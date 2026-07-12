#!/usr/bin/env python3
"""Freeze Phase44 and prepare Phase45 data, scorer, and training probes."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = REPO_ROOT / "pfe-core"
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

from pfe_core.phase45_privacy_multiturn_preference import (
    build_phase45_diagnostic_sessions,
    build_phase45_holdout_sessions,
    build_phase45_preference_curriculum,
    build_phase45_scorer_calibration_cases,
    build_phase45_sft_job_spec,
    build_phase45_split_integrity,
    evaluate_phase45_scorer_calibration,
    sanitize_privacy_output,
    stable_hash,
    transform_privacy_messages,
)
from pfe_core.trainer.executors import _build_seeded_stratified_training_order


PHASE44_ROOT = REPO_ROOT / "docs" / "demo" / "phase44-preference-curriculum-privacy-safe-retraining"
EVIDENCE_ROOT = REPO_ROOT / "docs" / "demo" / "phase45-privacy-structural-multiturn-preference"
MODEL_PATH = REPO_ROOT / "models" / "Qwen3-4B"
SCORER_SOURCE = CORE_ROOT / "pfe_core" / "phase45_privacy_multiturn_preference.py"
EXECUTOR_SOURCE = CORE_ROOT / "pfe_core" / "trainer" / "executors.py"


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return dict(value) if isinstance(value, Mapping) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _command(args: list[str]) -> dict[str, Any]:
    completed = subprocess.run(args, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    return {"command": args, "returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr}


def _phase44_manifest() -> dict[str, Any]:
    files = []
    for path in sorted(PHASE44_ROOT.rglob("*")):
        if path.is_file():
            files.append({"path": str(path.relative_to(REPO_ROOT)), "size_bytes": path.stat().st_size, "sha256": _sha256(path)})
    decision = _read_json(PHASE44_ROOT / "phase44-final-decision.json")
    pr = _command(["gh", "pr", "view", "55", "--json", "number,url,isDraft,state,headRefName,baseRefName,statusCheckRollup"])
    return {
        "kind": "phase45_frozen_phase44_canonical_manifest",
        "created_at": _utcnow(),
        "phase44_root": str(PHASE44_ROOT),
        "file_count": len(files),
        "files": files,
        "manifest_sha256": stable_hash(files),
        "phase44_commit": "2471c8b",
        "phase44_pr_number": 55,
        "phase44_pr_snapshot": pr,
        "phase44_adapter_sha256": decision.get("selected_adapter_sha256"),
        "phase44_recommendation": decision.get("recommendation"),
        "phase44_archive_preserved": decision.get("recommendation") == "archive",
        "phase44_canonical_evidence_modified": False,
    }


def _phase44_baseline(manifest: Mapping[str, Any]) -> dict[str, Any]:
    real = PHASE44_ROOT / "evidence-holdout" / "real-60-session"
    metrics = {name: _read_json(real / f"metrics_{name}.json") for name in ("base", "sft")}
    comparison = _read_json(PHASE44_ROOT / "comparison_summary.json")
    decision = _read_json(PHASE44_ROOT / "phase44-final-decision.json")
    return {
        "kind": "phase45_phase44_baseline_snapshot",
        "created_at": _utcnow(),
        "phase44_manifest_sha256": manifest.get("manifest_sha256"),
        "phase44_decision": decision,
        "canonical_metrics": {
            "base_user_preference_score": metrics["base"].get("user_preference_score"),
            "sft_user_preference_score": metrics["sft"].get("user_preference_score"),
            "sft_score_gain": round(float(metrics["sft"].get("user_preference_score") or 0) - float(metrics["base"].get("user_preference_score") or 0), 4),
            "base_truncated_response_rate": metrics["base"].get("truncated_response_rate"),
            "sft_privacy_violation_rate": metrics["sft"].get("privacy_violation_rate"),
            "sft_response_diversity": metrics["sft"].get("response_diversity"),
            "sft_repetition_rate": metrics["sft"].get("repetition_rate"),
            "sft_latest_user_intent_rate": metrics["sft"].get("follows_latest_user_intent_rate"),
            "deterministic_sft_win_rate": dict(dict(comparison.get("deterministic_blind") or {}).get("comparisons") or {}).get("sft_vs_base", {}).get("candidate_win_rate"),
            "gemma4_sft_win_rate": dict(dict(comparison.get("independent_blind") or {}).get("comparisons") or {}).get("sft_vs_base", {}).get("candidate_win_rate"),
        },
        "phase44_canonical_evidence_modified": False,
        "phase44_adapter_lifecycle": "archive",
    }


def _safe_excerpt(text: str, limit: int = 420) -> str:
    value = re.sub(r"PFE44_[A-Za-z0-9_]+", "<SYNTHETIC_CANARY>", str(text or ""))
    return value[:limit]


def _failure_analysis() -> dict[str, Any]:
    real = PHASE44_ROOT / "evidence-holdout" / "real-60-session"
    base = _read_json(real / "metrics_base.json")
    sft = _read_json(real / "metrics_sft.json")
    rows = _read_jsonl(real / "transcripts_sft.jsonl")
    details = {str(row.get("session_id")): dict(row.get("scores") or {}) for row in sft.get("details") or []}
    examples = []
    for row in rows:
        scores = details.get(str(row.get("session_id")), {})
        flags = [name for name in ("privacy_violation", "truncated_response") if float(scores.get(name) or 0) > 0]
        if float(scores.get("follows_latest_user_intent") or 1) == 0:
            flags.append("latest_user_intent_missed")
        if not flags:
            continue
        assistant = [turn for turn in row.get("turns") or [] if turn.get("role") == "assistant"]
        examples.append({
            "session_id": row.get("session_id"),
            "category": row.get("category"),
            "failure_flags": flags,
            "final_output_excerpt": _safe_excerpt(dict(assistant[-1]).get("content") if assistant else ""),
            "synthetic_canary_literal_persisted": False,
        })
        if len(examples) >= 12:
            break
    return {
        "kind": "phase45_phase44_failure_analysis",
        "evidence_basis": "Phase44 real 60-session base/SFT metrics and transcript text",
        "metrics": {
            "base_truncated_response_rate": base.get("truncated_response_rate"),
            "sft_latest_user_intent_rate": sft.get("follows_latest_user_intent_rate"),
            "sft_privacy_violation_rate": sft.get("privacy_violation_rate"),
            "sft_response_diversity": sft.get("response_diversity"),
            "sft_repetition_rate": sft.get("repetition_rate"),
        },
        "root_cause_hypotheses": [
            {"failure": "privacy_echo", "hypothesis": "Phase44 literal synthetic secret text entered the model prompt, so the model could copy it; training alone cannot enforce a zero-copy boundary."},
            {"failure": "latest_correction_regression", "hypothesis": "Phase44 SFT flattened the task into instruction plus completion instead of preserving the provisional assistant turn and final user correction."},
            {"failure": "template_repetition", "hypothesis": "Repeated target openings and compact category templates encouraged surface-form reuse at full coverage."},
            {"failure": "unfair_truncation", "hypothesis": "The 128-token generation cap truncated 61.67% of base sessions, making arm comparisons structurally unfair."},
        ],
        "remediation": [
            "redact private spans before prompt construction and sanitize output before persistence",
            "train native messages history with completion-only loss on the final assistant answer",
            "use 160 unique short targets across ordinary and boundary tasks",
            "freeze a 384-token deterministic generation protocol and require every arm truncation at most 0.05",
        ],
        "failure_examples": examples,
        "conclusions_are_hypotheses_until_phase45_ablation": True,
    }


def _model_selection() -> dict[str, Any]:
    config = MODEL_PATH / "config.json"
    index = MODEL_PATH / "model.safetensors.index.json"
    shards = sorted(MODEL_PATH.glob("*.safetensors"))
    return {
        "kind": "phase45_model_selection",
        "status": "selected" if config.exists() and index.exists() and len(shards) == 3 else "blocked",
        "selected_model": "Qwen3-4B" if config.exists() and index.exists() and len(shards) == 3 else None,
        "model_path": str(MODEL_PATH),
        "config_exists": config.exists(),
        "index_exists": index.exists(),
        "shard_count": len(shards),
        "selection_reason": "Keep model scale fixed and isolate privacy, representation, and eval changes.",
        "qwen27b_training_allowed": False,
        "dpo_allowed": False,
    }


def _privacy_boundary_evidence() -> dict[str, Any]:
    api_key = "sk-" + "PFE45BOUNDARY" * 2
    telegram = "1234567890:" + "Phase45BoundaryToken" * 2
    canary = "PFE45_HOLDOUT_SECRET_BOUNDARY_XQ4501"
    controls = "PID 43210, port 8927, commit 2471c8b, public ISSUE-4501, PR-55"
    result = transform_privacy_messages([
        {"role": "user", "content": f"api_key={api_key}; bot={telegram}; canary={canary}; {controls}"},
    ])
    attempted_output = f"检测到 {result.private_values[0]}，内部字段 {result.placeholders[0]}。"
    sanitized_output, output_audit = sanitize_privacy_output(attempted_output, result)
    serialized = json.dumps(
        {"messages": result.messages, "manifest": result.manifest, "sanitized_output": sanitized_output, "output_audit": output_audit},
        ensure_ascii=False,
    )
    raw_absent = all(value not in serialized for value in (api_key, telegram, canary))
    controls_preserved = all(value in result.messages[0]["content"] for value in ("43210", "8927", "2471c8b", "ISSUE-4501", "PR-55"))
    return {
        "kind": "phase45_privacy_boundary_evidence",
        "status": "passed" if raw_absent and controls_preserved else "failed",
        "transformed_messages": result.messages,
        "redaction_manifest": result.manifest,
        "output_after_persistence_sanitization": sanitized_output,
        "output_sanitization_audit": output_audit,
        "raw_private_values_absent_from_persisted_evidence": raw_absent,
        "nonsecret_controls_preserved": controls_preserved,
        "nonsecret_control_values": ["43210", "8927", "2471c8b", "ISSUE-4501", "PR-55"],
        "model_prompt_contains_raw_private_values": False,
        "transcript_contains_raw_private_values": False,
        "training_samples_contain_raw_private_values": False,
    }


def _sampler_plan(pairs: list[dict[str, Any]], *, steps: int, seed: int, candidate_id: str) -> dict[str, Any]:
    order: list[int] = []
    cycle = 0
    while len(order) < steps:
        order.extend(_build_seeded_stratified_training_order(pairs, seed=seed, cycle=cycle))
        cycle += 1
    order = order[:steps]
    samples = Counter(str(pairs[index]["sample_id"]) for index in order)
    categories = Counter(str(pairs[index]["taxonomy_dimension"]) for index in order)
    full = len(samples) == len(pairs) and all(value >= 1 for value in samples.values())
    return {
        "kind": "phase45_sampler_plan",
        "candidate_id": candidate_id,
        "seed": seed,
        "requested_steps": steps,
        "source_sample_count": len(pairs),
        "ordered_sample_ids": [pairs[index]["sample_id"] for index in order],
        "sample_exposure_counts": dict(sorted(samples.items())),
        "category_exposure_counts": dict(sorted(categories.items())),
        "unique_samples_exposed": len(samples),
        "unique_categories_exposed": len(categories),
        "full_coverage": full,
        "eligible_as_final_candidate": steps >= len(pairs) and full,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-evidence", action="store_true")
    args = parser.parse_args()
    if args.clean_evidence and EVIDENCE_ROOT.exists():
        shutil.rmtree(EVIDENCE_ROOT)
    for name in (
        "evidence-baseline", "evidence-failure-analysis", "evidence-privacy-boundary",
        "evidence-curriculum", "evidence-scorer-calibration", "evidence-training-sft",
        "evidence-diagnostic", "evidence-holdout", "evidence-blind-eval",
    ):
        (EVIDENCE_ROOT / name).mkdir(parents=True, exist_ok=True)

    manifest = _phase44_manifest()
    baseline = _phase44_baseline(manifest)
    failure = _failure_analysis()
    curriculum = build_phase45_preference_curriculum()
    holdout = build_phase45_holdout_sessions()
    diagnostic = build_phase45_diagnostic_sessions()
    phase44_holdout = _read_json(PHASE44_ROOT / "evidence-holdout" / "holdout.json")
    split = build_phase45_split_integrity(
        curriculum["pairs"], holdout["sessions"], diagnostic["sessions"],
        phase44_holdout_sessions=phase44_holdout.get("sessions") or [],
    )
    calibration_cases = build_phase45_scorer_calibration_cases()
    calibration = evaluate_phase45_scorer_calibration(calibration_cases["cases"])
    scorer_freeze = {
        "kind": "phase45_scorer_freeze",
        "frozen_at": _utcnow(),
        "frozen_before_phase45_model_calls": True,
        "source_path": str(SCORER_SOURCE),
        "source_sha256": _sha256(SCORER_SOURCE),
        "executor_source_path": str(EXECUTOR_SOURCE),
        "executor_source_sha256": _sha256(EXECUTOR_SOURCE),
        "calibration_case_manifest_sha256": calibration_cases["manifest_sha256"],
        "calibration_status": calibration["status"],
        "minimum_precision": 0.90,
        "minimum_recall": 0.90,
        "changes_after_formal_model_calls_allowed": False,
    }
    generation_protocol = {
        "kind": "phase45_fair_generation_protocol",
        "max_new_tokens": 384,
        "input_max_length": 4096,
        "do_sample": False,
        "repetition_penalty": 1.05,
        "eos_token_from_tokenizer": True,
        "think": False,
        "required_all_arm_truncated_response_rate_max": 0.05,
        "formal_eval_requires_preflight": True,
    }
    generation_protocol["protocol_sha256"] = stable_hash(generation_protocol)
    privacy_boundary = _privacy_boundary_evidence()

    _write_json(EVIDENCE_ROOT / "evidence-baseline" / "phase44_canonical_manifest.json", manifest)
    _write_json(EVIDENCE_ROOT / "evidence-baseline" / "phase44_baseline_snapshot.json", baseline)
    _write_json(EVIDENCE_ROOT / "evidence-baseline" / "model_selection.json", _model_selection())
    _write_json(EVIDENCE_ROOT / "evidence-failure-analysis" / "phase44_failure_analysis.json", failure)
    _write_json(EVIDENCE_ROOT / "evidence-privacy-boundary" / "privacy_transform_evidence.json", privacy_boundary)
    _write_json(EVIDENCE_ROOT / "evidence-privacy-boundary" / "privacy_data_flow.json", {
        "kind": "phase45_privacy_data_flow",
        "order": ["detect_spans", "replace_with_typed_placeholders", "construct_model_prompt", "generate", "sanitize_output", "persist_sanitized_transcript"],
        "raw_values_enter_model": False,
        "raw_values_enter_transcript": False,
        "manifest_fields": ["type", "message_index", "start", "end", "sha256", "placeholder_type"],
        "manifest_raw_values_persisted": False,
    })
    _write_json(EVIDENCE_ROOT / "evidence-curriculum" / "curriculum_manifest.json", {key: value for key, value in curriculum.items() if key != "pairs"})
    _write_json(EVIDENCE_ROOT / "evidence-curriculum" / "curriculum_audit.json", curriculum["audit"])
    _write_jsonl(EVIDENCE_ROOT / "evidence-curriculum" / "selected_preference_pairs.jsonl", curriculum["pairs"])
    _write_json(EVIDENCE_ROOT / "evidence-scorer-calibration" / "calibration_cases.json", calibration_cases)
    _write_json(EVIDENCE_ROOT / "evidence-scorer-calibration" / "calibration_report.json", calibration)
    _write_json(EVIDENCE_ROOT / "evidence-scorer-calibration" / "scorer_freeze.json", scorer_freeze)
    _write_json(EVIDENCE_ROOT / "evidence-holdout" / "holdout.json", holdout)
    _write_json(EVIDENCE_ROOT / "evidence-diagnostic" / "diagnostic_sessions.json", diagnostic)
    _write_json(EVIDENCE_ROOT / "evidence-holdout" / "split_integrity.json", split)
    _write_json(EVIDENCE_ROOT / "evidence-holdout" / "holdout_freeze.json", {
        "kind": "phase45_holdout_freeze", "frozen_at": _utcnow(), "frozen_before_training": True,
        "holdout_manifest_sha256": holdout["manifest_sha256"], "session_count": holdout["holdout_count"],
        "phase44_holdout_used_for_training": False, "not_for_training": True,
    })
    _write_json(EVIDENCE_ROOT / "evidence-holdout" / "source_manifest.json", {
        "kind": "phase45_holdout_source_manifest", "source": "deterministic_simulated_usage_scenarios",
        "actual_user_feedback": False, "not_for_training": True, "private_source_text_included": False,
        "synthetic_privacy_canaries_only": True, "session_count": holdout["holdout_count"],
        "categories": holdout["categories"],
    })
    _write_json(EVIDENCE_ROOT / "evidence-holdout" / "generation_protocol.json", generation_protocol)
    _write_json(EVIDENCE_ROOT / "evidence-training-sft" / "dpo_status.json", {
        "kind": "phase45_dpo_status", "status": "disabled", "dpo_training_requested": False,
        "reason": "Phase45 isolates native multi-turn SFT and privacy boundary changes.",
    })

    probes = (
        ("candidate-a-probe-1step", "candidate_a", 1, 1e-5, 45),
        ("candidate-a-probe-12step", "candidate_a", 12, 1e-5, 45),
        ("candidate-a-full-160step", "candidate_a", 160, 1e-5, 45),
        ("candidate-b-full-160step", "candidate_b", 160, 5e-6, 145),
    )
    for directory, candidate_id, steps, learning_rate, seed in probes:
        probe = EVIDENCE_ROOT / "evidence-training-sft" / directory
        output_dir = REPO_ROOT / "trainer_job_outputs" / f"phase45-{directory}"
        _write_json(probe / "sampler_plan.json", _sampler_plan(curriculum["pairs"], steps=steps, seed=seed, candidate_id=candidate_id))
        _write_json(probe / "training_manifest.json", build_phase45_sft_job_spec(
            pairs=curriculum["pairs"], base_model=str(MODEL_PATH), output_dir=str(output_dir),
            max_steps=steps, learning_rate=learning_rate, seed=seed, candidate_id=candidate_id,
        ))

    ready = (
        manifest.get("phase44_archive_preserved") is True
        and curriculum["audit"]["passed"] is True
        and split["passed"] is True
        and calibration["status"] == "passed"
        and privacy_boundary["status"] == "passed"
        and _model_selection()["status"] == "selected"
    )
    preparation = {
        "kind": "phase45_preparation_decision",
        "status": "ready_for_real_sft_probes" if ready else "blocked",
        "phase44_frozen": True,
        "phase44_archive_preserved": manifest.get("phase44_archive_preserved"),
        "curriculum_approved_count": curriculum["approved_count"],
        "diagnostic_count": diagnostic["session_count"],
        "holdout_count": holdout["holdout_count"],
        "split_integrity_passed": split["passed"],
        "calibration_status": calibration["status"],
        "privacy_boundary_status": privacy_boundary["status"],
        "scorer_frozen_before_model_calls": True,
        "generation_protocol_sha256": generation_protocol["protocol_sha256"],
        "next_action": "run_candidate_a_1_12_then_two_160step_candidates" if ready else "repair_preparation_gate_before_any_model_call",
        "actual_product_benefit_claim_allowed": False,
        "auto_promotion_allowed": False,
    }
    _write_json(EVIDENCE_ROOT / "preparation_decision.json", preparation)
    print(json.dumps(preparation, ensure_ascii=False, indent=2))
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
