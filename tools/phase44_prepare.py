#!/usr/bin/env python3
"""Freeze Phase43 and prepare Phase44 curriculum, calibration, and eval splits."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = REPO_ROOT / "pfe-core"
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

from pfe_core.phase44_preference_curriculum import (
    build_phase44_diagnostic_sessions,
    build_phase44_failure_analysis,
    build_phase44_holdout_sessions,
    build_phase44_preference_curriculum,
    build_phase44_scorer_calibration_cases,
    build_phase44_sft_job_spec,
    build_phase44_split_integrity,
    evaluate_phase44_scorer_calibration,
    stable_hash,
)
from pfe_core.trainer.executors import _build_seeded_stratified_training_order


PHASE43_ROOT = REPO_ROOT / "docs" / "demo" / "phase43-qwen3-4b-personal-preference-benefit-proof"
EVIDENCE_ROOT = REPO_ROOT / "docs" / "demo" / "phase44-preference-curriculum-privacy-safe-retraining"
MODEL_PATH = REPO_ROOT / "models" / "Qwen3-4B"
SCORER_SOURCE = CORE_ROOT / "pfe_core" / "phase44_preference_curriculum.py"


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


def _phase43_manifest() -> dict[str, Any]:
    files = []
    for path in sorted(PHASE43_ROOT.rglob("*")):
        if path.is_file():
            files.append({"path": str(path.relative_to(REPO_ROOT)), "size_bytes": path.stat().st_size, "sha256": _sha256(path)})
    return {
        "kind": "phase44_frozen_phase43_manifest",
        "created_at": _utcnow(),
        "phase43_root": str(PHASE43_ROOT),
        "file_count": len(files),
        "files": files,
        "manifest_sha256": stable_hash(files),
        "phase43_canonical_evidence_modified": False,
    }


def _phase43_baseline(manifest: Mapping[str, Any]) -> dict[str, Any]:
    comparison = _read_json(PHASE43_ROOT / "comparison_summary.json")
    decision = _read_json(PHASE43_ROOT / "phase43-final-decision.json")
    metrics = {
        name: _read_json(PHASE43_ROOT / "evidence-holdout" / f"metrics_{name}.json")
        for name in ("base", "runtime", "sft")
    }
    return {
        "kind": "phase44_frozen_phase43_baseline",
        "created_at": _utcnow(),
        "git": {
            "head": _command(["git", "rev-parse", "HEAD"]),
            "branch": _command(["git", "branch", "--show-current"]),
            "status": _command(["git", "status", "--short", "--branch"]),
        },
        "phase43_manifest_sha256": manifest.get("manifest_sha256"),
        "phase43_decision": decision,
        "phase43_comparison": comparison,
        "phase43_metrics": metrics,
        "canonical_summary": {
            "sft_status": dict(dict(decision.get("decision") or {}).get("candidate_decisions") or {}).get("sft", {}).get("status"),
            "sft_preference_score": metrics["sft"].get("user_preference_score"),
            "base_preference_score": metrics["base"].get("user_preference_score"),
            "runtime_preference_score": metrics["runtime"].get("user_preference_score"),
            "sft_privacy_violation_rate": metrics["sft"].get("privacy_violation_rate"),
            "sft_false_completion_rate": metrics["sft"].get("false_completion_rate"),
            "sft_independent_blind_win_rate": dict(
                dict(comparison.get("independent_blind") or {}).get("variants") or {}
            ).get("sft", {}).get("candidate_win_rate"),
            "runtime_independent_blind_win_rate": dict(
                dict(comparison.get("independent_blind") or {}).get("variants") or {}
            ).get("runtime", {}).get("candidate_win_rate"),
        },
        "phase43_canonical_evidence_modified": False,
    }


def _failure_analysis() -> dict[str, Any]:
    metrics = {
        name: _read_json(PHASE43_ROOT / "evidence-holdout" / f"metrics_{name}.json")
        for name in ("base", "runtime", "sft")
    }
    transcripts = {
        name: _read_jsonl(PHASE43_ROOT / "evidence-holdout" / f"transcripts_{name}.jsonl")
        for name in ("base", "runtime", "sft")
    }
    return build_phase44_failure_analysis(metrics, transcripts)


def _model_selection() -> dict[str, Any]:
    config = MODEL_PATH / "config.json"
    index = MODEL_PATH / "model.safetensors.index.json"
    shards = sorted(MODEL_PATH.glob("*.safetensors"))
    return {
        "kind": "phase44_model_selection",
        "status": "selected" if config.exists() and index.exists() and len(shards) == 3 else "blocked",
        "selected_model": "Qwen3-4B" if config.exists() and index.exists() and len(shards) == 3 else None,
        "model_path": str(MODEL_PATH),
        "config_exists": config.exists(),
        "index_exists": index.exists(),
        "shard_count": len(shards),
        "shards": [{"name": path.name, "size_bytes": path.stat().st_size} for path in shards],
        "selection_reason": "Reuse the Phase43 local unquantized model; change curriculum and exposure, not model scale.",
        "qwen27b_training_allowed": False,
    }


def _sampler_report(pairs: list[dict[str, Any]], steps: int, *, seed: int = 44) -> dict[str, Any]:
    order: list[int] = []
    cycle = 0
    while len(order) < steps:
        order.extend(_build_seeded_stratified_training_order(pairs, seed=seed, cycle=cycle))
        cycle += 1
    order = order[:steps]
    sample_counts = Counter(str(pairs[index]["sample_id"]) for index in order)
    category_counts = Counter(str(pairs[index]["taxonomy_dimension"]) for index in order)
    full_coverage = len(sample_counts) == len(pairs) and all(value >= 1 for value in sample_counts.values())
    values = list(category_counts.values())
    return {
        "kind": "phase44_sampler_plan",
        "seed": seed,
        "sampling_strategy": "seeded_stratified",
        "requested_steps": steps,
        "source_sample_count": len(pairs),
        "ordered_sample_ids": [pairs[index]["sample_id"] for index in order],
        "sample_exposure_counts": dict(sorted(sample_counts.items())),
        "category_exposure_counts": dict(sorted(category_counts.items())),
        "unique_samples_exposed": len(sample_counts),
        "unique_categories_exposed": len(category_counts),
        "category_exposure_spread": max(values) - min(values) if values else None,
        "full_coverage": full_coverage,
        "eligible_as_final_candidate": steps >= len(pairs) and full_coverage,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-evidence", action="store_true")
    args = parser.parse_args()
    if args.clean_evidence and EVIDENCE_ROOT.exists():
        shutil.rmtree(EVIDENCE_ROOT)
    directories = (
        "evidence-baseline", "evidence-failure-analysis", "evidence-curriculum",
        "evidence-scorer-calibration", "evidence-training-sft", "evidence-training-dpo-preflight",
        "evidence-holdout", "evidence-blind-eval",
    )
    for name in directories:
        (EVIDENCE_ROOT / name).mkdir(parents=True, exist_ok=True)

    manifest = _phase43_manifest()
    baseline = _phase43_baseline(manifest)
    failure = _failure_analysis()
    curriculum = build_phase44_preference_curriculum()
    holdout = build_phase44_holdout_sessions()
    diagnostic = build_phase44_diagnostic_sessions()
    split = build_phase44_split_integrity(curriculum["pairs"], holdout["sessions"], diagnostic["sessions"])
    calibration_cases = build_phase44_scorer_calibration_cases()
    calibration = evaluate_phase44_scorer_calibration(calibration_cases["cases"])
    scorer_freeze = {
        "kind": "phase44_scorer_freeze",
        "frozen_at": _utcnow(),
        "frozen_before_phase44_model_calls": True,
        "source_path": str(SCORER_SOURCE),
        "source_sha256": _sha256(SCORER_SOURCE),
        "calibration_case_manifest_sha256": calibration_cases["manifest_sha256"],
        "calibration_status": calibration["status"],
        "minimum_precision": 0.90,
        "minimum_recall": 0.90,
        "scorer_changes_after_freeze_allowed": False,
    }

    _write_json(EVIDENCE_ROOT / "evidence-baseline" / "phase43_canonical_manifest.json", manifest)
    _write_json(EVIDENCE_ROOT / "evidence-baseline" / "phase43_baseline_snapshot.json", baseline)
    _write_json(EVIDENCE_ROOT / "evidence-baseline" / "model_selection.json", _model_selection())
    _write_json(EVIDENCE_ROOT / "evidence-failure-analysis" / "failure_taxonomy.json", failure["failure_taxonomy"])
    _write_jsonl(EVIDENCE_ROOT / "evidence-failure-analysis" / "phase43_failure_examples.jsonl", failure["failure_examples"])
    _write_json(EVIDENCE_ROOT / "evidence-failure-analysis" / "failure_distribution.json", failure["failure_distribution"])
    _write_json(EVIDENCE_ROOT / "evidence-failure-analysis" / "remediation_plan.json", failure["remediation_plan"])
    _write_json(EVIDENCE_ROOT / "evidence-curriculum" / "curriculum_manifest.json", {key: value for key, value in curriculum.items() if key != "pairs"})
    _write_json(EVIDENCE_ROOT / "evidence-curriculum" / "curriculum_audit.json", curriculum["audit"])
    _write_jsonl(EVIDENCE_ROOT / "evidence-curriculum" / "selected_preference_pairs.jsonl", curriculum["pairs"])
    _write_json(EVIDENCE_ROOT / "evidence-scorer-calibration" / "calibration_cases.json", calibration_cases)
    _write_json(EVIDENCE_ROOT / "evidence-scorer-calibration" / "calibration_report.json", calibration)
    _write_json(EVIDENCE_ROOT / "evidence-scorer-calibration" / "scorer_freeze.json", scorer_freeze)
    _write_json(EVIDENCE_ROOT / "evidence-holdout" / "holdout.json", holdout)
    _write_json(EVIDENCE_ROOT / "evidence-holdout" / "diagnostic_sessions.json", diagnostic)
    _write_json(EVIDENCE_ROOT / "evidence-holdout" / "split_integrity.json", split)
    _write_json(EVIDENCE_ROOT / "evidence-holdout" / "holdout_freeze.json", {
        "kind": "phase44_holdout_freeze", "frozen_at": _utcnow(), "frozen_before_training": True,
        "holdout_manifest_sha256": holdout["manifest_sha256"], "session_count": holdout["holdout_count"],
        "phase43_holdout_reused": False, "not_for_training": True,
    })
    _write_json(EVIDENCE_ROOT / "evidence-holdout" / "source_manifest.json", {
        "kind": "phase44_holdout_source_manifest", "source": "deterministic_simulated_usage_scenarios",
        "actual_user_feedback": False, "not_for_training": True, "private_source_text_included": False,
        "session_count": holdout["holdout_count"], "categories": holdout["categories"],
    })
    for steps in (1, 12, 120):
        probe = EVIDENCE_ROOT / "evidence-training-sft" / f"probe-{steps}step"
        sampler = _sampler_report(curriculum["pairs"], steps)
        job = build_phase44_sft_job_spec(
            pairs=curriculum["pairs"], base_model=str(MODEL_PATH),
            output_dir=str(REPO_ROOT / "trainer_job_outputs" / f"phase44-qwen3-4b-sft-{steps}step"), max_steps=steps,
        )
        _write_json(probe / "sampler_plan.json", sampler)
        _write_json(probe / "training_manifest.json", job)
    _write_json(EVIDENCE_ROOT / "evidence-training-dpo-preflight" / "phase43_dpo_nan_regression.json", {
        "kind": "phase44_dpo_preflight", "status": "disabled", "real_dpo_requested": False,
        "reason": "Phase43 Qwen3-4B DPO produced non-finite metrics; Phase44 changes SFT curriculum first.",
        "phase43_attempt": _read_json(PHASE43_ROOT / "evidence-training-dpo" / "training_attempt.json"),
        "dpo_adapter_eligible": False,
    })
    preparation = {
        "kind": "phase44_preparation_decision",
        "status": "ready_for_real_sft_probes" if curriculum["audit"]["passed"] and split["passed"] and calibration["status"] == "passed" else "blocked",
        "phase43_frozen": True,
        "curriculum_approved_count": curriculum["approved_count"],
        "holdout_count": holdout["holdout_count"],
        "calibration_status": calibration["status"],
        "scorer_frozen_before_model_calls": True,
        "next_action": "run_qwen3_4b_sft_1_12_120" if calibration["status"] == "passed" else "repair_scorer_before_any_model_call",
        "actual_product_benefit_claim_allowed": False,
        "auto_promotion_allowed": False,
    }
    _write_json(EVIDENCE_ROOT / "preparation_decision.json", preparation)
    print(json.dumps(preparation, ensure_ascii=False, indent=2))
    return 0 if preparation["status"] == "ready_for_real_sft_probes" else 1


if __name__ == "__main__":
    raise SystemExit(main())
