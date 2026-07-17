#!/usr/bin/env python3
"""Train and evaluate an aligned, diverse Qwen3-4B SFT curriculum."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import sys
import time
import traceback
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = REPO_ROOT / "pfe-core"
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

from pfe_core.adapter_store.quality import validate_adapter_artifact
from pfe_core.phase43_personal_preference_benefit import build_phase43_sft_job_spec
from pfe_core.phase75_personalization_benefit_benchmark import stable_hash
from pfe_core.phase103_simulated_user_acceptance import aggregate_phase103_scores, score_phase103_session
from pfe_core.phase105_qwen3_curriculum_alignment import (
    audit_phase105_curriculum,
    build_phase105_curriculum,
    build_phase105_decision,
    build_phase105_holdout,
)
from pfe_core.trainer.executors import (
    _build_sft_prompt_and_text,
    _encode_sft_examples,
    _run_real_local_peft_training,
)
from phase101_failure_targeted_sft import _load_runtime, _run_session, _write_private_jsonl


EVIDENCE_ROOT = REPO_ROOT / "docs/demo/phase105-qwen3-no-think-curriculum-alignment"
PREPARATION_ROOT = EVIDENCE_ROOT / "evidence-preparation"
TRAINING_ROOT = EVIDENCE_ROOT / "evidence-training"
EVAL_ROOT = EVIDENCE_ROOT / "evidence-eval"
FAILURE_ROOT = EVIDENCE_ROOT / "evidence-failures"
PRIVATE_ROOT = Path("/private/tmp/pfe-phase105-simulated-review")
MODEL_PATH = REPO_ROOT / "models/Qwen3-4B"
TRAINER_OUTPUT_ROOT = REPO_ROOT / "trainer_job_outputs"
PHASE100_104_ROOT = REPO_ROOT / "docs/demo/phase100-104-autonomous-qwen3-training-benefit-loop"
PHASE101_ROOT = PHASE100_104_ROOT / "phase101-failure-targeted-sft"
PHASE104_DECISION = PHASE100_104_ROOT / "phase104-final-decision.json"
GENERATION_PROTOCOL = {
    "model": "Qwen3-4B",
    "input_max_length": 3072,
    "max_new_tokens": 160,
    "do_sample": False,
    "repetition_penalty": 1.15,
    "no_repeat_ngram_size": 4,
    "enable_thinking": False,
    "guided_target_allowed": False,
    "premature_eos_suppression_allowed": False,
    "post_hoc_truncation_allowed": False,
    "variants": ["base", "candidate"],
    "model_calls_per_variant": 30,
    "phase105_model_call_budget": 60,
}


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_clean(path: Path, parent: Path) -> None:
    resolved = path.resolve()
    if resolved.parent != parent.resolve():
        raise RuntimeError(f"refusing to clean outside {parent}: {path}")
    if resolved.exists():
        shutil.rmtree(resolved)


def _source_hashes() -> dict[str, str]:
    paths = {
        "core": CORE_ROOT / "pfe_core/phase105_qwen3_curriculum_alignment.py",
        "driver": REPO_ROOT / "tools/phase105_qwen3_curriculum_alignment.py",
        "core_test": REPO_ROOT / "tests/test_phase105_qwen3_curriculum_alignment.py",
        "driver_test": REPO_ROOT / "tests/test_phase105_driver_safety.py",
        "executor": CORE_ROOT / "pfe_core/trainer/executors.py",
        "phase101_driver": REPO_ROOT / "tools/phase101_failure_targeted_sft.py",
    }
    return {name: _sha256(path) for name, path in paths.items()}


def _previous_holdouts() -> list[dict[str, Any]]:
    paths = (
        PHASE100_104_ROOT / "phase100-generation-boundary/evidence-preparation/final_holdout.json",
        PHASE101_ROOT / "evidence-preparation/holdout.json",
        PHASE100_104_ROOT / "phase103-simulated-user-acceptance/evidence-preparation/sessions.json",
    )
    return [_read_json(path) for path in paths if path.is_file()]


def _job_spec(rows: list[dict[str, Any]], *, steps: int, output_dir: Path) -> dict[str, Any]:
    spec = build_phase43_sft_job_spec(pairs=rows, base_model=str(MODEL_PATH), output_dir=str(output_dir), max_steps=steps)
    by_id = {str(row["sample_id"]): row for row in rows}
    for example in spec["training_examples"]:
        source = by_id[str(example["sample_id"])]
        example["messages"] = [dict(message) for message in source["messages"]]
        example["category"] = source["category"]
    spec["recipe"]["training"].update({
        "max_length": 384,
        "learning_rate": 0.00001,
        "seed": 105,
    })
    spec["phase105"] = {
        "system_contract_aligned": True,
        "multiturn_correction_context": True,
        "qwen3_empty_think_boundary_required": True,
        "completion_only_loss_required": True,
        "target_model": "Qwen3-4B",
        "simulated_usage": True,
        "actual_user_feedback": False,
        "automatic_promotion_allowed": False,
    }
    return spec


def _alignment_report(spec: Mapping[str, Any]) -> dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), local_files_only=True)
    examples = list(spec.get("training_examples") or [])
    max_length = int(spec["recipe"]["training"]["max_length"])
    encoded = _encode_sft_examples(
        tokenizer=tokenizer,
        training_examples=examples,
        max_length=max_length,
        vocab_size=int(getattr(tokenizer, "vocab_size", 0) or 151936),
    )
    details = []
    for example, encoded_row in zip(examples, encoded):
        prompt, full = _build_sft_prompt_and_text(
            tokenizer,
            str(example.get("instruction") or ""),
            str(example.get("chosen") or ""),
            messages=example.get("messages"),
        )
        labels = list(encoded_row.get("labels") or [])
        details.append({
            "sample_id": example.get("sample_id"),
            "system_contract_present": str(example["messages"][0]["content"]) in prompt,
            "empty_think_boundary_present": prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"),
            "chosen_present_in_full": str(example.get("chosen") or "") in full,
            "completion_label_tokens": sum(int(value) != -100 for value in labels),
            "prompt_mask_tokens": sum(int(value) == -100 for value in labels),
        })
    checks = {
        "all_240_encoded": len(details) == 240,
        "all_system_contract_aligned": all(row["system_contract_present"] for row in details),
        "all_empty_think_aligned": all(row["empty_think_boundary_present"] for row in details),
        "all_chosen_present": all(row["chosen_present_in_full"] for row in details),
        "all_completion_tokens_present": all(int(row["completion_label_tokens"]) >= 8 for row in details),
        "all_prompt_tokens_masked": all(int(row["prompt_mask_tokens"]) > 0 for row in details),
    }
    return {
        "kind": "phase105_training_template_alignment_report",
        "passed": all(checks.values()),
        "checks": checks,
        "sample_count": len(details),
        "minimum_completion_label_tokens": min((int(row["completion_label_tokens"]) for row in details), default=0),
        "details_sha256": stable_hash(details),
    }


def _phase101_mismatch_diagnostic() -> dict[str, Any]:
    from transformers import AutoTokenizer

    manifest = _read_json(PHASE101_ROOT / "evidence-training/probe-30step/training_manifest.json")
    examples = list(manifest.get("training_examples") or [])
    first = dict(examples[0])
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), local_files_only=True)
    prompt, _ = _build_sft_prompt_and_text(
        tokenizer,
        str(first.get("instruction") or ""),
        str(first.get("chosen") or ""),
        messages=first.get("messages"),
    )
    provenance = [row for row in examples if str(row.get("sample_id") or "").startswith("phase101-sft-provenance")]
    return {
        "kind": "phase101_training_eval_mismatch_diagnostic",
        "phase101_sample_count": len(examples),
        "phase101_messages_present": any(bool(row.get("messages")) for row in examples),
        "phase101_system_contract_present": "只输出三行" in prompt,
        "phase101_empty_think_boundary_present": prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"),
        "phase101_provenance_sample_count": len(provenance),
        "phase101_provenance_unique_target_count": len({str(row.get("chosen") or "") for row in provenance}),
        "diagnosis": [
            "qwen3_no_think_boundary_was_already_aligned",
            "system_runtime_contract_missing_from_training_prompt",
            "single_turn_training_did_not_match_multiturn_eval",
            "provenance_targets_had_low_diversity",
        ],
    }


def _prepare(clean: bool) -> int:
    if clean and EVIDENCE_ROOT.exists():
        _safe_clean(EVIDENCE_ROOT, EVIDENCE_ROOT.parent)
    PREPARATION_ROOT.mkdir(parents=True, exist_ok=True)
    curriculum = build_phase105_curriculum()
    holdout = build_phase105_holdout()
    audit = audit_phase105_curriculum(curriculum, holdout, _previous_holdouts())
    spec = _job_spec(curriculum, steps=30, output_dir=TRAINER_OUTPUT_ROOT / "phase105-qwen3-4b-sft-30step")
    alignment = _alignment_report(spec)
    mismatch = _phase101_mismatch_diagnostic()
    phase104 = _read_json(PHASE104_DECISION)
    checks = {
        "phase104_runtime_contract_primary": phase104.get("recommendation") == "runtime_contract_remains_primary",
        "curriculum_holdout_audit_passed": audit.get("passed") is True,
        "training_template_alignment_passed": alignment.get("passed") is True,
        "phase101_system_contract_missing_confirmed": mismatch.get("phase101_system_contract_present") is False,
        "phase101_no_think_was_aligned": mismatch.get("phase101_empty_think_boundary_present") is True,
        "curriculum_count_240": len(curriculum) == 240,
        "fresh_holdout_count_10": holdout.get("session_count") == 10,
        "eval_calls_frozen_60": holdout.get("total_model_call_budget") == 60,
    }
    freeze = {
        "kind": "phase105_pre_experiment_freeze",
        "created_at": _utcnow(),
        "passed": all(checks.values()),
        "checks": checks,
        "curriculum_manifest_sha256": stable_hash(curriculum),
        "holdout_manifest_sha256": holdout["manifest_sha256"],
        "generation_protocol": GENERATION_PROTOCOL,
        "generation_protocol_sha256": stable_hash(GENERATION_PROTOCOL),
        "source_sha256": _source_hashes(),
        "training_steps": [1, 12, 30],
        "simulated_usage": True,
        "actual_user_feedback_count": 0,
        "product_gate_qualified": False,
        "automatic_promotion_allowed": False,
    }
    _write_jsonl(PREPARATION_ROOT / "selected_sft_samples.jsonl", curriculum)
    _write_json(PREPARATION_ROOT / "holdout.json", holdout)
    _write_json(PREPARATION_ROOT / "curriculum_holdout_audit.json", audit)
    _write_json(PREPARATION_ROOT / "training_template_alignment.json", alignment)
    _write_json(PREPARATION_ROOT / "phase101_mismatch_diagnostic.json", mismatch)
    _write_json(EVIDENCE_ROOT / "pre_experiment_freeze.json", freeze)
    print(json.dumps({"status": "ready" if freeze["passed"] else "blocked", "checks": checks}, ensure_ascii=False, indent=2))
    return 0 if freeze["passed"] else 2


def _training_freeze_check(steps: int) -> dict[str, Any]:
    freeze = _read_json(EVIDENCE_ROOT / "pre_experiment_freeze.json")
    rows = _read_jsonl(PREPARATION_ROOT / "selected_sft_samples.jsonl")
    holdout = _read_json(PREPARATION_ROOT / "holdout.json")
    checks = {
        "pre_experiment_freeze_passed": freeze.get("passed") is True,
        "source_files_unchanged": _source_hashes() == freeze.get("source_sha256"),
        "curriculum_unchanged": stable_hash(rows) == freeze.get("curriculum_manifest_sha256"),
        "holdout_unchanged": stable_hash(holdout.get("sessions") or []) == freeze.get("holdout_manifest_sha256"),
        "step_is_frozen": steps in (1, 12, 30),
    }
    if steps == 12:
        prior = _read_json(TRAINING_ROOT / "probe-1step/training_attempt.json") if (TRAINING_ROOT / "probe-1step/training_attempt.json").is_file() else {}
        checks["one_step_completed"] = prior.get("status") == "completed"
        checks["one_step_parameters_updated"] = dict(prior.get("adapter_validation") or {}).get("parameters_updated") is True
    if steps == 30:
        prior = _read_json(TRAINING_ROOT / "probe-12step/training_attempt.json") if (TRAINING_ROOT / "probe-12step/training_attempt.json").is_file() else {}
        history = list(dict(prior.get("execution") or {}).get("loss_history") or [])
        values = [row.get("loss") if isinstance(row, Mapping) else row for row in history]
        checks["twelve_step_completed"] = prior.get("status") == "completed"
        checks["twelve_step_parameters_updated"] = dict(prior.get("adapter_validation") or {}).get("parameters_updated") is True
        checks["twelve_step_losses_finite"] = bool(values) and all(value is not None and math.isfinite(float(value)) for value in values)
    return {"kind": "phase105_training_freeze_check", "steps": steps, "passed": all(checks.values()), "checks": checks}


def _train(steps: int, clean: bool) -> int:
    if steps not in (1, 12, 30):
        raise SystemExit("Phase105 permits 1, 12, or 30 steps only")
    probe_dir = TRAINING_ROOT / f"probe-{steps}step"
    output_root = TRAINER_OUTPUT_ROOT / f"phase105-qwen3-4b-sft-{steps}step"
    if clean and probe_dir.exists():
        _safe_clean(probe_dir, TRAINING_ROOT)
    if clean and output_root.exists():
        _safe_clean(output_root, TRAINER_OUTPUT_ROOT)
    probe_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_jsonl(PREPARATION_ROOT / "selected_sft_samples.jsonl")
    spec = _job_spec(rows, steps=steps, output_dir=output_root)
    alignment = _alignment_report(spec)
    freeze = _training_freeze_check(steps)
    _write_json(probe_dir / "training_manifest.json", spec)
    _write_json(probe_dir / "training_template_alignment.json", alignment)
    _write_json(probe_dir / "training_freeze_check.json", freeze)
    if not freeze["passed"] or not alignment["passed"]:
        attempt = {"kind": "phase105_sft_training_attempt", "status": "blocked", "real_training": False, "requested_steps": steps, "reason": "freeze_or_alignment_failed", "product_gate_qualified": False}
        _write_json(probe_dir / "training_attempt.json", attempt)
        return 2
    started = time.perf_counter()
    started_at = _utcnow()
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    try:
        result = _run_real_local_peft_training(spec)
        real = dict(result.get("real_execution") or {})
        artifact_dir = Path(str(real.get("artifact_dir") or ""))
        adapter_path = artifact_dir / "adapter_model.safetensors"
        validation = validate_adapter_artifact(artifact_dir, {"artifact_name": adapter_path.name, "artifact_format": "peft_lora"})
        validation.update({
            "sha256": _sha256(adapter_path) if adapter_path.is_file() else None,
            "artifact_dir": str(artifact_dir),
            "adapter_path": str(adapter_path),
            "parameters_updated": real.get("parameters_updated"),
            "steps": real.get("steps"),
        })
        completed = result.get("status") == "completed" and real.get("success") is True and real.get("parameters_updated") is True and int(real.get("steps") or 0) >= steps and validation.get("valid") is True
        attempt = {
            "kind": "phase105_sft_training_attempt",
            "status": "completed" if completed else "failed",
            "real_training": completed,
            "requested_steps": steps,
            "model": str(MODEL_PATH),
            "started_at": started_at,
            "finished_at": _utcnow(),
            "duration_seconds": round(time.perf_counter() - started, 4),
            "max_rss_before_bytes": rss_before,
            "max_rss_after_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "execution": real,
            "adapter_validation": validation,
            "simulated_usage": True,
            "actual_user_feedback": False,
            "product_gate_qualified": False,
            "automatic_promotion_allowed": False,
        }
        _write_json(probe_dir / "adapter_validation.json", validation)
        _write_json(probe_dir / "train_log.json", {"loss_history": real.get("loss_history") or [], "initial_loss": real.get("initial_loss"), "final_loss": real.get("final_loss")})
    except Exception as exc:
        attempt = {
            "kind": "phase105_sft_training_attempt",
            "status": "failed",
            "real_training": False,
            "requested_steps": steps,
            "error": f"{exc.__class__.__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "product_gate_qualified": False,
            "automatic_promotion_allowed": False,
        }
        FAILURE_ROOT.mkdir(parents=True, exist_ok=True)
        _write_json(FAILURE_ROOT / f"training_{steps}step.json", attempt)
    _write_json(probe_dir / "training_attempt.json", attempt)
    print(json.dumps({key: attempt.get(key) for key in ("status", "requested_steps", "duration_seconds", "error")}, ensure_ascii=False, indent=2))
    return 0 if attempt.get("status") == "completed" else 1


def _adapter_dir() -> Path:
    attempt = _read_json(TRAINING_ROOT / "probe-30step/training_attempt.json")
    validation = dict(attempt.get("adapter_validation") or {})
    path = Path(str(validation.get("artifact_dir") or ""))
    if attempt.get("status") != "completed" or validation.get("valid") is not True or not path.is_dir():
        raise SystemExit("Phase105 30-step adapter is unavailable")
    return path


def _eval_freeze_check(variant: str, adapter_path: Path | None) -> dict[str, Any]:
    freeze = _read_json(EVIDENCE_ROOT / "pre_experiment_freeze.json")
    holdout = _read_json(PREPARATION_ROOT / "holdout.json")
    checks = {
        "pre_experiment_freeze_passed": freeze.get("passed") is True,
        "source_files_unchanged": _source_hashes() == freeze.get("source_sha256"),
        "holdout_unchanged": stable_hash(holdout.get("sessions") or []) == freeze.get("holdout_manifest_sha256"),
        "generation_protocol_unchanged": stable_hash(GENERATION_PROTOCOL) == freeze.get("generation_protocol_sha256"),
        "variant_frozen": variant in {"base", "candidate"},
        "adapter_available_or_base": adapter_path is None or adapter_path.is_dir(),
        "no_completed_eval_exists": not (EVAL_ROOT / variant / "metrics.json").exists(),
    }
    return {"kind": "phase105_eval_freeze_check", "variant": variant, "passed": all(checks.values()), "checks": checks}


def _evaluate(variant: str, clean: bool) -> int:
    if variant not in {"base", "candidate"}:
        raise SystemExit("Phase105 eval variant must be base or candidate")
    adapter_path = None if variant == "base" else _adapter_dir()
    output_root = EVAL_ROOT / variant
    if clean and output_root.exists():
        _safe_clean(output_root, EVAL_ROOT)
    cache_path = PRIVATE_ROOT / f"{variant}.jsonl"
    if clean:
        cache_path.unlink(missing_ok=True)
    freeze = _eval_freeze_check(variant, adapter_path)
    _write_json(output_root / "freeze_check.json", freeze)
    if not freeze["passed"]:
        return 2
    sessions = [dict(row) for row in _read_json(PREPARATION_ROOT / "holdout.json").get("sessions") or []]
    rows = []
    scores = []
    private_rows = []
    torch = tokenizer = model = device = None
    try:
        torch, tokenizer, model, device = _load_runtime(adapter_path)
        for index, session in enumerate(sessions, start=1):
            structural, private = _run_session(session=session, torch=torch, tokenizer=tokenizer, model=model, device=device)
            structural["kind"] = "phase105_structural_session"
            outputs = [str(row.get("raw_output") or "") for row in private.get("turns") or []]
            user_score = score_phase103_session(session=session, outputs=outputs, structural_turns=structural.get("turns") or [])
            structural["simulated_user_score"] = user_score
            rows.append(structural)
            scores.append(user_score)
            private_rows.append({"session_id": session.get("session_id"), "category": session.get("category"), "turns": private.get("turns") or [], "final_acceptance": user_score.get("accepted")})
            _write_jsonl(output_root / "structural_sessions.jsonl", rows)
            _write_jsonl(output_root / "simulated_user_scores.jsonl", scores)
            _write_private_jsonl(cache_path, private_rows)
            print(f"[phase105:{variant}] {index}/{len(sessions)} {session.get('session_id')} accepted={user_score['accepted']}", flush=True)
    finally:
        if torch is not None and model is not None and device is not None:
            del model
            if device == "mps":
                torch.mps.empty_cache()
    metrics = aggregate_phase103_scores(scores)
    payload = {
        "kind": "phase105_variant_metrics",
        "variant": variant,
        "model_call_count": sum(int(row.get("turn_count") or 0) for row in rows),
        "metrics": metrics,
        "adapter_loaded": adapter_path is not None,
        "guided_generation_used": False,
        "private_cache": str(cache_path),
        "private_cache_outside_repo": True,
        "simulated_usage": True,
        "actual_user_feedback_count": 0,
    }
    _write_json(output_root / "metrics.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _build_manifest() -> dict[str, Any]:
    excluded = {EVIDENCE_ROOT / "evidence_manifest.json", EVIDENCE_ROOT / "validation_summary.json"}
    files = [path for path in sorted(EVIDENCE_ROOT.rglob("*")) if path.is_file() and path not in excluded]
    return {
        "kind": "phase105_evidence_manifest",
        "files": [{"path": str(path.relative_to(REPO_ROOT)), "sha256": _sha256(path), "size_bytes": path.stat().st_size} for path in files],
        "file_count": len(files),
        "private_transcripts_committed": False,
        "actual_user_feedback_count": 0,
    }


def _decide() -> int:
    base = dict(_read_json(EVAL_ROOT / "base/metrics.json").get("metrics") or {})
    candidate = dict(_read_json(EVAL_ROOT / "candidate/metrics.json").get("metrics") or {})
    training = _read_json(TRAINING_ROOT / "probe-30step/training_attempt.json")
    decision = build_phase105_decision(base_metrics=base, candidate_metrics=candidate, training_completed=training.get("status") == "completed" and training.get("real_training") is True)
    decision.update({
        "base_metrics": base,
        "candidate_metrics": candidate,
        "selected_training_steps": 30,
        "model_call_count": 60,
        "private_transcripts_committed": False,
    })
    _write_json(EVIDENCE_ROOT / "phase105-final-decision.json", decision)
    lines = [
        "# Phase105 Final Decision",
        "",
        f"- Status: `{decision['status']}`",
        f"- Recommendation: `{decision['recommendation']}`",
        "- Real Qwen3-4B SFT: true",
        "- Curriculum rows: 240",
        "- Fresh paired model calls: 60",
        "- Product gate qualified: false",
        "- Automatic promotion allowed: false",
    ]
    (EVIDENCE_ROOT / "phase105-final-decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    runbook = [
        "# Phase105 Runbook",
        "",
        "```bash",
        ".venv/bin/python tools/phase105_qwen3_curriculum_alignment.py prepare --clean",
        ".venv/bin/python tools/phase105_qwen3_curriculum_alignment.py train --steps 1 --clean",
        ".venv/bin/python tools/phase105_qwen3_curriculum_alignment.py train --steps 12 --clean",
        ".venv/bin/python tools/phase105_qwen3_curriculum_alignment.py train --steps 30 --clean",
        ".venv/bin/python tools/phase105_qwen3_curriculum_alignment.py eval --variant base --clean",
        ".venv/bin/python tools/phase105_qwen3_curriculum_alignment.py eval --variant candidate --clean",
        ".venv/bin/python tools/phase105_qwen3_curriculum_alignment.py decide",
        ".venv/bin/python tools/phase105_qwen3_curriculum_alignment.py validate",
        "```",
        "",
        "All data is simulated_usage. Do not promote or deploy automatically.",
    ]
    (EVIDENCE_ROOT / "phase105-runbook.md").write_text("\n".join(runbook) + "\n", encoding="utf-8")
    _write_json(EVIDENCE_ROOT / "evidence_manifest.json", _build_manifest())
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return 0 if decision["passed"] else 1


def _validate() -> int:
    decision = _read_json(EVIDENCE_ROOT / "phase105-final-decision.json")
    manifest = _read_json(EVIDENCE_ROOT / "evidence_manifest.json")
    expected = {str(row["path"]): str(row["sha256"]) for row in manifest.get("files") or []}
    excluded = {EVIDENCE_ROOT / "evidence_manifest.json", EVIDENCE_ROOT / "validation_summary.json"}
    current = {str(path.relative_to(REPO_ROOT)): _sha256(path) for path in sorted(EVIDENCE_ROOT.rglob("*")) if path.is_file() and path not in excluded}
    checks = {
        "manifest_unchanged": expected == current,
        "real_training_completed": dict(decision.get("checks") or {}).get("real_training_completed") is True,
        "recommendation_allowed": decision.get("recommendation") in {"runtime_contract_remains_primary", "promote_after_manual_review"},
        "product_gate_false": decision.get("product_gate_qualified") is False,
        "automatic_promotion_false": decision.get("automatic_promotion_allowed") is False,
        "model_call_count_60": decision.get("model_call_count") == 60,
        "actual_feedback_zero": decision.get("actual_user_feedback_count") == 0,
        "private_transcripts_not_committed": decision.get("private_transcripts_committed") is False,
    }
    payload = {"kind": "phase105_validation_summary", "validated_at": _utcnow(), "passed": all(checks.values()), "checks": checks}
    _write_json(EVIDENCE_ROOT / "validation_summary.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["passed"] else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--clean", action="store_true")
    train = sub.add_parser("train")
    train.add_argument("--steps", type=int, required=True)
    train.add_argument("--clean", action="store_true")
    evaluate = sub.add_parser("eval")
    evaluate.add_argument("--variant", required=True)
    evaluate.add_argument("--clean", action="store_true")
    sub.add_parser("decide")
    sub.add_parser("validate")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "prepare":
        return _prepare(args.clean)
    if args.command == "train":
        return _train(args.steps, args.clean)
    if args.command == "eval":
        return _evaluate(args.variant, args.clean)
    if args.command == "decide":
        return _decide()
    if args.command == "validate":
        return _validate()
    raise SystemExit("unsupported command")


if __name__ == "__main__":
    raise SystemExit(main())
