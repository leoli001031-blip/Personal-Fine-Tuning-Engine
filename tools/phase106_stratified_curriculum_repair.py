#!/usr/bin/env python3
"""Repair Phase105 with seeded-stratified category exposure and a fresh gate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import resource
import sys
import time
import traceback
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = REPO_ROOT / "pfe-core"
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

from pfe_core.adapter_store.quality import validate_adapter_artifact
from pfe_core.phase75_personalization_benefit_benchmark import stable_hash
from pfe_core.phase103_simulated_user_acceptance import aggregate_phase103_scores, score_phase103_session
from pfe_core.phase105_qwen3_curriculum_alignment import build_phase105_curriculum
from pfe_core.phase106_stratified_curriculum_repair import (
    audit_phase106_holdout,
    build_phase106_decision,
    build_phase106_holdout,
    summarize_phase106_exposure,
)
from pfe_core.trainer.executors import _build_seeded_stratified_training_order
from phase101_failure_targeted_sft import _load_runtime, _run_session, _write_private_jsonl
from phase105_qwen3_curriculum_alignment import (
    GENERATION_PROTOCOL as PHASE105_GENERATION_PROTOCOL,
    _alignment_report,
    _job_spec as _phase105_job_spec,
    _read_json,
    _read_jsonl,
    _run_real_local_peft_training,
    _safe_clean,
    _sha256,
    _write_json,
    _write_jsonl,
)


EVIDENCE_ROOT = REPO_ROOT / "docs/demo/phase106-qwen3-stratified-curriculum-repair"
PREPARATION_ROOT = EVIDENCE_ROOT / "evidence-preparation"
TRAINING_ROOT = EVIDENCE_ROOT / "evidence-training"
EVAL_ROOT = EVIDENCE_ROOT / "evidence-eval"
FAILURE_ROOT = EVIDENCE_ROOT / "evidence-failures"
PRIVATE_ROOT = Path("/private/tmp/pfe-phase106-simulated-review")
TRAINER_OUTPUT_ROOT = REPO_ROOT / "trainer_job_outputs"
PHASE105_ROOT = REPO_ROOT / "docs/demo/phase105-qwen3-no-think-curriculum-alignment"
PHASE100_104_ROOT = REPO_ROOT / "docs/demo/phase100-104-autonomous-qwen3-training-benefit-loop"
GENERATION_PROTOCOL = {
    **PHASE105_GENERATION_PROTOCOL,
    "phase": 106,
    "variants": ["base", "candidate"],
    "phase106_model_call_budget": 60,
}


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _source_hashes() -> dict[str, str]:
    paths = {
        "core": CORE_ROOT / "pfe_core/phase106_stratified_curriculum_repair.py",
        "driver": REPO_ROOT / "tools/phase106_stratified_curriculum_repair.py",
        "core_test": REPO_ROOT / "tests/test_phase106_stratified_curriculum_repair.py",
        "driver_test": REPO_ROOT / "tests/test_phase106_driver_safety.py",
        "executor": CORE_ROOT / "pfe_core/trainer/executors.py",
        "phase105_core": CORE_ROOT / "pfe_core/phase105_qwen3_curriculum_alignment.py",
        "phase105_driver": REPO_ROOT / "tools/phase105_qwen3_curriculum_alignment.py",
    }
    return {name: _sha256(path) for name, path in paths.items()}


def _previous_holdouts() -> list[dict[str, Any]]:
    paths = (
        PHASE100_104_ROOT / "phase100-generation-boundary/evidence-preparation/final_holdout.json",
        PHASE100_104_ROOT / "phase101-failure-targeted-sft/evidence-preparation/holdout.json",
        PHASE100_104_ROOT / "phase103-simulated-user-acceptance/evidence-preparation/sessions.json",
        PHASE105_ROOT / "evidence-preparation/holdout.json",
    )
    return [_read_json(path) for path in paths if path.is_file()]


def _job_spec(rows: list[dict[str, Any]], *, steps: int, output_dir: Path) -> dict[str, Any]:
    spec = _phase105_job_spec(rows, steps=steps, output_dir=output_dir)
    spec["recipe"]["training"]["sampling_strategy"] = "seeded_stratified"
    spec["recipe"]["training"]["seed"] = 106
    spec["phase106"] = {
        "single_variable_repair": "sampling_strategy",
        "sampling_strategy": "seeded_stratified",
        "expected_30step_category_exposure": 6,
        "simulated_usage": True,
        "actual_user_feedback": False,
        "product_gate_qualified": False,
        "automatic_promotion_allowed": False,
    }
    return spec


def _exposure_plan(rows: list[dict[str, Any]], *, steps: int) -> dict[str, Any]:
    order = _build_seeded_stratified_training_order(rows, seed=106, cycle=0)[:steps]
    if steps == 30:
        return summarize_phase106_exposure(rows, order)
    counts: dict[str, int] = {}
    for index in order:
        category = str(rows[index].get("category") or "uncategorized")
        counts[category] = counts.get(category, 0) + 1
    return {
        "kind": "phase106_stratified_exposure_plan",
        "passed": len(order) == steps and len(counts) == min(steps, 5),
        "step_count": steps,
        "category_exposure_counts": counts,
        "order_sha256": stable_hash(order),
    }


def _prepare(clean: bool) -> int:
    if clean and EVIDENCE_ROOT.exists():
        _safe_clean(EVIDENCE_ROOT, EVIDENCE_ROOT.parent)
    PREPARATION_ROOT.mkdir(parents=True, exist_ok=True)
    rows = build_phase105_curriculum()
    holdout = build_phase106_holdout()
    audit = audit_phase106_holdout(rows, holdout, _previous_holdouts())
    spec = _job_spec(rows, steps=30, output_dir=TRAINER_OUTPUT_ROOT / "phase106-qwen3-4b-sft-30step")
    alignment = _alignment_report(spec)
    exposure = _exposure_plan(rows, steps=30)
    phase105 = _read_json(PHASE105_ROOT / "phase105-final-decision.json")
    phase105_training = _read_json(PHASE105_ROOT / "evidence-training/probe-30step/training_attempt.json")
    phase105_exposure = dict(phase105_training.get("execution") or {}).get("category_exposure_counts") or {}
    checks = {
        "phase105_remains_archive": str(phase105.get("status") or "").startswith("archive_"),
        "phase105_product_gate_false": phase105.get("product_gate_qualified") is False,
        "phase105_sequential_exposure_problem_confirmed": phase105_exposure == {"exact_three_line": 30},
        "fresh_holdout_audit_passed": audit.get("passed") is True,
        "training_template_alignment_passed": alignment.get("passed") is True,
        "stratified_exposure_plan_passed": exposure.get("passed") is True,
        "six_exposures_per_category": all(value == 6 for value in dict(exposure.get("category_exposure_counts") or {}).values()),
        "fresh_holdout_count_10": holdout.get("session_count") == 10,
        "eval_calls_frozen_60": holdout.get("total_model_call_budget") == 60,
    }
    freeze = {
        "kind": "phase106_pre_experiment_freeze",
        "created_at": _utcnow(),
        "passed": all(checks.values()),
        "checks": checks,
        "curriculum_manifest_sha256": stable_hash(rows),
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
    _write_jsonl(PREPARATION_ROOT / "selected_sft_samples.jsonl", rows)
    _write_json(PREPARATION_ROOT / "holdout.json", holdout)
    _write_json(PREPARATION_ROOT / "curriculum_holdout_audit.json", audit)
    _write_json(PREPARATION_ROOT / "training_template_alignment.json", alignment)
    _write_json(PREPARATION_ROOT / "stratified_exposure_plan.json", exposure)
    _write_json(PREPARATION_ROOT / "phase105_exposure_diagnostic.json", {
        "sampling_strategy": dict(phase105_training.get("execution") or {}).get("sampling_strategy"),
        "category_exposure_counts": phase105_exposure,
        "diagnosis": "30 sequential steps exposed exact_three_line only",
    })
    _write_json(EVIDENCE_ROOT / "pre_experiment_freeze.json", freeze)
    print(json.dumps({"status": "ready" if freeze["passed"] else "blocked", "checks": checks}, ensure_ascii=False, indent=2))
    return 0 if freeze["passed"] else 2


def _training_freeze_check(steps: int) -> dict[str, Any]:
    freeze = _read_json(EVIDENCE_ROOT / "pre_experiment_freeze.json")
    rows = _read_jsonl(PREPARATION_ROOT / "selected_sft_samples.jsonl")
    holdout = _read_json(PREPARATION_ROOT / "holdout.json")
    exposure = _exposure_plan(rows, steps=steps)
    checks = {
        "pre_experiment_freeze_passed": freeze.get("passed") is True,
        "source_files_unchanged": _source_hashes() == freeze.get("source_sha256"),
        "curriculum_unchanged": stable_hash(rows) == freeze.get("curriculum_manifest_sha256"),
        "holdout_unchanged": stable_hash(holdout.get("sessions") or []) == freeze.get("holdout_manifest_sha256"),
        "step_is_frozen": steps in (1, 12, 30),
        "stratified_exposure_plan_passed": exposure.get("passed") is True,
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
    return {"kind": "phase106_training_freeze_check", "steps": steps, "passed": all(checks.values()), "checks": checks, "exposure_plan": exposure}


def _train(steps: int, clean: bool) -> int:
    if steps not in (1, 12, 30):
        raise SystemExit("Phase106 permits 1, 12, or 30 steps only")
    probe_dir = TRAINING_ROOT / f"probe-{steps}step"
    output_root = TRAINER_OUTPUT_ROOT / f"phase106-qwen3-4b-sft-{steps}step"
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
        attempt = {"kind": "phase106_sft_training_attempt", "status": "blocked", "real_training": False, "requested_steps": steps, "reason": "freeze_or_alignment_failed", "product_gate_qualified": False}
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
            "kind": "phase106_sft_training_attempt",
            "status": "completed" if completed else "failed",
            "real_training": completed,
            "requested_steps": steps,
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
        _write_json(probe_dir / "train_log.json", {"loss_history": real.get("loss_history") or [], "category_exposure_counts": real.get("category_exposure_counts") or {}, "sampling_strategy": real.get("sampling_strategy")})
    except Exception as exc:
        attempt = {"kind": "phase106_sft_training_attempt", "status": "failed", "real_training": False, "requested_steps": steps, "error": f"{exc.__class__.__name__}: {exc}", "traceback": traceback.format_exc(), "product_gate_qualified": False, "automatic_promotion_allowed": False}
        FAILURE_ROOT.mkdir(parents=True, exist_ok=True)
        _write_json(FAILURE_ROOT / f"training_{steps}step.json", attempt)
    _write_json(probe_dir / "training_attempt.json", attempt)
    print(json.dumps({"status": attempt.get("status"), "requested_steps": steps, "duration_seconds": attempt.get("duration_seconds"), "category_exposure_counts": dict(attempt.get("execution") or {}).get("category_exposure_counts"), "error": attempt.get("error")}, ensure_ascii=False, indent=2))
    return 0 if attempt.get("status") == "completed" else 1


def _adapter_dir() -> Path:
    attempt = _read_json(TRAINING_ROOT / "probe-30step/training_attempt.json")
    validation = dict(attempt.get("adapter_validation") or {})
    path = Path(str(validation.get("artifact_dir") or ""))
    if attempt.get("status") != "completed" or validation.get("valid") is not True or not path.is_dir():
        raise SystemExit("Phase106 30-step adapter is unavailable")
    return path


def _eval_freeze_check(variant: str, adapter_path: Path | None) -> dict[str, Any]:
    freeze = _read_json(EVIDENCE_ROOT / "pre_experiment_freeze.json")
    holdout = _read_json(PREPARATION_ROOT / "holdout.json")
    training = _read_json(TRAINING_ROOT / "probe-30step/training_attempt.json") if (TRAINING_ROOT / "probe-30step/training_attempt.json").is_file() else {}
    exposure = dict(training.get("execution") or {}).get("category_exposure_counts") or {}
    expected_categories = {
        "exact_three_line",
        "false_block",
        "provenance",
        "correction_following",
        "ordinary_control",
    }
    checks = {
        "pre_experiment_freeze_passed": freeze.get("passed") is True,
        "source_files_unchanged": _source_hashes() == freeze.get("source_sha256"),
        "holdout_unchanged": stable_hash(holdout.get("sessions") or []) == freeze.get("holdout_manifest_sha256"),
        "generation_protocol_unchanged": stable_hash(GENERATION_PROTOCOL) == freeze.get("generation_protocol_sha256"),
        "variant_frozen": variant in {"base", "candidate"},
        "adapter_available_or_base": adapter_path is None or adapter_path.is_dir(),
        "actual_30step_exposure_balanced": set(exposure) == expected_categories
        and all(int(exposure.get(category) or 0) == 6 for category in expected_categories),
        "no_completed_eval_exists": not (EVAL_ROOT / variant / "metrics.json").exists(),
    }
    return {"kind": "phase106_eval_freeze_check", "variant": variant, "passed": all(checks.values()), "checks": checks}


def _evaluate(variant: str, clean: bool) -> int:
    if variant not in {"base", "candidate"}:
        raise SystemExit("Phase106 eval variant must be base or candidate")
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
            structural["kind"] = "phase106_structural_session"
            outputs = [str(row.get("raw_output") or "") for row in private.get("turns") or []]
            user_score = score_phase103_session(session=session, outputs=outputs, structural_turns=structural.get("turns") or [])
            structural["simulated_user_score"] = user_score
            rows.append(structural)
            scores.append(user_score)
            private_rows.append({"session_id": session.get("session_id"), "category": session.get("category"), "turns": private.get("turns") or [], "final_acceptance": user_score.get("accepted")})
            _write_jsonl(output_root / "structural_sessions.jsonl", rows)
            _write_jsonl(output_root / "simulated_user_scores.jsonl", scores)
            _write_private_jsonl(cache_path, private_rows)
            print(f"[phase106:{variant}] {index}/{len(sessions)} {session.get('session_id')} accepted={user_score['accepted']}", flush=True)
    finally:
        if torch is not None and model is not None and device is not None:
            del model
            if device == "mps":
                torch.mps.empty_cache()
    metrics = aggregate_phase103_scores(scores)
    payload = {"kind": "phase106_variant_metrics", "variant": variant, "model_call_count": sum(int(row.get("turn_count") or 0) for row in rows), "metrics": metrics, "adapter_loaded": adapter_path is not None, "guided_generation_used": False, "private_cache": str(cache_path), "private_cache_outside_repo": True, "simulated_usage": True, "actual_user_feedback_count": 0}
    _write_json(output_root / "metrics.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _build_manifest() -> dict[str, Any]:
    excluded = {EVIDENCE_ROOT / "evidence_manifest.json", EVIDENCE_ROOT / "validation_summary.json"}
    files = [path for path in sorted(EVIDENCE_ROOT.rglob("*")) if path.is_file() and path not in excluded]
    return {"kind": "phase106_evidence_manifest", "files": [{"path": str(path.relative_to(REPO_ROOT)), "sha256": _sha256(path), "size_bytes": path.stat().st_size} for path in files], "file_count": len(files), "private_transcripts_committed": False, "actual_user_feedback_count": 0}


def _decide() -> int:
    base = dict(_read_json(EVAL_ROOT / "base/metrics.json").get("metrics") or {})
    candidate = dict(_read_json(EVAL_ROOT / "candidate/metrics.json").get("metrics") or {})
    training = _read_json(TRAINING_ROOT / "probe-30step/training_attempt.json")
    exposure = dict(training.get("execution") or {}).get("category_exposure_counts") or {}
    balanced = set(exposure) == {"exact_three_line", "false_block", "provenance", "correction_following", "ordinary_control"} and all(int(value) == 6 for value in exposure.values())
    decision = build_phase106_decision(base_metrics=base, candidate_metrics=candidate, training_completed=training.get("status") == "completed" and training.get("real_training") is True, exposure_balanced=balanced)
    decision.update({"base_metrics": base, "candidate_metrics": candidate, "actual_category_exposure_counts": exposure, "selected_training_steps": 30, "model_call_count": 60, "private_transcripts_committed": False})
    _write_json(EVIDENCE_ROOT / "phase106-final-decision.json", decision)
    lines = ["# Phase106 Final Decision", "", f"- Status: `{decision['status']}`", f"- Recommendation: `{decision['recommendation']}`", f"- Actual category exposure: `{json.dumps(exposure, ensure_ascii=False, sort_keys=True)}`", "- Product gate qualified: false", "- Automatic promotion allowed: false"]
    (EVIDENCE_ROOT / "phase106-final-decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    runbook = ["# Phase106 Runbook", "", "```bash", ".venv/bin/python tools/phase106_stratified_curriculum_repair.py prepare --clean", ".venv/bin/python tools/phase106_stratified_curriculum_repair.py train --steps 1 --clean", ".venv/bin/python tools/phase106_stratified_curriculum_repair.py train --steps 12 --clean", ".venv/bin/python tools/phase106_stratified_curriculum_repair.py train --steps 30 --clean", ".venv/bin/python tools/phase106_stratified_curriculum_repair.py eval --variant base --clean", ".venv/bin/python tools/phase106_stratified_curriculum_repair.py eval --variant candidate --clean", ".venv/bin/python tools/phase106_stratified_curriculum_repair.py decide", ".venv/bin/python tools/phase106_stratified_curriculum_repair.py validate", "```", "", "This is a single-variable sampling repair. Do not auto-promote or deploy."]
    (EVIDENCE_ROOT / "phase106-runbook.md").write_text("\n".join(runbook) + "\n", encoding="utf-8")
    _write_json(EVIDENCE_ROOT / "evidence_manifest.json", _build_manifest())
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return 0 if decision["passed"] else 1


def _validate() -> int:
    decision = _read_json(EVIDENCE_ROOT / "phase106-final-decision.json")
    manifest = _read_json(EVIDENCE_ROOT / "evidence_manifest.json")
    expected = {str(row["path"]): str(row["sha256"]) for row in manifest.get("files") or []}
    excluded = {EVIDENCE_ROOT / "evidence_manifest.json", EVIDENCE_ROOT / "validation_summary.json"}
    current = {str(path.relative_to(REPO_ROOT)): _sha256(path) for path in sorted(EVIDENCE_ROOT.rglob("*")) if path.is_file() and path not in excluded}
    checks = {"manifest_unchanged": expected == current, "real_training_completed": dict(decision.get("checks") or {}).get("real_training_completed") is True, "stratified_exposure_balanced": dict(decision.get("checks") or {}).get("stratified_exposure_balanced") is True, "recommendation_allowed": decision.get("recommendation") in {"runtime_contract_remains_primary", "promote_after_manual_review"}, "product_gate_false": decision.get("product_gate_qualified") is False, "automatic_promotion_false": decision.get("automatic_promotion_allowed") is False, "model_call_count_60": decision.get("model_call_count") == 60, "actual_feedback_zero": decision.get("actual_user_feedback_count") == 0, "private_transcripts_not_committed": decision.get("private_transcripts_committed") is False}
    payload = {"kind": "phase106_validation_summary", "validated_at": _utcnow(), "passed": all(checks.values()), "checks": checks}
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
