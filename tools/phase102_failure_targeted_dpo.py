#!/usr/bin/env python3
"""Run Phase102 Qwen3-4B DPO from frozen failure-targeted preferences."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import resource
import shutil
import sys
import time
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = REPO_ROOT / "pfe-core"
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

from pfe_core.adapter_store.quality import validate_adapter_artifact
from pfe_core.phase75_personalization_benefit_benchmark import stable_hash
from pfe_core.phase93_95_dpo_product_proof import aggregate_phase94_scores
from pfe_core.phase102_failure_targeted_dpo import (
    audit_phase102_pairs,
    build_phase102_dpo_decision,
    select_phase102_dpo_pairs,
)
from pfe_core.trainer.executors import execute_dpo_training
from phase101_failure_targeted_sft import _load_runtime, _run_session, _write_private_jsonl


EVIDENCE_ROOT = REPO_ROOT / "docs/demo/phase100-104-autonomous-qwen3-training-benefit-loop"
PHASE100_ROOT = EVIDENCE_ROOT / "phase100-generation-boundary"
PHASE101_ROOT = EVIDENCE_ROOT / "phase101-failure-targeted-sft"
PHASE_ROOT = EVIDENCE_ROOT / "phase102-failure-targeted-dpo"
PREPARATION_ROOT = PHASE_ROOT / "evidence-preparation"
TRAINING_ROOT = PHASE_ROOT / "evidence-training"
EVAL_ROOT = PHASE_ROOT / "evidence-eval"
PRIVATE_ROOT = Path("/private/tmp/pfe-phase102-simulated-review")
MODEL_PATH = REPO_ROOT / "models/Qwen3-4B"
TRAINER_OUTPUT_ROOT = REPO_ROOT / "trainer_job_outputs/phase102-qwen3-4b-dpo"
DPO_RUNTIME = {
    "runtime_device": "mps",
    "runtime_dtype": "float32",
    "learning_rate": 0.000005,
    "beta": 0.1,
    "max_length": 192,
    "max_prompt_length": 128,
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.0,
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
        "core": CORE_ROOT / "pfe_core/phase102_failure_targeted_dpo.py",
        "driver": REPO_ROOT / "tools/phase102_failure_targeted_dpo.py",
        "core_test": REPO_ROOT / "tests/test_phase102_failure_targeted_dpo.py",
        "driver_test": REPO_ROOT / "tests/test_phase102_driver_safety.py",
        "executor": CORE_ROOT / "pfe_core/trainer/executors.py",
        "phase101_driver": REPO_ROOT / "tools/phase101_failure_targeted_sft.py",
    }
    return {name: _sha256(path) for name, path in paths.items()}


def _job_spec(rows: list[dict[str, Any]], *, steps: int, output_dir: Path) -> dict[str, Any]:
    return {
        "backend": "dpo",
        "execution_backend": "dpo",
        "execution_executor": "dpo",
        "executor_mode": "real_import",
        "dry_run": True,
        "output_dir": str(output_dir),
        "recipe": {
            "training": {
                "method": "lora",
                "epochs": 1,
                "max_steps": steps,
                "learning_rate": DPO_RUNTIME["learning_rate"],
                "train_type": "dpo",
                "base_model": str(MODEL_PATH),
                "base_model_path": str(MODEL_PATH),
                "local_only": True,
                "num_train_samples": len(rows),
                "output_dir": str(output_dir),
                "runtime_device": DPO_RUNTIME["runtime_device"],
                "runtime_dtype": DPO_RUNTIME["runtime_dtype"],
            },
            "peft": {
                "trainer_class": "trl.DPOTrainer",
                "dpo_config": {
                    "beta": DPO_RUNTIME["beta"],
                    "label_smoothing": 0.0,
                    "max_length": DPO_RUNTIME["max_length"],
                    "max_prompt_length": DPO_RUNTIME["max_prompt_length"],
                },
                "lora_config": {
                    "r": DPO_RUNTIME["lora_r"],
                    "lora_alpha": DPO_RUNTIME["lora_alpha"],
                    "lora_dropout": DPO_RUNTIME["lora_dropout"],
                },
            },
        },
        "training_examples": [dict(row) for row in rows],
        "phase102": {
            "failure_targeted": True,
            "starts_from_base_because_phase101_sft_archived": True,
            "simulated_usage": True,
            "actual_user_feedback": False,
            "automatic_promotion_allowed": False,
        },
    }


def _prepare(clean: bool) -> int:
    if clean and PHASE_ROOT.exists():
        _safe_clean(PHASE_ROOT, EVIDENCE_ROOT)
    PREPARATION_ROOT.mkdir(parents=True, exist_ok=True)
    candidates = _read_jsonl(PHASE101_ROOT / "evidence-preparation/selected_sft_samples.jsonl")
    holdout = _read_json(PHASE101_ROOT / "evidence-preparation/holdout.json")
    pairs = select_phase102_dpo_pairs(candidates)
    audit = audit_phase102_pairs(pairs, holdout)
    phase101 = _read_json(PHASE101_ROOT / "phase101-decision.json")
    spec12 = _job_spec(pairs, steps=12, output_dir=TRAINER_OUTPUT_ROOT / "12step")
    spec30 = _job_spec(pairs, steps=30, output_dir=TRAINER_OUTPUT_ROOT / "30step")
    dry12 = execute_dpo_training(job_spec=spec12, dry_run=True)
    resolution = dict(dry12.get("training_config") or {}).get("runtime_resolution") or {}
    checks = {
        "phase101_sft_remains_archive": str(phase101.get("status") or "").startswith("archive_"),
        "phase101_product_gate_false": phase101.get("product_gate_qualified") is False,
        "pair_audit_passed": audit.get("passed") is True,
        "pair_count_24": len(pairs) == 24,
        "runtime_resolves_mps_float32": resolution.get("device") == "mps" and resolution.get("dtype") == "float32",
        "starts_from_base_not_archived_sft": "incremental_context" not in spec12["recipe"]["training"],
        "holdout_is_phase101_frozen": stable_hash(holdout.get("sessions") or []) == _read_json(PHASE101_ROOT / "pre_experiment_freeze.json").get("holdout_manifest_sha256"),
    }
    freeze = {
        "kind": "phase102_pre_training_freeze",
        "created_at": _utcnow(),
        "passed": all(checks.values()),
        "checks": checks,
        "pair_manifest_sha256": stable_hash(pairs),
        "holdout_manifest_sha256": stable_hash(holdout.get("sessions") or []),
        "job_spec_sha256": {"12": stable_hash(spec12), "30": stable_hash(spec30)},
        "source_sha256": _source_hashes(),
        "runtime": DPO_RUNTIME,
        "simulated_usage": True,
        "actual_user_feedback_count": 0,
        "automatic_promotion_allowed": False,
    }
    _write_jsonl(PREPARATION_ROOT / "selected_dpo_pairs.jsonl", pairs)
    _write_json(PREPARATION_ROOT / "pair_holdout_audit.json", audit)
    _write_json(PREPARATION_ROOT / "job_spec_12step.json", spec12)
    _write_json(PREPARATION_ROOT / "job_spec_30step.json", spec30)
    _write_json(PREPARATION_ROOT / "dry_run_12step.json", dry12)
    _write_json(PHASE_ROOT / "pre_training_freeze.json", freeze)
    print(json.dumps({"status": "ready" if freeze["passed"] else "blocked", "checks": checks}, ensure_ascii=False, indent=2))
    return 0 if freeze["passed"] else 2


def _training_freeze_check(steps: int) -> dict[str, Any]:
    freeze = _read_json(PHASE_ROOT / "pre_training_freeze.json")
    rows = _read_jsonl(PREPARATION_ROOT / "selected_dpo_pairs.jsonl")
    spec = _job_spec(rows, steps=steps, output_dir=TRAINER_OUTPUT_ROOT / f"{steps}step")
    checks = {
        "pre_training_freeze_passed": freeze.get("passed") is True,
        "source_files_unchanged": _source_hashes() == freeze.get("source_sha256"),
        "pairs_unchanged": stable_hash(rows) == freeze.get("pair_manifest_sha256"),
        "job_spec_unchanged": stable_hash(spec) == dict(freeze.get("job_spec_sha256") or {}).get(str(steps)),
        "step_is_frozen": steps in (12, 30),
    }
    if steps == 30:
        prior_path = TRAINING_ROOT / "12step/training_attempt.json"
        prior = _read_json(prior_path) if prior_path.is_file() else {}
        log_path = TRAINING_ROOT / "12step/train_log.json"
        log = _read_json(log_path) if log_path.is_file() else {}
        values = [row.get("loss") for row in log.get("loss_history") or [] if isinstance(row, Mapping) and "loss" in row]
        checks["twelve_step_completed"] = prior.get("status") == "completed"
        checks["twelve_step_adapter_valid"] = dict(prior.get("adapter_validation") or {}).get("valid") is True
        checks["twelve_step_metrics_finite"] = not bool(log.get("non_finite_metrics")) and all(math.isfinite(float(value)) for value in values)
    return {"kind": "phase102_training_freeze_check", "steps": steps, "passed": all(checks.values()), "checks": checks}


def _train(steps: int, clean: bool) -> int:
    if steps not in (12, 30):
        raise SystemExit("Phase102 permits 12-step and 30-step probes only")
    evidence_dir = TRAINING_ROOT / f"{steps}step"
    output_dir = TRAINER_OUTPUT_ROOT / f"{steps}step"
    if clean and evidence_dir.exists():
        _safe_clean(evidence_dir, TRAINING_ROOT)
    if clean and output_dir.exists():
        _safe_clean(output_dir, TRAINER_OUTPUT_ROOT)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_jsonl(PREPARATION_ROOT / "selected_dpo_pairs.jsonl")
    spec = _job_spec(rows, steps=steps, output_dir=output_dir)
    freeze = _training_freeze_check(steps)
    _write_json(evidence_dir / "freeze_check.json", freeze)
    _write_json(evidence_dir / "job_spec.json", spec)
    if not freeze["passed"]:
        attempt = {"kind": "phase102_dpo_training_attempt", "status": "blocked", "requested_steps": steps, "reason": "freeze_check_failed", "freeze_check": freeze, "product_gate_qualified": False}
        _write_json(evidence_dir / "training_attempt.json", attempt)
        return 2
    started = time.perf_counter()
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = execute_dpo_training(job_spec={**spec, "dry_run": False}, dry_run=False)
    duration = round(time.perf_counter() - started, 4)
    real = dict(result.get("real_execution") or {})
    artifact_dir = Path(str(real.get("artifact_dir") or ""))
    validation = validate_adapter_artifact(artifact_dir, {"artifact_name": "adapter_model.safetensors", "artifact_format": "peft_lora"}) if artifact_dir.is_dir() else {"valid": False, "reason": "artifact_dir_missing"}
    adapter_path = artifact_dir / "adapter_model.safetensors"
    validation.update({
        "artifact_dir": str(artifact_dir),
        "sha256": _sha256(adapter_path) if adapter_path.is_file() else None,
        "requested_steps": steps,
        "completed_steps": int(real.get("steps") or 0),
        "lineage": "qwen3_4b_base_to_phase102_dpo",
    })
    completed = (
        result.get("status") == "completed"
        and real.get("success") is True
        and int(real.get("steps") or 0) == steps
        and real.get("parameters_updated") is True
        and validation.get("valid") is True
    )
    attempt = {
        "kind": "phase102_dpo_training_attempt",
        "status": "completed" if completed else "failed",
        "real_training": completed,
        "requested_steps": steps,
        "completed_steps": int(real.get("steps") or 0),
        "duration_seconds": duration,
        "resource_usage": {
            "ru_maxrss_before": rss_before,
            "ru_maxrss_after": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "ru_maxrss_unit": "bytes_on_macos",
        },
        "runtime_resolution": dict(result.get("training_config") or {}).get("runtime_resolution"),
        "result": result,
        "adapter_validation": validation,
        "simulated_usage": True,
        "actual_user_feedback": False,
        "product_gate_qualified": False,
        "automatic_promotion_allowed": False,
    }
    _write_json(evidence_dir / "training_attempt.json", attempt)
    _write_json(evidence_dir / "train_log.json", {
        "status": attempt["status"],
        "requested_steps": steps,
        "completed_steps": attempt["completed_steps"],
        "loss_history": real.get("loss_history") or [],
        "non_finite_metrics": real.get("non_finite_metrics") or [],
        "parameter_fingerprint_before": real.get("parameter_fingerprint_before"),
        "parameter_fingerprint_after": real.get("parameter_fingerprint_after"),
        "parameters_updated": real.get("parameters_updated"),
        "runtime_audit": real.get("runtime_audit"),
        "error": result.get("error"),
    })
    _write_json(evidence_dir / "adapter_validation.json", validation)
    print(json.dumps({
        "status": attempt["status"],
        "requested_steps": steps,
        "completed_steps": attempt["completed_steps"],
        "duration_seconds": duration,
        "error": result.get("error"),
        "adapter_valid": validation.get("valid"),
    }, ensure_ascii=False, indent=2))
    return 0 if completed else 1


def _adapter_dir() -> Path:
    attempt = _read_json(TRAINING_ROOT / "30step/training_attempt.json")
    validation = dict(attempt.get("adapter_validation") or {})
    path = Path(str(validation.get("artifact_dir") or ""))
    if attempt.get("status") != "completed" or validation.get("valid") is not True or not path.is_dir():
        raise SystemExit("Phase102 30-step adapter is unavailable")
    return path


def _eval_freeze_check(adapter_path: Path) -> dict[str, Any]:
    freeze = _read_json(PHASE_ROOT / "pre_training_freeze.json")
    phase101_freeze = _read_json(PHASE101_ROOT / "pre_experiment_freeze.json")
    holdout = _read_json(PHASE101_ROOT / "evidence-preparation/holdout.json")
    checks = {
        "pre_training_freeze_passed": freeze.get("passed") is True,
        "source_files_unchanged": _source_hashes() == freeze.get("source_sha256"),
        "phase101_holdout_unchanged": stable_hash(holdout.get("sessions") or []) == phase101_freeze.get("holdout_manifest_sha256"),
        "adapter_available": adapter_path.is_dir(),
        "no_completed_eval_exists": not (EVAL_ROOT / "metrics.json").exists(),
    }
    return {"kind": "phase102_eval_freeze_check", "passed": all(checks.values()), "checks": checks}


def _evaluate(clean: bool) -> int:
    adapter_path = _adapter_dir()
    if clean and EVAL_ROOT.exists():
        _safe_clean(EVAL_ROOT, PHASE_ROOT)
    cache_path = PRIVATE_ROOT / "dpo.jsonl"
    if clean:
        cache_path.unlink(missing_ok=True)
    freeze = _eval_freeze_check(adapter_path)
    _write_json(EVAL_ROOT / "freeze_check.json", freeze)
    if not freeze["passed"]:
        return 2
    sessions = [dict(row) for row in _read_json(PHASE101_ROOT / "evidence-preparation/holdout.json").get("sessions") or []]
    rows = []
    private_rows = []
    torch = tokenizer = model = device = None
    try:
        torch, tokenizer, model, device = _load_runtime(adapter_path)
        for index, session in enumerate(sessions, start=1):
            structural, private = _run_session(session=session, torch=torch, tokenizer=tokenizer, model=model, device=device)
            structural["kind"] = "phase102_structural_session"
            rows.append(structural)
            private_rows.append(private)
            _write_jsonl(EVAL_ROOT / "structural_sessions.jsonl", rows)
            _write_private_jsonl(cache_path, private_rows)
            print(f"[phase102:dpo] {index}/{len(sessions)} {session.get('session_id')} completed", flush=True)
    finally:
        if torch is not None and model is not None and device is not None:
            del model
            if device == "mps":
                torch.mps.empty_cache()
    details = [{"category": row.get("category"), **dict(row.get("raw_score") or {})} for row in rows]
    turns = [turn for row in rows for turn in row.get("turns") or []]
    metrics = aggregate_phase94_scores(details)
    metrics.update({
        "extra_text_after_first_answer_rate": round(sum(row.get("extra_text_after_first_answer") is True for row in details) / len(details), 4),
        "forbidden_generation_rate": round(sum(row.get("forbidden_generation") is True for row in details) / len(details), 4),
        "complete_content_before_termination_rate": round(sum(row.get("complete_content_before_termination") is True for row in turns) / len(turns), 4),
        "native_termination_rate": round(sum(row.get("native_termination") is True for row in turns) / len(turns), 4),
        "runtime_control_dependency_rate": 0.0,
    })
    payload = {
        "kind": "phase102_dpo_metrics",
        "session_count": len(rows),
        "model_call_count": sum(int(row.get("turn_count") or 0) for row in rows),
        "metrics": metrics,
        "adapter_loaded": True,
        "guided_generation_used": False,
        "private_cache": str(cache_path),
        "private_cache_outside_repo": True,
        "simulated_usage": True,
        "actual_user_feedback_count": 0,
    }
    _write_json(EVAL_ROOT / "metrics.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _decide() -> int:
    base = dict(_read_json(PHASE101_ROOT / "evidence-eval/base/metrics.json").get("metrics") or {})
    sft = dict(_read_json(PHASE101_ROOT / "evidence-eval/sft/metrics.json").get("metrics") or {})
    runtime = dict(_read_json(PHASE100_ROOT / "evidence-eval/metrics.json").get("raw") or {})
    runtime["runtime_control_dependency_rate"] = 0.25
    training_path = TRAINING_ROOT / "30step/training_attempt.json"
    training = _read_json(training_path) if training_path.is_file() else {}
    eval_path = EVAL_ROOT / "metrics.json"
    candidate = dict(_read_json(eval_path).get("metrics") or {}) if eval_path.is_file() else {}
    decision = build_phase102_dpo_decision(
        base_metrics=base,
        sft_metrics=sft,
        runtime_metrics=runtime,
        candidate_metrics=candidate,
        training_completed=training.get("status") == "completed" and training.get("real_training") is True,
    )
    decision.update({
        "base_metrics": base,
        "archived_sft_metrics": sft,
        "runtime_contract_metrics": runtime,
        "candidate_metrics": candidate,
        "selected_training_steps": 30 if training else None,
        "phase102_model_call_count": 24 if candidate else 0,
        "cumulative_model_call_count": 120 if candidate else 96,
        "long_run_total_call_budget": 270,
    })
    _write_json(PHASE_ROOT / "phase102-decision.json", decision)
    lines = [
        "# Phase102 Decision",
        "",
        f"- Status: `{decision['status']}`",
        f"- Passed: {str(decision['passed']).lower()}",
        f"- Real DPO training completed: {str(decision['checks']['real_dpo_training_completed']).lower()}",
        "- Product gate qualified: false",
        "- Automatic promotion allowed: false",
    ]
    (PHASE_ROOT / "phase102-decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return 0 if decision["passed"] else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--clean", action="store_true")
    train = sub.add_parser("train")
    train.add_argument("--steps", type=int, required=True)
    train.add_argument("--clean", action="store_true")
    evaluate = sub.add_parser("eval")
    evaluate.add_argument("--clean", action="store_true")
    sub.add_parser("decide")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "prepare":
        return _prepare(args.clean)
    if args.command == "train":
        return _train(args.steps, args.clean)
    if args.command == "eval":
        return _evaluate(args.clean)
    if args.command == "decide":
        return _decide()
    raise SystemExit("unsupported command")


if __name__ == "__main__":
    raise SystemExit(main())
