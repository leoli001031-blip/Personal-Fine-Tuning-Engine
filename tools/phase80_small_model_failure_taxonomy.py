#!/usr/bin/env python3
"""Run the Phase80 small-model failure taxonomy experiment."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import resource
import shutil
import subprocess
import sys
import time
import traceback
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = REPO_ROOT / "pfe-core"
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

import phase75_personalization_benefit_benchmark as phase75_driver
import phase79_cpu_feasible_persona_probe as phase79_driver
from pfe_core.adapter_store.quality import validate_adapter_artifact
from pfe_core.phase75_personalization_benefit_benchmark import (
    aggregate_phase75_variant,
    stable_hash,
)
from pfe_core.phase77_private_value_guarded_runtime import build_phase77_holdout
from pfe_core.phase78_persona_internalization_training import (
    PHASE78_PERSONA_CATEGORIES,
    PHASE78_TRAINING_SAMPLE_COUNT,
    audit_phase78_training_samples,
    build_phase78_holdout,
    build_phase78_training_samples,
)
from pfe_core.phase79_cpu_feasible_persona_probe import build_phase79_holdout
from pfe_core.phase80_small_model_failure_taxonomy import (
    PHASE80_SESSION_COUNT,
    PHASE80_VARIANTS,
    audit_phase80_isolation,
    build_phase80_decision,
    build_phase80_holdout,
)
from pfe_core.trainer.executors import _run_real_local_peft_training


EVIDENCE_ROOT = REPO_ROOT / "docs/demo/phase80-small-model-failure-taxonomy"
PREPARATION_ROOT = EVIDENCE_ROOT / "evidence-preparation"
TRAINING_ROOT = EVIDENCE_ROOT / "evidence-real-training"
GENERATION_ROOT = EVIDENCE_ROOT / "evidence-real-generation"
FAILURE_ROOT = EVIDENCE_ROOT / "evidence-failures"
MODEL_05_PATH = REPO_ROOT / "models/Qwen2.5-0.5B-Instruct"
MODEL_4B_PATH = REPO_ROOT / "models/Qwen3-4B"
TRAINER_OUTPUT_ROOT = REPO_ROOT / "trainer_job_outputs"
PHASE79_ROOT = REPO_ROOT / "docs/demo/phase79-cpu-feasible-persona-adapter-probe"
CORE_SOURCE = CORE_ROOT / "pfe_core/phase80_small_model_failure_taxonomy.py"
DRIVER_SOURCE = REPO_ROOT / "tools/phase80_small_model_failure_taxonomy.py"
TEST_SOURCE = REPO_ROOT / "tests/test_phase80_small_model_failure_taxonomy.py"
EXECUTOR_SOURCE = CORE_ROOT / "pfe_core/trainer/executors.py"
GENERATION_PROTOCOL = {
    **phase75_driver.GENERATION_PROTOCOL,
    "kind": "phase80_frozen_failure_taxonomy_generation_protocol",
    "variants": list(PHASE80_VARIANTS),
    "standard_max_new_tokens": 192,
    "standard_repetition_penalty": 1.05,
    "stop_control_max_new_tokens": 192,
    "stop_control_repetition_penalty": 1.15,
    "stop_control_no_repeat_ngram_size": 4,
    "same_holdout_all_variants": True,
    "runtime_variants_use_phase77_conditional_contract": True,
    "private_value_guard_before_every_call": True,
    "score_or_gate_relaxation_allowed": False,
}
DYNAMIC_FILES = {
    "evidence_manifest.json",
    "evidence_integrity.json",
    "finalization_state.json",
    "validation_gate.txt",
    "validation_summary.json",
}


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return dict(value) if isinstance(value, Mapping) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(
            json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _phase79_attempt() -> dict[str, Any]:
    return _read_json(
        PHASE79_ROOT / "evidence-real-training/probe-12step/training_attempt.json"
    )


def _phase79_adapter_path() -> Path:
    attempt = _phase79_attempt()
    path = Path(str(dict(attempt.get("adapter_validation") or {}).get("artifact_dir") or ""))
    if attempt.get("status") != "completed" or not path.is_dir():
        raise SystemExit("Phase80 requires the saved valid Phase79 12-step adapter")
    return path.resolve()


def _low_lr_job_spec(samples: Iterable[Mapping[str, Any]], output_dir: Path) -> dict[str, Any]:
    spec = phase79_driver._phase79_job_spec(samples, output_dir, 12)
    spec["recipe"]["training"]["learning_rate"] = 1e-5
    spec["recipe"]["training"]["seed"] = 80
    spec["phase80"] = {
        "hypothesis": "lower_learning_rate_may_remove_phase79_adapter_instability",
        "selected_model": "Qwen2.5-0.5B-Instruct",
        "requested_steps": 12,
        "learning_rate": 1e-5,
        "same_curriculum_as_phase79": True,
        "phase79_adapter_reused_as_new_candidate": False,
        "simulated_usage": True,
        "actual_user_feedback": False,
        "auto_promotion_allowed": False,
    }
    return spec


def _previous_holdouts() -> list[dict[str, Any]]:
    return (
        build_phase77_holdout()["sessions"]
        + build_phase78_holdout()["sessions"]
        + build_phase79_holdout()["sessions"]
    )


def _model_inventory() -> dict[str, Any]:
    phase79_attempt = _phase79_attempt()
    phase79_validation = dict(phase79_attempt.get("adapter_validation") or {})
    checks = {
        "qwen25_0_5b_config_exists": (MODEL_05_PATH / "config.json").exists(),
        "qwen25_0_5b_weights_exist": bool(list(MODEL_05_PATH.glob("*.safetensors"))),
        "qwen3_4b_config_exists": (MODEL_4B_PATH / "config.json").exists(),
        "qwen3_4b_weights_exist": bool(list(MODEL_4B_PATH.glob("*.safetensors"))),
        "phase79_real_training_completed": phase79_attempt.get("status") == "completed",
        "phase79_adapter_valid": phase79_validation.get("valid") is True,
        "phase79_adapter_artifact_exists": _phase79_adapter_path().is_dir(),
    }
    return {
        "kind": "phase80_model_inventory",
        "passed": all(checks.values()),
        "checks": checks,
        "training_model": str(MODEL_05_PATH),
        "capacity_reference_model": str(MODEL_4B_PATH),
        "phase79_high_lr_adapter": str(_phase79_adapter_path()),
        "phase79_adapter_reused_as_new_candidate": False,
        "model_weight_bytes": {
            "Qwen2.5-0.5B-Instruct": sum(path.stat().st_size for path in MODEL_05_PATH.glob("*.safetensors")),
            "Qwen3-4B": sum(path.stat().st_size for path in MODEL_4B_PATH.glob("*.safetensors")),
        },
    }


def _prepare(clean: bool) -> int:
    if clean and EVIDENCE_ROOT.exists():
        shutil.rmtree(EVIDENCE_ROOT)
    required = (
        CORE_SOURCE,
        DRIVER_SOURCE,
        TEST_SOURCE,
        EXECUTOR_SOURCE,
        PHASE79_ROOT / "phase79-final-decision.json",
        PHASE79_ROOT / "comparison_summary.json",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"Phase80 required sources missing: {missing}")
    holdout = build_phase80_holdout()
    isolation = audit_phase80_isolation(holdout["sessions"], _previous_holdouts())
    samples = build_phase78_training_samples()
    quality = audit_phase78_training_samples(samples)
    inventory = _model_inventory()
    phase79_decision = _read_json(PHASE79_ROOT / "phase79-final-decision.json")
    phase79_comparison = _read_json(PHASE79_ROOT / "comparison_summary.json")
    preview = _low_lr_job_spec(samples, TRAINER_OUTPUT_ROOT / "phase80-preview")
    boundary = phase79_driver._completion_boundary_report(preview)
    phase79_checks = {
        "phase79_archived_on_sanity_failure": phase79_decision.get("status")
        == "archive_12_step_sanity_failed",
        "phase79_training_success_separate_from_benefit": phase79_decision.get("training_success")
        is True
        and phase79_decision.get("adapter_benefit") == "not_evaluated_on_full_holdout",
        "phase79_adapter_negative_on_sanity": float(
            phase79_comparison.get("adapter_target_gain_vs_base") or 0.0
        ) < 0.0,
        "phase79_no_product_claim": phase79_decision.get("actual_product_benefit_claim_allowed")
        is False,
    }
    checks = {
        "fresh_holdout_isolation_passed": isolation.get("passed") is True,
        "training_quality_passed": quality.get("passed") is True,
        "completion_boundary_passed": boundary.get("passed") is True,
        "model_inventory_passed": inventory.get("passed") is True,
        "phase79_failure_acknowledged": all(phase79_checks.values()),
    }
    freeze = {
        "kind": "phase80_pre_experiment_freeze",
        "frozen_at": _utcnow(),
        "frozen_before_low_lr_training": True,
        "passed": all(checks.values()),
        "checks": checks,
        "holdout_manifest_sha256": stable_hash(holdout["sessions"]),
        "training_manifest_sha256": stable_hash(samples),
        "isolation_audit_sha256": stable_hash(isolation),
        "training_quality_sha256": stable_hash(quality),
        "completion_boundary_sha256": stable_hash(boundary),
        "model_inventory_sha256": stable_hash(inventory),
        "phase79_decision_sha256": stable_hash(phase79_decision),
        "phase79_adapter_sha256": dict(_phase79_attempt().get("adapter_validation") or {}).get("sha256"),
        "core_source_sha256": _sha256(CORE_SOURCE),
        "driver_source_sha256": _sha256(DRIVER_SOURCE),
        "test_source_sha256": _sha256(TEST_SOURCE),
        "executor_source_sha256": _sha256(EXECUTOR_SOURCE),
        "generation_protocol_sha256": stable_hash(GENERATION_PROTOCOL),
        "frozen_low_lr": 1e-5,
        "frozen_steps": 12,
        "full_training_allowed": False,
        "score_or_gate_relaxation_allowed": False,
    }
    _write_json(PREPARATION_ROOT / "holdout.json", holdout)
    _write_json(PREPARATION_ROOT / "isolation_audit.json", isolation)
    _write_jsonl(PREPARATION_ROOT / "selected_training_samples.jsonl", samples)
    _write_json(PREPARATION_ROOT / "training_quality_audit.json", quality)
    _write_json(PREPARATION_ROOT / "completion_boundary_report.json", boundary)
    _write_json(PREPARATION_ROOT / "model_inventory.json", inventory)
    _write_json(PREPARATION_ROOT / "phase79_failure_snapshot.json", {
        "kind": "phase80_phase79_failure_snapshot",
        "checks": phase79_checks,
        "passed": all(phase79_checks.values()),
        "decision": phase79_decision,
        "comparison": phase79_comparison,
    })
    _write_json(EVIDENCE_ROOT / "generation_protocol.json", GENERATION_PROTOCOL)
    _write_json(EVIDENCE_ROOT / "pre_experiment_freeze.json", freeze)
    _write_json(EVIDENCE_ROOT / "preparation_decision.json", {
        "kind": "phase80_preparation_decision",
        "status": "ready_for_low_lr_12_step_probe" if freeze["passed"] else "blocked",
        "checks": checks,
        "failed_checks": [name for name, value in checks.items() if not value],
        "automatic_training_allowed": False,
    })
    existing_attempt_path = TRAINING_ROOT / "low-lr-12step/training_attempt.json"
    existing_manifest_path = TRAINING_ROOT / "low-lr-12step/training_manifest.json"
    if existing_attempt_path.exists() and existing_manifest_path.exists():
        existing_attempt = _read_json(existing_attempt_path)
        existing_manifest = _read_json(existing_manifest_path)
        current_manifest = _low_lr_job_spec(
            samples,
            TRAINER_OUTPUT_ROOT / "phase80-low-lr-12step",
        )
        validation = dict(existing_attempt.get("adapter_validation") or {})
        adapter_path = Path(str(validation.get("adapter_path") or ""))
        amendment_checks = {
            "existing_training_completed": existing_attempt.get("status") == "completed",
            "training_manifest_unchanged": stable_hash(existing_manifest)
            == stable_hash(current_manifest),
            "adapter_hash_unchanged": adapter_path.is_file()
            and _sha256(adapter_path) == validation.get("sha256"),
            "historical_adapter_not_reused": existing_attempt.get("historical_adapter_reused")
            is False,
        }
        _write_json(TRAINING_ROOT / "low-lr-12step/post_training_freeze_amendment.json", {
            "kind": "phase80_post_training_driver_amendment",
            "created_at": _utcnow(),
            "reason": (
                "The stop-control diagnostic now preserves the standard 192-token budget and adds "
                "no_repeat_ngram_size=4. The training recipe, samples, and low-LR adapter are unchanged."
            ),
            "checks": amendment_checks,
            "passed": all(amendment_checks.values()),
            "training_repetition_required": not all(amendment_checks.values()),
        })
    print(json.dumps({
        "status": "ready" if freeze["passed"] else "blocked",
        "holdout_session_count": holdout["session_count"],
        "maximum_full_token_count": boundary.get("maximum_full_token_count"),
        "checks": checks,
    }, ensure_ascii=False, indent=2))
    return 0 if freeze["passed"] else 1


def _training_freeze_check() -> dict[str, Any]:
    freeze = _read_json(EVIDENCE_ROOT / "pre_experiment_freeze.json")
    holdout = _read_json(PREPARATION_ROOT / "holdout.json")
    samples = _read_jsonl(PREPARATION_ROOT / "selected_training_samples.jsonl")
    isolation = _read_json(PREPARATION_ROOT / "isolation_audit.json")
    quality = _read_json(PREPARATION_ROOT / "training_quality_audit.json")
    boundary = _read_json(PREPARATION_ROOT / "completion_boundary_report.json")
    inventory = _read_json(PREPARATION_ROOT / "model_inventory.json")
    phase79_decision = _read_json(PHASE79_ROOT / "phase79-final-decision.json")
    phase79_adapter = _phase79_adapter_path() / "adapter_model.safetensors"
    checks = {
        "preparation_passed": freeze.get("passed") is True,
        "holdout_unchanged": stable_hash(holdout.get("sessions") or [])
        == freeze.get("holdout_manifest_sha256"),
        "training_unchanged": stable_hash(samples) == freeze.get("training_manifest_sha256"),
        "isolation_unchanged": stable_hash(isolation) == freeze.get("isolation_audit_sha256"),
        "training_quality_unchanged": stable_hash(quality) == freeze.get("training_quality_sha256"),
        "completion_boundary_unchanged": stable_hash(boundary)
        == freeze.get("completion_boundary_sha256"),
        "model_inventory_unchanged": stable_hash(inventory) == freeze.get("model_inventory_sha256"),
        "phase79_decision_unchanged": stable_hash(phase79_decision)
        == freeze.get("phase79_decision_sha256"),
        "phase79_adapter_unchanged": phase79_adapter.is_file()
        and _sha256(phase79_adapter) == freeze.get("phase79_adapter_sha256"),
        "core_unchanged": _sha256(CORE_SOURCE) == freeze.get("core_source_sha256"),
        "driver_unchanged": _sha256(DRIVER_SOURCE) == freeze.get("driver_source_sha256"),
        "test_unchanged": _sha256(TEST_SOURCE) == freeze.get("test_source_sha256"),
        "executor_unchanged": _sha256(EXECUTOR_SOURCE) == freeze.get("executor_source_sha256"),
    }
    return {"kind": "phase80_training_freeze_check", "passed": all(checks.values()), "checks": checks}


def _train(clean: bool) -> int:
    freeze = _training_freeze_check()
    if not freeze["passed"]:
        raise SystemExit(f"Phase80 training freeze failed: {freeze}")
    probe_dir = TRAINING_ROOT / "low-lr-12step"
    output_dir = TRAINER_OUTPUT_ROOT / "phase80-low-lr-12step"
    if clean:
        shutil.rmtree(probe_dir, ignore_errors=True)
        shutil.rmtree(output_dir, ignore_errors=True)
    probe_dir.mkdir(parents=True, exist_ok=True)
    samples = _read_jsonl(PREPARATION_ROOT / "selected_training_samples.jsonl")
    spec = _low_lr_job_spec(samples, output_dir)
    boundary = phase79_driver._completion_boundary_report(spec)
    _write_json(probe_dir / "training_manifest.json", spec)
    _write_json(probe_dir / "completion_boundary_report.json", boundary)
    _write_json(probe_dir / "training_freeze_check.json", freeze)
    if not boundary.get("passed") or len(samples) != PHASE78_TRAINING_SAMPLE_COUNT:
        attempt = {
            "kind": "phase80_low_lr_sft_training_attempt",
            "status": "blocked",
            "reason": "training_preflight_failed",
            "real_training": False,
            "requested_steps": 12,
            "historical_adapter_reused": False,
            "actual_product_benefit_claim_allowed": False,
            "auto_promotion_allowed": False,
        }
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
        validation = validate_adapter_artifact(
            artifact_dir,
            {"artifact_name": adapter_path.name, "artifact_format": "peft_lora"},
        )
        validation.update({
            "sha256": _sha256(adapter_path) if adapter_path.exists() else None,
            "artifact_dir": str(artifact_dir),
            "adapter_path": str(adapter_path),
            "parameters_updated": real.get("parameters_updated"),
            "steps": real.get("steps"),
        })
        completed = (
            result.get("status") == "completed"
            and real.get("success") is True
            and real.get("parameters_updated") is True
            and int(real.get("steps") or 0) >= 12
            and validation.get("valid") is True
        )
        attempt = {
            "kind": "phase80_low_lr_sft_training_attempt",
            "status": "completed" if completed else "failed",
            "real_training": completed,
            "requested_steps": 12,
            "selected_model": "Qwen2.5-0.5B-Instruct",
            "model": str(MODEL_05_PATH),
            "learning_rate": 1e-5,
            "seed": 80,
            "started_at": started_at,
            "finished_at": _utcnow(),
            "duration_seconds": round(time.perf_counter() - started, 4),
            "max_rss_before_bytes": rss_before,
            "max_rss_after_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "execution": real,
            "adapter_validation": validation,
            "historical_adapter_reused": False,
            "phase79_adapter_used_as_reference_only": True,
            "simulated_usage": True,
            "actual_user_feedback": False,
            "actual_product_benefit_claim_allowed": False,
            "auto_promotion_allowed": False,
        }
        _write_json(probe_dir / "adapter_validation.json", validation)
        _write_json(probe_dir / "train_log.json", {"loss_history": real.get("loss_history") or []})
        _write_json(probe_dir / "parameter_fingerprint_before_after.json", {
            "before": real.get("parameter_fingerprint_before"),
            "after": real.get("parameter_fingerprint_after"),
            "parameters_updated": real.get("parameters_updated"),
        })
        _write_json(probe_dir / "actual_exposure_report.json", {
            "requested_steps": 12,
            "actual_steps": real.get("steps"),
            "sample_exposure_counts": real.get("sample_exposure_counts") or {},
            "category_exposure_counts": real.get("category_exposure_counts") or {},
            "unique_samples_exposed": real.get("unique_samples_exposed"),
            "unique_categories_exposed": real.get("unique_categories_exposed"),
            "full_coverage": False,
        })
    except Exception as exc:
        attempt = {
            "kind": "phase80_low_lr_sft_training_attempt",
            "status": "failed",
            "real_training": False,
            "requested_steps": 12,
            "selected_model": "Qwen2.5-0.5B-Instruct",
            "learning_rate": 1e-5,
            "seed": 80,
            "started_at": started_at,
            "finished_at": _utcnow(),
            "duration_seconds": round(time.perf_counter() - started, 4),
            "error": f"{exc.__class__.__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "historical_adapter_reused": False,
            "actual_product_benefit_claim_allowed": False,
            "auto_promotion_allowed": False,
        }
        _write_json(FAILURE_ROOT / "low_lr_training_failure.json", attempt)
    _write_json(probe_dir / "training_attempt.json", attempt)
    _write_json(TRAINING_ROOT / "latest_training_attempt.json", attempt)
    print(json.dumps({key: attempt.get(key) for key in (
        "status", "requested_steps", "learning_rate", "duration_seconds", "error",
    )}, ensure_ascii=False, indent=2))
    return 0 if attempt.get("status") == "completed" else 1


def _low_lr_adapter_path() -> Path:
    attempt = _read_json(TRAINING_ROOT / "low-lr-12step/training_attempt.json")
    validation = dict(attempt.get("adapter_validation") or {})
    path = Path(str(validation.get("artifact_dir") or ""))
    if attempt.get("status") != "completed" or validation.get("valid") is not True or not path.is_dir():
        raise SystemExit("Phase80 low-LR adapter is unavailable")
    return path.resolve()


def _variant_config(variant: str) -> dict[str, Any]:
    configs = {
        "base_0_5b_minimal": {
            "model_path": MODEL_05_PATH,
            "adapter_path": None,
            "runtime_contract": False,
            "stop_control": False,
        },
        "runtime_0_5b": {
            "model_path": MODEL_05_PATH,
            "adapter_path": None,
            "runtime_contract": True,
            "stop_control": False,
        },
        "phase79_high_lr_adapter": {
            "model_path": MODEL_05_PATH,
            "adapter_path": _phase79_adapter_path(),
            "runtime_contract": False,
            "stop_control": False,
        },
        "phase80_low_lr_adapter": {
            "model_path": MODEL_05_PATH,
            "adapter_path": _low_lr_adapter_path(),
            "runtime_contract": False,
            "stop_control": False,
        },
        "phase79_high_lr_stop_control": {
            "model_path": MODEL_05_PATH,
            "adapter_path": _phase79_adapter_path(),
            "runtime_contract": False,
            "stop_control": True,
        },
        "base_4b_minimal": {
            "model_path": MODEL_4B_PATH,
            "adapter_path": None,
            "runtime_contract": False,
            "stop_control": False,
        },
        "runtime_4b": {
            "model_path": MODEL_4B_PATH,
            "adapter_path": None,
            "runtime_contract": True,
            "stop_control": False,
        },
    }
    if variant not in configs:
        raise SystemExit(f"unsupported Phase80 variant: {variant}")
    return configs[variant]


def _generation_freeze_check() -> dict[str, Any]:
    freeze = _read_json(EVIDENCE_ROOT / "pre_experiment_freeze.json")
    holdout = _read_json(PREPARATION_ROOT / "holdout.json")
    low_attempt = _read_json(TRAINING_ROOT / "low-lr-12step/training_attempt.json")
    low_validation = dict(low_attempt.get("adapter_validation") or {})
    low_adapter = Path(str(low_validation.get("adapter_path") or ""))
    high_attempt = _phase79_attempt()
    high_validation = dict(high_attempt.get("adapter_validation") or {})
    high_adapter = Path(str(high_validation.get("adapter_path") or ""))
    checks = {
        "fresh_holdout_unchanged": stable_hash(holdout.get("sessions") or [])
        == freeze.get("holdout_manifest_sha256"),
        "low_lr_training_completed": low_attempt.get("status") == "completed"
        and low_attempt.get("real_training") is True,
        "low_lr_adapter_unchanged": low_adapter.is_file()
        and _sha256(low_adapter) == low_validation.get("sha256"),
        "phase79_adapter_unchanged": high_adapter.is_file()
        and _sha256(high_adapter) == high_validation.get("sha256"),
        "historical_adapter_is_reference_only": low_attempt.get("historical_adapter_reused") is False,
        "generation_protocol_unchanged": stable_hash(GENERATION_PROTOCOL)
        == freeze.get("generation_protocol_sha256"),
    }
    return {"kind": "phase80_generation_freeze_check", "passed": all(checks.values()), "checks": checks}


def _generate_one_stop_control(
    torch: Any,
    tokenizer: Any,
    model: Any,
    device: str,
    messages: list[dict[str, str]],
) -> tuple[str, dict[str, Any]]:
    prompt = phase75_driver._render_prompt(tokenizer, messages)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=GENERATION_PROTOCOL["input_max_length"],
    )
    inputs = {name: value.to(device) for name, value in inputs.items()}
    input_length = int(inputs["input_ids"].shape[-1])
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=int(GENERATION_PROTOCOL["stop_control_max_new_tokens"]),
            do_sample=False,
            repetition_penalty=float(GENERATION_PROTOCOL["stop_control_repetition_penalty"]),
            no_repeat_ngram_size=int(GENERATION_PROTOCOL["stop_control_no_repeat_ngram_size"]),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = output[0][input_length:]
    raw = tokenizer.decode(generated, skip_special_tokens=True).strip()
    if not raw:
        raise RuntimeError("Phase80 stop-control generation returned empty output")
    cleaned = re.sub(
        r"<think>.*?</think>\s*", "", raw, flags=re.IGNORECASE | re.DOTALL
    ).strip() or raw
    return cleaned, {
        "input_tokens": input_length,
        "completion_tokens": int(generated.shape[-1]),
        "latency_seconds": round(time.perf_counter() - started, 4),
        "think_leak_detected": bool(re.search(r"</?think>", raw, flags=re.IGNORECASE)),
        "truncated": int(generated.shape[-1])
        >= int(GENERATION_PROTOCOL["stop_control_max_new_tokens"]),
        "stop_control": True,
        "no_repeat_ngram_size": int(GENERATION_PROTOCOL["stop_control_no_repeat_ngram_size"]),
    }


def _generate(variant: str, clean: bool) -> int:
    config = _variant_config(variant)
    freeze = _generation_freeze_check()
    if not freeze["passed"]:
        raise SystemExit(f"Phase80 generation freeze failed: {freeze}")
    sessions = [dict(row) for row in _read_json(PREPARATION_ROOT / "holdout.json").get("sessions") or []]
    output_path = GENERATION_ROOT / f"transcripts_{variant}.jsonl"
    metrics_path = GENERATION_ROOT / f"metrics_{variant}.json"
    if clean:
        output_path.unlink(missing_ok=True)
        metrics_path.unlink(missing_ok=True)
    rows = [] if clean else _read_jsonl(output_path)
    completed = {str(row.get("session_id")) for row in rows if row.get("status") == "completed"}
    old_model_path = phase79_driver.MODEL_PATH
    old_protocol = dict(phase75_driver.GENERATION_PROTOCOL)
    old_generate_one = phase75_driver._generate_one
    phase79_driver.MODEL_PATH = Path(config["model_path"])
    if config["stop_control"]:
        phase75_driver.GENERATION_PROTOCOL["max_new_tokens"] = int(
            GENERATION_PROTOCOL["stop_control_max_new_tokens"]
        )
        phase75_driver.GENERATION_PROTOCOL["repetition_penalty"] = float(
            GENERATION_PROTOCOL["stop_control_repetition_penalty"]
        )
        phase75_driver._generate_one = _generate_one_stop_control
    else:
        phase75_driver.GENERATION_PROTOCOL["max_new_tokens"] = int(
            GENERATION_PROTOCOL["standard_max_new_tokens"]
        )
        phase75_driver.GENERATION_PROTOCOL["repetition_penalty"] = float(
            GENERATION_PROTOCOL["standard_repetition_penalty"]
        )
    torch = tokenizer = model = device = None
    try:
        torch, tokenizer, model, device = phase79_driver._load_runtime(config["adapter_path"])
        for index, session in enumerate(sessions, start=1):
            session_id = str(session["session_id"])
            if session_id in completed:
                print(f"[{variant}] {index}/{len(sessions)} {session_id} resumed", flush=True)
                continue
            try:
                row = phase79_driver._run_session(
                    session=session,
                    variant="runtime_reference" if config["runtime_contract"] else "base_minimal_guarded",
                    torch=torch,
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                    adapter_loaded=config["adapter_path"] is not None,
                )
                row.update({
                    "kind": "phase80_real_multiturn_diagnostic_transcript",
                    "variant": variant,
                    "model_id": str(config["model_path"]),
                    "runtime_reference": bool(config["runtime_contract"]),
                    "stop_control": bool(config["stop_control"]),
                    "diagnostic_only": True,
                })
            except Exception as exc:
                row = {
                    "kind": "phase80_real_multiturn_diagnostic_transcript",
                    "session_id": session_id,
                    "category": session.get("category"),
                    "task_type": session.get("task_type"),
                    "variant": variant,
                    "model_id": str(config["model_path"]),
                    "adapter_loaded": config["adapter_path"] is not None,
                    "runtime_reference": bool(config["runtime_contract"]),
                    "stop_control": bool(config["stop_control"]),
                    "actual_model_call": False,
                    "status": "failed",
                    "error": f"{exc.__class__.__name__}: {exc}",
                    "turns": [],
                    "simulated_usage": True,
                    "actual_user_feedback": False,
                    "created_at": _utcnow(),
                }
            rows = [item for item in rows if item.get("session_id") != session_id]
            rows.append(row)
            rows.sort(key=lambda item: str(item.get("session_id")))
            _write_jsonl(output_path, rows)
            print(f"[{variant}] {index}/{len(sessions)} {session_id} {row['status']}", flush=True)
    finally:
        if torch is not None and model is not None and device is not None:
            phase79_driver._release_runtime(torch, model, device)
        phase79_driver.MODEL_PATH = old_model_path
        phase75_driver.GENERATION_PROTOCOL.clear()
        phase75_driver.GENERATION_PROTOCOL.update(old_protocol)
        phase75_driver._generate_one = old_generate_one
    metrics = aggregate_phase75_variant(rows, sessions)
    metrics.update({
        "kind": "phase80_variant_metrics",
        "variant": variant,
        "model_id": str(config["model_path"]),
        "adapter_loaded": config["adapter_path"] is not None,
        "runtime_reference": bool(config["runtime_contract"]),
        "stop_control": bool(config["stop_control"]),
        "model_call_count": sum(
            len(row.get("generation") or []) for row in rows if row.get("actual_model_call") is True
        ),
        "all_sessions_completed": len(rows) == len(sessions)
        and all(row.get("status") == "completed" for row in rows),
        "truncated_session_rate": round(
            sum(bool(row.get("truncated_response")) for row in rows) / len(rows), 4
        ) if rows else 0.0,
        "think_leak_rate": round(
            sum(bool(row.get("think_leak_detected")) for row in rows) / len(rows), 4
        ) if rows else 0.0,
        "returned_private_value_session_rate": round(
            sum(bool(row.get("returned_private_value_detected")) for row in rows) / len(rows), 4
        ) if rows else 0.0,
        "simulated_usage": True,
        "actual_user_feedback": False,
        "actual_product_benefit_claim_allowed": False,
    })
    _write_json(GENERATION_ROOT / f"freeze_check_{variant}.json", freeze)
    _write_json(metrics_path, metrics)
    print(json.dumps({key: metrics.get(key) for key in (
        "variant", "session_count", "model_call_count", "personalization_score",
        "hard_gate_pass_rate", "truncated_session_rate", "privacy_canary_echo_rate",
    )}, ensure_ascii=False, indent=2))
    return 0 if metrics["all_sessions_completed"] else 1


def _collect_metrics() -> dict[str, dict[str, Any]]:
    return {
        variant: _read_json(GENERATION_ROOT / f"metrics_{variant}.json")
        for variant in PHASE80_VARIANTS
    }


def _public_private_audit() -> dict[str, Any]:
    sessions = [dict(row) for row in _read_json(PREPARATION_ROOT / "holdout.json").get("sessions") or []]
    by_id = {str(row["session_id"]): row for row in sessions}
    details = []
    for variant in PHASE80_VARIANTS:
        for row in _read_jsonl(GENERATION_ROOT / f"transcripts_{variant}.jsonl"):
            session = by_id.get(str(row.get("session_id") or ""), {})
            values = [str(value) for value in session.get("declared_private_values") or [] if str(value)]
            serialized = json.dumps(row, ensure_ascii=False, sort_keys=True)
            leaked = [value for value in values if value in serialized]
            details.append({
                "variant": variant,
                "session_id": row.get("session_id"),
                "raw_private_match_count": len(leaked),
                "returned_private_value_detected": bool(row.get("returned_private_value_detected")),
                "model_input_private_value_detected": any(
                    bool(item.get("model_input_contains_declared_private_value"))
                    for item in row.get("private_input_guards") or []
                ),
            })
    expected = PHASE80_SESSION_COUNT * len(PHASE80_VARIANTS)
    checks = {
        "expected_transcript_count": len(details) == expected,
        "raw_private_match_count_zero": not any(row["raw_private_match_count"] for row in details),
        "returned_private_value_zero": not any(row["returned_private_value_detected"] for row in details),
        "model_input_private_value_zero": not any(row["model_input_private_value_detected"] for row in details),
    }
    return {
        "kind": "phase80_public_private_transcript_audit",
        "passed": all(checks.values()),
        "checks": checks,
        "transcript_count": len(details),
        "expected_transcript_count": expected,
        "details": details,
    }


def _output_examples(transcripts: Mapping[str, list[dict[str, Any]]]) -> str:
    selected = (
        "phase80-evidence_truthfulness-01",
        "phase80-provenance_labeling-01",
        "phase80-concise_workstyle-01",
        "phase80-ordinary_direct-01",
    )
    by_variant = {
        variant: {str(row.get("session_id")): row for row in rows}
        for variant, rows in transcripts.items()
    }
    lines = [
        "# Phase80 Output Examples",
        "",
        (
            "These are real local diagnostic outputs on fresh simulated_usage sessions. They compare "
            "learning rate, decoding stop control, and model capacity; they are not product-benefit evidence."
        ),
        "",
    ]
    for session_id in selected:
        lines.extend((f"## {session_id}", ""))
        for variant in PHASE80_VARIANTS:
            row = by_variant[variant][session_id]
            final = [
                str(turn.get("content") or "")
                for turn in row.get("turns") or []
                if turn.get("role") == "assistant"
            ][-1]
            final = "\n".join(line.rstrip() for line in final.splitlines())
            lines.extend((f"### {variant}", "", final, ""))
    return "\n".join(lines)


def _evidence_manifest() -> dict[str, Any]:
    files = []
    for path in sorted(EVIDENCE_ROOT.rglob("*")):
        if not path.is_file() or path.name in DYNAMIC_FILES:
            continue
        files.append({
            "path": str(path.relative_to(REPO_ROOT)),
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        })
    return {
        "kind": "phase80_evidence_manifest",
        "files": files,
        "file_count": len(files),
        "manifest_sha256": stable_hash(files),
    }


def _finalize() -> int:
    metrics = _collect_metrics()
    training = _read_json(TRAINING_ROOT / "low-lr-12step/training_attempt.json")
    isolation = _read_json(PREPARATION_ROOT / "isolation_audit.json")
    regression = _read_json(EVIDENCE_ROOT / "full_regression_summary.json")
    public_audit = _public_private_audit()
    _write_json(EVIDENCE_ROOT / "public_private_audit.json", public_audit)
    prerequisites = {
        "all_seven_variants_complete": all(
            row.get("all_sessions_completed") is True for row in metrics.values()
        ),
        "real_low_lr_training_complete": training.get("status") == "completed"
        and training.get("real_training") is True,
        "public_private_audit_passed": public_audit.get("passed") is True,
        "full_regression_passed": regression.get("passed") is True,
    }
    if not all(prerequisites.values()):
        raise SystemExit(f"Phase80 finalization prerequisites failed: {prerequisites}")
    decision = build_phase80_decision(
        metrics=metrics,
        low_lr_training_attempt=training,
        isolation_audit=isolation,
        public_private_audit=public_audit,
    )
    comparison = {
        "kind": "phase80_small_model_failure_taxonomy_comparison",
        "created_at": _utcnow(),
        "holdout_session_count_per_variant": PHASE80_SESSION_COUNT,
        "variant_count": len(PHASE80_VARIANTS),
        "real_generation_model_call_count": sum(
            int(row.get("model_call_count") or 0) for row in metrics.values()
        ),
        "real_low_lr_training": training,
        "metrics": metrics,
        "public_private_audit": public_audit,
        "decision": decision,
        "simulated_usage": True,
        "actual_user_feedback_count": 0,
        "actual_product_benefit_claim_allowed": False,
    }
    transcripts = {
        variant: _read_jsonl(GENERATION_ROOT / f"transcripts_{variant}.jsonl")
        for variant in PHASE80_VARIANTS
    }
    _write_json(EVIDENCE_ROOT / "phase80-final-decision.json", decision)
    _write_json(EVIDENCE_ROOT / "comparison_summary.json", comparison)
    _write_text(EVIDENCE_ROOT / "output_examples.md", _output_examples(transcripts))
    _write_text(EVIDENCE_ROOT / "phase80-final-decision.md", f"""# Phase80 Final Decision

Recommendation: **{decision['recommendation']}**

- Lifecycle status: `{decision['status']}`
- Failure classification: `{decision['failure_classification']}`
- Real low-LR 12-step training: `{training.get('status')}`
- 0.5B base target score: `{decision['target_scores']['base_0_5b_minimal']}`
- Phase79 high-LR adapter target score: `{decision['target_scores']['phase79_high_lr_adapter']}`
- Phase80 low-LR adapter target score: `{decision['target_scores']['phase80_low_lr_adapter']}`
- 0.5B runtime-contract score: `{decision['target_scores']['runtime_0_5b']}`
- 4B base score: `{decision['target_scores']['base_4b_minimal']}`
- 4B runtime-contract score: `{decision['target_scores']['runtime_4b']}`
- Low-LR adapter gain over 0.5B base: `{decision['low_lr_adapter_gain_vs_base']}`
- Runtime gain over 0.5B base: `{decision['runtime_gain_vs_base']}`
- 4B runtime gap over 0.5B runtime: `{decision['four_b_runtime_gap_vs_zero_point_five_b']}`
- Real generation calls: `{comparison['real_generation_model_call_count']}`

Phase80 is a diagnostic experiment, not an adapter promotion test. All sessions are fresh `simulated_usage`; there is no `actual_user_feedback`. Training completion, decoding stability, runtime-contract benefit, and model-capacity gap are reported separately. No adapter is promoted, attached to Hermes, or made a product default.
""")
    _write_text(EVIDENCE_ROOT / "phase80-runbook.md", """# Phase80 Runbook

```bash
.venv/bin/python tools/phase80_small_model_failure_taxonomy.py prepare --clean
.venv/bin/python tools/phase80_small_model_failure_taxonomy.py train --clean
.venv/bin/python tools/phase80_small_model_failure_taxonomy.py generate --variant base_0_5b_minimal --clean
.venv/bin/python tools/phase80_small_model_failure_taxonomy.py generate --variant runtime_0_5b --clean
.venv/bin/python tools/phase80_small_model_failure_taxonomy.py generate --variant phase79_high_lr_adapter --clean
.venv/bin/python tools/phase80_small_model_failure_taxonomy.py generate --variant phase80_low_lr_adapter --clean
.venv/bin/python tools/phase80_small_model_failure_taxonomy.py generate --variant phase79_high_lr_stop_control --clean
.venv/bin/python tools/phase80_small_model_failure_taxonomy.py generate --variant base_4b_minimal --clean
.venv/bin/python tools/phase80_small_model_failure_taxonomy.py generate --variant runtime_4b --clean
.venv/bin/python tools/phase80_small_model_failure_taxonomy.py full-regression
.venv/bin/python tools/phase80_small_model_failure_taxonomy.py finalize
.venv/bin/python tools/phase80_small_model_failure_taxonomy.py validate
```

The holdout and hypotheses are frozen before low-LR training. The Phase79 high-LR adapter is a read-only historical comparison, never the new candidate. Stop control changes only max generation tokens and repetition penalty, while all scores use the unchanged Phase75 deterministic rubric. This phase cannot promote or change defaults.
""")
    next_goal = {
        "optimization_instability_recoverable": (
            "Run a Phase81 low-LR full-coverage probe only after a fresh sanity gate. Keep the 0.5B "
            "candidate diagnostic and require independent blind judges before any simulated benefit claim."
        ),
        "small_model_capacity_dominant": (
            "Select the smallest locally trainable mid-size Qwen model, estimate a bounded training budget, "
            "and run a 4/12-step feasibility ladder before any full curriculum probe."
        ),
        "small_model_capacity_dominant_with_length_cost": (
            "Select the smallest trainable mid-size Qwen model and pair its 4/12-step feasibility ladder "
            "with a frozen length-control sanity gate before any full curriculum probe."
        ),
        "runtime_contract_dominant": (
            "Productize the Phase77 conditional runtime contract for persona tasks and postpone adapter "
            "training until stronger or actual consented preference signals exist."
        ),
        "curriculum_or_capacity_unresolved": (
            "Inspect completion losses and per-dimension outputs, then design a balanced micro-curriculum "
            "probe without changing the frozen evaluation rubric."
        ),
    }.get(decision["failure_classification"], "Repair incomplete Phase80 evidence before another probe.")
    _write_text(EVIDENCE_ROOT / "next-pursuit-goal.md", f"# Phase81 Pursuit Goal\n\n{next_goal}")
    _write_json(EVIDENCE_ROOT / "phase80-result-taxonomy.json", {
        "kind": "phase80_result_taxonomy",
        "training_proof": "real_low_lr_12_step_cpu_training_completed",
        "failure_classification": decision["failure_classification"],
        "adapter_benefit": "diagnostic_only",
        "runtime_contract": "measured_on_same_fresh_holdout",
        "capacity_gap": "measured_on_same_fresh_holdout",
        "actual_user_feedback": "absent",
        "product_benefit": "not_established",
        "promotion": "forbidden",
        "next_gate": decision["next_gate"],
    })
    manifest = _evidence_manifest()
    _write_json(EVIDENCE_ROOT / "evidence_manifest.json", manifest)
    integrity_checks = {
        **prerequisites,
        "decision_complete": decision.get("status") == "diagnosis_completed",
        "training_and_benefit_separate": decision.get("training_success") is True
        and decision.get("adapter_benefit") == "diagnostic_only",
        "no_actual_product_claim": decision.get("actual_product_benefit_claim_allowed") is False,
        "no_auto_promotion": decision.get("auto_promotion_allowed") is False,
    }
    _write_json(EVIDENCE_ROOT / "evidence_integrity.json", {
        "kind": "phase80_evidence_integrity",
        "passed": all(integrity_checks.values()),
        "checks": integrity_checks,
        "manifest_sha256": manifest["manifest_sha256"],
    })
    _write_json(EVIDENCE_ROOT / "finalization_state.json", {
        "kind": "phase80_finalization_state",
        "status": "completed",
        "decision": decision["recommendation"],
        "created_at": _utcnow(),
    })
    print(json.dumps({
        "status": decision["status"],
        "failure_classification": decision["failure_classification"],
        "recommendation": decision["recommendation"],
        "target_scores": decision["target_scores"],
    }, ensure_ascii=False, indent=2))
    return 0


def _run_logged(command: list[str]) -> dict[str, Any]:
    started = time.perf_counter()
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    lines = []
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    code = process.wait()
    return {
        "command": command,
        "exit_code": code,
        "duration_seconds": round(time.perf_counter() - started, 4),
        "output": "".join(lines),
    }


def _full_regression() -> int:
    commands = (
        [
            str(REPO_ROOT / ".venv/bin/pytest"),
            "-q",
            "tests/test_phase80_small_model_failure_taxonomy.py",
            "tests/test_phase79_cpu_feasible_persona_probe.py",
            "tests/test_phase78_persona_internalization_training.py",
        ],
        ["make", "test-unit", "test-surface", "test-e2e-mock", "smoke-beta"],
    )
    results = []
    for command in commands:
        result = _run_logged(command)
        results.append(result)
        if result["exit_code"] != 0:
            break
    summary = {
        "kind": "phase80_full_regression_summary",
        "created_at": _utcnow(),
        "passed": len(results) == len(commands) and all(row["exit_code"] == 0 for row in results),
        "results": results,
    }
    _write_json(EVIDENCE_ROOT / "full_regression_summary.json", summary)
    return 0 if summary["passed"] else 1


def _validate() -> int:
    manifest = _read_json(EVIDENCE_ROOT / "evidence_manifest.json")
    integrity = _read_json(EVIDENCE_ROOT / "evidence_integrity.json")
    decision = _read_json(EVIDENCE_ROOT / "phase80-final-decision.json")
    training = _read_json(TRAINING_ROOT / "low-lr-12step/training_attempt.json")
    manifest_failures = []
    for row in manifest.get("files") or []:
        path = REPO_ROOT / str(row.get("path") or "")
        if not path.is_file() or _sha256(path) != row.get("sha256"):
            manifest_failures.append(str(row.get("path") or ""))
    raw_private_locations = []
    for path in EVIDENCE_ROOT.rglob("*"):
        if not path.is_file() or path.name in DYNAMIC_FILES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if "SYNTHETIC_PHASE80_PRIVATE_" in text:
            raw_private_locations.append(str(path.relative_to(EVIDENCE_ROOT)))
    checks = {
        "manifest_files_unchanged": not manifest_failures,
        "integrity_passed": integrity.get("passed") is True,
        "real_low_lr_training_completed": training.get("status") == "completed"
        and training.get("real_training") is True,
        "historical_adapter_not_reused": training.get("historical_adapter_reused") is False,
        "private_canaries_only_in_frozen_holdout": raw_private_locations
        == ["evidence-preparation/holdout.json"],
        "no_actual_user_claim": decision.get("actual_product_benefit_claim_allowed") is False,
        "no_auto_promotion": decision.get("auto_promotion_allowed") is False,
        "no_hermes_attachment": decision.get("hermes_attachment_allowed") is False,
        "product_default_unchanged": decision.get("product_default_changed") is False,
    }
    summary = {
        "kind": "phase80_validation_summary",
        "created_at": _utcnow(),
        "passed": all(checks.values()),
        "checks": checks,
        "manifest_failures": manifest_failures,
        "raw_private_locations": raw_private_locations,
    }
    _write_json(EVIDENCE_ROOT / "validation_summary.json", summary)
    _write_text(EVIDENCE_ROOT / "validation_gate.txt", "PASS" if summary["passed"] else "FAIL")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["passed"] else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--clean", action="store_true")

    train = subparsers.add_parser("train")
    train.add_argument("--clean", action="store_true")

    generate = subparsers.add_parser("generate")
    generate.add_argument("--variant", choices=PHASE80_VARIANTS, required=True)
    generate.add_argument("--clean", action="store_true")

    subparsers.add_parser("full-regression")
    subparsers.add_parser("finalize")
    subparsers.add_parser("validate")
    args = parser.parse_args()

    if args.command == "prepare":
        return _prepare(args.clean)
    if args.command == "train":
        return _train(args.clean)
    if args.command == "generate":
        return _generate(args.variant, args.clean)
    if args.command == "full-regression":
        return _full_regression()
    if args.command == "finalize":
        return _finalize()
    if args.command == "validate":
        return _validate()
    raise SystemExit(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
