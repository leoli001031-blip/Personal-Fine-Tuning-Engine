#!/usr/bin/env python3
"""Run the Phase81 bounded mid-size Qwen training and evaluation ladder."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
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
    PHASE78_TRAINING_SAMPLE_COUNT,
    audit_phase78_training_samples,
    build_phase78_holdout,
    build_phase78_sft_job_spec,
    build_phase78_training_samples,
)
from pfe_core.phase79_cpu_feasible_persona_probe import build_phase79_holdout
from pfe_core.phase80_small_model_failure_taxonomy import build_phase80_holdout
from pfe_core.phase81_trainable_mid_model_selection import (
    PHASE81_SANITY_SESSION_COUNT,
    PHASE81_SANITY_VARIANTS,
    PHASE81_SESSION_COUNT,
    PHASE81_VARIANTS,
    audit_phase81_isolation,
    build_phase81_final_decision,
    build_phase81_holdout,
    build_phase81_model_selection,
    build_phase81_sanity_decision,
    build_phase81_sanity_holdout,
)
from pfe_core.trainer.executors import _run_real_local_peft_training


EVIDENCE_ROOT = REPO_ROOT / "docs/demo/phase81-trainable-mid-model-selection"
PREPARATION_ROOT = EVIDENCE_ROOT / "evidence-preparation"
TRAINING_ROOT = EVIDENCE_ROOT / "evidence-real-training"
GENERATION_ROOT = EVIDENCE_ROOT / "evidence-real-generation"
FAILURE_ROOT = EVIDENCE_ROOT / "evidence-failures"
MODEL_PATH = REPO_ROOT / "models/Qwen2.5-1.5B-Instruct"
TRAINER_OUTPUT_ROOT = REPO_ROOT / "trainer_job_outputs"
CORE_SOURCE = CORE_ROOT / "pfe_core/phase81_trainable_mid_model_selection.py"
DRIVER_SOURCE = REPO_ROOT / "tools/phase81_trainable_mid_model_selection.py"
TEST_SOURCE = REPO_ROOT / "tests/test_phase81_trainable_mid_model_selection.py"
EXECUTOR_SOURCE = CORE_ROOT / "pfe_core/trainer/executors.py"
MODEL_REPO = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
MODEL_EXPECTED_BYTES = 3_098_955_668
TRAINING_LEARNING_RATE = 2e-5
TRAINING_MAX_LENGTH = 176
GENERATION_PROTOCOL = {
    "kind": "phase81_frozen_mid_model_length_control_protocol",
    "input_max_length": 3072,
    "max_new_tokens": 128,
    "do_sample": False,
    "repetition_penalty": 1.15,
    "no_repeat_ngram_size": 4,
    "enable_thinking": False,
    "three_user_turns_per_session": True,
    "same_protocol_all_arms": True,
    "same_model_all_arms": True,
    "private_value_guard_before_every_call": True,
    "score_or_gate_relaxation_allowed": False,
}
FROZEN_THRESHOLDS = {
    "sanity_max_duration_seconds": 3600,
    "sanity_max_adapter_truncation_rate": 0.20,
    "sanity_max_target_regression": 0.10,
    "sanity_max_ordinary_regression": 0.15,
    "final_min_adapter_gain": 0.05,
    "final_min_runtime_gain": 0.04,
    "final_max_adapter_truncation_rate": 0.10,
    "final_max_ordinary_regression": 0.02,
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
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return dict(value) if isinstance(value, Mapping) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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
        "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
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


def _hardware_summary() -> dict[str, Any]:
    memory_bytes = 128 * 1024**3
    chip = "unknown"
    model = "unknown"
    try:
        completed = subprocess.run(
            ["system_profiler", "SPHardwareDataType", "-json"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        item = dict(json.loads(completed.stdout).get("SPHardwareDataType", [{}])[0])
        memory_text = str(item.get("physical_memory") or "")
        match = re.search(r"(\d+)\s*GB", memory_text, flags=re.IGNORECASE)
        if match:
            memory_bytes = int(match.group(1)) * 1024**3
        chip = str(item.get("chip_type") or "unknown")
        model = str(item.get("machine_model") or "unknown")
    except Exception:
        pass
    try:
        import torch

        mps_available = bool(torch.backends.mps.is_available())
    except Exception:
        mps_available = False
    return {
        "chip": chip,
        "machine_model": model,
        "system_memory_bytes": memory_bytes,
        "available_disk_bytes": shutil.disk_usage(REPO_ROOT).free,
        "mps_available": mps_available,
        "execution_device": "mps" if mps_available else "cpu",
        "serial_or_device_identifiers_persisted": False,
    }


def _model_complete(path: Path) -> bool:
    required = (
        "config.json",
        "generation_config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
    )
    return all((path / name).is_file() and (path / name).stat().st_size > 0 for name in required)


def _model_selection(hardware: Mapping[str, Any]) -> dict[str, Any]:
    selected_config = _read_json(MODEL_PATH / "config.json")
    candidates = (
        {
            "model_id": "Qwen2.5-1.5B-Instruct",
            "repo_id": MODEL_REPO,
            "revision": MODEL_REVISION,
            "official_model_card": "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct",
            "local_path": str(MODEL_PATH),
            "parameter_billions": 1.54,
            "download_bytes": MODEL_EXPECTED_BYTES,
            "estimated_training_memory_bytes": 16 * 1024**3,
            "official_qwen": True,
            "architecture_compatible": selected_config.get("model_type") == "qwen2",
            "download_complete": _model_complete(MODEL_PATH),
            "selection_note": "same Qwen2 architecture and PEFT path as the completed Phase79 CPU probe",
        },
        {
            "model_id": "Qwen3-1.7B",
            "repo_id": "Qwen/Qwen3-1.7B",
            "revision": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
            "official_model_card": "https://huggingface.co/Qwen/Qwen3-1.7B",
            "local_path": str(REPO_ROOT / "models/Qwen3-1.7B"),
            "parameter_billions": 1.7,
            "download_bytes": 4_079_423_234,
            "estimated_training_memory_bytes": 18 * 1024**3,
            "official_qwen": True,
            "architecture_compatible": True,
            "download_complete": _model_complete(REPO_ROOT / "models/Qwen3-1.7B"),
            "selection_note": "not preferred because thinking-mode behavior adds a second variable",
        },
    )
    return build_phase81_model_selection(
        candidates,
        available_disk_bytes=int(hardware.get("available_disk_bytes") or 0),
        system_memory_bytes=int(hardware.get("system_memory_bytes") or 0),
        mps_available=bool(hardware.get("mps_available")),
    )


def _job_spec(samples: Iterable[Mapping[str, Any]], output_dir: Path, steps: int) -> dict[str, Any]:
    spec = build_phase78_sft_job_spec(
        samples=samples,
        base_model=str(MODEL_PATH),
        output_dir=str(output_dir),
        max_steps=steps,
        learning_rate=TRAINING_LEARNING_RATE,
        seed=81,
    )
    spec["recipe"]["training"]["max_length"] = TRAINING_MAX_LENGTH
    spec["phase81"] = {
        "target_model": "Qwen2.5-1.5B-Instruct",
        "model_repo": MODEL_REPO,
        "model_revision": MODEL_REVISION,
        "bounded_probe": True,
        "completion_only_loss_required": True,
        "source_curriculum": "phase78_privacy_safe_persona_curriculum",
        "simulated_usage": True,
        "actual_user_feedback": False,
        "auto_promotion_allowed": False,
    }
    return spec


def _completion_boundary_report(spec: Mapping[str, Any]) -> dict[str, Any]:
    return phase79_driver._completion_boundary_report(spec)


def _prepare(clean: bool) -> int:
    if clean:
        shutil.rmtree(EVIDENCE_ROOT, ignore_errors=True)
    PREPARATION_ROOT.mkdir(parents=True, exist_ok=True)
    holdout = build_phase81_holdout()
    sanity = build_phase81_sanity_holdout(holdout)
    previous = (
        build_phase77_holdout()["sessions"]
        + build_phase78_holdout()["sessions"]
        + build_phase79_holdout()["sessions"]
        + build_phase80_holdout()["sessions"]
    )
    isolation = audit_phase81_isolation(holdout["sessions"], previous)
    samples = build_phase78_training_samples()
    quality = audit_phase78_training_samples(samples)
    hardware = _hardware_summary()
    selection = _model_selection(hardware)
    boundary = _completion_boundary_report(
        _job_spec(samples, TRAINER_OUTPUT_ROOT / "phase81-preflight", 4)
    )
    checks = {
        "model_selection_passed": selection.get("status") == "selected",
        "selected_expected_model": selection.get("selected_model") == "Qwen2.5-1.5B-Instruct",
        "model_download_complete": _model_complete(MODEL_PATH),
        "training_quality_passed": quality.get("passed") is True,
        "training_sample_count_120": len(samples) == PHASE78_TRAINING_SAMPLE_COUNT,
        "completion_boundary_passed": boundary.get("passed") is True,
        "holdout_isolation_passed": isolation.get("passed") is True,
        "sanity_holdout_count_7": sanity.get("session_count") == PHASE81_SANITY_SESSION_COUNT,
    }
    freeze = {
        "kind": "phase81_pre_experiment_freeze",
        "created_at": _utcnow(),
        "frozen_before_training": True,
        "passed": all(checks.values()),
        "checks": checks,
        "model_repo": MODEL_REPO,
        "model_revision": MODEL_REVISION,
        "model_config_sha256": _sha256(MODEL_PATH / "config.json") if (MODEL_PATH / "config.json").is_file() else None,
        "model_weight_size_bytes": (MODEL_PATH / "model.safetensors").stat().st_size
        if (MODEL_PATH / "model.safetensors").is_file()
        else 0,
        "training_manifest_sha256": stable_hash(samples),
        "holdout_manifest_sha256": stable_hash(holdout["sessions"]),
        "sanity_manifest_sha256": stable_hash(sanity["sessions"]),
        "generation_protocol_sha256": stable_hash(GENERATION_PROTOCOL),
        "thresholds_sha256": stable_hash(FROZEN_THRESHOLDS),
        "source_sha256": {
            "core": _sha256(CORE_SOURCE),
            "driver": _sha256(DRIVER_SOURCE),
            "test": _sha256(TEST_SOURCE),
            "executor": _sha256(EXECUTOR_SOURCE),
        },
        "score_or_gate_relaxation_allowed": False,
        "automatic_training_allowed": False,
        "automatic_promotion_allowed": False,
    }
    _write_json(PREPARATION_ROOT / "hardware_summary.json", hardware)
    _write_json(PREPARATION_ROOT / "model_selection.json", selection)
    _write_json(PREPARATION_ROOT / "holdout.json", holdout)
    _write_json(PREPARATION_ROOT / "sanity_holdout_manifest.json", {
        "kind": sanity["kind"],
        "session_count": sanity["session_count"],
        "session_ids": [row["session_id"] for row in sanity["sessions"]],
        "manifest_sha256": sanity["manifest_sha256"],
        "not_for_training": True,
    })
    _write_json(PREPARATION_ROOT / "isolation_audit.json", isolation)
    _write_jsonl(PREPARATION_ROOT / "selected_training_samples.jsonl", samples)
    _write_json(PREPARATION_ROOT / "training_quality_audit.json", quality)
    _write_json(PREPARATION_ROOT / "completion_boundary_report.json", boundary)
    _write_json(EVIDENCE_ROOT / "generation_protocol.json", GENERATION_PROTOCOL)
    _write_json(EVIDENCE_ROOT / "frozen_thresholds.json", FROZEN_THRESHOLDS)
    _write_json(EVIDENCE_ROOT / "pre_experiment_freeze.json", freeze)
    _write_json(EVIDENCE_ROOT / "preparation_decision.json", {
        "kind": "phase81_preparation_decision",
        "status": "ready_for_4_step_probe" if freeze["passed"] else "blocked",
        "checks": checks,
        "automatic_training_started": False,
    })
    print(json.dumps({
        "status": "ready_for_4_step_probe" if freeze["passed"] else "blocked",
        "selected_model": selection.get("selected_model"),
        "execution_device": selection.get("execution_device"),
        "checks": checks,
    }, ensure_ascii=False, indent=2))
    return 0 if freeze["passed"] else 2


def _training_freeze_check(steps: int) -> dict[str, Any]:
    freeze = _read_json(EVIDENCE_ROOT / "pre_experiment_freeze.json")
    samples = _read_jsonl(PREPARATION_ROOT / "selected_training_samples.jsonl")
    selection = _read_json(PREPARATION_ROOT / "model_selection.json")
    config_path = MODEL_PATH / "config.json"
    weights = MODEL_PATH / "model.safetensors"
    sanity = _read_json(EVIDENCE_ROOT / "sanity_decision.json")
    checks = {
        "pre_experiment_freeze_passed": freeze.get("passed") is True,
        "training_manifest_unchanged": stable_hash(samples) == freeze.get("training_manifest_sha256"),
        "model_selection_unchanged": selection.get("selected_model") == "Qwen2.5-1.5B-Instruct",
        "model_config_unchanged": config_path.is_file() and _sha256(config_path) == freeze.get("model_config_sha256"),
        "model_weight_size_unchanged": weights.is_file()
        and weights.stat().st_size == int(freeze.get("model_weight_size_bytes") or 0),
        "generation_protocol_unchanged": stable_hash(GENERATION_PROTOCOL)
        == freeze.get("generation_protocol_sha256"),
        "thresholds_unchanged": stable_hash(FROZEN_THRESHOLDS) == freeze.get("thresholds_sha256"),
        "twelve_step_requires_passed_sanity": steps != 12 or sanity.get("passed") is True,
    }
    return {"kind": "phase81_training_freeze_check", "passed": all(checks.values()), "checks": checks}


def _train(steps: int, clean: bool) -> int:
    if steps not in (4, 12):
        raise SystemExit("Phase81 only permits frozen 4-step and 12-step probes")
    freeze = _training_freeze_check(steps)
    probe_dir = TRAINING_ROOT / f"probe-{steps}step"
    output_dir = TRAINER_OUTPUT_ROOT / f"phase81-probe-{steps}step"
    if clean:
        shutil.rmtree(probe_dir, ignore_errors=True)
        shutil.rmtree(output_dir, ignore_errors=True)
    probe_dir.mkdir(parents=True, exist_ok=True)
    samples = _read_jsonl(PREPARATION_ROOT / "selected_training_samples.jsonl")
    spec = _job_spec(samples, output_dir, steps)
    boundary = _completion_boundary_report(spec)
    _write_json(probe_dir / "training_manifest.json", spec)
    _write_json(probe_dir / "training_freeze_check.json", freeze)
    _write_json(probe_dir / "completion_boundary_report.json", boundary)
    if not freeze["passed"] or not boundary.get("passed"):
        attempt = {
            "kind": "phase81_mid_model_training_attempt",
            "status": "blocked",
            "requested_steps": steps,
            "real_training": False,
            "reason": "training_freeze_or_boundary_failed",
            "freeze": freeze,
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
            "sha256": _sha256(adapter_path) if adapter_path.is_file() else None,
            "artifact_dir": str(artifact_dir),
            "adapter_path": str(adapter_path),
            "parameters_updated": real.get("parameters_updated"),
            "steps": real.get("steps"),
        })
        completed = (
            result.get("status") == "completed"
            and real.get("success") is True
            and real.get("parameters_updated") is True
            and int(real.get("steps") or 0) >= steps
            and validation.get("valid") is True
        )
        attempt = {
            "kind": "phase81_mid_model_training_attempt",
            "status": "completed" if completed else "failed",
            "real_training": completed,
            "candidate_eligible": False,
            "selected_model": "Qwen2.5-1.5B-Instruct",
            "model": str(MODEL_PATH),
            "model_revision": MODEL_REVISION,
            "requested_steps": steps,
            "learning_rate": TRAINING_LEARNING_RATE,
            "seed": 81,
            "started_at": started_at,
            "finished_at": _utcnow(),
            "duration_seconds": round(time.perf_counter() - started, 4),
            "max_rss_before_bytes": rss_before,
            "max_rss_after_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "execution": real,
            "adapter_validation": validation,
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
    except Exception as exc:
        attempt = {
            "kind": "phase81_mid_model_training_attempt",
            "status": "failed",
            "real_training": False,
            "candidate_eligible": False,
            "selected_model": "Qwen2.5-1.5B-Instruct",
            "requested_steps": steps,
            "started_at": started_at,
            "finished_at": _utcnow(),
            "duration_seconds": round(time.perf_counter() - started, 4),
            "error": f"{exc.__class__.__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "actual_product_benefit_claim_allowed": False,
            "auto_promotion_allowed": False,
        }
        _write_json(FAILURE_ROOT / f"training_probe_{steps}step.json", attempt)
    _write_json(probe_dir / "training_attempt.json", attempt)
    _write_json(TRAINING_ROOT / "latest_training_attempt.json", attempt)
    print(json.dumps({key: attempt.get(key) for key in (
        "status", "requested_steps", "duration_seconds", "error",
    )}, ensure_ascii=False, indent=2))
    return 0 if attempt.get("status") == "completed" else 1


def _adapter_path(steps: int) -> Path:
    attempt = _read_json(TRAINING_ROOT / f"probe-{steps}step/training_attempt.json")
    validation = dict(attempt.get("adapter_validation") or {})
    path = Path(str(validation.get("artifact_dir") or ""))
    if attempt.get("status") != "completed" or validation.get("valid") is not True or not path.is_dir():
        raise SystemExit(f"Phase81 {steps}-step adapter is unavailable")
    return path.resolve()


def _variant_config(scope: str, variant: str) -> dict[str, Any]:
    if scope == "sanity":
        configs = {
            "base_mid_4step_sanity": {"adapter_path": None, "runtime_contract": False},
            "adapter_mid_4step_sanity": {"adapter_path": _adapter_path(4), "runtime_contract": False},
        }
    else:
        configs = {
            "base_mid_length_control": {"adapter_path": None, "runtime_contract": False},
            "runtime_mid_length_control": {"adapter_path": None, "runtime_contract": True},
            "adapter_mid_12step_length_control": {"adapter_path": _adapter_path(12), "runtime_contract": False},
        }
    if variant not in configs:
        raise SystemExit(f"unsupported Phase81 {scope} variant: {variant}")
    return configs[variant]


def _generation_freeze_check(scope: str, config: Mapping[str, Any]) -> dict[str, Any]:
    freeze = _read_json(EVIDENCE_ROOT / "pre_experiment_freeze.json")
    holdout = _read_json(PREPARATION_ROOT / "holdout.json")
    adapter_path = config.get("adapter_path")
    checks = {
        "pre_experiment_freeze_passed": freeze.get("passed") is True,
        "fresh_holdout_unchanged": stable_hash(holdout.get("sessions") or [])
        == freeze.get("holdout_manifest_sha256"),
        "generation_protocol_unchanged": stable_hash(GENERATION_PROTOCOL)
        == freeze.get("generation_protocol_sha256"),
        "model_config_unchanged": _sha256(MODEL_PATH / "config.json")
        == freeze.get("model_config_sha256"),
        "adapter_available_or_base": adapter_path is None or Path(str(adapter_path)).is_dir(),
        "full_generation_requires_passed_sanity": scope != "full"
        or _read_json(EVIDENCE_ROOT / "sanity_decision.json").get("passed") is True,
    }
    return {"kind": "phase81_generation_freeze_check", "passed": all(checks.values()), "checks": checks}


def _generate_one_length_control(
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
            max_new_tokens=int(GENERATION_PROTOCOL["max_new_tokens"]),
            do_sample=False,
            repetition_penalty=float(GENERATION_PROTOCOL["repetition_penalty"]),
            no_repeat_ngram_size=int(GENERATION_PROTOCOL["no_repeat_ngram_size"]),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = output[0][input_length:]
    raw = tokenizer.decode(generated, skip_special_tokens=True).strip()
    if not raw:
        raise RuntimeError("Phase81 length-controlled generation returned empty output")
    cleaned = re.sub(r"<think>.*?</think>\s*", "", raw, flags=re.IGNORECASE | re.DOTALL).strip() or raw
    return cleaned, {
        "input_tokens": input_length,
        "completion_tokens": int(generated.shape[-1]),
        "latency_seconds": round(time.perf_counter() - started, 4),
        "think_leak_detected": bool(re.search(r"</?think>", raw, flags=re.IGNORECASE)),
        "truncated": int(generated.shape[-1]) >= int(GENERATION_PROTOCOL["max_new_tokens"]),
        "length_control": True,
        "no_repeat_ngram_size": int(GENERATION_PROTOCOL["no_repeat_ngram_size"]),
    }


def _scope_sessions(scope: str) -> list[dict[str, Any]]:
    holdout = _read_json(PREPARATION_ROOT / "holdout.json")
    sessions = [dict(row) for row in holdout.get("sessions") or []]
    if scope == "full":
        return sessions
    ids = set(_read_json(PREPARATION_ROOT / "sanity_holdout_manifest.json").get("session_ids") or [])
    return [row for row in sessions if row.get("session_id") in ids]


def _generate(scope: str, variant: str, clean: bool) -> int:
    config = _variant_config(scope, variant)
    freeze = _generation_freeze_check(scope, config)
    if not freeze["passed"]:
        raise SystemExit(f"Phase81 generation freeze failed: {freeze}")
    sessions = _scope_sessions(scope)
    root = GENERATION_ROOT / scope
    output_path = root / f"transcripts_{variant}.jsonl"
    metrics_path = root / f"metrics_{variant}.json"
    if clean:
        output_path.unlink(missing_ok=True)
        metrics_path.unlink(missing_ok=True)
    rows = [] if clean else _read_jsonl(output_path)
    completed = {str(row.get("session_id")) for row in rows if row.get("status") == "completed"}
    old_model_path = phase79_driver.MODEL_PATH
    old_protocol = dict(phase75_driver.GENERATION_PROTOCOL)
    old_generate_one = phase75_driver._generate_one
    phase79_driver.MODEL_PATH = MODEL_PATH
    phase75_driver.GENERATION_PROTOCOL.update(GENERATION_PROTOCOL)
    phase75_driver._generate_one = _generate_one_length_control
    torch = tokenizer = model = device = None
    try:
        torch, tokenizer, model, device = phase79_driver._load_runtime(config["adapter_path"])
        for index, session in enumerate(sessions, start=1):
            session_id = str(session["session_id"])
            if session_id in completed:
                print(f"[{scope}:{variant}] {index}/{len(sessions)} {session_id} resumed", flush=True)
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
                    "kind": "phase81_real_multiturn_mid_model_transcript",
                    "variant": variant,
                    "model_id": str(MODEL_PATH),
                    "model_revision": MODEL_REVISION,
                    "runtime_reference": bool(config["runtime_contract"]),
                    "length_control": True,
                    "diagnostic_only": True,
                })
            except Exception as exc:
                row = {
                    "kind": "phase81_real_multiturn_mid_model_transcript",
                    "session_id": session_id,
                    "category": session.get("category"),
                    "task_type": session.get("task_type"),
                    "variant": variant,
                    "model_id": str(MODEL_PATH),
                    "adapter_loaded": config["adapter_path"] is not None,
                    "runtime_reference": bool(config["runtime_contract"]),
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
            print(f"[{scope}:{variant}] {index}/{len(sessions)} {session_id} {row['status']}", flush=True)
    finally:
        if torch is not None and model is not None and device is not None:
            phase79_driver._release_runtime(torch, model, device)
        phase79_driver.MODEL_PATH = old_model_path
        phase75_driver.GENERATION_PROTOCOL.clear()
        phase75_driver.GENERATION_PROTOCOL.update(old_protocol)
        phase75_driver._generate_one = old_generate_one
    metrics = aggregate_phase75_variant(rows, sessions)
    metrics.update({
        "kind": "phase81_variant_metrics",
        "variant": variant,
        "scope": scope,
        "model_id": str(MODEL_PATH),
        "model_revision": MODEL_REVISION,
        "adapter_loaded": config["adapter_path"] is not None,
        "runtime_reference": bool(config["runtime_contract"]),
        "length_control": True,
        "model_call_count": sum(len(row.get("generation") or []) for row in rows if row.get("actual_model_call") is True),
        "all_sessions_completed": len(rows) == len(sessions)
        and all(row.get("status") == "completed" for row in rows),
        "truncated_session_rate": round(sum(bool(row.get("truncated_response")) for row in rows) / len(rows), 4)
        if rows
        else 0.0,
        "think_leak_rate": round(sum(bool(row.get("think_leak_detected")) for row in rows) / len(rows), 4)
        if rows
        else 0.0,
        "simulated_usage": True,
        "actual_user_feedback": False,
        "actual_product_benefit_claim_allowed": False,
    })
    _write_json(root / f"freeze_check_{variant}.json", freeze)
    _write_json(metrics_path, metrics)
    print(json.dumps({key: metrics.get(key) for key in (
        "variant", "session_count", "model_call_count", "personalization_score",
        "hard_gate_pass_rate", "truncated_session_rate", "privacy_canary_echo_rate",
    )}, ensure_ascii=False, indent=2))
    return 0 if metrics["all_sessions_completed"] else 1


def _collect_metrics(scope: str) -> dict[str, dict[str, Any]]:
    variants = PHASE81_SANITY_VARIANTS if scope == "sanity" else PHASE81_VARIANTS
    return {
        variant: _read_json(GENERATION_ROOT / scope / f"metrics_{variant}.json")
        for variant in variants
    }


def _sanity_decision() -> int:
    decision = build_phase81_sanity_decision(
        metrics=_collect_metrics("sanity"),
        training_attempt=_read_json(TRAINING_ROOT / "probe-4step/training_attempt.json"),
    )
    _write_json(EVIDENCE_ROOT / "sanity_decision.json", decision)
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return 0 if decision["passed"] else 2


def _public_private_audit() -> dict[str, Any]:
    sessions = [dict(row) for row in _read_json(PREPARATION_ROOT / "holdout.json").get("sessions") or []]
    by_id = {str(row["session_id"]): row for row in sessions}
    details = []
    for variant in PHASE81_VARIANTS:
        for row in _read_jsonl(GENERATION_ROOT / "full" / f"transcripts_{variant}.jsonl"):
            session = by_id.get(str(row.get("session_id") or ""), {})
            values = [str(value) for value in session.get("declared_private_values") or [] if str(value)]
            serialized = json.dumps(row, ensure_ascii=False, sort_keys=True)
            details.append({
                "variant": variant,
                "session_id": row.get("session_id"),
                "raw_private_match_count": sum(value in serialized for value in values),
                "returned_private_value_detected": bool(row.get("returned_private_value_detected")),
                "model_input_private_value_detected": any(
                    bool(item.get("model_input_contains_declared_private_value"))
                    for item in row.get("private_input_guards") or []
                ),
            })
    expected = PHASE81_SESSION_COUNT * len(PHASE81_VARIANTS)
    checks = {
        "expected_transcript_count": len(details) == expected,
        "raw_private_match_count_zero": not any(row["raw_private_match_count"] for row in details),
        "returned_private_value_zero": not any(row["returned_private_value_detected"] for row in details),
        "model_input_private_value_zero": not any(row["model_input_private_value_detected"] for row in details),
    }
    return {
        "kind": "phase81_public_private_transcript_audit",
        "passed": all(checks.values()),
        "checks": checks,
        "transcript_count": len(details),
        "expected_transcript_count": expected,
        "details": details,
    }


def _output_examples(transcripts: Mapping[str, list[dict[str, Any]]]) -> str:
    selected = (
        "phase81-evidence_truthfulness-01",
        "phase81-concise_workstyle-01",
        "phase81-privacy_non_echo-01",
        "phase81-ordinary_direct-01",
    )
    by_variant = {
        variant: {str(row.get("session_id")): row for row in rows}
        for variant, rows in transcripts.items()
    }
    lines = [
        "# Phase81 Output Examples",
        "",
        "Real local outputs on fresh simulated_usage sessions. This is laboratory evidence, not actual-user product evidence.",
        "",
    ]
    for session_id in selected:
        lines.extend((f"## {session_id}", ""))
        for variant in PHASE81_VARIANTS:
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
        "kind": "phase81_evidence_manifest",
        "files": files,
        "file_count": len(files),
        "manifest_sha256": stable_hash(files),
    }


def _post_experiment_decision_amendment() -> dict[str, Any]:
    freeze = _read_json(EVIDENCE_ROOT / "pre_experiment_freeze.json")
    holdout = _read_json(PREPARATION_ROOT / "holdout.json")
    training4 = _read_json(TRAINING_ROOT / "probe-4step/training_attempt.json")
    training12 = _read_json(TRAINING_ROOT / "probe-12step/training_attempt.json")
    metrics_paths = [
        GENERATION_ROOT / "full" / f"metrics_{variant}.json"
        for variant in PHASE81_VARIANTS
    ]
    transcript_paths = [
        GENERATION_ROOT / "full" / f"transcripts_{variant}.jsonl"
        for variant in PHASE81_VARIANTS
    ]
    current_sources = {
        "core": _sha256(CORE_SOURCE),
        "driver": _sha256(DRIVER_SOURCE),
        "test": _sha256(TEST_SOURCE),
        "executor": _sha256(EXECUTOR_SOURCE),
    }
    checks = {
        "frozen_thresholds_unchanged": stable_hash(FROZEN_THRESHOLDS)
        == freeze.get("thresholds_sha256"),
        "generation_protocol_unchanged": stable_hash(GENERATION_PROTOCOL)
        == freeze.get("generation_protocol_sha256"),
        "holdout_unchanged": stable_hash(holdout.get("sessions") or [])
        == freeze.get("holdout_manifest_sha256"),
        "real_4_step_training_preserved": training4.get("status") == "completed"
        and training4.get("real_training") is True,
        "real_12_step_training_preserved": training12.get("status") == "completed"
        and training12.get("real_training") is True,
        "all_full_metrics_preserved": all(path.is_file() for path in metrics_paths),
        "all_full_transcripts_preserved": all(path.is_file() for path in transcript_paths),
        "executor_source_unchanged": current_sources["executor"]
        == dict(freeze.get("source_sha256") or {}).get("executor"),
    }
    return {
        "kind": "phase81_post_experiment_decision_taxonomy_amendment",
        "created_at": _utcnow(),
        "passed": all(checks.values()),
        "reason": (
            "The frozen adapter truncation threshold was incorrectly grouped with evidence-completeness "
            "checks. It is now a benefit gate, so failure archives the adapter instead of claiming missing "
            "evidence. The numeric threshold, training, holdout, generation protocol, outputs, and scores are unchanged."
        ),
        "checks": checks,
        "frozen_source_sha256": freeze.get("source_sha256"),
        "current_source_sha256": current_sources,
        "source_change_expected": True,
        "threshold_or_score_change": False,
        "training_artifact_sha256": {
            "probe_4step": dict(training4.get("adapter_validation") or {}).get("sha256"),
            "probe_12step": dict(training12.get("adapter_validation") or {}).get("sha256"),
        },
        "full_metrics_sha256": {path.name: _sha256(path) for path in metrics_paths if path.is_file()},
        "full_transcripts_sha256": {path.name: _sha256(path) for path in transcript_paths if path.is_file()},
        "retraining_required": False,
        "regeneration_required": False,
    }


def _finalize() -> int:
    sanity = _read_json(EVIDENCE_ROOT / "sanity_decision.json")
    if sanity.get("passed") is not True:
        decision = {
            **sanity,
            "kind": "phase81_final_decision",
            "recommendation": "phase82_revise_mid_model_probe_before_12step",
            "promotion_allowed": False,
            "hermes_attachment_allowed": False,
            "product_default_changed": False,
        }
        _write_json(EVIDENCE_ROOT / "phase81-final-decision.json", decision)
        _write_text(EVIDENCE_ROOT / "phase81-final-decision.md", "# Phase81 Final Decision\n\nThe frozen 4-step sanity gate failed. The 12-step probe was not started.")
        _write_json(EVIDENCE_ROOT / "evidence_manifest.json", _evidence_manifest())
        return 2
    amendment = _post_experiment_decision_amendment()
    _write_json(EVIDENCE_ROOT / "post_experiment_decision_amendment.json", amendment)
    if amendment.get("passed") is not True:
        raise SystemExit(f"Phase81 decision amendment integrity failed: {amendment}")
    metrics = _collect_metrics("full")
    training = _read_json(TRAINING_ROOT / "probe-12step/training_attempt.json")
    selection = _read_json(PREPARATION_ROOT / "model_selection.json")
    isolation = _read_json(PREPARATION_ROOT / "isolation_audit.json")
    privacy = _public_private_audit()
    decision = build_phase81_final_decision(
        metrics=metrics,
        training_attempt=training,
        model_selection=selection,
        isolation_audit=isolation,
        public_private_audit=privacy,
    )
    transcripts = {
        variant: _read_jsonl(GENERATION_ROOT / "full" / f"transcripts_{variant}.jsonl")
        for variant in PHASE81_VARIANTS
    }
    comparison = {
        "kind": "phase81_mid_model_comparison",
        "created_at": _utcnow(),
        "selected_model": selection.get("selected_model"),
        "model_revision": MODEL_REVISION,
        "training_attempt": training,
        "sanity_decision": sanity,
        "metrics": metrics,
        "phase80_reference_ceiling": {
            "model": "Qwen3-4B",
            "variant": "runtime_4b",
            "target_score": 0.7122,
            "truncated_session_rate": 0.2381,
            "source": "docs/demo/phase80-small-model-failure-taxonomy/phase80-final-decision.json",
            "canonical_reference_only": True,
        },
        "decision": decision,
        "simulated_usage": True,
        "actual_user_feedback_count": 0,
        "actual_product_benefit_claim_allowed": False,
    }
    _write_json(EVIDENCE_ROOT / "public_private_audit.json", privacy)
    _write_json(EVIDENCE_ROOT / "comparison_summary.json", comparison)
    _write_json(EVIDENCE_ROOT / "phase81-final-decision.json", decision)
    _write_text(EVIDENCE_ROOT / "output_examples.md", _output_examples(transcripts))
    _write_text(EVIDENCE_ROOT / "phase81-final-decision.md", f"""# Phase81 Final Decision

Recommendation: **{decision['recommendation']}**

- Lifecycle status: `{decision['status']}`
- Selected model: `{selection.get('selected_model')}`
- Real 4-step training: `{_read_json(TRAINING_ROOT / 'probe-4step/training_attempt.json').get('status')}`
- Real 12-step training: `{training.get('status')}`
- Base target score: `{decision['target_scores']['base_mid_length_control']}`
- Runtime target score: `{decision['target_scores']['runtime_mid_length_control']}`
- Adapter target score: `{decision['target_scores']['adapter_mid_12step_length_control']}`
- Adapter gain over base: `{decision['adapter_gain_vs_base']}`
- Runtime gain over base: `{decision['runtime_gain_vs_base']}`

Phase81 uses fresh `simulated_usage`. Training completion and adapter benefit are separate claims. No actual user feedback, product-benefit claim, promotion, Hermes attachment, or product-default change is allowed.
""")
    _write_text(EVIDENCE_ROOT / "phase81-runbook.md", """# Phase81 Runbook

```bash
.venv/bin/python tools/phase81_trainable_mid_model_selection.py prepare --clean
.venv/bin/python tools/phase81_trainable_mid_model_selection.py train --steps 4 --clean
.venv/bin/python tools/phase81_trainable_mid_model_selection.py generate --scope sanity --variant base_mid_4step_sanity --clean
.venv/bin/python tools/phase81_trainable_mid_model_selection.py generate --scope sanity --variant adapter_mid_4step_sanity --clean
.venv/bin/python tools/phase81_trainable_mid_model_selection.py sanity
.venv/bin/python tools/phase81_trainable_mid_model_selection.py train --steps 12 --clean
.venv/bin/python tools/phase81_trainable_mid_model_selection.py generate --scope full --variant base_mid_length_control --clean
.venv/bin/python tools/phase81_trainable_mid_model_selection.py generate --scope full --variant runtime_mid_length_control --clean
.venv/bin/python tools/phase81_trainable_mid_model_selection.py generate --scope full --variant adapter_mid_12step_length_control --clean
.venv/bin/python tools/phase81_trainable_mid_model_selection.py full-regression
.venv/bin/python tools/phase81_trainable_mid_model_selection.py finalize
.venv/bin/python tools/phase81_trainable_mid_model_selection.py validate
```

The model revision, fresh holdout, curriculum, generation protocol, and success thresholds are frozen before training. The 12-step probe is blocked unless the 4-step training and seven-session sanity gate pass.
""")
    next_goal = {
        "phase82_full_coverage_mid_model_probe": "Run a frozen full-coverage mid-model curriculum probe only after the Phase81 simulated adapter gain is reproduced.",
        "phase82_mid_model_runtime_contract_path": "Productize the same-model runtime contract and archive the non-beneficial 12-step adapter.",
        "phase82_curriculum_or_training_objective_revision": "Diagnose curriculum coverage and loss objective before any longer mid-model training.",
        "repair_phase81_evidence": "Repair missing Phase81 evidence without changing frozen gates.",
    }.get(decision["recommendation"], "Review the Phase81 archive evidence before another training run.")
    _write_text(EVIDENCE_ROOT / "next-pursuit-goal.md", f"# Phase82 Pursuit Goal\n\n{next_goal}")
    manifest = _evidence_manifest()
    _write_json(EVIDENCE_ROOT / "evidence_manifest.json", manifest)
    integrity = {
        "kind": "phase81_evidence_integrity",
        "passed": True,
        "manifest_file_count": manifest["file_count"],
        "manifest_sha256": manifest["manifest_sha256"],
        "actual_user_feedback_count": 0,
        "actual_product_benefit_claim_allowed": False,
        "auto_promotion_allowed": False,
    }
    _write_json(EVIDENCE_ROOT / "evidence_integrity.json", integrity)
    _write_json(EVIDENCE_ROOT / "finalization_state.json", {
        "kind": "phase81_finalization_state",
        "created_at": _utcnow(),
        "status": "finalized",
    })
    print(json.dumps({
        "status": decision["status"],
        "recommendation": decision["recommendation"],
        "target_scores": decision["target_scores"],
        "adapter_gain_vs_base": decision["adapter_gain_vs_base"],
        "runtime_gain_vs_base": decision["runtime_gain_vs_base"],
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
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
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
            "tests/test_phase81_trainable_mid_model_selection.py",
            "tests/test_phase80_small_model_failure_taxonomy.py",
            "tests/test_phase79_cpu_feasible_persona_probe.py",
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
        "kind": "phase81_full_regression_summary",
        "created_at": _utcnow(),
        "passed": len(results) == len(commands) and all(row["exit_code"] == 0 for row in results),
        "results": results,
    }
    _write_json(EVIDENCE_ROOT / "full_regression_summary.json", summary)
    return 0 if summary["passed"] else 1


def _validate() -> int:
    manifest = _read_json(EVIDENCE_ROOT / "evidence_manifest.json")
    integrity = _read_json(EVIDENCE_ROOT / "evidence_integrity.json")
    decision = _read_json(EVIDENCE_ROOT / "phase81-final-decision.json")
    training4 = _read_json(TRAINING_ROOT / "probe-4step/training_attempt.json")
    training12 = _read_json(TRAINING_ROOT / "probe-12step/training_attempt.json")
    regression = _read_json(EVIDENCE_ROOT / "full_regression_summary.json")
    amendment = _read_json(EVIDENCE_ROOT / "post_experiment_decision_amendment.json")
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
        if "SYNTHETIC_PHASE81_PRIVATE_" in text:
            raw_private_locations.append(str(path.relative_to(EVIDENCE_ROOT)))
    checks = {
        "manifest_files_unchanged": not manifest_failures,
        "integrity_passed": integrity.get("passed") is True,
        "real_4_step_training_completed": training4.get("status") == "completed"
        and training4.get("real_training") is True,
        "real_12_step_training_completed": training12.get("status") == "completed"
        and training12.get("real_training") is True,
        "full_regression_passed": regression.get("passed") is True,
        "post_experiment_amendment_passed": amendment.get("passed") is True
        and amendment.get("threshold_or_score_change") is False
        and amendment.get("retraining_required") is False
        and amendment.get("regeneration_required") is False,
        "private_canaries_only_in_frozen_holdout": raw_private_locations
        == ["evidence-preparation/holdout.json"],
        "no_actual_user_claim": decision.get("actual_product_benefit_claim_allowed") is False,
        "no_auto_promotion": decision.get("auto_promotion_allowed") is False,
        "no_hermes_attachment": decision.get("hermes_attachment_allowed") is False,
        "product_default_unchanged": decision.get("product_default_changed") is False,
    }
    summary = {
        "kind": "phase81_validation_summary",
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
    train.add_argument("--steps", type=int, choices=(4, 12), required=True)
    train.add_argument("--clean", action="store_true")
    generate = subparsers.add_parser("generate")
    generate.add_argument("--scope", choices=("sanity", "full"), required=True)
    generate.add_argument("--variant", choices=PHASE81_SANITY_VARIANTS + PHASE81_VARIANTS, required=True)
    generate.add_argument("--clean", action="store_true")
    subparsers.add_parser("sanity")
    subparsers.add_parser("full-regression")
    subparsers.add_parser("finalize")
    subparsers.add_parser("validate")
    args = parser.parse_args()
    if args.command == "prepare":
        return _prepare(args.clean)
    if args.command == "train":
        return _train(args.steps, args.clean)
    if args.command == "generate":
        return _generate(args.scope, args.variant, args.clean)
    if args.command == "sanity":
        return _sanity_decision()
    if args.command == "full-regression":
        return _full_regression()
    if args.command == "finalize":
        return _finalize()
    if args.command == "validate":
        return _validate()
    raise SystemExit(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
