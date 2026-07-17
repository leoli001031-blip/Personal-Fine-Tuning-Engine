from __future__ import annotations

from typing import Any, Iterable, Mapping


PHASE92_PROBE_STEPS = 4
PHASE92_BASE_LEARNING_RATE = 5e-6
PHASE92_LOW_LEARNING_RATE = 1e-6


def build_phase92_probe_matrix(*, mps_available: bool) -> list[dict[str, Any]]:
    probes = [
        {
            "probe_id": "cpu_float32",
            "runtime_device": "cpu",
            "runtime_dtype": "float32",
            "learning_rate": PHASE92_BASE_LEARNING_RATE,
            "max_steps": PHASE92_PROBE_STEPS,
            "run_order": 1,
            "run_condition": "always",
        }
    ]
    if mps_available:
        probes.extend([
            {
                "probe_id": "mps_float32",
                "runtime_device": "mps",
                "runtime_dtype": "float32",
                "learning_rate": PHASE92_BASE_LEARNING_RATE,
                "max_steps": PHASE92_PROBE_STEPS,
                "run_order": 2,
                "run_condition": "after_cpu_float32",
            },
            {
                "probe_id": "mps_float32_low_lr",
                "runtime_device": "mps",
                "runtime_dtype": "float32",
                "learning_rate": PHASE92_LOW_LEARNING_RATE,
                "max_steps": PHASE92_PROBE_STEPS,
                "run_order": 3,
                "run_condition": "only_if_cpu_passes_and_mps_base_lr_is_non_finite",
            },
        ])
    return probes


def reconstruct_phase91_runtime(*, cuda_available: bool, mps_available: bool) -> dict[str, Any]:
    accelerator_available = cuda_available or mps_available
    use_cpu = not accelerator_available
    legacy_device_map = "auto" if cuda_available and not use_cpu else {"": "cpu"}
    legacy_dtype = "float16" if accelerator_available and not use_cpu else "float32"
    reported_device = "cpu" if use_cpu else ("cuda" if cuda_available else "mps")
    actual_device = "cuda" if legacy_device_map == "auto" else "cpu"
    return {
        "cuda_available": cuda_available,
        "mps_available": mps_available,
        "legacy_use_cpu": use_cpu,
        "legacy_device_map": legacy_device_map,
        "legacy_dtype": legacy_dtype,
        "legacy_reported_device": reported_device,
        "legacy_actual_load_device": actual_device,
        "device_report_mismatch": reported_device != actual_device,
        "cpu_float16_mismatch": actual_device == "cpu" and legacy_dtype == "float16",
    }


def assess_phase92_probe(
    attempt: Mapping[str, Any],
    *,
    required_steps: int = PHASE92_PROBE_STEPS,
) -> dict[str, Any]:
    result = dict(attempt.get("result") or {})
    real = dict(result.get("real_execution") or {})
    validation = dict(attempt.get("adapter_validation") or {})
    checks = {
        "status_completed": result.get("status") == "completed",
        "exact_optimizer_steps": int(real.get("steps") or 0) == required_steps,
        "parameters_updated": real.get("parameters_updated") is True,
        "adapter_valid": validation.get("valid") is True,
        "non_finite_error_absent": "non-finite" not in str(result.get("error") or "").lower(),
    }
    return {
        "probe_id": attempt.get("probe_id"),
        "passed": all(checks.values()),
        "checks": checks,
        "error": result.get("error"),
        "duration_seconds": attempt.get("duration_seconds"),
    }


def third_probe_allowed(assessments: Mapping[str, Mapping[str, Any]]) -> bool:
    cpu = dict(assessments.get("cpu_float32") or {})
    mps = dict(assessments.get("mps_float32") or {})
    return cpu.get("passed") is True and (
        "non-finite" in str(mps.get("error") or "").lower()
        or "nan" in str(mps.get("error") or "").lower()
    )


def select_phase92_runtime(
    attempts: Iterable[Mapping[str, Any]],
    *,
    mps_available: bool,
) -> dict[str, Any]:
    assessed = {
        str(row.get("probe_id")): assess_phase92_probe(row)
        for row in attempts
    }
    required = {"cpu_float32"}
    if mps_available:
        required.add("mps_float32")
    missing = sorted(required - set(assessed))
    if missing:
        return {
            "status": "blocked_missing_required_probes",
            "selected_probe_id": None,
            "missing_probes": missing,
            "assessments": assessed,
        }

    if assessed.get("mps_float32", {}).get("passed") is True:
        selected = "mps_float32"
    elif third_probe_allowed(assessed) and "mps_float32_low_lr" not in assessed:
        return {
            "status": "third_probe_required",
            "selected_probe_id": None,
            "missing_probes": ["mps_float32_low_lr"],
            "assessments": assessed,
        }
    elif assessed.get("mps_float32_low_lr", {}).get("passed") is True:
        selected = "mps_float32_low_lr"
    elif assessed.get("cpu_float32", {}).get("passed") is True:
        selected = "cpu_float32"
    else:
        return {
            "status": "archive_no_stable_runtime",
            "selected_probe_id": None,
            "missing_probes": [],
            "assessments": assessed,
        }

    return {
        "status": "stable_runtime_selected",
        "selected_probe_id": selected,
        "missing_probes": [],
        "assessments": assessed,
        "product_gate_qualified": False,
        "automatic_promotion_allowed": False,
    }
