from __future__ import annotations

from pfe_core.phase92_dpo_numerical_stability import (
    build_phase92_probe_matrix,
    reconstruct_phase91_runtime,
    select_phase92_runtime,
)


def _attempt(probe_id: str, *, passed: bool, error: str | None = None) -> dict:
    return {
        "probe_id": probe_id,
        "result": {
            "status": "completed" if passed else "failed",
            "error": error,
            "real_execution": {
                "steps": 4 if passed else 0,
                "parameters_updated": passed,
            },
        },
        "adapter_validation": {"valid": passed},
    }


def test_probe_matrix_is_bounded_and_cpu_runs_first() -> None:
    matrix = build_phase92_probe_matrix(mps_available=True)

    assert [row["probe_id"] for row in matrix] == [
        "cpu_float32",
        "mps_float32",
        "mps_float32_low_lr",
    ]
    assert all(row["max_steps"] == 4 for row in matrix)


def test_phase91_runtime_reconstruction_exposes_cpu_float16_mismatch() -> None:
    reconstructed = reconstruct_phase91_runtime(cuda_available=False, mps_available=True)

    assert reconstructed["legacy_reported_device"] == "mps"
    assert reconstructed["legacy_actual_load_device"] == "cpu"
    assert reconstructed["legacy_dtype"] == "float16"
    assert reconstructed["device_report_mismatch"] is True
    assert reconstructed["cpu_float16_mismatch"] is True


def test_selector_prefers_stable_mps_after_required_cpu_control() -> None:
    decision = select_phase92_runtime([
        _attempt("cpu_float32", passed=True),
        _attempt("mps_float32", passed=True),
    ], mps_available=True)

    assert decision["status"] == "stable_runtime_selected"
    assert decision["selected_probe_id"] == "mps_float32"


def test_selector_requests_third_probe_only_for_mps_non_finite_failure() -> None:
    decision = select_phase92_runtime([
        _attempt("cpu_float32", passed=True),
        _attempt("mps_float32", passed=False, error="TrainingError: non-finite grad_norm nan"),
    ], mps_available=True)

    assert decision["status"] == "third_probe_required"
    assert decision["missing_probes"] == ["mps_float32_low_lr"]


def test_selector_falls_back_to_cpu_when_mps_failure_is_not_numerical() -> None:
    decision = select_phase92_runtime([
        _attempt("cpu_float32", passed=True),
        _attempt("mps_float32", passed=False, error="RuntimeError: unsupported operation"),
    ], mps_available=True)

    assert decision["status"] == "stable_runtime_selected"
    assert decision["selected_probe_id"] == "cpu_float32"
