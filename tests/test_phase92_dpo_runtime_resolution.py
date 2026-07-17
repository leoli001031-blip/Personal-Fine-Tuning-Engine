from __future__ import annotations

import pytest

from pfe_core.errors import TrainingError
from pfe_core.trainer.executors import resolve_dpo_runtime_config


def test_auto_runtime_uses_real_mps_device_with_float32() -> None:
    resolved = resolve_dpo_runtime_config(
        requested_device="auto",
        requested_dtype="auto",
        cuda_available=False,
        mps_available=True,
    )

    assert resolved == {
        "requested_device": "auto",
        "requested_dtype": "auto",
        "device": "mps",
        "dtype": "float32",
        "device_map": {"": "mps"},
        "use_cpu": False,
    }


def test_explicit_cpu_is_float32_even_when_mps_is_available() -> None:
    resolved = resolve_dpo_runtime_config(
        requested_device="cpu",
        requested_dtype="auto",
        cuda_available=False,
        mps_available=True,
    )

    assert resolved["device"] == "cpu"
    assert resolved["dtype"] == "float32"
    assert resolved["device_map"] == {"": "cpu"}
    assert resolved["use_cpu"] is True


def test_auto_cuda_keeps_float16_default() -> None:
    resolved = resolve_dpo_runtime_config(
        requested_device="auto",
        requested_dtype="auto",
        cuda_available=True,
        mps_available=False,
    )

    assert resolved["device"] == "cuda"
    assert resolved["dtype"] == "float16"
    assert resolved["device_map"] == "auto"


def test_unavailable_requested_device_is_rejected() -> None:
    with pytest.raises(TrainingError, match="requested DPO runtime device mps is unavailable"):
        resolve_dpo_runtime_config(
            requested_device="mps",
            requested_dtype="float32",
            cuda_available=False,
            mps_available=False,
        )


def test_cpu_float16_is_rejected_instead_of_repeating_phase91_failure() -> None:
    with pytest.raises(TrainingError, match="CPU float16 is not allowed"):
        resolve_dpo_runtime_config(
            requested_device="cpu",
            requested_dtype="float16",
            cuda_available=False,
            mps_available=True,
        )
