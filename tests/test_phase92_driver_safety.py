from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DRIVER_PATH = ROOT / "tools/phase92_95_autonomous_dpo_stability_product_proof.py"
SPEC = importlib.util.spec_from_file_location("phase92_95_driver", DRIVER_PATH)
assert SPEC and SPEC.loader
driver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(driver)


def test_phase92_job_spec_freezes_exact_runtime_and_four_steps(tmp_path: Path) -> None:
    probe = {
        "probe_id": "cpu_float32",
        "runtime_device": "cpu",
        "runtime_dtype": "float32",
        "learning_rate": 5e-6,
        "max_steps": 4,
    }
    rows = [{
        "sample_id": "simulated-1",
        "instruction": "prompt",
        "chosen": "chosen",
        "rejected": "rejected",
        "simulated_usage": True,
        "actual_user_feedback": False,
    }]

    spec = driver.build_phase92_job_spec(rows, probe, tmp_path / "output")
    training = spec["recipe"]["training"]

    assert training["max_steps"] == 4
    assert training["runtime_device"] == "cpu"
    assert training["runtime_dtype"] == "float32"
    assert training["incremental_context"]["parent_adapter_path"] == str(driver.PARENT_ADAPTER_ROOT)
    assert spec["phase92"]["simulated_usage"] is True
    assert spec["phase92"]["actual_user_feedback"] is False
