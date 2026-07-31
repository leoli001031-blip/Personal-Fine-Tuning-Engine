from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DRIVER_PATH = ROOT / "tools/phase92_95_autonomous_dpo_stability_product_proof.py"
SPEC = importlib.util.spec_from_file_location("phase93_95_driver", DRIVER_PATH)
assert SPEC and SPEC.loader
driver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(driver)


def _row(index: int) -> dict:
    return {
        "sample_id": f"simulated-{index}",
        "instruction": "prompt",
        "chosen": "chosen",
        "rejected": "rejected",
        "simulated_usage": True,
        "actual_user_feedback": False,
    }


def test_phase93_12_and_30_jobs_start_independently_from_phase89(tmp_path: Path) -> None:
    runtime = {
        "probe_id": "mps_float32",
        "runtime_device": "mps",
        "runtime_dtype": "float32",
        "learning_rate": 5e-6,
    }
    job12 = driver.build_phase93_job_spec([_row(i) for i in range(12)], steps=12, output_dir=tmp_path / "12", runtime=runtime)
    job30 = driver.build_phase93_job_spec([_row(i) for i in range(30)], steps=30, output_dir=tmp_path / "30", runtime=runtime)

    for job, steps in ((job12, 12), (job30, 30)):
        training = job["recipe"]["training"]
        assert training["max_steps"] == steps
        assert training["runtime_device"] == "mps"
        assert training["runtime_dtype"] == "float32"
        assert training["incremental_context"]["parent_adapter_path"] == str(driver.PARENT_ADAPTER_ROOT)
        assert job["phase93"]["phase92_probe_adapter_used_as_parent"] is False
        assert job["phase93"]["phase90_holdout_used_for_training"] is False


def test_parser_rejects_external_variants_and_unbounded_steps() -> None:
    parser = driver._parser()

    try:
        parser.parse_args(["phase93-train", "--steps", "60"])
    except SystemExit:
        pass
    else:
        raise AssertionError("60-step training should be rejected")

    try:
        parser.parse_args(["phase94-generate", "--scope", "product", "--variant", "external"])
    except SystemExit:
        pass
    else:
        raise AssertionError("external provider variant should be rejected")


def test_private_review_cache_is_outside_repository() -> None:
    assert str(driver.PRIVATE_REVIEW_ROOT).startswith("/private/tmp/")
    assert driver.REPO_ROOT not in driver.PRIVATE_REVIEW_ROOT.parents
