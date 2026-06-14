from __future__ import annotations

import importlib.util
from pathlib import Path


TOOL_PATH = Path(__file__).resolve().parents[1] / "tools" / "verify_cuda_real_training.py"
SPEC = importlib.util.spec_from_file_location("verify_cuda_real_training", TOOL_PATH)
assert SPEC is not None
verify_cuda = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(verify_cuda)


def test_build_job_spec_uses_isolated_real_training_shape(tmp_path: Path) -> None:
    job = verify_cuda.build_job_spec(
        backend="peft",
        base_model="sshleifer/tiny-gpt2",
        output_dir=tmp_path,
        epochs=1,
        max_seq_length=64,
        learning_rate=1e-5,
        timeout_seconds=30,
    )

    assert job["backend"] == "peft"
    assert job["execution_executor"] == "peft"
    assert job["real_local"] is True
    assert job["real_training_enabled"] is True
    assert job["timeout_seconds"] == 30
    assert job["training_examples"][0]["chosen"] == "pong"
    assert job["recipe"]["training"]["output_dir"].endswith("peft_output")
    assert job["recipe"]["peft"]["lora_config"]["r"] == 2


def test_summarize_result_extracts_diagnostics_paths(tmp_path: Path) -> None:
    result = {
        "status": "completed",
        "dry_run": False,
        "returncode": 0,
        "diagnostics": {
            "signal_name": None,
            "stdout_log": str(tmp_path / "stdout.log"),
            "stderr_log": str(tmp_path / "stderr.log"),
        },
        "runner_result": {
            "status": "completed",
            "execution_mode": "real_import",
            "real_execution": {
                "kind": "real_peft",
                "success": True,
                "artifact_dir": str(tmp_path / "peft_lora"),
            },
        },
    }

    summary = verify_cuda.summarize_result(result, backend="peft", output_dir=tmp_path)

    assert summary["backend"] == "peft"
    assert summary["status"] == "completed"
    assert summary["returncode"] == 0
    assert summary["runner_status"] == "completed"
    assert summary["real_execution_kind"] == "real_peft"
    assert summary["real_execution_success"] is True
    assert summary["adapter_path"] == str(tmp_path / "peft_lora")
