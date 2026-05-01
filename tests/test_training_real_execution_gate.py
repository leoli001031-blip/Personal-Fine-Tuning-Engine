"""Tests for PFE_REAL_TRAINING execution gate (Phase 6)."""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any

import pytest
from pfe_core.trainer import real_execution, runtime_job
from pfe_core.trainer.preflight import TrainingPreflight
from pfe_core.trainer.runtime_job import dispatch_training_job


@pytest.fixture(autouse=True)
def clear_real_training_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure PFE_REAL_TRAINING is unset for each test unless explicitly set."""
    monkeypatch.delenv("PFE_REAL_TRAINING", raising=False)


class TestMockLocalBackend:
    """mock_local should never be gated."""

    def test_mock_local_allowed_without_env(self) -> None:
        result = dispatch_training_job({"backend": "mock_local"}, dry_run=False)
        assert result.get("status") != "blocked"
        assert "Real training execution disabled" not in str(result)

    def test_mock_local_allowed_with_env(self) -> None:
        os.environ["PFE_REAL_TRAINING"] = "1"
        try:
            result = dispatch_training_job({"backend": "mock_local"}, dry_run=False)
            assert result.get("status") != "blocked"
        finally:
            del os.environ["PFE_REAL_TRAINING"]

    def test_mock_local_dry_run_allowed(self) -> None:
        result = dispatch_training_job({"backend": "mock_local"}, dry_run=True)
        assert result.get("status") != "blocked"


class TestMLXBackend:
    """mlx is a real backend and should be gated."""

    def test_mlx_blocked_without_env(self) -> None:
        result = dispatch_training_job({"backend": "mlx"}, dry_run=False)
        assert result.get("status") == "blocked"
        assert "Real training execution disabled" in result.get("reason", "")
        assert result.get("backend") == "mlx"

    def test_mlx_allowed_with_env_uses_subprocess_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PFE_REAL_TRAINING", "1")
        monkeypatch.setattr(real_execution, "run_training_preflight", lambda job_spec, *, backend: {"ready": True})

        def fake_subprocess(job_spec: Any, *, backend: str, dry_run: bool) -> dict[str, Any]:
            return {"status": "completed", "backend": backend, "dry_run": dry_run, "subprocess": True}

        monkeypatch.setattr(real_execution, "run_backend_in_subprocess", fake_subprocess)
        result = dispatch_training_job({"backend": "mlx"}, dry_run=False)
        assert result == {"status": "completed", "backend": "mlx", "dry_run": False, "subprocess": True}

    def test_mlx_env_still_blocks_when_preflight_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PFE_REAL_TRAINING", "1")
        monkeypatch.setattr(
            real_execution,
            "run_training_preflight",
            lambda job_spec, *, backend: {"ready": False, "reasons": ["missing_base_model"]},
        )
        result = dispatch_training_job({"backend": "mlx"}, dry_run=False)
        assert result.get("status") == "blocked"
        assert result.get("reason") == "preflight failed"

    def test_mlx_dry_run_bypasses_gate(self) -> None:
        result = dispatch_training_job({"backend": "mlx"}, dry_run=True)
        assert result.get("status") != "blocked"
        assert "Real training execution disabled" not in str(result)


class TestPEFTBackend:
    """peft is a real backend and should be gated."""

    def test_peft_blocked_without_env(self) -> None:
        result = dispatch_training_job({"backend": "peft"}, dry_run=False)
        assert result.get("status") == "blocked"
        assert "Real training execution disabled" in result.get("reason", "")
        assert result.get("backend") == "peft"

    def test_peft_allowed_with_env_uses_subprocess_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PFE_REAL_TRAINING", "1")
        monkeypatch.setattr(real_execution, "run_training_preflight", lambda job_spec, *, backend: {"ready": True})

        def fake_subprocess(job_spec: Any, *, backend: str, dry_run: bool) -> dict[str, Any]:
            return {"status": "completed", "backend": backend, "dry_run": dry_run, "subprocess": True}

        monkeypatch.setattr(real_execution, "run_backend_in_subprocess", fake_subprocess)
        result = dispatch_training_job({"backend": "peft"}, dry_run=False)
        assert result == {"status": "completed", "backend": "peft", "dry_run": False, "subprocess": True}

    def test_peft_child_subprocess_executes_backend_directly(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PFE_REAL_TRAINING", "1")
        monkeypatch.setenv("PFE_TRAINING_SUBPROCESS", "1")
        monkeypatch.setattr(
            runtime_job,
            "execute_peft_training",
            lambda *, job_spec, dry_run: {"status": "prepared", "backend": "peft", "dry_run": dry_run},
        )
        monkeypatch.setattr(
            real_execution,
            "run_backend_in_subprocess",
            lambda job_spec, *, backend, dry_run: pytest.fail("child subprocess must not rematerialize itself"),
        )

        result = dispatch_training_job({"backend": "peft"}, dry_run=False)
        assert result == {"status": "prepared", "backend": "peft", "dry_run": False}

    def test_peft_dry_run_bypasses_gate(self) -> None:
        result = dispatch_training_job({"backend": "peft"}, dry_run=True)
        assert result.get("status") != "blocked"
        assert "Real training execution disabled" not in str(result)


class TestDPOBackend:
    """dpo is a real backend and should be gated."""

    def test_dpo_blocked_without_env(self) -> None:
        result = dispatch_training_job({"backend": "dpo"}, dry_run=False)
        assert result.get("status") == "blocked"
        assert "Real training execution disabled" in result.get("reason", "")
        assert result.get("backend") == "dpo"

    def test_dpo_allowed_with_env_uses_subprocess_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PFE_REAL_TRAINING", "1")
        monkeypatch.setattr(real_execution, "run_training_preflight", lambda job_spec, *, backend: {"ready": True})

        def fake_subprocess(job_spec: Any, *, backend: str, dry_run: bool) -> dict[str, Any]:
            return {"status": "completed", "backend": backend, "dry_run": dry_run, "subprocess": True}

        monkeypatch.setattr(real_execution, "run_backend_in_subprocess", fake_subprocess)
        result = dispatch_training_job(
            {
                "backend": "dpo",
                "base_model": "gpt2",
                "training_examples": [{"instruction": "A", "chosen": "B", "rejected": "C"}],
            },
            dry_run=False,
        )
        assert result == {"status": "completed", "backend": "dpo", "dry_run": False, "subprocess": True}

    def test_dpo_env_still_blocks_when_preflight_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PFE_REAL_TRAINING", "1")
        monkeypatch.setattr(
            real_execution,
            "run_training_preflight",
            lambda job_spec, *, backend: {"ready": False, "reasons": ["dpo_import_failed"]},
        )
        result = dispatch_training_job(
            {
                "backend": "dpo",
                "base_model": "gpt2",
                "training_examples": [{"instruction": "A", "chosen": "B", "rejected": "C"}],
            },
            dry_run=False,
        )
        assert result.get("status") == "blocked"
        assert result.get("reason") == "preflight failed"

    def test_dpo_child_subprocess_executes_backend_directly(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Any,
    ) -> None:
        monkeypatch.setenv("PFE_REAL_TRAINING", "1")
        monkeypatch.setenv("PFE_TRAINING_SUBPROCESS", "1")
        monkeypatch.setenv("PFE_HOME", str(tmp_path / ".pfe"))
        monkeypatch.setattr(
            runtime_job,
            "execute_dpo_training",
            lambda *, job_spec, dry_run: {"status": "prepared", "backend": "dpo", "dry_run": dry_run},
        )
        monkeypatch.setattr(
            real_execution,
            "run_backend_in_subprocess",
            lambda job_spec, *, backend, dry_run: pytest.fail("child subprocess must not rematerialize itself"),
        )

        result = dispatch_training_job({"backend": "dpo", "training_examples": []}, dry_run=False)
        assert result == {"status": "prepared", "backend": "dpo", "dry_run": False}

    def test_dpo_dry_run_bypasses_gate(self) -> None:
        result = dispatch_training_job({"backend": "dpo"}, dry_run=True)
        assert result.get("status") != "blocked"


class TestUnslothBackend:
    """unsloth is a real backend and should be gated."""

    def test_unsloth_blocked_without_env(self) -> None:
        result = dispatch_training_job({"backend": "unsloth"}, dry_run=False)
        assert result.get("status") == "blocked"
        assert "Real training execution disabled" in result.get("reason", "")
        assert result.get("backend") == "unsloth"

    def test_unsloth_allowed_with_env_uses_subprocess_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PFE_REAL_TRAINING", "1")
        monkeypatch.setattr(real_execution, "run_training_preflight", lambda job_spec, *, backend: {"ready": True})

        def fake_subprocess(job_spec: Any, *, backend: str, dry_run: bool) -> dict[str, Any]:
            return {"status": "completed", "backend": backend, "dry_run": dry_run, "subprocess": True}

        monkeypatch.setattr(real_execution, "run_backend_in_subprocess", fake_subprocess)
        result = dispatch_training_job({"backend": "unsloth"}, dry_run=False)
        assert result == {"status": "completed", "backend": "unsloth", "dry_run": False, "subprocess": True}

    def test_unsloth_dry_run_bypasses_gate(self) -> None:
        result = dispatch_training_job({"backend": "unsloth"}, dry_run=True)
        assert result.get("status") != "blocked"


class TestDefaultBackend:
    """Default backend (mock_local) should not be gated."""

    def test_default_backend_no_gate(self) -> None:
        result = dispatch_training_job({}, dry_run=False)
        assert result.get("status") != "blocked"
        assert "Real training execution disabled" not in str(result)


class TestExecutionExecutorKey:
    """The execution_executor key should also be gated."""

    def test_execution_executor_peft_blocked(self) -> None:
        result = dispatch_training_job({"execution_executor": "peft"}, dry_run=False)
        assert result.get("status") == "blocked"

    def test_execution_executor_mlx_blocked(self) -> None:
        result = dispatch_training_job({"execution_executor": "mlx"}, dry_run=False)
        assert result.get("status") == "blocked"


class TestSubprocessMaterialization:
    def test_relative_local_paths_are_resolved_before_child_cwd_changes(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Any,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        local_model = tmp_path / "models" / "tiny"
        local_model.mkdir(parents=True)
        captured: dict[str, Any] = {}

        class FakeMaterialized:
            ready = True

            def to_dict(self) -> dict[str, Any]:
                return {"ready": True}

        def fake_materialize_training_job_bundle(*, execution_plan: dict[str, Any], output_dir: Any) -> FakeMaterialized:
            captured["execution_plan"] = execution_plan
            captured["output_dir"] = output_dir
            return FakeMaterialized()

        def fake_run_materialized_training_job_bundle(*args: Any, **kwargs: Any) -> Any:
            return SimpleNamespace(
                success=True,
                failure_category=None,
                returncode=0,
                stdout_log=None,
                stderr_log=None,
                diagnostics={},
                runner_result={"status": "completed"},
                materialization={},
            )

        monkeypatch.setattr(
            real_execution,
            "materialize_training_job_bundle",
            fake_materialize_training_job_bundle,
        )
        monkeypatch.setattr(
            real_execution,
            "run_materialized_training_job_bundle",
            fake_run_materialized_training_job_bundle,
        )

        result = real_execution.run_backend_in_subprocess(
            {
                "backend": "mlx",
                "base_model": "models/tiny",
                "output_dir": "jobs/rt3",
                "recipe": {"training": {"base_model": "models/tiny", "output_dir": "outputs/mlx"}},
                "training_examples": [{"instruction": "ping", "output": "pong"}],
            },
            backend="mlx",
            dry_run=False,
        )

        child_spec = captured["execution_plan"]["job_spec"]
        assert result["status"] == "completed"
        assert child_spec["base_model"] == str(local_model.resolve())
        assert child_spec["recipe"]["training"]["base_model"] == str(local_model.resolve())
        assert child_spec["output_dir"] == str((tmp_path / "jobs" / "rt3").resolve())
        assert child_spec["recipe"]["training"]["output_dir"] == str((tmp_path / "outputs" / "mlx").resolve())
        assert child_spec["_pfe_training_subprocess"] is True


class TestMLXPreflight:
    def test_mlx_local_model_blocks_when_available_memory_is_too_low(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Any,
    ) -> None:
        monkeypatch.setenv("PFE_REAL_TRAINING", "1")
        model_path = tmp_path / "local-model"
        model_path.mkdir()
        (model_path / "model.safetensors").write_bytes(b"fake weights")
        monkeypatch.setattr(TrainingPreflight, "_available_memory_gb", staticmethod(lambda: 1.0))
        monkeypatch.setattr(TrainingPreflight, "_local_model_weight_size_gb", staticmethod(lambda _: 7.5))

        result = TrainingPreflight(
            {
                "backend": "mlx",
                "base_model": str(model_path),
                "output_dir": str(tmp_path / "out"),
                "training_examples": [{"instruction": "ping", "output": "pong"}],
            }
        ).check()

        assert result["status"] == "blocked"
        assert "insufficient_memory" in result["reasons"]
        assert result["checks"]["memory"]["status"] == "blocked"
        assert result["checks"]["memory"]["estimated_required_gb"] == 11.25
