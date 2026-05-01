"""Tests for PFE_REAL_TRAINING execution gate (Phase 6)."""

from __future__ import annotations

import os
from typing import Any

import pytest
from pfe_core.trainer import real_execution, runtime_job
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
