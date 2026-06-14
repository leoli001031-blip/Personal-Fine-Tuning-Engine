from __future__ import annotations

import os
from typing import Any

from typer.testing import CliRunner

from pfe_cli import main as cli_main  # noqa: E402

def _patch_cli_training_surfaces(monkeypatch: Any, service: Any) -> None:
    monkeypatch.setattr(cli_main, "_load_service", lambda *module_names: service)
    monkeypatch.setattr(cli_main, "_format_train_preview", lambda **kwargs: "preview")
    monkeypatch.setattr(cli_main, "_format_train_result", lambda result, *, workspace=None: "result")
    monkeypatch.setattr(cli_main, "_record_train_cli_state", lambda result, workspace=None: None)

def test_train_dry_run_backend_and_real_local_are_passed(monkeypatch: Any) -> None:
    calls: list[dict[str, Any]] = []

    class FakeService:
        def train_result(self, **kwargs: Any) -> dict[str, Any]:
            calls.append({"kwargs": dict(kwargs), "env": os.environ.get("PFE_REAL_TRAINING")})
            return {"version": "v-test"}

    monkeypatch.delenv("PFE_REAL_TRAINING", raising=False)
    _patch_cli_training_surfaces(monkeypatch, FakeService())

    result = CliRunner().invoke(cli_main.app, ["train", "--backend", "mlx", "--dry-run", "--real-local"])

    assert result.exit_code == 0, result.stdout
    assert calls[0]["kwargs"]["backend"] == "mlx"
    assert calls[0]["kwargs"]["dry_run"] is True
    assert calls[0]["kwargs"]["real_local"] is True
    assert calls[0]["env"] == "1"
    assert os.environ.get("PFE_REAL_TRAINING") is None

def test_train_preview_real_local_exits_before_handler(monkeypatch: Any) -> None:
    calls: list[dict[str, Any]] = []

    class FakeService:
        def train_result(self, **kwargs: Any) -> dict[str, Any]:
            calls.append(dict(kwargs))
            return {"version": "v-test"}

    monkeypatch.delenv("PFE_REAL_TRAINING", raising=False)
    _patch_cli_training_surfaces(monkeypatch, FakeService())

    result = CliRunner().invoke(cli_main.app, ["train", "--backend", "peft", "--real-local", "--preview"])

    assert result.exit_code == 0, result.stdout
    assert "preview" in result.stdout
    assert calls == []
    assert os.environ.get("PFE_REAL_TRAINING") is None

def test_dpo_dry_run_routes_to_training_handler(monkeypatch: Any) -> None:
    calls: list[dict[str, Any]] = []

    class FakeService:
        def train_dpo(self, **kwargs: Any) -> dict[str, Any]:
            calls.append(dict(kwargs))
            return {"version": "dpo-test"}

    _patch_cli_training_surfaces(monkeypatch, FakeService())

    result = CliRunner().invoke(cli_main.app, ["dpo", "--dry-run", "--backend", "dpo"])

    assert result.exit_code == 0, result.stdout
    assert len(calls) == 1
    assert calls[0]["backend"] == "dpo"
    assert calls[0]["dry_run"] is True
    assert calls[0]["real_local"] is False

def test_train_rejects_dpo_backend_for_sft() -> None:
    result = CliRunner().invoke(cli_main.app, ["train", "--backend", "dpo"])

    assert result.exit_code == 1
    assert "--backend dpo is only valid" in (result.stdout + result.stderr)
