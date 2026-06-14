from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any

import pytest

from pfe_core.trainer import executors, real_execution
from pfe_core.trainer.backends import (
    backend_executor_imports,
    get_backend_capability,
    real_training_backend_names,
    subprocess_isolated_backend_names,
)
from pfe_core.trainer.preflight import TrainingPreflight
from pfe_core.trainer.runtime_job import dispatch_training_job
from pfe_core.trainer.service import TrainerService

trainer_service_module = importlib.import_module("pfe_core.trainer.service")


def test_backend_import_contract_is_shared_by_service_and_executor() -> None:
    expected = backend_executor_imports()

    assert TrainerService.EXECUTOR_IMPORTS == expected
    assert executors._BACKEND_IMPORTS == expected
    assert "accelerate" in expected["peft"]
    assert "trl" in expected["dpo"]


def test_real_execution_backend_sets_come_from_backend_contract() -> None:
    assert real_execution.REAL_TRAINING_BACKENDS == real_training_backend_names()
    assert real_execution.SUBPROCESS_ISOLATED_BACKENDS == subprocess_isolated_backend_names()
    assert real_execution.REAL_TRAINING_BACKENDS == frozenset({"peft", "dpo", "unsloth", "mlx"})
    assert "mock_local" not in real_execution.REAL_TRAINING_BACKENDS


def test_runtime_job_normalizes_backend_alias_before_real_execution_gate() -> None:
    result = dispatch_training_job({"backend": "mlx-lm"}, dry_run=False)

    assert result["status"] == "blocked"
    assert result["backend"] == "mlx"
    assert "Real training execution disabled" in result["reason"]


def test_preflight_dependency_check_uses_backend_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    monkeypatch.setenv("PFE_REAL_TRAINING", "1")

    def fake_find_spec(name: str) -> object | None:
        if name == "accelerate":
            return None
        return SimpleNamespace(name=name)

    monkeypatch.setattr("pfe_core.trainer.preflight.importlib.util.find_spec", fake_find_spec)

    result = TrainingPreflight(
        {
            "backend": "peft",
            "base_model": "local-model",
            "output_dir": str(tmp_path / "out"),
            "training_examples": [{"instruction": "ping", "output": "pong"}],
        }
    ).check()

    dependency_check = result["checks"]["dependencies"]["peft_import"]
    assert result["status"] == "blocked"
    assert "peft_import_failed" in result["reasons"]
    assert dependency_check["source"] == "backend_capability"
    assert dependency_check["required_modules"] == ["torch", "transformers", "peft", "accelerate"]
    assert dependency_check["missing_modules"] == ["accelerate"]


def test_mlx_preflight_dependencies_are_deferred_to_isolated_subprocess(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    monkeypatch.setenv("PFE_REAL_TRAINING", "1")
    monkeypatch.setattr("pfe_core.trainer.preflight.importlib.util.find_spec", lambda name: None)
    monkeypatch.setattr(TrainingPreflight, "_available_memory_gb", staticmethod(lambda: 3.0))
    monkeypatch.setattr(TrainingPreflight, "_local_model_weight_size_gb", staticmethod(lambda _: 0.25))

    result = TrainingPreflight(
        {
            "backend": "mlx",
            "base_model": "local-model",
            "output_dir": str(tmp_path / "out"),
            "training_examples": [{"instruction": "ping", "output": "pong"}],
        }
    ).check()

    dependency_check = result["checks"]["dependencies"]["mlx_import"]
    assert result["status"] == "ok"
    assert "mlx_import_failed" not in result["reasons"]
    assert dependency_check["ok"] is False
    assert dependency_check["status"] == "warning"
    assert dependency_check["missing_modules"] == ["mlx", "mlx_lm"]


def test_service_dispatch_uses_registry_availability_for_fallback() -> None:
    dispatch = TrainerService()._dispatch_training_backend(
        backend_plan={
            "train_type": "sft",
            "recommended_backend": "peft",
            "requires_export_step": False,
            "export_steps": [],
        },
        runtime={
            "runtime_device": "cpu",
            "installed_packages": {
                "torch": True,
                "transformers": True,
                "peft": True,
                "accelerate": False,
            },
        },
        backend_hint="peft",
    )

    assert dispatch["requested_backend"] == "peft"
    assert dispatch["execution_backend"] == "mock_local"
    assert dispatch["dispatch_mode"] == "fallback"
    assert dispatch["available"]["peft"] is False
    assert "missing dependencies: accelerate" in dispatch["reasons"]
    assert dispatch["capability"] == get_backend_capability("mock_local").to_dict()


def test_service_dispatch_uses_registry_capability_for_dpo_reroute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_modules = {
        "torch": SimpleNamespace(nn=object()),
        "transformers": SimpleNamespace(AutoModelForCausalLM=object(), AutoTokenizer=object(), TrainingArguments=object()),
        "peft": SimpleNamespace(LoraConfig=object(), get_peft_model=object(), PeftModel=object()),
        "trl": SimpleNamespace(DPOTrainer=object()),
        "accelerate": SimpleNamespace(Accelerator=object()),
    }

    monkeypatch.setattr(trainer_service_module.importlib.util, "find_spec", lambda name: object() if name in fake_modules else None)
    monkeypatch.setattr(trainer_service_module.importlib, "import_module", lambda name: fake_modules[name])

    dispatch = TrainerService()._dispatch_training_backend(
        backend_plan={
            "train_type": "dpo",
            "recommended_backend": "peft",
            "requires_export_step": True,
            "export_steps": ["gguf_merged_export"],
            "export_format": "gguf_merged",
            "export_backend": "llama_cpp",
        },
        runtime={
            "runtime_device": "cpu",
            "installed_packages": {
                "torch": True,
                "transformers": True,
                "peft": True,
                "trl": True,
                "accelerate": True,
            },
        },
        backend_hint="peft",
    )

    assert dispatch["requested_backend"] == "peft"
    assert dispatch["execution_backend"] == "dpo"
    assert dispatch["dispatch_mode"] == "dpo_reroute"
    assert dispatch["executor_mode"] == "real_import"
    assert dispatch["capability"] == get_backend_capability("dpo").to_dict()
    assert dispatch["requires_export_step"] is True
    assert dispatch["export_backend"] == "llama_cpp"


def test_service_dispatch_normalizes_known_backend_alias() -> None:
    dispatch = TrainerService()._dispatch_training_backend(
        backend_plan={
            "train_type": "sft",
            "recommended_backend": "mock_local",
            "requires_export_step": False,
            "export_steps": [],
        },
        runtime={"runtime_device": "cpu", "installed_packages": {}},
        backend_hint="local-mock",
    )

    assert dispatch["requested_backend"] == "mock_local"
    assert dispatch["execution_backend"] == "mock_local"
    assert dispatch["dispatch_mode"] == "requested"
