"""Unit tests for DPO executor implementation.

These tests verify the DPO training backend components without requiring
a full server or external dependencies.
"""

import pytest
from unittest.mock import patch, MagicMock

pytestmark = pytest.mark.filterwarnings(
    "ignore:`torch\\.jit\\.script_method` is deprecated.*:DeprecationWarning"
)


class TestDPOExecutorImports:
    """Test DPO executor import configuration."""

    def test_dpo_in_backend_imports(self):
        from pfe_core.trainer.executors import _BACKEND_IMPORTS
        assert "dpo" in _BACKEND_IMPORTS
        assert "trl" in _BACKEND_IMPORTS["dpo"]
        assert "torch" in _BACKEND_IMPORTS["dpo"]
        assert "transformers" in _BACKEND_IMPORTS["dpo"]
        assert "peft" in _BACKEND_IMPORTS["dpo"]
        assert "accelerate" in _BACKEND_IMPORTS["dpo"]

    def test_dpo_in_backend_required_attrs(self):
        from pfe_core.trainer.executors import _BACKEND_REQUIRED_ATTRS
        assert "dpo" in _BACKEND_REQUIRED_ATTRS
        attrs = _BACKEND_REQUIRED_ATTRS["dpo"]
        assert "trl" in attrs
        assert "DPOTrainer" in attrs["trl"]
        assert "peft" in attrs
        assert "PeftModel" in attrs["peft"]


class TestDPOExecutorFunction:
    """Test DPO executor function."""

    def test_execute_dpo_training_dry_run(self):
        from pfe_core.trainer.executors import execute_dpo_training

        job_spec = {
            "recipe": {
                "training": {"base_model": "gpt2", "epochs": 3},
                "peft": {"dpo_config": {"beta": 0.1, "label_smoothing": 0.0}},
            },
            "training_examples": [
                {"instruction": "Test", "chosen": "Good", "rejected": "Bad"}
            ],
        }

        result = execute_dpo_training(job_spec=job_spec, dry_run=True)

        assert result["backend"] == "dpo"
        assert result["status"] == "prepared"
        assert result["dry_run"] is True
        assert result["num_examples"] == 1
        assert result["dpo_config"]["beta"] == 0.1


class TestDPOTrainerExecutor:
    """Test DPOTrainerExecutor class."""

    def test_dpo_trainer_executor_exists(self):
        from pfe_core.trainer.dpo_executor import DPOTrainerExecutor, TrainingResult

        assert DPOTrainerExecutor is not None
        assert TrainingResult is not None

    def test_check_dpo_dependencies(self):
        from pfe_core.trainer.dpo_executor import check_dpo_dependencies

        deps = check_dpo_dependencies()
        assert "torch" in deps
        assert "transformers" in deps
        assert "trl" in deps
        assert "peft" in deps
        assert "all_available" in deps


class TestDPORuntimeJob:
    """Test DPO integration with runtime_job."""

    def test_dispatch_dpo_job(self):
        from pfe_core.trainer.runtime_job import dispatch_training_job

        job_spec = {
            "execution_executor": "dpo",
            "recipe": {
                "training": {"base_model": "gpt2", "epochs": 3},
                "peft": {"dpo_config": {"beta": 0.1}},
            },
            "training_examples": [],
        }

        result = dispatch_training_job(job_spec, dry_run=True)
        assert result["backend"] == "dpo"


class TestDPOServiceIntegration:
    """Test DPO integration with TrainerService."""

    def test_service_has_dpo_executor_imports(self):
        from pfe_core.trainer.service import TrainerService

        service = TrainerService()
        assert "dpo" in service.EXECUTOR_IMPORTS

    def test_service_has_execute_dpo_backend(self):
        from pfe_core.trainer.service import TrainerService

        service = TrainerService()
        assert hasattr(service, "_execute_dpo_backend")

    def test_execute_dpo_backend_returns_correct_structure(self):
        from pfe_core.trainer.service import TrainerService

        service = TrainerService()
        result = service._execute_dpo_backend(
            executor_spec={
                "executor_mode": "real_import",
                "executor_kind": "dpo_executor",
                "callable_name": "_execute_dpo_backend",
            },
            base_model_name="gpt2",
            train_type="dpo",
            method="qlora",
            epochs=3,
            train_samples=[{"prompt": "test", "chosen": "good", "rejected": "bad"}],
        )

        assert result["execution_backend"] == "dpo"
        assert result["train_type"] == "dpo"
        assert result["backend_label"].startswith("dpo:")


class TestDPOExecutionRecipe:
    """Test DPO training execution recipe building."""

    def test_build_training_execution_recipe_for_dpo(self):
        from pfe_core.trainer.executors import build_training_execution_recipe

        result = build_training_execution_recipe(
            backend_dispatch={
                "requested_backend": "dpo",
                "execution_backend": "dpo",
                "capability": {"artifact_format": "peft_lora"},
            },
            runtime={"runtime_device": "cpu"},
            method="qlora",
            epochs=3,
            train_type="dpo",
            base_model_name="gpt2",
            num_train_samples=10,
            num_fresh_samples=8,
            num_replay_samples=2,
            replay_ratio=0.2,
            train_examples=[
                {
                    "instruction": "What is AI?",
                    "chosen": "AI is artificial intelligence.",
                    "rejected": "AI is a robot.",
                    "sample_type": "dpo",
                }
            ],
        )

        assert result["requested_backend"] == "dpo"
        assert result["execution_backend"] == "dpo"
        assert result["execution_executor"] == "dpo"
        assert result["executor_mode"] == "real_import"

        backend_recipe = result["backend_recipe"]
        assert backend_recipe["backend"] == "dpo"
        assert backend_recipe["callable_name"] == "execute_dpo_training"
        assert backend_recipe["entrypoint"] == "pfe_core.trainer.executors:execute_dpo_training"

        peft_config = backend_recipe.get("peft", {})
        assert peft_config.get("trainer_class") == "trl.DPOTrainer"
        assert "dpo_config" in peft_config
        assert peft_config["dpo_config"]["beta"] == 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
