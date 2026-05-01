from __future__ import annotations

import importlib
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
for package_dir in ("pfe-core", "pfe-cli", "pfe-server"):
    package_path = str(ROOT / package_dir)
    if package_path not in os.sys.path:
        os.sys.path.insert(0, package_path)

trainer_executor_module = importlib.import_module("pfe_core.trainer.executors")


class TrainerExecutorRecipeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.previous_home = os.environ.get("PFE_HOME")
        self.pfe_home = Path(self.tempdir.name) / ".pfe"
        os.environ["PFE_HOME"] = str(self.pfe_home)

    def tearDown(self) -> None:
        if self.previous_home is None:
            os.environ.pop("PFE_HOME", None)
        else:
            os.environ["PFE_HOME"] = self.previous_home
        self.tempdir.cleanup()

    def test_peft_execution_recipe_uses_real_import_shape_when_dependencies_exist(self) -> None:
        fake_modules = {
            "torch": SimpleNamespace(nn=object()),
            "transformers": SimpleNamespace(Trainer=object(), TrainingArguments=object()),
            "peft": SimpleNamespace(LoraConfig=object(), get_peft_model=lambda *args, **kwargs: None),
            "accelerate": SimpleNamespace(Accelerator=object()),
        }

        def fake_find_spec(name: str):
            return object() if name in fake_modules else None

        def fake_import_module(name: str):
            return fake_modules[name]

        backend_dispatch = {
            "requested_backend": "peft",
            "execution_backend": "peft",
            "execution_executor": "peft",
            "executor_mode": "real_import",
            "reasons": ["using requested backend peft"],
            "requires_export_step": True,
            "export_steps": ["gguf_merged_export"],
            "export_format": "gguf_merged",
            "export_backend": "llama_cpp",
            "capability": {"artifact_format": "peft_lora"},
        }

        with patch.object(trainer_executor_module.importlib.util, "find_spec", side_effect=fake_find_spec), patch.object(
            trainer_executor_module.importlib,
            "import_module",
            side_effect=fake_import_module,
        ):
            plan = trainer_executor_module.build_training_execution_recipe(
                backend_dispatch=backend_dispatch,
                runtime={"runtime_device": "cpu"},
                method="qlora",
                epochs=2,
                train_type="sft",
                base_model_name="mock-llama-target",
                num_train_samples=8,
                num_fresh_samples=6,
                num_replay_samples=2,
                replay_ratio=0.25,
                allow_mock_fallback=True,
            )

        self.assertEqual(plan["execution_backend"], "peft")
        self.assertEqual(plan["execution_executor"], "peft")
        self.assertEqual(plan["executor_mode"], "real_import")
        self.assertEqual(plan["backend_recipe"]["backend"], "peft")
        self.assertEqual(plan["backend_recipe"]["peft"]["peft_config_class"], "peft.LoraConfig")
        self.assertEqual(plan["backend_recipe"]["peft"]["trainer_class"], "transformers.Trainer")
        self.assertEqual(plan["job_spec"]["entrypoint"], "pfe_core.trainer.executors:execute_peft_training")
        self.assertTrue(plan["job_spec"]["ready"])
        self.assertIn("peft", plan["job_spec"]["recipe"]["backend"])
        self.assertEqual(plan["job_spec"]["recipe"]["peft"]["lora_config"]["r"], 16)

    def test_peft_execution_recipe_falls_back_to_mock_local_without_imports(self) -> None:
        backend_dispatch = {
            "requested_backend": "peft",
            "execution_backend": "peft",
            "execution_executor": "peft",
            "executor_mode": "requested",
            "reasons": ["using requested backend peft"],
            "requires_export_step": True,
            "export_steps": ["gguf_merged_export"],
            "export_format": "gguf_merged",
            "export_backend": "llama_cpp",
            "capability": {"artifact_format": "peft_lora"},
        }

        with patch.object(trainer_executor_module.importlib.util, "find_spec", return_value=None):
            plan = trainer_executor_module.build_training_execution_recipe(
                backend_dispatch=backend_dispatch,
                runtime={"runtime_device": "cpu"},
                method="qlora",
                epochs=1,
                train_type="sft",
                base_model_name="mock-llama-target",
                num_train_samples=4,
                num_fresh_samples=3,
                num_replay_samples=1,
                replay_ratio=0.25,
                allow_mock_fallback=True,
            )

        self.assertEqual(plan["execution_backend"], "peft")
        self.assertEqual(plan["execution_executor"], "mock_local")
        self.assertEqual(plan["executor_mode"], "fallback")
        self.assertEqual(plan["fallback_from"], "peft")
        self.assertEqual(plan["backend_recipe"]["backend"], "peft")
        self.assertEqual(plan["executor_recipe"]["backend"], "mock_local")
        self.assertEqual(plan["job_spec"]["execution_executor"], "mock_local")
        self.assertFalse(plan["job_spec"]["ready"])

    def test_materialized_training_job_bundle_writes_command_and_script(self) -> None:
        backend_dispatch = {
            "requested_backend": "peft",
            "execution_backend": "peft",
            "execution_executor": "peft",
            "executor_mode": "real_import",
            "reasons": ["using requested backend peft"],
            "requires_export_step": True,
            "export_steps": ["gguf_merged_export"],
            "export_format": "gguf_merged",
            "export_backend": "llama_cpp",
            "capability": {"artifact_format": "peft_lora"},
        }

        fake_modules = {
            "torch": SimpleNamespace(nn=object()),
            "transformers": SimpleNamespace(Trainer=object(), TrainingArguments=object()),
            "peft": SimpleNamespace(LoraConfig=object(), get_peft_model=lambda *args, **kwargs: None),
            "accelerate": SimpleNamespace(Accelerator=object()),
        }

        def fake_find_spec(name: str):
            return object() if name in fake_modules else None

        def fake_import_module(name: str):
            return fake_modules[name]

        with patch.object(trainer_executor_module.importlib.util, "find_spec", side_effect=fake_find_spec), patch.object(
            trainer_executor_module.importlib,
            "import_module",
            side_effect=fake_import_module,
        ):
            plan = trainer_executor_module.build_training_execution_recipe(
                backend_dispatch=backend_dispatch,
                runtime={"runtime_device": "cpu"},
                method="qlora",
                epochs=2,
                train_type="sft",
                base_model_name="mock-llama-target",
                num_train_samples=8,
                num_fresh_samples=6,
                num_replay_samples=2,
                replay_ratio=0.25,
                allow_mock_fallback=True,
            )

        bundle = trainer_executor_module.materialize_training_job_bundle(
            execution_plan=plan,
            output_dir=self.pfe_home / "adapters" / "user_default" / "20260323-997",
        )

        self.assertEqual(bundle.command[0], sys.executable)
        self.assertTrue(bundle.command[1].endswith("trainer_job.py"))
        self.assertTrue(bundle.command[3].endswith("trainer_job.json"))
        self.assertTrue(bundle.command[5].endswith("training_job_result.json"))
        self.assertFalse(bundle.dry_run)
        self.assertTrue(Path(bundle.script_path).exists())
        self.assertTrue(Path(bundle.job_json_path).exists())
        self.assertTrue(Path(bundle.result_json_path).name == "training_job_result.json")
        self.assertIn("run_training_job_file", bundle.script_text)
        self.assertEqual(bundle.job_json["execution_executor"], "peft")
        self.assertEqual(bundle.audit["execution_executor"], "peft")
        with patch.dict(os.environ, {"PFE_REAL_TRAINING": "1"}):
            executed = trainer_executor_module.run_materialized_training_job_bundle(bundle, force_dry_run=False)
        self.assertTrue(executed.attempted)
        self.assertTrue(executed.success)
        self.assertEqual(executed.status, "executed")
        self.assertEqual(executed.runner_result["backend"], "peft")
        self.assertEqual(executed.runner_result["status"], "ready")
        self.assertTrue(Path(bundle.result_json_path).exists())

    def test_materialized_training_job_bundle_falls_back_to_mock_local_command(self) -> None:
        plan = {
            "requested_backend": "peft",
            "execution_backend": "peft",
            "execution_executor": "mock_local",
            "executor_mode": "fallback",
            "ready": False,
            "fallback_from": "peft",
            "import_probe": {"ready": False},
            "backend_recipe": {"backend": "peft"},
            "executor_recipe": {"backend": "mock_local"},
            "job_spec": {"execution_executor": "mock_local"},
            "executor_kind": "mock_local_executor",
            "callable_name": "execute_mock_local_training",
            "requires_export_step": True,
            "export_steps": ["gguf_merged_export"],
            "export_format": "gguf_merged",
            "export_backend": "llama_cpp",
            "reasons": ["fallback"],
        }

        bundle = trainer_executor_module.materialize_training_job_bundle(
            execution_plan=plan,
            output_dir=self.pfe_home / "adapters" / "user_default" / "20260323-996",
        )

        self.assertEqual(bundle.execution_executor, "mock_local")
        self.assertTrue(bundle.dry_run)
        self.assertIn("--dry-run", bundle.command)
        self.assertIn("run_training_job_file", bundle.script_text)
        executed = trainer_executor_module.run_materialized_training_job_bundle(bundle)
        self.assertFalse(executed.attempted)
        self.assertTrue(executed.success)
        self.assertEqual(executed.status, "planned")

    def test_materialized_real_local_job_enables_real_training_in_child_env(self) -> None:
        plan = {
            "requested_backend": "peft",
            "execution_backend": "peft",
            "execution_executor": "peft",
            "executor_mode": "real_import",
            "ready": True,
            "fallback_from": None,
            "import_probe": {"ready": True},
            "backend_recipe": {"backend": "peft"},
            "executor_recipe": {"backend": "peft"},
            "job_spec": {
                "execution_executor": "peft",
                "real_local": True,
                "real_training_enabled": True,
            },
            "executor_kind": "peft_executor",
            "callable_name": "execute_peft_training",
            "requires_export_step": False,
            "export_steps": [],
            "export_format": None,
            "export_backend": None,
            "reasons": [],
        }
        bundle = trainer_executor_module.materialize_training_job_bundle(
            execution_plan=plan,
            output_dir=self.pfe_home / "adapters" / "user_default" / "20260323-real",
        )
        observed_env: dict[str, str] = {}

        def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            observed_env.update(dict(kwargs.get("env") or {}))  # type: ignore[arg-type]
            return subprocess.CompletedProcess(
                command,
                returncode=0,
                stdout='{"backend":"peft","status":"ready"}\n',
                stderr="",
            )

        with patch.object(trainer_executor_module.subprocess, "run", side_effect=fake_run):
            result = trainer_executor_module.run_materialized_training_job_bundle(bundle, force_dry_run=False)

        self.assertTrue(result.success)
        self.assertEqual(observed_env.get("PFE_TRAINING_SUBPROCESS"), "1")
        self.assertEqual(observed_env.get("PFE_REAL_TRAINING"), "1")

    def test_materialized_training_timeout_persists_text_diagnostics(self) -> None:
        plan = {
            "requested_backend": "peft",
            "execution_backend": "peft",
            "execution_executor": "peft",
            "executor_mode": "real_import",
            "ready": True,
            "fallback_from": None,
            "import_probe": {"ready": True},
            "backend_recipe": {"backend": "peft"},
            "executor_recipe": {"backend": "peft"},
            "job_spec": {"execution_executor": "peft"},
            "executor_kind": "peft_executor",
            "callable_name": "execute_peft_training",
            "requires_export_step": False,
            "export_steps": [],
            "export_format": None,
            "export_backend": None,
            "reasons": [],
        }
        bundle = trainer_executor_module.materialize_training_job_bundle(
            execution_plan=plan,
            output_dir=self.pfe_home / "adapters" / "user_default" / "20260323-timeout",
        )

        def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            raise subprocess.TimeoutExpired(
                command,
                timeout=3,
                output=b"partial stdout",
                stderr=b"partial stderr",
            )

        with patch.object(trainer_executor_module.subprocess, "run", side_effect=fake_run):
            result = trainer_executor_module.run_materialized_training_job_bundle(
                bundle,
                force_dry_run=False,
                timeout_seconds=3,
            )

        self.assertFalse(result.success)
        self.assertEqual(result.returncode, 124)
        self.assertEqual(result.failure_category, "timeout")
        self.assertIn("partial stdout", result.stdout)
        self.assertIn("partial stderr", result.stderr)
        self.assertIn("timed out after 3 seconds", result.stderr)
        self.assertTrue(Path(result.stdout_log or "").read_text(encoding="utf-8").startswith("partial stdout"))
        self.assertIn(
            "timed out after 3 seconds",
            Path(result.stderr_log or "").read_text(encoding="utf-8"),
        )

    def test_training_job_spec_materializes_runner_and_export_audit(self) -> None:
        fake_modules = {
            "torch": SimpleNamespace(nn=object()),
            "transformers": SimpleNamespace(Trainer=object(), TrainingArguments=object()),
            "peft": SimpleNamespace(LoraConfig=object(), get_peft_model=lambda *args, **kwargs: None),
            "accelerate": SimpleNamespace(Accelerator=object()),
        }

        def fake_find_spec(name: str):
            return object() if name in fake_modules else None

        def fake_import_module(name: str):
            return fake_modules[name]

        backend_dispatch = {
            "requested_backend": "peft",
            "execution_backend": "peft",
            "execution_executor": "peft",
            "executor_mode": "real_import",
            "reasons": ["using requested backend peft"],
            "requires_export_step": True,
            "export_steps": ["gguf_merged_export"],
            "export_format": "gguf_merged",
            "export_backend": "llama_cpp",
            "capability": {"artifact_format": "peft_lora"},
        }

        with patch.object(trainer_executor_module.importlib.util, "find_spec", side_effect=fake_find_spec), patch.object(
            trainer_executor_module.importlib,
            "import_module",
            side_effect=fake_import_module,
        ):
            plan = trainer_executor_module.build_training_execution_recipe(
                backend_dispatch=backend_dispatch,
                runtime={"runtime_device": "cpu"},
                method="qlora",
                epochs=2,
                train_type="sft",
                base_model_name="mock-llama-target",
                num_train_samples=8,
                num_fresh_samples=6,
                num_replay_samples=2,
                replay_ratio=0.25,
                allow_mock_fallback=True,
            )

        self.assertTrue(plan["job_spec"]["ready"])
        self.assertTrue(plan["job_spec"]["dry_run"])
        self.assertEqual(plan["job_spec"]["execution_backend"], "peft")
        self.assertEqual(plan["job_spec"]["execution_executor"], "peft")
        self.assertEqual(plan["job_spec"]["audit"]["backend_recipe"]["export"]["required"], True)
        self.assertEqual(plan["job_spec"]["recipe"]["backend"], "peft")
        self.assertEqual(plan["job_spec"]["recipe"]["export"]["format"], "gguf_merged")
        self.assertEqual(plan["backend_recipe"]["training"]["method"], "qlora")
        self.assertEqual(plan["backend_recipe"]["adapter"]["requires_export_step"], True)
        self.assertEqual(plan["executor_recipe"]["audit"]["execution_executor"], "peft")

        executed = trainer_executor_module.execute_peft_training(job_spec=plan["job_spec"], dry_run=True)
        self.assertEqual(executed["status"], "prepared")
        self.assertEqual(executed["execution_mode"], "real_import")
        self.assertEqual(executed["job_spec"]["execution_executor"], "peft")
        self.assertEqual(executed["recipe"]["export"]["format"], "gguf_merged")

    def test_execute_peft_training_reports_completed_when_real_import_peft_execution_succeeds(self) -> None:
        job_spec = {
            "backend": "peft",
            "execution_backend": "peft",
            "execution_executor": "peft",
            "ready": True,
            "recipe": {"training": {"method": "qlora"}},
            "training_examples": [{"sample_id": "s1", "instruction": "hello", "chosen": "world"}],
            "audit": {"import_probe": {"ready": True}},
        }

        with patch.object(trainer_executor_module, "_probe_real_peft_runtime", return_value={"available": True, "missing_modules": []}), patch.object(
            trainer_executor_module,
            "_run_real_import_peft_training",
            return_value={
                "backend": "peft",
                "dry_run": False,
                "execution_mode": "real_import",
                "job_spec": job_spec,
                "recipe": {"training": {"method": "qlora"}},
                "status": "completed",
                "real_execution": {"kind": "real_peft", "path": "real_import", "num_examples": 1},
            },
        ):
            result = trainer_executor_module.execute_peft_training(job_spec=job_spec, dry_run=False)

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["real_execution"]["kind"], "real_peft")


if __name__ == "__main__":
    unittest.main()
