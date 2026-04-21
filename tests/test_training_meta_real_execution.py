from __future__ import annotations

import importlib
import json
import os
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

from pfe_core.pipeline import PipelineService
from pfe_core.trainer.service import TrainerService

trainer_service_module = importlib.import_module("pfe_core.trainer.service")
trainer_executor_module = importlib.import_module("pfe_core.trainer.executors")


class TrainingMetaRealExecutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.previous_home = os.environ.get("PFE_HOME")
        self.previous_export_tool = os.environ.get("PFE_LLAMA_CPP_EXPORT_TOOL")
        self.pfe_home = Path(self.tempdir.name) / ".pfe"
        os.environ["PFE_HOME"] = str(self.pfe_home)

    def tearDown(self) -> None:
        if self.previous_home is None:
            os.environ.pop("PFE_HOME", None)
        else:
            os.environ["PFE_HOME"] = self.previous_home
        if self.previous_export_tool is None:
            os.environ.pop("PFE_LLAMA_CPP_EXPORT_TOOL", None)
        else:
            os.environ["PFE_LLAMA_CPP_EXPORT_TOOL"] = self.previous_export_tool
        self.tempdir.cleanup()

    def test_training_meta_and_manifest_persist_job_and_execution_summaries(self) -> None:
        pipeline = PipelineService()
        pipeline.generate(scenario="life-coach", style="温和、共情", num_samples=8)

        runtime_snapshot = {
            "platform_name": "Linux",
            "machine": "x86_64",
            "processor": "x86_64",
            "python_version": "3.9.6",
            "apple_silicon": False,
            "cuda_available": False,
            "mps_available": False,
            "cpu_only": True,
            "runtime_device": "cpu",
            "installed_packages": {"torch": True, "transformers": True, "peft": True, "accelerate": True},
            "dependency_versions": {"torch": "2.5.0", "peft": "0.10.0"},
            "notes": ["synthetic runtime"],
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

        with tempfile.TemporaryDirectory() as tool_dir:
            tool_path = Path(tool_dir) / "fake-llama-export.sh"
            tool_path.write_text("#!/bin/sh\necho \"$@\"\nexit 0\n", encoding="utf-8")
            tool_path.chmod(0o755)

            with patch.dict(os.environ, {"PFE_LLAMA_CPP_EXPORT_TOOL": str(tool_path)}), patch.object(
                trainer_service_module,
                "detect_trainer_runtime",
            ) as detect_runtime, patch.object(
                trainer_service_module.importlib.util,
                "find_spec",
                side_effect=fake_find_spec,
            ), patch.object(
                trainer_service_module.importlib,
                "import_module",
                side_effect=fake_import_module,
            ), patch.object(
                trainer_executor_module.importlib.util,
                "find_spec",
                side_effect=fake_find_spec,
            ), patch.object(
                trainer_executor_module.importlib,
                "import_module",
                side_effect=fake_import_module,
            ):
                detect_runtime.return_value = SimpleNamespace(to_dict=lambda: dict(runtime_snapshot))
                result = pipeline.trainer.train_result(
                    method="qlora",
                    epochs=1,
                    base_model="mock-llama-target",
                    train_type="sft",
                    backend_hint="peft",
                )

        version_dir = Path(result.adapter_path)
        training_meta = json.loads((version_dir / "training_meta.json").read_text(encoding="utf-8"))
        manifest = json.loads((version_dir / "adapter_manifest.json").read_text(encoding="utf-8"))

        self.assertEqual(result.execution_backend, "peft")
        self.assertEqual(result.execution_executor, "peft")
        self.assertEqual(result.training_config["job_bundle"]["command"][0], sys.executable)
        self.assertTrue(result.training_config["job_bundle"]["command"][1].endswith("trainer_job.py"))
        self.assertTrue(result.training_config["job_bundle"]["job_json_path"].endswith("trainer_job.json"))
        self.assertTrue(result.training_config["job_bundle"]["script_path"].endswith("trainer_job.py"))
        self.assertTrue(result.training_config["job_bundle"]["job_json"]["job_spec"]["ready"])
        self.assertEqual(result.training_config["job_bundle"]["job_json"]["execution_executor"], "peft")
        self.assertEqual(result.training_config["job_bundle"]["audit"]["execution_executor"], "peft")
        self.assertFalse(result.training_config["job_bundle"]["dry_run"])

        self.assertIn("job_bundle", training_meta)
        self.assertIn("job_spec", training_meta)
        self.assertIn("execution_recipe", training_meta)
        self.assertIn("export_execution", training_meta)
        self.assertEqual(training_meta["job_bundle"]["execution_executor"], "peft")
        self.assertTrue(training_meta["job_bundle"]["job_json"]["job_spec"]["ready"])
        self.assertEqual(training_meta["job_bundle"]["job_json"]["execution_executor"], "peft")
        self.assertEqual(training_meta["job_bundle"]["job_json"]["backend_recipe"]["backend"], "peft")
        self.assertEqual(training_meta["job_bundle"]["job_json"]["executor_recipe"]["backend"], "peft")
        self.assertIn(training_meta["export_execution"]["audit"]["status"], {"success", "executed", "artifact_missing"})
        self.assertEqual(training_meta["export_write"]["metadata"]["execution_status"], training_meta["export_execution"]["audit"]["status"])

        self.assertIn("metadata", manifest)
        self.assertEqual(manifest["metadata"]["training"]["backend"], "peft")
        self.assertEqual(manifest["metadata"]["backend_plan"]["recommended_backend"], "peft")
        self.assertEqual(manifest["artifact_format"], training_meta["export_runtime"]["target_artifact_format"])
        self.assertEqual(manifest["adapter_dir"], str(version_dir))


if __name__ == "__main__":
    unittest.main()
