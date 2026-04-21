from __future__ import annotations

import importlib
import json
import os
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

from pfe_core.adapter_store.store import AdapterStore
from pfe_core.pipeline import PipelineService

trainer_service_module = importlib.import_module("pfe_core.trainer.service")
trainer_executor_module = importlib.import_module("pfe_core.trainer.executors")


class AdapterArtifactConsistencyTests(unittest.TestCase):
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

    def test_train_artifact_bundle_manifest_and_training_meta_remain_path_consistent(self) -> None:
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

        artifact_tmp = tempfile.TemporaryDirectory()
        try:
            artifact_root = Path(artifact_tmp.name) / "peft_lora"
            artifact_root.mkdir(parents=True, exist_ok=True)
            adapter_model = artifact_root / "adapter_model.safetensors"
            adapter_config = artifact_root / "adapter_config.json"
            manifest_file = artifact_root / "real_peft_job_manifest.json"
            summary_file = artifact_root / "training_summary.json"
            real_execution_file = artifact_root / "real_execution.json"
            trainer_state_file = artifact_root / "trainer_state.json"
            metrics_file = artifact_root / "train_metrics.json"

            adapter_model.write_text("{\"artifact\": \"adapter_model\"}\n", encoding="utf-8")
            adapter_config.write_text("{\"artifact\": \"adapter_config\"}\n", encoding="utf-8")
            manifest_file.write_text("{\"artifact\": \"manifest\"}\n", encoding="utf-8")
            summary_file.write_text("{\"artifact\": \"summary\"}\n", encoding="utf-8")
            real_execution_file.write_text("{\"artifact\": \"real_execution\"}\n", encoding="utf-8")
            trainer_state_file.write_text("{\"artifact\": \"trainer_state\"}\n", encoding="utf-8")
            metrics_file.write_text("{\"artifact\": \"metrics\"}\n", encoding="utf-8")

            job_execution = {
                "attempted": True,
                "success": True,
                "status": "executed",
                "returncode": 0,
                "exit_code": 0,
                "stdout": "",
                "stderr": "",
                "command": [os.sys.executable, "trainer_job.py"],
                "runner_result": {
                    "backend": "peft",
                    "status": "ready",
                    "execution_mode": "real_import",
                    "real_execution": {
                        "kind": "real_peft",
                        "artifact_dir": str(artifact_root),
                        "output_dir": str(artifact_root),
                        "artifacts": {
                            "adapter_model": str(adapter_model),
                            "adapter_config": str(adapter_config),
                            "manifest": str(manifest_file),
                            "training_summary": str(summary_file),
                            "real_execution": str(real_execution_file),
                            "trainer_state": str(trainer_state_file),
                            "metrics": str(metrics_file),
                        },
                        "metrics": {"loss": 0.125, "num_examples": 1},
                        "artifact_manifest_path": str(manifest_file),
                        "summary_path": str(summary_file),
                        "real_execution_path": str(real_execution_file),
                        "trainer_state_path": str(trainer_state_file),
                        "success": True,
                        "message": "real peft execution completed",
                    },
                },
                "metadata": {
                    "execution_state": "executed",
                    "executor_mode": "real_import",
                },
                "audit": {
                    "status": "executed",
                    "runner_status": "ready",
                    "dry_run": False,
                    "result_json_path": str(artifact_root / "training_job_result.json"),
                },
            }

            with patch.object(trainer_service_module, "detect_trainer_runtime") as detect_runtime, patch.object(
                trainer_service_module.importlib.util,
                "find_spec",
                side_effect=fake_find_spec,
            ), patch.object(
                trainer_service_module.importlib,
                "import_module",
                side_effect=fake_import_module,
            ), patch.object(
                trainer_service_module,
                "run_materialized_training_job_bundle",
                return_value=SimpleNamespace(to_dict=lambda: dict(job_execution)),
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
                    base_model="mock-base",
                    train_type="sft",
                    backend_hint="peft",
                )
            version_dir = Path(result.adapter_path)
            target_model = version_dir / "adapter_model.safetensors"
            target_config = version_dir / "adapter_config.json"
            training_meta = json.loads((version_dir / "training_meta.json").read_text(encoding="utf-8"))
            manifest = json.loads((version_dir / "adapter_manifest.json").read_text(encoding="utf-8"))
            store = AdapterStore(home=self.pfe_home)
            row = store.list_version_records(limit=1)[0]

            self.assertEqual(result.execution_backend, "peft")
            self.assertEqual(result.execution_executor, "peft")
            self.assertEqual(result.metrics["real_execution_artifacts"]["artifact_dir"], str(artifact_root))
            self.assertEqual(result.metrics["artifact_sync"]["artifact_dir"], str(artifact_root))
            self.assertTrue(result.metrics["artifact_sync"]["available"])
            self.assertEqual(result.metrics["artifact_sync"]["synced_files"]["adapter_model"]["target"], str(target_model))
            self.assertEqual(result.metrics["artifact_sync"]["synced_files"]["adapter_config"]["target"], str(target_config))
            self.assertTrue(target_model.exists())
            self.assertTrue(target_config.exists())
            self.assertEqual(target_model.read_text(encoding="utf-8"), adapter_model.read_text(encoding="utf-8"))
            self.assertEqual(target_config.read_text(encoding="utf-8"), adapter_config.read_text(encoding="utf-8"))

            self.assertEqual(training_meta["real_execution_artifacts"]["artifact_dir"], str(artifact_root))
            self.assertEqual(training_meta["real_execution_artifacts"]["artifacts"]["adapter_model"], str(adapter_model))
            self.assertEqual(training_meta["artifact_sync"]["artifact_dir"], str(artifact_root))
            self.assertEqual(training_meta["artifact_sync"]["synced_files"]["adapter_model"]["source"], str(adapter_model))
            self.assertEqual(training_meta["artifact_sync"]["synced_files"]["adapter_model"]["target"], str(target_model))

            self.assertEqual(manifest["artifact_format"], "peft_lora")
            self.assertEqual(manifest["artifact_name"], "adapter_model.safetensors")
            self.assertEqual(manifest["metadata"]["training"]["backend"], "peft")
            self.assertEqual(manifest["metadata"]["backend_plan"]["recommended_backend"], "peft")
            self.assertEqual(manifest["metadata"]["runtime"]["runtime_device"], "cpu")

            self.assertEqual(row["artifact_format"], "peft_lora")
            self.assertEqual(row["artifact_path"], str(target_model))
            self.assertEqual(row["adapter_dir"], str(version_dir))
            self.assertEqual(row["manifest_path"], str(version_dir / "adapter_manifest.json"))
        finally:
            artifact_tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
