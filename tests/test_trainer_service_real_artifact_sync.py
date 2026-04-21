from __future__ import annotations

import json
import os
import tempfile
import unittest
import importlib
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
from pfe_core.trainer.service import TrainerService

import pytest

trainer_service_module = importlib.import_module("pfe_core.trainer.service")


@pytest.mark.slow
class TrainerServiceRealArtifactSyncTests(unittest.TestCase):
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

    def test_real_execution_artifacts_are_synced_into_standard_adapter_dir(self) -> None:
        pipeline = PipelineService()
        pipeline.generate(scenario="life-coach", style="温和", num_samples=8)
        trainer = TrainerService()

        def fake_run(bundle, *args, **kwargs):
            materialization = bundle.to_dict() if hasattr(bundle, "to_dict") else dict(bundle)
            version_dir = Path(materialization["job_json_path"]).parent
            artifact_dir = version_dir / "trainer_job_outputs" / "peft_lora"
            artifact_dir.mkdir(parents=True, exist_ok=True)
            (artifact_dir / "adapter_model.safetensors").write_text("real-peft-adapter\n", encoding="utf-8")
            (artifact_dir / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA"}) + "\n", encoding="utf-8")
            payload = {
                "attempted": True,
                "success": True,
                "status": "executed",
                "command": materialization["command"],
                "returncode": 0,
                "exit_code": 0,
                "stdout": "",
                "stderr": "",
                "runner_result": {
                    "backend": "peft",
                    "status": "completed",
                    "execution_mode": "real_import",
                    "real_execution": {
                        "kind": "real_local_peft",
                        "success": True,
                        "artifact_dir": str(artifact_dir),
                        "artifacts": {
                            "adapter_model": str(artifact_dir / "adapter_model.safetensors"),
                            "adapter_config": str(artifact_dir / "adapter_config.json"),
                        },
                        "metrics": {"loss": 0.42, "num_examples": 5},
                    },
                },
                "materialization": materialization,
                "audit": {"status": "executed", "runner_status": "completed"},
                "metadata": {"execution_state": "executed", "executor_mode": "real_import"},
            }
            return SimpleNamespace(to_dict=lambda: payload)

        with patch.object(trainer_service_module, "run_materialized_training_job_bundle", side_effect=fake_run):
            result = trainer.train_result(method="qlora", epochs=1, train_type="sft", backend_hint="peft")

        version_dir = Path(result.adapter_path)
        adapter_model = version_dir / "adapter_model.safetensors"
        adapter_config = version_dir / "adapter_config.json"
        self.assertTrue(adapter_model.exists())
        self.assertTrue(adapter_config.exists())
        self.assertEqual(adapter_model.read_text(encoding="utf-8"), "real-peft-adapter\n")

        training_meta = json.loads((version_dir / "training_meta.json").read_text(encoding="utf-8"))
        manifest = json.loads((version_dir / "adapter_manifest.json").read_text(encoding="utf-8"))
        metadata = dict(manifest.get("metadata") or {})
        store = AdapterStore(home=self.pfe_home)
        row = store.list_version_records(limit=1)[0]

        self.assertIn("real_execution_artifacts", training_meta)
        self.assertIn("artifact_sync", training_meta)
        self.assertTrue(training_meta["artifact_sync"]["available"])
        self.assertIn("adapter_model", training_meta["artifact_sync"]["synced_files"])
        self.assertEqual(training_meta["real_execution_artifacts"]["artifact_dir"], str(version_dir / "trainer_job_outputs" / "peft_lora"))
        self.assertEqual(training_meta["artifact_sync"]["synced_files"]["adapter_model"]["target"], str(adapter_model))
        self.assertEqual(manifest["artifact_format"], "peft_lora")
        self.assertEqual(metadata["training"]["backend"], result.execution_backend)
        self.assertEqual(row["artifact_path"], str(adapter_model))
        self.assertEqual(row["artifact_format"], "peft_lora")


if __name__ == "__main__":
    unittest.main()
