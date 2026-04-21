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

from pfe_core.pipeline import PipelineService
from pfe_core.trainer.service import TrainerService

trainer_service_module = importlib.import_module("pfe_core.trainer.service")


class _NoopTrainerStore:
    def __init__(self, version_dir: Path):
        self.version_dir = version_dir
        self.created_training_config: dict[str, object] | None = None

    def create_training_version(
        self,
        *,
        base_model: str,
        training_config: dict[str, object],
        artifact_format: str = "peft_lora",
    ) -> dict[str, object]:
        del base_model, artifact_format
        self.created_training_config = dict(training_config)
        self.version_dir.mkdir(parents=True, exist_ok=True)
        return {"version": "20260324-777", "path": str(self.version_dir), "manifest": {"version": "20260324-777"}}

    def mark_pending_eval(self, version: str, *, num_samples: int, metrics: dict[str, object] | None = None) -> None:
        del version, num_samples, metrics


class TrainerServiceRealArtifactsTests(unittest.TestCase):
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

    def test_real_execution_artifact_bundle_is_synced_into_standard_adapter_dir(self) -> None:
        pipeline = PipelineService()
        pipeline.generate(scenario="life-coach", style="温和、共情", num_samples=8)

        source_root = Path(self.tempdir.name) / "real_execution"
        artifact_dir = source_root / "peft_lora"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        adapter_model = artifact_dir / "adapter_model.safetensors"
        adapter_config = artifact_dir / "adapter_config.json"
        trainer_state = artifact_dir / "trainer_state.json"
        summary_path = source_root / "training_summary.json"
        real_execution_path = source_root / "real_execution.json"
        artifact_manifest_path = source_root / "peft_job_manifest.json"
        metrics_path = source_root / "train_metrics.json"
        adapter_model.write_text("real-local-adapter\n", encoding="utf-8")
        adapter_config.write_text(json.dumps({"peft_type": "LORA"}, ensure_ascii=False) + "\n", encoding="utf-8")
        trainer_state.write_text(json.dumps({"step": 1}, ensure_ascii=False) + "\n", encoding="utf-8")
        summary_path.write_text(json.dumps({"status": "completed"}, ensure_ascii=False) + "\n", encoding="utf-8")
        real_execution_path.write_text(json.dumps({"kind": "real_local_peft"}, ensure_ascii=False) + "\n", encoding="utf-8")
        artifact_manifest_path.write_text(json.dumps({"artifact_dir": str(artifact_dir)}, ensure_ascii=False) + "\n", encoding="utf-8")
        metrics_path.write_text(json.dumps({"loss": 0.125, "num_examples": 5}, ensure_ascii=False) + "\n", encoding="utf-8")

        def fake_run(bundle, *args, **kwargs):
            materialization = bundle.to_dict() if hasattr(bundle, "to_dict") else dict(bundle)
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
                            "adapter_model": str(adapter_model),
                            "adapter_config": str(adapter_config),
                        },
                        "artifact_manifest_path": str(artifact_manifest_path),
                        "summary_path": str(summary_path),
                        "real_execution_path": str(real_execution_path),
                        "trainer_state_path": str(trainer_state),
                        "metrics": {"loss": 0.125, "num_examples": 5},
                        "metrics_path": str(metrics_path),
                    },
                },
                "materialization": materialization,
                "audit": {"status": "executed", "runner_status": "completed"},
                "metadata": {"execution_state": "executed", "executor_mode": "real_import"},
            }
            return SimpleNamespace(to_dict=lambda: payload)

        with patch.object(trainer_service_module, "run_materialized_training_job_bundle", side_effect=fake_run):
            result = TrainerService(store=_NoopTrainerStore(self.pfe_home / "adapters" / "user_default" / "20260324-777")).train_result(
                method="qlora",
                epochs=1,
                train_type="sft",
                backend_hint="peft",
            )

        version_dir = Path(result.adapter_path)
        self.assertTrue((version_dir / "adapter_model.safetensors").exists())
        self.assertTrue((version_dir / "adapter_config.json").exists())
        self.assertEqual((version_dir / "adapter_model.safetensors").read_text(encoding="utf-8"), "real-local-adapter\n")
        self.assertIn("real_execution_artifacts", result.metrics)
        self.assertIn("artifact_sync", result.metrics)
        self.assertTrue(result.metrics["artifact_sync"]["available"])
        self.assertIn("adapter_model", result.metrics["artifact_sync"]["synced_files"])
        self.assertEqual(result.metrics["real_execution_artifacts"]["artifact_dir"], str(artifact_dir))

        training_meta = json.loads((version_dir / "training_meta.json").read_text(encoding="utf-8"))
        manifest = json.loads((version_dir / "adapter_manifest.json").read_text(encoding="utf-8"))
        metadata = dict(manifest.get("metadata") or {})

        self.assertEqual(training_meta["job_execution_summary"]["state"], "executed")
        self.assertEqual(training_meta["real_execution_summary"]["state"], "completed")
        self.assertEqual(training_meta["real_execution_artifacts"]["artifact_dir"], str(artifact_dir))
        self.assertTrue(training_meta["artifact_sync"]["available"])
        self.assertEqual(training_meta["real_execution_artifacts"]["artifact_dir"], str(artifact_dir))
        self.assertEqual(metadata["real_execution_artifacts"]["artifact_dir"], str(artifact_dir))
        self.assertTrue(metadata["artifact_sync"]["available"])
        self.assertEqual(metadata["artifact_sync"]["synced_files"]["adapter_model"]["target"], str(version_dir / "adapter_model.safetensors"))


if __name__ == "__main__":
    unittest.main()
