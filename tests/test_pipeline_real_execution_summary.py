from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for package_dir in ("pfe-core", "pfe-cli", "pfe-server"):
    package_path = str(ROOT / package_dir)
    if package_path not in os.sys.path:
        os.sys.path.insert(0, package_path)

from pfe_core.pipeline import PipelineService


@pytest.mark.slow
class PipelineRealExecutionSummaryTests(unittest.TestCase):
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

    def test_train_result_and_status_include_normalized_execution_summaries(self) -> None:
        service = PipelineService()
        service.generate(scenario="life-coach", style="温和", num_samples=8)

        result = service.train_result(method="qlora", epochs=1, train_type="sft")
        self.assertIn("state", result.job_execution_summary)
        self.assertIn("attempted", result.job_execution_summary)
        self.assertIn("success", result.job_execution_summary)
        self.assertIn("status", result.export_toolchain_summary)
        self.assertIn("required", result.export_toolchain_summary)

        snapshot = service.status()
        last_run = snapshot["trainer"]["last_run"]
        self.assertEqual(last_run["version"], result.version)
        self.assertEqual(last_run["job_execution_summary"]["state"], result.job_execution_summary["state"])
        self.assertEqual(last_run["export_toolchain_summary"]["status"], result.export_toolchain_summary["status"])
        self.assertIn("required", last_run["export_toolchain_summary"])

    def test_training_meta_and_manifest_metadata_persist_execution_summaries(self) -> None:
        service = PipelineService()
        service.generate(scenario="life-coach", style="温和", num_samples=8)
        result = service.train_result(method="qlora", epochs=1, train_type="sft")

        version_dir = self.pfe_home / "adapters" / "user_default" / result.version
        training_meta = json.loads((version_dir / "training_meta.json").read_text(encoding="utf-8"))
        manifest = json.loads((version_dir / "adapter_manifest.json").read_text(encoding="utf-8"))
        manifest_metadata = dict(manifest.get("metadata") or {})

        self.assertIn("job_execution_summary", training_meta)
        self.assertIn("export_toolchain_summary", training_meta)
        self.assertIn("job_execution_summary", training_meta["audit"])
        self.assertIn("export_toolchain_summary", training_meta["audit"])

        self.assertIn("training", manifest_metadata)
        self.assertIn("backend_plan", manifest_metadata)
        self.assertIn("supported_inference_backends", manifest_metadata)
        self.assertEqual(training_meta["job_execution_summary"]["state"], result.job_execution_summary["state"])
        self.assertEqual(training_meta["export_toolchain_summary"]["status"], result.export_toolchain_summary["status"])
        self.assertEqual(manifest_metadata["training"]["backend"], result.training_config["backend"])
        self.assertEqual(manifest_metadata["backend_plan"]["recommended_backend"], result.backend_plan["recommended_backend"])


if __name__ == "__main__":
    unittest.main()
