from __future__ import annotations

import asyncio
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

from pfe_core.adapter_store.store import AdapterStore
from pfe_core.pipeline import PipelineService
from pfe_server.app import create_app, smoke_test_request


@pytest.mark.slow
class Phase0ClosedLoopRegressionTests(unittest.TestCase):
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

    def test_train_eval_promote_and_status_keep_latest_and_artifacts_in_sync(self) -> None:
        pipeline = PipelineService()
        pipeline.generate(scenario="life-coach", style="温和、共情", num_samples=12)

        train_result = pipeline.train_result(method="qlora", epochs=1, train_type="sft")
        version = train_result.version
        version_dir = self.pfe_home / "adapters" / "user_default" / version

        store = AdapterStore(home=self.pfe_home)
        self.assertIsNone(store.current_latest_version())

        manifest_before = json.loads((version_dir / "adapter_manifest.json").read_text(encoding="utf-8"))
        training_meta = json.loads((version_dir / "training_meta.json").read_text(encoding="utf-8"))
        training_meta_real_execution = dict(training_meta.get("real_execution_summary") or {})
        training_meta_job_execution = dict(training_meta.get("job_execution_summary") or {})
        training_meta_export_toolchain = dict(training_meta.get("export_toolchain_summary") or {})
        self.assertEqual(manifest_before["version"], version)
        self.assertEqual(manifest_before["state"], "pending_eval")
        self.assertEqual(training_meta_job_execution["state"], train_result.job_execution_summary["state"])
        if training_meta_real_execution:
            self.assertEqual(training_meta_real_execution, train_result.real_execution_summary)
        self.assertEqual(training_meta_export_toolchain["status"], train_result.export_toolchain_summary["status"])

        status_before = pipeline.status()
        self.assertIsNone(status_before["latest_adapter_version"])
        self.assertEqual(status_before["recent_adapter_version"], version)
        self.assertEqual(status_before["recent_adapter"]["state"], "pending_eval")
        self.assertEqual(status_before["trainer"]["last_run"]["version"], version)
        self.assertEqual(status_before["trainer"]["last_run"]["job_execution_summary"]["state"], train_result.job_execution_summary["state"])
        if train_result.real_execution_summary:
            self.assertEqual(status_before["trainer"]["last_run"]["real_execution_summary"], train_result.real_execution_summary)
        self.assertEqual(status_before["trainer"]["last_run"]["export_toolchain_summary"]["status"], train_result.export_toolchain_summary["status"])

        eval_report = json.loads(pipeline.evaluate(base_model="base", adapter="latest", num_samples=3))
        self.assertGreaterEqual(eval_report["num_test_samples"], 1)

        promoted = store.promote(version)
        self.assertIn(version, promoted)
        self.assertEqual(store.current_latest_version(), version)

        manifest_after = json.loads((version_dir / "adapter_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest_after["state"], "promoted")
        self.assertEqual(manifest_after["version"], version)
        self.assertIn("promoted_at", manifest_after)

        status_after = pipeline.status()
        self.assertEqual(status_after["latest_adapter_version"], version)
        self.assertEqual(status_after["latest_adapter"]["version"], version)
        self.assertEqual(status_after["latest_adapter"]["state"], "promoted")
        self.assertEqual(status_after["recent_adapter_version"], version)
        self.assertEqual(status_after["trainer"]["last_run"]["version"], version)
        self.assertEqual(status_after["trainer"]["last_run"]["job_execution_summary"]["state"], train_result.job_execution_summary["state"])
        if train_result.real_execution_summary:
            self.assertEqual(status_after["trainer"]["last_run"]["real_execution_summary"], train_result.real_execution_summary)
        self.assertEqual(status_after["trainer"]["last_run"]["export_toolchain_summary"]["status"], train_result.export_toolchain_summary["status"])

        app = create_app()
        status_response = asyncio.run(smoke_test_request(app, path="/pfe/status", query_params={"detail": "full"}))
        self.assertEqual(status_response["status_code"], 200)
        body = status_response["body"]

        self.assertEqual(body["latest_adapter"]["version"], version)
        self.assertEqual(body["latest_adapter"]["state"], "promoted")
        self.assertEqual(body["runtime"]["latest_adapter_version"], version)
        self.assertEqual(body["metadata"]["snapshot"]["latest_adapter_version"], version)
        self.assertIn("state", body["metadata"]["trainer"]["job_execution"])
        if body["metadata"]["trainer"]["job_execution"]["state"] != "unknown":
            self.assertEqual(body["metadata"]["trainer"]["job_execution"]["state"], train_result.job_execution_summary["state"])
        if train_result.real_execution_summary:
            self.assertEqual(body["metadata"]["trainer"]["real_execution"], train_result.real_execution_summary)
            self.assertEqual(body["metadata"]["lifecycle"]["promotion"]["last_real_execution_state"], train_result.real_execution_summary["state"])
        self.assertIn("status", body["metadata"]["trainer"]["export_toolchain"])
        if body["metadata"]["trainer"]["export_toolchain"]["status"] != "unknown":
            self.assertEqual(body["metadata"]["trainer"]["export_toolchain"]["status"], train_result.export_toolchain_summary["status"])
        self.assertEqual(body["metadata"]["lifecycle"]["promotion"]["last_export_toolchain_status"], train_result.export_toolchain_summary["status"])


if __name__ == "__main__":
    unittest.main()
