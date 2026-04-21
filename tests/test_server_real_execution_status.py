from __future__ import annotations

import asyncio
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
from pfe_server.app import create_app, smoke_test_request


@pytest.mark.slow
class ServerRealExecutionStatusTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.previous_home = os.environ.get("PFE_HOME")
        self.pfe_home = Path(self.tempdir.name) / ".pfe"
        os.environ["PFE_HOME"] = str(self.pfe_home)
        self.pipeline = PipelineService()
        self.pipeline.generate(scenario="life-coach", style="warm", num_samples=6)
        self.pipeline.train_result(method="qlora", epochs=1, train_type="sft")
        self.app = create_app()

    def tearDown(self) -> None:
        if self.previous_home is None:
            os.environ.pop("PFE_HOME", None)
        else:
            os.environ["PFE_HOME"] = self.previous_home
        self.tempdir.cleanup()

    def test_status_includes_real_execution_and_export_toolchain(self) -> None:
        result = asyncio.run(smoke_test_request(self.app, path="/pfe/status"))
        self.assertEqual(result["status_code"], 200)
        body = result["body"]
        metadata = body["metadata"]

        self.assertIn("trainer", metadata)
        self.assertIn("real_execution", metadata)
        self.assertIn("export_toolchain", metadata)
        self.assertIn("lifecycle", metadata)

        real_execution = metadata["real_execution"]
        self.assertIn("state", real_execution)
        self.assertIn("backend", real_execution)
        self.assertIn("execution_mode", real_execution)
        self.assertIn("output_dir", real_execution)

        export_toolchain = metadata["export_toolchain"]
        self.assertIn("state", export_toolchain)
        self.assertIn("toolchain_status", export_toolchain)
        self.assertIn("output_dir", export_toolchain)
        self.assertIn("write_state", export_toolchain)

        export = metadata["export"]
        self.assertIn("output_artifact_path", export)
        self.assertIn("output_artifact_valid", export)
        self.assertIn("output_artifact_size_bytes", export)
        self.assertIn("converter_kind", export)
        self.assertIn("outtype", export)
        self.assertIn("toolchain_resolved_path", export)

        trainer = metadata["trainer"]
        self.assertIn("real_execution_summary", trainer)
        self.assertIn("export_toolchain_summary", trainer)
        self.assertIn("export_artifact_summary", trainer)
        self.assertEqual(trainer["real_execution_summary"].get("state"), real_execution.get("state"))
        self.assertEqual(trainer["export_toolchain_summary"].get("toolchain_status"), export_toolchain.get("toolchain_status"))
        self.assertEqual(trainer["export_artifact_summary"].get("path"), export.get("export_artifact_path"))
        self.assertEqual(trainer["export_artifact_summary"].get("valid"), export.get("export_artifact_valid"))

        lifecycle = metadata["lifecycle"]
        self.assertIn("promotion", lifecycle)
        self.assertEqual(lifecycle["promotion"].get("last_real_execution_state"), real_execution.get("state"))
        self.assertEqual(lifecycle["promotion"].get("last_export_toolchain_status"), export_toolchain.get("toolchain_status"))
        self.assertIn("serve", lifecycle)
        self.assertIn("latest_adapter_exists", lifecycle["serve"])
        self.assertIn("adapter_resolution_state", lifecycle["serve"])
        self.assertIn("using_promoted_adapter", lifecycle["serve"])


if __name__ == "__main__":
    unittest.main()
