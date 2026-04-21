from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
for package_dir in ("pfe-core", "pfe-cli", "pfe-server"):
    package_path = str(ROOT / package_dir)
    if package_path not in os.sys.path:
        os.sys.path.insert(0, package_path)

from pfe_core.adapter_store.store import AdapterStore
from pfe_core.pipeline import PipelineService
from pfe_core.server_services import InferenceServiceAdapter, PipelineServiceAdapter
from pfe_server.app import ServiceBundle, build_serve_plan, create_app, serve, smoke_test_request
from pfe_server.auth import ServerSecurityConfig


@pytest.mark.slow
class Phase0LifecycleE2ETests(unittest.TestCase):
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

    def test_train_eval_promote_and_serve_status_keep_latest_lifecycle_export_and_runtime_in_sync(self) -> None:
        pipeline = PipelineService()
        pipeline.generate(scenario="life-coach", style="温和、共情", num_samples=12)

        train_result = pipeline.train_result(method="qlora", epochs=1, train_type="sft")
        self.assertEqual(train_result.metrics["train_type"], "sft")
        self.assertIn("backend_dispatch", train_result.metrics)
        self.assertIn("export_write", train_result.metrics)
        self.assertEqual(train_result.audit_info["backend_dispatch"]["execution_backend"], train_result.execution_backend)

        store = AdapterStore(home=self.pfe_home)
        version_records = store.list_version_records(limit=1)
        self.assertEqual(len(version_records), 1)
        version = version_records[0]["version"]
        version_dir = self.pfe_home / "adapters" / "user_default" / version
        manifest_before_promote = json.loads((version_dir / "adapter_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest_before_promote["state"], "pending_eval")
        self.assertEqual(store.current_latest_version(), None)

        eval_report = json.loads(pipeline.evaluate(base_model="base", adapter="latest", num_samples=3))
        self.assertGreaterEqual(eval_report["num_test_samples"], 1)
        self.assertGreaterEqual(eval_report["scores"]["quality_preservation"], 0.8)

        promote_result = store.promote(version)
        self.assertIn(version, promote_result)
        self.assertEqual(store.current_latest_version(), version)

        manifest_after_promote = json.loads((version_dir / "adapter_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest_after_promote["state"], "promoted")
        self.assertIn("promoted_at", manifest_after_promote)

        fake_uvicorn = ModuleType("uvicorn")
        run_calls: list[dict[str, object]] = []

        def fake_run(app: object, **kwargs: object) -> None:
            run_calls.append({"app": app, **kwargs})

        fake_uvicorn.run = fake_run  # type: ignore[attr-defined]

        with patch.dict(sys.modules, {"uvicorn": fake_uvicorn}):
            serve_plan = build_serve_plan(workspace="user_default", dry_run=False)
            self.assertFalse(serve_plan.runtime.dry_run)
            self.assertTrue(serve_plan.runtime.uvicorn_available)
            self.assertEqual(serve_plan.runtime_probe["launch_mode"], "uvicorn.run")
            self.assertEqual(serve_plan.runtime_probe["runner"]["kind"], "uvicorn.run")
            self.assertIn("uvicorn", " ".join(serve_plan.runtime_probe["command"]))

            bundle = ServiceBundle(
                inference=InferenceServiceAdapter(pipeline),
                pipeline=PipelineServiceAdapter(pipeline),
                security=ServerSecurityConfig(),
                provider="core",
                workspace="user_default",
                runtime_probe=serve_plan.runtime_probe,
            )
            app = create_app(bundle)
            status_result = asyncio.run(smoke_test_request(app, path="/pfe/status", query_params={"detail": "full"}))
            self.assertEqual(status_result["status_code"], 200)
            body = status_result["body"]

            self.assertEqual(body["latest_adapter"]["version"], version)
            self.assertEqual(body["latest_adapter"]["state"], "promoted")
            self.assertEqual(body["runtime"]["latest_adapter_version"], version)
            self.assertEqual(body["metadata"]["snapshot"]["latest_adapter_version"], version)
            self.assertEqual(body["metadata"]["export"]["latest_adapter_version"], version)
            self.assertEqual(body["metadata"]["lifecycle"]["promotion"]["state"], "promoted")
            self.assertTrue(body["metadata"]["lifecycle"]["promotion"]["latest_is_promoted"])
            self.assertTrue(body["metadata"]["lifecycle"]["promotion"]["latest_is_latest"])
            self.assertEqual(body["metadata"]["export"]["latest_adapter_state"], "promoted")
            self.assertEqual(body["metadata"]["export"]["write_state"], "materialized")
            self.assertTrue(body["metadata"]["export"]["materialized"])
            self.assertEqual(body["metadata"]["server_runtime"]["launch_mode"], "uvicorn.run")
            self.assertEqual(body["metadata"]["server_runtime"]["command"], serve_plan.runtime_probe["command"])
            self.assertEqual(body["metadata"]["server_runtime"]["runner"]["kind"], "uvicorn.run")

            serve_output = serve(workspace="user_default", dry_run=False)

        self.assertEqual(serve_output, "started uvicorn on 127.0.0.1:8921")
        self.assertEqual(len(run_calls), 1)
        self.assertEqual(run_calls[0]["host"], "127.0.0.1")
        self.assertEqual(run_calls[0]["port"], 8921)
        self.assertFalse(run_calls[0]["reload"])


if __name__ == "__main__":
    unittest.main()
