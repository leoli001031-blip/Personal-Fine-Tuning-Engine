from __future__ import annotations

import asyncio
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
class ServerServeRealRegressionTests(unittest.TestCase):
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

    def test_real_serve_plan_keeps_status_and_command_preview_coherent(self) -> None:
        pipeline = PipelineService()

        pipeline.generate(scenario="life-coach", style="warm", num_samples=8)
        first_result = pipeline.train_result(method="qlora", epochs=1, train_type="sft")
        pipeline.evaluate(base_model="base", adapter=first_result.version, num_samples=3)

        store = AdapterStore(home=self.pfe_home)
        store.promote(first_result.version)
        self.assertEqual(store.current_latest_version(), first_result.version)

        pipeline.generate(scenario="work-coach", style="direct", num_samples=8)
        second_result = pipeline.train_result(method="qlora", epochs=1, train_type="sft")
        pipeline.evaluate(base_model="base", adapter=second_result.version, num_samples=3)

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
            self.assertEqual(serve_plan.runtime_probe["command"], serve_plan.runtime.command)
            self.assertEqual(serve_plan.runtime_probe["runner"]["kind"], "uvicorn.run")

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

            self.assertTrue(body["strict_local"])
            self.assertEqual(body["latest_adapter"]["version"], first_result.version)
            self.assertEqual(body["latest_adapter"]["state"], "promoted")
            self.assertEqual(body["metadata"]["snapshot"]["latest_adapter_version"], first_result.version)
            self.assertEqual(body["metadata"]["snapshot"]["recent_adapter_version"], second_result.version)
            self.assertFalse(body["metadata"]["snapshot"]["recent_adapter"]["latest"])
            self.assertEqual(body["metadata"]["server_runtime"]["launch_mode"], "uvicorn.run")
            self.assertEqual(body["metadata"]["server_runtime"]["command"], serve_plan.runtime.command)
            self.assertEqual(body["metadata"]["server_runtime"]["runner"]["kind"], "uvicorn.run")
            self.assertTrue(body["metadata"]["lifecycle"]["promotion"]["latest_is_promoted"])
            self.assertTrue(body["metadata"]["lifecycle"]["promotion"]["latest_is_latest"])
            self.assertEqual(body["metadata"]["lifecycle"]["promotion"]["latest_adapter_version"], first_result.version)

            serve_output = serve(workspace="user_default", dry_run=False)

        self.assertEqual(serve_output, "started uvicorn on 127.0.0.1:8921")
        self.assertEqual(len(run_calls), 1)
        self.assertEqual(run_calls[0]["host"], "127.0.0.1")
        self.assertEqual(run_calls[0]["port"], 8921)
        self.assertFalse(run_calls[0]["reload"])


if __name__ == "__main__":
    unittest.main()
