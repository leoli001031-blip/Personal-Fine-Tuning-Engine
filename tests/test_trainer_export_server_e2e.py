from __future__ import annotations

import asyncio
import importlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
for package_dir in ("pfe-core", "pfe-cli", "pfe-server"):
    package_path = str(ROOT / package_dir)
    if package_path not in os.sys.path:
        os.sys.path.insert(0, package_path)

from pfe_core.pipeline import PipelineService
from pfe_core.server_services import InferenceServiceAdapter, PipelineServiceAdapter
from pfe_core.inference.export_runtime import build_llama_cpp_export_command_plan
from pfe_core.trainer.service import TrainerService
from pfe_server.app import ServiceBundle, build_serve_plan, create_app, serve, smoke_test_request
from pfe_server.auth import ServerSecurityConfig

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
        return {"version": "20260323-991", "path": str(self.version_dir), "manifest": {"version": "20260323-991"}}

    def mark_pending_eval(self, version: str, *, num_samples: int, metrics: dict[str, object] | None = None) -> None:
        del version, num_samples, metrics


class TrainerExportServerE2ETests(unittest.TestCase):
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

    def test_trainer_real_execution_fallback_records_audit_and_persisted_meta(self) -> None:
        pipeline = PipelineService()
        pipeline.generate(scenario="life-coach", style="温和、共情", num_samples=8)

        version_dir = self.pfe_home / "adapters" / "user_default" / "20260323-991"
        trainer = TrainerService(store=_NoopTrainerStore(version_dir))
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
            "installed_packages": {"torch": True, "transformers": True},
            "dependency_versions": {"torch": "2.5.0"},
            "notes": ["synthetic runtime"],
        }

        with patch.object(trainer_service_module, "detect_trainer_runtime") as detect_runtime, patch.object(
            trainer_service_module.importlib.util,
            "find_spec",
            return_value=None,
        ), patch.object(trainer_service_module.importlib, "import_module", side_effect=ImportError("missing dependency")):
            detect_runtime.return_value = SimpleNamespace(to_dict=lambda: dict(runtime_snapshot))
            result = trainer.train_result(
                method="qlora",
                epochs=1,
                base_model="mock-llama-target",
                train_type="sft",
                backend_hint="peft",
            )

        training_meta = json.loads((version_dir / "training_meta.json").read_text(encoding="utf-8"))
        self.assertEqual(result.execution_backend, "mock_local")
        self.assertEqual(result.execution_executor, "mock_local")
        self.assertEqual(result.executor_mode, "phase0_mock")
        self.assertEqual(result.backend_dispatch["requested_backend"], "peft")
        self.assertEqual(result.backend_dispatch["execution_backend"], "mock_local")
        self.assertEqual(result.backend_dispatch["dispatch_mode"], "fallback")
        self.assertEqual(result.audit_info["backend_dispatch"]["requested_backend"], "peft")
        self.assertEqual(result.audit_info["backend_dispatch"]["execution_executor"], "mock_local")
        self.assertEqual(result.audit_info["backend_dispatch"]["dispatch_mode"], "fallback")
        self.assertIn("export_write", result.audit_info)
        self.assertTrue(result.audit_info["export_write"]["metadata"]["materialized"])
        self.assertEqual(result.audit_info["export_write"]["metadata"]["written_count"], 3)
        self.assertEqual(result.training_config["backend_dispatch"]["execution_executor"], "mock_local")
        self.assertEqual(training_meta["execution_backend"], "mock_local")
        self.assertEqual(training_meta["execution_executor"], "mock_local")
        self.assertEqual(training_meta["executor_mode"], "phase0_mock")
        self.assertEqual(training_meta["audit"]["backend_dispatch"]["execution_executor"], "mock_local")
        self.assertEqual(training_meta["audit"]["export_write"]["metadata"]["written_count"], 3)

    def test_gguf_export_command_plan_records_runner_and_audit_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            adapter_dir = tmp / "adapter"
            adapter_dir.mkdir()
            tool = tmp / "fake-llama-export.sh"
            tool.write_text("#!/bin/sh\necho export:$@\n", encoding="utf-8")
            tool.chmod(0o755)

            plan = build_llama_cpp_export_command_plan(
                target_backend="llama_cpp",
                source_artifact_format="peft_lora",
                adapter_dir=adapter_dir,
                tool_path=tool,
                source_model_path=tmp / "source-model",
                extra_args=["--dry-run"],
            )

        self.assertEqual(plan.tool_resolution["resolved_path"], str(tool))
        self.assertTrue(plan.tool_resolution["exists"])
        self.assertTrue(plan.tool_resolution["executable"])
        self.assertEqual(plan.tool_resolution["source"], "explicit")
        self.assertEqual(plan.output_dir, str(adapter_dir / "gguf_merged"))
        self.assertEqual(plan.output_artifact_path, str(adapter_dir / "gguf_merged" / "adapter_model.gguf"))
        self.assertEqual(plan.command[0], str(tool))
        self.assertIn("--output-dir", plan.command)
        self.assertIn(str(adapter_dir / "gguf_merged"), plan.command)
        self.assertIn("--target-format", plan.command)
        self.assertEqual(plan.audit["tool_available"], True)
        self.assertEqual(plan.audit["required_export"], True)
        self.assertEqual(plan.audit["placeholder_only"], False)
        self.assertIn("merged GGUF", plan.audit["constraint"])
        self.assertEqual(plan.metadata["backend"], "llama_cpp")
        self.assertEqual(plan.metadata["extra_args_count"], 1)

    def test_serve_runtime_metadata_and_launch_mode_are_consistent_with_probe(self) -> None:
        pipeline = PipelineService()
        fake_uvicorn = ModuleType("uvicorn")
        run_calls: list[dict[str, object]] = []

        def fake_run(app: object, **kwargs: object) -> None:
            run_calls.append({"app": app, **kwargs})

        fake_uvicorn.run = fake_run  # type: ignore[attr-defined]

        with patch.dict(sys.modules, {"uvicorn": fake_uvicorn}):
            serve_plan = build_serve_plan(workspace=str(self.pfe_home), dry_run=False)
            self.assertFalse(serve_plan.runtime.dry_run)
            self.assertTrue(serve_plan.runtime.uvicorn_available)
            self.assertEqual(serve_plan.runtime_probe["launch_mode"], "uvicorn.run")
            self.assertEqual(serve_plan.runtime_probe["serve_summary"]["launch_mode"], "uvicorn.run")
            self.assertEqual(serve_plan.runtime_probe["runner"]["kind"], "uvicorn.run")

            bundle = ServiceBundle(
                inference=InferenceServiceAdapter(pipeline),
                pipeline=PipelineServiceAdapter(pipeline),
                security=ServerSecurityConfig(),
                provider="core",
                workspace=str(self.pfe_home),
                runtime_probe=serve_plan.runtime_probe,
            )
            app = create_app(bundle)
            status_result = asyncio.run(smoke_test_request(app, path="/pfe/status", query_params={"detail": "full"}))

            self.assertEqual(status_result["status_code"], 200)
            body = status_result["body"]
            self.assertEqual(body["metadata"]["server_runtime"]["launch_mode"], "uvicorn.run")
            self.assertEqual(body["metadata"]["server_runtime"]["uvicorn_available"], True)
            self.assertEqual(body["metadata"]["server_runtime"]["command"], serve_plan.runtime_probe["command"])
            self.assertEqual(body["metadata"]["server_runtime"]["runner"]["kind"], "uvicorn.run")
            self.assertIn("/pfe/status", {item["path"] for item in body["metadata"]["server_runtime"]["probe_paths"]})

            serve_output = serve(workspace=str(self.pfe_home), dry_run=False)

        self.assertEqual(serve_output, "started uvicorn on 127.0.0.1:8921")
        self.assertEqual(len(run_calls), 1)
        self.assertEqual(run_calls[0]["host"], "127.0.0.1")
        self.assertEqual(run_calls[0]["port"], 8921)
        self.assertFalse(run_calls[0]["reload"])


if __name__ == "__main__":
    unittest.main()
