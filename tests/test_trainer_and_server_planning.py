from __future__ import annotations

import asyncio
import importlib
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

from pfe_core.pipeline import PipelineService
from pfe_core.server_services import InferenceServiceAdapter, PipelineServiceAdapter
from pfe_core.inference.export_runtime import write_materialized_export_plan
from pfe_core.trainer.runtime import plan_trainer_backend
from pfe_core.trainer.service import TrainerService
from pfe_server.app import ServiceBundle, build_serve_plan, create_app, serve, smoke_test_request
from pfe_server.auth import ServerSecurityConfig

trainer_service_module = importlib.import_module("pfe_core.trainer.service")


class _FakeTrainerStore:
    def __init__(self, version_dir: Path):
        self.version_dir = version_dir
        self.created_training_config: dict[str, object] | None = None
        self.pending_eval_calls: list[dict[str, object]] = []

    def create_training_version(self, *, base_model: str, training_config: dict[str, object], artifact_format: str = "peft_lora") -> dict[str, object]:
        del base_model, artifact_format
        self.created_training_config = dict(training_config)
        self.version_dir.mkdir(parents=True, exist_ok=True)
        return {"version": "20260323-999", "path": str(self.version_dir), "manifest": {"version": "20260323-999"}}

    def mark_pending_eval(self, version: str, *, num_samples: int, metrics: dict[str, object] | None = None) -> None:
        self.pending_eval_calls.append({"version": version, "num_samples": num_samples, "metrics": dict(metrics or {})})


@pytest.mark.slow
class TrainerAndServerPlanningTests(unittest.TestCase):
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

    def test_trainer_service_persists_backend_planning_fields(self) -> None:
        pipeline = PipelineService()
        pipeline.generate(scenario="life-coach", style="温和、共情", num_samples=8)

        version_dir = self.pfe_home / "adapters" / "user_default" / "20260323-999"
        fake_store = _FakeTrainerStore(version_dir)
        trainer = TrainerService(store=fake_store)

        result = trainer.train_result(
            method="qlora",
            epochs=2,
            base_model="mock-llama-target",
            train_type="sft",
        )
        self.assertEqual(result.version, "20260323-999")
        expected_backend = fake_store.created_training_config["backend"]
        self.assertIn(expected_backend, {"mock_local", "peft"})
        self.assertIn("replay_ratio", fake_store.created_training_config)

        training_meta = json.loads((version_dir / "training_meta.json").read_text(encoding="utf-8"))
        self.assertEqual(training_meta["backend"], expected_backend)
        self.assertEqual(training_meta["method"], "qlora")
        self.assertEqual(training_meta["epochs"], 2)
        self.assertEqual(training_meta["train_type"], "sft")
        self.assertIn("replay_ratio", training_meta)
        self.assertGreaterEqual(training_meta["num_train_samples"], 1)

    def test_status_and_serve_outputs_expose_planning_fields(self) -> None:
        pipeline = PipelineService()
        app = create_app(
            ServiceBundle(
                inference=InferenceServiceAdapter(pipeline),
                pipeline=PipelineServiceAdapter(pipeline),
                security=ServerSecurityConfig(),
                provider="core",
                workspace=str(self.pfe_home),
            )
        )
        status_result = asyncio.run(smoke_test_request(app, path="/pfe/status"))
        self.assertEqual(status_result["status_code"], 200)
        status_body = status_result["body"]
        self.assertIn("runtime", status_body)
        self.assertTrue(status_body["runtime"]["management_local_only"])
        self.assertIn("metadata", status_body)
        self.assertIn("inference", status_body["metadata"])
        self.assertIn("backend_plan", status_body["metadata"]["inference"])
        self.assertIn("capabilities", status_body["metadata"])
        self.assertIn("generate", status_body["metadata"]["capabilities"])
        self.assertIn("user_modeling", status_body["metadata"])
        self.assertEqual(status_body["metadata"]["user_modeling"]["primary_runtime_system"], "user_memory")
        self.assertIn("pipeline", status_body["metadata"])

        serve_plan = build_serve_plan(workspace=str(self.pfe_home))
        self.assertIn("uvicorn", serve_plan.command)
        self.assertIn("kind", serve_plan.runtime.runner)
        self.assertIn("target", serve_plan.runtime.runner)
        self.assertTrue(serve_plan.runtime.dry_run)

        serve_output = serve(workspace=str(self.pfe_home), dry_run=True)
        self.assertIn("provider=", serve_output)
        self.assertIn("dry_run=True", serve_output)
        self.assertIn("command=", serve_output)

    def test_trainer_backend_plan_remains_serializable(self) -> None:
        plan = plan_trainer_backend(
            train_type="sft",
            runtime={
                "platform_name": "Linux",
                "machine": "x86_64",
                "runtime_device": "cpu",
                "cuda_available": False,
                "mps_available": False,
                "apple_silicon": False,
                "installed_packages": {"torch": True, "transformers": True, "peft": True, "accelerate": True},
                "dependency_versions": {"torch": "2.5.0"},
            },
        )

        payload = plan.to_dict()
        self.assertIn("recommended_backend", payload)
        self.assertIn("runtime_summary", payload)
        self.assertIn("capability", payload)
        self.assertEqual(payload["runtime_summary"]["runtime_device"], "cpu")

    def test_train_status_and_serve_keep_planning_fields_consistent(self) -> None:
        pipeline = PipelineService()
        pipeline.generate(scenario="life-coach", style="温和、共情", num_samples=8)

        version_dir = self.pfe_home / "adapters" / "user_default" / "20260323-998"
        fake_store = _FakeTrainerStore(version_dir)
        pipeline.trainer = TrainerService(store=fake_store)

        train_result = pipeline.train_result(
            method="qlora",
            epochs=1,
            base_model="mock-llama-target",
            train_type="sft",
        )
        training_meta = json.loads((version_dir / "training_meta.json").read_text(encoding="utf-8"))
        self.assertEqual(training_meta["backend_plan"]["recommended_backend"], train_result.backend_plan["recommended_backend"])
        self.assertEqual(training_meta["backend_plan"]["requires_export_step"], train_result.backend_plan["requires_export_step"])

        status_snapshot = pipeline.status()
        trainer_status = status_snapshot["trainer"]
        self.assertIn("runtime", trainer_status)
        self.assertIn("plans", trainer_status)
        self.assertIn("sft", trainer_status["plans"])
        self.assertIn("dpo", trainer_status["plans"])
        self.assertEqual(
            trainer_status["plans"]["sft"]["recommended_backend"],
            train_result.backend_plan["recommended_backend"],
        )
        self.assertEqual(trainer_status["runtime"]["runtime_device"], train_result.runtime["runtime_device"])
        self.assertIn("requires_export_step", trainer_status["plans"]["sft"])

        app = create_app(
            ServiceBundle(
                inference=InferenceServiceAdapter(pipeline),
                pipeline=PipelineServiceAdapter(pipeline),
                security=ServerSecurityConfig(),
                provider="core",
                workspace=str(self.pfe_home),
            )
        )
        status_result = asyncio.run(smoke_test_request(app, path="/pfe/status"))
        self.assertEqual(status_result["status_code"], 200)
        status_body = status_result["body"]
        self.assertEqual(
            status_body["metadata"]["pipeline"]["trainer"]["plans"]["sft"]["recommended_backend"],
            train_result.backend_plan["recommended_backend"],
        )
        self.assertIn("requires_export_step", status_body["metadata"]["pipeline"]["trainer"]["plans"]["sft"])

        serve_plan = build_serve_plan(workspace=str(self.pfe_home))
        self.assertIn("uvicorn", serve_plan.command)
        self.assertEqual(serve_plan.runtime.app_target, "pfe_server.app:app")
        self.assertTrue(serve_plan.runtime.dry_run)
        self.assertIn("target", serve_plan.runtime.runner)
        self.assertIn("kind", serve_plan.runtime.runner)

        serve_output = serve(workspace=str(self.pfe_home), dry_run=True)
        self.assertIn("provider=", serve_output)
        self.assertIn("app=", serve_output)
        self.assertIn("uvicorn_available=", serve_output)
        self.assertIn("dry_run=True", serve_output)
        self.assertIn("command=", serve_output)

    def test_train_result_backend_dispatch_and_export_runtime_match_status_and_serve(self) -> None:
        pipeline = PipelineService()
        pipeline.generate(scenario="life-coach", style="温和、共情", num_samples=8)

        version_dir = self.pfe_home / "adapters" / "user_default" / "20260323-997"
        fake_store = _FakeTrainerStore(version_dir)
        trainer = TrainerService(store=fake_store)

        def _stub_write_adapter_artifacts(**kwargs: object) -> dict[str, object]:
            train_samples = kwargs.get("train_samples") or []
            fresh = kwargs.get("fresh") or []
            replay = kwargs.get("replay") or []
            training_config = kwargs.get("training_config") or {}
            return {
                "loss": round(1.0 / max(len(train_samples), 1), 6),
                "epochs": kwargs.get("epochs", 0),
                "replay_ratio": training_config.get("replay_ratio", 0.0),
                "train_type": kwargs.get("train_type", "sft"),
                "num_fresh_samples": len(fresh),
                "num_replay_samples": len(replay),
            }

        trainer._write_adapter_artifacts = _stub_write_adapter_artifacts  # type: ignore[assignment]
        train_result = trainer.train_result(method="qlora", epochs=1, train_type="sft")
        pipeline.trainer = trainer
        self.assertIn("backend_plan", train_result.metrics)
        self.assertIn("export_runtime", train_result.metrics)
        self.assertEqual(train_result.backend_plan["recommended_backend"], train_result.metrics["backend_plan"]["recommended_backend"])
        self.assertIn(
            train_result.backend_dispatch["execution_backend"],
            {train_result.backend_plan["recommended_backend"], "mock_local"},
        )
        self.assertEqual(
            train_result.export_runtime["target_artifact_format"],
            train_result.backend_plan["capability"]["artifact_format"],
        )
        self.assertEqual(
            train_result.export_runtime["backend_plan"]["target_artifact_format"],
            train_result.export_runtime["target_artifact_format"],
        )

        status_snapshot = pipeline.status()
        trainer_plan = status_snapshot["trainer"]["plans"]["sft"]
        self.assertEqual(trainer_plan["recommended_backend"], train_result.backend_plan["recommended_backend"])
        self.assertEqual(trainer_plan["export_format"], train_result.backend_plan["export_format"])
        self.assertEqual(trainer_plan["requires_export_step"], train_result.backend_plan["requires_export_step"])

        app = create_app(
            ServiceBundle(
                inference=InferenceServiceAdapter(pipeline),
                pipeline=PipelineServiceAdapter(pipeline),
                security=ServerSecurityConfig(),
                provider="core",
                workspace=str(self.pfe_home),
            )
        )
        status_result = asyncio.run(smoke_test_request(app, path="/pfe/status"))
        self.assertEqual(status_result["status_code"], 200)
        status_body = status_result["body"]
        self.assertEqual(status_body["provider"], "core")
        self.assertEqual(status_body["metadata"]["pipeline"]["trainer"]["plans"]["sft"]["recommended_backend"], train_result.backend_plan["recommended_backend"])
        self.assertEqual(status_body["metadata"]["pipeline"]["trainer"]["plans"]["sft"]["export_format"], train_result.backend_plan["export_format"])

        serve_plan = build_serve_plan(workspace=str(self.pfe_home))
        self.assertEqual(serve_plan.runtime.provider, status_body["provider"])
        self.assertEqual(serve_plan.runtime.app_target, "pfe_server.app:app")
        self.assertTrue(serve_plan.runtime.dry_run)

        serve_output = serve(workspace=str(self.pfe_home), dry_run=True)
        self.assertIn(f"provider={serve_plan.runtime.provider}", serve_output)
        self.assertIn("app=", serve_output)
        self.assertIn("dry_run=True", serve_output)
        self.assertIn("command=", serve_output)

    def test_dry_run_false_serve_and_export_audit_paths_remain_consistent(self) -> None:
        pipeline = PipelineService()
        pipeline.generate(scenario="life-coach", style="温和、共情", num_samples=8)

        version_dir = self.pfe_home / "adapters" / "user_default" / "20260323-996"
        fake_store = _FakeTrainerStore(version_dir)
        trainer = TrainerService(store=fake_store)

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
            "dependency_versions": {"torch": "2.5.0"},
            "notes": ["synthetic runtime"],
        }
        fake_uvicorn = ModuleType("uvicorn")
        run_calls: list[dict[str, object]] = []

        def fake_run(app: object, **kwargs: object) -> None:
            run_calls.append({"app": app, **kwargs})

        fake_uvicorn.run = fake_run  # type: ignore[attr-defined]

        with patch.object(trainer_service_module, "detect_trainer_runtime") as detect_runtime, patch.dict(sys.modules, {"uvicorn": fake_uvicorn}):
            detect_runtime.return_value = type("_Runtime", (), {"to_dict": lambda self: dict(runtime_snapshot)})()
            result = trainer.train_result(
                method="qlora",
                epochs=2,
                base_model="mock-llama-target",
                train_type="sft",
                backend_hint="peft",
            )

            pipeline.trainer = trainer
            status_snapshot = pipeline.status()
            trainer_plan = status_snapshot["trainer"]["plans"]["sft"]
            self.assertEqual(trainer_plan["recommended_backend"], result.backend_plan["recommended_backend"])
            self.assertEqual(trainer_plan["requires_export_step"], result.backend_plan["requires_export_step"])

            serve_plan = build_serve_plan(workspace=str(self.pfe_home), dry_run=False)
            self.assertTrue(serve_plan.runtime.uvicorn_available)
            self.assertFalse(serve_plan.runtime.dry_run)
            self.assertEqual(serve_plan.runtime.app_target, "pfe_server.app:app")

            serve_output = serve(workspace=str(self.pfe_home), dry_run=False)
            self.assertEqual(serve_output, "started uvicorn on 127.0.0.1:8921")
            self.assertEqual(len(run_calls), 1)
            self.assertEqual(run_calls[0]["host"], "127.0.0.1")
            self.assertEqual(run_calls[0]["port"], 8921)
            self.assertFalse(run_calls[0]["reload"])

        self.assertEqual(result.execution_backend, result.backend_dispatch["execution_backend"])
        self.assertEqual(result.training_config["execution_backend"], result.execution_backend)
        self.assertIn("export_write", result.audit_info)
        self.assertTrue(result.audit_info["export_write"]["metadata"]["materialized"])
        self.assertEqual(result.audit_info["export_write"]["metadata"]["written_count"], 3)
        self.assertEqual(result.training_config["export_write"]["metadata"]["written_count"], 3)

        export_failure_message = None
        try:
            write_materialized_export_plan(
                target_backend="peft",
                source_artifact_format="peft_lora",
                adapter_dir=None,
            )
        except ValueError as exc:
            export_failure_message = str(exc)
        self.assertIsNotNone(export_failure_message)
        self.assertIn("adapter_dir is required", export_failure_message or "")

    def test_train_status_and_real_serve_probe_share_export_and_runtime_signals(self) -> None:
        pipeline = PipelineService()
        pipeline.generate(scenario="life-coach", style="温和、共情", num_samples=8)

        version_dir = self.pfe_home / "adapters" / "user_default" / "20260323-995"
        fake_store = _FakeTrainerStore(version_dir)
        trainer = TrainerService(store=fake_store)
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
        fake_uvicorn = ModuleType("uvicorn")
        run_calls: list[dict[str, object]] = []

        def fake_run(app: object, **kwargs: object) -> None:
            run_calls.append({"app": app, **kwargs})

        fake_uvicorn.run = fake_run  # type: ignore[attr-defined]

        class _TrainingRunResultStub:
            def __init__(self, **kwargs: object):
                self.__dict__.update(kwargs)

        with patch.object(trainer_service_module, "detect_trainer_runtime") as detect_runtime, patch.dict(sys.modules, {"uvicorn": fake_uvicorn}):
            detect_runtime.return_value = type("_Runtime", (), {"to_dict": lambda self: dict(runtime_snapshot)})()
            with patch.object(trainer_service_module, "TrainingRunResult", _TrainingRunResultStub):
                trainer._resolve_training_executor = lambda **kwargs: {  # type: ignore[assignment]
                    "execution_backend": kwargs["backend_dispatch"]["execution_backend"],
                    "execution_executor": kwargs["backend_dispatch"]["execution_backend"],
                    "execution_mode": "real_import",
                    "executor_mode": "real_import",
                }
                trainer._simulate_backend_execution = lambda **kwargs: {  # type: ignore[assignment]
                    "state": "ok",
                    "branch": kwargs["backend_dispatch"]["execution_backend"],
                    "executor_mode": kwargs["backend_dispatch"]["executor_mode"],
                    "execution_backend": kwargs["backend_dispatch"]["execution_backend"],
                }
                train_result = trainer.train_result(
                    method="qlora",
                    epochs=1,
                    base_model="mock-llama-target",
                    train_type="sft",
                    backend_hint="peft",
                )
                pipeline.trainer = trainer

                serve_plan = build_serve_plan(workspace=str(self.pfe_home), dry_run=False)
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

                self.assertEqual(train_result.execution_backend, train_result.backend_dispatch["execution_backend"])
                self.assertEqual(train_result.training_config["execution_backend"], train_result.execution_backend)
                self.assertTrue(train_result.audit_info["export_write"]["metadata"]["materialized"])
                self.assertEqual(train_result.audit_info["export_write"]["metadata"]["written_count"], 3)
                self.assertEqual(train_result.training_config["export_write"]["metadata"]["written_count"], 3)
                self.assertEqual(body["metadata"]["server_runtime"]["launch_mode"], serve_plan.runtime_probe["launch_mode"])
                self.assertEqual(body["metadata"]["server_runtime"]["command"], serve_plan.runtime_probe["command"])
                self.assertEqual(body["metadata"]["server_runtime"]["runner"]["kind"], serve_plan.runtime_probe["runner"]["kind"])
                self.assertEqual(body["metadata"]["export"]["write_state"], "materialized")
                self.assertEqual(body["metadata"]["export"]["materialized"], True)
                self.assertEqual(body["metadata"]["export"]["artifact_directory"], body["metadata"]["trainer"]["output_dir"])
                self.assertEqual(body["metadata"]["trainer"]["recommended_backend"], body["metadata"]["export"]["recommended_backend"])
                self.assertEqual(body["metadata"]["trainer"]["requires_export_step"], body["metadata"]["export"]["requires_export_step"])
                self.assertEqual(
                    body["metadata"]["trainer"]["export_artifact_summary"]["path"],
                    train_result.training_config["export_artifact_summary"]["path"],
                )
                self.assertIn("export_artifact_path", body["metadata"]["export"])
                self.assertEqual(body["metadata"]["pipeline"]["trainer"]["last_run"]["execution_backend"], train_result.execution_backend)
                self.assertEqual(body["metadata"]["pipeline"]["trainer"]["last_run"]["export_write"]["metadata"]["written_count"], 3)
                self.assertEqual(len(run_calls), 0)

                serve_output = serve(workspace=str(self.pfe_home), dry_run=False)
                self.assertEqual(serve_output, "started uvicorn on 127.0.0.1:8921")
                self.assertEqual(len(run_calls), 1)
                self.assertEqual(run_calls[0]["host"], "127.0.0.1")
                self.assertEqual(run_calls[0]["port"], 8921)
                self.assertFalse(run_calls[0]["reload"])


if __name__ == "__main__":
    unittest.main()
