from __future__ import annotations

import importlib
import json
import os
import tempfile
import unittest
from email.message import Message
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pfe_core.errors import TrainingError  # noqa: E402
from pfe_core.config import PFEConfig  # noqa: E402
from pfe_core.pipeline import PipelineService  # noqa: E402
from pfe_core.trainer import detect_trainer_runtime, plan_trainer_backend, trainer_runtime_summary  # noqa: E402
from pfe_core.trainer.service import TrainerService  # noqa: E402

trainer_runtime_module = importlib.import_module("pfe_core.trainer.runtime")
trainer_executor_module = importlib.import_module("pfe_core.trainer.executors")

trainer_service_module = importlib.import_module("pfe_core.trainer.service")

class _NoopTrainerStore:
    def __init__(self, version_dir: Path):
        self.version_dir = version_dir
        self.created_training_config: dict[str, object] | None = None
        self.pending_eval_calls: list[dict[str, object]] = []
        self.failed_eval_calls: list[dict[str, object]] = []

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
        return {"version": "20260323-999", "path": str(self.version_dir), "manifest": {"version": "20260323-999"}}

    def mark_pending_eval(self, version: str, *, num_samples: int, metrics: dict[str, object] | None = None) -> None:
        self.pending_eval_calls.append({"version": version, "num_samples": num_samples, "metrics": dict(metrics or {})})

    def mark_failed_eval(self, version: str, report: dict[str, object] | None = None) -> None:
        self.failed_eval_calls.append({"version": version, "report": dict(report or {})})


def _successful_peft_job_execution(version_dir: Path, result_json_path: str | None = None) -> dict[str, object]:
    artifact_dir = version_dir / "real-peft-artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    adapter_model = artifact_dir / "adapter_model.safetensors"
    adapter_config = artifact_dir / "adapter_config.json"
    real_execution_path = artifact_dir / "real_execution.json"
    adapter_model.write_text("adapter weights\n", encoding="utf-8")
    adapter_config.write_text(
        json.dumps(
            {
                "base_model_name_or_path": "mock-llama-target",
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    real_execution_path.write_text(
        json.dumps({"status": "completed", "train_loss": 0.125}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    payload = {
        "attempted": True,
        "success": True,
        "status": "executed",
        "command": [],
        "returncode": 0,
        "exit_code": 0,
        "stdout": "",
        "stderr": "",
        "runner_result": {
            "backend": "peft",
            "status": "completed",
            "execution_mode": "real_import",
            "real_execution": {
                "kind": "peft",
                "path": "real_import",
                "runtime_path": "real_import",
                "attempted": True,
                "available": True,
                "success": True,
                "train_loss": 0.125,
                "num_examples": 8,
                "artifact_dir": str(artifact_dir),
                "artifacts": {
                    "adapter_model": str(adapter_model),
                    "adapter_config": str(adapter_config),
                },
                "real_execution_path": str(real_execution_path),
            },
            "job_spec": {},
        },
        "audit": {"status": "executed"},
        "metadata": {"execution_state": "executed"},
    }
    if result_json_path:
        Path(result_json_path).write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
    return payload

class TrainerRuntimeTests(unittest.TestCase):
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

    def test_runtime_planner_is_exported_and_serializable(self) -> None:
        runtime = detect_trainer_runtime().to_dict()
        plan = plan_trainer_backend(
            train_type="sft",
            runtime=runtime,
            target_inference_backend="llama_cpp",
        )
        summary = trainer_runtime_summary()

        self.assertIn("platform_name", runtime)
        self.assertIn("recommended_backend", plan.to_dict())
        self.assertIn("capabilities", summary)

    def test_runtime_uses_package_metadata_for_versions_and_python_compatibility(self) -> None:
        probed_packages = {"torch", "transformers", "peft"}
        package_versions = {
            "torch": "2.5.0",
            "transformers": "4.48.0",
            "peft": "0.12.0",
        }
        package_requires_python = {
            "torch": ">=3.0",
            "transformers": ">=3.0,<4.0",
            "peft": ">=3.0",
        }

        original_import = __import__

        def guarded_import(name: str, globals=None, locals=None, fromlist=(), level=0):  # type: ignore[no-untyped-def]
            if name in probed_packages:
                raise AssertionError(f"unexpected import for version probing: {name}")
            return original_import(name, globals, locals, fromlist, level)

        def fake_metadata(name: str) -> Message:
            meta = Message()
            meta["Requires-Python"] = package_requires_python[name]
            return meta

        with patch.object(trainer_runtime_module, "_probe_package", side_effect=lambda name: name in probed_packages), patch.object(
            trainer_runtime_module,
            "_torch_cuda_available",
            return_value=False,
        ), patch.object(trainer_runtime_module, "_torch_mps_available", return_value=False), patch.object(
            trainer_runtime_module,
            "_python_supported_for_requires_python",
            side_effect=lambda requirement: requirement != ">=3.0,<4.0",
        ), patch.object(
            trainer_runtime_module.importlib_metadata,
            "version",
            side_effect=lambda name: package_versions[name],
        ), patch.object(
            trainer_runtime_module.importlib_metadata,
            "metadata",
            side_effect=fake_metadata,
        ), patch("builtins.__import__", side_effect=guarded_import):
            runtime = trainer_runtime_module.detect_trainer_runtime().to_dict()

        self.assertEqual(runtime["dependency_versions"]["torch"], "2.5.0")
        self.assertEqual(runtime["dependency_versions"]["transformers"], "4.48.0")
        self.assertEqual(runtime["requires_python"]["torch"], ">=3.0")
        self.assertEqual(runtime["requires_python"]["transformers"], ">=3.0,<4.0")
        self.assertTrue(runtime["python_supported"]["torch"])
        self.assertFalse(runtime["python_supported"]["transformers"])
        self.assertEqual(runtime["installed_packages"]["peft"], True)

    def test_train_result_records_runtime_backend_plan_and_export_audit(self) -> None:
        pipeline = PipelineService()
        pipeline.generate(scenario="life-coach", style="温和、共情", num_samples=8)

        version_dir = self.pfe_home / "adapters" / "user_default" / "20260323-999"
        store = _NoopTrainerStore(version_dir)
        trainer = TrainerService(store=store)

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
        fake_modules = {
            "torch": SimpleNamespace(nn=object()),
            "transformers": SimpleNamespace(Trainer=object(), TrainingArguments=object()),
            "peft": SimpleNamespace(LoraConfig=object(), get_peft_model=lambda *args, **kwargs: None),
            "accelerate": SimpleNamespace(Accelerator=object()),
        }

        def fake_find_spec(name: str):
            if name in fake_modules:
                return object()
            return None

        def fake_import_module(name: str):
            return fake_modules[name]

        with patch.dict(
            os.environ,
            {
                "PFE_LLAMA_CPP_EXPORT_TOOL": "/tmp/does-not-exist-llama-export-tool",
                "PFE_REAL_TRAINING": "1",
            },
        ), patch.object(
            trainer_service_module, "detect_trainer_runtime"
        ) as detect_runtime, patch.object(
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
            side_effect=lambda bundle: SimpleNamespace(
                to_dict=lambda: _successful_peft_job_execution(version_dir, bundle.result_json_path)
            ),
        ):
            detect_runtime.return_value = type("_Runtime", (), {"to_dict": lambda self: dict(runtime_snapshot)})()
            result = trainer.train_result(
                method="qlora",
                epochs=2,
                base_model="mock-llama-target",
                train_type="sft",
                backend_hint="peft",
            )

        self.assertEqual(result.version, "20260323-999")
        self.assertIn("runtime", result.metrics)
        self.assertIn("backend_plan", result.metrics)
        self.assertIn("backend_dispatch", result.metrics)
        self.assertTrue(hasattr(result, "audit_info"))
        self.assertIn("runtime_device", result.runtime)
        self.assertIn("recommended_backend", result.backend_plan)
        self.assertEqual(result.execution_backend, "peft")
        self.assertEqual(result.execution_executor, "peft")
        self.assertEqual(result.executor_mode, "real_import")
        self.assertEqual(result.backend_dispatch["execution_backend"], "peft")
        self.assertEqual(result.backend_dispatch["dispatch_mode"], "requested")
        self.assertEqual(result.backend_dispatch["executor_mode"], "real_import")
        self.assertIn("executor_spec", result.metrics)
        self.assertIn("execution_summary", result.training_config)
        self.assertEqual(result.training_config["backend_dispatch"]["execution_backend"], "peft")
        self.assertEqual(result.training_config["execution_executor"], "peft")
        self.assertEqual(result.training_config["executor_mode"], "real_import")
        self.assertEqual(result.executor_spec["execution_executor"], "peft")
        self.assertEqual(result.executor_spec["executor_mode"], "real_import")
        self.assertIn("job_bundle", result.metrics)
        self.assertIn("job_execution", result.metrics)
        self.assertEqual(result.job_execution["status"], "executed")
        self.assertEqual(result.job_execution["runner_result"]["backend"], "peft")
        self.assertTrue(Path(result.job_bundle["result_json_path"]).exists())
        self.assertIn("dry_run", result.export_runtime)
        self.assertIn("manifest_updates", result.export_runtime)
        self.assertEqual(result.export_command_plan["target_backend"], "transformers")
        self.assertEqual(result.export_command_plan["status"], "not_required")
        self.assertEqual(result.export_command_plan["target_artifact_format"], "peft_lora")
        self.assertIn("audit", result.export_execution)
        self.assertEqual(result.export_execution["audit"]["status"], "not_required")
        self.assertIn("requires_export_step", result.audit_info)
        self.assertEqual(store.created_training_config["backend"], "peft")
        self.assertEqual(store.created_training_config["requested_backend"], "peft")
        self.assertEqual(store.created_training_config["runtime"]["runtime_device"], result.runtime["runtime_device"])
        self.assertEqual(
            store.created_training_config["requires_export_step"],
            result.backend_plan["requires_export_step"],
        )
        self.assertEqual(result.training_config["export_runtime"]["artifact_directory"], result.export_runtime["artifact_directory"])

        training_meta = json.loads((version_dir / "training_meta.json").read_text(encoding="utf-8"))
        self.assertIn("runtime", training_meta)
        self.assertIn("backend_plan", training_meta)
        self.assertIn("backend_dispatch", training_meta)
        self.assertIn("executor_spec", training_meta)
        self.assertIn("job_bundle", training_meta)
        self.assertIn("job_execution", training_meta)
        self.assertIn("export_runtime", training_meta)
        self.assertIn("export_command_plan", training_meta)
        self.assertIn("export_execution", training_meta)
        self.assertIn("export_artifact_summary", training_meta)
        self.assertIn("audit", training_meta)
        self.assertEqual(training_meta["execution_backend"], "peft")
        self.assertEqual(training_meta["execution_executor"], "peft")
        self.assertEqual(training_meta["executor_mode"], "real_import")
        self.assertEqual(training_meta["executor_spec"]["execution_executor"], "peft")
        self.assertEqual(training_meta["backend_dispatch"]["execution_backend"], "peft")
        self.assertEqual(training_meta["audit"]["requires_export_step"], result.audit_info["requires_export_step"])
        self.assertEqual(training_meta["job_execution"]["runner_result"]["backend"], "peft")
        self.assertTrue(Path(training_meta["job_bundle"]["result_json_path"]).exists())
        self.assertEqual(training_meta["backend_plan"]["recommended_backend"], result.backend_plan["recommended_backend"])
        self.assertEqual(training_meta["export_runtime"]["dry_run"], True)
        self.assertEqual(training_meta["export_execution"]["audit"]["status"], "not_required")
        self.assertEqual(training_meta["export_artifact_summary"]["status"], result.export_execution["audit"]["status"])

    def test_train_result_defaults_base_model_from_initialized_config(self) -> None:
        config = PFEConfig()
        config.model.base_model = "/models/init-default"
        config.save(home=self.pfe_home)

        pipeline = PipelineService()
        pipeline.generate(scenario="life-coach", style="温和、共情", num_samples=8)

        version_dir = self.pfe_home / "adapters" / "user_default" / "20260323-999"
        store = _NoopTrainerStore(version_dir)
        trainer = TrainerService(store=store)

        result = trainer.train_result(
            method="qlora",
            epochs=1,
            train_type="sft",
            backend_hint="mock_local",
        )

        self.assertEqual(result.training_config["executor_recipe"]["training"]["base_model"], "/models/init-default")
        self.assertEqual(store.created_training_config["execution_summary"]["base_model"], "/models/init-default")
        manifest = json.loads((version_dir / "adapter_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["base_model"], "/models/init-default")

    def test_materialized_training_job_bundle_writes_runner_and_result_files(self) -> None:
        backend_dispatch = {
            "requested_backend": "peft",
            "execution_backend": "peft",
            "execution_executor": "peft",
            "executor_mode": "real_import",
            "reasons": ["using requested backend peft"],
            "requires_export_step": True,
            "export_steps": ["gguf_merged_export"],
            "export_format": "gguf_merged",
            "export_backend": "llama_cpp",
            "capability": {"artifact_format": "peft_lora"},
        }
        plan = trainer_executor_module.build_training_execution_recipe(
            backend_dispatch=backend_dispatch,
            runtime={"runtime_device": "cpu"},
            method="qlora",
            epochs=1,
            train_type="sft",
            base_model_name="mock-llama-target",
            num_train_samples=4,
            num_fresh_samples=3,
            num_replay_samples=1,
            replay_ratio=0.25,
            allow_mock_fallback=True,
        )

        bundle = trainer_executor_module.materialize_training_job_bundle(
            execution_plan=plan,
            output_dir=self.pfe_home / "adapters" / "user_default" / "20260323-998",
        )
        job_json_path = Path(bundle.job_json_path)
        script_path = Path(bundle.script_path)
        result_json_path = Path(bundle.result_json_path)
        failed_payload = {
            "backend": bundle.job_json["execution_backend"],
            "status": "failed",
            "error": "fixture runner failed",
        }

        def fake_run(command: list[str], **kwargs: object):
            result_json_path.write_text(json.dumps(failed_payload, sort_keys=True) + "\n", encoding="utf-8")
            return trainer_executor_module.subprocess.CompletedProcess(
                command,
                returncode=0,
                stdout=json.dumps(failed_payload, sort_keys=True) + "\n",
                stderr="",
            )

        with patch.dict(os.environ, {"PFE_REAL_TRAINING": "1"}), patch.object(
            trainer_executor_module.subprocess,
            "run",
            side_effect=fake_run,
        ):
            runner_result = trainer_executor_module.run_materialized_training_job_bundle(bundle, force_dry_run=False)

        self.assertTrue(job_json_path.exists())
        self.assertTrue(script_path.exists())
        self.assertTrue(result_json_path.exists())
        self.assertIn("trainer_job.py", str(script_path))
        self.assertIn("trainer_job.json", str(job_json_path))
        self.assertEqual(bundle.audit["materialized_files"], [str(job_json_path), str(script_path)])
        self.assertEqual(bundle.audit["result_json_path"], str(result_json_path))
        self.assertEqual(bundle.metadata["result_json_name"], result_json_path.name)
        self.assertIn("job_spec", bundle.job_json)
        expected_executor = plan["job_spec"]["execution_executor"]
        self.assertIn(expected_executor, {"mock_local", "peft"})
        self.assertEqual(bundle.job_json["execution_executor"], expected_executor)
        self.assertEqual(runner_result.status, "failed")
        self.assertTrue(runner_result.attempted)
        self.assertFalse(runner_result.success)
        self.assertEqual(runner_result.materialization["execution_executor"], expected_executor)
        self.assertEqual(runner_result.audit["status"], "failed")
        self.assertEqual(runner_result.audit["result_json_path"], str(result_json_path))
        self.assertEqual(runner_result.metadata["execution_state"], "failed")
        self.assertEqual(runner_result.metadata["result_json_path"], str(result_json_path))
        self.assertEqual(runner_result.failure_category, "runner_failed")
        result_payload = json.loads(result_json_path.read_text(encoding="utf-8"))
        self.assertIn(result_payload["backend"], {bundle.job_json["execution_backend"], "mock_local"})
        self.assertEqual(result_payload["status"], "failed")

    def test_backend_dispatch_skeleton_distinguishes_training_branches(self) -> None:
        trainer = TrainerService()
        base_plan = {
            "train_type": "sft",
            "recommended_backend": "mock_local",
            "requires_export_step": False,
            "export_steps": [],
            "export_format": None,
            "export_backend": None,
        }
        fake_modules = {
            "torch": SimpleNamespace(nn=object()),
            "transformers": SimpleNamespace(Trainer=object(), TrainingArguments=object()),
            "peft": SimpleNamespace(LoraConfig=object(), get_peft_model=lambda *args, **kwargs: None),
            "accelerate": SimpleNamespace(Accelerator=object()),
            "unsloth": SimpleNamespace(FastLanguageModel=object()),
        }
        imported_names: list[str] = []

        def fake_find_spec(name: str):
            if name in fake_modules or name in {"mlx", "mlx_lm"}:
                return object()
            return None

        def fake_import_module(name: str):
            if name in {"mlx", "mlx_lm"}:
                raise AssertionError(f"{name} should not be hard-imported during dispatch")
            imported_names.append(name)
            return fake_modules[name]

        with patch.object(
            trainer_service_module.importlib.util,
            "find_spec",
            side_effect=fake_find_spec,
        ), patch.object(
            trainer_service_module.importlib,
            "import_module",
            side_effect=fake_import_module,
        ):
            peft_dispatch = trainer._dispatch_training_backend(
                backend_plan=base_plan,
                runtime={
                    "installed_packages": {"torch": True, "transformers": True, "peft": True, "accelerate": True},
                    "cuda_available": False,
                    "mps_available": False,
                },
                backend_hint="peft",
            )
            unsloth_dispatch = trainer._dispatch_training_backend(
                backend_plan=base_plan,
                runtime={
                    "installed_packages": {"torch": True, "transformers": True, "unsloth": True},
                    "cuda_available": True,
                    "mps_available": False,
                },
                backend_hint="unsloth",
            )
            mlx_dispatch = trainer._dispatch_training_backend(
                backend_plan=base_plan,
                runtime={
                    "installed_packages": {"mlx": True, "mlx_lm": True},
                    "cuda_available": False,
                    "mps_available": True,
                },
                backend_hint="mlx",
            )
            fallback_dispatch = trainer._dispatch_training_backend(
                backend_plan=base_plan,
                runtime={"installed_packages": {}},
                backend_hint="unknown-backend",
            )

        self.assertIn(peft_dispatch["execution_backend"], {"peft", "mock_local"})
        self.assertIn(peft_dispatch["execution_executor"], {"peft", "mock_local"})
        self.assertIn(peft_dispatch["dispatch_mode"], {"requested", "fallback"})
        self.assertIn(unsloth_dispatch["execution_backend"], {"unsloth", "mock_local"})
        self.assertIn(unsloth_dispatch["execution_executor"], {"unsloth", "mock_local"})
        self.assertIn(unsloth_dispatch["dispatch_mode"], {"requested", "fallback"})
        self.assertEqual(mlx_dispatch["execution_backend"], "mlx")
        self.assertEqual(mlx_dispatch["execution_executor"], "mlx")
        self.assertEqual(mlx_dispatch["dispatch_mode"], "requested")
        self.assertEqual(mlx_dispatch["executor_mode"], "deferred_import")
        self.assertTrue(any(item.get("deferred_import") for item in mlx_dispatch["import_attempts"]))
        self.assertEqual(fallback_dispatch["execution_backend"], "mock_local")
        self.assertEqual(fallback_dispatch["execution_executor"], "mock_local")
        self.assertEqual(fallback_dispatch["dispatch_mode"], "fallback")
        self.assertIn("unknown backend hint", fallback_dispatch["reasons"][0])
        self.assertNotIn("mlx", imported_names)
        self.assertNotIn("mlx_lm", imported_names)
        if peft_dispatch["execution_backend"] == "mock_local":
            self.assertEqual(peft_dispatch["dispatch_mode"], "fallback")
        if unsloth_dispatch["execution_backend"] == "mock_local":
            self.assertEqual(unsloth_dispatch["dispatch_mode"], "fallback")

    def test_peft_executor_without_imports_raises_when_fallback_disabled(self) -> None:
        trainer = TrainerService()
        backend_dispatch = {
            "requested_backend": "peft",
            "execution_backend": "peft",
            "dispatch_mode": "requested",
            "available": {"peft": False},
            "capability": {"artifact_format": "peft_lora"},
            "reasons": ["peft dependencies unavailable"],
            "requires_export_step": False,
            "export_steps": [],
            "export_format": None,
            "export_backend": None,
        }

        with patch.object(trainer_service_module.importlib.util, "find_spec", return_value=None):
            with self.assertRaises(TrainingError):
                trainer._resolve_training_executor(
                    backend_dispatch=backend_dispatch,
                    runtime={"installed_packages": {}},
                    backend_hint="peft",
                    allow_mock_fallback=False,
                )

    def test_peft_executor_missing_imports_keeps_selected_backend_but_records_mock_fallback(self) -> None:
        pipeline = PipelineService()
        pipeline.generate(scenario="life-coach", style="温和、共情", num_samples=8)

        version_dir = self.pfe_home / "adapters" / "user_default" / "20260323-998"
        store = _NoopTrainerStore(version_dir)
        trainer = TrainerService(store=store)

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

        with patch.object(trainer_service_module, "detect_trainer_runtime") as detect_runtime, patch.object(
            trainer_service_module.importlib.util,
            "find_spec",
            return_value=None,
        ), patch.object(trainer_service_module.importlib, "import_module", side_effect=ImportError("missing dependency")):
            base_gguf = Path(self.tempdir.name) / "base-model.gguf"
            base_gguf.write_text("GGUF", encoding="utf-8")
            detect_runtime.return_value = type("_Runtime", (), {"to_dict": lambda self: dict(runtime_snapshot)})()
            result = trainer.train_result(
                method="qlora",
                epochs=1,
                base_model=str(base_gguf),
                train_type="sft",
                backend_hint="peft",
            )

        self.assertEqual(result.execution_backend, "peft")
        self.assertEqual(result.execution_executor, "mock_local")
        self.assertEqual(result.executor_mode, "fallback")
        self.assertEqual(result.backend_dispatch["execution_backend"], "peft")
        self.assertEqual(result.executor_spec["execution_backend"], "peft")
        self.assertEqual(result.executor_spec["execution_executor"], "mock_local")
        self.assertEqual(result.executor_spec["fallback_from"], "peft")
        self.assertEqual(result.export_runtime["artifact_directory"], "gguf_merged")
        self.assertEqual(result.export_write["plan"]["artifact_directory"], "gguf_merged")
        training_meta = json.loads((version_dir / "training_meta.json").read_text(encoding="utf-8"))
        self.assertEqual(training_meta["execution_backend"], "peft")
        self.assertEqual(training_meta["execution_executor"], "mock_local")
        self.assertEqual(training_meta["executor_mode"], "fallback")
        self.assertEqual(training_meta["export_write"]["plan"]["artifact_directory"], "gguf_merged")

    def test_llama_cpp_export_execution_runs_fake_tool_when_available(self) -> None:
        pipeline = PipelineService()
        pipeline.generate(scenario="life-coach", style="温和、共情", num_samples=8)

        version_dir = self.pfe_home / "adapters" / "user_default" / "20260323-997"
        store = _NoopTrainerStore(version_dir)
        trainer = TrainerService(store=store)

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
        fake_modules = {
            "torch": SimpleNamespace(nn=object()),
            "transformers": SimpleNamespace(Trainer=object(), TrainingArguments=object()),
            "peft": SimpleNamespace(LoraConfig=object(), get_peft_model=lambda *args, **kwargs: None),
            "accelerate": SimpleNamespace(Accelerator=object()),
        }

        def fake_find_spec(name: str):
            if name in fake_modules:
                return object()
            return None

        def fake_import_module(name: str):
            return fake_modules[name]

        with tempfile.TemporaryDirectory() as tool_dir:
            base_gguf = Path(tool_dir) / "base-model.gguf"
            base_gguf.write_text("GGUF", encoding="utf-8")
            tool_path = Path(tool_dir) / "fake-llama-export.sh"
            tool_path.write_text(
                "#!/bin/sh\n"
                "outdir=\"\"\n"
                "outname=\"adapter_model.gguf\"\n"
                "while [ $# -gt 0 ]; do\n"
                "  if [ \"$1\" = \"--output-dir\" ]; then outdir=\"$2\"; shift 2; continue; fi\n"
                "  if [ \"$1\" = \"--output-name\" ]; then outname=\"$2\"; shift 2; continue; fi\n"
                "  shift\n"
                "done\n"
                "mkdir -p \"$outdir\"\n"
                "printf 'gguf output\\n' > \"$outdir/$outname\"\n"
                "echo \"$@\"\n"
                "exit 0\n",
                encoding="utf-8",
            )
            tool_path.chmod(0o755)

            with patch.dict(os.environ, {"PFE_LLAMA_CPP_EXPORT_TOOL": str(tool_path)}), patch.object(
                trainer_service_module, "detect_trainer_runtime"
            ) as detect_runtime, patch.object(
                trainer_service_module.importlib.util,
                "find_spec",
                side_effect=fake_find_spec,
            ), patch.object(trainer_service_module.importlib, "import_module", side_effect=fake_import_module), patch.object(
                trainer_service_module,
                "run_materialized_training_job_bundle",
                side_effect=lambda bundle: SimpleNamespace(
                    to_dict=lambda: _successful_peft_job_execution(version_dir, bundle.result_json_path)
                ),
            ):
                detect_runtime.return_value = type("_Runtime", (), {"to_dict": lambda self: dict(runtime_snapshot)})()
                result = trainer.train_result(
                    method="qlora",
                    epochs=1,
                    base_model=str(base_gguf),
                    train_type="sft",
                    backend_hint="peft",
                )

        self.assertEqual(result.export_command_plan["tool_resolution"]["resolved_path"], str(tool_path))
        self.assertTrue(result.export_execution["attempted"])
        self.assertTrue(result.export_execution["success"])
        self.assertIn(result.export_execution["audit"]["status"], {"success", "executed"})
        self.assertTrue(result.export_execution["output_artifact_validation"]["valid"])
        self.assertEqual(result.export_write["write_state"], "validated")

    def test_local_qwen_hf_model_keeps_transformers_peft_artifact(self) -> None:
        pipeline = PipelineService()
        pipeline.generate(scenario="life-coach", style="温和、共情", num_samples=8)

        version_dir = self.pfe_home / "adapters" / "user_default" / "20260323-996"
        store = _NoopTrainerStore(version_dir)
        trainer = TrainerService(store=store)

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
        fake_modules = {
            "torch": SimpleNamespace(nn=object()),
            "transformers": SimpleNamespace(Trainer=object(), TrainingArguments=object()),
            "peft": SimpleNamespace(LoraConfig=object(), get_peft_model=lambda *args, **kwargs: None),
            "accelerate": SimpleNamespace(Accelerator=object()),
        }

        def fake_find_spec(name: str):
            if name in fake_modules:
                return object()
            return None

        def fake_import_module(name: str):
            return fake_modules[name]

        local_qwen_dir = Path(self.tempdir.name) / "Qwen3-4B"
        local_qwen_dir.mkdir(parents=True, exist_ok=True)
        (local_qwen_dir / "config.json").write_text(
            json.dumps({"model_type": "qwen2", "vocab_size": 128}) + "\n",
            encoding="utf-8",
        )

        with tempfile.TemporaryDirectory() as tool_dir:
            tool_path = Path(tool_dir) / "convert_lora_to_gguf"
            tool_path.write_text(
                "#!/bin/sh\n"
                "outfile=\"\"\n"
                "adapter_dir=\"\"\n"
                "while [ $# -gt 0 ]; do\n"
                "  if [ \"$1\" = \"--outfile\" ]; then outfile=\"$2\"; shift 2; continue; fi\n"
                "  case \"$1\" in\n"
                "    --*) shift 2; continue ;;\n"
                "  esac\n"
                "  adapter_dir=\"$1\"\n"
                "  shift\n"
                "done\n"
                "test -f \"$adapter_dir/adapter_model.safetensors\" || exit 31\n"
                "test -f \"$adapter_dir/adapter_config.json\" || exit 32\n"
                "mkdir -p \"$(dirname \"$outfile\")\"\n"
                "printf 'GGUF' > \"$outfile\"\n"
                "exit 0\n",
                encoding="utf-8",
            )
            tool_path.chmod(0o755)

            with patch.dict(os.environ, {"PFE_LLAMA_CPP_EXPORT_TOOL": str(tool_path), "PFE_REAL_TRAINING": "1"}), patch.object(
                trainer_service_module, "detect_trainer_runtime"
            ) as detect_runtime, patch.object(
                trainer_service_module.importlib.util,
                "find_spec",
                side_effect=fake_find_spec,
            ), patch.object(trainer_service_module.importlib, "import_module", side_effect=fake_import_module), patch.object(
                trainer_service_module,
                "run_materialized_training_job_bundle",
                side_effect=lambda bundle: SimpleNamespace(
                    to_dict=lambda: _successful_peft_job_execution(version_dir, bundle.result_json_path)
                ),
            ):
                detect_runtime.return_value = type("_Runtime", (), {"to_dict": lambda self: dict(runtime_snapshot)})()
                result = trainer.train_result(
                    method="qlora",
                    epochs=1,
                    base_model=str(local_qwen_dir),
                    train_type="sft",
                    backend_hint="peft",
                )

        self.assertEqual(result.export_runtime["target_backend"], "transformers")
        self.assertEqual(result.export_runtime["target_artifact_format"], "peft_lora")
        self.assertFalse(result.export_execution["attempted"])
        self.assertTrue(result.export_execution["success"])
        self.assertTrue(result.training_config["pre_export_artifact_sync"]["available"])

if __name__ == "__main__":
    unittest.main()
