from __future__ import annotations

import os
import unittest
from pathlib import Path

from typer.testing import CliRunner

ROOT = Path(__file__).resolve().parents[1]
for package_dir in ("pfe-core", "pfe-cli", "pfe-server"):
    package_path = str(ROOT / package_dir)
    if package_path not in os.sys.path:
        os.sys.path.insert(0, package_path)

from pfe_cli import main as cli_main


class CLIDoctorTests(unittest.TestCase):
    def test_doctor_command_reports_compact_readiness_summary(self) -> None:
        runner = CliRunner()
        original_optional_call = cli_main._optional_module_call
        original_load_latest_adapter_manifest = cli_main._load_latest_adapter_manifest
        original_lookup_latest = cli_main._lookup_adapter_snapshot
        original_lookup_recent = cli_main._lookup_recent_adapter_snapshot
        try:
            cli_main._load_latest_adapter_manifest = lambda workspace=None: {"base_model": "local-model-or-base"}

            def fake_optional_module_call(module_name: str, attr_name: str, *args: object, **kwargs: object):
                if module_name == "pfe_core.trainer.runtime" and attr_name == "detect_trainer_runtime":
                    return {
                        "python_version": "3.11.8",
                        "runtime_device": "cpu",
                        "installed_packages": {
                            "torch": True,
                            "transformers": True,
                            "peft": True,
                            "accelerate": True,
                            "trl": True,
                            "datasets": True,
                            "unsloth": False,
                            "mlx": False,
                            "mlx_lm": False,
                        },
                        "requires_python": {
                            "torch": ">=3.0",
                            "transformers": ">=3.0,<4.0",
                            "peft": ">=3.0",
                            "accelerate": ">=3.0",
                            "trl": ">=3.0",
                            "datasets": ">=3.0",
                        },
                        "python_supported": {
                            "torch": True,
                            "transformers": True,
                            "peft": True,
                            "accelerate": True,
                            "trl": True,
                            "datasets": True,
                        },
                    }
                if module_name == "pfe_core.trainer.executors" and attr_name == "_resolve_real_local_model_source":
                    return {
                        "available": True,
                        "requested_base_model": "local-model-or-base",
                        "source_kind": "path",
                        "source_path": "/models/local/base",
                        "config_path": None,
                        "load_mode": "from_pretrained",
                        "reason": "local model path resolved",
                    }
                if module_name == "pfe_core.inference.export_runtime" and attr_name == "resolve_llama_cpp_export_tool_path":
                    return {
                        "resolved_path": "/usr/local/bin/llama_export_tool",
                        "exists": True,
                        "executable": True,
                        "source": "which",
                        "reason": "resolved via PATH lookup",
                    }
                if module_name == "pfe_core.inference.export_runtime" and attr_name == "validate_llama_cpp_export_toolchain":
                    return {
                        "status": "planned",
                        "allowed": True,
                        "resolved_path": "/usr/local/bin/llama_export_tool",
                        "reason": "llama.cpp export tool is ready",
                    }
                return original_optional_call(module_name, attr_name, *args, **kwargs)

            cli_main._optional_module_call = fake_optional_module_call
            cli_main._lookup_adapter_snapshot = (
                lambda version, workspace=None: {
                    "version": "20260323-005",
                    "state": "promoted",
                    "latest": True,
                    "export_status": "executed",
                    "export_write_state": "validated",
                    "export_artifact_valid": True,
                    "export_artifact_exists": True,
                    "export_artifact_size_bytes": 11534336,
                    "export_artifact_path": "/tmp/pfe/adapters/user_default/20260323-005/gguf_merged/adapter_model.gguf",
                }
                if version == "latest"
                else None
            )
            cli_main._lookup_recent_adapter_snapshot = (
                lambda workspace=None: {
                    "version": "20260323-006",
                    "state": "pending_eval",
                    "latest": False,
                    "export_status": "executed",
                    "export_write_state": "validated",
                    "export_artifact_valid": True,
                    "export_artifact_exists": True,
                    "export_artifact_size_bytes": 11534336,
                    "export_artifact_path": "/tmp/pfe/adapters/user_default/20260323-006/gguf_merged/adapter_model.gguf",
                }
            )

            result = runner.invoke(cli_main.app, ["doctor", "--workspace", "user_default"])
        finally:
            cli_main._optional_module_call = original_optional_call
            cli_main._load_latest_adapter_manifest = original_load_latest_adapter_manifest
            cli_main._lookup_adapter_snapshot = original_lookup_latest
            cli_main._lookup_recent_adapter_snapshot = original_lookup_recent

        self.assertEqual(result.exit_code, 0, msg=result.stdout)
        text = result.stdout
        self.assertIn("PFE doctor", text)
        self.assertIn("trainer deps: ready=yes", text)
        self.assertIn("torch=yes", text)
        self.assertIn("transformers=yes", text)
        self.assertIn("python_version=", text)
        self.assertIn("requires_python=torch=>=3.0", text)
        self.assertIn("python_supported=torch=yes", text)
        self.assertIn("runtime_device=cpu", text)
        self.assertIn(
            "local model: available=yes | requested_base_model=local-model-or-base | source_kind=path | source_path=/models/local/base | load_mode=from_pretrained",
            text,
        )
        self.assertIn(
            "llama.cpp export tool: status=planned | allowed=yes | resolved_path=/usr/local/bin/llama_export_tool | reason=llama.cpp export tool is ready",
            text,
        )
        self.assertIn("blocked capabilities: none", text)
        self.assertIn("next steps: run pfe train or pfe eval as needed", text)
        self.assertIn(
            "capability boundaries: train/core | eval/core | serve/core | generate/heuristic | distill/heuristic | profile/heuristic | route/heuristic",
            text,
        )
        self.assertIn("user modeling: runtime=user_memory | analysis=user_profile", text)
        self.assertIn(
            "adapter home: home=",
            text,
        )
        self.assertIn("latest promoted=version=20260323-005 | state=promoted | latest=yes", text)
        self.assertIn("recent training=version=20260323-006 | state=pending_eval | latest=no", text)
        self.assertIn(
            "latest export artifact: status=executed | write_state=validated | valid=yes | exists=yes | size=11.0MB | path=/tmp/pfe/adapters/user_default/20260323-005/gguf_merged/adapter_model.gguf",
            text,
        )
        self.assertIn(
            "recent export artifact: status=executed | write_state=validated | valid=yes | exists=yes | size=11.0MB | path=/tmp/pfe/adapters/user_default/20260323-006/gguf_merged/adapter_model.gguf",
            text,
        )

    def test_doctor_command_surfaces_blocked_capabilities_without_base_model(self) -> None:
        runner = CliRunner()
        original_optional_call = cli_main._optional_module_call
        original_load_latest_adapter_manifest = cli_main._load_latest_adapter_manifest
        original_lookup_latest = cli_main._lookup_adapter_snapshot
        original_lookup_recent = cli_main._lookup_recent_adapter_snapshot
        try:
            cli_main._load_latest_adapter_manifest = lambda workspace=None: {}

            def fake_optional_module_call(module_name: str, attr_name: str, *args: object, **kwargs: object):
                if module_name == "pfe_core.trainer.runtime" and attr_name == "detect_trainer_runtime":
                    return {
                        "runtime_device": "cpu",
                        "installed_packages": {
                            "torch": True,
                            "transformers": True,
                            "peft": True,
                            "accelerate": True,
                            "trl": True,
                            "datasets": True,
                        },
                    }
                if module_name == "pfe_core.inference.export_runtime" and attr_name == "resolve_llama_cpp_export_tool_path":
                    return {
                        "resolved_path": "/usr/local/bin/llama_export_tool",
                        "exists": True,
                        "executable": True,
                        "source": "which",
                        "reason": "resolved via PATH lookup",
                    }
                if module_name == "pfe_core.inference.export_runtime" and attr_name == "validate_llama_cpp_export_toolchain":
                    return {
                        "status": "planned",
                        "allowed": True,
                        "resolved_path": "/usr/local/bin/llama_export_tool",
                        "reason": "llama.cpp export tool is ready",
                    }
                return original_optional_call(module_name, attr_name, *args, **kwargs)

            cli_main._optional_module_call = fake_optional_module_call
            cli_main._lookup_adapter_snapshot = lambda version, workspace=None: None
            cli_main._lookup_recent_adapter_snapshot = lambda workspace=None: None

            result = runner.invoke(cli_main.app, ["doctor", "--workspace", "user_default"])
        finally:
            cli_main._optional_module_call = original_optional_call
            cli_main._load_latest_adapter_manifest = original_load_latest_adapter_manifest
            cli_main._lookup_adapter_snapshot = original_lookup_latest
            cli_main._lookup_recent_adapter_snapshot = original_lookup_recent

        self.assertEqual(result.exit_code, 0, msg=result.stdout)
        text = result.stdout
        self.assertIn("local model: available=no | requested_base_model=n/a | reason=no base model configured", text)
        self.assertIn("blocked capabilities: train, eval, serve", text)
        self.assertIn(
            "next steps: set a base_model in the latest adapter manifest or pass --base-model; train an adapter to create a workspace snapshot",
            text,
        )
        self.assertIn(
            "capability boundaries: train/core | eval/core | serve/core | generate/heuristic | distill/heuristic | profile/heuristic | route/heuristic",
            text,
        )
        self.assertIn("user modeling: runtime=user_memory | analysis=user_profile", text)
        self.assertIn("adapter home: home=", text)
        self.assertIn("latest promoted=n/a", text)
        self.assertIn("recent training=n/a", text)

    def test_doctor_command_accepts_explicit_base_model_override(self) -> None:
        runner = CliRunner()
        original_optional_call = cli_main._optional_module_call
        original_load_latest_adapter_manifest = cli_main._load_latest_adapter_manifest
        original_lookup_latest = cli_main._lookup_adapter_snapshot
        original_lookup_recent = cli_main._lookup_recent_adapter_snapshot
        try:
            cli_main._load_latest_adapter_manifest = lambda workspace=None: {}

            def fake_optional_module_call(module_name: str, attr_name: str, *args: object, **kwargs: object):
                if module_name == "pfe_core.trainer.runtime" and attr_name == "detect_trainer_runtime":
                    return {
                        "runtime_device": "cpu",
                        "installed_packages": {
                            "torch": True,
                            "transformers": True,
                            "peft": True,
                            "accelerate": True,
                            "trl": True,
                            "datasets": True,
                        },
                    }
                if module_name == "pfe_core.trainer.executors" and attr_name == "_resolve_real_local_model_source":
                    payload = args[0] if args else {}
                    return {
                        "available": True,
                        "requested_base_model": payload.get("base_model"),
                        "source_kind": "path",
                        "source_path": payload.get("base_model"),
                        "config_path": None,
                        "load_mode": "from_pretrained",
                        "reason": "local model path resolved",
                    }
                return original_optional_call(module_name, attr_name, *args, **kwargs)

            cli_main._optional_module_call = fake_optional_module_call
            cli_main._lookup_adapter_snapshot = lambda version, workspace=None: None
            cli_main._lookup_recent_adapter_snapshot = lambda workspace=None: None

            result = runner.invoke(
                cli_main.app,
                [
                    "doctor",
                    "--workspace",
                    "user_default",
                    "--base-model",
                    "/Users/lichenhao/Desktop/PFE/models/Qwen3-4B",
                ],
            )
        finally:
            cli_main._optional_module_call = original_optional_call
            cli_main._load_latest_adapter_manifest = original_load_latest_adapter_manifest
            cli_main._lookup_adapter_snapshot = original_lookup_latest
            cli_main._lookup_recent_adapter_snapshot = original_lookup_recent

        self.assertEqual(result.exit_code, 0, msg=result.stdout)
        self.assertIn(
            "local model: available=yes | requested_base_model=/Users/lichenhao/Desktop/PFE/models/Qwen3-4B | source_kind=path | source_path=/Users/lichenhao/Desktop/PFE/models/Qwen3-4B | load_mode=from_pretrained",
            result.stdout,
        )


if __name__ == "__main__":
    unittest.main()
