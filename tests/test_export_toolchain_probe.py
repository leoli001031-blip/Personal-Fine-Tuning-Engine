from __future__ import annotations

import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
for package_dir in ("pfe-core", "pfe-cli", "pfe-server"):
    package_path = str(ROOT / package_dir)
    if package_path not in os.sys.path:
        os.sys.path.insert(0, package_path)

from pfe_core.adapter_store.lifecycle import AdapterArtifactFormat
from pfe_core.inference.export_runtime import (
    build_llama_cpp_export_command_plan,
    probe_llama_cpp_export_toolchain,
    summarize_llama_cpp_export_toolchain,
)


class ExportToolchainProbeTests(unittest.TestCase):
    def test_probe_reports_missing_tool_with_actionable_env_hints(self) -> None:
        probe = probe_llama_cpp_export_toolchain(tool_path="/tmp/definitely-missing-llama-export-tool")

        self.assertEqual(probe["status"], "tool_missing")
        self.assertFalse(probe["ready"])
        self.assertEqual(probe["readiness_state"], "blocked")
        self.assertIn("PFE_LLAMA_CPP_EXPORT_TOOL", probe["recommended_action"])
        self.assertIn("tool_path explicitly", probe["recommended_action"])
        self.assertTrue(probe["searched_env"])
        self.assertEqual(
            probe["tool_resolution"]["metadata"]["searched_env"],
            probe["searched_env"],
        )

    def test_probe_reports_python_script_ready_state_from_env_directory(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tool_dir = Path(tmpdir) / "llama.cpp"
            tool_dir.mkdir()
            tool = tool_dir / "convert_lora_to_gguf.py"
            tool.write_text("print('ok')\n", encoding="utf-8")

            probe = probe_llama_cpp_export_toolchain(env={"LLAMA_CPP_PATH": str(tool_dir)})

            self.assertEqual(probe["status"], "planned")
            self.assertTrue(probe["ready"])
            self.assertEqual(probe["readiness_state"], "ready")
            self.assertEqual(probe["source"], "env_dir")
            self.assertEqual(probe["resolved_path"], str(tool))
            self.assertEqual(probe["env_name"], "LLAMA_CPP_PATH")
            self.assertTrue(probe["runnable_with_python"])
            self.assertIn("resolved_path=", probe["recommended_action"])
            self.assertIn("convert_lora_to_gguf.py", probe["tool_resolution"]["checked_paths"][-1])

    def test_summary_surface_exposes_readiness_probe_details(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            adapter_dir = tmp / "adapter"
            adapter_dir.mkdir()
            tool = tmp / "fake-llama-export.sh"
            tool.write_text("#!/bin/sh\necho ready\n", encoding="utf-8")
            tool.chmod(0o755)

            plan = build_llama_cpp_export_command_plan(
                target_backend="llama_cpp",
                source_artifact_format=AdapterArtifactFormat.PEFT_LORA.value,
                adapter_dir=adapter_dir,
                tool_path=tool,
            )
            summary = summarize_llama_cpp_export_toolchain(
                toolchain_status=plan.toolchain_status,
                toolchain_validation=plan.toolchain_validation,
                output_artifact_validation=plan.output_artifact_validation,
                command_metadata=plan.command_metadata,
                execution_status="planned",
                outcome="planned",
            )

            self.assertTrue(summary["toolchain_ready"])
            self.assertEqual(summary["toolchain_readiness_state"], "ready")
            self.assertTrue(summary["probe_summary"]["ready"])
            self.assertEqual(summary["probe_summary"]["status"], "planned")
            self.assertIn("recommended_action", summary["probe_summary"])
            self.assertIn("toolchain_searched_env", summary)


if __name__ == "__main__":
    unittest.main()
