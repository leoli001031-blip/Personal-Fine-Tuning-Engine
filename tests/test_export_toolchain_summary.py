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
    execute_llama_cpp_export_command,
    summarize_llama_cpp_export_toolchain,
    validate_llama_cpp_export_output_artifact,
)


class ExportToolchainSummaryTests(unittest.TestCase):
    def test_planned_summary_exposes_toolchain_and_artifact_checks(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            adapter_dir = tmp / "adapter"
            adapter_dir.mkdir()
            tool = tmp / "fake-llama-export.sh"
            tool.write_text("#!/bin/sh\necho planned\n", encoding="utf-8")
            tool.chmod(0o755)

            plan = build_llama_cpp_export_command_plan(
                target_backend="llama_cpp",
                source_artifact_format=AdapterArtifactFormat.PEFT_LORA.value,
                adapter_dir=adapter_dir,
                tool_path=tool,
            )

            summary = plan.toolchain_summary
            self.assertEqual(summary["summary"], "planned")
            self.assertEqual(summary["toolchain_status"], "planned")
            self.assertTrue(summary["toolchain_allowed"])
            self.assertFalse(summary["output_artifact_valid"])
            self.assertEqual(summary["source_artifact_role"], "lora_adapter")
            self.assertEqual(summary["target_artifact_role"], "merged_gguf")
            self.assertEqual(summary["artifact_contract"]["base_model_role"], "base_model")
            self.assertEqual(summary["audit_summary"]["summary"], "planned")
            self.assertEqual(summary["command_metadata"]["execution_state"], "planned")

            rebuilt = summarize_llama_cpp_export_toolchain(
                toolchain_status=plan.toolchain_status,
                toolchain_validation=plan.toolchain_validation,
                output_artifact_validation=plan.output_artifact_validation,
                command_metadata=plan.command_metadata,
                execution_status="planned",
                outcome="planned",
            )
            self.assertEqual(rebuilt["toolchain_status"], "planned")
            self.assertEqual(rebuilt["output_artifact_reason"], "output artifact is not present yet")

    def test_executed_summary_reports_valid_artifact(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            adapter_dir = tmp / "adapter"
            adapter_dir.mkdir()
            tool = tmp / "fake-llama-export.sh"
            tool.write_text(
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
                "exit 0\n",
                encoding="utf-8",
            )
            tool.chmod(0o755)

            result = execute_llama_cpp_export_command(
                target_backend="llama_cpp",
                source_artifact_format=AdapterArtifactFormat.PEFT_LORA.value,
                adapter_dir=adapter_dir,
                tool_path=tool,
                dry_run=False,
            )

            self.assertEqual(result.status, "executed")
            self.assertEqual(result.toolchain_summary["summary"], "executed")
            self.assertTrue(result.output_artifact_validation["valid"])
            self.assertTrue((adapter_dir / "gguf_merged" / "adapter_model.gguf").exists())
            self.assertEqual(result.toolchain_summary["toolchain_status"], "planned")
            self.assertEqual(result.toolchain_summary["execution_status"], "executed")
            self.assertEqual(result.toolchain_summary["outcome"], "success")
            self.assertEqual(result.toolchain_summary["artifact_contract"]["target_artifact_role"], "merged_gguf")
            self.assertEqual(result.toolchain_summary["base_model_role"], "base_model")

    def test_missing_tool_summary_is_structured(self) -> None:
        with TemporaryDirectory() as tmpdir:
            adapter_dir = Path(tmpdir) / "adapter"
            adapter_dir.mkdir()

            result = execute_llama_cpp_export_command(
                target_backend="llama_cpp",
                source_artifact_format=AdapterArtifactFormat.PEFT_LORA.value,
                adapter_dir=adapter_dir,
                tool_path="/tmp/definitely-missing-llama-export-tool",
            )

            self.assertEqual(result.status, "tool_missing")
            self.assertEqual(result.toolchain_summary["summary"], "tool_missing")
            self.assertFalse(result.toolchain_summary["toolchain_allowed"])
            self.assertEqual(result.toolchain_summary["toolchain_status"], "tool_missing")
            self.assertIn("not found", result.toolchain_summary["toolchain_reason"])

    def test_invalid_tool_summary_is_structured(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            adapter_dir = tmp / "adapter"
            adapter_dir.mkdir()
            tool = tmp / "invalid-llama-export-tool"
            tool.write_text("#!/bin/sh\necho invalid\n", encoding="utf-8")
            tool.chmod(0o644)

            result = execute_llama_cpp_export_command(
                target_backend="llama_cpp",
                source_artifact_format=AdapterArtifactFormat.PEFT_LORA.value,
                adapter_dir=adapter_dir,
                tool_path=tool,
            )

            self.assertEqual(result.status, "tool_invalid")
            self.assertEqual(result.toolchain_summary["summary"], "tool_invalid")
            self.assertFalse(result.toolchain_summary["toolchain_allowed"])
            self.assertEqual(result.toolchain_summary["toolchain_status"], "tool_invalid")
            self.assertIn("not", result.toolchain_summary["toolchain_reason"])
            self.assertFalse(result.toolchain_summary["toolchain_validation"].get("runnable_with_python", False))

    def test_validate_output_artifact_helper_keeps_state_consistent(self) -> None:
        with TemporaryDirectory() as tmpdir:
            artifact = Path(tmpdir) / "adapter_model.gguf"
            validation = validate_llama_cpp_export_output_artifact(artifact, required=False)
            self.assertFalse(validation.valid)
            self.assertEqual(validation.reason, "output artifact is not present yet")


if __name__ == "__main__":
    unittest.main()
