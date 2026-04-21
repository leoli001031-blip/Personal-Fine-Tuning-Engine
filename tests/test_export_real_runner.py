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
    execute_llama_cpp_export_command,
    validate_llama_cpp_export_output_artifact,
)


class ExportRealRunnerTests(unittest.TestCase):
    def test_validate_output_artifact_rejects_empty_files(self) -> None:
        with TemporaryDirectory() as tmpdir:
            artifact = Path(tmpdir) / "adapter_model.gguf"
            artifact.write_text("", encoding="utf-8")

            validation = validate_llama_cpp_export_output_artifact(artifact, required=True)
            self.assertFalse(validation.valid)
            self.assertEqual(validation.reason, "output artifact is empty")

    def test_execute_llama_cpp_export_command_reports_successful_real_artifact(self) -> None:
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
                "echo 'exported'\n"
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
            self.assertEqual(result.failure_category, "success")
            self.assertTrue(result.success)
            self.assertTrue(result.output_artifact_validation["valid"])
            self.assertTrue((adapter_dir / "gguf_merged" / "adapter_model.gguf").exists())
            self.assertIn("failure_category", result.audit)
            self.assertEqual(result.audit["failure_category"], "success")
            self.assertEqual(result.audit_summary["summary"], "executed")
            self.assertEqual(result.toolchain_summary["failure_category"], "success")

    def test_execute_llama_cpp_export_command_flags_missing_artifact(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            adapter_dir = tmp / "adapter"
            adapter_dir.mkdir()
            tool = tmp / "fake-llama-export-missing.sh"
            tool.write_text(
                "#!/bin/sh\n"
                "echo 'no artifact'\n"
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
            self.assertEqual(result.failure_category, "artifact_missing")
            self.assertTrue(result.success)
            self.assertIn("missing", result.failure_reason or "")
            self.assertFalse(result.output_artifact_validation["valid"])
            self.assertEqual(result.audit_summary["summary"], "executed")

    def test_execute_llama_cpp_export_command_flags_empty_artifact(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            adapter_dir = tmp / "adapter"
            adapter_dir.mkdir()
            tool = tmp / "fake-llama-export-empty.sh"
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
                ": > \"$outdir/$outname\"\n"
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
            self.assertEqual(result.failure_category, "artifact_empty")
            self.assertTrue(result.success)
            self.assertEqual(result.output_artifact_validation["reason"], "output artifact is empty")
            self.assertEqual(result.audit_summary["summary"], "executed")

    def test_execute_llama_cpp_export_command_reports_command_failure(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            adapter_dir = tmp / "adapter"
            adapter_dir.mkdir()
            tool = tmp / "fake-llama-export-failed.sh"
            tool.write_text(
                "#!/bin/sh\n"
                "echo 'boom' >&2\n"
                "exit 7\n",
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

            self.assertEqual(result.status, "command_failed")
            self.assertEqual(result.failure_category, "command_failed")
            self.assertFalse(result.success)
            self.assertEqual(result.returncode, 7)
            self.assertIn("boom", result.stderr)
            self.assertIn("exit code 7", result.failure_reason or "")

    def test_execute_llama_cpp_export_command_reports_missing_tool(self) -> None:
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
            self.assertEqual(result.failure_category, "tool_missing")
            self.assertFalse(result.success)
            self.assertEqual(result.toolchain_summary["failure_category"], "tool_missing")


if __name__ == "__main__":
    unittest.main()
