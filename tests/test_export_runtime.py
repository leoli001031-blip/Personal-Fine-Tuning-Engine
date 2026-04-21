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
    build_export_runtime_spec,
    build_llama_cpp_export_command_plan,
    describe_llama_cpp_export_constraint,
    dry_run_export_spec,
    execute_llama_cpp_export_command,
    materialize_export_plan,
    resolve_llama_cpp_export_tool_path,
    run_export_command_plan,
    summarize_llama_cpp_export_audit,
    validate_llama_cpp_export_toolchain,
    validate_llama_cpp_export_output_artifact,
    write_materialized_export_plan,
)


class ExportRuntimeTests(unittest.TestCase):
    def test_dry_run_export_spec_uses_standard_gguf_directory(self) -> None:
        spec = build_export_runtime_spec(
            target_backend="llama_cpp",
            source_artifact_format=AdapterArtifactFormat.PEFT_LORA.value,
            adapter_dir="/tmp/pfe-adapter",
            workspace="user_default",
            source_adapter_version="20260323-001",
            source_model="base-model",
            training_run_id="run-42",
            num_samples=9,
            extra_metadata={"note": "dry-run"},
        )

        self.assertTrue(spec.dry_run)
        self.assertTrue(spec.required)
        self.assertEqual(spec.artifact_directory, "gguf_merged")
        self.assertEqual(spec.artifact_name, "adapter_model.gguf")
        self.assertEqual(spec.output_dir, "/tmp/pfe-adapter/gguf_merged")
        self.assertIn("merged GGUF", spec.constraint)
        self.assertEqual(spec.manifest_updates["artifact_format"], AdapterArtifactFormat.GGUF_MERGED.value)
        self.assertEqual(spec.manifest_updates["artifact_name"], "adapter_model.gguf")
        self.assertEqual(spec.manifest_updates["export"]["export_directory"], "gguf_merged")
        self.assertEqual(spec.manifest_updates["export"]["note"], "dry-run")
        self.assertEqual(spec.manifest_updates["artifact_contract"]["source_artifact_role"], "lora_adapter")
        self.assertEqual(spec.manifest_updates["artifact_contract"]["target_artifact_role"], "merged_gguf")
        self.assertEqual(spec.manifest_updates["export"]["base_model_role"], "base_model")

    def test_transformers_export_keeps_peft_layout(self) -> None:
        spec = build_export_runtime_spec(
            target_backend="transformers",
            source_artifact_format=AdapterArtifactFormat.PEFT_LORA.value,
            adapter_dir="/tmp/pfe-adapter",
        )

        self.assertFalse(spec.required)
        self.assertEqual(spec.artifact_directory, "peft_lora")
        self.assertEqual(spec.artifact_name, "adapter_model.safetensors")
        self.assertEqual(spec.output_dir, "/tmp/pfe-adapter/peft_lora")
        self.assertIn("does not require a merged GGUF export", spec.constraint)
        self.assertEqual(spec.manifest_updates["inference_backend"], "transformers")

    def test_dry_run_export_spec_helper_matches_runtime_spec(self) -> None:
        spec = dry_run_export_spec(
            target_backend="llama.cpp",
            source_artifact_format=AdapterArtifactFormat.GGUF_MERGED.value,
        )

        self.assertFalse(spec["required"])
        self.assertEqual(spec["target_artifact_format"], AdapterArtifactFormat.GGUF_MERGED.value)
        self.assertEqual(spec["artifact_directory"], "gguf_merged")
        self.assertEqual(spec["metadata"]["source_artifact_role"], "merged_gguf")

    def test_constraint_message_mentions_llama_cpp_rule(self) -> None:
        message = describe_llama_cpp_export_constraint(
            target_backend="llama_cpp",
            source_artifact_format=AdapterArtifactFormat.PEFT_LORA.value,
        )

        self.assertIn("llama.cpp targets must be exported from a LoRA adapter to merged GGUF first", message)
        self.assertIn("LoRA safetensors", message)

    def test_materialize_export_plan_expands_placeholders_and_manifest_patch(self) -> None:
        plan = materialize_export_plan(
            target_backend="llama_cpp",
            source_artifact_format=AdapterArtifactFormat.PEFT_LORA.value,
            adapter_dir="/tmp/pfe-adapter",
            workspace="user_default",
            source_adapter_version="20260323-001",
            source_model="base-model",
            training_run_id="run-42",
            num_samples=9,
            extra_metadata={"note": "dry-run"},
        )

        self.assertTrue(plan.required)
        self.assertEqual(plan.output_dir, "/tmp/pfe-adapter/gguf_merged")
        self.assertEqual(
            plan.placeholder_files,
            ["adapter_manifest.json", "adapter_model.gguf", "export_plan.json", "EXPORT_NOTES.txt"],
        )
        self.assertIn("update adapter_manifest.json", plan.manifest_patch_description[0])
        self.assertTrue(any(item.endswith("artifact_name=adapter_model.gguf") for item in plan.manifest_patch_description))
        self.assertTrue(any(item.endswith("source_artifact_role=lora_adapter") for item in plan.manifest_patch_description))
        self.assertTrue(any(item.endswith("target_artifact_role=merged_gguf") for item in plan.manifest_patch_description))
        self.assertEqual(plan.manifest_patch["export"]["note"], "dry-run")
        self.assertEqual(plan.manifest_patch["export"]["source_artifact_role"], "lora_adapter")
        self.assertTrue(plan.metadata["materialized"])
        self.assertEqual(plan.metadata["placeholder_count"], 4)

    def test_materialized_export_plan_helper_matches_runtime_shape(self) -> None:
        payload = materialize_export_plan(
            target_backend="transformers",
            source_artifact_format=AdapterArtifactFormat.PEFT_LORA.value,
        ).to_dict()

        self.assertFalse(payload["required"])
        self.assertEqual(payload["artifact_directory"], "peft_lora")
        self.assertEqual(payload["placeholder_files"][1], "adapter_model.safetensors")
        self.assertIn("does not require a merged GGUF export", payload["constraint"])

    def test_write_materialized_export_plan_writes_placeholder_layout(self) -> None:
        with TemporaryDirectory() as tmpdir:
            adapter_dir = Path(tmpdir) / "adapter"
            result = write_materialized_export_plan(
                target_backend="llama_cpp",
                source_artifact_format=AdapterArtifactFormat.PEFT_LORA.value,
                adapter_dir=adapter_dir,
                workspace="user_default",
                source_adapter_version="20260323-001",
                source_model="base-model",
                training_run_id="run-99",
                num_samples=11,
                extra_metadata={"note": "write"},
            )

            output_dir = adapter_dir / "gguf_merged"
            artifact_path = output_dir / "adapter_model.gguf"
            export_plan_path = output_dir / "export_plan.json"
            notes_path = output_dir / "EXPORT_NOTES.txt"

            self.assertEqual(result.output_dir, str(output_dir))
            self.assertTrue(output_dir.is_dir())
            self.assertTrue(artifact_path.exists())
            self.assertTrue(export_plan_path.exists())
            self.assertTrue(notes_path.exists())
            self.assertEqual(
                result.written_files,
                [str(artifact_path), str(export_plan_path), str(notes_path)],
            )
            self.assertIn("placeholder artifact", artifact_path.read_text(encoding="utf-8"))
            self.assertIn("llama.cpp targets must be exported from a LoRA adapter to merged GGUF first", notes_path.read_text(encoding="utf-8"))
            self.assertIn("adapter_model.gguf", export_plan_path.read_text(encoding="utf-8"))
            self.assertTrue(result.metadata["materialized"])
            self.assertEqual(result.metadata["written_count"], 3)

    def test_write_materialized_export_plan_preserves_real_artifact_when_present(self) -> None:
        with TemporaryDirectory() as tmpdir:
            adapter_dir = Path(tmpdir) / "adapter"
            output_dir = adapter_dir / "gguf_merged"
            output_dir.mkdir(parents=True)
            artifact_path = output_dir / "adapter_model.gguf"
            artifact_path.write_bytes(b"GGUF-real-artifact")

            result = write_materialized_export_plan(
                target_backend="llama_cpp",
                source_artifact_format=AdapterArtifactFormat.PEFT_LORA.value,
                adapter_dir=adapter_dir,
            )

            self.assertEqual(result.artifact_path, str(artifact_path))
            self.assertEqual(artifact_path.read_bytes(), b"GGUF-real-artifact")
            self.assertTrue(result.metadata["artifact_preexisting"])
            self.assertTrue(result.metadata["artifact_preserved"])
            self.assertEqual(result.metadata["artifact_state"], "preserved")

    def test_resolve_llama_cpp_export_tool_path_reports_missing_tool(self) -> None:
        resolution = resolve_llama_cpp_export_tool_path("/tmp/does-not-exist-llama-export-tool")

        self.assertFalse(resolution.exists)
        self.assertFalse(resolution.executable)
        self.assertIn("export tool not found", resolution.reason)
        self.assertTrue(resolution.checked_paths)

    def test_resolve_llama_cpp_export_tool_path_accepts_python_script(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tool = Path(tmpdir) / "convert_lora_to_gguf.py"
            tool.write_text("print('ok')\n", encoding="utf-8")

            resolution = resolve_llama_cpp_export_tool_path(tool)
            validation = validate_llama_cpp_export_toolchain(resolution)

            self.assertTrue(resolution.exists)
            self.assertFalse(resolution.executable)
            self.assertTrue(resolution.metadata["runnable_with_python"])
            self.assertEqual(validation["status"], "planned")
            self.assertTrue(validation["allowed"])
            self.assertTrue(validation["runnable_with_python"])

    def test_resolve_llama_cpp_export_tool_path_uses_directory_env_override(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tool_dir = Path(tmpdir) / "llama.cpp"
            tool_dir.mkdir()
            tool = tool_dir / "convert_lora_to_gguf.py"
            tool.write_text("print('ok')\n", encoding="utf-8")

            resolution = resolve_llama_cpp_export_tool_path(env={"LLAMA_CPP_PATH": str(tool_dir)})

            self.assertTrue(resolution.exists)
            self.assertEqual(resolution.source, "env_dir")
            self.assertEqual(resolution.resolved_path, str(tool))
            self.assertEqual(resolution.metadata["env_name"], "LLAMA_CPP_PATH")

    def test_build_llama_cpp_export_command_plan_uses_real_lora_converter_shape(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            adapter_dir = tmp / "adapter"
            adapter_dir.mkdir()
            tool = tmp / "convert_lora_to_gguf.py"
            tool.write_text("print('export')\n", encoding="utf-8")
            base_model = tmp / "base-model"
            base_model.mkdir()

            plan = build_llama_cpp_export_command_plan(
                target_backend="llama_cpp",
                source_artifact_format=AdapterArtifactFormat.PEFT_LORA.value,
                adapter_dir=adapter_dir,
                tool_path=tool,
                source_model_path=base_model,
                extra_args=["--dry-run"],
            )

            self.assertEqual(plan.tool_resolution["resolved_path"], str(tool))
            self.assertEqual(plan.toolchain_status, "planned")
            self.assertFalse(plan.output_artifact_validation["valid"])
            self.assertEqual(plan.output_dir, str(adapter_dir / "gguf_merged"))
            self.assertEqual(plan.output_artifact_path, str(adapter_dir / "gguf_merged" / "adapter_model.gguf"))
            self.assertEqual(plan.command[0], os.sys.executable)
            self.assertEqual(plan.command[1], str(tool))
            self.assertIn("--outfile", plan.command)
            self.assertIn(str(adapter_dir / "gguf_merged" / "adapter_model.gguf"), plan.command)
            self.assertIn("--base", plan.command)
            self.assertIn(str(base_model), plan.command)
            self.assertIn(str(adapter_dir), plan.command)
            self.assertIn("--dry-run", plan.command)
            self.assertEqual(plan.command_metadata["execution_state"], "planned")
            self.assertEqual(plan.command_metadata["converter_kind"], "lora")
            self.assertEqual(plan.command_metadata["source_artifact_role"], "lora_adapter")
            self.assertEqual(plan.command_metadata["target_artifact_role"], "merged_gguf")
            self.assertEqual(plan.command_metadata["base_model_reference_kind"], "local_base_model")
            self.assertEqual(validate_llama_cpp_export_toolchain(plan.tool_resolution)["status"], "planned")

    def test_build_llama_cpp_export_command_plan_wraps_python_script_with_interpreter(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            adapter_dir = tmp / "adapter"
            adapter_dir.mkdir()
            tool = tmp / "convert_lora_to_gguf.py"
            tool.write_text("print('ok')\n", encoding="utf-8")

            plan = build_llama_cpp_export_command_plan(
                target_backend="llama_cpp",
                source_artifact_format=AdapterArtifactFormat.PEFT_LORA.value,
                adapter_dir=adapter_dir,
                tool_path=tool,
            )

            self.assertEqual(plan.command[0], os.sys.executable)
            self.assertEqual(plan.command[1], str(tool))
            self.assertEqual(plan.toolchain_status, "planned")

    def test_build_llama_cpp_export_command_plan_uses_base_model_id_for_non_local_reference(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            adapter_dir = tmp / "adapter"
            adapter_dir.mkdir()
            tool = tmp / "convert_lora_to_gguf.py"
            tool.write_text("print('ok')\n", encoding="utf-8")

            plan = build_llama_cpp_export_command_plan(
                target_backend="llama_cpp",
                source_artifact_format=AdapterArtifactFormat.PEFT_LORA.value,
                adapter_dir=adapter_dir,
                tool_path=tool,
                source_model_path="Qwen/Qwen3-4B",
            )

            self.assertIn("--base-model-id", plan.command)
            self.assertIn("Qwen/Qwen3-4B", plan.command)
            self.assertEqual(plan.command_metadata["converter_kind"], "lora")
            self.assertEqual(plan.command_metadata["base_model_reference_kind"], "base_model_id")
            self.assertIn("f16", plan.command)

    def test_build_llama_cpp_export_command_plan_defaults_lora_outtype_to_f16(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            adapter_dir = tmp / "adapter"
            adapter_dir.mkdir()
            tool = tmp / "convert_lora_to_gguf.py"
            tool.write_text("print('ok')\n", encoding="utf-8")
            base_model = tmp / "Qwen3-4B"
            base_model.mkdir()

            plan = build_llama_cpp_export_command_plan(
                target_backend="llama_cpp",
                source_artifact_format=AdapterArtifactFormat.PEFT_LORA.value,
                adapter_dir=adapter_dir,
                tool_path=tool,
                source_model_path=base_model,
            )

            self.assertIn("--outtype", plan.command)
            self.assertIn("f16", plan.command)
            self.assertFalse(plan.command_metadata["no_lazy"])

    def test_validate_llama_cpp_export_output_artifact_handles_missing_and_present_files(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            missing = validate_llama_cpp_export_output_artifact(tmp / "missing.gguf", required=True)
            self.assertFalse(missing.valid)
            self.assertEqual(missing.reason, "required output artifact is missing")

            artifact = tmp / "adapter.gguf"
            artifact.write_text("ok", encoding="utf-8")
            present = validate_llama_cpp_export_output_artifact(artifact, required=True)
            self.assertTrue(present.valid)
            self.assertTrue(present.exists)
            self.assertTrue(present.is_file)
            self.assertEqual(present.size_bytes, 2)

    def test_summarize_llama_cpp_export_audit_uses_stable_statuses(self) -> None:
        summary = summarize_llama_cpp_export_audit(
            toolchain_status="planned",
            execution_status="executed",
            outcome="success",
            output_artifact_validation={"valid": True, "reason": "output artifact exists"},
            command_metadata={"execution_state": "executed"},
        )

        self.assertEqual(summary["summary"], "executed")
        self.assertEqual(summary["toolchain_status"], "planned")
        self.assertEqual(summary["execution_status"], "executed")
        self.assertTrue(summary["artifact_valid"])
        self.assertEqual(summary["command_metadata"]["execution_state"], "executed")

    def test_run_export_command_plan_dry_run_reuses_plan_audit(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            adapter_dir = tmp / "adapter"
            adapter_dir.mkdir()
            tool = tmp / "fake-llama-export.sh"
            tool.write_text("#!/bin/sh\necho export:$@\n", encoding="utf-8")
            tool.chmod(0o755)

            plan = build_llama_cpp_export_command_plan(
                target_backend="llama_cpp",
                source_artifact_format=AdapterArtifactFormat.PEFT_LORA.value,
                adapter_dir=adapter_dir,
                tool_path=tool,
            )
            result = run_export_command_plan(plan, dry_run=True)

            self.assertFalse(result.attempted)
            self.assertTrue(result.success)
            self.assertEqual(result.status, "planned")
            self.assertEqual(result.outcome, "planned")
            self.assertEqual(result.audit["status"], "planned")
            self.assertEqual(result.metadata["execution_mode"], "dry_run")
            self.assertEqual(result.command[0], str(tool))
            self.assertEqual(result.command_metadata["execution_state"], "planned")
            self.assertFalse(result.output_artifact_validation["valid"])
            self.assertEqual(result.audit_summary["summary"], "planned")

    def test_execute_llama_cpp_export_command_skips_when_tool_missing(self) -> None:
        with TemporaryDirectory() as tmpdir:
            adapter_dir = Path(tmpdir) / "adapter"
            adapter_dir.mkdir()

            result = execute_llama_cpp_export_command(
                target_backend="llama_cpp",
                source_artifact_format=AdapterArtifactFormat.PEFT_LORA.value,
                adapter_dir=adapter_dir,
                tool_path="/tmp/does-not-exist-llama-export-tool",
                dry_run=False,
            )

            self.assertFalse(result.attempted)
            self.assertFalse(result.success)
            self.assertIsNone(result.returncode)
            self.assertIsNone(result.exit_code)
            self.assertEqual(result.audit["status"], "tool_missing")
            self.assertIn("missing or not executable", result.audit["failure_reason"])

    def test_execute_llama_cpp_export_command_reports_invalid_tool(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            adapter_dir = tmp / "adapter"
            adapter_dir.mkdir()
            invalid_tool = tmp / "invalid-llama-export-tool"
            invalid_tool.write_text("#!/bin/sh\necho invalid\n", encoding="utf-8")
            invalid_tool.chmod(0o644)

            result = execute_llama_cpp_export_command(
                target_backend="llama_cpp",
                source_artifact_format=AdapterArtifactFormat.PEFT_LORA.value,
                adapter_dir=adapter_dir,
                tool_path=invalid_tool,
                dry_run=False,
            )

            self.assertFalse(result.attempted)
            self.assertFalse(result.success)
            self.assertIsNone(result.exit_code)
            self.assertEqual(result.status, "tool_invalid")
            self.assertEqual(result.audit["status"], "tool_invalid")
            self.assertIn("not executable", result.audit["failure_reason"])
            self.assertEqual(result.audit_summary["summary"], "tool_invalid")

    def test_run_export_command_plan_reports_missing_tool_structure(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            adapter_dir = tmp / "adapter"
            adapter_dir.mkdir()
            tool = tmp / "missing-llama-export-tool"

            plan = build_llama_cpp_export_command_plan(
                target_backend="llama_cpp",
                source_artifact_format=AdapterArtifactFormat.PEFT_LORA.value,
                adapter_dir=adapter_dir,
                tool_path=tool,
            )
            result = run_export_command_plan(plan, dry_run=False)

            self.assertFalse(result.attempted)
            self.assertFalse(result.success)
            self.assertIsNone(result.exit_code)
            self.assertEqual(result.audit["status"], "tool_missing")
            self.assertEqual(result.status, "tool_missing")
            self.assertEqual(result.command_metadata["execution_state"], "blocked")
            self.assertEqual(result.audit_summary["summary"], "tool_missing")

    def test_execute_llama_cpp_export_command_dry_run_returns_audit_only(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            adapter_dir = tmp / "adapter"
            adapter_dir.mkdir()
            tool = tmp / "fake-llama-export.sh"
            tool.write_text("#!/bin/sh\necho \"$@\"\nexit 0\n", encoding="utf-8")
            tool.chmod(0o755)

            result = execute_llama_cpp_export_command(
                target_backend="llama_cpp",
                source_artifact_format=AdapterArtifactFormat.PEFT_LORA.value,
                adapter_dir=adapter_dir,
                tool_path=tool,
                source_model_path=tmp / "source-model",
                extra_args=["--dry-run"],
                dry_run=True,
            )

            self.assertFalse(result.attempted)
            self.assertTrue(result.success)
            self.assertIsNone(result.returncode)
            self.assertIsNone(result.exit_code)
            self.assertEqual(result.status, "planned")
            self.assertEqual(result.outcome, "planned")
            self.assertEqual(result.audit["status"], "planned")
            self.assertEqual(result.metadata["execution_mode"], "dry_run")

    def test_execute_llama_cpp_export_command_runs_fake_tool(self) -> None:
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
                "  if [ \"$1\" = \"--output-dir\" ]; then\n"
                "    outdir=\"$2\"\n"
                "    shift 2\n"
                "    continue\n"
                "  fi\n"
                "  if [ \"$1\" = \"--output-name\" ]; then\n"
                "    outname=\"$2\"\n"
                "    shift 2\n"
                "    continue\n"
                "  fi\n"
                "  shift\n"
                "done\n"
                "mkdir -p \"$outdir\"\n"
                "printf 'gguf output\\n' > \"$outdir/$outname\"\n"
                "echo \"$@\"\n"
                "exit 0\n",
                encoding="utf-8",
            )
            tool.chmod(0o755)

            result = execute_llama_cpp_export_command(
                target_backend="llama_cpp",
                source_artifact_format=AdapterArtifactFormat.PEFT_LORA.value,
                adapter_dir=adapter_dir,
                tool_path=tool,
                source_model_path=tmp / "source-model",
                extra_args=["--dry-run"],
                dry_run=False,
            )

            self.assertTrue(result.attempted)
            self.assertTrue(result.success)
            self.assertEqual(result.returncode, 0)
            self.assertEqual(result.exit_code, 0)
            self.assertEqual(result.status, "executed")
            self.assertEqual(result.outcome, "success")
            self.assertEqual(result.audit["status"], "executed")
            self.assertEqual(result.audit["outcome"], "success")
            self.assertEqual(result.audit["exit_code"], 0)
            self.assertTrue((adapter_dir / "gguf_merged" / "adapter_model.gguf").exists())
            self.assertTrue(result.output_artifact_validation["valid"])
            self.assertEqual(result.toolchain_summary["source_artifact_role"], "lora_adapter")
            self.assertEqual(result.toolchain_summary["target_artifact_role"], "merged_gguf")
            self.assertEqual(result.audit_summary["summary"], "executed")

    def test_execute_llama_cpp_export_command_creates_output_dir_before_running_tool(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            adapter_dir = tmp / "adapter"
            adapter_dir.mkdir()
            tool = tmp / "convert_lora_to_gguf"
            tool.write_text(
                "#!/bin/sh\n"
                "outfile=\"\"\n"
                "while [ $# -gt 0 ]; do\n"
                "  if [ \"$1\" = \"--outfile\" ]; then\n"
                "    outfile=\"$2\"\n"
                "    shift 2\n"
                "    continue\n"
                "  fi\n"
                "  shift\n"
                "done\n"
                "printf 'GGUF' > \"$outfile\"\n",
                encoding="utf-8",
            )
            tool.chmod(0o755)

            result = execute_llama_cpp_export_command(
                target_backend="llama_cpp",
                source_artifact_format=AdapterArtifactFormat.PEFT_LORA.value,
                adapter_dir=adapter_dir,
                tool_path=tool,
                source_model_path=tmp / "base-model",
                dry_run=False,
            )

            self.assertTrue(result.attempted)
            self.assertTrue(result.success)
            self.assertEqual(result.status, "executed")
            self.assertTrue((adapter_dir / "gguf_merged").is_dir())
            self.assertTrue((adapter_dir / "gguf_merged" / "adapter_model.gguf").exists())

    def test_execute_llama_cpp_export_command_surfaces_stderr_summary_on_failure(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            adapter_dir = tmp / "adapter"
            adapter_dir.mkdir()
            tool = tmp / "fake-llama-export.sh"
            tool.write_text(
                "#!/bin/sh\n"
                "echo 'warning: preflight note' 1>&2\n"
                "echo 'Traceback (most recent call last):' 1>&2\n"
                "echo \"AttributeError: 'LoraTorchTensor' object has no attribute 'dim'\" 1>&2\n"
                "exit 1\n",
                encoding="utf-8",
            )
            tool.chmod(0o755)

            result = execute_llama_cpp_export_command(
                target_backend="llama_cpp",
                source_artifact_format=AdapterArtifactFormat.PEFT_LORA.value,
                adapter_dir=adapter_dir,
                tool_path=tool,
                source_model_path=tmp / "source-model",
                dry_run=False,
            )

            self.assertTrue(result.attempted)
            self.assertFalse(result.success)
            self.assertEqual(result.returncode, 1)
            self.assertEqual(result.status, "command_failed")
            self.assertEqual(result.audit["status"], "command_failed")
            self.assertIn("exit code 1", result.failure_reason or "")
            self.assertIn("LoraTorchTensor", result.failure_reason or "")
            self.assertEqual(result.audit_summary["summary"], "command_failed")


if __name__ == "__main__":
    unittest.main()
