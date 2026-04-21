from __future__ import annotations

import os
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for package_dir in ("pfe-core", "pfe-cli", "pfe-server"):
    package_path = str(ROOT / package_dir)
    if package_path not in os.sys.path:
        os.sys.path.insert(0, package_path)

from pfe_core.adapter_store.lifecycle import AdapterArtifactFormat
from pfe_core.inference.backends import plan_inference_backend
from pfe_core.inference.export import plan_export


class InferencePlanningTests(unittest.TestCase):
    def test_explicit_llama_cpp_requires_gguf_merged_export(self) -> None:
        decision = plan_inference_backend(
            requested_backend="llama_cpp",
            artifact_format=AdapterArtifactFormat.PEFT_LORA.value,
            platform_name="Linux",
            machine_name="x86_64",
        )

        self.assertEqual(decision.selected_backend, "llama_cpp")
        self.assertTrue(decision.requires_export)
        self.assertEqual(decision.required_artifact_format, AdapterArtifactFormat.GGUF_MERGED.value)
        self.assertIn("merged GGUF export required", decision.reason)

    def test_auto_backend_prefers_llama_cpp_for_gguf_artifacts(self) -> None:
        decision = plan_inference_backend(
            requested_backend="auto",
            artifact_format=AdapterArtifactFormat.GGUF_MERGED.value,
            platform_name="Linux",
            machine_name="x86_64",
        )

        self.assertEqual(decision.selected_backend, "llama_cpp")
        self.assertFalse(decision.requires_export)
        self.assertEqual(decision.required_artifact_format, AdapterArtifactFormat.GGUF_MERGED.value)

    def test_auto_backend_falls_back_to_transformers_on_non_apple_local_runtime(self) -> None:
        decision = plan_inference_backend(
            requested_backend="auto",
            artifact_format=AdapterArtifactFormat.PEFT_LORA.value,
            platform_name="Linux",
            machine_name="x86_64",
        )

        self.assertEqual(decision.selected_backend, "transformers")
        self.assertFalse(decision.requires_export)
        self.assertIn("fallback local backend is transformers", decision.reason)

    def test_plan_export_includes_manifest_updates_for_gguf_merged(self) -> None:
        plan = plan_export(
            target_backend="llama_cpp",
            source_artifact_format=AdapterArtifactFormat.PEFT_LORA.value,
            workspace="user_default",
            adapter_dir="/tmp/pfe-adapter",
            source_adapter_version="20260323-001",
            source_model="base-model",
            training_run_id="run-1",
            num_samples=7,
            extra_metadata={"note": "integration"},
        )

        self.assertTrue(plan.required)
        self.assertEqual(plan.target_artifact_format, AdapterArtifactFormat.GGUF_MERGED.value)
        self.assertEqual(plan.artifact_name, "adapter_model.gguf")
        self.assertEqual(plan.artifact_directory, "gguf_merged")
        self.assertEqual(plan.metadata["source_artifact_role"], "lora_adapter")
        self.assertEqual(plan.metadata["target_artifact_role"], "merged_gguf")
        self.assertEqual(plan.metadata["artifact_contract"]["base_model_role"], "base_model")

        manifest_updates = plan.manifest_updates
        self.assertEqual(manifest_updates["workspace"], "user_default")
        self.assertEqual(manifest_updates["adapter_dir"], "/tmp/pfe-adapter")
        self.assertEqual(manifest_updates["source_adapter_version"], "20260323-001")
        self.assertEqual(manifest_updates["source_model"], "base-model")
        self.assertEqual(manifest_updates["training_run_id"], "run-1")
        self.assertEqual(manifest_updates["num_samples"], 7)
        self.assertEqual(manifest_updates["artifact_format"], AdapterArtifactFormat.GGUF_MERGED.value)
        self.assertEqual(manifest_updates["artifact_name"], "adapter_model.gguf")
        self.assertEqual(manifest_updates["inference_backend"], "llama_cpp")
        self.assertTrue(manifest_updates["requires_export"])
        self.assertEqual(manifest_updates["artifact_contract"]["source_artifact_role"], "lora_adapter")
        self.assertEqual(manifest_updates["artifact_contract"]["target_artifact_role"], "merged_gguf")
        self.assertEqual(manifest_updates["export"]["target_backend"], "llama_cpp")
        self.assertEqual(manifest_updates["export"]["target_artifact_format"], AdapterArtifactFormat.GGUF_MERGED.value)
        self.assertEqual(manifest_updates["export"]["export_directory"], "gguf_merged")
        self.assertEqual(manifest_updates["export"]["source_artifact_role"], "lora_adapter")
        self.assertEqual(manifest_updates["export"]["target_artifact_role"], "merged_gguf")
        self.assertEqual(manifest_updates["export"]["note"], "integration")


if __name__ == "__main__":
    unittest.main()
