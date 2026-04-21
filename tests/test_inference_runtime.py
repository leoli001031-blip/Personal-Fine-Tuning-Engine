from __future__ import annotations

import os
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
for package_dir in ("pfe-core", "pfe-cli", "pfe-server"):
    package_path = str(ROOT / package_dir)
    if package_path not in os.sys.path:
        os.sys.path.insert(0, package_path)

from pfe_core.inference.engine import (
    InferenceConfig,
    InferenceEngine,
    _clean_llama_cpp_output,
    _resolve_llama_cpp_runtime_binary,
    _strip_thinking_output,
    resolve_base_model_reference,
)
from pfe_core.pipeline import PipelineService


class InferenceRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.previous_base_model = os.environ.get("PFE_BASE_MODEL")
        os.environ["PFE_BASE_MODEL"] = str(Path(self.tempdir.name) / "fake-base-model")

    def tearDown(self) -> None:
        if self.previous_base_model is None:
            os.environ.pop("PFE_BASE_MODEL", None)
        else:
            os.environ["PFE_BASE_MODEL"] = self.previous_base_model
        os.environ.pop("PFE_DISABLE_AUTO_LOCAL_BASE_MODEL", None)
        self.tempdir.cleanup()

    def test_resolve_base_model_reference_uses_environment_override_for_local_default(self) -> None:
        self.assertEqual(
            resolve_base_model_reference("local-default"),
            str(Path(self.tempdir.name) / "fake-base-model"),
        )

    def test_strip_thinking_output_removes_think_block(self) -> None:
        text = "<think>\n先想想\n</think>\n\n你好，我在。"
        self.assertEqual(_strip_thinking_output(text), "你好，我在。")

    def test_clean_llama_cpp_output_removes_prompt_rollover_and_perf_lines(self) -> None:
        raw = (
            "你好，有什么可以帮助你的吗？\n"
            "USER: 你是谁\n"
            "common_perf_print: total time = 1 ms\n"
            "llama_memory_breakdown_print: host ...\n"
            "AI"
        )
        self.assertEqual(_clean_llama_cpp_output(raw), "你好，有什么可以帮助你的吗？")

    def test_resolve_base_model_reference_can_disable_repo_auto_discovery(self) -> None:
        os.environ.pop("PFE_BASE_MODEL", None)
        os.environ["PFE_DISABLE_AUTO_LOCAL_BASE_MODEL"] = "1"
        self.assertEqual(resolve_base_model_reference("local-default"), "Qwen/Qwen2.5-3B-Instruct")

    def test_generate_prefers_real_response_when_runtime_succeeds(self) -> None:
        engine = InferenceEngine(InferenceConfig(base_model="local-default"))
        with patch.object(
            engine,
            "_generate_real_response",
            return_value={
                "text": "这是一条真实推理回复。",
                "served_by": "local",
                "runtime_path": "real_local",
                "resolved_base_model": "/tmp/fake-base",
                "adapter_loaded": False,
                "thinking_stripped": False,
            },
        ):
            text = engine.generate([{"role": "user", "content": "你好"}], metadata={"enable_real_local": True})

        self.assertEqual(text, "这是一条真实推理回复。")
        self.assertEqual(engine.status()["served_by"], "local")
        self.assertEqual(engine.status()["runtime_path"], "real_local")

    def test_generate_falls_back_to_template_when_real_runtime_fails(self) -> None:
        engine = InferenceEngine(InferenceConfig(base_model="local-default"))
        with patch.object(engine, "_generate_real_response", side_effect=RuntimeError("boom")):
            text = engine.generate([{"role": "user", "content": "你好"}], metadata={"enable_real_local": True})

        self.assertIn("[base]", text)
        self.assertEqual(engine.status()["served_by"], "mock")
        self.assertIn("fallback_reason", engine.status()["generation"])

    def test_generate_does_not_attempt_remote_model_download_for_non_local_reference(self) -> None:
        os.environ.pop("PFE_BASE_MODEL", None)
        os.environ["PFE_DISABLE_AUTO_LOCAL_BASE_MODEL"] = "1"
        engine = InferenceEngine(InferenceConfig(base_model="base"))

        text = engine.generate([{"role": "user", "content": "你好"}], metadata={"enable_real_local": True})

        self.assertIn("[base]", text)
        self.assertEqual(engine.status()["served_by"], "mock")
        self.assertIn("requires a local base model path", engine.status()["generation"]["fallback_reason"])

    def test_llama_cpp_runtime_resolution_prefers_cpu_build(self) -> None:
        resolution = _resolve_llama_cpp_runtime_binary()
        self.assertTrue(resolution["available"])
        self.assertIn("build-cpu/bin/llama-completion", str(resolution["path"]))

    def test_generate_prefers_llama_cpp_for_gguf_manifest(self) -> None:
        adapter_dir = Path(self.tempdir.name) / "adapter"
        (adapter_dir / "gguf_merged").mkdir(parents=True)
        manifest = {
            "version": "20260325-001",
            "base_model": str(Path(self.tempdir.name) / "fake-base-model"),
            "artifact_format": "gguf_merged",
            "metadata": {
                "export_artifact_summary": {
                    "path": str(adapter_dir / "gguf_merged" / "adapter_model.gguf"),
                    "valid": True,
                }
            },
        }
        (adapter_dir / "adapter_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (adapter_dir / "gguf_merged" / "adapter_model.gguf").write_text("GGUF", encoding="utf-8")
        engine = InferenceEngine(
            InferenceConfig(
                base_model="local-default",
                adapter_path=str(adapter_dir),
            )
        )
        with patch.object(
            engine,
            "_generate_llama_cpp_response",
            return_value={
                "text": "这是 llama.cpp 回复。",
                "served_by": "local",
                "runtime_path": "llama_cpp",
            },
        ) as llama_runtime, patch.object(
            engine,
            "_generate_real_response",
            side_effect=AssertionError("transformers runtime should not be used first"),
        ):
            text = engine.generate([{"role": "user", "content": "你好"}], metadata={"enable_real_local": True})

        self.assertEqual(text, "这是 llama.cpp 回复。")
        self.assertEqual(engine.status()["runtime_path"], "llama_cpp")
        llama_runtime.assert_called_once()

    def test_generate_records_llama_cpp_failure_without_transformers_fallback_for_gguf_manifest(self) -> None:
        adapter_dir = Path(self.tempdir.name) / "adapter"
        adapter_dir.mkdir(parents=True)
        manifest = {
            "version": "20260325-001",
            "base_model": str(Path(self.tempdir.name) / "fake-base-model"),
            "artifact_format": "gguf_merged",
        }
        (adapter_dir / "adapter_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        engine = InferenceEngine(
            InferenceConfig(
                base_model="local-default",
                adapter_path=str(adapter_dir),
            )
        )
        with patch.object(engine, "_generate_llama_cpp_response", side_effect=RuntimeError("llama failed")), patch.object(
            engine,
            "_generate_real_response",
            side_effect=AssertionError("transformers fallback should not run for explicit llama.cpp backend"),
        ):
            text = engine.generate([{"role": "user", "content": "你好"}], metadata={"enable_real_local": True})

        self.assertIn("[adapter:20260325-001]", text)
        generation = engine.status()["generation"]
        self.assertIn("llama_cpp: RuntimeError: llama failed", generation["fallback_reason"])
        self.assertEqual(generation["previous_runtime_failures"], ["llama_cpp: RuntimeError: llama failed"])

    def test_pipeline_chat_completion_surfaces_local_served_by_and_usage(self) -> None:
        service = PipelineService()
        with patch("pfe_core.pipeline.InferenceEngine.generate", return_value="真实模型输出"), patch(
            "pfe_core.pipeline.InferenceEngine.status",
            return_value={"backend": "transformers", "healthy": True, "served_by": "local", "runtime_path": "real_local"},
        ):
            payload = service.chat_completion(
                messages=[{"role": "user", "content": "你好"}],
                model="local-default",
                metadata={"style_hint": "helpful", "enable_real_local": True},
                request_id="req-1",
                session_id="sess-1",
            )

        self.assertEqual(payload["served_by"], "local")
        self.assertEqual(payload["request_id"], "req-1")
        self.assertEqual(payload["session_id"], "sess-1")
        self.assertEqual(payload["choices"][0]["message"]["content"], "真实模型输出")
        self.assertIn("usage", payload)
        self.assertGreaterEqual(payload["usage"]["completion_tokens"], 1)

    def test_generate_skips_real_runtime_when_not_explicitly_enabled(self) -> None:
        engine = InferenceEngine(InferenceConfig(base_model="local-default"))
        with patch.object(engine, "_generate_real_response", side_effect=AssertionError("should not be called")):
            text = engine.generate([{"role": "user", "content": "你好"}], metadata={"style_hint": "helpful"})

        self.assertIn("[base]", text)
        self.assertEqual(engine.status()["served_by"], "mock")
        self.assertFalse(engine.status()["real_local_enabled"])


if __name__ == "__main__":
    unittest.main()
