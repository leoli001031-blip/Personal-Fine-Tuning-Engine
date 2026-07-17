from __future__ import annotations

import json
import subprocess
from unittest.mock import patch

import pytest

from pfe_core.inference.contracts import BOUNDARY_CONTRACT_ID, PERSONA_V4_CONTRACT_ID
from pfe_core.inference.engine import InferenceConfig, InferenceEngine, InferenceError


def _phase85_messages() -> list[dict[str, str]]:
    return [
        {
            "role": "user",
            "content": "给当前状态，并安排追踪上传记录和补导后验收。",
        }
    ]


def test_boundary_fallback_status_keeps_only_structural_contract_output() -> None:
    engine = InferenceEngine(InferenceConfig(base_model="local-default"))
    user_marker = "BOUNDARY_FALLBACK_PRIVATE_MARKER"

    output = engine.generate(
        [{"role": "user", "content": f"总结当前材料：{user_marker}"}],
        metadata={"response_contract": BOUNDARY_CONTRACT_ID},
    )

    generation = engine.status()["generation"]
    contract_output = generation["contract_output"]
    serialized = json.dumps(engine.status(), ensure_ascii=False, sort_keys=True)
    assert contract_output["complete"] is True
    assert contract_output["fallback_used"] is True
    assert contract_output["raw_output_persisted"] is False
    assert "raw_output" not in contract_output
    assert "normalized_output" not in contract_output
    assert output not in serialized
    assert user_marker not in serialized


def test_phase85_v4_status_discards_raw_and_normalized_text_but_keeps_flags() -> None:
    engine = InferenceEngine(
        InferenceConfig(base_model="local-default", backend="transformers")
    )
    raw_output = (
        "不能确认已完成。\n"
        "依据：PHASE85_RAW_OUTPUT_MARKER，文件仍缺失。\n"
        "下一步：追踪上传记录。"
    )
    with patch.object(
        engine,
        "_generate_real_response",
        return_value={
            "text": raw_output,
            "raw_text": "PHASE85_TRANSPORT_RAW_MARKER",
            "served_by": "local",
            "runtime_path": "real_local",
        },
    ):
        normalized_output = engine.generate(
            _phase85_messages(),
            metadata={
                "enable_real_local": True,
                "response_contract": PERSONA_V4_CONTRACT_ID,
            },
        )

    generation = engine.status()["generation"]
    contract_output = generation["contract_output"]
    serialized_status = json.dumps(engine.status(), ensure_ascii=False, sort_keys=True)
    serialized_last_generation = json.dumps(
        engine.last_generation_info, ensure_ascii=False, sort_keys=True
    )
    assert contract_output["guard_applied"] is True
    assert contract_output["factual_guard_evaluated"] is True
    assert contract_output["semantic_repair_used"] is True
    assert contract_output["native_format"] is False
    assert contract_output["fallback_used"] is False
    assert contract_output["raw_output_persisted"] is False
    assert "raw_output" not in contract_output
    assert "normalized_output" not in contract_output
    assert "text" not in generation
    assert "raw_text" not in generation
    for serialized in (serialized_status, serialized_last_generation):
        assert raw_output not in serialized
        assert normalized_output not in serialized
        assert "PHASE85_RAW_OUTPUT_MARKER" not in serialized
        assert "PHASE85_TRANSPORT_RAW_MARKER" not in serialized


def test_first_runtime_success_records_one_backend_attempt() -> None:
    engine = InferenceEngine(
        InferenceConfig(base_model="local-default", backend="transformers")
    )
    with patch.object(
        engine,
        "_generate_real_response",
        return_value={
            "text": "first runtime succeeded",
            "served_by": "local",
            "runtime_path": "real_local",
        },
    ) as transformers_runtime, patch.object(
        engine,
        "_generate_llama_cpp_response",
        side_effect=AssertionError("second backend must not run"),
    ) as llama_cpp_runtime:
        engine.generate(
            [{"role": "user", "content": "run once"}],
            metadata={"enable_real_local": True},
        )

    generation = engine.status()["generation"]
    assert generation["runtime_attempt_count"] == 1
    assert generation["attempted_backends"] == ["transformers"]
    assert "previous_runtime_failures" not in generation
    transformers_runtime.assert_called_once()
    llama_cpp_runtime.assert_not_called()


def test_llama_cpp_status_discards_command_and_prompt_fields() -> None:
    engine = InferenceEngine(
        InferenceConfig(base_model="local-default", backend="llama_cpp")
    )
    private_marker = "PHASE85_LLAMA_COMMAND_PRIVATE_MARKER"
    engine.backend_plan = {"selected_backend": "llama_cpp"}
    with patch.object(
        engine,
        "_generate_llama_cpp_response",
        return_value={
            "text": "llama.cpp response",
            "raw_text": "raw llama.cpp response",
            "command": ["llama-cli", "-p", f"private prompt {private_marker}"],
            "prompt": f"private prompt {private_marker}",
            "formatted_prompt": f"formatted private prompt {private_marker}",
            "served_by": "local",
            "runtime_path": "llama_cpp",
        },
    ):
        output = engine.generate(
            [{"role": "user", "content": private_marker}],
            metadata={"enable_real_local": True},
        )

    generation = engine.status()["generation"]
    serialized = json.dumps(engine.status(), ensure_ascii=False, sort_keys=True)
    assert output == "llama.cpp response"
    assert generation["runtime_attempt_count"] == 1
    assert generation["attempted_backends"] == ["llama_cpp"]
    for key in ("text", "raw_text", "command", "prompt", "formatted_prompt"):
        assert key not in generation
    assert private_marker not in serialized


def test_engine_status_discards_adapter_failure_reason_text() -> None:
    engine = InferenceEngine(
        InferenceConfig(base_model="local-default", backend="transformers")
    )
    private_marker = "PHASE85_ADAPTER_FAILURE_PRIVATE_MARKER"
    with patch.object(
        engine,
        "_generate_real_response",
        return_value={
            "text": "runtime response",
            "served_by": "local",
            "runtime_path": "real_local",
            "adapter_reason": f"adapter failure {private_marker}",
        },
    ):
        engine.generate(
            [{"role": "user", "content": "run adapter"}],
            metadata={"enable_real_local": True},
        )

    generation = engine.status()["generation"]
    assert "adapter_reason" not in generation
    assert private_marker not in json.dumps(engine.status(), ensure_ascii=False)


def test_uncontracted_response_does_not_persist_raw_or_normalized_output() -> None:
    engine = InferenceEngine(
        InferenceConfig(base_model="local-default", backend="transformers")
    )
    private_marker = "PHASE85_UNCONTRACTED_PRIVATE_MARKER"
    with patch.object(
        engine,
        "_generate_real_response",
        return_value={
            "text": f"normal response {private_marker}",
            "raw_text": f"raw response {private_marker}",
            "raw_output": f"raw output {private_marker}",
            "normalized_output": f"normalized output {private_marker}",
            "served_by": "local",
            "runtime_path": "real_local",
        },
    ):
        output = engine.generate(
            [{"role": "user", "content": private_marker}],
            metadata={"enable_real_local": True},
        )

    generation = engine.status()["generation"]
    serialized_status = json.dumps(engine.status(), ensure_ascii=False, sort_keys=True)
    serialized_last_generation = json.dumps(
        engine.last_generation_info, ensure_ascii=False, sort_keys=True
    )
    assert output == f"normal response {private_marker}"
    for key in ("text", "raw_text", "raw_output", "normalized_output"):
        assert key not in generation
    assert generation["raw_output_persisted"] is False
    for serialized in (serialized_status, serialized_last_generation):
        assert private_marker not in serialized


def test_direct_template_path_records_explicit_mock_attempt() -> None:
    engine = InferenceEngine(InferenceConfig(base_model="local-default"))

    engine.generate([{"role": "user", "content": "use template"}])

    generation = engine.status()["generation"]
    assert generation["runtime_attempt_count"] == 1
    assert generation["attempted_backends"] == ["mock"]


def test_runtime_failure_status_never_persists_exception_text() -> None:
    engine = InferenceEngine(
        InferenceConfig(base_model="local-default", backend="transformers")
    )
    private_marker = "PHASE85_RUNTIME_FAILURE_PRIVATE_MARKER"
    with patch.object(
        engine,
        "_generate_real_response",
        side_effect=RuntimeError(f"transformers failure {private_marker}"),
    ), patch.object(
        engine,
        "_generate_llama_cpp_response",
        side_effect=RuntimeError(f"llama failure {private_marker}"),
    ):
        engine.generate(
            [{"role": "user", "content": private_marker}],
            metadata={"enable_real_local": True},
        )

    generation = engine.status()["generation"]
    serialized = json.dumps(engine.status(), ensure_ascii=False, sort_keys=True)
    assert generation["fallback_reason"] == "transformers: RuntimeError"
    assert generation["previous_runtime_failures"] == [
        "transformers: RuntimeError",
        "llama_cpp: RuntimeError",
    ]
    assert private_marker not in serialized


def test_llama_cpp_timeout_exception_does_not_repeat_command_prompt() -> None:
    engine = InferenceEngine(
        InferenceConfig(base_model="local-default", backend="llama_cpp")
    )
    private_marker = "PHASE85_TIMEOUT_COMMAND_PRIVATE_MARKER"
    with patch(
        "pfe_core.inference.engine._resolve_llama_cpp_runtime_binary",
        return_value={"available": True, "path": "/tmp/llama-cli"},
    ), patch(
        "pfe_core.inference.engine._resolve_base_gguf_path",
        return_value={"available": True, "path": "/tmp/model.gguf"},
    ), patch(
        "pfe_core.inference.engine.subprocess.run",
        side_effect=subprocess.TimeoutExpired(
            cmd=["llama-cli", "-p", private_marker],
            timeout=30,
        ),
    ):
        with pytest.raises(InferenceError) as error:
            engine._generate_llama_cpp_response(
                [{"role": "user", "content": private_marker}],
                resolved_base_model="/tmp/model.gguf",
            )

    assert str(error.value) == "llama.cpp runtime timed out"
    assert private_marker not in str(error.value)
