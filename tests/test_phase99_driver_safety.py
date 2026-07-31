from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "tools/phase99_qwen3_native_generation_boundary.py"
SPEC = importlib.util.spec_from_file_location("phase99_driver", DRIVER)
assert SPEC and SPEC.loader
driver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(driver)


class _FakeQwen3Tokenizer:
    eos_token_id = 151645
    pad_token_id = 151643

    _encoded_sequences = {
        "<think>": [1],
        "</think>": [2],
        "<|im_start|>": [3],
        "<tool_response>": [4],
        "Human:": [5],
        "Assistant:": [6],
        "AI:": [7],
    }

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> str:
        assert messages == [{"role": "user", "content": "test"}]
        assert tokenize is False
        assert add_generation_prompt is True
        assert enable_thinking is False
        return "<|im_start|>user\ntest<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    def encode(self, sequence: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return list(self._encoded_sequences[sequence])

    def convert_tokens_to_ids(self, token: str) -> int:
        if token == "<|im_end|>":
            return 151645
        return -1


def test_phase99_uses_only_local_qwen3_4b_and_24_calls() -> None:
    assert driver.MODEL_PATH == ROOT / "models/Qwen3-4B"
    assert driver.GENERATION_PROTOCOL["model_call_budget"] == 24
    assert driver.GENERATION_PROTOCOL["post_hoc_truncation_allowed"] is False


def test_qwen3_tokenizer_contract_has_frozen_no_think_and_stop_controls() -> None:
    tokenizer = _FakeQwen3Tokenizer()
    prompt = driver.render_qwen3_no_think_prompt(tokenizer, [{"role": "user", "content": "test"}])

    assert prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert len(driver.qwen3_bad_words_ids(tokenizer)) == 7
    assert driver.qwen3_eos_token_ids(tokenizer) == [151645, 151643]


def test_parser_exposes_no_external_provider_variant() -> None:
    parser = driver._parser()
    assert "variant" not in parser.format_help().lower()
