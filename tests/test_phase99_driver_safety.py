from __future__ import annotations

import importlib.util
from pathlib import Path

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "tools/phase99_qwen3_native_generation_boundary.py"
SPEC = importlib.util.spec_from_file_location("phase99_driver", DRIVER)
assert SPEC and SPEC.loader
driver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(driver)


def test_phase99_uses_only_local_qwen3_4b_and_24_calls() -> None:
    assert driver.MODEL_PATH == ROOT / "models/Qwen3-4B"
    assert driver.GENERATION_PROTOCOL["model_call_budget"] == 24
    assert driver.GENERATION_PROTOCOL["post_hoc_truncation_allowed"] is False


def test_real_qwen3_tokenizer_has_frozen_no_think_and_stop_controls() -> None:
    tokenizer = AutoTokenizer.from_pretrained(str(driver.MODEL_PATH), local_files_only=True)
    prompt = driver.render_qwen3_no_think_prompt(tokenizer, [{"role": "user", "content": "test"}])

    assert prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert len(driver.qwen3_bad_words_ids(tokenizer)) == 7
    assert driver.qwen3_eos_token_ids(tokenizer) == [151645, 151643]


def test_parser_exposes_no_external_provider_variant() -> None:
    parser = driver._parser()
    assert "variant" not in parser.format_help().lower()
