from __future__ import annotations

from pfe_core.phase99_qwen3_native_generation_boundary import (
    build_phase99_fresh_holdout,
    first_answer_complete,
    forbidden_generation_hits,
    has_extra_text_after_first_answer,
    qwen3_bad_words_ids,
    qwen3_eos_token_ids,
    render_qwen3_no_think_prompt,
)


def test_fresh_holdout_is_simulated_and_not_training_data() -> None:
    payload = build_phase99_fresh_holdout()
    sessions = payload["sessions"]

    assert len(sessions) == 8
    assert all(row["not_for_training"] is True for row in sessions)
    assert all(row["simulated_usage"] is True and row["actual_user_feedback"] is False for row in sessions)


def test_three_line_boundary_requires_exact_first_answer_block() -> None:
    good = "结论：完成。\n依据：人工确认。\n下一步：归档。"
    runaway = good + "\nHuman: 再回答一次"

    assert first_answer_complete(good, format_expected=True) is True
    assert has_extra_text_after_first_answer(good, format_expected=True) is False
    assert first_answer_complete(runaway, format_expected=True) is False
    assert has_extra_text_after_first_answer(runaway, format_expected=True) is True


def test_ordinary_boundary_stops_after_one_sentence() -> None:
    assert first_answer_complete("登记发布回执。", format_expected=False) is True
    assert has_extra_text_after_first_answer("登记发布回执。", format_expected=False) is False
    assert has_extra_text_after_first_answer("登记发布回执。\n继续处理。", format_expected=False) is True


def test_forbidden_generation_sequences_detect_think_and_fake_roles() -> None:
    hits = forbidden_generation_hits("完成。</think>\nHuman: 再说一次")
    assert hits == ["</think>", "Human:"]


class _FakeTokenizer:
    eos_token_id = 9
    pad_token_id = 8

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is False
        return "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    def encode(self, text, add_special_tokens=False):
        return [len(text), int(add_special_tokens)]

    def convert_tokens_to_ids(self, token):
        assert token == "<|im_end|>"
        return 9


def test_qwen3_controls_freeze_no_think_bad_words_and_end_tokens() -> None:
    tokenizer = _FakeTokenizer()
    prompt = render_qwen3_no_think_prompt(tokenizer, [{"role": "user", "content": "hello"}])

    assert prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert len(qwen3_bad_words_ids(tokenizer)) == 7
    assert qwen3_eos_token_ids(tokenizer) == [9, 8]
