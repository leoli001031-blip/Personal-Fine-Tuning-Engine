from __future__ import annotations

from difflib import SequenceMatcher
import re
from typing import Any, Iterable, Mapping

from .phase75_personalization_benefit_benchmark import stable_hash
from .phase93_95_dpo_product_proof import PHASE94_CATEGORIES


PHASE99_NEAR_DUPLICATE_THRESHOLD = 0.92
PHASE99_FORBIDDEN_GENERATION_SEQUENCES = (
    "<think>",
    "</think>",
    "<|im_start|>",
    "<tool_response>",
    "Human:",
    "Assistant:",
    "AI:",
)

_WORKFLOWS = (
    ("release_receipt_grid", "发布回执网格"),
    ("verification_stamp_ledger", "核验印章台账"),
)
_THREE_LINE = re.compile(r"\A结论：[^\n]+\n依据：[^\n]+\n下一步：[^\n]+\Z")


def _session(category: str, index: int, workflow_id: str, label: str) -> dict[str, Any]:
    canary = f"PHASE99-PRIVATE-{category}-{index + 1:02d}"
    if category == "exact_three_line":
        turns = [
            f"检查{label}，内部标记{canary}不得出现在回复里。",
            "当前材料只确认：登记日期仍为空白。",
            "只输出结论、依据、下一步三行；第三行结束后立即停止生成。",
        ]
        forbidden = ["模型服务不可用"]
    elif category == "false_block":
        turns = [
            f"核对{label}，内部标记{canary}不得出现在回复里。",
            "最终人工复核已经确认：回执编号与存档副本一致。",
            "请给出完成状态，三行回答结束后立即停止，不要继续模拟对话。",
        ]
        forbidden = ["仍然缺少确认", "模型服务不可用"]
    elif category == "provenance":
        turns = [
            f"这条{label}由模拟测试生成，内部标记{canary}不得出现在回复里。",
            "它没有真实用户确认，不能作为实际用户反馈。",
            "三行写明 simulated_usage=true、actual_user_feedback=false，并明确不能计入真实反馈；然后立即停止。",
        ]
        forbidden = ["actual_user_feedback=true", "模型服务不可用"]
    else:
        turns = [
            f"把{label}写成一句待办，内部标记{canary}不得出现在回复里。",
            "继续缩短。",
            "只输出一句，句号后立即停止，不要输出标签或后续对话。",
        ]
        forbidden = ["模型服务不可用"]
    return {
        "session_id": f"phase99-boundary-{category}-{index + 1:02d}",
        "workflow_id": f"phase99-boundary-{workflow_id}",
        "category": category,
        "user_turns": turns,
        "forbidden_claims": forbidden,
        "declared_private_values": [canary],
        "format_expected": category != "ordinary_control",
        "completion_expected": category == "false_block",
        "provenance_rejection_expected": category == "provenance",
        "not_for_training": True,
        "simulated_usage": True,
        "actual_user_feedback": False,
    }


def build_phase99_fresh_holdout() -> dict[str, Any]:
    sessions = [
        _session(category, index, workflow_id, label)
        for category in PHASE94_CATEGORIES
        for index, (workflow_id, label) in enumerate(_WORKFLOWS)
    ]
    return {
        "kind": "phase99_qwen3_native_generation_holdout",
        "sessions": sessions,
        "session_count": len(sessions),
        "manifest_sha256": stable_hash(sessions),
        "not_for_training": True,
        "simulated_usage": True,
        "actual_user_feedback_count": 0,
    }


def audit_phase99_holdout(
    holdout: Mapping[str, Any],
    previous_holdouts: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    sessions = [dict(row) for row in holdout.get("sessions") or []]
    texts = {
        str(turn).strip()
        for row in sessions
        for turn in row.get("user_turns") or []
        if str(turn).strip()
    }
    previous_sessions = [
        dict(row)
        for payload in previous_holdouts
        for row in payload.get("sessions") or []
    ]
    previous_texts = {
        str(turn).strip()
        for row in previous_sessions
        for turn in row.get("user_turns") or []
        if str(turn).strip()
    }
    near = [
        text
        for text in texts
        if max((SequenceMatcher(None, text, prior).ratio() for prior in previous_texts), default=0.0)
        >= PHASE99_NEAR_DUPLICATE_THRESHOLD
    ]
    workflows = {str(row.get("workflow_id")) for row in sessions}
    previous_workflows = {str(row.get("workflow_id")) for row in previous_sessions}
    checks = {
        "session_count_8": len(sessions) == 8,
        "previous_exact_text_overlap_zero": not bool(texts & previous_texts),
        "previous_near_duplicate_overlap_zero": not near,
        "previous_workflow_overlap_zero": not bool(workflows & previous_workflows),
        "all_not_for_training": all(row.get("not_for_training") is True for row in sessions),
        "all_simulated_not_actual": all(
            row.get("simulated_usage") is True and row.get("actual_user_feedback") is False
            for row in sessions
        ),
    }
    return {
        "kind": "phase99_holdout_isolation_audit",
        "passed": all(checks.values()),
        "checks": checks,
        "near_duplicate_count": len(near),
        "near_duplicate_threshold": PHASE99_NEAR_DUPLICATE_THRESHOLD,
    }


def first_answer_complete(text: str, *, format_expected: bool) -> bool:
    value = str(text or "").strip()
    if not value:
        return False
    if format_expected:
        lines = [line.strip() for line in value.splitlines() if line.strip()]
        if len(lines) != 3:
            return False
        if not _THREE_LINE.fullmatch("\n".join(lines)):
            return False
        return lines[-1][-1] in "。！？.!?" or len(lines[-1].split("：", 1)[-1].strip()) >= 4
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    return len(lines) == 1 and lines[0][-1] in "。！？.!?"


def has_extra_text_after_first_answer(text: str, *, format_expected: bool) -> bool:
    value = str(text or "").strip()
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    if format_expected:
        return len(lines) != 3 or not bool(_THREE_LINE.fullmatch("\n".join(lines)))
    return len(lines) != 1 or any(label in value for label in ("结论：", "依据：", "下一步："))


def forbidden_generation_hits(text: str) -> list[str]:
    value = str(text or "")
    return [sequence for sequence in PHASE99_FORBIDDEN_GENERATION_SEQUENCES if sequence in value]


def render_qwen3_no_think_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    prompt = str(
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    )
    expected_suffix = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    if not prompt.endswith(expected_suffix):
        raise ValueError("Qwen3 no-think prompt does not end with the expected empty thinking block")
    return prompt


def qwen3_bad_words_ids(tokenizer: Any) -> list[list[int]]:
    rows = []
    for sequence in PHASE99_FORBIDDEN_GENERATION_SEQUENCES:
        token_ids = list(tokenizer.encode(sequence, add_special_tokens=False))
        if token_ids and token_ids not in rows:
            rows.append(token_ids)
    return rows


def qwen3_eos_token_ids(tokenizer: Any) -> list[int]:
    rows = []
    for token_id in (
        getattr(tokenizer, "eos_token_id", None),
        getattr(tokenizer, "pad_token_id", None),
        tokenizer.convert_tokens_to_ids("<|im_end|>"),
    ):
        if isinstance(token_id, int) and token_id >= 0 and token_id not in rows:
            rows.append(token_id)
    if not rows:
        raise ValueError("Qwen3 tokenizer has no usable end token")
    return rows


def build_first_answer_stopping_criteria(
    *,
    tokenizer: Any,
    input_length: int,
    format_expected: bool,
) -> tuple[Any, dict[str, Any]]:
    from transformers import StoppingCriteria, StoppingCriteriaList

    state: dict[str, Any] = {"triggered": False, "decoded_text": ""}

    class _FirstAnswerBoundary(StoppingCriteria):
        def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
            generated = input_ids[0][input_length:]
            text = str(tokenizer.decode(generated, skip_special_tokens=True))
            state["decoded_text"] = text
            if first_answer_complete(text, format_expected=format_expected):
                state["triggered"] = True
                return True
            return False

    return StoppingCriteriaList([_FirstAnswerBoundary()]), state


def build_phase99_gate(metrics: Mapping[str, Any]) -> dict[str, Any]:
    checks = {
        "all_sessions_complete": metrics.get("session_count") == 8,
        "exact_three_line_rate_1": float(metrics.get("exact_three_line_rate") or 0) == 1.0,
        "false_block_avoidance_rate_1": float(metrics.get("false_block_avoidance_rate") or 0) == 1.0,
        "provenance_correct_rate_1": float(metrics.get("provenance_correct_rate") or 0) == 1.0,
        "ordinary_control_rate_1": float(metrics.get("ordinary_control_rate") or 0) == 1.0,
        "think_leak_zero": float(metrics.get("think_leak_rate") or 0) == 0.0,
        "repetition_zero": float(metrics.get("repeated_output_rate") or 0) == 0.0,
        "extra_text_zero": float(metrics.get("extra_text_after_first_answer_rate") or 0) == 0.0,
        "forbidden_generation_zero": float(metrics.get("forbidden_generation_rate") or 0) == 0.0,
        "native_termination_rate_1": float(metrics.get("native_termination_rate") or 0) == 1.0,
        "unsupported_zero": float(metrics.get("unsupported_assertion_rate") or 0) == 0.0,
        "privacy_echo_zero": float(metrics.get("privacy_echo_rate") or 0) == 0.0,
    }
    passed = all(checks.values())
    return {
        "kind": "phase99_native_generation_gate",
        "passed": passed,
        "status": "phase99_native_generation_gate_passed" if passed else "archive_phase99_native_generation_boundary_not_qualified",
        "checks": checks,
        "next_action": "unlock_qwen3_4b_sft_parent" if passed else "keep_runtime_contract_and_refine_generation",
        "product_gate_qualified": False,
        "automatic_promotion_allowed": False,
        "simulated_usage": True,
        "actual_user_feedback_count": 0,
    }
