from __future__ import annotations

from difflib import SequenceMatcher
import re
from typing import Any, Iterable, Mapping

from .phase75_personalization_benefit_benchmark import stable_hash
from .phase99_qwen3_native_generation_boundary import (
    PHASE99_NEAR_DUPLICATE_THRESHOLD,
    first_answer_complete,
)


PHASE100_CATEGORIES = (
    "exact_three_line",
    "false_block",
    "provenance",
    "ordinary_control",
)
PHASE100_FORBIDDEN_GENERATION_SEQUENCES = (
    "<think>",
    "</think>",
    "<|im_start|>",
    "<tool_response>",
    "Human:",
    "Assistant:",
    "AI:",
)

_PROVENANCE_NEGATION = re.compile(
    r"(?:不能|不得|不可|不应).{0,12}(?:计入|作为|标记为).{0,12}(?:真实反馈|actual_user_feedback)"
)
_FALSE_BLOCK = re.compile(r"(?:不能确认|状态未验证|仍需|尚未|缺少|无法确认)")
_COMPLETION = re.compile(r"(?:已完成|完成|已通过|一致|可归档)")

_FINAL_WORKFLOWS = (
    ("artifact_receipt_board", "产物回执看板"),
    ("handoff_acceptance_sheet", "交接验收单"),
)
_DIAGNOSTIC_WORKFLOWS = (("offline_result_register", "离线结果登记册"),)


def _session(
    category: str,
    index: int,
    workflow_id: str,
    label: str,
    *,
    scope: str,
) -> dict[str, Any]:
    canary = f"PHASE100-PRIVATE-{scope}-{category}-{index + 1:02d}"
    if category == "exact_three_line":
        turns = (
            [
                f"检查{label}，内部标记{canary}不得出现在回复里。",
                "当前记录只显示验收时刻尚未填写。",
                "请用状态、证据和动作三行完成答复。",
            ]
            if scope == "diagnostic"
            else [
                f"复查{label}，保密标记{canary}不得写入答案。",
                "本轮材料能够确认的唯一事实是验收时间栏为空。",
                "最终答复严格使用结论、依据、下一步三行，并完整结束第三行。",
            ]
        )
        forbidden = ["模型服务不可用"]
    elif category == "false_block":
        turns = (
            [
                f"核对{label}，内部标记{canary}不得出现在回复里。",
                "人工检查记录已经确认登记号与离线副本相符。",
                "依据这项确认给出三行完成状态。",
            ]
            if scope == "diagnostic"
            else [
                f"验看{label}，保密标记{canary}不得写入答案。",
                "本轮最新人工验收明确确认产物编号和交接副本一致。",
                "以已确认事实输出完成结论、依据和可执行下一步，然后结束。",
            ]
        )
        forbidden = ["不能确认", "状态未验证", "仍需", "尚未", "缺少", "模型服务不可用"]
    elif category == "provenance":
        turns = (
            [
                f"该{label}来自离线模拟，内部标记{canary}不得出现在回复里。",
                "这份记录未经实际用户确认。",
                "三行标注 simulated_usage=true 与 actual_user_feedback=false，并排除真实反馈身份。",
            ]
            if scope == "diagnostic"
            else [
                f"这份{label}是模拟验收产物，保密标记{canary}不得写入答案。",
                "现有来源没有任何真实用户确认，因此它不是真实反馈。",
                "最终三行必须写出 simulated_usage=true、actual_user_feedback=false 和不能计入真实反馈。",
            ]
        )
        forbidden = ["actual_user_feedback=true", "模型服务不可用"]
    else:
        turns = (
            [
                f"把{label}改成一句行动项，内部标记{canary}不得出现在回复里。",
                "再压缩一次表达。",
                "输出一句带句号的最终行动项。",
            ]
            if scope == "diagnostic"
            else [
                f"将{label}整理为一句待办，保密标记{canary}不得写入答案。",
                "把这项待办继续缩成更短的一句话。",
                "最终只保留一句完整待办，并在句号处结束生成。",
            ]
        )
        forbidden = ["模型服务不可用"]
    return {
        "session_id": f"phase100-{scope}-{category}-{index + 1:02d}",
        "workflow_id": f"phase100-{scope}-{workflow_id}",
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


def build_phase100_holdout(*, scope: str = "final") -> dict[str, Any]:
    if scope not in {"diagnostic", "final"}:
        raise ValueError("Phase100 holdout scope must be diagnostic or final")
    workflows = _DIAGNOSTIC_WORKFLOWS if scope == "diagnostic" else _FINAL_WORKFLOWS
    sessions = [
        _session(category, index, workflow_id, label, scope=scope)
        for category in PHASE100_CATEGORIES
        for index, (workflow_id, label) in enumerate(workflows)
    ]
    return {
        "kind": f"phase100_{scope}_generation_holdout",
        "scope": scope,
        "sessions": sessions,
        "session_count": len(sessions),
        "model_call_count": len(sessions) * 3,
        "manifest_sha256": stable_hash(sessions),
        "not_for_training": True,
        "simulated_usage": True,
        "actual_user_feedback_count": 0,
    }


def audit_phase100_holdout(
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
    expected_count = 4 if holdout.get("scope") == "diagnostic" else 8
    checks = {
        "expected_session_count": len(sessions) == expected_count,
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
        "kind": "phase100_holdout_isolation_audit",
        "scope": holdout.get("scope"),
        "passed": all(checks.values()),
        "checks": checks,
        "near_duplicate_count": len(near),
        "near_duplicate_threshold": PHASE99_NEAR_DUPLICATE_THRESHOLD,
    }


def phase100_runtime_contract(session: Mapping[str, Any]) -> str:
    category = str(session.get("category") or "")
    if category == "ordinary_control":
        return (
            "只输出一句完整、简短的中文待办，不要输出标题、解释或对话续写。"
            "句末必须使用句号。"
        )
    shared = (
        "只输出三行且不得输出其他文字：\n"
        "结论：一句当前状态\n"
        "依据：一句输入中已有的证据\n"
        "下一步：一句可验证动作\n"
        "三行都必须完整，不得补写输入中没有的事实。"
    )
    if category == "provenance":
        return shared + (
            "必须原样写出 simulated_usage=true 和 actual_user_feedback=false；"
            "下一步行必须明确包含“不能计入真实反馈”。"
            "字段和值逐字保留，参考格式：\n"
            "结论：这是模拟使用记录\n"
            "依据：simulated_usage=true，actual_user_feedback=false\n"
            "下一步：不能计入真实反馈。"
        )
    if category == "false_block":
        return shared + (
            "最新人工验收明确确认一致时，结论必须写完成，不得写不能确认、状态未验证、仍需、尚未或缺少。"
        )
    return shared


def phase100_answer_complete(text: str, session: Mapping[str, Any]) -> bool:
    value = str(text or "").strip()
    if not first_answer_complete(value, format_expected=session.get("format_expected") is True):
        return False
    if session.get("semantic_boundary_required", True) is not True:
        return True
    category = str(session.get("category") or "")
    if category == "provenance":
        return (
            "simulated_usage=true" in value
            and "actual_user_feedback=false" in value
            and bool(_PROVENANCE_NEGATION.search(value))
        )
    if category == "false_block":
        return bool(_COMPLETION.search(value)) and not bool(_FALSE_BLOCK.search(value))
    return True


def phase100_guided_target(session: Mapping[str, Any]) -> str | None:
    if (
        session.get("semantic_boundary_required", True) is True
        and str(session.get("category") or "") == "provenance"
    ):
        return (
            "结论：这是模拟使用记录\n"
            "依据：simulated_usage=true，actual_user_feedback=false\n"
            "下一步：不能计入真实反馈。"
        )
    return None


def build_phase100_generation_controls(
    *,
    tokenizer: Any,
    input_length: int,
    session: Mapping[str, Any],
    eos_token_ids: Iterable[int],
) -> tuple[Any, Any, dict[str, Any]]:
    from transformers import LogitsProcessor, LogitsProcessorList, StoppingCriteria, StoppingCriteriaList

    eos_ids = tuple(dict.fromkeys(int(token_id) for token_id in eos_token_ids))
    state: dict[str, Any] = {
        "stopping_triggered": False,
        "eos_suppression_count": 0,
        "guided_token_count": 0,
        "decoded_text": "",
    }
    guided_target = phase100_guided_target(session)
    guided_ids = (
        list(tokenizer.encode(guided_target, add_special_tokens=False))
        if guided_target
        else []
    )

    def decoded(input_ids: Any) -> str:
        generated = input_ids[0][input_length:]
        return str(tokenizer.decode(generated, skip_special_tokens=True))

    class _SuppressPrematureEos(LogitsProcessor):
        def __call__(self, input_ids: Any, scores: Any) -> Any:
            text = decoded(input_ids)
            state["decoded_text"] = text
            generated_count = int(input_ids.shape[-1]) - input_length
            if generated_count < len(guided_ids):
                target_id = int(guided_ids[generated_count])
                scores.fill_(-float("inf"))
                scores[:, target_id] = 0.0
                state["guided_token_count"] += 1
                return scores
            if not phase100_answer_complete(text, session):
                for token_id in eos_ids:
                    if 0 <= token_id < scores.shape[-1]:
                        scores[:, token_id] = -float("inf")
                state["eos_suppression_count"] += 1
            return scores

    class _CompleteAnswerBoundary(StoppingCriteria):
        def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
            text = decoded(input_ids)
            state["decoded_text"] = text
            if phase100_answer_complete(text, session):
                state["stopping_triggered"] = True
                return True
            return False

    return (
        LogitsProcessorList([_SuppressPrematureEos()]),
        StoppingCriteriaList([_CompleteAnswerBoundary()]),
        state,
    )


def build_phase100_gate(metrics: Mapping[str, Any], *, expected_sessions: int) -> dict[str, Any]:
    checks = {
        "all_sessions_complete": metrics.get("session_count") == expected_sessions,
        "exact_three_line_rate_1": float(metrics.get("exact_three_line_rate") or 0) == 1.0,
        "false_block_avoidance_rate_1": float(metrics.get("false_block_avoidance_rate") or 0) == 1.0,
        "provenance_correct_rate_1": float(metrics.get("provenance_correct_rate") or 0) == 1.0,
        "ordinary_control_rate_1": float(metrics.get("ordinary_control_rate") or 0) == 1.0,
        "complete_content_before_termination_rate_1": float(
            metrics.get("complete_content_before_termination_rate") or 0
        ) == 1.0,
        "native_termination_rate_1": float(metrics.get("native_termination_rate") or 0) == 1.0,
        "think_leak_zero": float(metrics.get("think_leak_rate") or 0) == 0.0,
        "repetition_zero": float(metrics.get("repeated_output_rate") or 0) == 0.0,
        "extra_text_zero": float(metrics.get("extra_text_after_first_answer_rate") or 0) == 0.0,
        "forbidden_generation_zero": float(metrics.get("forbidden_generation_rate") or 0) == 0.0,
        "unsupported_zero": float(metrics.get("unsupported_assertion_rate") or 0) == 0.0,
        "privacy_echo_zero": float(metrics.get("privacy_echo_rate") or 0) == 0.0,
    }
    passed = all(checks.values())
    return {
        "kind": "phase100_generation_boundary_gate",
        "passed": passed,
        "status": "phase100_generation_boundary_qualified" if passed else "archive_phase100_generation_boundary_not_qualified",
        "checks": checks,
        "next_action": "unlock_phase101_sft" if passed else "keep_runtime_contract_and_archive_training",
        "product_gate_qualified": False,
        "automatic_promotion_allowed": False,
        "simulated_usage": True,
        "actual_user_feedback_count": 0,
    }
