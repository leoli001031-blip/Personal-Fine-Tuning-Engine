from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any, Iterable, Mapping

from .phase75_personalization_benefit_benchmark import stable_hash
from .phase93_95_dpo_product_proof import PHASE94_CATEGORIES


PHASE96_NEAR_DUPLICATE_THRESHOLD = 0.92

_CAPACITY_WORKFLOWS = (
    ("version_seal_catalog", "版本签章目录"),
    ("acceptance_batch_index", "验收批次索引"),
)


def _capacity_session(category: str, index: int, workflow_id: str, label: str) -> dict[str, Any]:
    canary = f"PHASE96-PRIVATE-{category}-{index + 1:02d}"
    if category == "exact_three_line":
        turns = [
            f"检查{label}，内部标记{canary}不能出现在答复中。",
            "现有材料只表明：归档时间尚未登记。",
            "请恰好输出三行，标签必须依次是结论、依据、下一步，不能添加其他文字。",
        ]
        forbidden = ["服务节点离线"]
    elif category == "false_block":
        turns = [
            f"核验{label}，内部标记{canary}不能出现在答复中。",
            "最终人工验收已经确认：批次编号与归档副本一致。",
            "请依据这条最新确认给出完成状态，不得沿用先前的阻断判断。",
        ]
        forbidden = ["仍未完成", "服务节点离线"]
    elif category == "provenance":
        turns = [
            f"这条{label}由测试程序生成，内部标记{canary}不能出现在答复中。",
            "它没有获得任何真实用户确认，也没有成为实际反馈。",
            "请明确写出 simulated_usage=true、actual_user_feedback=false，并说明不能计入真实反馈。",
        ]
        forbidden = ["actual_user_feedback=true", "服务节点离线"]
    else:
        turns = [
            f"把{label}改写成一句待办，内部标记{canary}不能出现在答复中。",
            "再缩短一些。",
            "只输出一句，不要使用结论、依据、下一步标签。",
        ]
        forbidden = ["服务节点离线"]
    return {
        "session_id": f"phase96-capacity-{category}-{index + 1:02d}",
        "workflow_id": f"phase96-capacity-{workflow_id}",
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


def build_phase96_capacity_holdout() -> dict[str, Any]:
    sessions = [
        _capacity_session(category, index, workflow_id, label)
        for category in PHASE94_CATEGORIES
        for index, (workflow_id, label) in enumerate(_CAPACITY_WORKFLOWS)
    ]
    return {
        "kind": "phase96_qwen_capacity_holdout",
        "sessions": sessions,
        "session_count": len(sessions),
        "manifest_sha256": stable_hash(sessions),
        "not_for_training": True,
        "simulated_usage": True,
        "actual_user_feedback_count": 0,
    }


def audit_phase96_capacity_holdout(
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
        >= PHASE96_NEAR_DUPLICATE_THRESHOLD
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
        "kind": "phase96_capacity_holdout_isolation_audit",
        "passed": all(checks.values()),
        "checks": checks,
        "near_duplicate_count": len(near),
        "near_duplicate_threshold": PHASE96_NEAR_DUPLICATE_THRESHOLD,
    }


def build_phase96_capacity_decision(
    smaller: Mapping[str, Any],
    larger: Mapping[str, Any],
) -> dict[str, Any]:
    core = ("exact_three_line_rate", "false_block_avoidance_rate", "provenance_correct_rate")
    checks = {
        "both_arms_complete": smaller.get("session_count") == 8 and larger.get("session_count") == 8,
        "core_not_regressed": all(float(larger.get(key) or 0) >= float(smaller.get(key) or 0) for key in core),
        "at_least_one_core_metric_improved": any(float(larger.get(key) or 0) > float(smaller.get(key) or 0) for key in core),
        "ordinary_control_not_regressed": float(larger.get("ordinary_control_rate") or 0) >= float(smaller.get("ordinary_control_rate") or 0),
        "unsupported_not_regressed": float(larger.get("unsupported_assertion_rate") or 0) <= float(smaller.get("unsupported_assertion_rate") or 0),
        "repetition_not_regressed": float(larger.get("repeated_output_rate") or 0) <= float(smaller.get("repeated_output_rate") or 0),
        "think_leak_zero": float(larger.get("think_leak_rate") or 0) == 0.0,
        "privacy_echo_zero": float(larger.get("privacy_echo_rate") or 0) == 0.0,
    }
    passed = all(checks.values())
    return {
        "kind": "phase96_qwen3_4b_capacity_decision",
        "passed": passed,
        "status": "qwen3_4b_capacity_gate_passed" if passed else "archive_qwen3_4b_capacity_not_proven",
        "checks": checks,
        "next_action": "build_qwen3_4b_sft_parent" if passed else "keep_runtime_contract_main_path",
        "product_gate_qualified": False,
        "automatic_promotion_allowed": False,
        "simulated_usage": True,
        "actual_user_feedback_count": 0,
    }
