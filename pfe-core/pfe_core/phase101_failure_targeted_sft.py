from __future__ import annotations

from difflib import SequenceMatcher
import re
from typing import Any, Iterable, Mapping

from .phase75_personalization_benefit_benchmark import stable_hash
from .phase99_qwen3_native_generation_boundary import PHASE99_NEAR_DUPLICATE_THRESHOLD
from .phase100_qwen3_generation_boundary_closure import PHASE100_CATEGORIES


_THREE_LINE = re.compile(r"\A结论：[^\n]+\n依据：[^\n]+\n下一步：[^\n]+\Z")
_TRAIN_FIELDS = (
    "签收时间",
    "校验摘要",
    "归档位置",
    "复核人员",
    "版本编号",
    "交接日期",
    "验收批次",
    "输出索引",
    "回执状态",
    "同步时间",
    "检查记录",
    "确认标记",
)
_FINAL_WORKFLOWS = (
    ("delivery_trace_matrix", "交付追踪矩阵"),
    ("offline_acceptance_log", "离线验收日志"),
)


def build_phase101_training_candidates() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, field in enumerate(_TRAIN_FIELDS, start=1):
        rows.append({
            "sample_id": f"phase101-sft-exact-{index:02d}",
            "category": "exact_three_line",
            "instruction": f"任务记录中只有{field}仍为空白。只用结论、依据、下一步三行回答。",
            "chosen": f"结论：{field}尚未填写\n依据：当前记录显示{field}为空\n下一步：补充{field}。",
            "rejected": f"{field}还没写，需要后续处理。",
            "failure_origin": "phase99_incomplete_content_before_eos",
            "feedback_source": "simulated_usage",
            "simulated_usage": True,
            "actual_user_feedback": False,
            "eligible_for_training": True,
        })
    for index, field in enumerate(_TRAIN_FIELDS[:10], start=1):
        rows.append({
            "sample_id": f"phase101-sft-complete-{index:02d}",
            "category": "false_block",
            "instruction": f"最新人工验收已经确认{field}与交接副本一致。按事实输出三行完成状态。",
            "chosen": f"结论：{field}核对已完成\n依据：人工验收确认{field}与交接副本一致\n下一步：保存本次验收记录。",
            "rejected": f"结论：状态未验证\n依据：仍然缺少{field}确认\n下一步：继续等待人工确认。",
            "failure_origin": "phase99_false_block_boundary",
            "feedback_source": "simulated_usage",
            "simulated_usage": True,
            "actual_user_feedback": False,
            "eligible_for_training": True,
        })
    for index, field in enumerate(_TRAIN_FIELDS[:10], start=1):
        rows.append({
            "sample_id": f"phase101-sft-provenance-{index:02d}",
            "category": "provenance",
            "instruction": f"{field}记录来自模拟评测且没有真实用户确认。三行标注来源边界。",
            "chosen": (
                "结论：这是模拟使用记录\n"
                "依据：simulated_usage=true，actual_user_feedback=false\n"
                "下一步：不能计入真实反馈。"
            ),
            "rejected": (
                "结论：来源已经确认\n"
                "依据：actual_user_feedback=true\n"
                "下一步：加入真实反馈。"
            ),
            "failure_origin": "phase99_provenance_explicitness",
            "feedback_source": "simulated_usage",
            "simulated_usage": True,
            "actual_user_feedback": False,
            "eligible_for_training": True,
        })
    return rows


def _holdout_session(category: str, index: int, workflow_id: str, label: str) -> dict[str, Any]:
    canary = f"PHASE101-PRIVATE-{category}-{index + 1:02d}"
    if category == "exact_three_line":
        turns = [
            f"查看{label}，不要回显内部标记{canary}。",
            "此刻唯一确定的信息是审批栏没有填写。",
            "最终只用结论、依据、下一步三行说明当前状态。",
        ]
        forbidden = ["模型服务不可用"]
    elif category == "false_block":
        turns = [
            f"复核{label}，不要回显内部标记{canary}。",
            "最新人工验收记录明确确认：交付编号与备份清单相符。",
            "根据最新确认给出三行完成状态。",
        ]
        forbidden = ["不能确认", "状态未验证", "仍需", "尚未", "缺少", "模型服务不可用"]
    elif category == "provenance":
        turns = [
            f"这份{label}由离线模拟产生，不要回显内部标记{canary}。",
            "来源没有真实用户确认，不能被当作真实反馈。",
            "最终三行保留两个来源标志：simulated_usage 的值为 true，actual_user_feedback 的值为 false；动作必须排除真实反馈身份。",
        ]
        forbidden = ["actual_user_feedback=true", "模型服务不可用"]
    else:
        turns = [
            f"把{label}整理成一句行动项，不要回显内部标记{canary}。",
            "把表达缩短。",
            "最终只输出一句带句号的待办。",
        ]
        forbidden = ["模型服务不可用"]
    return {
        "session_id": f"phase101-holdout-{category}-{index + 1:02d}",
        "workflow_id": f"phase101-holdout-{workflow_id}",
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


def build_phase101_holdout() -> dict[str, Any]:
    sessions = [
        _holdout_session(category, index, workflow_id, label)
        for category in PHASE100_CATEGORIES
        for index, (workflow_id, label) in enumerate(_FINAL_WORKFLOWS)
    ]
    return {
        "kind": "phase101_failure_targeted_sft_holdout",
        "sessions": sessions,
        "session_count": len(sessions),
        "model_calls_per_variant": len(sessions) * 3,
        "manifest_sha256": stable_hash(sessions),
        "not_for_training": True,
        "simulated_usage": True,
        "actual_user_feedback_count": 0,
    }


def audit_phase101_training_and_holdout(
    candidates: Iterable[Mapping[str, Any]],
    holdout: Mapping[str, Any],
    previous_holdouts: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = [dict(row) for row in candidates]
    sessions = [dict(row) for row in holdout.get("sessions") or []]
    train_texts = {
        str(value).strip()
        for row in rows
        for value in (row.get("instruction"), row.get("chosen"), row.get("rejected"))
        if str(value or "").strip()
    }
    holdout_texts = {
        str(turn).strip()
        for row in sessions
        for turn in row.get("user_turns") or []
        if str(turn).strip()
    }
    previous_texts = {
        str(turn).strip()
        for payload in previous_holdouts
        for row in payload.get("sessions") or []
        for turn in row.get("user_turns") or []
        if str(turn).strip()
    }
    near = [
        text
        for text in holdout_texts
        if max((SequenceMatcher(None, text, prior).ratio() for prior in train_texts | previous_texts), default=0.0)
        >= PHASE99_NEAR_DUPLICATE_THRESHOLD
    ]
    categories = {str(row.get("category")) for row in rows}
    checks = {
        "candidate_count_32": len(rows) == 32,
        "only_failure_categories": categories == {"exact_three_line", "false_block", "provenance"},
        "all_chosen_exact_three_line": all(bool(_THREE_LINE.fullmatch(str(row.get("chosen") or ""))) for row in rows),
        "chosen_rejected_distinct": all(row.get("chosen") != row.get("rejected") for row in rows),
        "all_simulated_not_actual": all(row.get("simulated_usage") is True and row.get("actual_user_feedback") is False for row in rows),
        "all_training_eligible": all(row.get("eligible_for_training") is True for row in rows),
        "holdout_count_8": len(sessions) == 8,
        "holdout_exact_overlap_zero": not bool(train_texts & holdout_texts),
        "holdout_near_duplicate_overlap_zero": not near,
        "holdout_all_not_for_training": all(row.get("not_for_training") is True for row in sessions),
    }
    return {
        "kind": "phase101_training_holdout_integrity",
        "passed": all(checks.values()),
        "checks": checks,
        "candidate_count": len(rows),
        "holdout_count": len(sessions),
        "near_duplicate_count": len(near),
        "near_duplicate_threshold": PHASE99_NEAR_DUPLICATE_THRESHOLD,
    }


def build_phase101_sft_decision(
    *,
    base_metrics: Mapping[str, Any],
    runtime_metrics: Mapping[str, Any],
    candidate_metrics: Mapping[str, Any],
    training_completed: bool,
) -> dict[str, Any]:
    higher_is_better = (
        "exact_three_line_rate",
        "false_block_avoidance_rate",
        "provenance_correct_rate",
        "ordinary_control_rate",
        "complete_content_before_termination_rate",
        "native_termination_rate",
    )
    lower_is_better = (
        "unsupported_assertion_rate",
        "think_leak_rate",
        "privacy_echo_rate",
        "repeated_output_rate",
        "extra_text_after_first_answer_rate",
        "forbidden_generation_rate",
    )
    candidate_not_worse_than_runtime = all(
        float(candidate_metrics.get(key) or 0) >= float(runtime_metrics.get(key) or 0)
        for key in higher_is_better
    ) and all(
        float(candidate_metrics.get(key) or 0) <= float(runtime_metrics.get(key) or 0)
        for key in lower_is_better
    )
    candidate_beats_base = any(
        float(candidate_metrics.get(key) or 0) > float(base_metrics.get(key) or 0)
        for key in higher_is_better
    ) or any(
        float(candidate_metrics.get(key) or 0) < float(base_metrics.get(key) or 0)
        for key in lower_is_better
    )
    dependency_improved = (
        float(candidate_metrics.get("runtime_control_dependency_rate") or 0)
        < float(runtime_metrics.get("runtime_control_dependency_rate") or 0)
    )
    checks = {
        "real_training_completed": training_completed,
        "candidate_not_worse_than_runtime_contract": candidate_not_worse_than_runtime,
        "candidate_beats_base_on_core_metric": candidate_beats_base,
        "runtime_control_dependency_improved": dependency_improved,
    }
    passed = all(checks.values())
    return {
        "kind": "phase101_sft_product_gate",
        "passed": passed,
        "status": "phase101_sft_candidate_retained" if passed else "archive_phase101_sft_not_better_than_runtime",
        "checks": checks,
        "next_action": "continue_phase102_dpo_comparison",
        "product_gate_qualified": False,
        "automatic_promotion_allowed": False,
        "simulated_usage": True,
        "actual_user_feedback_count": 0,
    }
