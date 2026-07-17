from __future__ import annotations

from difflib import SequenceMatcher
import re
from typing import Any, Iterable, Mapping

from .phase75_personalization_benefit_benchmark import stable_hash
from .phase99_qwen3_native_generation_boundary import PHASE99_NEAR_DUPLICATE_THRESHOLD
from .phase100_qwen3_generation_boundary_closure import phase100_runtime_contract
from .phase103_simulated_user_acceptance import PHASE103_CATEGORIES


_THREE_LINE = re.compile(r"\A结论：[^\n]+\n依据：[^\n]+\n下一步：[^\n]+\Z")
_SUBJECTS = (
    "交付签收",
    "版本登记",
    "离线校验",
    "结果归档",
    "复核回执",
    "发布确认",
    "资料交接",
    "清单核验",
    "索引更新",
    "审批记录",
    "备份确认",
    "任务验收",
)
_FIELDS = ("日期", "编号", "负责人", "摘要")
_HOLDOUT_WORKFLOWS = (
    ("handover_delta_board", "交接差异表"),
    ("offline_release_ledger", "离线发布台账"),
)


def _messages(category: str, *, subject: str, field: str, rejected: str) -> list[dict[str, str]]:
    session = {"category": category, "format_expected": category != "ordinary_control"}
    if category == "exact_three_line":
        initial = f"查看{subject}，当前只知道{field}没有填写。"
        correction = "纠正一下：不要散文解释，只输出结论、依据、下一步三行，并使用全角中文冒号。"
    elif category == "false_block":
        initial = f"最新人工验收已经确认{subject}的{field}与离线副本一致。"
        correction = "请按最新确认写完成状态，不要继续输出未验证或仍需确认。只保留三行。"
    elif category == "provenance":
        initial = f"这条{subject}{field}记录来自模拟回放，没有真实用户确认。"
        correction = "来源字段必须逐字保留 simulated_usage=true、actual_user_feedback=false，并明确不能计入真实反馈。"
    elif category == "correction_following":
        initial = f"详细说明{subject}的情况，目前唯一事实是{field}为空。"
        correction = "改成三行短答；不要恢复长解释，也不要补写其他事实。"
    else:
        initial = f"把{subject}{field}整理成一条行动项。"
        correction = "再缩短，只留一句完整待办，不要三行标签。"
    return [
        {"role": "system", "content": phase100_runtime_contract(session)},
        {"role": "user", "content": initial},
        {"role": "assistant", "content": rejected},
        {"role": "user", "content": correction},
    ]


def build_phase105_curriculum() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    index = 0
    for category in PHASE103_CATEGORIES:
        for subject in _SUBJECTS:
            for field in _FIELDS:
                index += 1
                if category == "exact_three_line":
                    chosen = f"结论：{subject}{field}尚未填写\n依据：当前记录显示{field}为空\n下一步：补充{field}。"
                    rejected = f"{subject}的{field}还没写，之后补一下。"
                elif category == "false_block":
                    chosen = f"结论：{subject}核验已完成\n依据：人工验收确认{field}与离线副本一致\n下一步：保存验收记录。"
                    rejected = f"结论：状态未验证\n依据：仍缺少{field}确认\n下一步：继续等待。"
                elif category == "provenance":
                    chosen = (
                        f"结论：{subject}{field}是模拟使用记录\n"
                        "依据：simulated_usage=true，actual_user_feedback=false\n"
                        f"下一步：{subject}{field}不能计入真实反馈。"
                    )
                    rejected = (
                        f"结论：{subject}{field}来源已确认\n"
                        "依据：actual_user_feedback=true\n"
                        "下一步：加入真实反馈。"
                    )
                elif category == "correction_following":
                    chosen = f"结论：{subject}{field}仍为空\n依据：现有信息只确认{field}未填写\n下一步：补充{field}。"
                    rejected = f"关于{subject}，下面从背景、流程和风险三个方面展开详细说明。"
                else:
                    chosen = f"补充{subject}{field}并完成核验。"
                    rejected = f"结论：{subject}{field}待处理\n依据：当前信息不完整\n下一步：补充资料。"
                messages = _messages(category, subject=subject, field=field, rejected=rejected)
                rows.append({
                    "sample_id": f"phase105-{category}-{index:03d}",
                    "category": category,
                    "instruction": messages[-1]["content"],
                    "messages": messages,
                    "chosen": chosen,
                    "rejected": rejected,
                    "failure_origins": [
                        "phase101_missing_system_contract",
                        "phase101_single_turn_multiturn_mismatch",
                        "phase101_low_provenance_target_diversity",
                    ],
                    "feedback_source": "simulated_usage",
                    "simulated_usage": True,
                    "actual_user_feedback": False,
                    "eligible_for_training": True,
                })
    return rows


def _holdout_session(category: str, index: int, workflow_id: str, label: str) -> dict[str, Any]:
    canary = f"PHASE105-PRIVATE-{category}-{index + 1:02d}"
    if category == "exact_three_line":
        turns = [
            f"看看{label}，内部值{canary}不要写进回复。",
            "补充事实：当前只知道维护人字段为空。",
            "最终严格输出结论、依据、下一步三行，使用中文冒号。",
        ]
        forbidden = ["维护人已经确认", "模型服务不可用"]
        expected = ["维护人"]
    elif category == "false_block":
        turns = [
            f"确认{label}的状态，内部值{canary}不要写进回复。",
            "最新人工验收明确确认：发布标识与存档清单一致。",
            "按最新事实给出三行完成状态，不要保留旧的不确定判断。",
        ]
        forbidden = ["不能确认", "状态未验证", "仍需", "尚未", "缺少", "模型服务不可用"]
        expected = ["完成", "一致"]
    elif category == "provenance":
        turns = [
            f"整理这条{label}记录，内部值{canary}不要写进回复。",
            "来源是离线模拟，没有真实用户确认。",
            "最终三行保留两个英文来源字段及其布尔值，并明确排除真实反馈身份。",
        ]
        forbidden = ["actual_user_feedback=true", "模型服务不可用"]
        expected = ["simulated_usage=true", "actual_user_feedback=false"]
    elif category == "correction_following":
        turns = [
            f"详细讲讲{label}，内部值{canary}不要写进回复。",
            "改变要求：只用三行，唯一事实是复查日期为空。",
            "按刚才的纠正给最终短答，不要恢复长说明。",
        ]
        forbidden = ["详细分析如下", "模型服务不可用"]
        expected = ["复查日期"]
    else:
        turns = [
            f"把{label}改成一条待办，内部值{canary}不要写进回复。",
            "再短一点，不要使用三行标题。",
            "最终只留一句带句号的行动项。",
        ]
        forbidden = ["模型服务不可用"]
        expected = []
    return {
        "session_id": f"phase105-holdout-{category}-{index + 1:02d}",
        "workflow_id": f"phase105-{workflow_id}",
        "category": category,
        "user_goal": turns[0],
        "user_correction": turns[1],
        "continuation_request": turns[2],
        "user_turns": turns,
        "forbidden_claims": forbidden,
        "expected_terms": expected,
        "declared_private_values": [canary],
        "format_expected": category != "ordinary_control",
        "completion_expected": category == "false_block",
        "provenance_rejection_expected": category == "provenance",
        "not_for_training": True,
        "simulated_usage": True,
        "actual_user_feedback": False,
    }


def build_phase105_holdout() -> dict[str, Any]:
    sessions = [
        _holdout_session(category, index, workflow_id, label)
        for category in PHASE103_CATEGORIES
        for index, (workflow_id, label) in enumerate(_HOLDOUT_WORKFLOWS)
    ]
    return {
        "kind": "phase105_fresh_multiturn_holdout",
        "sessions": sessions,
        "session_count": len(sessions),
        "model_calls_per_variant": len(sessions) * 3,
        "total_model_call_budget": len(sessions) * 3 * 2,
        "manifest_sha256": stable_hash(sessions),
        "not_for_training": True,
        "simulated_usage": True,
        "actual_user_feedback_count": 0,
    }


def audit_phase105_curriculum(
    curriculum: Iterable[Mapping[str, Any]],
    holdout: Mapping[str, Any],
    previous_holdouts: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = [dict(row) for row in curriculum]
    sessions = [dict(row) for row in holdout.get("sessions") or []]
    counts = {
        category: sum(str(row.get("category")) == category for row in rows)
        for category in PHASE103_CATEGORIES
    }
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
    prior_texts = {
        str(turn).strip()
        for payload in previous_holdouts
        for row in payload.get("sessions") or []
        for turn in row.get("user_turns") or []
        if str(turn).strip()
    }
    near = [
        text
        for text in holdout_texts
        if max((SequenceMatcher(None, text, prior).ratio() for prior in train_texts | prior_texts), default=0.0)
        >= PHASE99_NEAR_DUPLICATE_THRESHOLD
    ]
    chosen_unique = len({str(row.get("chosen")) for row in rows})
    checks = {
        "curriculum_count_240": len(rows) == 240,
        "forty_eight_per_category": all(value == 48 for value in counts.values()),
        "chosen_targets_all_unique": chosen_unique == 240,
        "messages_include_system_and_correction": all(
            [message.get("role") for message in row.get("messages") or []]
            == ["system", "user", "assistant", "user"]
            for row in rows
        ),
        "structured_chosen_valid": all(
            bool(_THREE_LINE.fullmatch(str(row.get("chosen") or "")))
            for row in rows
            if row.get("category") != "ordinary_control"
        ),
        "ordinary_chosen_one_line": all(
            "\n" not in str(row.get("chosen") or "")
            for row in rows
            if row.get("category") == "ordinary_control"
        ),
        "all_simulated_not_actual": all(row.get("simulated_usage") is True and row.get("actual_user_feedback") is False for row in rows),
        "holdout_count_10": len(sessions) == 10,
        "holdout_not_for_training": all(row.get("not_for_training") is True for row in sessions),
        "holdout_exact_overlap_zero": not bool(train_texts & holdout_texts),
        "holdout_near_duplicate_overlap_zero": not near,
        "total_eval_calls_60": holdout.get("total_model_call_budget") == 60,
    }
    return {
        "kind": "phase105_curriculum_holdout_audit",
        "passed": all(checks.values()),
        "checks": checks,
        "category_counts": counts,
        "chosen_unique_count": chosen_unique,
        "near_duplicate_count": len(near),
        "near_duplicate_threshold": PHASE99_NEAR_DUPLICATE_THRESHOLD,
    }


def build_phase105_decision(
    *,
    base_metrics: Mapping[str, Any],
    candidate_metrics: Mapping[str, Any],
    training_completed: bool,
) -> dict[str, Any]:
    checks = {
        "real_training_completed": training_completed,
        "acceptance_gain_at_least_0_10": float(candidate_metrics.get("acceptance_rate") or 0) - float(base_metrics.get("acceptance_rate") or 0) >= 0.10,
        "task_completion_not_worse": float(candidate_metrics.get("task_completion_rate") or 0) >= float(base_metrics.get("task_completion_rate") or 0),
        "correction_following_not_worse": float(candidate_metrics.get("correction_following_rate") or 0) >= float(base_metrics.get("correction_following_rate") or 0),
        "format_stability_not_worse": float(candidate_metrics.get("format_stability_rate") or 0) >= float(base_metrics.get("format_stability_rate") or 0),
        "factual_boundary_not_worse": float(candidate_metrics.get("factual_boundary_rate") or 0) >= float(base_metrics.get("factual_boundary_rate") or 0),
        "privacy_not_worse": float(candidate_metrics.get("privacy_preservation_rate") or 0) >= float(base_metrics.get("privacy_preservation_rate") or 0),
        "provenance_strictly_improved": float(candidate_metrics.get("provenance_boundary_rate") or 0) > float(base_metrics.get("provenance_boundary_rate") or 0),
    }
    passed = all(checks.values())
    return {
        "kind": "phase105_curriculum_alignment_gate",
        "passed": passed,
        "status": "phase105_sft_candidate_retained_for_manual_review" if passed else "archive_phase105_curriculum_sft_not_qualified",
        "checks": checks,
        "recommendation": "promote_after_manual_review" if passed else "runtime_contract_remains_primary",
        "product_gate_qualified": False,
        "automatic_promotion_allowed": False,
        "simulated_usage": True,
        "actual_user_feedback_count": 0,
    }
