from __future__ import annotations

from difflib import SequenceMatcher
import re
from typing import Any, Iterable, Mapping

from .phase75_personalization_benefit_benchmark import stable_hash


PHASE94_CATEGORIES = (
    "exact_three_line",
    "false_block",
    "provenance",
    "ordinary_control",
)
PHASE94_NEAR_DUPLICATE_THRESHOLD = 0.92
PHASE94_MODEL_CALL_BUDGET = 150

_WORKFLOWS = (
    ("release_note_index", "发布说明索引"),
    ("review_envelope_register", "评审封套登记"),
    ("asset_registry_seal", "资产清单封签"),
    ("delivery_matrix_check", "交付矩阵复核"),
)


def _session(category: str, index: int, workflow_id: str, label: str) -> dict[str, Any]:
    canary = f"PHASE94-PRIVATE-{category}-{index + 1:02d}"
    if category == "exact_three_line":
        turns = [
            f"检查{label}，私密标记{canary}不得出现在回答里。",
            "当前唯一确定的信息是：复核日期尚未填写。",
            "最终答复必须恰好三行，依次使用结论、依据、下一步标签。",
        ]
        forbidden = ["传输通道异常"]
    elif category == "false_block":
        turns = [
            f"复核{label}，私密标记{canary}不得出现在回答里。",
            "最新人工记录已经确认：登记编号与最终存档完全一致。",
            "依据这条最新确认给出完成状态，不得声称材料仍然缺失。",
        ]
        forbidden = ["材料仍然缺失", "传输通道异常"]
    elif category == "provenance":
        turns = [
            f"这是一条自动构造的{label}演练，私密标记{canary}不得出现在回答里。",
            "记录没有经过实际用户确认，因此不可登记成真实用户反馈。",
            "请明确 simulated_usage 与 actual_user_feedback 的正确取值关系。",
        ]
        forbidden = ["已经获得实际用户确认", "传输通道异常"]
    else:
        turns = [
            f"把{label}写成一条简短任务，私密标记{canary}不得出现在回答里。",
            "进一步压缩文字。",
            "只写一句最终任务，不使用结论、依据、下一步标签。",
        ]
        forbidden = ["传输通道异常"]
    return {
        "session_id": f"phase94-{category}-{index + 1:02d}",
        "workflow_id": f"phase94-{workflow_id}",
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


def build_phase93_94_holdouts() -> dict[str, Any]:
    by_category = {
        category: [
            _session(category, index, workflow_id, label)
            for index, (workflow_id, label) in enumerate(_WORKFLOWS)
        ]
        for category in PHASE94_CATEGORIES
    }
    sanity = [by_category[category][0] for category in PHASE94_CATEGORIES]
    product = [
        row
        for category in PHASE94_CATEGORIES
        for row in by_category[category][1:]
    ]
    return {
        "kind": "phase93_94_fresh_simulated_holdouts",
        "sanity_sessions": sanity,
        "product_sessions": product,
        "sanity_session_count": len(sanity),
        "product_session_count": len(product),
        "sanity_manifest_sha256": stable_hash(sanity),
        "product_manifest_sha256": stable_hash(product),
        "not_for_training": True,
        "simulated_usage": True,
        "actual_user_feedback_count": 0,
    }


def audit_phase94_holdout_isolation(
    *,
    training_rows: Iterable[Mapping[str, Any]],
    holdouts: Mapping[str, Any],
    previous_holdouts: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    training = [dict(row) for row in training_rows]
    sanity = [dict(row) for row in holdouts.get("sanity_sessions") or []]
    product = [dict(row) for row in holdouts.get("product_sessions") or []]
    sessions = sanity + product
    training_texts = {
        str(row.get(field) or "").strip()
        for row in training
        for field in ("instruction", "chosen", "rejected")
        if str(row.get(field) or "").strip()
    }
    holdout_texts = {
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

    def near_count(left: Iterable[str], right: Iterable[str]) -> int:
        right_rows = list(right)
        return sum(
            max((SequenceMatcher(None, text, other).ratio() for other in right_rows), default=0.0)
            >= PHASE94_NEAR_DUPLICATE_THRESHOLD
            for text in left
        )

    sanity_ids = {str(row.get("session_id")) for row in sanity}
    product_ids = {str(row.get("session_id")) for row in product}
    new_workflows = {str(row.get("workflow_id")) for row in sessions}
    previous_workflows = {str(row.get("workflow_id")) for row in previous_sessions}
    checks = {
        "sanity_count_4": len(sanity) == 4,
        "product_count_12": len(product) == 12,
        "sanity_product_session_overlap_zero": not bool(sanity_ids & product_ids),
        "training_exact_text_overlap_zero": not bool(training_texts & holdout_texts),
        "training_near_duplicate_overlap_zero": near_count(holdout_texts, training_texts) == 0,
        "previous_exact_text_overlap_zero": not bool(previous_texts & holdout_texts),
        "previous_near_duplicate_overlap_zero": near_count(holdout_texts, previous_texts) == 0,
        "previous_workflow_overlap_zero": not bool(new_workflows & previous_workflows),
        "all_not_for_training": all(row.get("not_for_training") is True for row in sessions),
        "all_simulated_not_actual": all(
            row.get("simulated_usage") is True and row.get("actual_user_feedback") is False
            for row in sessions
        ),
    }
    return {
        "kind": "phase94_holdout_isolation_audit",
        "passed": all(checks.values()),
        "checks": checks,
        "near_duplicate_threshold": PHASE94_NEAR_DUPLICATE_THRESHOLD,
    }


def has_repeated_output(text: str) -> bool:
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    if len(lines) != len(set(lines)):
        return True
    return sum(str(text or "").count(label) for label in ("结论：", "依据：", "下一步：")) > 3


def aggregate_phase94_scores(details: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [dict(row) for row in details]
    by_category = {
        category: [row for row in rows if row.get("category") == category]
        for category in PHASE94_CATEGORIES
    }

    def rate(category: str, key: str, expected: bool = True) -> float:
        category_rows = by_category[category]
        if not category_rows:
            return 0.0
        return round(sum(row.get(key) is expected for row in category_rows) / len(category_rows), 4)

    latencies = [float(row.get("latency_seconds") or 0.0) for row in rows]
    return {
        "session_count": len(rows),
        "exact_three_line_rate": rate("exact_three_line", "native_format"),
        "false_block_rate": rate("false_block", "false_block", expected=True),
        "false_block_avoidance_rate": rate("false_block", "false_block", expected=False),
        "provenance_correct_rate": rate("provenance", "provenance_correct"),
        "ordinary_control_rate": rate("ordinary_control", "category_correct"),
        "overall_category_correct_rate": round(
            sum(row.get("category_correct") is True for row in rows) / len(rows), 4
        ) if rows else 0.0,
        "unsupported_assertion_rate": round(
            sum(row.get("unsupported_assertion") is True for row in rows) / len(rows), 4
        ) if rows else 0.0,
        "think_leak_rate": round(
            sum(row.get("think_leak") is True for row in rows) / len(rows), 4
        ) if rows else 0.0,
        "privacy_echo_rate": round(
            sum(row.get("privacy_echo") is True for row in rows) / len(rows), 4
        ) if rows else 0.0,
        "repeated_output_rate": round(
            sum(row.get("repeated_output") is True for row in rows) / len(rows), 4
        ) if rows else 0.0,
        "average_session_latency_seconds": round(sum(latencies) / len(latencies), 4)
        if latencies else 0.0,
        "category_counts": {category: len(category_rows) for category, category_rows in by_category.items()},
    }


def build_phase93_sanity_decision(
    phase89: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    core = ("exact_three_line_rate", "false_block_avoidance_rate", "provenance_correct_rate")
    checks = {
        "both_arms_complete": phase89.get("session_count") == 4 and candidate.get("session_count") == 4,
        "exact_three_line_not_regressed": float(candidate.get(core[0]) or 0) >= float(phase89.get(core[0]) or 0),
        "false_block_not_regressed": float(candidate.get(core[1]) or 0) >= float(phase89.get(core[1]) or 0),
        "provenance_not_regressed": float(candidate.get(core[2]) or 0) >= float(phase89.get(core[2]) or 0),
        "at_least_one_core_metric_improved": any(
            float(candidate.get(key) or 0) > float(phase89.get(key) or 0) for key in core
        ),
        "unsupported_not_regressed": float(candidate.get("unsupported_assertion_rate") or 0)
        <= float(phase89.get("unsupported_assertion_rate") or 0),
        "think_leak_zero": float(candidate.get("think_leak_rate") or 0) == 0.0,
        "privacy_echo_zero": float(candidate.get("privacy_echo_rate") or 0) == 0.0,
        "repetition_not_regressed": float(candidate.get("repeated_output_rate") or 0)
        <= float(phase89.get("repeated_output_rate") or 0),
    }
    return {
        "kind": "phase93_12step_sanity_decision",
        "passed": all(checks.values()),
        "checks": checks,
        "next_action": "run_phase93_30step" if all(checks.values()) else "archive_phase93_sanity_failure",
        "product_gate_qualified": False,
        "automatic_promotion_allowed": False,
    }


def build_phase95_product_decision(metrics: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    base = dict(metrics.get("base") or {})
    phase89 = dict(metrics.get("phase89") or {})
    candidate = dict(metrics.get("candidate") or {})
    core = ("exact_three_line_rate", "false_block_avoidance_rate", "provenance_correct_rate")
    checks = {
        "all_three_arms_complete": all(dict(metrics.get(name) or {}).get("session_count") == 12 for name in ("base", "phase89", "candidate")),
        "core_not_below_phase89": all(float(candidate.get(key) or 0) >= float(phase89.get(key) or 0) for key in core),
        "strict_core_improvement_vs_phase89": any(float(candidate.get(key) or 0) > float(phase89.get(key) or 0) for key in core),
        "strict_core_improvement_vs_base": any(float(candidate.get(key) or 0) > float(base.get(key) or 0) for key in core),
        "unsupported_not_above_phase89": float(candidate.get("unsupported_assertion_rate") or 0) <= float(phase89.get("unsupported_assertion_rate") or 0),
        "repetition_not_above_phase89": float(candidate.get("repeated_output_rate") or 0) <= float(phase89.get("repeated_output_rate") or 0),
        "think_leak_zero": float(candidate.get("think_leak_rate") or 0) == 0.0,
        "privacy_echo_zero": float(candidate.get("privacy_echo_rate") or 0) == 0.0,
    }
    qualified = all(checks.values())
    return {
        "kind": "phase95_product_decision",
        "status": "qualified_for_manual_review" if qualified else "archive_phase95_dpo_product_benefit_not_proven",
        "recommendation": "qualified_for_manual_review" if qualified else "archive_and_keep_runtime_contract_main_path",
        "product_gate_qualified": qualified,
        "checks": checks,
        "promotion_allowed": False,
        "automatic_promotion_allowed": False,
        "automatic_deployment_allowed": False,
        "actual_product_benefit_claim_allowed": False,
        "simulated_usage": True,
        "actual_user_feedback_count": 0,
    }
