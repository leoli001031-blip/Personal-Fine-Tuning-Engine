from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping

from .phase75_personalization_benefit_benchmark import stable_hash
from .phase103_simulated_user_acceptance import PHASE103_CATEGORIES
from .phase105_qwen3_curriculum_alignment import audit_phase105_curriculum


_WORKFLOWS = (
    ("audit_change_register", "审核变更登记"),
    ("release_delta_index", "发布差异索引"),
)


def _session(category: str, index: int, workflow_id: str, label: str) -> dict[str, Any]:
    canary = f"PHASE106-PRIVATE-{category}-{index + 1:02d}"
    if category == "exact_three_line":
        turns = [
            f"检查{label}，不要输出内部值{canary}。",
            "目前唯一可确认的信息是审核人字段未填写。",
            "最后严格用结论、依据、下一步三行回答，标签使用中文冒号。",
        ]
        forbidden = ["审核人已经确认", "模型服务不可用"]
        expected = ["审核人"]
    elif category == "false_block":
        turns = [
            f"核验{label}，不要输出内部值{canary}。",
            "最新人工验收明确确认：变更编号与离线存档一致。",
            "按最新确认写三行完成状态，不要沿用未验证结论。",
        ]
        forbidden = ["不能确认", "状态未验证", "仍需", "尚未", "缺少", "模型服务不可用"]
        expected = ["完成", "一致"]
    elif category == "provenance":
        turns = [
            f"整理{label}的来源，内部值{canary}不能回显。",
            "这份内容由模拟流程生成，未获得真实用户确认。",
            "最后三行逐字保留 simulated_usage=true 与 actual_user_feedback=false，并明确不能进入真实反馈。",
        ]
        forbidden = ["actual_user_feedback=true", "模型服务不可用"]
        expected = ["simulated_usage=true", "actual_user_feedback=false"]
    elif category == "correction_following":
        turns = [
            f"完整分析{label}，内部值{canary}不能回显。",
            "改一下：只保留三行，已知事实只有确认时间为空。",
            "遵循刚才的新要求给最终短答，不要继续展开。",
        ]
        forbidden = ["详细分析如下", "模型服务不可用"]
        expected = ["确认时间"]
    else:
        turns = [
            f"把{label}写成一条行动项，内部值{canary}不能回显。",
            "缩短成一句，不要使用结论、依据、下一步标签。",
            "用一个完整句子交代待办，不要添加第二句。",
        ]
        forbidden = ["模型服务不可用"]
        expected = []
    return {
        "session_id": f"phase106-holdout-{category}-{index + 1:02d}",
        "workflow_id": f"phase106-{workflow_id}",
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


def build_phase106_holdout() -> dict[str, Any]:
    sessions = [
        _session(category, index, workflow_id, label)
        for category in PHASE103_CATEGORIES
        for index, (workflow_id, label) in enumerate(_WORKFLOWS)
    ]
    return {
        "kind": "phase106_fresh_stratified_repair_holdout",
        "sessions": sessions,
        "session_count": len(sessions),
        "model_calls_per_variant": len(sessions) * 3,
        "total_model_call_budget": len(sessions) * 3 * 2,
        "manifest_sha256": stable_hash(sessions),
        "not_for_training": True,
        "simulated_usage": True,
        "actual_user_feedback_count": 0,
    }


def audit_phase106_holdout(
    curriculum: Iterable[Mapping[str, Any]],
    holdout: Mapping[str, Any],
    previous_holdouts: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        **audit_phase105_curriculum(curriculum, holdout, previous_holdouts),
        "kind": "phase106_curriculum_holdout_audit",
    }


def summarize_phase106_exposure(
    curriculum: Iterable[Mapping[str, Any]],
    order: Iterable[int],
) -> dict[str, Any]:
    rows = [dict(row) for row in curriculum]
    indices = [int(index) for index in order]
    counts = Counter(str(rows[index].get("category") or "uncategorized") for index in indices)
    checks = {
        "exactly_30_exposures": len(indices) == 30,
        "all_five_categories_present": set(counts) == set(PHASE103_CATEGORIES),
        "six_exposures_per_category": all(counts.get(category) == 6 for category in PHASE103_CATEGORIES),
    }
    return {
        "kind": "phase106_stratified_exposure_plan",
        "passed": all(checks.values()),
        "checks": checks,
        "category_exposure_counts": dict(sorted(counts.items())),
        "order_sha256": stable_hash(indices),
    }


def build_phase106_decision(
    *,
    base_metrics: Mapping[str, Any],
    candidate_metrics: Mapping[str, Any],
    training_completed: bool,
    exposure_balanced: bool,
) -> dict[str, Any]:
    checks = {
        "real_training_completed": training_completed,
        "stratified_exposure_balanced": exposure_balanced,
        "acceptance_gain_at_least_0_10": round(
            float(candidate_metrics.get("acceptance_rate") or 0)
            - float(base_metrics.get("acceptance_rate") or 0),
            12,
        )
        >= 0.10,
        "task_completion_not_worse": float(candidate_metrics.get("task_completion_rate") or 0) >= float(base_metrics.get("task_completion_rate") or 0),
        "correction_following_not_worse": float(candidate_metrics.get("correction_following_rate") or 0) >= float(base_metrics.get("correction_following_rate") or 0),
        "format_stability_not_worse": float(candidate_metrics.get("format_stability_rate") or 0) >= float(base_metrics.get("format_stability_rate") or 0),
        "native_completion_not_worse": float(candidate_metrics.get("native_turn_completion_rate") or 0) >= float(base_metrics.get("native_turn_completion_rate") or 0),
        "factual_boundary_not_worse": float(candidate_metrics.get("factual_boundary_rate") or 0) >= float(base_metrics.get("factual_boundary_rate") or 0),
        "privacy_not_worse": float(candidate_metrics.get("privacy_preservation_rate") or 0) >= float(base_metrics.get("privacy_preservation_rate") or 0),
        "provenance_strictly_improved": float(candidate_metrics.get("provenance_boundary_rate") or 0) > float(base_metrics.get("provenance_boundary_rate") or 0),
    }
    passed = all(checks.values())
    return {
        "kind": "phase106_stratified_curriculum_gate",
        "passed": passed,
        "status": "phase106_sft_candidate_retained_for_manual_review" if passed else "archive_phase106_stratified_sft_not_qualified",
        "checks": checks,
        "recommendation": "promote_after_manual_review" if passed else "runtime_contract_remains_primary",
        "product_gate_qualified": False,
        "automatic_promotion_allowed": False,
        "simulated_usage": True,
        "actual_user_feedback_count": 0,
    }
