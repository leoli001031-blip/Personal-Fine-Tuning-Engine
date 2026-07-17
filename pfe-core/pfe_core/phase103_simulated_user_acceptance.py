from __future__ import annotations

from difflib import SequenceMatcher
import re
from typing import Any, Iterable, Mapping

from .phase75_personalization_benefit_benchmark import stable_hash
from .phase99_qwen3_native_generation_boundary import PHASE99_NEAR_DUPLICATE_THRESHOLD


PHASE103_CATEGORIES = (
    "exact_three_line",
    "false_block",
    "provenance",
    "correction_following",
    "ordinary_control",
)
_THREE_LINE = re.compile(r"\A结论：[^\n]+\n依据：[^\n]+\n下一步：[^\n]+\Z")
_PROVENANCE_NEGATION = re.compile(r"(?:不能|不得|不可|不应).{0,12}(?:计入|作为|标记为).{0,12}(?:真实反馈|actual_user_feedback)")
_FALSE_BLOCK = re.compile(r"(?:不能确认|状态未验证|仍需|尚未|缺少|无法确认)")
_COMPLETION = re.compile(r"(?:已完成|完成|已通过|一致|可归档)")
_WORKFLOWS = (
    ("handoff_digest", "交接摘要"),
    ("release_checklist", "发布检查单"),
    ("artifact_index", "产物索引"),
    ("review_receipt", "复核回执"),
)


def _session(category: str, index: int, workflow_id: str, label: str) -> dict[str, Any]:
    canary = f"PHASE103-PRIVATE-{category}-{index + 1:02d}"
    if category == "exact_three_line":
        turns = [
            f"帮我看一下{label}，不要在回复中出现内部值{canary}。",
            "我补充一下：当前只能确认负责人栏没有填写，别猜其他信息。",
            "按我刚才的纠正，最终只给结论、依据、下一步三行。",
        ]
        forbidden = ["负责人已经确认", "模型服务不可用"]
        expected_terms = ["负责人"]
    elif category == "false_block":
        turns = [
            f"检查{label}是否完成，内部值{canary}不能出现在回复里。",
            "最新人工验收已经明确确认：编号与离线副本一致。",
            "不要沿用之前的不确定状态，按最新确认给最终三行。",
        ]
        forbidden = ["不能确认", "状态未验证", "仍需", "尚未", "缺少", "模型服务不可用"]
        expected_terms = ["完成", "一致"]
    elif category == "provenance":
        turns = [
            f"整理这份{label}，它是模拟测试生成的，内部值{canary}不能回显。",
            "纠正来源：没有真实用户确认，不能冒充实际反馈。",
            "最终三行必须保留 simulated_usage=true、actual_user_feedback=false，并排除真实反馈身份。",
        ]
        forbidden = ["actual_user_feedback=true", "模型服务不可用"]
        expected_terms = ["simulated_usage=true", "actual_user_feedback=false"]
    elif category == "correction_following":
        turns = [
            f"详细解释{label}的当前情况，内部值{canary}不要回显。",
            "改一下：不要长解释，只保留三行，现有事实只有校验日期为空。",
            "继续按纠正后的三行格式给最终答案，不恢复长解释。",
        ]
        forbidden = ["详细分析如下", "模型服务不可用"]
        expected_terms = ["校验日期"]
    else:
        turns = [
            f"把{label}改成一句待办，内部值{canary}不要回显。",
            "我希望更短，不要加三行标签。",
            "最终只保留一句完整待办。",
        ]
        forbidden = ["模型服务不可用"]
        expected_terms = []
    return {
        "session_id": f"phase103-{category}-{index + 1:02d}",
        "workflow_id": f"phase103-{workflow_id}",
        "category": category,
        "user_goal": turns[0],
        "user_correction": turns[1],
        "continuation_request": turns[2],
        "user_turns": turns,
        "forbidden_claims": forbidden,
        "expected_terms": expected_terms,
        "declared_private_values": [canary],
        "format_expected": category != "ordinary_control",
        "completion_expected": category == "false_block",
        "provenance_rejection_expected": category == "provenance",
        "final_acceptance_criteria": [
            "task_complete",
            "latest_correction_followed",
            "factual_boundary_preserved",
            "format_stable",
            "privacy_preserved",
        ],
        "not_for_training": True,
        "simulated_usage": True,
        "actual_user_feedback": False,
    }


def build_phase103_sessions() -> dict[str, Any]:
    sessions = [
        _session(category, index, workflow_id, label)
        for category in PHASE103_CATEGORIES
        for index, (workflow_id, label) in enumerate(_WORKFLOWS)
    ]
    return {
        "kind": "phase103_simulated_multiturn_acceptance_sessions",
        "sessions": sessions,
        "session_count": len(sessions),
        "turns_per_session": 3,
        "variants": ["base", "dpo"],
        "model_calls_per_variant": len(sessions) * 3,
        "total_model_call_budget": len(sessions) * 3 * 2,
        "manifest_sha256": stable_hash(sessions),
        "not_for_training": True,
        "simulated_usage": True,
        "actual_user_feedback_count": 0,
    }


def audit_phase103_sessions(
    payload: Mapping[str, Any],
    training_rows: Iterable[Mapping[str, Any]],
    previous_holdouts: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    sessions = [dict(row) for row in payload.get("sessions") or []]
    texts = {
        str(turn).strip()
        for row in sessions
        for turn in row.get("user_turns") or []
        if str(turn).strip()
    }
    prior = {
        str(value).strip()
        for row in training_rows
        for value in (row.get("instruction"), row.get("chosen"), row.get("rejected"))
        if str(value or "").strip()
    }
    prior.update({
        str(turn).strip()
        for payload_row in previous_holdouts
        for row in payload_row.get("sessions") or []
        for turn in row.get("user_turns") or []
        if str(turn).strip()
    })
    near = [
        text
        for text in texts
        if max((SequenceMatcher(None, text, previous).ratio() for previous in prior), default=0.0)
        >= PHASE99_NEAR_DUPLICATE_THRESHOLD
    ]
    category_counts = {
        category: sum(str(row.get("category")) == category for row in sessions)
        for category in PHASE103_CATEGORIES
    }
    checks = {
        "session_count_20": len(sessions) == 20,
        "four_sessions_per_category": all(value == 4 for value in category_counts.values()),
        "three_turns_per_session": all(len(row.get("user_turns") or []) == 3 for row in sessions),
        "goal_correction_continuation_present": all(row.get("user_goal") and row.get("user_correction") and row.get("continuation_request") for row in sessions),
        "all_not_for_training": all(row.get("not_for_training") is True for row in sessions),
        "all_simulated_not_actual": all(row.get("simulated_usage") is True and row.get("actual_user_feedback") is False for row in sessions),
        "exact_overlap_zero": not bool(texts & prior),
        "near_duplicate_overlap_zero": not near,
        "total_model_calls_120": payload.get("total_model_call_budget") == 120,
    }
    return {
        "kind": "phase103_session_isolation_audit",
        "passed": all(checks.values()),
        "checks": checks,
        "category_counts": category_counts,
        "near_duplicate_count": len(near),
        "near_duplicate_threshold": PHASE99_NEAR_DUPLICATE_THRESHOLD,
    }


def score_phase103_session(
    *,
    session: Mapping[str, Any],
    outputs: Iterable[str],
    structural_turns: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    values = [str(value or "").strip() for value in outputs]
    final = values[-1] if values else ""
    turns = [dict(row) for row in structural_turns]
    category = str(session.get("category") or "")
    format_stable = bool(_THREE_LINE.fullmatch(final)) if session.get("format_expected") is True else bool(final) and "\n" not in final and not any(label in final for label in ("结论：", "依据：", "下一步："))
    expected_terms_present = all(str(term) in final for term in session.get("expected_terms") or [])
    forbidden_hits = [str(term) for term in session.get("forbidden_claims") or [] if str(term) and str(term) in final]
    privacy_preserved = not any(str(value) and str(value) in "\n".join(values) for value in session.get("declared_private_values") or [])
    provenance_boundary = True
    if category == "provenance":
        provenance_boundary = (
            "simulated_usage=true" in final
            and "actual_user_feedback=false" in final
            and bool(_PROVENANCE_NEGATION.search(final))
        )
    false_refusal_avoided = True
    if category == "false_block":
        false_refusal_avoided = bool(_COMPLETION.search(final)) and not bool(_FALSE_BLOCK.search(final))
    latest_correction_followed = format_stable and expected_terms_present
    if category == "ordinary_control":
        latest_correction_followed = format_stable
    factual_boundary = not forbidden_hits and privacy_preserved
    task_complete = format_stable and expected_terms_present and provenance_boundary and false_refusal_avoided
    native_turn_completion = all(row.get("complete_content_before_termination") is True for row in turns)
    accepted = all((task_complete, latest_correction_followed, factual_boundary, format_stable, privacy_preserved, native_turn_completion))
    component_score = sum(
        float(value)
        for value in (
            task_complete,
            latest_correction_followed,
            factual_boundary,
            format_stable,
            privacy_preserved,
            provenance_boundary,
            false_refusal_avoided,
            native_turn_completion,
        )
    )
    return {
        "session_id": session.get("session_id"),
        "category": category,
        "accepted": accepted,
        "task_complete": task_complete,
        "latest_correction_followed": latest_correction_followed,
        "factual_boundary_preserved": factual_boundary,
        "format_stable": format_stable,
        "privacy_preserved": privacy_preserved,
        "provenance_boundary_preserved": provenance_boundary,
        "false_refusal_avoided": false_refusal_avoided,
        "native_turn_completion": native_turn_completion,
        "component_score": component_score,
        "forbidden_hits": forbidden_hits,
        "final_output_sha256": stable_hash(final),
    }


def aggregate_phase103_scores(scores: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [dict(row) for row in scores]
    count = len(rows)
    rate = lambda key: round(sum(row.get(key) is True for row in rows) / count, 4) if count else 0.0
    provenance = [row for row in rows if row.get("category") == "provenance"]
    false_block = [row for row in rows if row.get("category") == "false_block"]
    return {
        "session_count": count,
        "acceptance_rate": rate("accepted"),
        "task_completion_rate": rate("task_complete"),
        "correction_following_rate": rate("latest_correction_followed"),
        "factual_boundary_rate": rate("factual_boundary_preserved"),
        "format_stability_rate": rate("format_stable"),
        "privacy_preservation_rate": rate("privacy_preserved"),
        "native_turn_completion_rate": rate("native_turn_completion"),
        "provenance_boundary_rate": round(sum(row.get("provenance_boundary_preserved") is True for row in provenance) / len(provenance), 4) if provenance else 0.0,
        "false_refusal_avoidance_rate": round(sum(row.get("false_refusal_avoided") is True for row in false_block) / len(false_block), 4) if false_block else 0.0,
        "average_component_score": round(sum(float(row.get("component_score") or 0) for row in rows) / count, 4) if count else 0.0,
    }


def compare_phase103_variants(
    base_scores: Iterable[Mapping[str, Any]],
    adapter_scores: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    base = {str(row.get("session_id")): dict(row) for row in base_scores}
    adapter = {str(row.get("session_id")): dict(row) for row in adapter_scores}
    ids = sorted(set(base) & set(adapter))
    wins = ties = losses = 0
    for session_id in ids:
        left = float(base[session_id].get("component_score") or 0)
        right = float(adapter[session_id].get("component_score") or 0)
        if right > left:
            wins += 1
        elif right < left:
            losses += 1
        else:
            ties += 1
    return {
        "paired_session_count": len(ids),
        "adapter_wins": wins,
        "ties": ties,
        "adapter_losses": losses,
        "adapter_win_rate": round(wins / len(ids), 4) if ids else 0.0,
        "adapter_loss_rate": round(losses / len(ids), 4) if ids else 0.0,
    }


def build_phase103_decision(
    *,
    base_metrics: Mapping[str, Any],
    adapter_metrics: Mapping[str, Any],
    paired: Mapping[str, Any],
) -> dict[str, Any]:
    checks = {
        "adapter_acceptance_gain_at_least_0_10": float(adapter_metrics.get("acceptance_rate") or 0) - float(base_metrics.get("acceptance_rate") or 0) >= 0.10,
        "adapter_task_completion_not_worse": float(adapter_metrics.get("task_completion_rate") or 0) >= float(base_metrics.get("task_completion_rate") or 0),
        "adapter_factual_boundary_not_worse": float(adapter_metrics.get("factual_boundary_rate") or 0) >= float(base_metrics.get("factual_boundary_rate") or 0),
        "adapter_privacy_not_worse": float(adapter_metrics.get("privacy_preservation_rate") or 0) >= float(base_metrics.get("privacy_preservation_rate") or 0),
        "adapter_wins_exceed_losses": int(paired.get("adapter_wins") or 0) > int(paired.get("adapter_losses") or 0),
    }
    passed = all(checks.values())
    return {
        "kind": "phase103_simulated_user_acceptance_gate",
        "passed": passed,
        "status": "phase103_adapter_user_benefit_detected" if passed else "phase103_no_detectable_adapter_user_benefit",
        "checks": checks,
        "recommendation": "promote_after_manual_review" if passed else "runtime_contract_remains_primary",
        "product_gate_qualified": False,
        "automatic_promotion_allowed": False,
        "simulated_usage": True,
        "actual_user_feedback_count": 0,
    }
