"""Phase41 simulated user review and preference-candidate primitives."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import re
from typing import Any, Iterable, Mapping

from pfe_core.phase32_personal_agent_preference import contains_raw_private_text
from pfe_core.phase40_user_acceptance_simulation import (
    PHASE40_FEEDBACK_SOURCE,
    PHASE40_MIN_REVIEWED_PREFERENCES,
    build_phase40_manual_review_summary,
    build_phase40_preference_candidate_manifest,
    score_phase40_candidate,
)


PHASE41_KIND = "phase41_simulated_review_preference_candidates"
PHASE41_FEEDBACK_SOURCE = PHASE40_FEEDBACK_SOURCE
PHASE41_DEFAULT_REVIEW_COUNT = 24
PHASE41_MIN_REVIEWED_PREFERENCES = PHASE40_MIN_REVIEWED_PREFERENCES
PHASE41_REVIEWER_ID = "phase41-simulated-user-reviewer"
PHASE41_EVIDENCE_TYPE = "simulated_manual_review_preference_evidence"


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _candidate_text(candidate: Mapping[str, Any]) -> str:
    return str(candidate.get("agent_response") or "")


def _contains_all_sections(text: str) -> bool:
    return all(section in text for section in ("摘要", "风险提示", "引用依据", "人工确认"))


def _review_candidate_score(candidate: Mapping[str, Any]) -> dict[str, Any]:
    metric_scores = score_phase40_candidate(candidate)
    text = _candidate_text(candidate)
    weighted_score = (
        2.0 * metric_scores["would_user_keep_using"]
        + 1.5 * metric_scores["follows_latest_user_intent"]
        + 1.5 * metric_scores["checks_real_state_before_claim"]
        + 1.2 * metric_scores["correction_responsiveness"]
        + 1.2 * metric_scores["separates_actual_vs_simulated_evidence"]
        + 1.0 * metric_scores["preserves_privacy_boundary"]
        + 1.0 * metric_scores["useful_next_step"]
        + 0.8 * metric_scores["concise_and_actionable"]
        + 0.8 * metric_scores["no_false_completion"]
    )
    if _contains_all_sections(text):
        weighted_score += 0.4
    if re.search(r"整体分析|综合判断|继续优化即可|后续继续推进即可", text):
        weighted_score -= 0.8
    if re.search(r"先核对|先查|git|PR|gate|evidence|真实 PID|工作区|命令输出", text):
        weighted_score += 0.3
    if re.search(r"不提交私密|不把 simulated|不能 claim|不能宣称|actual product benefit", text):
        weighted_score += 0.3
    return {
        "score": round(weighted_score, 3),
        "metrics": metric_scores,
        "has_four_section_boundary": _contains_all_sections(text),
        "generic_answer_penalty": bool(re.search(r"整体分析|综合判断|继续优化即可|后续继续推进即可", text)),
    }


def _variant_scores(review_payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    scores: dict[str, dict[str, Any]] = {}
    for label in ("variant_a", "variant_b", "variant_c", "variant_d"):
        candidate = _dict(review_payload.get(label))
        if candidate:
            scores[label] = _review_candidate_score(candidate)
    return scores


def build_phase41_simulated_review_decisions(
    *,
    review_items: Iterable[Mapping[str, Any]],
    reviewer_id: str = PHASE41_REVIEWER_ID,
    review_count: int = PHASE41_DEFAULT_REVIEW_COUNT,
) -> list[dict[str, Any]]:
    """Choose preference pairs from public blind review payloads.

    The simulated reviewer only sees anonymous variant labels and response text.
    Model identity can be audited later with the hidden key, but it is not used
    to choose the preferred response.
    """

    decisions: list[dict[str, Any]] = []
    for item in list(review_items)[: max(0, int(review_count))]:
        payload = _dict(item.get("review_payload"))
        scores = _variant_scores(payload)
        ranked = sorted(scores.items(), key=lambda entry: (entry[1]["score"], entry[0]), reverse=True)
        if len(ranked) < 2:
            state = "both_bad"
            chosen_variant = None
            rejected_variant = None
            reason = "Not enough anonymous variants were available for a useful preference decision."
        else:
            chosen_variant, chosen_score = ranked[0]
            rejected_variant, rejected_score = ranked[-1]
            score_gap = float(chosen_score["score"]) - float(rejected_score["score"])
            if float(chosen_score["score"]) < 4.5:
                state = "both_bad"
                reason = "No response met the minimum evidence-first user acceptance bar."
            elif score_gap < 1.0:
                state = "tie"
                reason = "The best and weakest anonymous responses were too close for a stable preference."
            else:
                state = "prefer_a" if chosen_variant == "variant_a" else "prefer_b"
                reason = (
                    "Chosen response better follows the latest user intent, checks evidence before claims, "
                    "preserves privacy boundaries, and gives a concrete next step."
                )
        decision: dict[str, Any] = {
            "kind": "phase41_simulated_manual_review_decision",
            "review_item_id": item.get("review_item_id"),
            "pair_id": item.get("pair_id"),
            "scenario_id": item.get("scenario_id"),
            "category": item.get("category"),
            "decision": state,
            "decision_semantics": "anonymous_preference_pair",
            "chosen_variant": chosen_variant,
            "rejected_variant": rejected_variant,
            "reviewer_id": reviewer_id,
            "timestamp": _utcnow_iso(),
            "reason": reason,
            "review_rubric_scores": scores,
            "consent_for_training_candidate_review": state in {"prefer_a", "prefer_b"},
            "source": "simulated_user_review",
            "feedback_source": PHASE41_FEEDBACK_SOURCE,
            "simulated_usage": True,
            "simulated_user_review": True,
            "not_actual_user_feedback": True,
            "confirmed_actual_user_feedback": False,
            "actual_product_benefit_claim_allowed": False,
            "auto_training_allowed": False,
            "auto_promotion_allowed": False,
        }
        if state not in {"prefer_a", "prefer_b"}:
            decision.pop("chosen_variant", None)
            decision.pop("rejected_variant", None)
        decisions.append(decision)
    return decisions


def build_phase41_review_summary(
    *,
    review_items: list[Mapping[str, Any]],
    review_decisions: list[Mapping[str, Any]],
) -> dict[str, Any]:
    summary = build_phase40_manual_review_summary(
        review_items=review_items,
        review_decisions=review_decisions,
    )
    decision_counts = Counter(str(item.get("decision") or "unknown") for item in review_decisions)
    summary.update(
        {
            "kind": "phase41_simulated_manual_review_summary",
            "phase40_compatible": True,
            "decision_counts": dict(sorted(decision_counts.items())),
            "reviewer_mode": "simulated_user_perspective",
            "reviewer_blinded_to_model_identity": True,
            "source": "simulated_user_review",
            "feedback_source": PHASE41_FEEDBACK_SOURCE,
            "simulated_usage": True,
            "not_actual_user_feedback": True,
            "confirmed_actual_user_feedback": False,
            "actual_product_benefit_claim_allowed": False,
            "auto_training_allowed": False,
            "auto_promotion_allowed": False,
        }
    )
    return summary


def build_phase41_candidate_manifest(
    *,
    review_items: list[Mapping[str, Any]],
    review_summary: Mapping[str, Any],
) -> dict[str, Any]:
    manifest = build_phase40_preference_candidate_manifest(
        review_items=review_items,
        manual_review_summary=review_summary,
    )
    manifest.update(
        {
            "kind": "phase41_preference_candidate_manifest",
            "preference_source": "simulated_user_review_preference",
            "feedback_source": PHASE41_FEEDBACK_SOURCE,
            "simulated_usage": True,
            "not_actual_user_feedback": True,
            "confirmed_actual_user_feedback": False,
            "actual_product_benefit_claim_allowed": False,
            "actual_user_feedback_count": 0,
            "auto_training_allowed": False,
            "auto_promotion_allowed": False,
            "training_launch_policy": "manual_only_after_review",
        }
    )
    return manifest


def build_phase41_review_decision_audit(
    *,
    review_decisions: Iterable[Mapping[str, Any]],
    blind_variant_key: Mapping[str, Any],
) -> dict[str, Any]:
    key_by_pair = {
        str(item.get("pair_id")): _dict(item.get("blind_variant_map"))
        for item in blind_variant_key.get("items") or []
        if isinstance(item, Mapping)
    }
    chosen_counts: Counter[str] = Counter()
    rejected_counts: Counter[str] = Counter()
    audited_preferences: list[dict[str, Any]] = []
    for decision in review_decisions:
        if decision.get("decision") not in {"prefer_a", "prefer_b"}:
            continue
        pair_id = str(decision.get("pair_id") or "")
        label_map = key_by_pair.get(pair_id, {})
        chosen_variant = str(decision.get("chosen_variant") or "")
        rejected_variant = str(decision.get("rejected_variant") or "")
        chosen_model = str(label_map.get(chosen_variant) or "unknown")
        rejected_model = str(label_map.get(rejected_variant) or "unknown")
        chosen_counts[chosen_model] += 1
        rejected_counts[rejected_model] += 1
        audited_preferences.append(
            {
                "review_item_id": decision.get("review_item_id"),
                "pair_id": pair_id,
                "chosen_anonymous_variant": chosen_variant,
                "rejected_anonymous_variant": rejected_variant,
                "chosen_model_variant_for_audit_only": chosen_model,
                "rejected_model_variant_for_audit_only": rejected_model,
            }
        )
    return {
        "kind": "phase41_simulated_review_decision_audit",
        "reviewer_input_was_blinded": True,
        "audit_uses_hidden_key_after_decisions": True,
        "preference_count": len(audited_preferences),
        "chosen_model_counts": dict(sorted(chosen_counts.items())),
        "rejected_model_counts": dict(sorted(rejected_counts.items())),
        "audited_preferences": audited_preferences,
        "actual_product_benefit_claim_allowed": False,
        "actual_user_feedback_count": 0,
        "created_at": _utcnow_iso(),
    }


def validate_phase41_boundaries(
    *,
    review_items: Iterable[Mapping[str, Any]],
    review_decisions: Iterable[Mapping[str, Any]],
    review_summary: Mapping[str, Any],
    candidate_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    problems: list[dict[str, str]] = []
    for collection_name, rows in (
        ("review_item", list(review_items)),
        ("review_decision", list(review_decisions)),
    ):
        for item in rows:
            item_id = str(item.get("review_item_id") or item.get("pair_id") or "unknown")
            if item.get("feedback_source") != PHASE41_FEEDBACK_SOURCE:
                problems.append({"item_id": item_id, "collection": collection_name, "reason": "feedback_source_not_simulated_usage"})
            if item.get("confirmed_actual_user_feedback") is True:
                problems.append({"item_id": item_id, "collection": collection_name, "reason": "actual_feedback_mislabel"})
            if item.get("actual_product_benefit_claim_allowed") is True:
                problems.append({"item_id": item_id, "collection": collection_name, "reason": "actual_product_claim_allowed"})
            if contains_raw_private_text(item):
                problems.append({"item_id": item_id, "collection": collection_name, "reason": "raw_private_text_detected"})
            public_text = json.dumps(item, ensure_ascii=False, sort_keys=True)
            if collection_name == "review_decision" and re.search(
                r"chosen_model_variant_for_audit_only|adapter_runtime_contract|runtime_contract|\"adapter\"|\"base\"",
                public_text,
            ):
                problems.append({"item_id": item_id, "collection": collection_name, "reason": "model_identity_leaked_to_review_decision"})
    for name, payload in (("review_summary", review_summary), ("candidate_manifest", candidate_manifest)):
        if payload.get("actual_product_benefit_claim_allowed") is True:
            problems.append({"item_id": name, "collection": name, "reason": "actual_product_claim_allowed"})
        if payload.get("actual_user_feedback_count"):
            problems.append({"item_id": name, "collection": name, "reason": "actual_user_feedback_count_nonzero"})
        if contains_raw_private_text(payload):
            problems.append({"item_id": name, "collection": name, "reason": "raw_private_text_detected"})
    return {
        "kind": "phase41_boundary_check",
        "passed": not problems,
        "problem_count": len(problems),
        "problems": problems[:100],
        "created_at": _utcnow_iso(),
    }


def phase41_final_decision(
    *,
    phase40_summary: Mapping[str, Any],
    review_summary: Mapping[str, Any],
    candidate_manifest: Mapping[str, Any],
    boundary_check: Mapping[str, Any],
    decision_audit: Mapping[str, Any],
) -> dict[str, Any]:
    reviewed_count = int(review_summary.get("manual_reviewed_preference_count") or 0)
    ready = (
        boundary_check.get("passed") is True
        and candidate_manifest.get("training_candidate_status") == "ready"
        and reviewed_count >= PHASE41_MIN_REVIEWED_PREFERENCES
    )
    if not boundary_check.get("passed"):
        recommendation = "fix_phase41_boundary_violations"
        status = "blocked"
    elif ready:
        recommendation = "ready_for_small_model_training_probe_from_simulated_preferences"
        status = "ready_for_manual_training_probe"
    else:
        recommendation = "collect_more_simulated_manual_reviews"
        status = "continue"
    return {
        "kind": "phase41_final_decision",
        "status": status,
        "recommendation": recommendation,
        "evidence_type": PHASE41_EVIDENCE_TYPE,
        "phase40_recommendation": phase40_summary.get("final_recommendation") or phase40_summary.get("recommendation"),
        "manual_reviewed_preference_count": reviewed_count,
        "required_manual_reviewed_preferences": PHASE41_MIN_REVIEWED_PREFERENCES,
        "training_candidate_status": candidate_manifest.get("training_candidate_status"),
        "selected_preference_pair_count": candidate_manifest.get("selected_preference_pair_count"),
        "chosen_model_counts_for_audit_only": decision_audit.get("chosen_model_counts"),
        "rejected_model_counts_for_audit_only": decision_audit.get("rejected_model_counts"),
        "actual_user_feedback_count": 0,
        "actual_product_benefit_claim_allowed": False,
        "cannot_claim_actual_product_benefit": True,
        "auto_training_allowed": False,
        "auto_promotion_allowed": False,
        "manual_training_probe_allowed": ready,
        "next_action": "run_small_model_training_probe_with_simulated_preference_candidates"
        if ready
        else "continue_review_sampling",
        "created_at": _utcnow_iso(),
    }


def build_phase41_comparison_summary(
    *,
    review_summary: Mapping[str, Any],
    candidate_manifest: Mapping[str, Any],
    boundary_check: Mapping[str, Any],
    final_decision: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "kind": "phase41_simulated_review_preference_summary",
        "status": "completed",
        "manual_reviewed_preference_count": review_summary.get("manual_reviewed_preference_count"),
        "required_manual_reviewed_preferences": PHASE41_MIN_REVIEWED_PREFERENCES,
        "training_candidate_status": candidate_manifest.get("training_candidate_status"),
        "selected_preference_pair_count": candidate_manifest.get("selected_preference_pair_count"),
        "boundary_passed": boundary_check.get("passed"),
        "boundary_problem_count": boundary_check.get("problem_count"),
        "evidence_type": final_decision.get("evidence_type"),
        "final_recommendation": final_decision.get("recommendation"),
        "actual_user_feedback_count": 0,
        "actual_product_benefit_claim_allowed": False,
        "auto_training_allowed": False,
        "auto_promotion_allowed": False,
        "created_at": _utcnow_iso(),
    }


__all__ = [
    "PHASE41_DEFAULT_REVIEW_COUNT",
    "PHASE41_EVIDENCE_TYPE",
    "PHASE41_FEEDBACK_SOURCE",
    "PHASE41_KIND",
    "PHASE41_MIN_REVIEWED_PREFERENCES",
    "build_phase41_candidate_manifest",
    "build_phase41_comparison_summary",
    "build_phase41_review_decision_audit",
    "build_phase41_review_summary",
    "build_phase41_simulated_review_decisions",
    "phase41_final_decision",
    "validate_phase41_boundaries",
]
