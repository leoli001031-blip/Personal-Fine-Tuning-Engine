from __future__ import annotations

from pfe_core.phase40_user_acceptance_simulation import (
    PHASE40_MIN_REVIEWED_PREFERENCES,
    PHASE40_MODEL_VARIANTS,
    build_phase40_blind_eval_pairs,
    build_phase40_manual_review_items,
    build_phase40_phase39_recap,
    build_phase40_scenario_bank,
    build_phase40_transcripts,
)
from pfe_core.phase41_simulated_review_preferences import (
    PHASE41_EVIDENCE_TYPE,
    build_phase41_candidate_manifest,
    build_phase41_comparison_summary,
    build_phase41_review_decision_audit,
    build_phase41_review_summary,
    build_phase41_simulated_review_decisions,
    phase41_final_decision,
    validate_phase41_boundaries,
)


def _phase41_review_bundle(count: int = 100, review_count: int = 24) -> tuple[list[dict], list[dict], dict]:
    recap = build_phase40_phase39_recap(
        {
            "evidence_type": "simulated_lab_evidence",
            "actual_product_benefit_claim_allowed": False,
            "final_recommendation": "collect_manual_review",
        }
    )
    scenarios = build_phase40_scenario_bank(count=count, phase39_recap=recap)
    transcripts_by_variant = {
        variant: build_phase40_transcripts(scenarios=scenarios, model_variant=variant)
        for variant in PHASE40_MODEL_VARIANTS
    }
    blind_pairs, blind_key = build_phase40_blind_eval_pairs(
        scenarios=scenarios,
        transcripts_by_variant=transcripts_by_variant,
    )
    review_items = build_phase40_manual_review_items(blind_pairs=blind_pairs, sample_count=review_count)
    review_decisions = build_phase41_simulated_review_decisions(
        review_items=review_items,
        review_count=review_count,
    )
    return review_items, review_decisions, blind_key


def test_phase41_simulated_reviewer_generates_blinded_preference_decisions() -> None:
    review_items, review_decisions, _blind_key = _phase41_review_bundle()

    assert len(review_decisions) == 24
    assert sum(1 for item in review_decisions if item["decision"] in {"prefer_a", "prefer_b"}) >= PHASE40_MIN_REVIEWED_PREFERENCES
    assert all(item["feedback_source"] == "simulated_usage" for item in review_decisions)
    assert all(item["simulated_user_review"] is True for item in review_decisions)
    assert all(item["confirmed_actual_user_feedback"] is False for item in review_decisions)
    assert all(item["actual_product_benefit_claim_allowed"] is False for item in review_decisions)
    public_text = str(review_decisions)
    assert "chosen_model_variant_for_audit_only" not in public_text
    assert "adapter_runtime_contract" not in public_text
    assert '"adapter"' not in public_text
    assert '"base"' not in public_text
    assert review_items[0]["review_payload"]["kind"] == "phase40_blind_eval_pair"


def test_phase41_duplicate_candidate_is_blocked_and_stays_simulated_only() -> None:
    review_items, review_decisions, blind_key = _phase41_review_bundle()
    review_summary = build_phase41_review_summary(
        review_items=review_items,
        review_decisions=review_decisions,
    )
    candidate = build_phase41_candidate_manifest(
        review_items=review_items,
        review_summary=review_summary,
    )
    audit = build_phase41_review_decision_audit(
        review_decisions=review_decisions,
        blind_variant_key=blind_key,
    )

    assert review_summary["manual_reviewed_preference_count"] >= PHASE40_MIN_REVIEWED_PREFERENCES
    assert candidate["training_candidate_status"] == "blocked"
    assert candidate["blocked_reason"] == "candidate_quality_gate_failed"
    assert candidate["evaluated_preference_pair_count"] >= PHASE40_MIN_REVIEWED_PREFERENCES
    assert candidate["candidate_quality"]["passed"] is False
    assert candidate["preference_source"] == "simulated_user_review_preference"
    assert candidate["actual_user_feedback_count"] == 0
    assert candidate["actual_product_benefit_claim_allowed"] is False
    assert candidate["auto_training_allowed"] is False
    assert audit["reviewer_input_was_blinded"] is True
    assert audit["audit_uses_hidden_key_after_decisions"] is True
    assert audit["chosen_model_counts"]


def test_phase41_boundary_blocks_actual_feedback_or_product_claim() -> None:
    review_items, review_decisions, _blind_key = _phase41_review_bundle()
    review_decisions[0]["confirmed_actual_user_feedback"] = True
    review_decisions[1]["actual_product_benefit_claim_allowed"] = True
    review_summary = build_phase41_review_summary(
        review_items=review_items,
        review_decisions=review_decisions,
    )
    candidate = build_phase41_candidate_manifest(
        review_items=review_items,
        review_summary=review_summary,
    )
    boundary = validate_phase41_boundaries(
        review_items=review_items,
        review_decisions=review_decisions,
        review_summary=review_summary,
        candidate_manifest=candidate,
    )

    assert boundary["passed"] is False
    reasons = {item["reason"] for item in boundary["problems"]}
    assert "actual_feedback_mislabel" in reasons
    assert "actual_product_claim_allowed" in reasons


def test_phase41_final_decision_requires_diverse_preferences_before_training() -> None:
    review_items, review_decisions, blind_key = _phase41_review_bundle()
    review_summary = build_phase41_review_summary(
        review_items=review_items,
        review_decisions=review_decisions,
    )
    candidate = build_phase41_candidate_manifest(
        review_items=review_items,
        review_summary=review_summary,
    )
    audit = build_phase41_review_decision_audit(
        review_decisions=review_decisions,
        blind_variant_key=blind_key,
    )
    boundary = validate_phase41_boundaries(
        review_items=review_items,
        review_decisions=review_decisions,
        review_summary=review_summary,
        candidate_manifest=candidate,
    )
    decision = phase41_final_decision(
        phase40_summary={"final_recommendation": "collect_manual_review"},
        review_summary=review_summary,
        candidate_manifest=candidate,
        boundary_check=boundary,
        decision_audit=audit,
    )
    summary = build_phase41_comparison_summary(
        review_summary=review_summary,
        candidate_manifest=candidate,
        boundary_check=boundary,
        final_decision=decision,
    )

    assert boundary["passed"] is True
    assert decision["recommendation"] == "regenerate_diverse_simulated_preferences"
    assert decision["evidence_type"] == PHASE41_EVIDENCE_TYPE
    assert decision["manual_training_probe_allowed"] is False
    assert decision["actual_product_benefit_claim_allowed"] is False
    assert decision["auto_training_allowed"] is False
    assert decision["auto_promotion_allowed"] is False
    assert summary["final_recommendation"] == decision["recommendation"]


def test_phase41_stays_blocked_below_review_threshold() -> None:
    review_items, review_decisions, blind_key = _phase41_review_bundle(review_count=24)
    review_decisions = review_decisions[: PHASE40_MIN_REVIEWED_PREFERENCES - 1]
    review_summary = build_phase41_review_summary(
        review_items=review_items,
        review_decisions=review_decisions,
    )
    candidate = build_phase41_candidate_manifest(
        review_items=review_items,
        review_summary=review_summary,
    )
    audit = build_phase41_review_decision_audit(
        review_decisions=review_decisions,
        blind_variant_key=blind_key,
    )
    boundary = validate_phase41_boundaries(
        review_items=review_items,
        review_decisions=review_decisions,
        review_summary=review_summary,
        candidate_manifest=candidate,
    )
    decision = phase41_final_decision(
        phase40_summary={"final_recommendation": "collect_manual_review"},
        review_summary=review_summary,
        candidate_manifest=candidate,
        boundary_check=boundary,
        decision_audit=audit,
    )

    assert candidate["training_candidate_status"] == "blocked"
    assert candidate["blocked_reason"] == "insufficient_manual_reviewed_preferences"
    assert decision["recommendation"] == "collect_more_simulated_manual_reviews"
    assert decision["manual_training_probe_allowed"] is False
