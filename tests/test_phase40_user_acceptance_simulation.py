from __future__ import annotations

from pfe_core.phase36_39_local_feedback_product_benefit import (
    build_phase36_39_comparison_summary,
    build_phase36_39_simulated_lab_records,
    build_phase36_review_summary,
    build_phase37_candidate_artifacts,
    build_phase37_holdout,
    build_phase38_training_attempt,
    build_phase38_training_manifest,
    build_phase39_blind_eval_pairs,
    build_phase39_eval_report,
    build_phase39_simulated_sessions,
    build_phase39_transcripts,
    phase39_final_decision,
    validate_phase39_boundaries,
)
from pfe_core.phase40_user_acceptance_simulation import (
    PHASE40_FEEDBACK_SOURCE,
    PHASE40_MIN_REVIEWED_PREFERENCES,
    PHASE40_MODEL_VARIANTS,
    build_phase40_blind_eval_pairs,
    build_phase40_comparison_summary,
    build_phase40_manual_review_items,
    build_phase40_manual_review_summary,
    build_phase40_phase39_recap,
    build_phase40_preference_candidate_manifest,
    build_phase40_scenario_bank,
    build_phase40_transcripts,
    build_phase40_user_acceptance_scores,
    phase40_final_decision,
    score_phase40_candidate,
    validate_phase40_boundaries,
    validate_phase40_manual_review_decision,
    validate_phase40_scenario_bank,
    validate_phase40_transcript_structure,
)


def _phase39_summary() -> dict:
    review_summary = build_phase36_review_summary(state={"interactions": []}, review_decisions=[])
    holdout = build_phase37_holdout(count=40)
    actual = build_phase37_candidate_artifacts(
        records=[],
        review_decisions=[],
        holdout=holdout,
        lane="actual_candidate_lane",
    )
    simulated = build_phase37_candidate_artifacts(
        records=build_phase36_39_simulated_lab_records(count=12),
        review_decisions=[],
        holdout=holdout,
        lane="simulated_lab_candidate_lane",
    )
    training_manifest = build_phase38_training_manifest(
        candidate_artifacts=simulated,
        model_selection={"status": "selected", "selected_model": "/models/Qwen2.5-0.5B-Instruct"},
    )
    training_attempt = build_phase38_training_attempt(
        training_manifest=training_manifest,
        execution_result={"status": "completed", "real_execution": {"success": True, "artifact_dir": "/tmp/adapter"}},
    )
    sessions = build_phase39_simulated_sessions(count=50)
    transcripts_by_variant = {
        variant: build_phase39_transcripts(sessions=sessions, model_variant=variant)
        for variant in PHASE40_MODEL_VARIANTS
    }
    blind_pairs = build_phase39_blind_eval_pairs(sessions=sessions, transcripts_by_variant=transcripts_by_variant)
    eval_report = build_phase39_eval_report(transcripts_by_variant=transcripts_by_variant)
    boundary = validate_phase39_boundaries(
        sessions=sessions,
        transcripts=[item for rows in transcripts_by_variant.values() for item in rows],
        blind_pairs=blind_pairs,
    )
    decision = phase39_final_decision(
        review_summary=review_summary,
        actual_candidates=actual,
        simulated_candidates=simulated,
        training_attempt=training_attempt,
        eval_report=eval_report,
        boundary_check=boundary,
    )
    return build_phase36_39_comparison_summary(
        review_summary=review_summary,
        actual_candidates=actual,
        simulated_candidates=simulated,
        training_attempt=training_attempt,
        eval_report=eval_report,
        final_decision=decision,
    )


def _phase40_bundle(count: int = 100) -> tuple[list[dict], dict, dict, dict]:
    recap = build_phase40_phase39_recap(_phase39_summary())
    scenarios = build_phase40_scenario_bank(count=count, phase39_recap=recap)
    transcripts_by_variant = {
        variant: build_phase40_transcripts(scenarios=scenarios, model_variant=variant)
        for variant in PHASE40_MODEL_VARIANTS
    }
    blind_pairs, blind_key = build_phase40_blind_eval_pairs(
        scenarios=scenarios,
        transcripts_by_variant=transcripts_by_variant,
    )
    scores = build_phase40_user_acceptance_scores(blind_pairs=blind_pairs, blind_variant_key=blind_key)
    return scenarios, transcripts_by_variant, blind_pairs, {"key": blind_key, "scores": scores, "recap": recap}


def test_phase40_scenario_bank_schema_and_simulated_labeling() -> None:
    recap = build_phase40_phase39_recap(_phase39_summary())
    scenarios = build_phase40_scenario_bank(count=120, phase39_recap=recap)
    validation = validate_phase40_scenario_bank(scenarios)

    assert len(scenarios) == 120
    assert validation["passed"] is True
    assert all(scenario["feedback_source"] == PHASE40_FEEDBACK_SOURCE for scenario in scenarios)
    assert all(scenario["simulated_usage"] is True for scenario in scenarios)
    assert all(scenario["confirmed_actual_user_feedback"] is False for scenario in scenarios)
    assert all(scenario["actual_product_benefit_claim_allowed"] is False for scenario in scenarios)
    assert {scenario["category"] for scenario in scenarios} >= {
        "development_status",
        "execute_next",
        "course_correction",
        "submit_pr",
        "process_check",
        "next_goal",
        "no_showcase_assets",
        "training_effect",
        "evidence_boundary",
    }


def test_phase40_transcripts_have_required_multiround_structure() -> None:
    scenarios, transcripts_by_variant, _blind_pairs, _bundle = _phase40_bundle(count=100)
    assert len(scenarios) == 100

    for variant in PHASE40_MODEL_VARIANTS:
        rows = transcripts_by_variant[variant]
        assert len(rows) == 100
        assert all(validate_phase40_transcript_structure(row)["passed"] is True for row in rows)
        stages = {turn["stage"] for turn in rows[0]["turns"]}
        assert {"initial_answer", "response_to_correction", "next_action", "final_status_summary"} <= stages


def test_phase40_blind_eval_hides_model_variant_and_scores_acceptance() -> None:
    _scenarios, _transcripts, blind_pairs, bundle = _phase40_bundle(count=100)
    public_text = str(blind_pairs[0])

    assert "model_variant" not in public_text
    assert "adapter_runtime_contract" not in public_text
    assert "runtime_contract" not in public_text
    assert '"adapter"' not in public_text
    assert '"base"' not in public_text
    assert bundle["key"]["not_visible_to_scorer"] is True

    scores = bundle["scores"]["variants"]
    assert scores["adapter"]["scores"]["would_user_keep_using"] > scores["base"]["scores"]["would_user_keep_using"]
    assert scores["adapter_runtime_contract"]["scores"]["would_user_keep_using"] >= scores["runtime_contract"]["scores"]["would_user_keep_using"]
    assert score_phase40_candidate(blind_pairs[0]["variant_a"])["no_false_completion"] == 1.0


def test_phase40_manual_review_items_are_pending_and_not_actual_feedback() -> None:
    _scenarios, _transcripts, blind_pairs, _bundle = _phase40_bundle(count=100)
    review_items = build_phase40_manual_review_items(blind_pairs=blind_pairs, sample_count=24)
    summary = build_phase40_manual_review_summary(review_items=review_items, review_decisions=[])
    candidate = build_phase40_preference_candidate_manifest(
        review_items=review_items,
        manual_review_summary=summary,
    )

    assert len(review_items) == 24
    assert all(item["status"] == "pending_manual_review" for item in review_items)
    assert all(item["confirmed_actual_user_feedback"] is False for item in review_items)
    assert summary["pending_manual_review_count"] == 24
    assert summary["manual_reviewed_preference_count"] == 0
    assert candidate["training_candidate_status"] == "blocked"
    assert candidate["blocked_reason"] == "insufficient_manual_reviewed_preferences"
    assert candidate["actual_user_feedback_count"] == 0


def test_phase40_manual_review_decision_schema_and_candidate_threshold() -> None:
    _scenarios, _transcripts, blind_pairs, _bundle = _phase40_bundle(count=100)
    review_items = build_phase40_manual_review_items(blind_pairs=blind_pairs, sample_count=24)
    bad = validate_phase40_manual_review_decision(
        {"decision": "prefer_a", "reviewer_id": "", "reason": ""},
        review_items[0],
    )
    assert bad["status"] == "blocked"
    assert "reviewer_id_required" in bad["errors"]
    assert "timestamp_required" in bad["errors"]

    decisions = []
    for item in review_items[:PHASE40_MIN_REVIEWED_PREFERENCES]:
        decisions.append(
            {
                "review_item_id": item["review_item_id"],
                "decision": "prefer_a",
                "reviewer_id": "manual-reviewer-001",
                "timestamp": "2026-07-06T00:00:00+00:00",
                "reason": "variant_a is more concrete and evidence-first.",
                "chosen_variant": "variant_a",
                "rejected_variant": "variant_b",
                "consent_for_training_candidate_review": True,
            }
        )
    summary = build_phase40_manual_review_summary(review_items=review_items, review_decisions=decisions)
    candidate = build_phase40_preference_candidate_manifest(
        review_items=review_items,
        manual_review_summary=summary,
    )

    assert summary["manual_reviewed_preference_count"] == PHASE40_MIN_REVIEWED_PREFERENCES
    assert candidate["training_candidate_status"] == "ready"
    assert candidate["preference_source"] == "simulated_acceptance_preference"
    assert candidate["actual_product_benefit_claim_allowed"] is False
    assert candidate["selected_preference_pair_count"] == PHASE40_MIN_REVIEWED_PREFERENCES


def test_phase40_decision_gate_collects_manual_review_by_default() -> None:
    scenarios, transcripts_by_variant, blind_pairs, bundle = _phase40_bundle(count=100)
    review_items = build_phase40_manual_review_items(blind_pairs=blind_pairs, sample_count=24)
    manual_summary = build_phase40_manual_review_summary(review_items=review_items, review_decisions=[])
    candidate = build_phase40_preference_candidate_manifest(
        review_items=review_items,
        manual_review_summary=manual_summary,
    )
    boundary = validate_phase40_boundaries(
        scenarios=scenarios,
        transcripts=[item for rows in transcripts_by_variant.values() for item in rows],
        blind_pairs=blind_pairs,
        review_items=review_items,
        candidate_manifest=candidate,
    )
    decision = phase40_final_decision(
        phase39_recap=bundle["recap"],
        acceptance_scores=bundle["scores"],
        manual_review_summary=manual_summary,
        candidate_manifest=candidate,
        boundary_check=boundary,
    )
    comparison = build_phase40_comparison_summary(
        scenario_validation=validate_phase40_scenario_bank(scenarios),
        acceptance_scores=bundle["scores"],
        manual_review_summary=manual_summary,
        candidate_manifest=candidate,
        final_decision=decision,
    )

    assert boundary["passed"] is True
    assert decision["recommendation"] in {"continue_lab_validation", "collect_manual_review"}
    assert decision["recommendation"] == "collect_manual_review"
    assert decision["actual_product_benefit_claim_allowed"] is False
    assert decision["auto_promotion_allowed"] is False
    assert comparison["evidence_type"] == "simulated_lab_evidence"


def test_phase40_decision_can_reach_manual_training_probe_without_actual_claim() -> None:
    scenarios, transcripts_by_variant, blind_pairs, bundle = _phase40_bundle(count=100)
    review_items = build_phase40_manual_review_items(blind_pairs=blind_pairs, sample_count=24)
    decisions = [
        {
            "review_item_id": item["review_item_id"],
            "decision": "prefer_a",
            "reviewer_id": "manual-reviewer-001",
            "timestamp": "2026-07-06T00:00:00+00:00",
            "reason": "chosen output is more actionable.",
            "chosen_variant": "variant_a",
            "rejected_variant": "variant_b",
            "consent_for_training_candidate_review": True,
        }
        for item in review_items[:PHASE40_MIN_REVIEWED_PREFERENCES]
    ]
    manual_summary = build_phase40_manual_review_summary(review_items=review_items, review_decisions=decisions)
    candidate = build_phase40_preference_candidate_manifest(
        review_items=review_items,
        manual_review_summary=manual_summary,
    )
    boundary = validate_phase40_boundaries(
        scenarios=scenarios,
        transcripts=[item for rows in transcripts_by_variant.values() for item in rows],
        blind_pairs=blind_pairs,
        review_items=review_items,
        candidate_manifest=candidate,
    )
    decision = phase40_final_decision(
        phase39_recap=bundle["recap"],
        acceptance_scores=bundle["scores"],
        manual_review_summary=manual_summary,
        candidate_manifest=candidate,
        boundary_check=boundary,
    )

    assert decision["recommendation"] == "ready_for_manual_training_probe"
    assert decision["evidence_type"] == "manual_reviewed_preference_evidence"
    assert decision["actual_product_benefit_claim_allowed"] is False
    assert decision["actual_user_feedback_count"] == 0


def test_phase36_39_regression_still_reports_lab_only_summary() -> None:
    summary = _phase39_summary()

    assert summary["phase36_review_queue_available"] is True
    assert summary["phase37_candidate_generation_available"] is True
    assert summary["approved_actual_candidate_count"] == 0
    assert summary["actual_candidate_lane_status"] == "blocked"
    assert summary["evidence_type"] == "simulated_lab_evidence"
