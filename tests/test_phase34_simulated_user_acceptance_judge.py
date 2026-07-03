from __future__ import annotations

from pfe_core.phase33_simulated_usage_replay import phase33_final_decision
from pfe_core.phase34_simulated_user_acceptance_judge import (
    PHASE34_FEEDBACK_SOURCE,
    PHASE34_MAX_SCENARIOS,
    PHASE34_MIN_SCENARIOS,
    aggregate_phase34_judgements,
    blind_pair_public_view,
    build_phase34_acceptance_scenarios,
    build_phase34_blind_eval_pairs,
    build_phase34_default_inputs,
    build_phase34_phase33_review,
    judge_phase34_blind_pair,
    phase34_final_decision,
    validate_phase34_blind_pair,
    validate_phase34_simulation_boundaries,
)


def _phase33_summary() -> dict:
    return {
        "status": "completed",
        "session_count": 64,
        "actual_user_feedback_count": 0,
        "final_recommendation": "promote_after_manual_review",
        "eval_report": {
            "base": {"scores": {"overall_replay_score": 0.633}},
            "adapter": {"scores": {"overall_replay_score": 0.912}},
            "score_delta": {"overall_replay_score": 0.279},
        },
    }


def test_phase34_phase33_review_states_scope_and_boundaries() -> None:
    review = build_phase34_phase33_review(
        phase33_summary=_phase33_summary(),
        phase33_decision_text="Phase33 simulated replay only.",
    )

    assert review["phase33_completed"] is True
    assert review["phase33_actual_user_feedback_count"] == 0
    assert review["phase34_does_not_train"] is True
    assert review["phase34_does_not_auto_promote"] is True
    assert review["phase34_does_not_collect_actual_feedback"] is True


def test_phase34_acceptance_scenarios_cover_required_range_and_are_not_training_data() -> None:
    batch = build_phase34_acceptance_scenarios(count=100, phase33_reference=_phase33_summary())

    assert PHASE34_MIN_SCENARIOS <= batch["scenario_count"] <= PHASE34_MAX_SCENARIOS
    assert batch["actual_user_feedback_count"] == 0
    assert batch["source"] == PHASE34_FEEDBACK_SOURCE
    assert len(batch["categories"]) >= 10
    for scenario in batch["scenarios"]:
        assert scenario["feedback_source"] == PHASE34_FEEDBACK_SOURCE
        assert scenario["simulated_user_judgement"] is True
        assert scenario["confirmed_actual_user_feedback"] is False
        assert scenario["not_actual_user_feedback"] is True
        assert scenario["not_for_training"] is True
        assert scenario["user_intent"]
        assert scenario["expected_outcome"]
        assert scenario["user_correction"]
        assert scenario["continuation_need"]
        assert "no_false_completion" in scenario["acceptance_lens"]


def test_phase34_blind_eval_public_view_does_not_expose_variant_identity() -> None:
    generated = build_phase34_default_inputs(scenario_count=80, phase33_summary=_phase33_summary())
    pair = generated["blind_eval_pairs"][0]
    public = blind_pair_public_view(pair)
    validation = validate_phase34_blind_pair(pair)

    assert "blind_variant_map" not in public
    assert public["variant_a"]["label"] == "variant_a"
    assert public["variant_b"]["label"] == "variant_b"
    assert "model_variant" not in str(public)
    assert validation["passed"] is True
    assert validation["identity_leaked_to_judge"] is False


def test_phase34_simulated_user_judge_schema_and_preference_unblind() -> None:
    generated = build_phase34_default_inputs(scenario_count=80, phase33_summary=_phase33_summary())
    judgement = judge_phase34_blind_pair(generated["blind_eval_pairs"][0])

    assert judgement["kind"] == "phase34_simulated_user_judgement"
    assert judgement["feedback_source"] == PHASE34_FEEDBACK_SOURCE
    assert judgement["confirmed_actual_user_feedback"] is False
    assert judgement["acceptance_decision"] in {"accept", "reject", "needs_edit", "blocked"}
    assert judgement["preferred_variant"] in {"variant_a", "variant_b", "tie"}
    assert judgement["preferred_model_after_unblind"] in {"base", "adapter", "tie"}
    assert 0.0 <= judgement["perceived_value_score"] <= 1.0
    assert 0.0 <= judgement["trust_score"] <= 1.0
    assert 0.0 <= judgement["user_effort_reduction_score"] <= 1.0
    assert 0.0 <= judgement["frustration_score"] <= 1.0


def test_phase34_acceptance_scores_and_decision_gate_promote_only_after_user_value_win() -> None:
    generated = build_phase34_default_inputs(scenario_count=100, phase33_summary=_phase33_summary())
    scores = generated["acceptance_scores"]
    decision = generated["decision"]

    assert scores["adapter_win_rate"] > scores["base_win_rate"]
    assert scores["adapter"]["user_effort_reduction_rate"] > scores["base"]["user_effort_reduction_rate"]
    assert scores["adapter"]["frustration_score"] < scores["base"]["frustration_score"]
    assert scores["adapter"]["would_continue_using_rate"] > scores["base"]["would_continue_using_rate"]
    assert decision["recommendation"] == "promote_after_manual_review"
    assert decision["auto_promotion_allowed"] is False
    assert decision["product_benefit_claim_allowed"] is False


def test_phase34_decision_archives_when_adapter_does_not_win() -> None:
    scores = aggregate_phase34_judgements([])
    scores = {
        **scores,
        "adapter_win_rate": 0.0,
        "base_win_rate": 1.0,
        "base": {
            "acceptance_rate": 1.0,
            "user_effort_reduction_rate": 1.0,
            "frustration_score": 0.0,
            "false_completion_penalty_rate": 0.0,
            "privacy_boundary_trust_rate": 1.0,
            "would_continue_using_rate": 1.0,
            "overall_product_value_score": 1.0,
        },
        "adapter": {
            "acceptance_rate": 0.0,
            "user_effort_reduction_rate": 0.0,
            "frustration_score": 1.0,
            "false_completion_penalty_rate": 0.0,
            "privacy_boundary_trust_rate": 1.0,
            "would_continue_using_rate": 0.0,
            "overall_product_value_score": 0.0,
        },
    }
    decision = phase34_final_decision(
        acceptance_scores=scores,
        boundary_check={"passed": True},
    )

    assert decision["recommendation"] == "archive"
    assert decision["auto_promotion_allowed"] is False
    assert "adapter_win_rate_not_above_base" in decision["reasons"]


def test_phase34_boundary_check_blocks_actual_feedback_and_private_text() -> None:
    scenarios = build_phase34_acceptance_scenarios(count=80)["scenarios"]
    bad_scenario = {
        **scenarios[0],
        "feedback_source": "actual_user_feedback",
        "confirmed_actual_user_feedback": True,
        "user_intent": "/Users/lichenhao/AgentMemory/Conversations/2026-06-22_14-01_Codex.md",
    }
    check = validate_phase34_simulation_boundaries(scenarios=[bad_scenario], pairs=[], judgements=[])

    assert check["passed"] is False
    reasons = {item["reason"] for item in check["problems"]}
    assert "feedback_source_not_simulated_user_judgement" in reasons
    assert "confirmed_actual_user_feedback_true" in reasons
    assert "raw_private_text_detected" in reasons


def test_phase33_regression_final_decision_never_auto_promotes() -> None:
    decision = phase33_final_decision(
        eval_report={
            "status": "completed",
            "source": "simulated_usage",
            "actual_user_feedback_count": 0,
            "same_session_comparison": True,
            "base": {"scores": {"overall_replay_score": 0.5, "raw_private_text_leak_rate": 0.0, "actual_feedback_mislabel_rate": 0.0}},
            "adapter": {
                "scores": {
                    "overall_replay_score": 0.8,
                    "execution_first_rate": 1.0,
                    "evidence_grounding_rate": 1.0,
                    "boundary_awareness_rate": 1.0,
                    "correction_responsiveness_rate": 1.0,
                    "persistence_rate": 1.0,
                    "local_context_awareness_rate": 1.0,
                    "final_acceptance_rate": 1.0,
                    "raw_private_text_leak_rate": 0.0,
                    "actual_feedback_mislabel_rate": 0.0,
                    "hallucinated_completion_rate": 0.0,
                }
            },
        },
        phase32_reference={"real_training": "completed"},
    )

    assert decision["recommendation"] == "promote_after_manual_review"
    assert decision["auto_promotion_allowed"] is False
