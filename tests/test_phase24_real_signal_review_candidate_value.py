from __future__ import annotations

from pfe_core.phase23_runtime_contract_loop import build_phase23_holdout, evaluate_runtime_contract_holdout
from pfe_core.phase24_real_signal_review_candidate_value import (
    PHASE24_FEEDBACK_SOURCES,
    PHASE24_FEEDBACK_TYPES,
    apply_phase24_review_decisions,
    build_phase24_candidate_artifacts,
    build_phase24_feedback_signals,
    build_phase24_holdout,
    build_phase24_interactions,
    build_phase24_model_selection,
    build_phase24_review_queue,
    build_phase24_routing_report,
    build_phase24_training_feasibility,
    build_phase24_training_job_specs,
    evaluate_phase24_runtime_contract_holdout,
    phase24_holdout_integrity_check,
    phase24_runtime_product_decision,
    phase24_training_decision,
)


def _phase24_flow():
    capture = build_phase24_interactions(count=80)
    feedback = build_phase24_feedback_signals(capture)
    queue = build_phase24_review_queue(feedback["signals"])
    reviewed = apply_phase24_review_decisions(queue, feedback["signals"])
    routing = build_phase24_routing_report(reviewed, feedback["signals"])
    holdout = build_phase24_holdout(regression_count=50, hard_count=50)
    holdout_chunk_ids = {item["chunk_id"] for item in holdout["prompts"] if item.get("chunk_id")}
    candidates = build_phase24_candidate_artifacts(
        signals=feedback["signals"],
        reviewed=reviewed,
        routing_report=routing,
        holdout_chunk_ids=holdout_chunk_ids,
    )
    integrity = phase24_holdout_integrity_check(
        holdout=holdout,
        sft_samples=candidates["sft_samples"],
        dpo_pairs=candidates["dpo_pairs"],
    )
    return capture, feedback, queue, reviewed, routing, holdout, candidates, integrity


def test_phase24_interaction_capture_schema_and_runtime_contract_smoke() -> None:
    capture = build_phase24_interactions(count=80)

    assert capture["interaction_count"] == 80
    assert capture["runtime_output_count"] == 80
    assert capture["real_runtime_contract_calls"] is True
    first_output = capture["runtime_outputs"][0]
    assert first_output["real_runtime_contract_call"] is True
    assert first_output["scores"]["structure_hit_rate"] == 1.0
    assert first_output["scores"]["citation_hit_rate"] == 1.0
    assert first_output["scores"]["think_leak_rate"] == 0.0


def test_phase24_feedback_signal_schema_tracks_types_and_provenance() -> None:
    feedback = build_phase24_feedback_signals(build_phase24_interactions(count=80))

    assert feedback["signal_count"] == 80
    assert set(feedback["feedback_type_counts"]) == PHASE24_FEEDBACK_TYPES
    assert set(feedback["feedback_source_counts"]) == PHASE24_FEEDBACK_SOURCES
    assert feedback["actual_user_feedback_count"] == 0
    assert all(signal["feedback_source_is_actual_user_feedback"] is False for signal in feedback["signals"])


def test_phase24_review_queue_state_transitions() -> None:
    _, feedback, queue, reviewed, _, _, _, _ = _phase24_flow()

    assert queue["queue_count"] == feedback["signal_count"]
    assert "pending_review" in queue["state_counts"]
    assert reviewed["reviewed_count"] == feedback["signal_count"]
    assert "approved_for_candidate" in reviewed["state_counts"]
    assert "excluded" in reviewed["state_counts"]
    assert "needs_more_context" in queue["state_counts"]


def test_phase24_routing_exclusions_and_candidate_generation() -> None:
    _, feedback, _, reviewed, routing, _, candidates, integrity = _phase24_flow()

    assert routing["signal_count"] == feedback["signal_count"]
    assert routing["eligible_training_count"] > 0
    assert routing["product_value_training_allowed_count"] == 0
    assert routing["excluded_reason_counts"]
    assert "excluded" in routing["route_counts"]
    assert candidates["candidate_manifest"]["sft_sample_count"] > 0
    assert candidates["candidate_manifest"]["dpo_pair_count"] > 0
    assert candidates["quality_report"]["passed"] is True
    assert integrity["passed"] is True
    assert integrity["contaminated_chunk_ids"] == []


def test_phase24_holdout_has_100_prompts_and_runtime_eval_stable() -> None:
    holdout = build_phase24_holdout(regression_count=50, hard_count=50)
    eval_report = evaluate_phase24_runtime_contract_holdout(holdout)
    decision = phase24_runtime_product_decision(eval_report)

    assert holdout["holdout_count"] == 100
    assert holdout["regression_holdout_count"] == 50
    assert holdout["hard_holdout_count"] == 50
    assert eval_report["holdout_count"] == 100
    assert eval_report["scores"]["structure_hit_rate"] == 1.0
    assert eval_report["scores"]["citation_hit_rate"] == 1.0
    assert eval_report["scores"]["safety_boundary_rate"] == 1.0
    assert eval_report["scores"]["unsupported_assertions"] == 0
    assert decision["recommendation"] == "primary_product_path"


def test_phase24_job_specs_and_decision_archive_without_actual_user_feedback() -> None:
    _, feedback, _, _, _, _, candidates, _ = _phase24_flow()
    model_selection = build_phase24_model_selection(
        local_models=[{"name": "Qwen-test-7B", "path": "/models/qwen-test-7b", "trainable": True}]
    )
    feasibility = build_phase24_training_feasibility(
        candidate_manifest=candidates["candidate_manifest"],
        model_selection=model_selection,
        actual_user_feedback_count=feedback["actual_user_feedback_count"],
    )
    job_specs = build_phase24_training_job_specs(
        candidate_manifest=candidates["candidate_manifest"],
        model_selection=model_selection,
        feasibility=feasibility,
    )
    decision = phase24_training_decision(
        runtime_scores={
            "structure_hit_rate": 1.0,
            "citation_hit_rate": 1.0,
            "safety_boundary_rate": 1.0,
            "explicit_boundary_rate": 1.0,
            "unsupported_assertions": 0,
            "external_law_reference_rate": 0.0,
            "think_leak_rate": 0.0,
            "extra_text_after_first_block_rate": 0.0,
        },
        sft_scores=None,
        dpo_scores=None,
        feasibility=feasibility,
        candidate_manifest=candidates["candidate_manifest"],
    )

    assert feasibility["status"] == "blocked"
    assert "insufficient_actual_user_feedback_for_product_value_training_probe" in feasibility["blockers"]
    assert job_specs["status"] == "dry_run_blocked"
    assert decision["recommendation"] == "archive"
    assert decision["auto_promotion_allowed"] is False


def test_phase23_regression_runtime_contract_still_stable() -> None:
    holdout = build_phase23_holdout(count=50)
    eval_report = evaluate_runtime_contract_holdout(holdout)

    assert eval_report["holdout_count"] == 50
    assert eval_report["scores"]["structure_hit_rate"] == 1.0
    assert eval_report["scores"]["citation_hit_rate"] == 1.0
    assert eval_report["scores"]["unsupported_assertions"] == 0
