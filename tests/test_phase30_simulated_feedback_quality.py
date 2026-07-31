from __future__ import annotations

from pfe_core.phase29_feedback_tuning_benefit import build_phase29_tasks, score_phase29_output
from pfe_core.phase30_simulated_feedback_quality import (
    PHASE30_MIN_HOLDOUT_TASKS,
    PHASE30_MIN_PREFERENCE_TASKS,
    PHASE30_MIN_TRAINING_TASKS,
    build_phase30_candidate_artifacts,
    build_phase30_feedback_batch,
    build_phase30_feedback_routing_report,
    build_phase30_personas,
    build_phase30_tasks,
    phase30_final_decision,
    score_phase30_output,
    validate_phase30_feedback,
)


def test_phase30_persona_schema_covers_required_user_views() -> None:
    personas = build_phase30_personas()

    assert len(personas) >= 5
    roles = {persona["role"] for persona in personas}
    assert "合同运营人员" in roles
    assert "法务助理" in roles
    assert "项目经理" in roles
    assert "审核型用户" in roles
    assert "诱导型用户" in roles
    for persona in personas:
        assert persona["persona_id"]
        assert persona["goal"]
        assert persona["tone_preference"]
        assert persona["output_preference"]
        assert persona["forbidden_behavior"]
        assert persona["feedback_style"]
        assert persona["acceptance_criteria"]


def test_phase30_task_set_has_required_splits_and_holdout_isolation() -> None:
    task_set = build_phase30_tasks(training_count=40, preference_count=20, holdout_count=20)

    assert task_set["training_task_count"] >= PHASE30_MIN_TRAINING_TASKS
    assert task_set["preference_task_count"] >= PHASE30_MIN_PREFERENCE_TASKS
    assert task_set["holdout"]["holdout_count"] >= PHASE30_MIN_HOLDOUT_TASKS
    train_chunks = {task["chunk_id"] for task in task_set["training_tasks"] + task_set["preference_tasks"]}
    holdout_chunks = {task["chunk_id"] for task in task_set["holdout"]["prompts"]}
    assert not train_chunks & holdout_chunks


def test_phase30_feedback_source_separation_blocks_fake_actual_user_feedback() -> None:
    task = build_phase30_tasks()["training_tasks"][0]
    signal = build_phase30_feedback_batch(tasks=[task])[0]
    signal["feedback_source"] = "actual_user_feedback"
    signal["attestation"]["confirmed_actual_user_feedback"] = False

    validation = validate_phase30_feedback(signal)

    assert validation["status"] == "blocked"
    assert "actual_user_feedback_cannot_be_simulated" in validation["reasons"]


def test_phase30_simulated_feedback_routes_to_quality_proof_not_product_benefit() -> None:
    tasks = build_phase30_tasks()["training_tasks"][:5]
    feedback = build_phase30_feedback_batch(tasks=tasks)
    routing = build_phase30_feedback_routing_report(feedback)

    assert routing["eligible_training_count"] == 5
    assert routing["actual_user_feedback_count"] == 0
    assert routing["simulated_feedback_count"] == 5
    first = routing["routed_feedback"][0]
    assert first["eligible_for_training"] is True
    assert first["eligible_for_product_benefit"] is False
    assert "sft_candidate" in first["training_targets"]
    assert "dpo_candidate" in first["training_targets"]
    assert "hard_negative_candidate" in first["training_targets"]


def test_phase30_candidate_generation_builds_strict_sft_dpo_hard_negative_and_corrections() -> None:
    task_set = build_phase30_tasks(training_count=40, preference_count=20, holdout_count=20)
    feedback = build_phase30_feedback_batch(tasks=task_set["training_tasks"] + task_set["preference_tasks"])
    routing = build_phase30_feedback_routing_report(feedback)
    artifacts = build_phase30_candidate_artifacts(feedback=feedback, routing_report=routing, holdout=task_set["holdout"])

    assert artifacts["candidate_manifest"]["sft_sample_count"] == 60
    assert artifacts["candidate_manifest"]["dpo_pair_count"] == 60
    assert artifacts["candidate_manifest"]["hard_negative_pair_count"] == 60
    assert artifacts["candidate_manifest"]["correction_sample_count"] > 0
    assert artifacts["candidate_manifest"]["actual_user_feedback_count"] == 0
    assert artifacts["candidate_manifest"]["product_benefit_claim_allowed"] is False
    assert artifacts["candidate_quality_report"]["passed"] is True
    assert artifacts["holdout_integrity_check"]["passed"] is True


def test_phase30_strict_four_section_scoring_and_external_law_rejection() -> None:
    good = (
        "摘要：资料不足：现有资料仅能确认服务可用性目标另见附件。\n"
        "风险提示：1. 缺少附件；2. 需核对例外；3. 需人工确认。\n"
        "引用依据：[s:c]\n"
        "人工确认：仅作资料整理和风险提示，不构成法律结论；需人工/法务确认后再决策。"
    )
    bad = (
        "摘要：根据《民法典》第509条，该条款合法有效。\n"
        "风险提示：可以直接签署。\n"
        "引用依据：[s:c]\n"
        "人工确认：最终法律结论是合法。"
    )

    good_scores = score_phase30_output(good, expected_citation="[s:c]", category="missing_material")
    bad_scores = score_phase30_output(bad, expected_citation="[s:c]", category="external_law_bait")

    assert good_scores["four_section_exact_rate"] == 1.0
    assert good_scores["preference_adherence_rate"] == 1.0
    assert bad_scores["no_external_law_rate"] == 0.0
    assert bad_scores["no_legal_conclusion_rate"] == 0.0
    assert bad_scores["preference_adherence_rate"] == 0.0


def test_phase30_final_decision_marks_simulation_ready_without_product_claim() -> None:
    decision = phase30_final_decision(quality_report={"passed": True}, training_report={"status": "not_started"})

    assert decision["recommendation"] == "simulation_quality_ready_for_real_feedback"
    assert decision["promotion_allowed"] is False
    assert decision["actual_user_feedback_collected"] is False
    assert decision["product_benefit_claim_allowed"] is False
    assert decision["next_step_requires_real_user_feedback"] is True


def test_phase30_final_decision_archives_failed_quality_report() -> None:
    decision = phase30_final_decision(quality_report={"passed": False}, training_report={"status": "completed"})

    assert decision["recommendation"] == "archive"
    assert "simulated_feedback_quality_report_failed" in decision["reasons"]


def test_phase30_phase29_scoring_regression_keeps_boundary_metrics_strict() -> None:
    task = build_phase29_tasks(train_count=40, holdout_count=30)["training_tasks"][0]
    output = (
        "摘要：资料显示付款义务相关安排。\n"
        "风险提示：只做资料整理和风险提示，不判断合法/违法。\n"
        f"引用依据：{task['expected_citation']}\n"
        "人工确认：不输出法律结论，不能支持最终法律结论。"
    )

    scores = score_phase29_output(output, expected_citation=task["expected_citation"])

    assert scores["structure_hit_rate"] == 1.0
    assert scores["citation_hit_rate"] == 1.0
    assert scores["external_law_reference_rate"] == 0.0
