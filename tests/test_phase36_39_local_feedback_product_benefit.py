from __future__ import annotations

from pfe_core.phase35_local_interaction_capture import build_phase35_interaction_record
from pfe_core.phase36_39_local_feedback_product_benefit import (
    PHASE37_MIN_ACTUAL_APPROVED,
    PHASE39_FEEDBACK_SOURCE,
    build_phase36_39_simulated_lab_records,
    build_phase36_review_decision,
    build_phase36_review_queue,
    build_phase36_review_summary,
    build_phase37_candidate_artifacts,
    build_phase37_holdout,
    build_phase38_model_selection,
    build_phase38_training_attempt,
    build_phase38_training_manifest,
    build_phase39_blind_eval_pairs,
    build_phase39_eval_report,
    build_phase39_simulated_sessions,
    build_phase39_transcripts,
    is_phase36_actual_attested,
    phase39_final_decision,
    validate_phase36_review_decision,
    validate_phase39_boundaries,
)


def _actual_record(index: int = 1) -> dict:
    return build_phase35_interaction_record(
        workspace="phase36-test",
        user_goal=f"真实本地交互 {index}：先查证据再提交。",
        assistant_response="我先检查 git status、测试和 gate，再提交。",
        feedback_action="accept",
        user_feedback="真实用户确认这条符合偏好。",
        operator_id="local-user",
        confirmed_actual_user_feedback=True,
        consent_for_training_candidate_review=True,
        not_scripted_or_curated=True,
    )


def test_phase36_review_decision_approves_only_attested_actual_feedback() -> None:
    record = _actual_record()
    decision = build_phase36_review_decision(
        record,
        state="approve_for_candidate",
        reviewer_id="reviewer-001",
        reason="完整 attestation，可进入候选。",
    )
    queue = build_phase36_review_queue({"interactions": [record], "review_decisions": []})

    assert is_phase36_actual_attested(record) is True
    assert decision["validation"]["passed"] is True
    assert queue["pending_review_count"] == 1

    simulated = build_phase36_39_simulated_lab_records(count=1)[0]
    blocked = build_phase36_review_decision(
        simulated,
        state="approve_for_candidate",
        reviewer_id="reviewer-001",
        reason="尝试错误批准模拟记录。",
    )

    assert blocked["validation"]["status"] == "blocked"
    assert "simulated_usage_cannot_approve_for_actual_candidate" in blocked["validation"]["errors"]
    assert "approve_requires_attested_actual_feedback" in blocked["validation"]["errors"]


def test_phase36_review_decision_requires_reason_reviewer_and_quarantines_private_text() -> None:
    record = _actual_record()
    record["feedback"]["user_feedback"] = "/Users/lichenhao/AgentMemory/Conversations/2026-06-22_14-01_Codex.md"
    validation = validate_phase36_review_decision(
        {"state": "approve_for_candidate", "reviewer_id": "", "reason": ""},
        record,
    )

    assert validation["status"] == "blocked"
    assert "reviewer_id_required" in validation["errors"]
    assert "reason_required" in validation["errors"]
    assert "raw_private_text_detected" in validation["quarantine_reasons"]


def test_phase37_actual_lane_blocks_below_threshold_and_simulated_lane_is_explicit_lab_only() -> None:
    actual = [_actual_record(1)]
    decisions = [
        build_phase36_review_decision(
            actual[0],
            state="approve_for_candidate",
            reviewer_id="reviewer-001",
            reason="approved but below threshold",
        )
    ]
    holdout = build_phase37_holdout(count=40)

    actual_artifacts = build_phase37_candidate_artifacts(
        records=actual,
        review_decisions=decisions,
        holdout=holdout,
        lane="actual_candidate_lane",
    )
    simulated_artifacts = build_phase37_candidate_artifacts(
        records=build_phase36_39_simulated_lab_records(count=PHASE37_MIN_ACTUAL_APPROVED),
        review_decisions=[],
        holdout=holdout,
        lane="simulated_lab_candidate_lane",
    )

    assert actual_artifacts["candidate_manifest"]["status"] == "blocked"
    assert actual_artifacts["candidate_manifest"]["blocked_reason"] == "approved_actual_feedback_below_threshold"
    assert actual_artifacts["candidate_manifest"]["actual_user_feedback_count"] == 1
    assert actual_artifacts["candidate_manifest"]["actual_product_benefit_claim_allowed"] is False
    assert actual_artifacts["candidate_manifest"]["sft_sample_count"] == 0
    assert actual_artifacts["candidate_manifest"]["dpo_pair_count"] == 0
    assert simulated_artifacts["candidate_manifest"]["status"] == "ready"
    assert simulated_artifacts["candidate_manifest"]["actual_product_benefit_claim_allowed"] is False
    assert simulated_artifacts["candidate_manifest"]["simulated_lab_sample_count"] == PHASE37_MIN_ACTUAL_APPROVED
    assert simulated_artifacts["candidate_quality_report"]["passed"] is True


def test_phase37_candidates_have_dpo_contrast_and_holdout_isolation() -> None:
    holdout = build_phase37_holdout(count=40)
    records = build_phase36_39_simulated_lab_records(count=12)
    artifacts = build_phase37_candidate_artifacts(
        records=records,
        review_decisions=[],
        holdout=holdout,
        lane="simulated_lab_candidate_lane",
    )

    pair = artifacts["dpo_pairs"][0]
    sample = artifacts["sft_samples"][0]

    assert pair["chosen"] != pair["rejected"]
    assert "不假装" in sample["output"]
    assert artifacts["holdout_integrity_check"]["passed"] is True
    assert artifacts["candidate_quality_report"]["aggregate"]["holdout_isolation_rate"] == 1.0


def test_phase37_holdout_contamination_is_excluded() -> None:
    holdout = build_phase37_holdout(count=40)
    record = build_phase36_39_simulated_lab_records(count=1)[0]
    record["user_goal"] = f"污染 {holdout['prompts'][0]['prompt_id']}"

    artifacts = build_phase37_candidate_artifacts(
        records=[record],
        review_decisions=[],
        holdout=holdout,
        lane="simulated_lab_candidate_lane",
    )

    assert artifacts["candidate_manifest"]["excluded_count"] == 1
    assert artifacts["excluded"][0]["reason"] == "holdout_contamination"


def test_phase38_model_selection_prefers_small_unquantized_qwen_and_training_manifest() -> None:
    selection = build_phase38_model_selection(
        local_models=[
            {"name": "Qwen3.6-27B", "path": "/models/Qwen3.6-27B", "trainable": True},
            {"name": "Qwen2.5-0.5B-Instruct-4bit", "path": "/models/Qwen2.5-0.5B-Instruct-4bit", "trainable": True, "quantization": "4bit"},
            {"name": "Qwen2.5-0.5B-Instruct", "path": "/models/Qwen2.5-0.5B-Instruct", "trainable": True},
        ]
    )
    holdout = build_phase37_holdout(count=40)
    artifacts = build_phase37_candidate_artifacts(
        records=build_phase36_39_simulated_lab_records(count=12),
        review_decisions=[],
        holdout=holdout,
        lane="simulated_lab_candidate_lane",
    )
    manifest = build_phase38_training_manifest(
        candidate_artifacts=artifacts,
        model_selection=selection,
        step_count=12,
    )
    attempt = build_phase38_training_attempt(
        training_manifest=manifest,
        execution_result={
            "status": "completed",
            "real_execution": {
                "success": True,
                "artifact_dir": "/tmp/phase38/adapter",
                "artifact_kind": "real_local_peft",
                "runtime_path": "real_local",
                "train_loss": 1.23,
            },
        },
    )

    assert selection["status"] == "selected"
    assert selection["selected_model_name"] == "Qwen2.5-0.5B-Instruct"
    assert selection["not_27b_training"] is True
    assert manifest["lane"] == "simulated_lab_candidate_lane"
    assert manifest["step_equivalent_count"] == 12
    assert attempt["status"] == "completed"
    assert attempt["adapter_validation"]["valid"] is True


def test_phase39_simulated_sessions_blind_eval_and_boundary_checks() -> None:
    sessions = build_phase39_simulated_sessions(count=50)
    transcripts_by_variant = {
        variant: build_phase39_transcripts(sessions=sessions, model_variant=variant)
        for variant in ("base", "runtime_contract", "adapter", "adapter_runtime_contract")
    }
    pairs = build_phase39_blind_eval_pairs(sessions=sessions, transcripts_by_variant=transcripts_by_variant)
    eval_report = build_phase39_eval_report(transcripts_by_variant=transcripts_by_variant)
    boundary = validate_phase39_boundaries(
        sessions=sessions,
        transcripts=[item for rows in transcripts_by_variant.values() for item in rows],
        blind_pairs=pairs,
    )

    assert len(sessions) == 50
    assert all(session["feedback_source"] == PHASE39_FEEDBACK_SOURCE for session in sessions)
    assert "blind_variant_map" in pairs[0]
    assert "model_variant" not in str({key: value for key, value in pairs[0].items() if key != "blind_variant_map"})
    assert boundary["passed"] is True
    assert eval_report["variants"]["adapter"]["scores"]["usefulness_as_personal_agent"] > eval_report["variants"]["base"]["scores"]["usefulness_as_personal_agent"]
    assert eval_report["variants"]["adapter_runtime_contract"]["scores"]["usefulness_as_personal_agent"] >= eval_report["variants"]["adapter"]["scores"]["usefulness_as_personal_agent"]


def test_phase39_decision_is_lab_only_without_actual_approved_feedback() -> None:
    review_summary = build_phase36_review_summary(
        state={"interactions": [], "review_decisions": []},
        review_decisions=[],
    )
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
    manifest = build_phase38_training_manifest(
        candidate_artifacts=simulated,
        model_selection={"status": "selected", "selected_model": "/models/Qwen2.5-0.5B-Instruct"},
    )
    attempt = build_phase38_training_attempt(
        training_manifest=manifest,
        execution_result={"status": "completed", "real_execution": {"success": True, "artifact_dir": "/tmp/adapter"}},
    )
    sessions = build_phase39_simulated_sessions(count=50)
    transcripts_by_variant = {
        variant: build_phase39_transcripts(sessions=sessions, model_variant=variant)
        for variant in ("base", "runtime_contract", "adapter", "adapter_runtime_contract")
    }
    eval_report = build_phase39_eval_report(transcripts_by_variant=transcripts_by_variant)
    boundary = validate_phase39_boundaries(
        sessions=sessions,
        transcripts=[item for rows in transcripts_by_variant.values() for item in rows],
        blind_pairs=build_phase39_blind_eval_pairs(sessions=sessions, transcripts_by_variant=transcripts_by_variant),
    )
    decision = phase39_final_decision(
        review_summary=review_summary,
        actual_candidates=actual,
        simulated_candidates=simulated,
        training_attempt=attempt,
        eval_report=eval_report,
        boundary_check=boundary,
    )

    assert decision["simulated_lab_evidence_only"] is True
    assert decision["actual_product_benefit_claim_allowed"] is False
    assert decision["auto_promotion_allowed"] is False
    assert decision["recommendation"] in {"continue_lab_validation", "promote_after_manual_review_for_lab_only"}
