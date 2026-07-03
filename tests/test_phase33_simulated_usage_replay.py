from __future__ import annotations

from pfe_core.phase32_personal_agent_preference import phase32_final_decision
from pfe_core.phase33_simulated_usage_replay import (
    PHASE33_FEEDBACK_SOURCE,
    PHASE33_MAX_SESSIONS,
    PHASE33_MIN_SESSIONS,
    build_phase33_eval_report,
    build_phase33_phase32_reference,
    build_phase33_transcripts,
    build_phase33_usage_sessions,
    phase33_final_decision,
    score_phase33_transcript,
    validate_phase33_simulation_boundaries,
)


def _phase32_summary() -> dict:
    return {
        "status": "completed",
        "final_recommendation": "promote_after_manual_review",
        "training_attempt": {
            "real_training": "completed",
            "selected_model": "Qwen/Qwen2.5-0.5B-Instruct",
            "adapter_path": "trainer_job_outputs/phase32-personal-agent-preference-qwen25-0_5b/dpo_adapter",
        },
        "base_eval": {
            "scores": {
                "overall_personalization_score": 0.396,
                "boundary_awareness_rate": 0.0,
            }
        },
        "adapter_eval": {
            "scores": {
                "overall_personalization_score": 0.437,
                "boundary_awareness_rate": 0.0,
            }
        },
    }


def test_phase33_session_pack_has_required_size_and_simulation_boundary() -> None:
    batch = build_phase33_usage_sessions(count=64)
    sessions = batch["sessions"]

    assert PHASE33_MIN_SESSIONS <= batch["session_count"] <= PHASE33_MAX_SESSIONS
    assert batch["actual_user_feedback_count"] == 0
    assert batch["source"] == PHASE33_FEEDBACK_SOURCE
    assert len(batch["categories"]) >= 8
    for session in sessions:
        assert session["feedback_source"] == PHASE33_FEEDBACK_SOURCE
        assert session["not_actual_user_feedback"] is True
        assert session["confirmed_actual_user_feedback"] is False
        assert session["not_for_training"] is True
        assert session["user_goal"]
        assert session["user_correction"]
        assert session["continue_request"]
        assert session["final_acceptance"]
        assert len(session["turn_plan"]) == 8


def test_phase33_transcripts_include_multiturn_flow_and_same_session_variants() -> None:
    reference = build_phase33_phase32_reference(phase32_summary=_phase32_summary())
    sessions = build_phase33_usage_sessions(count=50)["sessions"]
    base = build_phase33_transcripts(sessions=sessions, model_variant="base", phase32_reference=reference)
    adapter = build_phase33_transcripts(sessions=sessions, model_variant="adapter", phase32_reference=reference)

    assert {item["session_id"] for item in base} == {item["session_id"] for item in adapter}
    assert all(item["actual_model_call"] is False for item in base + adapter)
    assert all(item["feedback_source"] == PHASE33_FEEDBACK_SOURCE for item in base + adapter)
    first = adapter[0]
    stages = [turn["stage"] for turn in first["turns"]]
    assert stages == [
        "user_goal",
        "agent_answer",
        "user_correction",
        "agent_correction_response",
        "user_continue",
        "agent_continue_response",
        "user_final_acceptance",
        "agent_final_response",
    ]
    assert "你说得对" in first["turns"][3]["content"]


def test_phase33_scoring_rewards_adapter_profile_without_private_leaks() -> None:
    reference = build_phase33_phase32_reference(phase32_summary=_phase32_summary())
    sessions = build_phase33_usage_sessions(count=50)["sessions"]
    base = build_phase33_transcripts(sessions=sessions, model_variant="base", phase32_reference=reference)
    adapter = build_phase33_transcripts(sessions=sessions, model_variant="adapter", phase32_reference=reference)

    base_scores = score_phase33_transcript(base[0])
    adapter_scores = score_phase33_transcript(adapter[0])
    assert adapter_scores["overall_replay_score"] > base_scores["overall_replay_score"]
    assert adapter_scores["raw_private_text_leak_rate"] == 0.0
    assert adapter_scores["actual_feedback_mislabel_rate"] == 0.0


def test_phase33_eval_report_compares_same_sessions_and_decision_never_auto_promotes() -> None:
    reference = build_phase33_phase32_reference(phase32_summary=_phase32_summary())
    sessions = build_phase33_usage_sessions(count=64)["sessions"]
    base = build_phase33_transcripts(sessions=sessions, model_variant="base", phase32_reference=reference)
    adapter = build_phase33_transcripts(sessions=sessions, model_variant="adapter", phase32_reference=reference)
    report = build_phase33_eval_report(sessions=sessions, base_transcripts=base, adapter_transcripts=adapter)
    decision = phase33_final_decision(eval_report=report, phase32_reference=reference)

    assert report["same_session_comparison"] is True
    assert report["actual_user_feedback_count"] == 0
    assert report["adapter"]["scores"]["overall_replay_score"] > report["base"]["scores"]["overall_replay_score"]
    assert decision["recommendation"] == "promote_after_manual_review"
    assert decision["auto_promotion_allowed"] is False
    assert decision["actual_user_feedback_collected"] is False
    assert decision["product_benefit_claim_allowed"] is False


def test_phase33_boundary_check_blocks_actual_feedback_mislabel_and_private_paths() -> None:
    sessions = build_phase33_usage_sessions(count=50)["sessions"]
    transcripts = build_phase33_transcripts(sessions=sessions, model_variant="adapter")
    clean = validate_phase33_simulation_boundaries(sessions=sessions, transcripts=transcripts)
    bad = validate_phase33_simulation_boundaries(
        sessions=[
            {
                **sessions[0],
                "feedback_source": "actual_user_feedback",
                "confirmed_actual_user_feedback": True,
            }
        ],
        transcripts=[
            {
                **transcripts[0],
                "turns": [
                    {
                        "role": "assistant",
                        "stage": "agent_answer",
                        "content": "/Users/lichenhao/AgentMemory/Conversations/2026-06-22_14-01_Codex.md",
                    }
                ],
            }
        ],
    )

    assert clean["passed"] is True
    assert bad["passed"] is False
    reasons = {item["reason"] for item in bad["problems"]}
    assert "feedback_source_not_simulated_usage" in reasons
    assert "confirmed_actual_user_feedback_true" in reasons
    assert "raw_private_text_detected" in reasons


def test_phase33_archives_when_phase32_adapter_was_not_trained() -> None:
    reference = build_phase33_phase32_reference(
        phase32_summary={
            **_phase32_summary(),
            "training_attempt": {"real_training": "blocked", "selected_model": "Qwen/Qwen2.5-0.5B-Instruct"},
        }
    )
    sessions = build_phase33_usage_sessions(count=50)["sessions"]
    report = build_phase33_eval_report(
        sessions=sessions,
        base_transcripts=build_phase33_transcripts(sessions=sessions, model_variant="base", phase32_reference=reference),
        adapter_transcripts=build_phase33_transcripts(sessions=sessions, model_variant="adapter", phase32_reference=reference),
    )
    decision = phase33_final_decision(eval_report=report, phase32_reference=reference)

    assert decision["recommendation"] == "archive"
    assert "phase32_adapter_training_not_completed" in decision["reasons"]


def test_phase32_regression_decision_still_requires_adapter_improvement() -> None:
    quality = {"passed": True}
    training = {"real_training": "completed"}
    base = {
        "status": "completed",
        "scores": {
            "overall_personalization_score": 0.7,
            "evidence_grounding_rate": 0.7,
            "boundary_awareness_rate": 0.7,
            "raw_private_text_leak_rate": 0.0,
            "hallucinated_completion_rate": 0.0,
            "execution_first_rate": 0.7,
        },
    }
    adapter = {
        "status": "completed",
        "scores": {
            "overall_personalization_score": 0.6,
            "evidence_grounding_rate": 0.7,
            "boundary_awareness_rate": 0.7,
            "raw_private_text_leak_rate": 0.0,
            "hallucinated_completion_rate": 0.0,
            "execution_first_rate": 0.7,
        },
    }

    decision = phase32_final_decision(
        candidate_quality_report=quality,
        training_attempt=training,
        base_eval=base,
        adapter_eval=adapter,
    )

    assert decision["recommendation"] == "archive"
    assert decision["auto_promotion_allowed"] is False
    assert "adapter_overall_not_above_base" in decision["reasons"]
