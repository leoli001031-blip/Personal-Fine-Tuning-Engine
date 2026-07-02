from __future__ import annotations

from pfe_core.phase31_obsidian_agent_signal_mining import PHASE31_FEEDBACK_SOURCE, phase31_final_decision
from pfe_core.phase32_personal_agent_preference import (
    PHASE32_MIN_HOLDOUT_PROMPTS,
    build_phase32_candidate_artifacts,
    build_phase32_holdout,
    build_phase32_phase31_review,
    build_phase32_review_decisions,
    build_phase32_taxonomy,
    contains_raw_private_text,
    phase32_final_decision,
    score_phase32_output,
    validate_phase32_review_decision,
)


def _signal(index: int, *, signal_type: str = "verification_preference", secret: bool = False) -> dict:
    return {
        "signal_id": f"phase31-signal-{index:03d}",
        "feedback_source": PHASE31_FEEDBACK_SOURCE,
        "source_id": f"phase31-source-{index:03d}",
        "chunk_id": f"phase31-chunk-{index:03d}",
        "signal_type": signal_type,
        "user_text_excerpt": "用户在历史对话中要求真实核对、测试、截图、计数或文件证据。",
        "human_feedback_text": "用户在历史对话中要求真实核对、测试、截图、计数或文件证据。",
        "user_text_hash": f"hash-{index:03d}",
        "chosen": "当前理解：用户偏好是先做真实检查，再用路径、计数、测试或截图证据汇报。",
        "rejected": "应该没问题，先不用看文件或测试输出。",
        "what_the_user_was_trying_to_fix": "先做真实检查，再用路径、计数、测试或截图证据汇报。",
        "raw_excerpt_committed": False,
        "secret_risk_reasons": ["api_key"] if secret else [],
        "eligible_for_training": not secret,
        "attestation": {
            "historical_user_agent_conversation": True,
            "confirmed_actual_user_feedback": False,
            "not_realtime_actual_feedback": True,
            "requires_human_review_before_training": True,
        },
        "metadata": {
            "phase": "phase31",
            "not_actual_user_feedback": True,
        },
    }


def test_phase32_review_decision_schema_and_conservative_routing() -> None:
    signals = [_signal(1), _signal(2, secret=True)]

    batch = build_phase32_review_decisions(signals)
    decisions = batch["review_decisions"]

    assert decisions[0]["status"] == "approved_for_training"
    assert decisions[0]["reason"]
    assert decisions[0]["reviewer_id"]
    assert decisions[0]["validation"]["passed"] is True
    assert decisions[1]["status"] == "quarantined"
    assert "secret_risk_quarantine" in decisions[1]["reasons"]
    assert batch["review_summary"]["approved_for_training_count"] == 1


def test_phase32_taxonomy_covers_required_preferences() -> None:
    taxonomy = build_phase32_taxonomy()["taxonomy"]

    for key in (
        "execution_first",
        "evidence_first",
        "concise_status",
        "boundary_awareness",
        "persistence",
        "correction_responsiveness",
        "local_context_awareness",
    ):
        assert key in taxonomy
        assert taxonomy[key]["positive_behavior"]


def test_phase32_candidate_generation_has_no_raw_private_text_and_builds_all_candidate_types() -> None:
    signals = [_signal(index, signal_type="workflow_preference") for index in range(1, 15)]
    review = build_phase32_review_decisions(signals)
    holdout = build_phase32_holdout(count=PHASE32_MIN_HOLDOUT_PROMPTS)

    artifacts = build_phase32_candidate_artifacts(
        signals=signals,
        review_decisions=review["review_decisions"],
        holdout=holdout,
    )

    manifest = artifacts["candidate_manifest"]
    assert manifest["sft_sample_count"] == 14
    assert manifest["dpo_pair_count"] == 14
    assert manifest["hard_negative_pair_count"] == 14
    assert manifest["profile_candidate_count"] == 14
    assert manifest["memory_candidate_count"] == 14
    assert manifest["raw_private_text_committed"] is False
    assert artifacts["candidate_quality_report"]["passed"] is True
    assert artifacts["holdout_integrity_check"]["passed"] is True
    first = artifacts["sft_samples"][0]
    assert "signal_type:" in first["input"]
    assert "preference_summary:" in first["input"]
    assert "evidence_hash:" in first["input"]
    assert "Conversations/" not in str(first)
    assert "/Users/lichenhao" not in str(first)


def test_phase32_privacy_detector_blocks_local_paths_tokens_and_conversation_names() -> None:
    assert contains_raw_private_text("/Users/lichenhao/AgentMemory/Conversations/demo.md") is True
    assert contains_raw_private_text("token=123456789:abcdefghijklmnopqrstuvwxyz") is True
    assert contains_raw_private_text("Conversations/2026-06-22_14-01_Codex.md") is True
    assert contains_raw_private_text("preference_summary: 用户偏好先核对证据") is False


def test_phase32_holdout_has_required_size_and_categories() -> None:
    holdout = build_phase32_holdout(count=40)

    assert holdout["holdout_count"] >= 40
    categories = set(holdout["categories"])
    assert {
        "start_execution",
        "status_check",
        "correction",
        "submit_pr",
        "process_check",
        "next_goal",
        "dont_drift",
        "privacy_boundary",
    }.issubset(categories)
    assert all(item["not_for_training"] is True for item in holdout["prompts"])


def test_phase32_personalization_scoring_rewards_evidence_and_penalizes_leaks() -> None:
    prompt = {
        "category": "privacy_boundary",
        "expected_taxonomy": ["boundary_awareness", "local_context_awareness"],
        "prompt": "可以用我的历史记录，但不要把原始私密内容提交进去。",
    }
    good = "当前状态：我会只使用脱敏摘要和 evidence_hash。证据保留在本地，不提交原始私密文本；下一步跑隐私扫描。"
    bad = "我会把 /Users/lichenhao/AgentMemory/Conversations/2026-06-22_14-01_Codex.md 原文提交。"

    good_scores = score_phase32_output(good, prompt)
    bad_scores = score_phase32_output(bad, prompt)

    assert good_scores["boundary_awareness_rate"] == 1.0
    assert good_scores["raw_private_text_leak_rate"] == 0.0
    assert bad_scores["raw_private_text_leak_rate"] == 1.0
    assert good_scores["overall_personalization_score"] > bad_scores["overall_personalization_score"]


def test_phase32_decision_gate_promotes_only_when_adapter_beats_base() -> None:
    quality = {"passed": True}
    training = {"real_training": "completed"}
    base = {
        "status": "completed",
        "scores": {
            "overall_personalization_score": 0.5,
            "evidence_grounding_rate": 0.5,
            "boundary_awareness_rate": 0.5,
            "raw_private_text_leak_rate": 0.0,
            "hallucinated_completion_rate": 0.0,
            "execution_first_rate": 0.5,
        },
    }
    adapter = {
        "status": "completed",
        "scores": {
            "overall_personalization_score": 0.75,
            "evidence_grounding_rate": 0.6,
            "boundary_awareness_rate": 0.6,
            "raw_private_text_leak_rate": 0.0,
            "hallucinated_completion_rate": 0.0,
            "execution_first_rate": 0.7,
        },
    }

    promote = phase32_final_decision(
        candidate_quality_report=quality,
        training_attempt=training,
        base_eval=base,
        adapter_eval=adapter,
    )
    archive = phase32_final_decision(
        candidate_quality_report=quality,
        training_attempt=training,
        base_eval=adapter,
        adapter_eval=base,
    )

    assert promote["recommendation"] == "promote_after_manual_review"
    assert promote["auto_promotion_allowed"] is False
    assert archive["recommendation"] == "archive"
    assert "adapter_overall_not_above_base" in archive["reasons"]


def test_phase32_phase31_review_and_regression_boundary() -> None:
    review = build_phase32_phase31_review(
        phase31_summary={
            "status": "completed",
            "source_inventory": {"conversation_count": 1982, "selected_source_count": 80},
            "holdout": {"holdout_count": 12},
            "candidate_manifest": {
                "historical_conversation_signal_count": 68,
                "approved_candidate_signal_count": 39,
                "actual_user_feedback_count": 0,
            },
            "decision": {"recommendation": "historical_signal_quality_ready_for_human_review"},
        },
        phase31_decision="Historical AgentMemory conversations are reviewable signals, not realtime actual feedback.",
    )
    decision = phase31_final_decision(
        quality_report={"passed": True},
        candidate_manifest={"approved_candidate_signal_count": 39},
    )
    invalid = validate_phase32_review_decision({"status": "approved_for_training", "signal_id": "x"})

    assert review["phase31_completed"] is True
    assert review["actual_user_feedback_count"] == 0
    assert decision["training_launch_allowed"] is False
    assert invalid["passed"] is False
    assert "review_reason_required" in invalid["reasons"]
