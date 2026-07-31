from __future__ import annotations

from pfe_core.phase25_actual_user_feedback_loop import PHASE25_MIN_APPROVED_ACTUAL_CANDIDATES
from pfe_core.phase26_actual_feedback_collection_probe import (
    build_phase26_collection_pack,
    build_phase26_comparison_summary,
    build_phase26_empty_state,
    build_phase26_feedback_batch,
    build_phase26_probe_readiness,
)


def _payload_from_collection(item: dict, index: int) -> dict:
    return {
        "prompt": item["prompt"],
        "metadata": item["metadata"],
        "feedback_source": "actual_user_feedback",
        "feedback": {
            "action": "correction",
            "edited_text": item["suggested_target_template"],
            "user_feedback": "真实用户确认：请使用这个更明确的四段式。",
            "signal_id": f"phase26-actual-signal-{index:03d}",
        },
        "attestation": {
            "operator_id": "human-reviewer-001",
            "capture_method": "phase26_collection_pack",
            "captured_at": "2026-06-21T10:00:00+08:00",
            "confirmed_actual_user_feedback": True,
            "not_scripted_or_curated": True,
            "consent_for_training_candidate_review": True,
        },
        "request_id": f"phase26-request-{index:03d}",
        "session_id": "phase26-actual-feedback-session",
    }


def test_phase26_collection_pack_prepares_twelve_real_feedback_tasks() -> None:
    pack = build_phase26_collection_pack()

    assert pack["collection_count"] == PHASE25_MIN_APPROVED_ACTUAL_CANDIDATES
    assert pack["attestation_template"]["feedback_source"] == "actual_user_feedback"
    assert all(item["actual_feedback_required"] for item in pack["items"])
    assert all(item["not_training_data_until_attested_and_approved"] for item in pack["items"])
    assert pack["items"][0]["runtime_scores"]["structure_hit_rate"] == 1.0


def test_phase26_feedback_batch_rejects_unattested_or_curated_payloads() -> None:
    pack = build_phase26_collection_pack()
    bad = _payload_from_collection(pack["items"][0], 1)
    bad["feedback_source"] = "curated_review_feedback"

    batch = build_phase26_feedback_batch([bad])

    assert batch["payload_count"] == 1
    assert batch["accepted_pending_review_count"] == 0
    assert batch["blocked_count"] == 1
    assert "feedback_source_must_be_actual_user_feedback" in batch["blocked"][0]["errors"]


def test_phase26_feedback_batch_accepts_attested_actual_feedback_pending_review() -> None:
    pack = build_phase26_collection_pack()
    payload = _payload_from_collection(pack["items"][0], 1)

    batch = build_phase26_feedback_batch([payload])
    readiness = build_phase26_probe_readiness(
        signals=batch["accepted_signals"],
        local_models=[{"name": "Qwen3-4B", "path": "/models/qwen3-4b", "trainable": True}],
    )

    assert batch["accepted_pending_review_count"] == 1
    assert batch["accepted_signals"][0]["feedback_source"] == "actual_user_feedback"
    assert readiness["training_readiness"]["status"] == "collect_actual_feedback"
    assert readiness["routing_report"]["product_value_training_allowed_count"] == 0


def test_phase26_probe_readiness_opens_after_twelve_approved_actual_feedback_items() -> None:
    pack = build_phase26_collection_pack()
    payloads = [
        _payload_from_collection(item, index)
        for index, item in enumerate(pack["items"], start=1)
    ]
    batch = build_phase26_feedback_batch(payloads)
    approved = {signal["signal_id"] for signal in batch["accepted_signals"]}

    readiness = build_phase26_probe_readiness(
        signals=batch["accepted_signals"],
        approved_signal_ids=approved,
        local_models=[{"name": "Qwen3-4B", "path": "/models/qwen3-4b", "trainable": True}],
    )

    assert batch["accepted_pending_review_count"] == PHASE25_MIN_APPROVED_ACTUAL_CANDIDATES
    assert readiness["routing_report"]["product_value_training_allowed_count"] == PHASE25_MIN_APPROVED_ACTUAL_CANDIDATES
    assert readiness["candidate_artifacts"]["candidate_manifest"]["sft_sample_count"] == PHASE25_MIN_APPROVED_ACTUAL_CANDIDATES
    assert readiness["candidate_artifacts"]["candidate_manifest"]["dpo_pair_count"] == PHASE25_MIN_APPROVED_ACTUAL_CANDIDATES
    assert readiness["training_readiness"]["status"] == "ready_for_real_training_probe"
    assert readiness["training_job_specs"]["status"] == "ready"


def test_phase26_empty_state_summary_blocks_training_without_real_feedback() -> None:
    state = build_phase26_empty_state(
        local_models=[{"name": "Qwen3-4B", "path": "/models/qwen3-4b", "trainable": True}]
    )
    summary = build_phase26_comparison_summary(state)

    assert summary["collection_count"] == PHASE25_MIN_APPROVED_ACTUAL_CANDIDATES
    assert summary["actual_feedback_count"] == 0
    assert summary["training_readiness"]["status"] == "collect_actual_feedback"
    assert summary["final_recommendation"] == "collect_and_review_actual_user_feedback"
    assert summary["auto_promotion_allowed"] is False
