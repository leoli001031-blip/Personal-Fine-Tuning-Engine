from __future__ import annotations

from pfe_core.phase24_real_signal_review_candidate_value import (
    build_phase24_candidate_artifacts,
    build_phase24_holdout,
    build_phase24_model_selection,
    evaluate_phase24_runtime_contract_holdout,
    phase24_holdout_integrity_check,
    phase24_runtime_product_decision,
)
from pfe_core.phase25_actual_user_feedback_loop import (
    PHASE25_MIN_APPROVED_ACTUAL_CANDIDATES,
    apply_phase25_review_decisions,
    build_phase25_actual_feedback_signal,
    build_phase25_empty_readiness,
    build_phase25_review_queue,
    build_phase25_routing_report,
    build_phase25_training_readiness_report,
    validate_phase25_actual_feedback_payload,
)


def _edited(citation: str, excerpt: str = "资料说明客户需在发票日后三十日内付款。") -> str:
    return (
        f"摘要：资料显示：{excerpt}\n"
        "风险提示：需核对资料完整性和附件位置；只做资料整理和风险提示，不判断合法/违法。\n"
        f"引用依据：{citation}\n"
        "人工确认：不输出法律结论，不能支持最终法律结论；需人工/法务结合完整材料确认。"
    )


def _payload(index: int = 1) -> dict:
    citation = f"[phase25-source-{index:03d}:phase25-chunk-{index:03d}]"
    excerpt = "资料说明客户需在发票日后三十日内付款。"
    return {
        "prompt": (
            "任务：请整理付款义务相关摘要、风险提示、引用依据和人工确认项。\n"
            f"资料引用：{citation}\n"
            f"资料摘录：{excerpt}\n"
            "只基于给定资料回答，不输出法律结论。"
        ),
        "metadata": {
            "response_contract": "contract_boundary_summary",
            "expected_citation": citation,
            "source_excerpt": excerpt,
        },
        "feedback_source": "actual_user_feedback",
        "feedback": {
            "action": "correction",
            "edited_text": _edited(citation, excerpt),
            "user_feedback": "真实用户修正：风险提示需要更明确。",
            "signal_id": f"phase25-actual-signal-{index:03d}",
        },
        "attestation": {
            "operator_id": "human-reviewer-001",
            "capture_method": "api_review_session",
            "captured_at": "2026-06-21T10:00:00+08:00",
            "confirmed_actual_user_feedback": True,
            "not_scripted_or_curated": True,
            "consent_for_training_candidate_review": True,
        },
        "request_id": f"phase25-request-{index:03d}",
        "session_id": "phase25-session-real-feedback",
    }


def test_phase25_actual_feedback_attestation_rejects_non_actual_sources() -> None:
    payload = _payload()
    payload["feedback_source"] = "curated_review_feedback"

    validation = validate_phase25_actual_feedback_payload(payload)
    intake = build_phase25_actual_feedback_signal(payload)

    assert validation["passed"] is False
    assert "feedback_source_must_be_actual_user_feedback" in validation["errors"]
    assert intake["status"] == "blocked"
    assert intake["auto_promotion_allowed"] is False


def test_phase25_actual_feedback_signal_preserves_actual_provenance_pending_review() -> None:
    intake = build_phase25_actual_feedback_signal(_payload())

    assert intake["status"] == "accepted_pending_review"
    signal = intake["signal"]
    assert signal["feedback_source"] == "actual_user_feedback"
    assert signal["feedback_source_is_actual_user_feedback"] is True
    assert signal["metadata"]["attestation"]["confirmed_actual_user_feedback"] is True
    assert intake["phase25_route"]["eligible_for_training"] is False
    assert intake["phase25_route"]["excluded_reason"] == "not_review_approved"


def test_phase25_review_approval_is_required_before_product_value_training() -> None:
    signals = [build_phase25_actual_feedback_signal(_payload(index))["signal"] for index in range(1, 3)]
    queue = build_phase25_review_queue(signals)
    pending = apply_phase25_review_decisions(queue, signals)
    approved = apply_phase25_review_decisions(queue, signals, approved_signal_ids={signals[0]["signal_id"]})
    pending_routing = build_phase25_routing_report(pending, signals)
    approved_routing = build_phase25_routing_report(approved, signals)

    assert pending["state_counts"]["pending_review"] == 2
    assert pending_routing["product_value_training_allowed_count"] == 0
    assert approved["state_counts"]["approved_for_candidate"] == 1
    assert approved_routing["product_value_training_allowed_count"] == 1


def test_phase25_training_readiness_blocks_empty_actual_feedback() -> None:
    readiness = build_phase25_empty_readiness(
        local_models=[{"name": "Qwen3-4B", "path": "/models/qwen3-4b", "trainable": True}]
    )

    report = readiness["training_readiness"]
    assert readiness["actual_feedback_count"] == 0
    assert report["status"] == "collect_actual_feedback"
    assert "insufficient_approved_actual_user_feedback" in report["blockers"]
    assert report["auto_promotion_allowed"] is False


def test_phase25_training_readiness_allows_probe_after_approved_actual_candidates() -> None:
    signals = [
        build_phase25_actual_feedback_signal(_payload(index))["signal"]
        for index in range(1, PHASE25_MIN_APPROVED_ACTUAL_CANDIDATES + 1)
    ]
    queue = build_phase25_review_queue(signals)
    reviewed = apply_phase25_review_decisions(queue, signals, approved_signal_ids={signal["signal_id"] for signal in signals})
    routing = build_phase25_routing_report(reviewed, signals)
    holdout = build_phase24_holdout(regression_count=50, hard_count=50)
    candidates = build_phase24_candidate_artifacts(
        signals=signals,
        reviewed=reviewed,
        routing_report=routing,
        holdout_chunk_ids={item["chunk_id"] for item in holdout["prompts"] if item.get("chunk_id")},
    )
    integrity = phase24_holdout_integrity_check(
        holdout=holdout,
        sft_samples=candidates["sft_samples"],
        dpo_pairs=candidates["dpo_pairs"],
    )
    runtime_eval = evaluate_phase24_runtime_contract_holdout(holdout)
    runtime_decision = phase24_runtime_product_decision(runtime_eval)
    model_selection = build_phase24_model_selection(
        local_models=[{"name": "Qwen3-4B", "path": "/models/qwen3-4b", "trainable": True}]
    )
    readiness = build_phase25_training_readiness_report(
        reviewed=reviewed,
        routing_report=routing,
        candidate_manifest=candidates["candidate_manifest"],
        candidate_quality_report=candidates["quality_report"],
        holdout_integrity=integrity,
        runtime_decision=runtime_decision,
        model_selection=model_selection,
    )

    assert routing["product_value_training_allowed_count"] == PHASE25_MIN_APPROVED_ACTUAL_CANDIDATES
    assert candidates["candidate_manifest"]["sft_sample_count"] == PHASE25_MIN_APPROVED_ACTUAL_CANDIDATES
    assert candidates["candidate_manifest"]["dpo_pair_count"] == PHASE25_MIN_APPROVED_ACTUAL_CANDIDATES
    assert readiness["status"] == "ready_for_real_training_probe"
    assert readiness["blockers"] == []
