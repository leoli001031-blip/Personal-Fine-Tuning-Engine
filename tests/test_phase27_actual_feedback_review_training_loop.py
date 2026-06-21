from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from pfe_core.phase27_actual_feedback_review_training_loop import (
    PHASE27_MIN_APPROVED_ACTUAL_CANDIDATES,
    append_phase27_import_batch,
    apply_phase27_review_decision,
    build_phase27_collection_pack,
    build_phase27_feedback_templates,
    build_phase27_import_batch,
    build_phase27_readiness,
    load_phase27_state,
    phase27_payloads_from_csv,
    phase27_payloads_from_jsonl,
    phase27_store_path,
    validate_phase27_feedback_payload,
)


def _actual_payload(item: dict, index: int) -> dict:
    return {
        "collection_id": item["collection_id"],
        "prompt": item["prompt"],
        "messages": item["messages"],
        "runtime_output": item["runtime_output"],
        "response_under_review": item["runtime_output"],
        "metadata": item["metadata"],
        "feedback_source": "actual_user_feedback",
        "feedback": {
            "action": "correction",
            "edited_text": item["suggested_target_template"],
            "user_feedback": "真实用户确认：这版边界更清楚，可进入候选审阅。",
            "signal_id": f"phase27-actual-signal-{index:03d}",
        },
        "attestation": {
            "operator_id": "human-reviewer-001",
            "capture_method": "phase27_collection_pack",
            "captured_at": "2026-06-21T10:00:00+08:00",
            "confirmed_actual_user_feedback": True,
            "not_scripted_or_curated": True,
            "consent_for_training_candidate_review": True,
        },
        "request_id": f"phase27-request-{index:03d}",
        "session_id": "phase27-actual-feedback-session",
    }


def _local_qwen_model() -> list[dict]:
    return [{"name": "Qwen3-4B", "path": "/models/qwen3-4b", "trainable": True}]


def test_phase27_jsonl_and_csv_import_schema_round_trip() -> None:
    pack = build_phase27_collection_pack()
    payload = _actual_payload(pack["items"][0], 1)
    jsonl = json.dumps(payload, ensure_ascii=False)

    parsed_jsonl = phase27_payloads_from_jsonl(jsonl)

    assert parsed_jsonl[0]["feedback"]["action"] == "correction"
    assert parsed_jsonl[0]["feedback_source"] == "actual_user_feedback"

    row = {
        "collection_id": payload["collection_id"],
        "prompt": payload["prompt"],
        "messages": json.dumps(payload["messages"], ensure_ascii=False),
        "runtime_output": payload["runtime_output"],
        "response_under_review": payload["response_under_review"],
        "metadata": json.dumps(payload["metadata"], ensure_ascii=False),
        "feedback_source": payload["feedback_source"],
        "feedback_action": "correction",
        "edited_text": payload["feedback"]["edited_text"],
        "user_feedback": payload["feedback"]["user_feedback"],
        "signal_id": payload["feedback"]["signal_id"],
        "attestation": json.dumps(payload["attestation"], ensure_ascii=False),
        "reviewer_decision": "pending_review",
        "reviewer_reason": "",
    }
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(row))
    writer.writeheader()
    writer.writerow(row)

    parsed_csv = phase27_payloads_from_csv(buffer.getvalue())

    assert parsed_csv[0]["feedback"]["edited_text"] == payload["feedback"]["edited_text"]
    assert parsed_csv[0]["attestation"]["confirmed_actual_user_feedback"] is True


def test_phase27_attestation_validation_blocks_missing_consent() -> None:
    pack = build_phase27_collection_pack()
    payload = _actual_payload(pack["items"][0], 1)
    payload["attestation"]["consent_for_training_candidate_review"] = False

    validation = validate_phase27_feedback_payload(payload)
    batch = build_phase27_import_batch([payload])

    assert validation["status"] == "blocked"
    assert "attestation_consent_for_training_candidate_review_must_be_true" in validation["errors"]
    assert batch["accepted_pending_review_count"] == 0
    assert batch["blocked_count"] == 1


def test_phase27_template_feedback_is_non_training_not_actual() -> None:
    pack = build_phase27_collection_pack()
    templates = build_phase27_feedback_templates(pack)
    template_payload = templates["jsonl_rows"][0]

    batch = build_phase27_import_batch([template_payload])

    assert batch["accepted_pending_review_count"] == 0
    assert batch["non_training_count"] == 1
    assert "template_feedback_not_training_data" in batch["non_training"][0]["reasons"]


def test_phase27_quarantines_pii_and_missing_citation() -> None:
    pack = build_phase27_collection_pack()
    payload = _actual_payload(pack["items"][0], 1)
    payload["metadata"]["expected_citation"] = ""
    payload["feedback"]["edited_text"] += "\n联系邮箱：reviewer@example.com"

    batch = build_phase27_import_batch([payload])

    assert batch["accepted_pending_review_count"] == 0
    assert batch["quarantined_count"] == 1
    assert "missing_required_citation" in batch["quarantined"][0]["reasons"]
    assert "pii_detected" in batch["quarantined"][0]["reasons"]


def test_phase27_review_decision_persistence_round_trips(tmp_path: Path) -> None:
    pack = build_phase27_collection_pack()
    batch = build_phase27_import_batch([_actual_payload(pack["items"][0], 1)])
    store = phase27_store_path(tmp_path, "phase27-test")

    append_phase27_import_batch(store, batch)
    result = apply_phase27_review_decision(
        store,
        {
            "signal_id": "phase27-actual-signal-001",
            "state": "approved_for_candidate",
            "reason": "passes four-section citation boundary",
            "reviewer_id": "reviewer-001",
        },
    )
    state = load_phase27_state(store)

    assert result["status"] == "completed"
    assert state["review_decisions"][0]["state"] == "approved_for_candidate"
    assert state["signals"][0]["signal_id"] == "phase27-actual-signal-001"


def test_phase27_pending_review_does_not_generate_training_candidates() -> None:
    pack = build_phase27_collection_pack()
    batch = build_phase27_import_batch([_actual_payload(pack["items"][0], 1)])

    readiness = build_phase27_readiness(
        signals=batch["accepted_signals"],
        review_decisions=[],
        local_models=_local_qwen_model(),
    )

    assert readiness["training_readiness"]["status"] == "collect_actual_feedback"
    assert readiness["candidate_artifacts"]["candidate_manifest"]["sft_sample_count"] == 0
    assert readiness["candidate_artifacts"]["candidate_manifest"]["dpo_pair_count"] == 0


def test_phase27_approved_threshold_generates_candidates_and_ready_gate() -> None:
    pack = build_phase27_collection_pack()
    payloads = [
        _actual_payload(item, index)
        for index, item in enumerate(pack["items"], start=1)
    ]
    batch = build_phase27_import_batch(payloads)
    decisions = [
        {
            "signal_id": signal["signal_id"],
            "state": "approved_for_candidate",
            "reason": "passes boundary contract",
            "reviewer_id": "reviewer-001",
        }
        for signal in batch["accepted_signals"]
    ]

    readiness = build_phase27_readiness(
        signals=batch["accepted_signals"],
        review_decisions=decisions,
        local_models=_local_qwen_model(),
    )

    assert batch["accepted_pending_review_count"] == PHASE27_MIN_APPROVED_ACTUAL_CANDIDATES
    assert readiness["review_state"]["approved_for_candidate_count"] == PHASE27_MIN_APPROVED_ACTUAL_CANDIDATES
    assert readiness["candidate_artifacts"]["candidate_manifest"]["sft_sample_count"] == PHASE27_MIN_APPROVED_ACTUAL_CANDIDATES
    assert readiness["candidate_artifacts"]["candidate_manifest"]["dpo_pair_count"] == PHASE27_MIN_APPROVED_ACTUAL_CANDIDATES
    assert readiness["holdout_integrity_check"]["passed"] is True
    assert readiness["training_readiness"]["status"] == "ready_for_real_training_probe"


def test_phase27_holdout_chunk_contamination_blocks_readiness() -> None:
    pack = build_phase27_collection_pack()
    payload = _actual_payload(pack["items"][0], 1)
    payload["metadata"]["chunk_id"] = "phase24-hard-holdout-chunk-001"
    payload["metadata"]["expected_citation"] = "[phase24-hard-holdout-source-001:phase24-hard-holdout-chunk-001]"
    payload["feedback"]["edited_text"] = payload["feedback"]["edited_text"].replace(
        "[phase26-collection-source-001:phase26-collection-chunk-001]",
        "[phase24-hard-holdout-source-001:phase24-hard-holdout-chunk-001]",
    )
    batch = build_phase27_import_batch([payload])
    decisions = [
        {
            "signal_id": "phase27-actual-signal-001",
            "state": "approved_for_candidate",
            "reason": "operator attempted approval",
        }
    ]

    readiness = build_phase27_readiness(
        signals=batch["accepted_signals"],
        review_decisions=decisions,
        local_models=_local_qwen_model(),
    )

    assert readiness["holdout_integrity_check"]["passed"] is True
    assert readiness["candidate_artifacts"]["candidate_manifest"]["sft_sample_count"] == 0
    assert readiness["candidate_artifacts"]["excluded"][0]["reason"] in {"holdout_contamination", "missing_required_citation"}
