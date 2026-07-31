from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from pfe_core.phase28_real_feedback_loop_engineering import (
    PHASE28_MIN_APPROVED_ACTUAL_CANDIDATES,
    append_phase28_import_batch,
    apply_phase28_review_decision,
    build_phase28_import_batch,
    build_phase28_loop_state,
    build_phase28_readiness,
    build_phase28_review_state,
    build_phase28_task_pack,
    build_phase28_training_attempt,
    load_phase28_state,
    phase28_payloads_from_csv,
    phase28_payloads_from_jsonl,
    phase28_store_path,
    validate_phase28_feedback_payload,
)


def _actual_payload(task: dict, index: int) -> dict:
    return {
        "task_id": task["task_id"],
        "collection_id": task["collection_id"],
        "scenario_id": task["scenario_id"],
        "prompt": task["user_prompt"],
        "messages": task["messages"],
        "runtime_output": task["runtime_output"],
        "response_under_review": task["runtime_output"],
        "metadata": task["source_metadata"],
        "feedback_source": "actual_user_feedback",
        "feedback": {
            "action": "correction",
            "edited_text": task["suggested_target_template"],
            "user_feedback": "真实用户确认：这版四段式边界可进入候选审阅。",
            "signal_id": f"phase28-actual-signal-{index:03d}",
        },
        "attestation": {
            "operator_id": "human-reviewer-001",
            "capture_method": "phase28_collection_pack",
            "captured_at": "2026-06-21T10:00:00+08:00",
            "confirmed_actual_user_feedback": True,
            "not_scripted_or_curated": True,
            "consent_for_training_candidate_review": True,
        },
        "request_id": f"phase28-request-{index:03d}",
        "session_id": "phase28-actual-feedback-session",
    }


def _local_qwen_model() -> list[dict]:
    return [{"name": "Qwen3-4B", "path": "/models/qwen3-4b", "trainable": True}]


def test_phase28_task_pack_schema_marks_tasks_non_training() -> None:
    pack = build_phase28_task_pack(count=36)

    assert pack["task_count"] == 36
    assert pack["template_not_training_data"] is True
    first = pack["tasks"][0]
    assert first["task_id"] == "phase28-task-001"
    assert first["scenario_id"] == "contract_risk_summary"
    assert first["source_id"]
    assert first["chunk_id"]
    assert first["source_excerpt"]
    assert first["user_prompt"]
    assert first["runtime_output"]
    assert first["suggested_target_template"]
    assert first["expected_citation_boundary"]
    assert first["template_not_training_data"] is True
    assert "source_metadata" in first
    assert first["source_metadata"]["expected_citation"] == first["expected_citation_boundary"]


def test_phase28_jsonl_and_csv_import_schema_round_trip() -> None:
    task = build_phase28_task_pack(count=1)["tasks"][0]
    payload = _actual_payload(task, 1)
    parsed_jsonl = phase28_payloads_from_jsonl(json.dumps(payload, ensure_ascii=False))

    assert parsed_jsonl[0]["feedback_source"] == "actual_user_feedback"
    assert parsed_jsonl[0]["metadata"]["source_id"] == task["source_id"]

    row = {
        "task_id": payload["task_id"],
        "collection_id": payload["collection_id"],
        "scenario_id": payload["scenario_id"],
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
    writer = csv.DictWriter(buffer, fieldnames=list(row), lineterminator="\n")
    writer.writeheader()
    writer.writerow(row)

    parsed_csv = phase28_payloads_from_csv(buffer.getvalue())

    assert parsed_csv[0]["feedback"]["edited_text"] == payload["feedback"]["edited_text"]
    assert parsed_csv[0]["attestation"]["confirmed_actual_user_feedback"] is True


def test_phase28_attestation_validation_blocks_missing_consent_and_operator() -> None:
    task = build_phase28_task_pack(count=1)["tasks"][0]
    payload = _actual_payload(task, 1)
    payload["attestation"]["operator_id"] = ""
    payload["attestation"]["consent_for_training_candidate_review"] = False

    validation = validate_phase28_feedback_payload(payload)
    batch = build_phase28_import_batch([payload])

    assert validation["status"] == "blocked"
    assert "attestation_operator_id_required" in validation["errors"]
    assert "attestation_consent_for_training_candidate_review_must_be_true" in validation["errors"]
    assert batch["blocked_count"] == 1
    assert batch["accepted_pending_review_count"] == 0


def test_phase28_excludes_simulation_and_template_payloads_from_training() -> None:
    task = build_phase28_task_pack(count=1)["tasks"][0]
    simulation_payload = _actual_payload(task, 1)
    simulation_payload["simulation_only"] = True
    simulation_payload["metadata"]["simulation_only"] = True
    template_payload = _actual_payload(task, 2)
    template_payload["feedback_source"] = "template_feedback"
    template_payload["metadata"]["template_not_training_data"] = True

    batch = build_phase28_import_batch([simulation_payload, template_payload])

    assert batch["accepted_pending_review_count"] == 0
    assert batch["non_training_count"] == 2
    reasons = {reason for item in batch["non_training"] for reason in item["reasons"]}
    assert "phase28_simulation_not_training_data" in reasons
    assert "phase28_template_not_training_data" in reasons


def test_phase28_quarantines_pii_missing_citation_and_legal_conclusion() -> None:
    task = build_phase28_task_pack(count=1)["tasks"][0]
    payload = _actual_payload(task, 1)
    payload["metadata"]["expected_citation"] = ""
    payload["feedback"]["edited_text"] += "\n合法有效，可直接签。联系 reviewer@example.com"

    batch = build_phase28_import_batch([payload])

    assert batch["accepted_pending_review_count"] == 0
    assert batch["quarantined_count"] == 1
    assert "missing_required_citation" in batch["quarantined"][0]["reasons"]
    assert "pii_detected" in batch["quarantined"][0]["reasons"]
    assert "legal_conclusion_or_direct_sign_target" in batch["quarantined"][0]["reasons"]


def test_phase28_review_persistence_and_audit_log_round_trip(tmp_path: Path) -> None:
    task = build_phase28_task_pack(count=1)["tasks"][0]
    batch = build_phase28_import_batch([_actual_payload(task, 1)])
    store = phase28_store_path(tmp_path, "phase28-test")

    append_phase28_import_batch(store, batch)
    result = apply_phase28_review_decision(
        store,
        {
            "signal_id": "phase28-actual-signal-001",
            "state": "approved_for_candidate",
            "reason": "passes four-section citation boundary",
            "reviewer_id": "reviewer-001",
        },
    )
    state = load_phase28_state(store)

    assert result["status"] == "completed"
    assert state["review_decisions"][0]["state"] == "approved_for_candidate"
    assert state["reviewer_audit_log"][0]["reason"] == "passes four-section citation boundary"


def test_phase28_candidate_threshold_and_ready_gate() -> None:
    pack = build_phase28_task_pack(count=PHASE28_MIN_APPROVED_ACTUAL_CANDIDATES)
    payloads = [_actual_payload(task, index) for index, task in enumerate(pack["tasks"], start=1)]
    batch = build_phase28_import_batch(payloads)
    decisions = [
        {
            "signal_id": signal["signal_id"],
            "state": "approved_for_candidate",
            "reason": "passes boundary contract",
            "reviewer_id": "reviewer-001",
        }
        for signal in batch["accepted_signals"]
    ]

    readiness = build_phase28_readiness(
        signals=batch["accepted_signals"],
        review_decisions=decisions,
        local_models=_local_qwen_model(),
    )
    attempt = build_phase28_training_attempt(readiness)
    loop_state = build_phase28_loop_state(
        readiness_payload=readiness,
        training_attempt=attempt,
        evidence_path="test",
        import_batch=batch,
    )

    assert readiness["training_readiness"]["status"] == "ready_for_real_training_probe"
    assert readiness["candidate_artifacts"]["candidate_manifest"]["sft_sample_count"] == PHASE28_MIN_APPROVED_ACTUAL_CANDIDATES
    assert readiness["candidate_artifacts"]["candidate_manifest"]["dpo_pair_count"] == PHASE28_MIN_APPROVED_ACTUAL_CANDIDATES
    assert attempt["status"] == "ready_to_launch"
    assert loop_state["current_state"] == "train_ready"
    assert loop_state["auto_action_allowed"] is False


def test_phase28_under_threshold_stays_train_blocked_after_review() -> None:
    task = build_phase28_task_pack(count=1)["tasks"][0]
    batch = build_phase28_import_batch([_actual_payload(task, 1)])
    decisions = [
        {
            "signal_id": "phase28-actual-signal-001",
            "state": "approved_for_candidate",
            "reason": "passes boundary contract",
        }
    ]
    readiness = build_phase28_readiness(
        signals=batch["accepted_signals"],
        review_decisions=decisions,
        local_models=_local_qwen_model(),
    )
    attempt = build_phase28_training_attempt(readiness)
    loop_state = build_phase28_loop_state(
        readiness_payload=readiness,
        training_attempt=attempt,
        evidence_path="test",
        import_batch=batch,
    )

    assert readiness["training_readiness"]["status"] == "collect_actual_feedback"
    assert "insufficient_approved_actual_user_feedback" in readiness["training_readiness"]["blockers"]
    assert attempt["status"] == "blocked"
    assert loop_state["current_state"] == "train_blocked"


def test_phase28_pending_review_loop_state() -> None:
    task = build_phase28_task_pack(count=1)["tasks"][0]
    batch = build_phase28_import_batch([_actual_payload(task, 1)])
    readiness = build_phase28_readiness(
        signals=batch["accepted_signals"],
        review_decisions=[],
        local_models=_local_qwen_model(),
    )
    attempt = build_phase28_training_attempt(readiness)
    loop_state = build_phase28_loop_state(
        readiness_payload=readiness,
        training_attempt=attempt,
        evidence_path="test",
        import_batch=batch,
    )

    assert loop_state["current_state"] == "review"
    assert loop_state["required_human_action"] == "review_pending_actual_feedback"


def test_phase28_holdout_chunk_contamination_excludes_candidate() -> None:
    task = build_phase28_task_pack(count=1)["tasks"][0]
    payload = _actual_payload(task, 1)
    payload["metadata"]["chunk_id"] = "phase24-hard-holdout-chunk-001"
    payload["metadata"]["expected_citation"] = "[phase24-hard-holdout-source-001:phase24-hard-holdout-chunk-001]"
    payload["feedback"]["edited_text"] = payload["feedback"]["edited_text"].replace(
        task["expected_citation_boundary"],
        "[phase24-hard-holdout-source-001:phase24-hard-holdout-chunk-001]",
    )
    batch = build_phase28_import_batch([payload])
    decisions = [
        {
            "signal_id": "phase28-actual-signal-001",
            "state": "approved_for_candidate",
            "reason": "operator attempted approval",
        }
    ]

    readiness = build_phase28_readiness(
        signals=batch["accepted_signals"],
        review_decisions=decisions,
        local_models=_local_qwen_model(),
    )

    assert readiness["holdout_integrity_check"]["passed"] is True
    assert readiness["candidate_artifacts"]["candidate_manifest"]["sft_sample_count"] == 0
    assert readiness["candidate_artifacts"]["excluded"][0]["reason"] in {
        "holdout_contamination",
        "missing_required_citation",
        "prompt_or_output_copy_noise",
    }
