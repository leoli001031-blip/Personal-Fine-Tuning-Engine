"""Phase28 real-feedback loop-engineering primitives.

Phase28 keeps Phase27's actual-feedback gate intact and adds a durable loop
state for the next operator action. It treats templates and simulations as
workflow evidence only; they never count as actual feedback or training fuel.
"""

from __future__ import annotations

from collections import Counter
import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .phase27_actual_feedback_review_training_loop import (
    PHASE27_MIN_APPROVED_ACTUAL_CANDIDATES,
    PHASE27_REVIEW_STATES,
    build_phase27_collection_pack,
    build_phase27_import_batch,
    build_phase27_readiness,
    build_phase27_review_state,
    build_phase27_training_attempt,
    validate_phase27_feedback_payload,
)


PHASE28_KIND = "phase28_real_feedback_loop_engineering"
PHASE28_MIN_APPROVED_ACTUAL_CANDIDATES = PHASE27_MIN_APPROVED_ACTUAL_CANDIDATES
PHASE28_REVIEW_STATES = PHASE27_REVIEW_STATES
PHASE28_LOOP_STATES = {
    "observe",
    "ingest",
    "validate",
    "review",
    "build_candidates",
    "train_ready",
    "train_blocked",
    "eval_ready",
    "archive",
    "promote_after_manual_review",
}


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _json_loads(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    try:
        return json.loads(text)
    except Exception:
        return value


def _messages_from_payload(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = payload.get("messages")
    if isinstance(raw, str):
        raw = _json_loads(raw)
    if isinstance(raw, list):
        messages = [dict(item) for item in raw if isinstance(item, Mapping)]
        if messages:
            return messages
    prompt = str(payload.get("prompt") or payload.get("user_prompt") or payload.get("user_input") or "")
    return [{"role": "user", "content": prompt}]


def _row_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    metadata = _dict(_json_loads(row.get("metadata")))
    attestation = _dict(_json_loads(row.get("attestation")))
    feedback = _dict(_json_loads(row.get("feedback")))
    if not feedback:
        feedback = {
            "action": row.get("feedback_action") or row.get("action"),
            "edited_text": row.get("edited_text"),
            "user_feedback": row.get("user_feedback"),
            "signal_id": row.get("signal_id"),
        }
    messages = _json_loads(row.get("messages"))
    payload = {
        "task_id": row.get("task_id"),
        "collection_id": row.get("collection_id") or row.get("task_id"),
        "scenario_id": row.get("scenario_id"),
        "prompt": row.get("prompt") or row.get("user_prompt") or row.get("user_input"),
        "messages": messages if isinstance(messages, list) else [],
        "runtime_output": row.get("runtime_output") or row.get("base_output"),
        "response_under_review": row.get("response_under_review") or row.get("runtime_output") or row.get("base_output"),
        "metadata": metadata,
        "feedback_source": row.get("feedback_source") or metadata.get("feedback_source"),
        "feedback": feedback,
        "attestation": attestation,
        "request_id": row.get("request_id"),
        "session_id": row.get("session_id"),
        "simulation_only": row.get("simulation_only"),
        "template_not_training_data": row.get("template_not_training_data"),
        "not_valid_for_production_training": row.get("not_valid_for_production_training"),
    }
    return {key: value for key, value in payload.items() if value not in (None, "")}


def _truthy(value: Any) -> bool:
    if value is True:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return False


def _has_simulation_marker(value: Mapping[str, Any]) -> bool:
    feedback = _dict(value.get("feedback"))
    metadata = _dict(value.get("metadata"))
    attestation = _dict(value.get("attestation"))
    markers = (
        value.get("simulation_only"),
        value.get("not_valid_for_production_training"),
        metadata.get("simulation_only"),
        metadata.get("not_valid_for_production_training"),
        feedback.get("simulation_only"),
        attestation.get("simulation_only"),
    )
    if any(_truthy(marker) for marker in markers):
        return True
    text = " ".join(
        str(part)
        for part in (
            feedback.get("user_feedback"),
            feedback.get("edited_text"),
            metadata.get("simulation_policy"),
            value.get("request_id"),
            value.get("session_id"),
        )
    ).lower()
    return "simulation only" in text or "phase27-simulation" in text


def _has_template_marker(value: Mapping[str, Any]) -> bool:
    metadata = _dict(value.get("metadata"))
    return any(
        _truthy(marker)
        for marker in (
            value.get("template_not_training_data"),
            metadata.get("template_not_training_data"),
            metadata.get("task_not_training_data"),
        )
    )


def _source_metadata_from_phase27_item(item: Mapping[str, Any], task_id: str) -> dict[str, Any]:
    metadata = _dict(item.get("metadata"))
    return {
        "phase": "phase28",
        "scenario_id": "contract_risk_summary",
        "task_id": task_id,
        "source_id": metadata.get("source_id"),
        "chunk_id": metadata.get("chunk_id"),
        "source_excerpt": metadata.get("source_excerpt"),
        "expected_citation": metadata.get("expected_citation"),
        "response_contract": metadata.get("response_contract") or "contract_boundary_summary",
    }


def build_phase28_task_pack(*, count: int = 36) -> dict[str, Any]:
    source_pack = build_phase27_collection_pack()
    source_items = [dict(item) for item in source_pack.get("items") or [] if isinstance(item, Mapping)]
    tasks: list[dict[str, Any]] = []
    if not source_items:
        count = 0
    for index in range(1, count + 1):
        source = source_items[(index - 1) % len(source_items)]
        task_id = f"phase28-task-{index:03d}"
        source_metadata = _source_metadata_from_phase27_item(source, task_id)
        prompt = str(source.get("prompt") or "")
        tasks.append(
            {
                "task_id": task_id,
                "collection_id": source.get("collection_id"),
                "scenario_id": "contract_risk_summary",
                "source_id": source_metadata.get("source_id"),
                "chunk_id": source_metadata.get("chunk_id"),
                "source_excerpt": source_metadata.get("source_excerpt"),
                "user_prompt": prompt,
                "messages": source.get("messages") or [{"role": "user", "content": prompt}],
                "runtime_output": source.get("runtime_output"),
                "base_output": source.get("runtime_output"),
                "suggested_target_template": source.get("suggested_target_template"),
                "expected_citation_boundary": source_metadata.get("expected_citation"),
                "source_metadata": source_metadata,
                "reviewer_feedback_fields": {
                    "feedback_source": "actual_user_feedback",
                    "allowed_actions": ["accept", "reject", "edit", "correction", "preference", "safety_block"],
                    "required_attestation": {
                        "operator_id": "human-reviewer-id",
                        "captured_at": "ISO-8601 timestamp",
                        "confirmed_actual_user_feedback": True,
                        "not_scripted_or_curated": True,
                        "consent_for_training_candidate_review": True,
                    },
                    "required_review_decision": "pending_review",
                },
                "task_not_training_data": True,
                "template_not_training_data": True,
            }
        )
    return {
        "kind": "phase28_real_feedback_task_pack",
        "scenario_id": "contract_risk_summary",
        "task_count": len(tasks),
        "tasks": tasks,
        "template_not_training_data": True,
        "simulation_not_training_data": True,
        "actual_feedback_required": True,
        "created_at": _utcnow_iso(),
    }


def phase28_feedback_template_payload(task: Mapping[str, Any], index: int) -> dict[str, Any]:
    return {
        "task_id": task.get("task_id"),
        "collection_id": task.get("collection_id"),
        "scenario_id": task.get("scenario_id"),
        "prompt": task.get("user_prompt"),
        "messages": task.get("messages") or [],
        "runtime_output": task.get("runtime_output"),
        "response_under_review": task.get("runtime_output"),
        "metadata": {**_dict(task.get("source_metadata")), "template_not_training_data": True},
        "feedback_source": "template_feedback",
        "feedback": {
            "action": "correction",
            "edited_text": "",
            "user_feedback": "",
            "signal_id": f"phase28-template-signal-{index:03d}",
        },
        "attestation": {
            "operator_id": "",
            "capture_method": "phase28_template",
            "captured_at": "",
            "confirmed_actual_user_feedback": False,
            "not_scripted_or_curated": False,
            "consent_for_training_candidate_review": False,
        },
        "reviewer_decision": "pending_review",
        "reviewer_reason": "template row; not training data",
        "template_not_training_data": True,
    }


def build_phase28_feedback_templates(task_pack: Mapping[str, Any]) -> dict[str, Any]:
    rows = [
        phase28_feedback_template_payload(task, index)
        for index, task in enumerate(task_pack.get("tasks") or [], start=1)
        if isinstance(task, Mapping)
    ]
    csv_rows = []
    for row in rows:
        feedback = _dict(row.get("feedback"))
        csv_rows.append(
            {
                "task_id": row.get("task_id"),
                "collection_id": row.get("collection_id"),
                "scenario_id": row.get("scenario_id"),
                "prompt": row.get("prompt"),
                "messages": json.dumps(row.get("messages") or [], ensure_ascii=False),
                "runtime_output": row.get("runtime_output"),
                "response_under_review": row.get("response_under_review"),
                "metadata": json.dumps(row.get("metadata") or {}, ensure_ascii=False, sort_keys=True),
                "feedback_source": row.get("feedback_source"),
                "feedback_action": feedback.get("action"),
                "edited_text": feedback.get("edited_text"),
                "user_feedback": feedback.get("user_feedback"),
                "signal_id": feedback.get("signal_id"),
                "attestation": json.dumps(row.get("attestation") or {}, ensure_ascii=False, sort_keys=True),
                "reviewer_decision": row.get("reviewer_decision"),
                "reviewer_reason": row.get("reviewer_reason"),
            }
        )
    return {
        "kind": "phase28_feedback_templates",
        "jsonl_rows": rows,
        "csv_rows": csv_rows,
        "created_at": _utcnow_iso(),
    }


def phase28_payloads_from_jsonl(text: str) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, Mapping):
            payloads.append(_row_payload(payload))
    return payloads


def phase28_payloads_from_csv(text: str) -> list[dict[str, Any]]:
    reader = csv.DictReader(io.StringIO(text))
    return [_row_payload(row) for row in reader]


def validate_phase28_feedback_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    body = _row_payload(payload)
    phase27_validation = validate_phase27_feedback_payload(body)
    non_training_reasons = list(phase27_validation.get("non_training_reasons") or [])
    if _has_template_marker(body):
        non_training_reasons.append("phase28_template_not_training_data")
    if _has_simulation_marker(body):
        non_training_reasons.append("phase28_simulation_not_training_data")

    status = str(phase27_validation.get("status") or "blocked")
    if non_training_reasons:
        status = "non_training"

    return {
        "kind": "phase28_actual_feedback_validation",
        "passed": status == "passed",
        "status": status,
        "errors": list(phase27_validation.get("errors") or []),
        "non_training_reasons": sorted(set(non_training_reasons)),
        "quarantine_reasons": list(phase27_validation.get("quarantine_reasons") or []),
        "phase27_validation": phase27_validation,
        "created_at": _utcnow_iso(),
    }


def build_phase28_import_batch(payloads: list[Mapping[str, Any]]) -> dict[str, Any]:
    accepted_signals: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    quarantined: list[dict[str, Any]] = []
    non_training: list[dict[str, Any]] = []
    intakes: list[dict[str, Any]] = []

    for index, raw_payload in enumerate(payloads):
        payload = _row_payload(raw_payload)
        validation = validate_phase28_feedback_payload(payload)
        status = validation["status"]
        if status == "non_training":
            record = {
                "batch_index": index,
                "status": "non_training",
                "training_disposition": "non_training",
                "reasons": validation["non_training_reasons"],
                "validation": validation,
            }
            non_training.append(record)
            intakes.append(record)
            continue
        if status == "quarantined":
            record = {
                "batch_index": index,
                "status": "quarantined",
                "training_disposition": "quarantined",
                "reasons": validation["quarantine_reasons"],
                "validation": validation,
            }
            quarantined.append(record)
            intakes.append(record)
            continue
        if status == "blocked":
            record = {
                "batch_index": index,
                "status": "blocked",
                "training_disposition": "blocked",
                "reasons": validation["errors"] or ["invalid_phase28_actual_feedback"],
                "validation": validation,
            }
            blocked.append(record)
            intakes.append(record)
            continue

        phase27_batch = build_phase27_import_batch([payload])
        signals = [dict(item) for item in phase27_batch.get("accepted_signals") or [] if isinstance(item, Mapping)]
        if signals:
            signal = signals[0]
            signal.setdefault("metadata", {})["phase"] = "phase28"
            signal.setdefault("metadata", {})["phase28_imported_at"] = _utcnow_iso()
            signal.setdefault("metadata", {})["task_id"] = payload.get("task_id")
            signal["phase28_review_state"] = "pending_review"
            signal["eligible_for_training"] = False
            accepted_signals.append(signal)
            record = {
                "batch_index": index,
                "status": "accepted_pending_review",
                "validation": validation,
                "signal": signal,
                "auto_promotion_allowed": False,
            }
            intakes.append(record)
            continue
        record = {
            "batch_index": index,
            "status": "blocked",
            "training_disposition": "blocked",
            "reasons": ["phase27_intake_blocked"],
            "validation": validation,
        }
        blocked.append(record)
        intakes.append(record)

    return {
        "kind": "phase28_actual_feedback_import_batch",
        "payload_count": len(payloads),
        "accepted_pending_review_count": len(accepted_signals),
        "blocked_count": len(blocked),
        "quarantined_count": len(quarantined),
        "non_training_count": len(non_training),
        "intakes": intakes,
        "accepted_signals": accepted_signals,
        "blocked": blocked,
        "quarantined": quarantined,
        "non_training": non_training,
        "auto_promotion_allowed": False,
        "created_at": _utcnow_iso(),
    }


def phase28_store_path(root: Path, workspace: str = "user_default") -> Path:
    return Path(root) / "data" / f"phase28_real_feedback_{workspace}.json"


def load_phase28_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "kind": "phase28_persisted_state",
            "signals": [],
            "review_decisions": [],
            "reviewer_audit_log": [],
            "import_batches": [],
            "loop_runs": [],
            "created_at": _utcnow_iso(),
            "updated_at": _utcnow_iso(),
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        payload = {}
    state = dict(payload) if isinstance(payload, Mapping) else {}
    state.setdefault("kind", "phase28_persisted_state")
    state.setdefault("signals", [])
    state.setdefault("review_decisions", [])
    state.setdefault("reviewer_audit_log", [])
    state.setdefault("import_batches", [])
    state.setdefault("loop_runs", [])
    state.setdefault("created_at", _utcnow_iso())
    state["updated_at"] = state.get("updated_at") or _utcnow_iso()
    return state


def save_phase28_state(path: Path, state: Mapping[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(state)
    payload["updated_at"] = _utcnow_iso()
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def append_phase28_import_batch(path: Path, batch: Mapping[str, Any]) -> dict[str, Any]:
    state = load_phase28_state(path)
    existing = {str(signal.get("signal_id")) for signal in state.get("signals") or [] if isinstance(signal, Mapping)}
    new_signals = [
        dict(signal)
        for signal in batch.get("accepted_signals") or []
        if isinstance(signal, Mapping) and str(signal.get("signal_id")) not in existing
    ]
    state["signals"] = [dict(signal) for signal in state.get("signals") or [] if isinstance(signal, Mapping)] + new_signals
    state["import_batches"] = [dict(item) for item in state.get("import_batches") or [] if isinstance(item, Mapping)] + [
        {
            "kind": batch.get("kind"),
            "payload_count": batch.get("payload_count", 0),
            "accepted_pending_review_count": batch.get("accepted_pending_review_count", 0),
            "blocked_count": batch.get("blocked_count", 0),
            "quarantined_count": batch.get("quarantined_count", 0),
            "non_training_count": batch.get("non_training_count", 0),
            "created_at": batch.get("created_at") or _utcnow_iso(),
        }
    ]
    return save_phase28_state(path, state)


def apply_phase28_review_decision(path: Path, decision: Mapping[str, Any]) -> dict[str, Any]:
    state = load_phase28_state(path)
    signal_ids = decision.get("signal_ids") or decision.get("signal_id") or []
    if isinstance(signal_ids, str):
        signal_ids = [signal_ids]
    review_state = str(decision.get("state") or decision.get("review_state") or "")
    reason = str(decision.get("reason") or "").strip()
    if review_state not in PHASE28_REVIEW_STATES:
        return {
            "kind": "phase28_review_decision_result",
            "status": "blocked",
            "reason": "unsupported_review_state",
            "allowed_states": sorted(PHASE28_REVIEW_STATES),
            "auto_promotion_allowed": False,
        }
    if not reason:
        return {
            "kind": "phase28_review_decision_result",
            "status": "blocked",
            "reason": "review_reason_required",
            "auto_promotion_allowed": False,
        }
    known_signal_ids = {str(signal.get("signal_id")) for signal in state.get("signals") or [] if isinstance(signal, Mapping)}
    applied: list[dict[str, Any]] = []
    decisions = [dict(item) for item in state.get("review_decisions") or [] if isinstance(item, Mapping)]
    audit_log = [dict(item) for item in state.get("reviewer_audit_log") or [] if isinstance(item, Mapping)]
    for signal_id in signal_ids:
        signal_id = str(signal_id)
        if signal_id not in known_signal_ids:
            applied.append({"signal_id": signal_id, "status": "blocked", "reason": "signal_not_found"})
            continue
        record = {
            "signal_id": signal_id,
            "state": review_state,
            "reason": reason,
            "reviewer_id": decision.get("reviewer_id") or "manual-reviewer",
            "decided_at": _utcnow_iso(),
        }
        decisions = [item for item in decisions if str(item.get("signal_id")) != signal_id]
        decisions.append(record)
        audit_log.append({"kind": "phase28_reviewer_audit_event", **record})
        applied.append({"signal_id": signal_id, "status": "applied", "state": review_state})
    state["review_decisions"] = decisions
    state["reviewer_audit_log"] = audit_log
    saved = save_phase28_state(path, state)
    return {
        "kind": "phase28_review_decision_result",
        "status": "completed" if any(item["status"] == "applied" for item in applied) else "blocked",
        "applied": applied,
        "state": saved,
        "auto_promotion_allowed": False,
        "created_at": _utcnow_iso(),
    }


def build_phase28_review_state(
    *,
    signals: list[Mapping[str, Any]],
    review_decisions: list[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    review_state = build_phase27_review_state(signals=signals, review_decisions=review_decisions or [])
    return {
        **review_state,
        "kind": "phase28_review_state",
    }


def build_phase28_readiness(
    *,
    signals: list[Mapping[str, Any]],
    review_decisions: list[Mapping[str, Any]] | None = None,
    local_models: list[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    readiness = build_phase27_readiness(
        signals=signals,
        review_decisions=review_decisions or [],
        local_models=local_models or [],
    )
    return {
        **readiness,
        "kind": "phase28_training_readiness",
    }


def build_phase28_training_attempt(readiness_payload: Mapping[str, Any]) -> dict[str, Any]:
    attempt = build_phase27_training_attempt(readiness_payload)
    return {
        **attempt,
        "kind": "phase28_training_attempt",
        "real_training_executed": False,
        "optional_real_probe_requires_explicit_operator_action": True,
        "auto_promotion_allowed": False,
    }


def build_phase28_loop_state(
    *,
    readiness_payload: Mapping[str, Any],
    training_attempt: Mapping[str, Any],
    evidence_path: str,
    import_batch: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    readiness = _dict(readiness_payload.get("training_readiness"))
    review_state = _dict(readiness_payload.get("review_state"))
    reviewed = _dict(review_state.get("reviewed"))
    state_counts = _dict(reviewed.get("state_counts"))
    actual_feedback_count = int(readiness_payload.get("actual_feedback_count", 0) or 0)
    approved_count = int(readiness.get("approved_actual_candidate_count", 0) or 0)
    pending_count = int(review_state.get("pending_review_count", 0) or state_counts.get("pending_review", 0) or 0)
    blockers = list(readiness.get("blockers") or [])
    batch = _dict(import_batch)

    current_state = "observe"
    next_action = "collect_attested_actual_feedback"
    required_human_action = "fill_phase28_feedback_batch_from_real_user_interactions"
    if int(batch.get("payload_count", 0) or 0) > 0 and actual_feedback_count == 0:
        current_state = "validate"
        next_action = "fix_or_remove_invalid_feedback_rows"
        required_human_action = "supply_attested_non_template_non_simulation_feedback"
    elif pending_count > 0:
        current_state = "review"
        next_action = "approve_exclude_or_quarantine_pending_feedback"
        required_human_action = "review_pending_actual_feedback"
    elif training_attempt.get("status") == "ready_to_launch":
        current_state = "train_ready"
        next_action = "operator_may_launch_qwen_probe"
        required_human_action = "explicitly_launch_real_training_probe_or_keep_collecting_feedback"
    elif approved_count > 0:
        current_state = "train_blocked"
        next_action = "collect_more_approved_actual_feedback"
        required_human_action = "approve_at_least_12_quality_actual_feedback_items"
    elif actual_feedback_count > 0:
        current_state = "build_candidates"
        next_action = "review_candidate_quality_and_continue_approval"
        required_human_action = "approve_more_feedback_or_exclude_low_quality_rows"

    return {
        "kind": "phase28_loop_state",
        "current_state": current_state,
        "allowed_states": sorted(PHASE28_LOOP_STATES),
        "evidence_path": evidence_path,
        "blockers": blockers,
        "next_action": next_action,
        "required_human_action": required_human_action,
        "auto_action_allowed": False,
        "auto_promotion_allowed": False,
        "actual_feedback_count": actual_feedback_count,
        "approved_actual_candidate_count": approved_count,
        "pending_review_count": pending_count,
        "training_attempt_status": training_attempt.get("status"),
        "created_at": _utcnow_iso(),
    }


def build_phase28_simulation_review(simulation_dir: Path) -> dict[str, Any]:
    summary_path = Path(simulation_dir) / "simulation_summary.json"
    guardrail_path = Path(simulation_dir) / "guardrail_replay_batch.json"
    if not summary_path.exists():
        return {
            "kind": "phase28_phase27_simulation_review",
            "status": "missing",
            "simulation_dir": str(simulation_dir),
            "conclusion": "phase27 simulation evidence not found",
            "simulation_rows_allowed_for_training": False,
            "created_at": _utcnow_iso(),
        }
    try:
        summary = _dict(json.loads(summary_path.read_text(encoding="utf-8")))
    except Exception:
        summary = {}
    try:
        guardrail = _dict(json.loads(guardrail_path.read_text(encoding="utf-8"))) if guardrail_path.exists() else {}
    except Exception:
        guardrail = {}
    return {
        "kind": "phase28_phase27_simulation_review",
        "status": "reviewed",
        "simulation_dir": str(simulation_dir),
        "simulation_only": summary.get("simulation_only") is True,
        "not_valid_for_production_training": summary.get("not_valid_for_production_training") is True,
        "phase27_simulated_readiness_status": summary.get("readiness_status"),
        "phase27_simulated_training_attempt_status": summary.get("training_attempt_status"),
        "guardrail_counts": {
            "blocked_count": guardrail.get("blocked_count"),
            "non_training_count": guardrail.get("non_training_count"),
            "quarantined_count": guardrail.get("quarantined_count"),
        },
        "simulation_rows_allowed_for_training": False,
        "conclusion": "simulation proves workflow shape only and must remain excluded from Phase28 actual feedback",
        "created_at": _utcnow_iso(),
    }


def build_phase28_comparison_summary(
    *,
    task_pack: Mapping[str, Any],
    import_batch: Mapping[str, Any],
    readiness_payload: Mapping[str, Any],
    training_attempt: Mapping[str, Any],
    loop_state: Mapping[str, Any],
    simulation_review: Mapping[str, Any],
) -> dict[str, Any]:
    readiness = _dict(readiness_payload.get("training_readiness"))
    final = (
        "ready_for_real_training_probe"
        if training_attempt.get("status") == "ready_to_launch"
        else "collect_more_actual_feedback"
    )
    return {
        "kind": "phase28_comparison_summary",
        "status": "completed",
        "task_count": task_pack.get("task_count", 0),
        "actual_feedback_count": readiness_payload.get("actual_feedback_count", 0),
        "accepted_pending_review_count": import_batch.get("accepted_pending_review_count", 0),
        "approved_actual_candidate_count": readiness.get("approved_actual_candidate_count", 0),
        "training_readiness": readiness,
        "training_attempt": training_attempt,
        "loop_state": loop_state,
        "phase27_simulation_review": simulation_review,
        "final_recommendation": final,
        "auto_promotion_allowed": False,
        "created_at": _utcnow_iso(),
    }


__all__ = [
    "PHASE28_KIND",
    "PHASE28_LOOP_STATES",
    "PHASE28_MIN_APPROVED_ACTUAL_CANDIDATES",
    "PHASE28_REVIEW_STATES",
    "append_phase28_import_batch",
    "apply_phase28_review_decision",
    "build_phase28_comparison_summary",
    "build_phase28_feedback_templates",
    "build_phase28_import_batch",
    "build_phase28_loop_state",
    "build_phase28_readiness",
    "build_phase28_review_state",
    "build_phase28_simulation_review",
    "build_phase28_task_pack",
    "build_phase28_training_attempt",
    "load_phase28_state",
    "phase28_payloads_from_csv",
    "phase28_payloads_from_jsonl",
    "phase28_store_path",
    "save_phase28_state",
    "validate_phase28_feedback_payload",
]
