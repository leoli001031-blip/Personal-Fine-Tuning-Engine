"""Phase27 actual-feedback review and training-loop primitives.

Phase27 turns the Phase26 collection gate into a resumable workflow:
import attested actual feedback, persist manual review decisions, generate
candidates only after approval, and keep training blocked until the evidence is
strong enough.
"""

from __future__ import annotations

from collections import Counter
import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

from .phase24_real_signal_review_candidate_value import (
    build_phase24_candidate_artifacts,
    build_phase24_holdout,
    build_phase24_model_selection,
    evaluate_phase24_runtime_contract_holdout,
    phase24_holdout_integrity_check,
    phase24_runtime_product_decision,
)
from .phase25_actual_user_feedback_loop import (
    PHASE25_MIN_APPROVED_ACTUAL_CANDIDATES,
    apply_phase25_review_decisions,
    build_phase25_actual_feedback_signal,
    build_phase25_review_queue,
    build_phase25_routing_report,
    build_phase25_training_job_specs,
    build_phase25_training_readiness_report,
    validate_phase25_actual_feedback_payload,
)
from .phase26_actual_feedback_collection_probe import build_phase26_collection_pack


PHASE27_KIND = "phase27_actual_feedback_review_training_loop"
PHASE27_MIN_APPROVED_ACTUAL_CANDIDATES = PHASE25_MIN_APPROVED_ACTUAL_CANDIDATES
PHASE27_REVIEW_STATES = {"pending_review", "approved_for_candidate", "excluded", "quarantined"}

_NON_TRAINING_FEEDBACK_SOURCES = {
    "curated_review_feedback",
    "scripted_feedback",
    "sample_feedback",
    "template_feedback",
    "synthetic_feedback",
}
_PHONE_PATTERN = re.compile(r"(?<!\d)(?:\+?\d[\d -]{7,}\d)(?!\d)")
_EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")
_EXTERNAL_LAW_PATTERN = re.compile(r"(民法典|司法解释|判例|案例|法条|court|statute|regulation)", re.IGNORECASE)
_LEGAL_CONCLUSION_PATTERN = re.compile(r"(合法有效|一定合法|一定违法|可以直接签|建议直接签署|无需人工确认)")


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
    prompt = str(payload.get("prompt") or payload.get("user_input") or "")
    return [{"role": "user", "content": prompt}]


def _payload_text_bundle(payload: Mapping[str, Any]) -> str:
    feedback = _dict(payload.get("feedback"))
    metadata = _dict(payload.get("metadata"))
    messages = _messages_from_payload(payload)
    return "\n".join(
        [
            str(payload.get("prompt") or payload.get("user_input") or ""),
            "\n".join(str(message.get("content") or "") for message in messages),
            str(payload.get("runtime_output") or payload.get("response_under_review") or ""),
            str(feedback.get("edited_text") or payload.get("edited_text") or ""),
            str(feedback.get("user_feedback") or payload.get("user_feedback") or ""),
            str(metadata.get("source_excerpt") or ""),
            str(metadata.get("expected_citation") or ""),
        ]
    )


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
        "collection_id": row.get("collection_id"),
        "prompt": row.get("prompt") or row.get("user_input"),
        "messages": messages if isinstance(messages, list) else [],
        "runtime_output": row.get("runtime_output"),
        "response_under_review": row.get("response_under_review"),
        "metadata": metadata,
        "feedback_source": row.get("feedback_source") or metadata.get("feedback_source"),
        "feedback": feedback,
        "attestation": attestation,
        "request_id": row.get("request_id"),
        "session_id": row.get("session_id"),
    }
    return {key: value for key, value in payload.items() if value not in (None, "")}


def phase27_payloads_from_jsonl(text: str) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, Mapping):
            payloads.append(_row_payload(payload))
    return payloads


def phase27_payloads_from_csv(text: str) -> list[dict[str, Any]]:
    reader = csv.DictReader(io.StringIO(text))
    return [_row_payload(row) for row in reader]


def build_phase27_collection_pack() -> dict[str, Any]:
    pack = build_phase26_collection_pack()
    items = []
    for item in pack.get("items") or []:
        if not isinstance(item, Mapping):
            continue
        copied = dict(item)
        metadata = _dict(copied.get("metadata"))
        metadata["phase"] = "phase27"
        copied["metadata"] = metadata
        copied["template_not_training_data"] = True
        copied["phase27_collection_policy"] = (
            "collection item is a prompt for gathering actual user feedback; "
            "it is not training data until attested and approved"
        )
        items.append(copied)
    return {
        **pack,
        "kind": "phase27_actual_feedback_collection_pack",
        "items": items,
        "collection_count": len(items),
        "template_not_training_data": True,
        "created_at": _utcnow_iso(),
    }


def phase27_feedback_template_payload(item: Mapping[str, Any], index: int) -> dict[str, Any]:
    metadata = _dict(item.get("metadata"))
    return {
        "collection_id": item.get("collection_id"),
        "prompt": item.get("prompt"),
        "messages": item.get("messages") or [],
        "runtime_output": item.get("runtime_output"),
        "response_under_review": item.get("runtime_output"),
        "metadata": {
            **metadata,
            "feedback_source": "template_feedback",
            "template_not_training_data": True,
            "source_policy": "example row only; replace with real user feedback before import",
        },
        "feedback_source": "template_feedback",
        "feedback": {
            "action": "correction",
            "edited_text": "",
            "user_feedback": "",
            "signal_id": f"phase27-template-signal-{index:03d}",
        },
        "attestation": {
            "operator_id": "",
            "capture_method": "phase27_template",
            "captured_at": "",
            "confirmed_actual_user_feedback": False,
            "not_scripted_or_curated": False,
            "consent_for_training_candidate_review": False,
        },
        "reviewer_decision": "pending_review",
        "reviewer_reason": "template row; not training data",
        "template_not_training_data": True,
    }


def build_phase27_feedback_templates(collection_pack: Mapping[str, Any]) -> dict[str, Any]:
    rows = [
        phase27_feedback_template_payload(item, index)
        for index, item in enumerate(collection_pack.get("items") or [], start=1)
        if isinstance(item, Mapping)
    ]
    return {
        "kind": "phase27_feedback_templates",
        "jsonl_rows": rows,
        "csv_rows": [
            {
                "collection_id": row.get("collection_id"),
                "prompt": row.get("prompt"),
                "messages": json.dumps(row.get("messages") or [], ensure_ascii=False),
                "runtime_output": row.get("runtime_output"),
                "response_under_review": row.get("response_under_review"),
                "metadata": json.dumps(row.get("metadata") or {}, ensure_ascii=False, sort_keys=True),
                "feedback_source": row.get("feedback_source"),
                "feedback_action": _dict(row.get("feedback")).get("action"),
                "edited_text": "",
                "user_feedback": "",
                "signal_id": _dict(row.get("feedback")).get("signal_id"),
                "attestation": json.dumps(row.get("attestation") or {}, ensure_ascii=False, sort_keys=True),
                "reviewer_decision": row.get("reviewer_decision"),
                "reviewer_reason": row.get("reviewer_reason"),
            }
            for row in rows
        ],
        "created_at": _utcnow_iso(),
    }


def validate_phase27_feedback_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    body = _row_payload(payload)
    metadata = _dict(body.get("metadata"))
    feedback_source = str(body.get("feedback_source") or metadata.get("feedback_source") or "")
    phase25_validation = validate_phase25_actual_feedback_payload(body)
    errors = list(phase25_validation.get("errors") or [])
    quarantine_reasons: list[str] = []
    non_training_reasons: list[str] = []
    bundle = _payload_text_bundle(body)
    expected_citation = str(metadata.get("expected_citation") or metadata.get("source_ref") or "")

    if feedback_source in _NON_TRAINING_FEEDBACK_SOURCES:
        non_training_reasons.append("template_or_scripted_feedback_not_training_data")
    if metadata.get("template_not_training_data") is True or body.get("template_not_training_data") is True:
        non_training_reasons.append("template_feedback_not_training_data")
    if str(metadata.get("source_policy") or "").lower().startswith("example row"):
        non_training_reasons.append("sample_feedback_not_training_data")
    if _PHONE_PATTERN.search(bundle) or _EMAIL_PATTERN.search(bundle):
        quarantine_reasons.append("pii_detected")
    if not expected_citation or expected_citation == "未提供可验证引用":
        quarantine_reasons.append("missing_required_citation")
    if _EXTERNAL_LAW_PATTERN.search(bundle):
        quarantine_reasons.append("external_law_reference")
    if _LEGAL_CONCLUSION_PATTERN.search(bundle):
        quarantine_reasons.append("legal_conclusion_or_direct_sign_target")

    status = "passed"
    if non_training_reasons:
        status = "non_training"
    elif quarantine_reasons:
        status = "quarantined"
    elif errors:
        status = "blocked"

    return {
        "kind": "phase27_actual_feedback_validation",
        "passed": status == "passed",
        "status": status,
        "errors": errors,
        "non_training_reasons": sorted(set(non_training_reasons)),
        "quarantine_reasons": sorted(set(quarantine_reasons)),
        "phase25_validation": phase25_validation,
        "feedback_source": feedback_source,
        "created_at": _utcnow_iso(),
    }


def build_phase27_import_batch(payloads: list[Mapping[str, Any]]) -> dict[str, Any]:
    accepted_signals: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    quarantined: list[dict[str, Any]] = []
    non_training: list[dict[str, Any]] = []
    intakes: list[dict[str, Any]] = []

    for index, raw_payload in enumerate(payloads):
        payload = _row_payload(raw_payload)
        validation = validate_phase27_feedback_payload(payload)
        if validation["status"] == "non_training":
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
        if validation["status"] == "quarantined":
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
        if validation["status"] == "blocked":
            record = {
                "batch_index": index,
                "status": "blocked",
                "training_disposition": "blocked",
                "reasons": validation["errors"] or ["invalid_actual_feedback"],
                "validation": validation,
            }
            blocked.append(record)
            intakes.append(record)
            continue

        intake = build_phase25_actual_feedback_signal(payload)
        record = {"batch_index": index, **intake}
        if intake.get("status") == "accepted_pending_review" and isinstance(intake.get("signal"), Mapping):
            signal = dict(intake["signal"])
            signal.setdefault("metadata", {})["phase"] = "phase27"
            signal.setdefault("metadata", {})["phase27_imported_at"] = _utcnow_iso()
            signal["phase27_review_state"] = "pending_review"
            signal["eligible_for_training"] = False
            accepted_signals.append(signal)
            record["signal"] = signal
        else:
            blocked.append(
                {
                    "batch_index": index,
                    "status": "blocked",
                    "training_disposition": "blocked",
                    "reasons": _dict(intake.get("validation")).get("errors") or ["phase25_intake_blocked"],
                    "validation": validation,
                }
            )
        intakes.append(record)

    return {
        "kind": "phase27_actual_feedback_import_batch",
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


def phase27_store_path(root: Path, workspace: str = "user_default") -> Path:
    return Path(root) / "data" / f"phase27_actual_feedback_{workspace}.json"


def load_phase27_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "kind": "phase27_persisted_state",
            "signals": [],
            "review_decisions": [],
            "import_batches": [],
            "created_at": _utcnow_iso(),
            "updated_at": _utcnow_iso(),
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        payload = {}
    state = dict(payload) if isinstance(payload, Mapping) else {}
    state.setdefault("kind", "phase27_persisted_state")
    state.setdefault("signals", [])
    state.setdefault("review_decisions", [])
    state.setdefault("import_batches", [])
    state.setdefault("created_at", _utcnow_iso())
    state["updated_at"] = state.get("updated_at") or _utcnow_iso()
    return state


def save_phase27_state(path: Path, state: Mapping[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(state)
    payload["updated_at"] = _utcnow_iso()
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def append_phase27_import_batch(path: Path, batch: Mapping[str, Any]) -> dict[str, Any]:
    state = load_phase27_state(path)
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
    return save_phase27_state(path, state)


def apply_phase27_review_decision(path: Path, decision: Mapping[str, Any]) -> dict[str, Any]:
    state = load_phase27_state(path)
    signal_ids = decision.get("signal_ids") or decision.get("signal_id") or []
    if isinstance(signal_ids, str):
        signal_ids = [signal_ids]
    review_state = str(decision.get("state") or decision.get("review_state") or "")
    reason = str(decision.get("reason") or "").strip()
    if review_state not in PHASE27_REVIEW_STATES:
        return {
            "kind": "phase27_review_decision_result",
            "status": "blocked",
            "reason": "unsupported_review_state",
            "allowed_states": sorted(PHASE27_REVIEW_STATES),
            "auto_promotion_allowed": False,
        }
    if not reason:
        return {
            "kind": "phase27_review_decision_result",
            "status": "blocked",
            "reason": "review_reason_required",
            "auto_promotion_allowed": False,
        }
    known_signal_ids = {str(signal.get("signal_id")) for signal in state.get("signals") or [] if isinstance(signal, Mapping)}
    applied: list[dict[str, Any]] = []
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
        applied.append({"signal_id": signal_id, "status": "applied", "state": review_state})
        decisions = [
            dict(item)
            for item in state.get("review_decisions") or []
            if isinstance(item, Mapping) and str(item.get("signal_id")) != signal_id
        ]
        decisions.append(record)
        state["review_decisions"] = decisions
    saved = save_phase27_state(path, state)
    return {
        "kind": "phase27_review_decision_result",
        "status": "completed" if any(item["status"] == "applied" for item in applied) else "blocked",
        "applied": applied,
        "state": saved,
        "auto_promotion_allowed": False,
        "created_at": _utcnow_iso(),
    }


def _decisions_by_signal(review_decisions: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(item.get("signal_id")): dict(item)
        for item in review_decisions
        if isinstance(item, Mapping) and item.get("signal_id")
    }


def build_phase27_review_state(
    *,
    signals: list[Mapping[str, Any]],
    review_decisions: list[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    decisions = _decisions_by_signal(review_decisions or [])
    approved_ids = {
        signal_id
        for signal_id, decision in decisions.items()
        if decision.get("state") == "approved_for_candidate"
    }
    excluded_ids = {
        signal_id
        for signal_id, decision in decisions.items()
        if decision.get("state") in {"excluded", "quarantined"}
    }
    queue = build_phase25_review_queue(signals)
    reviewed = apply_phase25_review_decisions(
        queue,
        signals,
        approved_signal_ids=approved_ids,
        excluded_signal_ids=excluded_ids,
    )
    patched: list[dict[str, Any]] = []
    for item in reviewed.get("items") or []:
        if not isinstance(item, Mapping):
            continue
        record = dict(item)
        decision = decisions.get(str(record.get("signal_id")))
        if decision:
            state = str(decision.get("state") or record.get("state") or "pending_review")
            if state == "quarantined":
                record["state"] = "quarantined"
            record["phase27_decision_reason"] = decision.get("reason")
            record["phase27_reviewer_id"] = decision.get("reviewer_id")
        patched.append(record)
    reviewed = {
        **reviewed,
        "items": patched,
        "state_counts": dict(Counter(str(item.get("state") or "unknown") for item in patched)),
    }
    return {
        "kind": "phase27_review_state",
        "queue": queue,
        "reviewed": reviewed,
        "review_decision_count": len(decisions),
        "pending_review_count": reviewed["state_counts"].get("pending_review", 0),
        "approved_for_candidate_count": reviewed["state_counts"].get("approved_for_candidate", 0),
        "excluded_count": reviewed["state_counts"].get("excluded", 0),
        "quarantined_count": reviewed["state_counts"].get("quarantined", 0),
        "created_at": _utcnow_iso(),
    }


def build_phase27_readiness(
    *,
    signals: list[Mapping[str, Any]],
    review_decisions: list[Mapping[str, Any]] | None = None,
    local_models: list[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    holdout = build_phase24_holdout(regression_count=50, hard_count=50)
    review_state = build_phase27_review_state(signals=signals, review_decisions=review_decisions or [])
    reviewed = _dict(review_state.get("reviewed"))
    routing = build_phase25_routing_report(reviewed, signals)
    holdout_chunk_ids = {str(item.get("chunk_id")) for item in holdout["prompts"] if item.get("chunk_id")}
    candidates = build_phase24_candidate_artifacts(
        signals=signals,
        reviewed=reviewed,
        routing_report=routing,
        holdout_chunk_ids=holdout_chunk_ids,
    )
    integrity = phase24_holdout_integrity_check(
        holdout=holdout,
        sft_samples=candidates["sft_samples"],
        dpo_pairs=candidates["dpo_pairs"],
    )
    runtime_eval = evaluate_phase24_runtime_contract_holdout(holdout)
    runtime_decision = phase24_runtime_product_decision(runtime_eval)
    model_selection = build_phase24_model_selection(local_models=local_models or [])
    readiness = build_phase25_training_readiness_report(
        reviewed=reviewed,
        routing_report=routing,
        candidate_manifest=candidates["candidate_manifest"],
        candidate_quality_report=candidates["quality_report"],
        holdout_integrity=integrity,
        runtime_decision=runtime_decision,
        model_selection=model_selection,
    )
    job_specs = build_phase25_training_job_specs(readiness, model_selection)
    return {
        "kind": "phase27_training_readiness",
        "actual_feedback_count": len(signals),
        "review_state": review_state,
        "routing_report": routing,
        "candidate_artifacts": candidates,
        "holdout_integrity_check": integrity,
        "runtime_eval": runtime_eval,
        "runtime_decision": runtime_decision,
        "model_selection": model_selection,
        "training_readiness": readiness,
        "training_job_specs": job_specs,
        "created_at": _utcnow_iso(),
    }


def build_phase27_training_attempt(readiness_payload: Mapping[str, Any]) -> dict[str, Any]:
    readiness = _dict(readiness_payload.get("training_readiness"))
    if readiness.get("status") != "ready_for_real_training_probe":
        return {
            "kind": "phase27_training_attempt",
            "status": "blocked",
            "reason": ";".join(readiness.get("blockers") or ["training_readiness_blocked"]),
            "adapter_artifact_created": False,
            "auto_promotion_allowed": False,
            "created_at": _utcnow_iso(),
        }
    return {
        "kind": "phase27_training_attempt",
        "status": "ready_to_launch",
        "reason": "approved actual feedback threshold met; launch requires explicit operator action",
        "adapter_artifact_created": False,
        "auto_promotion_allowed": False,
        "job_specs": readiness_payload.get("training_job_specs"),
        "created_at": _utcnow_iso(),
    }


def build_phase27_comparison_summary(
    *,
    phase26_summary: Mapping[str, Any],
    collection_pack: Mapping[str, Any],
    import_batch: Mapping[str, Any],
    readiness_payload: Mapping[str, Any],
    training_attempt: Mapping[str, Any],
) -> dict[str, Any]:
    readiness = _dict(readiness_payload.get("training_readiness"))
    final = "run_qwen3_4b_training_probe" if training_attempt.get("status") == "ready_to_launch" else "collect_more_actual_feedback"
    return {
        "kind": "phase27_comparison_summary",
        "status": "completed",
        "phase26_review": {
            "collection_count": phase26_summary.get("collection_count"),
            "actual_feedback_count": phase26_summary.get("actual_feedback_count"),
            "approved_actual_candidate_count": phase26_summary.get("approved_actual_candidate_count"),
            "final_recommendation": phase26_summary.get("final_recommendation"),
        },
        "collection_count": collection_pack.get("collection_count", 0),
        "actual_feedback_count": readiness_payload.get("actual_feedback_count", 0),
        "accepted_pending_review_count": import_batch.get("accepted_pending_review_count", 0),
        "approved_actual_candidate_count": readiness.get("approved_actual_candidate_count", 0),
        "review_state_counts": _dict(_dict(readiness_payload.get("review_state")).get("reviewed")).get("state_counts") or {},
        "candidate_manifest": readiness.get("candidate_manifest"),
        "training_readiness": readiness,
        "training_attempt": training_attempt,
        "final_recommendation": final,
        "auto_promotion_allowed": False,
        "feedback_source_policy": "only attested actual user feedback can be approved for training candidates",
        "created_at": _utcnow_iso(),
    }


__all__ = [
    "PHASE27_KIND",
    "PHASE27_MIN_APPROVED_ACTUAL_CANDIDATES",
    "PHASE27_REVIEW_STATES",
    "append_phase27_import_batch",
    "apply_phase27_review_decision",
    "build_phase27_collection_pack",
    "build_phase27_comparison_summary",
    "build_phase27_feedback_templates",
    "build_phase27_import_batch",
    "build_phase27_readiness",
    "build_phase27_review_state",
    "build_phase27_training_attempt",
    "load_phase27_state",
    "phase27_payloads_from_csv",
    "phase27_payloads_from_jsonl",
    "phase27_store_path",
    "save_phase27_state",
    "validate_phase27_feedback_payload",
]
