"""Phase25 actual-user feedback readiness loop primitives.

Phase25 turns the Phase24 lesson into a product contract: only explicitly
attested actual user feedback can unlock product-value training readiness.
Curated and scripted lab feedback remain useful for testing the loop, but they
must not be counted as real training evidence.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Any, Mapping

from .inference.contracts import BOUNDARY_CONTRACT_ID
from .phase23_runtime_contract_loop import (
    build_runtime_contract_response,
    signal_record_from_contract_feedback,
)
from .phase24_real_signal_review_candidate_value import (
    build_phase24_candidate_artifacts,
    build_phase24_candidate_quality_report,
    build_phase24_holdout,
    build_phase24_model_selection,
    build_phase24_routing_report,
    evaluate_phase24_runtime_contract_holdout,
    phase24_holdout_integrity_check,
    phase24_route_signal,
    phase24_runtime_product_decision,
)


PHASE25_KIND = "phase25_actual_user_feedback_readiness_loop"
PHASE25_MIN_APPROVED_ACTUAL_CANDIDATES = 12
PHASE25_ATTESTATION_VERSION = "phase25-actual-user-feedback-attestation-v1"


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _messages_from_payload(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_messages = payload.get("messages")
    if isinstance(raw_messages, list):
        messages = [dict(item) for item in raw_messages if isinstance(item, Mapping)]
        if messages:
            return messages
    prompt = str(payload.get("prompt") or payload.get("user_input") or "")
    return [{"role": "user", "content": prompt}]


def build_phase25_attestation_template() -> dict[str, Any]:
    return {
        "kind": "phase25_actual_feedback_attestation_template",
        "feedback_source": "actual_user_feedback",
        "attestation_version": PHASE25_ATTESTATION_VERSION,
        "required_attestation": {
            "operator_id": "human-reviewer-id",
            "capture_method": "api_or_review_ui",
            "captured_at": "ISO-8601 timestamp",
            "confirmed_actual_user_feedback": True,
            "not_scripted_or_curated": True,
            "consent_for_training_candidate_review": True,
        },
        "required_feedback": {
            "action": "correction_or_edit",
            "edited_text": "four-section boundary-preserving target when action is edit/correction",
            "user_feedback": "human feedback note",
        },
        "auto_promotion_allowed": False,
    }


def validate_phase25_actual_feedback_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    body = _dict(payload)
    feedback = _dict(body.get("feedback"))
    metadata = _dict(body.get("metadata"))
    attestation = _dict(body.get("attestation") or feedback.get("attestation") or metadata.get("attestation"))
    errors: list[str] = []

    feedback_source = str(
        body.get("feedback_source")
        or feedback.get("feedback_source")
        or metadata.get("feedback_source")
        or attestation.get("feedback_source")
        or ""
    )
    if feedback_source != "actual_user_feedback":
        errors.append("feedback_source_must_be_actual_user_feedback")

    for key in ("operator_id", "capture_method", "captured_at"):
        if not str(attestation.get(key) or "").strip():
            errors.append(f"attestation_{key}_required")
    required_bools = (
        "confirmed_actual_user_feedback",
        "not_scripted_or_curated",
        "consent_for_training_candidate_review",
    )
    for key in required_bools:
        if attestation.get(key) is not True:
            errors.append(f"attestation_{key}_must_be_true")

    action = str(feedback.get("action") or body.get("action") or "").strip().lower()
    if action not in {"accept", "reject", "edit", "preference", "correction", "safety_block"}:
        errors.append("feedback_action_unsupported")
    if action in {"edit", "correction"} and not str(feedback.get("edited_text") or body.get("edited_text") or "").strip():
        errors.append("edited_text_required_for_edit_or_correction")
    if not str(body.get("prompt") or body.get("user_input") or "").strip() and not _messages_from_payload(body)[0].get("content"):
        errors.append("prompt_or_messages_required")

    return {
        "kind": "phase25_actual_feedback_validation",
        "passed": not errors,
        "errors": errors,
        "feedback_source": feedback_source,
        "attestation_version": PHASE25_ATTESTATION_VERSION,
        "attestation": attestation,
        "created_at": _utcnow_iso(),
    }


def build_phase25_actual_feedback_signal(payload: Mapping[str, Any]) -> dict[str, Any]:
    body = _dict(payload)
    validation = validate_phase25_actual_feedback_payload(body)
    if not validation["passed"]:
        return {
            "kind": "phase25_actual_feedback_intake",
            "status": "blocked",
            "validation": validation,
            "signal": None,
            "phase25_route": {
                "eligible_for_training": False,
                "product_value_training_allowed": False,
                "excluded_reason": "invalid_actual_feedback_attestation",
            },
            "auto_promotion_allowed": False,
            "created_at": _utcnow_iso(),
        }

    feedback = _dict(body.get("feedback"))
    metadata = _dict(body.get("metadata"))
    attestation = _dict(validation.get("attestation"))
    mode = str(body.get("mode") or metadata.get("response_contract") or "contract_boundary_summary")
    messages = _messages_from_payload(body)
    runtime_response = build_runtime_contract_response(
        messages=messages,
        metadata={**metadata, "response_contract": mode},
        mode=mode,
    )
    signal = signal_record_from_contract_feedback(
        action=str(feedback.get("action") or body.get("action") or "accept"),
        runtime_response=runtime_response,
        edited_text=str(feedback.get("edited_text") or body.get("edited_text") or ""),
        user_feedback=str(feedback.get("user_feedback") or body.get("user_feedback") or ""),
        confidence=float(feedback.get("confidence", body.get("confidence", 0.95)) or 0.95),
        session_id=str(body.get("session_id") or feedback.get("session_id") or ""),
        request_id=str(body.get("request_id") or feedback.get("request_id") or ""),
        signal_id=str(feedback.get("signal_id") or body.get("signal_id") or "") or None,
        metadata={
            **metadata,
            "phase": "phase25",
            "feedback_source": "actual_user_feedback",
            "feedback_source_is_actual_user_feedback": True,
            "attestation": attestation,
            "attestation_version": PHASE25_ATTESTATION_VERSION,
            "consent_for_training_candidate_review": True,
        },
    )
    signal["feedback_source"] = "actual_user_feedback"
    signal["feedback_source_is_actual_user_feedback"] = True
    signal["attestation"] = attestation
    signal.setdefault("metadata", {})["feedback_source"] = "actual_user_feedback"
    signal.setdefault("metadata", {})["feedback_source_is_actual_user_feedback"] = True
    signal.setdefault("metadata", {})["attestation"] = attestation
    route = phase24_route_signal(signal, {"state": "pending_review"}).to_dict()
    return {
        "kind": "phase25_actual_feedback_intake",
        "status": "accepted_pending_review",
        "validation": validation,
        "runtime_contract": runtime_response,
        "signal": signal,
        "phase25_route": route,
        "auto_promotion_allowed": False,
        "created_at": _utcnow_iso(),
    }


def build_phase25_review_queue(signals: list[Mapping[str, Any]]) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    for signal in signals:
        pending_route = phase24_route_signal(signal, {"state": "pending_review"}).to_dict()
        if pending_route.get("excluded_reason") and pending_route.get("excluded_reason") != "not_review_approved":
            state = "excluded"
            reason = str(pending_route.get("excluded_reason"))
        elif _dict(signal.get("metadata")).get("feedback_source") != "actual_user_feedback":
            state = "excluded"
            reason = "not_actual_user_feedback"
        else:
            state = "pending_review"
            reason = "actual_feedback_requires_manual_review"
        items.append(
            {
                "queue_id": f"phase25-review-{signal.get('signal_id')}",
                "signal_id": signal.get("signal_id"),
                "state": state,
                "reason": reason,
                "feedback_source": signal.get("feedback_source") or _dict(signal.get("metadata")).get("feedback_source"),
                "signal_type": signal.get("signal_type"),
                "phase25_pending_route": pending_route,
                "updated_at": _utcnow_iso(),
            }
        )
    return {
        "kind": "phase25_actual_feedback_review_queue",
        "queue_count": len(items),
        "state_counts": dict(Counter(str(item["state"]) for item in items)),
        "items": items,
        "created_at": _utcnow_iso(),
    }


def apply_phase25_review_decisions(
    queue: Mapping[str, Any],
    signals: list[Mapping[str, Any]],
    *,
    approved_signal_ids: set[str] | None = None,
    excluded_signal_ids: set[str] | None = None,
) -> dict[str, Any]:
    approved = set(approved_signal_ids or set())
    excluded = set(excluded_signal_ids or set())
    by_signal_id = {str(signal.get("signal_id")): dict(signal) for signal in signals}
    reviewed: list[dict[str, Any]] = []
    for item in queue.get("items") or []:
        if not isinstance(item, Mapping):
            continue
        signal_id = str(item.get("signal_id") or "")
        signal = by_signal_id.get(signal_id, {})
        state = str(item.get("state") or "pending_review")
        reason = str(item.get("reason") or "")
        if signal_id in excluded:
            state = "excluded"
            reason = "manual_review_excluded"
        elif signal_id in approved and state != "excluded":
            approved_route = phase24_route_signal(signal, {"state": "approved_for_candidate"}).to_dict()
            if approved_route.get("eligible_for_training") and approved_route.get("product_value_training_allowed"):
                state = "approved_for_candidate"
                reason = "manual_review_approved_actual_user_feedback"
            else:
                state = "excluded"
                reason = str(approved_route.get("excluded_reason") or "not_training_eligible_after_review")
        reviewed.append({**dict(item), "state": state, "decision_reason": reason, "decided_at": _utcnow_iso()})
    return {
        "kind": "phase25_reviewed_actual_feedback",
        "reviewed_count": len(reviewed),
        "state_counts": dict(Counter(str(item["state"]) for item in reviewed)),
        "items": reviewed,
        "created_at": _utcnow_iso(),
    }


def build_phase25_routing_report(reviewed: Mapping[str, Any], signals: list[Mapping[str, Any]]) -> dict[str, Any]:
    return build_phase24_routing_report(reviewed, signals)


def build_phase25_training_readiness_report(
    *,
    reviewed: Mapping[str, Any],
    routing_report: Mapping[str, Any],
    candidate_manifest: Mapping[str, Any],
    candidate_quality_report: Mapping[str, Any],
    holdout_integrity: Mapping[str, Any],
    runtime_decision: Mapping[str, Any],
    model_selection: Mapping[str, Any],
) -> dict[str, Any]:
    approved_actual_count = int(routing_report.get("product_value_training_allowed_count", 0) or 0)
    blockers: list[str] = []
    if approved_actual_count < PHASE25_MIN_APPROVED_ACTUAL_CANDIDATES:
        blockers.append("insufficient_approved_actual_user_feedback")
    if not candidate_quality_report.get("passed"):
        blockers.append("candidate_quality_report_failed")
    if not holdout_integrity.get("passed"):
        blockers.append("holdout_contamination")
    if runtime_decision.get("recommendation") != "primary_product_path":
        blockers.append("runtime_contract_not_stable")
    if not _dict(model_selection.get("selected_model")):
        blockers.append("no_feasible_qwen_training_model_selected")
    if int(candidate_manifest.get("sft_sample_count", 0) or 0) < PHASE25_MIN_APPROVED_ACTUAL_CANDIDATES:
        blockers.append("insufficient_sft_candidate_samples")
    if int(candidate_manifest.get("dpo_pair_count", 0) or 0) < PHASE25_MIN_APPROVED_ACTUAL_CANDIDATES:
        blockers.append("insufficient_dpo_candidate_pairs")
    return {
        "kind": "phase25_training_readiness_report",
        "status": "ready_for_real_training_probe" if not blockers else "collect_actual_feedback",
        "blockers": blockers,
        "approved_actual_candidate_count": approved_actual_count,
        "minimum_approved_actual_candidates": PHASE25_MIN_APPROVED_ACTUAL_CANDIDATES,
        "review_state_counts": reviewed.get("state_counts") or {},
        "candidate_manifest": dict(candidate_manifest),
        "runtime_recommendation": runtime_decision.get("recommendation"),
        "selected_model": model_selection.get("selected_model"),
        "auto_promotion_allowed": False,
        "created_at": _utcnow_iso(),
    }


def build_phase25_training_job_specs(readiness: Mapping[str, Any], model_selection: Mapping[str, Any]) -> dict[str, Any]:
    selected = _dict(model_selection.get("selected_model"))
    status = "ready" if readiness.get("status") == "ready_for_real_training_probe" else "blocked"
    return {
        "kind": "phase25_training_job_specs",
        "status": status,
        "jobs": [
            {
                "job_id": "phase25-sft-actual-feedback-12-step",
                "method": "sft",
                "model": selected.get("path") or selected.get("name") or "unselected",
                "steps": 12,
                "dataset": "actual_feedback_sft_candidates.jsonl",
                "status": status,
            },
            {
                "job_id": "phase25-dpo-actual-feedback-12-step",
                "method": "dpo",
                "model": selected.get("path") or selected.get("name") or "unselected",
                "steps": 12,
                "dataset": "actual_feedback_dpo_pairs.jsonl",
                "status": status,
            },
        ],
        "auto_promotion_allowed": False,
        "created_at": _utcnow_iso(),
    }


def build_phase25_empty_readiness(*, local_models: list[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    holdout = build_phase24_holdout(regression_count=50, hard_count=50)
    runtime_eval = evaluate_phase24_runtime_contract_holdout(holdout)
    runtime_decision = phase24_runtime_product_decision(runtime_eval)
    model_selection = build_phase24_model_selection(local_models=local_models or [])
    queue = build_phase25_review_queue([])
    reviewed = apply_phase25_review_decisions(queue, [])
    routing = build_phase25_routing_report(reviewed, [])
    candidates = build_phase24_candidate_artifacts(
        signals=[],
        reviewed=reviewed,
        routing_report=routing,
        holdout_chunk_ids={str(item.get("chunk_id")) for item in holdout["prompts"] if item.get("chunk_id")},
    )
    integrity = phase24_holdout_integrity_check(
        holdout=holdout,
        sft_samples=candidates["sft_samples"],
        dpo_pairs=candidates["dpo_pairs"],
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
    return {
        "kind": "phase25_empty_actual_feedback_readiness",
        "actual_feedback_count": 0,
        "queue": queue,
        "reviewed": reviewed,
        "routing_report": routing,
        "candidate_artifacts": candidates,
        "holdout_integrity_check": integrity,
        "runtime_eval": runtime_eval,
        "runtime_decision": runtime_decision,
        "model_selection": model_selection,
        "training_readiness": readiness,
        "training_job_specs": build_phase25_training_job_specs(readiness, model_selection),
        "attestation_template": build_phase25_attestation_template(),
        "created_at": _utcnow_iso(),
    }


def build_phase25_comparison_summary(readiness_payload: Mapping[str, Any]) -> dict[str, Any]:
    readiness = _dict(readiness_payload.get("training_readiness"))
    runtime_eval = _dict(readiness_payload.get("runtime_eval"))
    return {
        "kind": "phase25_comparison_summary",
        "status": "completed",
        "actual_feedback_count": readiness_payload.get("actual_feedback_count", 0),
        "approved_actual_candidate_count": readiness.get("approved_actual_candidate_count", 0),
        "runtime_contract_eval": {
            "holdout_count": runtime_eval.get("holdout_count"),
            "scores": runtime_eval.get("scores"),
            "decision": readiness_payload.get("runtime_decision"),
        },
        "training_readiness": readiness,
        "training_job_specs": readiness_payload.get("training_job_specs"),
        "final_recommendation": (
            "ready_for_real_training_probe"
            if readiness.get("status") == "ready_for_real_training_probe"
            else "collect_actual_user_feedback"
        ),
        "auto_promotion_allowed": False,
        "created_at": _utcnow_iso(),
    }


__all__ = [
    "PHASE25_ATTESTATION_VERSION",
    "PHASE25_KIND",
    "PHASE25_MIN_APPROVED_ACTUAL_CANDIDATES",
    "apply_phase25_review_decisions",
    "build_phase25_actual_feedback_signal",
    "build_phase25_attestation_template",
    "build_phase25_comparison_summary",
    "build_phase25_empty_readiness",
    "build_phase25_review_queue",
    "build_phase25_routing_report",
    "build_phase25_training_job_specs",
    "build_phase25_training_readiness_report",
    "validate_phase25_actual_feedback_payload",
]
