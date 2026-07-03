"""Phase35 lightweight local interaction capture primitives."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

from pfe_core.phase32_personal_agent_preference import contains_raw_private_text, write_jsonl
from pfe_core.phase34_simulated_user_acceptance_judge import PHASE34_FEEDBACK_SOURCE


PHASE35_KIND = "phase35_local_interaction_capture"
PHASE35_CAPTURE_SOURCE = "pfe_local_interaction_capture"
PHASE35_SIMULATED_SOURCE = "simulated_local_interaction"
PHASE35_REVIEW_STATES = {"pending_review", "approved_for_phase36_review", "excluded", "quarantined"}
PHASE35_FEEDBACK_ACTIONS = {"accept", "reject", "edit", "correction", "continue", "final_acceptance"}


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _stable_id(*parts: str, length: int = 12) -> str:
    digest = hashlib.sha256("\n".join(str(part) for part in parts).encode("utf-8")).hexdigest()
    return digest[:length]


def _truthy(value: Any) -> bool:
    if value is True:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return False


def _slug(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value or "user_default")).strip("-")
    return text or "user_default"


def phase35_store_path(root: Path, workspace: str = "user_default") -> Path:
    return Path(root) / "data" / f"phase35_local_interactions_{_slug(workspace)}.json"


def render_phase35_agent_response(*, user_goal: str, model_variant: str = "adapter") -> dict[str, Any]:
    variant = model_variant.strip().lower()
    if variant not in {"base", "adapter"}:
        raise ValueError("model_variant must be base or adapter")
    if variant == "base":
        response = (
            "我可以先给你一个整体规划，再逐步推进。你可以补充更多上下文，"
            "我再判断下一步应该怎么做。"
        )
    else:
        response = (
            "我先按当前目标做最小可验证推进：确认工作区状态、执行能跑的检查、"
            "保存证据和阻塞原因；不会宣称未实际完成的提交、PR 或关停。"
        )
    return {
        "kind": "phase35_agent_response",
        "model_variant": variant,
        "generation_mode": "phase35_local_profile_replay",
        "actual_model_call": False,
        "user_goal": user_goal,
        "assistant_response": response,
        "created_at": _utcnow_iso(),
    }


def build_phase35_interaction_record(
    *,
    workspace: str = "user_default",
    user_goal: str,
    assistant_response: str,
    feedback_action: str,
    user_feedback: str = "",
    edited_text: str = "",
    model_variant: str = "adapter",
    session_id: str = "",
    interaction_id: str = "",
    operator_id: str = "",
    confirmed_actual_user_feedback: bool = False,
    consent_for_training_candidate_review: bool = False,
    not_scripted_or_curated: bool = False,
    simulated_local_interaction: bool = False,
) -> dict[str, Any]:
    action = feedback_action.strip().lower()
    session = session_id or f"phase35-session-{_stable_id(workspace, user_goal, length=10)}"
    captured_at = _utcnow_iso()
    source = PHASE35_SIMULATED_SOURCE if simulated_local_interaction else PHASE35_CAPTURE_SOURCE
    record_id = interaction_id or f"phase35-interaction-{_stable_id(workspace, session, user_goal, assistant_response, action, length=12)}"
    actual_attested = (
        bool(confirmed_actual_user_feedback)
        and bool(consent_for_training_candidate_review)
        and bool(not_scripted_or_curated)
        and bool(str(operator_id or "").strip())
        and not simulated_local_interaction
    )
    initial_review_state = "pending_review" if actual_attested else "excluded"
    return {
        "kind": "phase35_local_interaction_record",
        "interaction_id": record_id,
        "workspace": workspace,
        "session_id": session,
        "source": source,
        "feedback_source": "actual_user_feedback" if actual_attested else source,
        "capture_method": PHASE35_CAPTURE_SOURCE,
        "simulated_local_interaction": bool(simulated_local_interaction),
        "confirmed_actual_user_feedback": bool(confirmed_actual_user_feedback) and not simulated_local_interaction,
        "not_actual_user_feedback": not actual_attested,
        "consent_for_training_candidate_review": bool(consent_for_training_candidate_review) and not simulated_local_interaction,
        "not_scripted_or_curated": bool(not_scripted_or_curated) and not simulated_local_interaction,
        "operator_id": operator_id,
        "captured_at": captured_at,
        "model_variant": model_variant,
        "generation_mode": "phase35_local_profile_replay",
        "actual_model_call": False,
        "user_goal": user_goal,
        "assistant_response": assistant_response,
        "feedback": {
            "action": action,
            "user_feedback": user_feedback,
            "edited_text": edited_text,
            "signal_id": f"{record_id}-signal",
        },
        "review_state": initial_review_state,
        "eligible_for_training": False,
        "eligible_for_phase36_review": actual_attested,
        "auto_training_allowed": False,
        "auto_promotion_allowed": False,
        "metadata": {
            "phase": "phase35",
            "source_phase34": PHASE34_FEEDBACK_SOURCE,
            "requires_phase36_review": True,
            "not_training_data_until_reviewed": True,
        },
    }


def validate_phase35_interaction_record(record: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    quarantine_reasons: list[str] = []
    action = str(_dict(record.get("feedback")).get("action") or "").strip().lower()
    simulated = bool(record.get("simulated_local_interaction"))
    attestation = {
        "confirmed_actual_user_feedback": bool(record.get("confirmed_actual_user_feedback")),
        "consent_for_training_candidate_review": bool(record.get("consent_for_training_candidate_review")),
        "not_scripted_or_curated": bool(record.get("not_scripted_or_curated")),
        "operator_id": bool(str(record.get("operator_id") or "").strip()),
    }
    actual_attested = all(attestation.values()) and not simulated
    partial_attestation = any(attestation.values()) and not actual_attested and not simulated
    if not str(record.get("user_goal") or "").strip():
        errors.append("user_goal_required")
    if not str(record.get("assistant_response") or "").strip():
        errors.append("assistant_response_required")
    if action not in PHASE35_FEEDBACK_ACTIONS:
        errors.append("unsupported_feedback_action")
    if record.get("feedback_source") == "actual_user_feedback" or partial_attestation:
        if not attestation["confirmed_actual_user_feedback"]:
            errors.append("confirmed_actual_user_feedback_required")
        if not attestation["consent_for_training_candidate_review"]:
            errors.append("consent_for_training_candidate_review_required")
        if not attestation["not_scripted_or_curated"]:
            errors.append("not_scripted_or_curated_required")
        if not attestation["operator_id"]:
            errors.append("operator_id_required")
    if record.get("simulated_local_interaction") and record.get("feedback_source") == "actual_user_feedback":
        errors.append("simulated_interaction_cannot_be_actual_feedback")
    if contains_raw_private_text(record):
        quarantine_reasons.append("raw_private_text_detected")
    status = "passed"
    if quarantine_reasons:
        status = "quarantined"
    if errors:
        status = "blocked"
    if status == "passed" and not actual_attested:
        status = "non_training"
    non_training_reasons: list[str] = []
    if status == "non_training":
        if simulated:
            non_training_reasons.append("simulated_local_interaction_not_actual_feedback")
        else:
            non_training_reasons.append("actual_feedback_attestation_required")
    return {
        "kind": "phase35_interaction_validation",
        "passed": status == "passed",
        "status": status,
        "errors": errors,
        "quarantine_reasons": quarantine_reasons,
        "non_training_reasons": non_training_reasons,
        "created_at": _utcnow_iso(),
    }


def build_phase35_capture_batch(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    accepted_pending_review: list[dict[str, Any]] = []
    non_training: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    quarantined: list[dict[str, Any]] = []
    intakes: list[dict[str, Any]] = []
    for index, raw in enumerate(records):
        record = dict(raw)
        validation = validate_phase35_interaction_record(record)
        status = validation["status"]
        record["validation"] = validation
        if status == "passed":
            record["review_state"] = "pending_review"
            record["eligible_for_training"] = False
            accepted_pending_review.append(record)
            intake = {"batch_index": index, "status": "accepted_pending_review", "record": record}
        elif status == "non_training":
            record["review_state"] = "excluded"
            non_training.append(record)
            intake = {"batch_index": index, "status": "non_training", "record": record}
        elif status == "quarantined":
            record["review_state"] = "quarantined"
            quarantined.append(record)
            intake = {"batch_index": index, "status": "quarantined", "record": record}
        else:
            record["review_state"] = "blocked"
            blocked.append(record)
            intake = {"batch_index": index, "status": "blocked", "record": record}
        intakes.append(intake)
    return {
        "kind": "phase35_local_interaction_capture_batch",
        "record_count": len(records),
        "accepted_pending_review_count": len(accepted_pending_review),
        "non_training_count": len(non_training),
        "blocked_count": len(blocked),
        "quarantined_count": len(quarantined),
        "accepted_pending_review": accepted_pending_review,
        "non_training": non_training,
        "blocked": blocked,
        "quarantined": quarantined,
        "intakes": intakes,
        "auto_training_allowed": False,
        "auto_promotion_allowed": False,
        "created_at": _utcnow_iso(),
    }


def load_phase35_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "kind": "phase35_persisted_state",
            "interactions": [],
            "capture_batches": [],
            "review_decisions": [],
            "reviewer_audit_log": [],
            "created_at": _utcnow_iso(),
            "updated_at": _utcnow_iso(),
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        payload = {}
    state = dict(payload) if isinstance(payload, Mapping) else {}
    state.setdefault("kind", "phase35_persisted_state")
    state.setdefault("interactions", [])
    state.setdefault("capture_batches", [])
    state.setdefault("review_decisions", [])
    state.setdefault("reviewer_audit_log", [])
    state.setdefault("created_at", _utcnow_iso())
    state["updated_at"] = state.get("updated_at") or _utcnow_iso()
    return state


def save_phase35_state(path: Path, state: Mapping[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(state)
    payload["updated_at"] = _utcnow_iso()
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def append_phase35_capture_batch(path: Path, batch: Mapping[str, Any]) -> dict[str, Any]:
    state = load_phase35_state(path)
    existing = {
        str(item.get("interaction_id"))
        for item in state.get("interactions") or []
        if isinstance(item, Mapping)
    }
    new_records: list[dict[str, Any]] = []
    for lane in ("accepted_pending_review", "non_training", "blocked", "quarantined"):
        new_records.extend(
            dict(item)
            for item in batch.get(lane) or []
            if isinstance(item, Mapping) and str(item.get("interaction_id")) not in existing
        )
        existing.update(str(item.get("interaction_id")) for item in new_records)
    state["interactions"] = [
        dict(item) for item in state.get("interactions") or [] if isinstance(item, Mapping)
    ] + new_records
    state["capture_batches"] = [
        dict(item) for item in state.get("capture_batches") or [] if isinstance(item, Mapping)
    ] + [
        {
            "kind": batch.get("kind"),
            "record_count": batch.get("record_count", 0),
            "accepted_pending_review_count": batch.get("accepted_pending_review_count", 0),
            "non_training_count": batch.get("non_training_count", 0),
            "blocked_count": batch.get("blocked_count", 0),
            "quarantined_count": batch.get("quarantined_count", 0),
            "created_at": batch.get("created_at") or _utcnow_iso(),
        }
    ]
    return save_phase35_state(path, state)


def build_phase35_review_queue(state: Mapping[str, Any]) -> dict[str, Any]:
    decisions = {
        str(item.get("interaction_id")): item
        for item in state.get("review_decisions") or []
        if isinstance(item, Mapping)
    }
    pending = []
    reviewed_counts: Counter[str] = Counter()
    for item in state.get("interactions") or []:
        if not isinstance(item, Mapping):
            continue
        decision = _dict(decisions.get(str(item.get("interaction_id"))))
        review_state = str(decision.get("state") or item.get("review_state") or "pending_review")
        if review_state == "pending_review":
            pending.append(dict(item))
        else:
            reviewed_counts[review_state] += 1
    return {
        "kind": "phase35_review_queue",
        "pending_review_count": len(pending),
        "pending_review": pending,
        "reviewed_counts": dict(sorted(reviewed_counts.items())),
        "auto_training_allowed": False,
        "auto_promotion_allowed": False,
        "created_at": _utcnow_iso(),
    }


def build_phase35_readiness(state: Mapping[str, Any]) -> dict[str, Any]:
    interactions = [dict(item) for item in state.get("interactions") or [] if isinstance(item, Mapping)]
    queue = build_phase35_review_queue(state)
    attested_count = sum(1 for item in interactions if item.get("eligible_for_phase36_review"))
    action_counts = Counter(str(_dict(item.get("feedback")).get("action") or "unknown") for item in interactions)
    return {
        "kind": "phase35_local_interaction_readiness",
        "interaction_count": len(interactions),
        "attested_actual_pending_review_count": attested_count,
        "pending_review_count": queue["pending_review_count"],
        "feedback_action_counts": dict(sorted(action_counts.items())),
        "current_state": "review" if queue["pending_review_count"] else "observe",
        "next_action": "review_pending_local_interactions" if queue["pending_review_count"] else "capture_local_interaction",
        "training_status": "blocked",
        "training_blocked_reason": "phase35_capture_only_phase36_review_required",
        "ready_for_phase36_review": attested_count > 0,
        "auto_training_allowed": False,
        "auto_promotion_allowed": False,
        "created_at": _utcnow_iso(),
    }


def build_phase35_phase34_review(*, phase34_summary: Mapping[str, Any]) -> dict[str, Any]:
    scores = _dict(phase34_summary.get("acceptance_scores"))
    return {
        "kind": "phase35_phase34_review",
        "phase34_completed": phase34_summary.get("status") == "completed",
        "phase34_final_recommendation": phase34_summary.get("final_recommendation"),
        "phase34_actual_user_feedback_count": phase34_summary.get("actual_user_feedback_count"),
        "phase34_adapter_win_rate": scores.get("adapter_win_rate"),
        "phase34_base_win_rate": scores.get("base_win_rate"),
        "phase35_scope": "local_interaction_capture_without_hermes_without_training",
        "hermes_integration_required": False,
        "created_at": _utcnow_iso(),
    }


def build_phase35_comparison_summary(
    *,
    phase34_review: Mapping[str, Any],
    capture_batch: Mapping[str, Any],
    state: Mapping[str, Any],
    readiness: Mapping[str, Any],
) -> dict[str, Any]:
    final_recommendation = (
        "review_local_interactions_before_training"
        if readiness.get("pending_review_count", 0)
        else "capture_attested_actual_local_interactions"
    )
    return {
        "kind": "phase35_local_interaction_capture_summary",
        "status": "completed",
        "phase34_review": dict(phase34_review),
        "capture_batch": dict(capture_batch),
        "interaction_count": readiness.get("interaction_count", 0),
        "attested_actual_pending_review_count": readiness.get("attested_actual_pending_review_count", 0),
        "simulated_local_interaction_count": capture_batch.get("non_training_count", 0),
        "pending_review_count": readiness.get("pending_review_count", 0),
        "training_status": readiness.get("training_status"),
        "training_blocked_reason": readiness.get("training_blocked_reason"),
        "final_recommendation": final_recommendation,
        "actual_training_run": False,
        "hermes_integration_used": False,
        "auto_training_allowed": False,
        "auto_promotion_allowed": False,
        "state_summary": {
            "kind": state.get("kind"),
            "interaction_count": len(state.get("interactions") or []),
            "capture_batch_count": len(state.get("capture_batches") or []),
        },
        "created_at": _utcnow_iso(),
    }


__all__ = [
    "PHASE35_CAPTURE_SOURCE",
    "PHASE35_FEEDBACK_ACTIONS",
    "PHASE35_KIND",
    "PHASE35_REVIEW_STATES",
    "PHASE35_SIMULATED_SOURCE",
    "append_phase35_capture_batch",
    "build_phase35_capture_batch",
    "build_phase35_comparison_summary",
    "build_phase35_interaction_record",
    "build_phase35_phase34_review",
    "build_phase35_readiness",
    "build_phase35_review_queue",
    "load_phase35_state",
    "phase35_store_path",
    "render_phase35_agent_response",
    "save_phase35_state",
    "validate_phase35_interaction_record",
    "write_jsonl",
]
