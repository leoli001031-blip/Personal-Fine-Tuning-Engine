"""Candidate action history and timeline helpers for PipelineService."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping


def candidate_history_entry(action: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": str(action.get("action") or "candidate_action"),
        "status": str(action.get("status") or "noop"),
        "reason": str(action.get("reason") or ""),
        "candidate_version": action.get("candidate_version"),
        "promoted_version": action.get("promoted_version"),
        "archived_version": action.get("archived_version"),
        "operator_note": action.get("operator_note"),
        "previous_candidate_state": action.get("previous_candidate_state"),
        "triggered": bool(action.get("triggered", False)),
    }


def normalize_candidate_history(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    return [dict(item) for item in raw if isinstance(item, dict)]


def candidate_history_summary(history: list[dict[str, Any]]) -> dict[str, Any]:
    latest = history[-1] if history else {}
    return {
        "count": len(history),
        "latest_timestamp": latest.get("timestamp"),
        "last_action": latest.get("action"),
        "last_status": latest.get("status"),
        "last_reason": latest.get("reason"),
        "last_candidate_version": latest.get("candidate_version"),
        "last_note": latest.get("operator_note"),
        "action_counts": {
            "promote_candidate": sum(1 for item in history if str(item.get("action")) == "promote_candidate"),
            "archive_candidate": sum(1 for item in history if str(item.get("action")) == "archive_candidate"),
        },
        "items": history[-5:],
    }


def candidate_history_payload(
    *,
    history: list[dict[str, Any]],
    workspace: str | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    bounded_limit = max(1, int(limit or 10))
    latest = history[-1] if history else {}
    return {
        "workspace": workspace or "user_default",
        "count": len(history),
        "limit": bounded_limit,
        "last_action": latest.get("action"),
        "last_status": latest.get("status"),
        "last_reason": latest.get("reason"),
        "last_candidate_version": latest.get("candidate_version"),
        "last_note": latest.get("operator_note"),
        "latest_timestamp": latest.get("timestamp"),
        "items": history[-bounded_limit:],
    }


def candidate_timeline_stage(action: str, status: str) -> str:
    if action == "promote_candidate" and status == "completed":
        return "promoted"
    if action == "archive_candidate" and status == "completed":
        return "archived"
    if status == "blocked":
        return "blocked"
    if status == "noop":
        return "noop"
    return "candidate_action"


def candidate_timeline_summary(history: list[dict[str, Any]]) -> dict[str, Any]:
    latest = history[-1] if history else {}
    transitions = sum(1 for item in history if str(item.get("status") or "") in {"completed", "blocked", "noop"})
    current_stage = "idle"
    if latest:
        current_stage = candidate_timeline_stage(
            str(latest.get("action") or ""),
            str(latest.get("status") or ""),
        )
    return {
        "count": len(history),
        "transition_count": transitions,
        "current_stage": current_stage,
        "last_transition": latest,
        "last_reason": latest.get("reason"),
        "last_candidate_version": latest.get("candidate_version"),
        "latest_timestamp": latest.get("timestamp"),
    }


def candidate_timeline_payload(
    *,
    history: list[dict[str, Any]],
    workspace: str | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    bounded_limit = max(1, int(limit or 10))
    latest = history[-1] if history else {}
    timeline_items = []
    for item in history[-bounded_limit:]:
        action = str(item.get("action") or "candidate_action")
        status = str(item.get("status") or "noop")
        timeline_items.append(
            {
                **dict(item),
                "stage": candidate_timeline_stage(action, status),
                "label": f"{action}:{status}",
            }
        )
    summary = candidate_timeline_summary(history)
    return {
        "workspace": workspace or "user_default",
        "count": len(history),
        "limit": bounded_limit,
        "current_stage": summary.get("current_stage"),
        "transition_count": summary.get("transition_count"),
        "last_transition": latest,
        "last_reason": latest.get("reason"),
        "last_candidate_version": latest.get("candidate_version"),
        "latest_timestamp": latest.get("timestamp"),
        "items": timeline_items,
    }


__all__ = [
    "candidate_history_entry",
    "candidate_history_payload",
    "candidate_history_summary",
    "candidate_timeline_payload",
    "candidate_timeline_stage",
    "candidate_timeline_summary",
    "normalize_candidate_history",
]
