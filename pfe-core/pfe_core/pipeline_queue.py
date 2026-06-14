"""Train queue history and summary helpers for PipelineService."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping


def queue_history_entry(
    *,
    event: str,
    item: Mapping[str, Any],
    reason: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    timestamp = (
        item.get("updated_at")
        or item.get("triggered_at")
        or item.get("created_at")
        or datetime.now(timezone.utc).isoformat()
    )
    entry = {
        "timestamp": str(timestamp),
        "event": str(event),
        "state": str(item.get("state") or "unknown"),
        "job_id": str(item.get("job_id") or ""),
    }
    if reason:
        entry["reason"] = str(reason)
    if metadata:
        entry["metadata"] = dict(metadata)
        if metadata.get("note") is not None:
            entry["note"] = str(metadata["note"])
    return entry


def queue_sort_key(item: Mapping[str, Any]) -> tuple[int, str]:
    priority = int(item.get("priority", 0) or 0)
    ordered_at = str(
        item.get("triggered_at")
        or item.get("created_at")
        or item.get("updated_at")
        or ""
    )
    return (-priority, ordered_at)


def normalize_queue_items(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    return [dict(item) for item in raw if isinstance(item, dict)]


def queue_state_counts(items: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        state = str(item.get("state") or "unknown")
        counts[state] = counts.get(state, 0) + 1
    return counts


def queue_recent_history(items: list[dict[str, Any]], *, limit: int = 5) -> list[dict[str, Any]]:
    recent_history: list[dict[str, Any]] = []
    for item in items:
        recent_history.extend(list(item.get("history") or [])[-1:])
    return sorted(recent_history, key=lambda entry: str(entry.get("timestamp") or ""), reverse=True)[: max(1, int(limit or 5))]


def queue_history_summary(items: list[dict[str, Any]], recent_history: list[dict[str, Any]]) -> dict[str, Any]:
    last_transition = recent_history[0] if recent_history else {}
    return {
        "transition_count": sum(int(item.get("history_count", 0) or 0) for item in items),
        "last_transition": last_transition,
        "last_reason": last_transition.get("reason"),
    }


def queue_review_entries(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    entries = [
        dict(entry)
        for item in items
        for entry in list(item.get("history") or [])
        if str(entry.get("event") or "") in {"approved", "rejected"}
    ]
    return sorted(entries, key=lambda entry: str(entry.get("timestamp") or ""), reverse=True)


def queue_review_summary(
    *,
    review_entries: list[dict[str, Any]],
    awaiting_item: Mapping[str, Any] | None,
) -> dict[str, Any]:
    last_review = review_entries[0] if review_entries else {}
    awaiting = dict(awaiting_item or {})
    return {
        "reviewed_transition_count": len(review_entries),
        "approved_transition_count": sum(1 for entry in review_entries if str(entry.get("event")) == "approved"),
        "rejected_transition_count": sum(1 for entry in review_entries if str(entry.get("event")) == "rejected"),
        "last_review_event": last_review.get("event"),
        "last_review_reason": last_review.get("reason"),
        "last_review_note": last_review.get("note"),
        "next_job_id": awaiting.get("job_id"),
        "next_confirmation_reason": awaiting.get("confirmation_reason"),
    }


def queue_review_policy_summary(
    *,
    queue_mode: str,
    require_queue_confirmation: bool,
    awaiting_confirmation_count: int,
    awaiting_item: Mapping[str, Any] | None,
    queued_item: Mapping[str, Any] | None,
) -> dict[str, Any]:
    review_required_by_policy = bool(str(queue_mode) == "deferred" and require_queue_confirmation)
    if awaiting_confirmation_count > 0:
        queue_entry_mode = "awaiting_confirmation"
        next_action = "review_queue_confirmation"
        review_reason = dict(awaiting_item or {}).get("confirmation_reason")
    elif queued_item is not None:
        queue_entry_mode = "queued"
        next_action = "process_next_queue_item"
        review_reason = None
    elif str(queue_mode) == "deferred":
        queue_entry_mode = "deferred_idle"
        next_action = "await_new_queue_item"
        review_reason = None
    else:
        queue_entry_mode = "inline_execute"
        next_action = "await_signal_trigger"
        review_reason = None

    review_mode = "manual_review" if (review_required_by_policy or awaiting_confirmation_count > 0) else "auto_queue"
    return {
        "review_mode": review_mode,
        "queue_entry_mode": queue_entry_mode,
        "review_required_by_policy": review_required_by_policy,
        "review_required_now": awaiting_confirmation_count > 0,
        "review_reason": review_reason,
        "next_action": next_action,
        "summary_line": " | ".join(
            [
                f"mode={review_mode}",
                f"entry={queue_entry_mode}",
                f"next={next_action}",
            ]
            + ([f"reason={review_reason}"] if review_reason else [])
        ),
    }


def train_queue_history_payload(
    *,
    payload: Mapping[str, Any],
    workspace: str | None = None,
    job_id: str | None = None,
    limit: int = 10,
    history_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    bounded_limit = max(1, int(limit or 10))
    items = normalize_queue_items(payload.get("items") or [])
    target_item: dict[str, Any] = {}
    if job_id:
        for item in items:
            if str(item.get("job_id") or "") == str(job_id):
                target_item = item
                break
    elif items:
        last_item = payload.get("last_item")
        target_item = dict(last_item) if isinstance(last_item, dict) else dict(items[0])

    history = list(target_item.get("history") or [])
    return {
        "workspace": workspace or "user_default",
        "job_id": target_item.get("job_id"),
        "state": target_item.get("state"),
        "count": len(history),
        "limit": bounded_limit,
        "history": history[-bounded_limit:],
        "history_count": int(target_item.get("history_count", len(history)) or len(history)),
        "available_job_ids": [item.get("job_id") for item in items[:10] if item.get("job_id")],
        "history_summary": dict(history_summary or {}),
    }


__all__ = [
    "normalize_queue_items",
    "queue_history_entry",
    "queue_history_summary",
    "queue_recent_history",
    "queue_review_entries",
    "queue_review_policy_summary",
    "queue_review_summary",
    "queue_sort_key",
    "queue_state_counts",
    "train_queue_history_payload",
]
