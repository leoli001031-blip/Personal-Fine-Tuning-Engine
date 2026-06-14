"""Queue review policy derivation helpers for console rendering."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .console_app_data_core import _mapping, _summary_field, _value


def _resolved_queue_review_policy(
    *,
    console: Mapping[str, Any],
    overview: Mapping[str, Any],
    train_queue: Mapping[str, Any],
    trigger_policy: Mapping[str, Any],
    trigger_blocked_reason: Any,
    trigger_blocked_action: Any,
) -> dict[str, Any]:
    queue_review_policy = (
        _mapping(console.get("queue_review_policy"))
        or _mapping(overview.get("queue_review_policy"))
        or _mapping(_mapping(train_queue).get("review_policy_summary"))
    )
    if queue_review_policy:
        return queue_review_policy

    review_mode = (
        _summary_field(overview.get("trigger_policy_summary"), "review")
        or _value(trigger_policy, "review_mode", default="")
    )
    queue_entry_mode = _value(trigger_policy, "queue_entry_mode", default="")
    next_action = ""
    blocked_reason = str(trigger_blocked_reason or "").strip().lower()
    blocked_action_text = str(trigger_blocked_action or "").strip()

    if blocked_reason == "queue_pending_review":
        next_action = blocked_action_text or "review_queue_confirmation"
    elif blocked_reason == "queue_waiting_execution":
        next_action = blocked_action_text or "process_next_queue_item"
    elif blocked_reason:
        next_action = blocked_action_text or blocked_reason
    elif queue_entry_mode == "awaiting_confirmation":
        next_action = "review_queue_confirmation"
    else:
        next_action = "await_signal_trigger"

    if not review_mode:
        review_mode = "manual_review" if queue_entry_mode == "awaiting_confirmation" else "auto_queue"
    if not queue_entry_mode:
        queue_entry_mode = "awaiting_confirmation" if review_mode == "manual_review" else "inline_execute"

    return {
        "review_mode": review_mode,
        "queue_entry_mode": queue_entry_mode,
        "next_action": next_action,
    }


__all__ = ["_resolved_queue_review_policy"]
