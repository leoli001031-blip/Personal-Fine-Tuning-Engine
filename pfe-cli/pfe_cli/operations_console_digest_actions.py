"""Derive next actions for operations console digests."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def build_derived_next_actions(
    *,
    overview: Mapping[str, Any],
    candidate_summary: Mapping[str, Any],
    queue_confirm: Mapping[str, Any],
    worker: Mapping[str, Any],
) -> list[str]:
    """Derive fallback next actions from operations state."""
    actions: list[str] = []
    attention_reason = overview.get("attention_reason")
    candidate_needs_promotion = bool(candidate_summary.get("candidate_needs_promotion"))
    if attention_reason == "awaiting_confirmation":
        actions.append("review_queue_confirmation")
    elif attention_reason == "candidate_ready_for_promotion" or candidate_needs_promotion:
        actions.append("review_candidate_promotion")
    elif attention_reason:
        actions.append(str(attention_reason))

    if str(worker.get("lock_state") or "") == "stale":
        actions.append("inspect_worker_stale_lock")
    if bool(worker.get("active")) and bool(worker.get("stop_requested")):
        actions.append("wait_for_runner_shutdown")
    if int(queue_confirm.get("awaiting_confirmation_count", 0) or 0) > 0:
        if "review_queue_confirmation" not in actions:
            actions.append("review_queue_confirmation")
    return actions


__all__ = ["build_derived_next_actions"]
