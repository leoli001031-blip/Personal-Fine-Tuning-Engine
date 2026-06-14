"""Helpers for existing operations console digest payloads."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def augment_console_with_timelines(
    console: Mapping[str, Any],
    *,
    daemon_timeline: Mapping[str, Any],
    runner_timeline: Mapping[str, Any],
) -> dict[str, Any]:
    """Add compact timeline snapshots to a provided console digest."""
    enriched = dict(console)
    if daemon_timeline and "daemon" not in enriched:
        enriched["daemon"] = {
            "count": daemon_timeline.get("count"),
            "recovery_event_count": daemon_timeline.get("recovery_event_count"),
            "last_event": daemon_timeline.get("last_event"),
            "last_reason": daemon_timeline.get("last_reason"),
            "last_recovery_event": daemon_timeline.get("last_recovery_event"),
            "last_recovery_reason": daemon_timeline.get("last_recovery_reason"),
            "last_recovery_note": daemon_timeline.get("last_recovery_note"),
            "latest_timestamp": daemon_timeline.get("latest_timestamp"),
        }
    if runner_timeline and "runner_timeline" not in enriched:
        enriched["runner_timeline"] = {
            "count": runner_timeline.get("count"),
            "last_event": runner_timeline.get("last_event"),
            "last_reason": runner_timeline.get("last_reason"),
            "current_active": runner_timeline.get("current_active"),
            "current_lock_state": runner_timeline.get("current_lock_state"),
            "current_stop_requested": runner_timeline.get("current_stop_requested"),
            "current_lease_expires_at": runner_timeline.get("current_lease_expires_at"),
            "latest_timestamp": runner_timeline.get("latest_timestamp"),
        }
    return enriched


__all__ = ["augment_console_with_timelines"]
