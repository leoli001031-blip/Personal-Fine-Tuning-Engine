"""Section builders for derived operations console digests."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

def build_candidate_section(
    *,
    candidate_summary: Mapping[str, Any],
    candidate_history: Mapping[str, Any],
    candidate_timeline: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "current_stage": candidate_timeline.get("current_stage") or candidate_summary.get("candidate_state"),
        "last_candidate_version": candidate_timeline.get("last_candidate_version")
        or candidate_history.get("last_candidate_version")
        or candidate_summary.get("candidate_version"),
        "last_reason": candidate_timeline.get("last_reason") or candidate_history.get("last_reason"),
        "latest_timestamp": candidate_timeline.get("latest_timestamp") or candidate_history.get("latest_timestamp"),
        "transition_count": candidate_timeline.get("transition_count") or candidate_history.get("count"),
        "history_count": candidate_history.get("count") or candidate_timeline.get("history_count"),
    }


def build_queue_section(
    *,
    train_queue: Mapping[str, Any],
    queue_history: Mapping[str, Any],
    queue_review: Mapping[str, Any],
    queue_confirm: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "count": train_queue.get("count"),
        "awaiting_confirmation_count": queue_confirm.get("awaiting_confirmation_count"),
        "next_confirmation_reason": queue_confirm.get("next_confirmation_reason"),
        "last_transition": queue_history.get("last_transition"),
        "last_reason": queue_history.get("last_reason"),
        "reviewed_transition_count": queue_review.get("reviewed_transition_count"),
        "last_review_event": queue_review.get("last_review_event"),
        "last_review_note": queue_review.get("last_review_note"),
    }


def build_runner_section(*, worker: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "active": worker.get("active"),
        "lock_state": worker.get("lock_state"),
        "last_event": worker.get("last_event"),
        "last_event_reason": worker.get("last_event_reason"),
        "lease_expires_at": worker.get("lease_expires_at"),
        "history_count": worker.get("history_count"),
    }


def build_daemon_section(*, daemon_timeline: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "count": daemon_timeline.get("count"),
        "recovery_event_count": daemon_timeline.get("recovery_event_count"),
        "last_event": daemon_timeline.get("last_event"),
        "last_reason": daemon_timeline.get("last_reason"),
        "last_recovery_event": daemon_timeline.get("last_recovery_event"),
        "last_recovery_reason": daemon_timeline.get("last_recovery_reason"),
        "last_recovery_note": daemon_timeline.get("last_recovery_note"),
        "recent_anomaly_reason": daemon_timeline.get("recent_anomaly_reason"),
        "latest_timestamp": daemon_timeline.get("latest_timestamp"),
    }


def build_runner_timeline_section(*, runner_timeline: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "count": runner_timeline.get("count"),
        "last_event": runner_timeline.get("last_event"),
        "last_reason": runner_timeline.get("last_reason"),
        "takeover_event_count": runner_timeline.get("takeover_event_count"),
        "last_takeover_event": runner_timeline.get("last_takeover_event"),
        "last_takeover_reason": runner_timeline.get("last_takeover_reason"),
        "recent_anomaly_reason": runner_timeline.get("recent_anomaly_reason"),
        "latest_timestamp": runner_timeline.get("latest_timestamp"),
    }


__all__ = [
    "build_candidate_section",
    "build_daemon_section",
    "build_queue_section",
    "build_runner_section",
    "build_runner_timeline_section",
]
