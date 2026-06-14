"""Primary key/value rendering for operations console digest sections."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .operations_formatting_deps import OperationsFormattingDeps


def render_primary_section_parts(
    label: str,
    section: Mapping[str, Any],
    *,
    deps: OperationsFormattingDeps,
) -> list[str]:
    if label == "candidate":
        return _render_keys(
            section,
            (
                "current_stage",
                "last_candidate_version",
                "last_reason",
                "latest_timestamp",
                "transition_count",
                "history_count",
            ),
            deps=deps,
        )
    if label == "queue":
        return _render_queue_section(section, deps=deps)
    if label in {"runner", "daemon"}:
        return _render_keys(
            section,
            (
                "active",
                "lock_state",
                "health_state",
                "lease_state",
                "heartbeat_state",
                "restart_policy_state",
                "recovery_action",
                "last_event",
                "last_event_reason",
                "lease_expires_at",
                "history_count",
                "recovery_needed",
                "can_recover",
                "recovery_reason",
                "recovery_state",
                "recovery_event_count",
                "last_recovery_event",
                "last_recovery_reason",
                "last_recovery_note",
                "recent_anomaly_reason",
            ),
            deps=deps,
        )
    if label == "runner_timeline":
        return _render_keys(
            section,
            (
                "count",
                "last_event",
                "last_reason",
                "takeover_event_count",
                "last_takeover_event",
                "last_takeover_reason",
                "recent_anomaly_reason",
                "latest_timestamp",
            ),
            deps=deps,
        )
    return []


def _render_queue_section(section: Mapping[str, Any], *, deps: OperationsFormattingDeps) -> list[str]:
    section_parts = _render_keys(
        section,
        (
            "count",
            "awaiting_confirmation_count",
            "next_confirmation_reason",
            "last_reason",
            "reviewed_transition_count",
            "last_review_event",
            "last_review_note",
        ),
        deps=deps,
    )
    last_transition = deps.coerce_mapping(section.get("last_transition"))
    if last_transition:
        transition_parts: list[str] = []
        for key in ("job_id", "event", "state"):
            value = last_transition.get(key)
            if value is not None:
                transition_parts.append(deps.format_scalar(value))
        if transition_parts:
            section_parts.append("last_transition=" + ",".join(transition_parts))
    return section_parts


def _render_keys(section: Mapping[str, Any], keys: tuple[str, ...], *, deps: OperationsFormattingDeps) -> list[str]:
    parts: list[str] = []
    for key in keys:
        value = section.get(key)
        if value is not None:
            parts.append(f"{key}={deps.format_scalar(value)}")
    return parts


__all__ = ["render_primary_section_parts"]
