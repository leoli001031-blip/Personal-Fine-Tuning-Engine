"""Queue, runner, and daemon console summary text."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .console_routing_deps import ConsoleRoutingDeps
from .console_routing_summary_helpers import append_mapping_parts, render_summary


def console_queue_summary_text(
    payload: Mapping[str, Any],
    *,
    deps: ConsoleRoutingDeps,
    history: Mapping[str, Any] | None = None,
) -> str:
    mapping = deps.coerce_mapping(payload) or {}
    queue_summary = deps.coerce_mapping(mapping.get("train_queue")) or {}
    queue_history = deps.coerce_mapping(history) or deps.coerce_mapping(mapping.get("train_queue")) or {}

    parts: list[str] = []
    append_mapping_parts(
        parts,
        queue_summary,
        (
            "count",
            "max_priority",
            "current_job_id",
            "awaiting_confirmation_count",
            "reviewed_transition_count",
            "approved_transition_count",
            "rejected_transition_count",
        ),
        deps=deps,
    )
    history_summary = deps.coerce_mapping(queue_history.get("history_summary")) or {}
    append_mapping_parts(parts, history_summary, ("transition_count", "last_reason"), deps=deps)
    last_transition = deps.coerce_mapping(history_summary.get("last_transition")) or {}
    if last_transition.get("event") is not None:
        parts.append(f"last_event={deps.format_scalar(last_transition.get('event'))}")
    return render_summary("PFE train queue summary", parts, fallback="count=0")


def console_runner_summary_text(
    payload: Mapping[str, Any],
    *,
    deps: ConsoleRoutingDeps,
    history: Mapping[str, Any] | None = None,
) -> str:
    mapping = deps.coerce_mapping(payload) or {}
    runner_summary = deps.coerce_mapping(mapping.get("train_queue_worker_runner")) or deps.coerce_mapping(
        mapping.get("worker_runner")
    ) or {}
    runner_history = deps.coerce_mapping(history) or deps.coerce_mapping(mapping.get("runner_timeline")) or {}

    parts: list[str] = []
    append_mapping_parts(
        parts,
        runner_summary,
        ("active", "lock_state", "stop_requested", "processed_count", "failed_count", "loop_cycles"),
        deps=deps,
    )
    append_mapping_parts(
        parts,
        runner_history,
        ("count", "last_event", "last_reason", "takeover_event_count", "current_lock_state"),
        deps=deps,
    )
    return render_summary("PFE worker runner summary", parts, fallback="state=idle")


def console_daemon_summary_text(result: Any, *, deps: ConsoleRoutingDeps) -> str:
    mapping = deps.coerce_mapping(result) or {}
    parts: list[str] = []
    append_mapping_parts(
        parts,
        mapping,
        (
            "workspace",
            "desired_state",
            "observed_state",
            "command_status",
            "lock_state",
            "health_state",
            "lease_state",
            "heartbeat_state",
            "recovery_action",
        ),
        deps=deps,
    )
    return render_summary("PFE worker daemon summary", parts, fallback="state=unknown")


__all__ = ["console_daemon_summary_text", "console_queue_summary_text", "console_runner_summary_text"]
