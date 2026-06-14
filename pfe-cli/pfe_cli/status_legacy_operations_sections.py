"""Legacy operations detail surface formatting."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .status_legacy_operations_section_helpers import (
    append_lines,
    append_split_lines,
    operations_alert_policy_lines,
)


def append_legacy_operations_detail_lines(
    lines: list[str],
    *,
    operations_overview: Mapping[str, Any] | None,
    operations_alerts: list[dict[str, Any]],
    operations_health: Mapping[str, Any] | None,
    operations_recovery: Mapping[str, Any] | None,
    operations_next_actions: list[str],
    operations_dashboard: Mapping[str, Any] | None,
    operations_alert_policy: Mapping[str, Any] | None,
    operations_console: Mapping[str, Any] | None,
    operations_event_stream: Mapping[str, Any] | None,
    operations_timeline: Mapping[str, Any] | None,
    candidate_summary: Mapping[str, Any] | None,
    candidate_history: Mapping[str, Any] | None,
    candidate_timeline: Mapping[str, Any] | None,
    daemon_timeline: Mapping[str, Any] | None,
    runner_timeline: Mapping[str, Any] | None,
    train_queue: Mapping[str, Any] | None,
    deps: Any,
) -> None:
    append_lines(
        lines,
        deps.format_operations_alert_surface(
            {
                "operations_alerts": operations_alerts,
                "operations_health": operations_health,
                "operations_recovery": operations_recovery,
                "operations_next_actions": operations_next_actions,
                "operations_dashboard": operations_dashboard,
                "operations_alert_policy": operations_alert_policy,
                "operations_console": operations_console,
                "operations_overview": operations_overview,
            }
        ),
    )
    append_lines(
        lines,
        deps.format_operations_console_digest(
            {
                "operations_console": operations_console,
                "operations_overview": operations_overview,
                "candidate_summary": candidate_summary,
                "candidate_history": candidate_history,
                "candidate_timeline": candidate_timeline,
                "daemon_timeline": daemon_timeline,
                "runner_timeline": runner_timeline,
                "train_queue": train_queue,
            }
        ),
    )
    append_lines(
        lines,
        deps.format_operations_dashboard(operations_dashboard) if operations_dashboard is not None else None,
    )
    append_lines(
        lines,
        operations_alert_policy_lines(
            operations_dashboard=operations_dashboard,
            operations_alert_policy=operations_alert_policy,
            deps=deps,
        ),
    )
    append_lines(
        lines,
        deps.format_operations_event_stream(operations_event_stream)
        if operations_event_stream is not None
        else None,
    )
    append_lines(
        lines,
        deps.format_operations_timeline(operations_timeline) if operations_timeline is not None else None,
    )
    append_split_lines(
        lines,
        deps.format_runner_timeline_summary(runner_timeline) if runner_timeline is not None else None,
    )
    append_split_lines(
        lines,
        deps.format_daemon_timeline_summary(daemon_timeline) if daemon_timeline is not None else None,
    )


__all__ = ["append_legacy_operations_detail_lines"]
