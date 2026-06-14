"""Legacy plain-text operations status formatting."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .status_legacy_operations_overview import append_legacy_operations_overview_lines
from .status_legacy_operations_sections import append_legacy_operations_detail_lines


def append_legacy_operations_surface_lines(
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
    """Append legacy operations overview and surface formatter lines."""
    append_legacy_operations_overview_lines(
        lines,
        operations_overview=operations_overview,
        operations_dashboard=operations_dashboard,
        operations_alert_policy=operations_alert_policy,
        deps=deps,
    )
    append_legacy_operations_detail_lines(
        lines,
        operations_overview=operations_overview,
        operations_alerts=operations_alerts,
        operations_health=operations_health,
        operations_recovery=operations_recovery,
        operations_next_actions=operations_next_actions,
        operations_dashboard=operations_dashboard,
        operations_alert_policy=operations_alert_policy,
        operations_console=operations_console,
        operations_event_stream=operations_event_stream,
        operations_timeline=operations_timeline,
        candidate_summary=candidate_summary,
        candidate_history=candidate_history,
        candidate_timeline=candidate_timeline,
        daemon_timeline=daemon_timeline,
        runner_timeline=runner_timeline,
        train_queue=train_queue,
        deps=deps,
    )


__all__ = ["append_legacy_operations_surface_lines"]
