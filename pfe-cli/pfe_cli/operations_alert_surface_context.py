"""Normalized inputs for operations alert surfaces."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .operations_alert_surface_resolution import surface_action, surface_focus
from .operations_formatting_deps import OperationsFormattingDeps


@dataclass(frozen=True)
class AlertSurfaceContext:
    alerts: list[dict[str, Any]]
    health: dict[str, Any]
    recovery: dict[str, Any]
    next_actions: list[Any]
    dashboard: dict[str, Any]
    alert_policy: dict[str, Any]
    overview: dict[str, Any]
    console: dict[str, Any]
    current_focus: Any
    required_action: Any
    summary_line: Any
    inspection_summary_line: Any

    @property
    def has_explicit_surface(self) -> bool:
        return bool(self.alerts or self.health or self.recovery or self.next_actions)


def build_alert_surface_context(
    *,
    operations_alerts: Any | None,
    operations_health: Any | None,
    operations_recovery: Any | None,
    operations_next_actions: Any | None,
    operations_dashboard: Mapping[str, Any] | None,
    operations_alert_policy: Mapping[str, Any] | None,
    operations_console: Mapping[str, Any] | None,
    operations_overview: Mapping[str, Any] | None,
    deps: OperationsFormattingDeps,
) -> AlertSurfaceContext:
    alerts = deps.coerce_sequence_of_mappings(operations_alerts)
    health = deps.coerce_mapping(operations_health) or {}
    recovery = deps.coerce_mapping(operations_recovery) or {}
    next_actions = deps.coerce_sequence_of_scalars(operations_next_actions)
    dashboard = deps.coerce_mapping(operations_dashboard) or {}
    alert_policy = deps.coerce_mapping(operations_alert_policy) or {}
    overview = deps.coerce_mapping(operations_overview) or {}
    console = deps.coerce_mapping(operations_console) or {}

    current_focus = surface_focus(
        alert_policy=alert_policy,
        console=console,
        dashboard=dashboard,
        overview=overview,
    )
    required_action = surface_action(
        alert_policy=alert_policy,
        console=console,
        dashboard=dashboard,
        overview=overview,
    )
    inspection_summary_line = (
        overview.get("inspection_summary_line")
        or dashboard.get("inspection_summary_line")
        or alert_policy.get("inspection_summary_line")
        or console.get("inspection_summary_line")
    )
    summary_line = (
        health.get("summary_line")
        or recovery.get("summary_line")
        or console.get("summary_line")
        or overview.get("summary_line")
    )
    summary_line, inspection_summary_line = deps.prefer_inspection_summary_for_generic_monitor(
        focus=current_focus,
        summary_line=summary_line,
        inspection_summary_line=inspection_summary_line,
    )
    return AlertSurfaceContext(
        alerts=alerts,
        health=health,
        recovery=recovery,
        next_actions=next_actions,
        dashboard=dashboard,
        alert_policy=alert_policy,
        overview=overview,
        console=console,
        current_focus=current_focus,
        required_action=required_action,
        summary_line=summary_line,
        inspection_summary_line=inspection_summary_line,
    )


__all__ = ["AlertSurfaceContext", "build_alert_surface_context"]
