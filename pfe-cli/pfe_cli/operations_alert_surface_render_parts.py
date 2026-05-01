"""Text sections for operations alert surface rendering."""

from __future__ import annotations

from typing import Any

from .operations_formatting_deps import OperationsFormattingDeps


def append_alert_summary(lines: list[str], alert_surface: dict[str, Any], deps: OperationsFormattingDeps) -> None:
    alert_parts: list[str] = []
    if alert_surface.get("attention_needed") is not None:
        alert_parts.append(f"attention_needed={deps.format_scalar(alert_surface.get('attention_needed'))}")
    if alert_surface.get("current_focus") is not None:
        alert_parts.append(f"current_focus={deps.format_scalar(alert_surface.get('current_focus'))}")
    if alert_surface.get("required_action") is not None:
        alert_parts.append(f"required_action={deps.format_scalar(alert_surface.get('required_action'))}")
    summary_line = alert_surface.get("summary_line")
    if summary_line:
        alert_parts.append(f"summary={deps.format_scalar(summary_line)}")
    inspection_summary_line = alert_surface.get("inspection_summary_line")
    if inspection_summary_line and inspection_summary_line != summary_line:
        alert_parts.append(f"inspection_summary={deps.format_scalar(inspection_summary_line)}")
    if alert_parts:
        lines.append("  " + " | ".join(alert_parts))


def append_alert_items(lines: list[str], alert_surface: dict[str, Any], deps: OperationsFormattingDeps) -> None:
    alerts = deps.coerce_sequence_of_mappings(alert_surface.get("alerts"))
    if not alerts:
        return

    lines.append("  alerts:")
    for alert in alerts:
        parts: list[str] = []
        for key in ("reason", "detail", "candidate_stage", "queue_count", "runner_lock_state", "severity"):
            value = alert.get(key)
            if value is not None:
                parts.append(f"{key}={deps.format_scalar(value)}")
        lines.append("    - " + " | ".join(parts) if parts else "    - " + deps.format_scalar(alert))


def append_alert_health(lines: list[str], alert_surface: dict[str, Any], deps: OperationsFormattingDeps) -> None:
    health = deps.coerce_mapping(alert_surface.get("health")) or {}
    if not health:
        return

    parts: list[str] = []
    for key in (
        "status",
        "daemon_lock_state",
        "health_state",
        "daemon_health_state",
        "lease_state",
        "daemon_lease_state",
        "heartbeat_state",
        "daemon_heartbeat_state",
        "restart_policy_state",
        "daemon_restart_policy_state",
        "recovery_action",
        "daemon_recovery_action",
        "runner_lock_state",
        "candidate_state",
        "queue_state",
    ):
        value = health.get(key)
        if value is not None:
            parts.append(f"{key}={deps.format_scalar(value)}")
    if parts:
        lines.append("  health: " + " | ".join(parts))


def append_alert_recovery(lines: list[str], alert_surface: dict[str, Any], deps: OperationsFormattingDeps) -> None:
    recovery = deps.coerce_mapping(alert_surface.get("recovery")) or {}
    if not recovery:
        return

    parts: list[str] = []
    for key in (
        "daemon_recovery_needed",
        "daemon_recovery_reason",
        "daemon_recovery_state",
        "daemon_recovery_action",
        "recovery_needed",
        "recovery_reason",
    ):
        value = recovery.get(key)
        if value is not None:
            parts.append(f"{key}={deps.format_scalar(value)}")
    if parts:
        lines.append("  recovery: " + " | ".join(parts))


def append_alert_next_actions(lines: list[str], alert_surface: dict[str, Any], deps: OperationsFormattingDeps) -> None:
    next_actions = deps.coerce_sequence_of_scalars(alert_surface.get("next_actions"))
    if next_actions:
        lines.append("  next actions: " + ", ".join(deps.format_scalar(action) for action in next_actions))


__all__ = [
    "append_alert_health",
    "append_alert_items",
    "append_alert_next_actions",
    "append_alert_recovery",
    "append_alert_summary",
]
