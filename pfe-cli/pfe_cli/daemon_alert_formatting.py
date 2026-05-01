"""Daemon alert formatting."""

from __future__ import annotations

from typing import Any

from .daemon_formatting_deps import DaemonFormattingDeps


def format_daemon_alerts(result: Any, *, deps: DaemonFormattingDeps) -> str:
    """Format daemon alerts for CLI output."""
    mapping = deps.coerce_mapping(result)
    if mapping is None:
        return deps.format_scalar(result)

    lines = ["PFE daemon alerts"]

    filters = mapping.get("filters") or {}
    filter_parts: list[str] = []
    for key in ("level", "scope"):
        value = filters.get(key)
        if value:
            filter_parts.append(f"{key}={value}")
    if filter_parts:
        lines.append("filters: " + " | ".join(filter_parts))

    count = mapping.get("count", 0)
    lines.append(f"count: {count}")

    alerts = list(mapping.get("alerts") or [])
    if alerts:
        lines.append("alerts:")
        for alert in alerts:
            level = alert.get("level", "unknown")
            scope = alert.get("scope", "unknown")
            reason = alert.get("reason", "unknown")
            message = alert.get("message", "")
            timestamp = alert.get("timestamp", "")

            lines.append(f"  - [{level}] {scope}: {reason}")
            if message:
                lines.append(f"    message: {message}")
            if timestamp:
                lines.append(f"    timestamp: {timestamp}")
    else:
        lines.append("alerts: none")

    summary = mapping.get("summary") or {}
    if summary:
        total = summary.get("total_active", 0)
        critical = summary.get("critical_count", 0)
        error = summary.get("error_count", 0)
        warning = summary.get("warning_count", 0)
        lines.append(f"summary: total_active={total} critical={critical} error={error} warning={warning}")

    checked_at = mapping.get("checked_at")
    if checked_at:
        lines.append(f"checked_at: {checked_at}")

    return "\n".join(lines)


__all__ = ["format_daemon_alerts"]
