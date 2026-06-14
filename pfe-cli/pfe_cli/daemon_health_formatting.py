"""Daemon health status formatting."""

from __future__ import annotations

from typing import Any

from .daemon_formatting_deps import DaemonFormattingDeps


def format_daemon_health_status(result: Any, *, deps: DaemonFormattingDeps) -> str:
    """Format daemon health status for CLI output."""
    mapping = deps.coerce_mapping(result)
    if mapping is None:
        return deps.format_scalar(result)

    lines = ["PFE daemon health status"]

    overall = mapping.get("overall_health", "unknown")
    lines.append(f"overall: {overall}")

    issues = list(mapping.get("issues") or [])
    if issues:
        lines.append("issues:")
        for issue in issues:
            component = issue.get("component", "unknown")
            state = issue.get("state", "unknown")
            message = issue.get("message", "")
            lines.append(f"  - [{component}] {state}: {message}")
    else:
        lines.append("issues: none")

    daemon = mapping.get("daemon") or {}
    daemon_parts: list[str] = []
    for key in ("health_state", "lock_state", "heartbeat_state", "lease_state", "restart_policy_state"):
        value = daemon.get(key)
        if value is not None:
            daemon_parts.append(f"{key}={value}")
    if daemon_parts:
        lines.append("daemon: " + " | ".join(daemon_parts))

    runner = mapping.get("runner") or {}
    runner_parts: list[str] = []
    for key in ("lock_state", "active", "lease_expires_at"):
        value = runner.get(key)
        if value is not None:
            runner_parts.append(f"{key}={deps.format_scalar(value)}")
    if runner_parts:
        lines.append("runner: " + " | ".join(runner_parts))

    reliability = mapping.get("reliability") or {}
    if reliability:
        rel_parts: list[str] = []
        for key in ("active_runners", "stalled_jobs", "expired_leases"):
            value = reliability.get(key)
            if value is not None:
                rel_parts.append(f"{key}={value}")
        if rel_parts:
            lines.append("reliability: " + " | ".join(rel_parts))

        alerts = reliability.get("alerts_summary") or {}
        if alerts:
            total = alerts.get("total_active", 0)
            critical = alerts.get("critical_count", 0)
            error = alerts.get("error_count", 0)
            warning = alerts.get("warning_count", 0)
            if total > 0:
                lines.append(f"alerts: total={total} critical={critical} error={error} warning={warning}")

    checked_at = mapping.get("checked_at")
    if checked_at:
        lines.append(f"checked_at: {checked_at}")

    return "\n".join(lines)


__all__ = ["format_daemon_health_status"]
