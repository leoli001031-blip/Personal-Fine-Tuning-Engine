"""Daemon heartbeat formatting."""

from __future__ import annotations

from typing import Any

from .daemon_formatting_deps import DaemonFormattingDeps


def format_daemon_heartbeat_status(result: Any, *, deps: DaemonFormattingDeps) -> str:
    """Format daemon heartbeat status for CLI output."""
    mapping = deps.coerce_mapping(result)
    if mapping is None:
        return deps.format_scalar(result)

    lines = ["PFE daemon heartbeat status"]

    daemon = mapping.get("daemon") or {}
    daemon_parts: list[str] = []
    for key in ("heartbeat_state", "heartbeat_age_seconds", "lease_state"):
        value = daemon.get(key)
        if value is not None:
            daemon_parts.append(f"{key}={deps.format_scalar(value)}")
    if daemon_parts:
        lines.append("daemon: " + " | ".join(daemon_parts))

    last_hb = daemon.get("last_heartbeat_at")
    if last_hb:
        lines.append(f"  last_heartbeat: {last_hb}")
    lease_expires = daemon.get("lease_expires_at")
    if lease_expires:
        lines.append(f"  lease_expires: {lease_expires}")

    runner = mapping.get("runner") or {}
    runner_parts: list[str] = []
    for key in ("heartbeat_age_seconds", "stale_after_seconds"):
        value = runner.get(key)
        if value is not None:
            runner_parts.append(f"{key}={deps.format_scalar(value)}")
    if runner_parts:
        lines.append("runner: " + " | ".join(runner_parts))

    runner_hb = runner.get("last_heartbeat_at")
    if runner_hb:
        lines.append(f"  last_heartbeat: {runner_hb}")
    runner_lease = runner.get("lease_expires_at")
    if runner_lease:
        lines.append(f"  lease_expires: {runner_lease}")

    checked_at = mapping.get("checked_at")
    if checked_at:
        lines.append(f"checked_at: {checked_at}")

    return "\n".join(lines)


__all__ = ["format_daemon_heartbeat_status"]
