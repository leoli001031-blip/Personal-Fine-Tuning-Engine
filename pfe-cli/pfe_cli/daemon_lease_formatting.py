"""Daemon lease formatting."""

from __future__ import annotations

from typing import Any

from .daemon_formatting_deps import DaemonFormattingDeps


def format_daemon_lease_status(result: Any, *, deps: DaemonFormattingDeps) -> str:
    """Format daemon lease status for CLI output."""
    mapping = deps.coerce_mapping(result)
    if mapping is None:
        return deps.format_scalar(result)

    lines = ["PFE daemon lease status"]

    daemon_lease = mapping.get("daemon_lease") or {}
    daemon_parts: list[str] = []
    for key in ("lease_state", "heartbeat_state"):
        value = daemon_lease.get(key)
        if value is not None:
            daemon_parts.append(f"{key}={value}")
    if daemon_parts:
        lines.append("daemon: " + " | ".join(daemon_parts))

    expires = daemon_lease.get("lease_expires_at")
    if expires:
        lines.append(f"  lease_expires: {expires}")

    runner_lease = mapping.get("runner_lease") or {}
    runner_parts: list[str] = []
    for key in ("lock_state", "stale_after_seconds"):
        value = runner_lease.get(key)
        if value is not None:
            runner_parts.append(f"{key}={deps.format_scalar(value)}")
    if runner_parts:
        lines.append("runner: " + " | ".join(runner_parts))

    runner_expires = runner_lease.get("lease_expires_at")
    if runner_expires:
        lines.append(f"  lease_expires: {runner_expires}")

    expired_count = mapping.get("expired_leases_count", 0)
    lines.append(f"expired_leases: {expired_count}")

    expired = list(mapping.get("expired_leases") or [])
    if expired:
        lines.append("recent expired:")
        for lease in expired[:5]:
            lid = lease.get("lease_id", "unknown")[:8]
            job = lease.get("job_id", "unknown")[:8]
            state = lease.get("state", "unknown")
            lines.append(f"  - {lid}... job={job}... state={state}")

    checked_at = mapping.get("checked_at")
    if checked_at:
        lines.append(f"checked_at: {checked_at}")

    return "\n".join(lines)


__all__ = ["format_daemon_lease_status"]
