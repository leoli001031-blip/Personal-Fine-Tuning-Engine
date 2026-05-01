"""Daemon stale-check formatting."""

from __future__ import annotations

from typing import Any

from .daemon_formatting_deps import DaemonFormattingDeps


def format_daemon_stale_check(result: Any, *, deps: DaemonFormattingDeps) -> str:
    """Format daemon stale check results for CLI output."""
    mapping = deps.coerce_mapping(result)
    if mapping is None:
        return deps.format_scalar(result)

    lines = ["PFE daemon stale check"]

    takeover = mapping.get("takeover_requested", False)
    lines.append(f"takeover_requested: {takeover}")

    daemon = mapping.get("daemon") or {}
    daemon_stale = daemon.get("is_stale", False)
    daemon_parts: list[str] = [f"is_stale={daemon_stale}"]
    for key in ("lock_state", "heartbeat_state", "can_recover"):
        value = daemon.get(key)
        if value is not None:
            daemon_parts.append(f"{key}={value}")
    lines.append("daemon: " + " | ".join(daemon_parts))

    runner = mapping.get("runner") or {}
    runner_stale = runner.get("is_stale", False)
    runner_parts: list[str] = [f"is_stale={runner_stale}"]
    for key in ("lock_state", "active"):
        value = runner.get(key)
        if value is not None:
            runner_parts.append(f"{key}={value}")
    lines.append("runner: " + " | ".join(runner_parts))

    actions = list(mapping.get("actions_taken") or [])
    if actions:
        lines.append("actions_taken:")
        for action in actions:
            component = action.get("component", "unknown")
            act = action.get("action", "unknown")
            result_status = action.get("result", "")
            note = action.get("note", "")
            if result_status:
                lines.append(f"  - [{component}] {act}: {result_status}")
            elif note:
                lines.append(f"  - [{component}] {act}: {note}")
            else:
                lines.append(f"  - [{component}] {act}")

    checked_at = mapping.get("checked_at")
    if checked_at:
        lines.append(f"checked_at: {checked_at}")

    return "\n".join(lines)


__all__ = ["format_daemon_stale_check"]
