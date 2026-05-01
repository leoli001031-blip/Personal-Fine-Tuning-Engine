"""Worker daemon status formatting."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .operations_history_common import OperationsHistoryFormattingDeps


def format_train_queue_daemon_status(result: Any, *, deps: OperationsHistoryFormattingDeps) -> str:
    mapping = deps.coerce_mapping(result)
    if mapping is None:
        return deps.format_scalar(result)

    lines = ["PFE worker daemon"]
    summary_parts: list[str] = []
    for key in ("workspace", "desired_state", "requested_action", "command_status", "last_event", "last_reason"):
        value = mapping.get(key)
        if value is not None:
            summary_parts.append(f"{key}={deps.format_scalar(value)}")
    if summary_parts:
        lines.append("summary: " + " | ".join(summary_parts))

    state_parts: list[str] = []
    for key in (
        "active",
        "observed_state",
        "lock_state",
        "recovery_state",
        "restart_attempts",
        "auto_restart_enabled",
        "auto_recover_enabled",
        "heartbeat_interval_seconds",
        "lease_timeout_seconds",
        "next_restart_after",
        "last_requested_by",
        "last_requested_at",
        "history_count",
        "auto_recovery_count",
    ):
        value = mapping.get(key)
        if value is not None:
            state_parts.append(f"{key}={deps.format_scalar(value)}")
    if state_parts:
        lines.append("state: " + " | ".join(state_parts))

    state_path = mapping.get("state_path")
    if state_path is not None:
        lines.append(f"state path: {deps.format_scalar(state_path)}")

    history = list(mapping.get("history") or [])
    if history:
        lines.append("history:")
        for item in history:
            if not isinstance(item, Mapping):
                lines.append(f"  - {deps.format_scalar(item)}")
                continue
            parts: list[str] = []
            for key in ("timestamp", "event", "reason", "note"):
                value = item.get(key)
                if value is not None:
                    parts.append(f"{key}={deps.format_scalar(value)}")
            lines.append("  - " + " | ".join(parts))
    else:
        lines.append("history: none")
    return "\n".join(lines)


__all__ = ["format_train_queue_daemon_status"]
