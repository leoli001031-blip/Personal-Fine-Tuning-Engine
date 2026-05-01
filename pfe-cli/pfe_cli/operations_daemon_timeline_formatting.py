"""Worker daemon timeline summary formatting."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .operations_history_common import OperationsHistoryFormattingDeps


def format_daemon_timeline_summary(result: Any, *, deps: OperationsHistoryFormattingDeps) -> str:
    mapping = deps.coerce_mapping(result)
    if mapping is None:
        return deps.format_scalar(result)

    summary_parts: list[str] = []
    for key in (
        "count",
        "recovery_event_count",
        "last_event",
        "last_reason",
        "last_recovery_event",
        "last_recovery_reason",
        "last_recovery_note",
        "recent_anomaly_event",
        "recent_anomaly_reason",
        "latest_timestamp",
    ):
        value = mapping.get(key)
        if value is not None:
            summary_parts.append(f"{key}={deps.format_scalar(value)}")
    lines = ["daemon timeline: " + " | ".join(summary_parts) if summary_parts else "daemon timeline:"]

    recent_recovery_events = list(mapping.get("recent_recovery_events") or [])
    if recent_recovery_events:
        lines.append("  recent recovery events:")
        for item in recent_recovery_events[:3]:
            if not isinstance(item, Mapping):
                lines.append(f"    - {deps.format_scalar(item)}")
                continue
            parts: list[str] = []
            for key in ("timestamp", "event", "reason", "note"):
                value = item.get(key)
                if value is not None:
                    parts.append(f"{key}={deps.format_scalar(value)}")
            lines.append("    - " + " | ".join(parts) if parts else "    - " + deps.format_scalar(item))
    return "\n".join(lines)


__all__ = ["format_daemon_timeline_summary"]
