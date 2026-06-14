"""Worker runner history and timeline formatting."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .operations_history_common import OperationsHistoryFormattingDeps, history_latest_timestamp


def format_worker_runner_history(result: Any, *, deps: OperationsHistoryFormattingDeps) -> str:
    mapping = deps.coerce_mapping(result)
    if mapping is None:
        return deps.format_scalar(result)

    lines = ["PFE worker runner history"]
    summary_parts: list[str] = []
    for key in ("workspace", "count", "last_event", "last_reason"):
        value = mapping.get(key)
        if value is not None:
            summary_parts.append(f"{key}={deps.format_scalar(value)}")
    if summary_parts:
        lines.append("summary: " + " | ".join(summary_parts))

    items = list(mapping.get("items") or [])
    latest_timestamp = history_latest_timestamp(items, deps=deps)
    if latest_timestamp is not None:
        lines.append(f"latest timestamp: {latest_timestamp}")
    if items:
        lines.append("items:")
        for item in items:
            if not isinstance(item, Mapping):
                lines.append(f"  - {deps.format_scalar(item)}")
                continue
            parts: list[str] = []
            for key in ("timestamp", "event", "reason"):
                value = item.get(key)
                if value is not None:
                    parts.append(f"{key}={deps.format_scalar(value)}")
            metadata = deps.coerce_mapping(item.get("metadata")) or {}
            for key in ("pid", "takeover", "previous_pid", "processed_count", "failed_count"):
                value = metadata.get(key)
                if value is not None:
                    parts.append(f"{key}={deps.format_scalar(value)}")
            lines.append("  - " + " | ".join(parts))
        else:
            lines.append("items: none")
    return "\n".join(lines)


def format_runner_timeline_summary(result: Any, *, deps: OperationsHistoryFormattingDeps) -> str:
    mapping = deps.coerce_mapping(result)
    if mapping is None:
        return deps.format_scalar(result)

    summary_parts: list[str] = []
    for key in (
        "count",
        "last_event",
        "last_reason",
        "takeover_event_count",
        "last_takeover_event",
        "last_takeover_reason",
        "current_active",
        "current_lock_state",
        "current_stop_requested",
        "current_lease_expires_at",
        "recent_anomaly_reason",
        "latest_timestamp",
    ):
        value = mapping.get(key)
        if value is not None:
            summary_parts.append(f"{key}={deps.format_scalar(value)}")
    lines = ["runner timeline: " + " | ".join(summary_parts) if summary_parts else "runner timeline:"]

    recent_events = list(mapping.get("recent_events") or [])
    if recent_events:
        lines.append("  recent events:")
        for item in recent_events[:3]:
            if not isinstance(item, Mapping):
                lines.append(f"    - {deps.format_scalar(item)}")
                continue
            parts: list[str] = []
            for key in ("timestamp", "event", "reason"):
                value = item.get(key)
                if value is not None:
                    parts.append(f"{key}={deps.format_scalar(value)}")
            lines.append("    - " + " | ".join(parts) if parts else "    - " + deps.format_scalar(item))
    recent_takeover_events = list(mapping.get("recent_takeover_events") or [])
    if recent_takeover_events:
        lines.append("  recent takeover events:")
        for item in recent_takeover_events[:3]:
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


__all__ = ["format_runner_timeline_summary", "format_worker_runner_history"]
