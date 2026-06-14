"""Worker daemon history list formatting."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .operations_history_common import OperationsHistoryFormattingDeps, history_latest_timestamp


def format_train_queue_daemon_history(result: Any, *, deps: OperationsHistoryFormattingDeps) -> str:
    mapping = deps.coerce_mapping(result)
    if mapping is None:
        return deps.format_scalar(result)

    lines = ["PFE worker daemon history"]
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
            for key in ("timestamp", "event", "reason", "note"):
                value = item.get(key)
                if value is not None:
                    parts.append(f"{key}={deps.format_scalar(value)}")
            lines.append("  - " + " | ".join(parts))
    else:
        lines.append("items: none")
    return "\n".join(lines)


__all__ = ["format_train_queue_daemon_history"]
