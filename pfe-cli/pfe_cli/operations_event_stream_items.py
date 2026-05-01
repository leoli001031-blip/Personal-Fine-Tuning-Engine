"""Recent item formatting for operations event streams."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .operations_formatting_deps import OperationsFormattingDeps


def append_event_stream_items(
    lines: list[str],
    mapping: Mapping[str, Any],
    *,
    deps: OperationsFormattingDeps,
) -> None:
    items = list(mapping.get("items") or [])
    if not items:
        return

    lines.append("  recent events:")
    for item in items[:5]:
        if not isinstance(item, Mapping):
            lines.append(f"    - {deps.format_scalar(item)}")
            continue
        parts = _event_stream_item_parts(item, deps=deps)
        lines.append("    - " + " | ".join(parts))


def _event_stream_item_parts(item: Mapping[str, Any], *, deps: OperationsFormattingDeps) -> list[str]:
    parts: list[str] = []
    for key in (
        "timestamp",
        "source",
        "event",
        "reason",
        "severity",
        "attention",
        "status",
        "version",
        "job_id",
        "note",
        "message",
    ):
        value = item.get(key)
        if value is not None:
            parts.append(f"{key}={deps.format_scalar(value)}")
    return parts


__all__ = ["append_event_stream_items"]
