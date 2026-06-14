"""Legacy train queue summary formatting."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .status_legacy_queue_helpers import compact_item


def append_legacy_queue_summary_lines(lines: list[str], train_queue: Mapping[str, Any], *, deps: Any) -> None:
    queue_parts: list[str] = []
    for key in ("count", "max_priority"):
        value = train_queue.get(key)
        if value is not None:
            queue_parts.append(f"{key}={deps.format_scalar(value)}")
    counts = deps.coerce_mapping(train_queue.get("counts")) or {}
    if counts:
        queue_parts.append(
            "states="
            + ",".join(f"{name}:{deps.format_scalar(counts.get(name))}" for name in sorted(counts))
        )
    current_item = deps.coerce_mapping(train_queue.get("current"))
    if current_item:
        current_text = compact_item(current_item, ("job_id", "state"), deps=deps)
        if current_text:
            queue_parts.append("current=" + current_text)
    last_item = deps.coerce_mapping(train_queue.get("last_item"))
    if last_item:
        last_text = compact_item(last_item, ("job_id", "state", "adapter_version"), deps=deps)
        if last_text:
            queue_parts.append("last=" + last_text)
    if queue_parts:
        lines.append("train queue: " + " | ".join(queue_parts))


__all__ = ["append_legacy_queue_summary_lines"]
