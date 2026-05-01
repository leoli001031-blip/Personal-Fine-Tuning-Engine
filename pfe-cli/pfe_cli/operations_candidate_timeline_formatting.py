"""Candidate timeline formatting."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .operations_history_common import OperationsHistoryFormattingDeps, history_latest_timestamp


def candidate_timeline_stage(item: Mapping[str, Any] | None, *, deps: OperationsHistoryFormattingDeps) -> str | None:
    if item is None:
        return None

    stage = item.get("stage")
    if stage is not None:
        return deps.format_scalar(stage)

    action = str(item.get("action") or "")
    status = str(item.get("status") or "")
    if action == "promote_candidate" and status == "completed":
        return "promoted"
    if action == "archive_candidate" and status == "completed":
        return "archived"
    if status == "blocked":
        return "blocked"
    if status == "noop":
        return "noop"
    return "candidate_action"


def format_candidate_timeline_item(item: Any, *, index: int, deps: OperationsHistoryFormattingDeps) -> str:
    if not isinstance(item, Mapping):
        return f"  - {index}. {deps.format_scalar(item)}"

    action = str(item.get("action") or "")
    status = str(item.get("status") or "")
    label = item.get("label") or f"{action}:{status}"

    parts: list[str] = []
    for key, value in (
        ("timestamp", item.get("timestamp")),
        ("stage", item.get("stage")),
        ("label", label if label else None),
        ("action", action if action else None),
        ("status", status if status else None),
        ("reason", item.get("reason")),
        ("operator_note", item.get("operator_note")),
        ("candidate_version", item.get("candidate_version")),
        ("promoted_version", item.get("promoted_version")),
        ("archived_version", item.get("archived_version")),
    ):
        if value is not None:
            parts.append(f"{key}={deps.format_scalar(value)}")

    if not parts:
        return f"  - {index}. {deps.format_scalar(item)}"
    return f"  - {index}. " + " | ".join(parts)


def format_candidate_timeline(result: Any, *, deps: OperationsHistoryFormattingDeps) -> str:
    mapping = deps.coerce_mapping(result)
    if mapping is None:
        return deps.format_scalar(result)

    lines = ["PFE candidate timeline"]
    summary_parts: list[str] = []
    for key in ("workspace", "count", "limit", "current_stage", "transition_count", "last_reason", "last_candidate_version"):
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
        lines.append("timeline:")
        for index, item in enumerate(items, 1):
            if isinstance(item, Mapping) and "stage" not in item:
                item = {**dict(item), "stage": candidate_timeline_stage(item, deps=deps)}
            lines.append(format_candidate_timeline_item(item, index=index, deps=deps))
    else:
        lines.append("timeline: none")
    return "\n".join(lines)


__all__ = [
    "candidate_timeline_stage",
    "format_candidate_timeline",
    "format_candidate_timeline_item",
]
