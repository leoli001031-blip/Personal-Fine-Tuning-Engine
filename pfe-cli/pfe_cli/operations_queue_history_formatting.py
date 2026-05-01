"""Train queue history formatting."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .operations_history_common import OperationsHistoryFormattingDeps, history_latest_timestamp


def format_train_queue_history(result: Any, *, deps: OperationsHistoryFormattingDeps) -> str:
    mapping = deps.coerce_mapping(result)
    if mapping is None:
        return deps.format_scalar(result)

    lines = ["PFE train queue history"]
    summary_parts: list[str] = []
    for key in ("workspace", "job_id", "state", "count", "history_count"):
        value = mapping.get(key)
        if value is not None:
            summary_parts.append(f"{key}={deps.format_scalar(value)}")
    if summary_parts:
        lines.append("summary: " + " | ".join(summary_parts))

    available_job_ids = list(mapping.get("available_job_ids") or [])
    if available_job_ids:
        lines.append("available jobs: " + ", ".join(str(item) for item in available_job_ids))

    history_summary = deps.coerce_mapping(mapping.get("history_summary")) or {}
    if history_summary:
        history_summary_parts: list[str] = []
        if history_summary.get("transition_count") is not None:
            history_summary_parts.append(f"transition_count={deps.format_scalar(history_summary.get('transition_count'))}")
        if history_summary.get("last_reason") is not None:
            history_summary_parts.append(f"last_reason={deps.format_scalar(history_summary.get('last_reason'))}")
        last_transition = deps.coerce_mapping(history_summary.get("last_transition")) or {}
        if last_transition.get("event") is not None:
            history_summary_parts.append(f"last_event={deps.format_scalar(last_transition.get('event'))}")
        if history_summary_parts:
            lines.append("history summary: " + " | ".join(history_summary_parts))

    history = list(mapping.get("history") or [])
    latest_timestamp = history_latest_timestamp(history, deps=deps)
    if latest_timestamp is not None:
        lines.append(f"latest timestamp: {latest_timestamp}")
    if history:
        lines.append("history:")
        for item in history:
            if not isinstance(item, Mapping):
                lines.append(f"  - {deps.format_scalar(item)}")
                continue
            parts: list[str] = []
            for key in ("timestamp", "event", "state", "reason", "note"):
                value = item.get(key)
                if value is not None:
                    parts.append(f"{key}={deps.format_scalar(value)}")
            lines.append("  - " + " | ".join(parts))
    else:
        lines.append("history: none")
    return "\n".join(lines)


__all__ = ["format_train_queue_history"]
