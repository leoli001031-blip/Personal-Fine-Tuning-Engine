"""Operations event stream surface formatter."""

from __future__ import annotations

from typing import Any

from .operations_event_stream_items import append_event_stream_items
from .operations_event_stream_parts import append_event_stream_dashboard, event_stream_summary_parts
from .operations_formatting_deps import OperationsFormattingDeps


def format_operations_event_stream(
    result: Any,
    *,
    deps: OperationsFormattingDeps,
) -> list[str] | None:
    mapping = deps.coerce_mapping(result)
    if not mapping:
        return None

    lines = ["operations event stream:"]
    summary_parts = event_stream_summary_parts(mapping, deps=deps)
    if summary_parts:
        lines.append("  " + " | ".join(summary_parts))
    next_actions = mapping.get("next_actions")
    if next_actions:
        lines.append("  next_actions=" + deps.format_scalar(next_actions))
    append_event_stream_dashboard(lines, mapping, deps=deps)
    append_event_stream_items(lines, mapping, deps=deps)
    return lines


__all__ = ["format_operations_event_stream"]
