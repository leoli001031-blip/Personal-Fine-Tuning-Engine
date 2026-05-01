"""Legacy auto-train trigger result formatting."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .status_legacy_auto_train_trigger_fields import LAST_RESULT_KEYS, append_scalar_parts


def append_trigger_note_lines(lines: list[str], auto_trigger: Mapping[str, Any], *, deps: Any) -> None:
    blocked_summary = auto_trigger.get("blocked_summary")
    if blocked_summary:
        lines.append(f"auto train trigger blocked summary: {deps.format_scalar(blocked_summary)}")

    summary = auto_trigger.get("last_result_summary")
    if summary:
        lines.append(f"auto train trigger summary: {deps.format_scalar(summary)}")


def append_trigger_last_result_lines(lines: list[str], auto_trigger: Mapping[str, Any], *, deps: Any) -> None:
    last_result = deps.coerce_mapping(auto_trigger.get("last_result"))
    if last_result is None:
        return
    last_parts: list[str] = []
    append_scalar_parts(last_parts, last_result, LAST_RESULT_KEYS, deps=deps)
    if last_parts:
        lines.append("auto train trigger last result: " + " | ".join(last_parts))


__all__ = ["append_trigger_last_result_lines", "append_trigger_note_lines"]
