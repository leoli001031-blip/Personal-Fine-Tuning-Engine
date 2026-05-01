"""Legacy auto-train trigger status formatting."""

from __future__ import annotations

from typing import Any

from .status_legacy_auto_train_trigger_gate import (
    append_trigger_policy_lines,
    append_trigger_threshold_lines,
)
from .status_legacy_auto_train_trigger_result import (
    append_trigger_last_result_lines,
    append_trigger_note_lines,
)
from .status_legacy_auto_train_trigger_summary import append_trigger_summary_lines


def append_legacy_auto_train_trigger_lines(
    lines: list[str],
    mapping: dict[str, Any],
    *,
    deps: Any,
) -> None:
    auto_trigger = deps.coerce_mapping(mapping.pop("auto_train_trigger", None))
    if auto_trigger is None:
        return

    append_trigger_summary_lines(lines, auto_trigger, deps=deps)
    append_trigger_policy_lines(lines, auto_trigger, deps=deps)
    append_trigger_threshold_lines(lines, auto_trigger, deps=deps)
    append_trigger_note_lines(lines, auto_trigger, deps=deps)
    append_trigger_last_result_lines(lines, auto_trigger, deps=deps)


__all__ = ["append_legacy_auto_train_trigger_lines"]
