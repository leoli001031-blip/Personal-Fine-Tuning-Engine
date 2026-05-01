"""Auto-train action status section for Matrix terminal output."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .matrix_formatting_common import _coerce_mapping
from .terminal_theme import draw_box, format_key_value


def append_auto_train_action_section(lines: list[str], mapping: Mapping[str, Any]) -> None:
    """Append auto-train action status box."""
    auto_trigger_action = _coerce_mapping(mapping.get("auto_train_trigger_action"))
    if auto_trigger_action:
        action_lines = []
        for key in (
            "action",
            "status",
            "reason",
            "triggered",
            "queue_job_id",
            "confirmation_reason",
            "approval_reason",
            "rejection_reason",
            "processed_count",
            "completed_count",
            "failed_count",
            "limit",
            "max_iterations",
            "max_cycles",
            "loop_cycles",
            "idle_rounds",
            "remaining_queued",
            "drained",
            "stopped_reason",
            "triggered_version",
            "promoted_version",
        ):
            value = auto_trigger_action.get(key)
            if value is not None:
                action_lines.append(format_key_value(key.replace("_", " "), value))
        if action_lines:
            lines.append(draw_box("AUTO TRAIN ACTION", action_lines))
            lines.append("")


__all__ = ["append_auto_train_action_section"]
