"""Operations overview status sections for Matrix terminal output."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .matrix_formatting_common import _coerce_mapping, _format_scalar
from .terminal_theme import MatrixColors, draw_box, format_key_value, status_badge


def append_operations_overview_sections(lines: list[str], mapping: Mapping[str, Any]) -> None:
    """Append operations overview, dashboard, and alert policy boxes."""
    operations = _coerce_mapping(mapping.get("operations_overview"))
    if operations:
        ops_content = []

        attention = operations.get("attention_needed", False)
        if attention:
            reason = operations.get("attention_reason", "unknown")
            ops_content.append(f"{MatrixColors.AMBER}⚠ ATTENTION REQUIRED: {reason}{MatrixColors.RESET}")

        trigger_state = operations.get("trigger_state", "unknown")
        ops_content.append(format_key_value("trigger state", status_badge(trigger_state)))

        summary = operations.get("summary_line", "")
        if summary:
            ops_content.append(format_key_value("summary", summary))

        for key in (
            "current_focus",
            "monitor_focus",
            "required_action",
            "candidate_version",
            "candidate_state",
            "queue_count",
            "awaiting_confirmation_count",
            "runner_active",
            "runner_lock_state",
            "runner_last_event",
        ):
            value = operations.get(key)
            if value is not None:
                ops_content.append(format_key_value(key.replace("_", " "), value))

        auto_train_blocker = _coerce_mapping(operations.get("auto_train_blocker"))
        if auto_train_blocker:
            block = []
            for k, v in auto_train_blocker.items():
                if v is not None:
                    block.append(f"{k}={_format_scalar(v)}")
            if block:
                ops_content.append(format_key_value("blocker", " | ".join(block)))

        lines.append(draw_box("OPERATIONS", ops_content))
        lines.append("")

    operations_dashboard = _coerce_mapping(mapping.get("operations_dashboard"))
    if operations_dashboard:
        dash_content = []
        for key, value in operations_dashboard.items():
            if value is not None and key not in ("dashboard",):
                dash_content.append(format_key_value(key.replace("_", " "), value))
        nested_dashboard = _coerce_mapping(operations_dashboard.get("dashboard"))
        if nested_dashboard:
            for key, value in nested_dashboard.items():
                if value is not None:
                    dash_content.append(format_key_value(f"dashboard {key.replace('_', ' ')}".strip(), value))
        if dash_content:
            lines.append(draw_box("OPERATIONS DASHBOARD", dash_content))
            lines.append("")

    operations_alert_policy = _coerce_mapping(mapping.get("operations_alert_policy"))
    if operations_alert_policy:
        policy_content = []
        for key, value in operations_alert_policy.items():
            if value is not None:
                policy_content.append(format_key_value(key.replace("_", " "), value))
        if policy_content:
            lines.append(draw_box("OPERATIONS ALERT POLICY", policy_content))
            lines.append("")


__all__ = ["append_operations_overview_sections"]
