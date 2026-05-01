"""Operations console status sections for Matrix terminal output."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .matrix_formatting_common import _coerce_mapping, _coerce_sequence_of_mappings, _format_scalar
from .terminal_theme import MatrixColors, draw_box, format_key_value, status_badge


def append_operations_console_sections(lines: list[str], mapping: Mapping[str, Any]) -> None:
    """Append operations console, alerts, and event stream boxes."""
    operations_console = _coerce_mapping(mapping.get("operations_console"))
    if operations_console:
        console_content = []
        attention_needed = operations_console.get("attention_needed")
        if attention_needed is not None:
            console_content.append(format_key_value("attention needed", "yes" if attention_needed else "no"))
        attention_reason = operations_console.get("attention_reason")
        if attention_reason:
            console_content.append(format_key_value("attention reason", attention_reason))
        summary_line = operations_console.get("summary_line")
        if summary_line:
            console_content.append(format_key_value("summary", summary_line))
        next_actions = operations_console.get("next_actions")
        if next_actions:
            console_content.append(format_key_value("next actions", ", ".join(str(a) for a in next_actions)))
        candidate = _coerce_mapping(operations_console.get("candidate"))
        if candidate:
            cands = []
            for key, value in candidate.items():
                if value is not None:
                    cands.append(f"{key.replace('_', ' ')}={_format_scalar(value)}")
            if cands:
                console_content.append(format_key_value("candidate", " | ".join(cands)))
        queue = _coerce_mapping(operations_console.get("queue"))
        if queue:
            qs = []
            for key, value in queue.items():
                if value is not None:
                    qs.append(f"{key.replace('_', ' ')}={_format_scalar(value)}")
            if qs:
                console_content.append(format_key_value("queue", " | ".join(qs)))
        runner = _coerce_mapping(operations_console.get("runner"))
        if runner:
            rs = []
            for key, value in runner.items():
                if value is not None:
                    rs.append(f"{key.replace('_', ' ')}={_format_scalar(value)}")
            if rs:
                console_content.append(format_key_value("runner", " | ".join(rs)))
        timelines = _coerce_mapping(operations_console.get("timelines"))
        if timelines:
            ts = []
            for key, value in timelines.items():
                if value is not None:
                    ts.append(f"{key.replace('_', ' ')}={_format_scalar(value)}")
            if ts:
                console_content.append(format_key_value("timelines", " | ".join(ts)))
        if console_content:
            lines.append(draw_box("OPERATIONS CONSOLE", console_content))
            lines.append("")

    operations_alerts = _coerce_sequence_of_mappings(mapping.get("operations_alerts"))
    if operations_alerts:
        alert_content = []
        for alert in operations_alerts:
            severity = alert.get("severity", "info")
            message = alert.get("message", alert.get("alert", "unknown alert"))
            badge = (
                status_badge(severity)
                if severity in ("info", "warning", "error", "critical")
                else f"{MatrixColors.GRAY}[ {severity.upper()} ]{MatrixColors.RESET}"
            )
            alert_content.append(f"  {badge} {message}")
        if alert_content:
            lines.append(draw_box("OPERATIONS ALERTS", alert_content))
            lines.append("")

    operations_event_stream = _coerce_mapping(mapping.get("operations_event_stream"))
    if operations_event_stream is None:
        operations_console = _coerce_mapping(mapping.get("operations_console"))
        if operations_console:
            operations_event_stream = _coerce_mapping(operations_console.get("event_stream"))
    if operations_event_stream:
        es_content = []
        for key, value in operations_event_stream.items():
            if value is not None and key != "dashboard" and not isinstance(value, (list, tuple)):
                es_content.append(format_key_value(key.replace("_", " "), value))
        nested_dashboard = _coerce_mapping(operations_event_stream.get("dashboard"))
        if nested_dashboard:
            for key, value in nested_dashboard.items():
                if value is not None:
                    es_content.append(format_key_value(f"dashboard {key.replace('_', ' ')}".strip(), value))
        if es_content:
            lines.append(draw_box("OPERATIONS EVENT STREAM", es_content))
            lines.append("")


__all__ = ["append_operations_console_sections"]
