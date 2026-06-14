"""Text sections for operations event stream formatting."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .operations_formatting_deps import OperationsFormattingDeps, resolved_focus as _resolved_focus


def event_stream_summary_parts(mapping: Mapping[str, Any], *, deps: OperationsFormattingDeps) -> list[str]:
    summary_parts: list[str] = []
    for key in (
        "count",
        "severity",
        "status",
        "attention_needed",
        "attention_reason",
        "attention_source",
        "current_focus",
        "required_action",
        "last_recovery_event",
        "last_recovery_reason",
        "last_recovery_note",
        "highest_priority_action",
        "active_recovery_hint",
        "latest_recovery",
        "latest_source",
        "latest_event",
        "latest_reason",
        "latest_timestamp",
        "alert_count",
        "escalated_reasons",
    ):
        value = _resolved_focus(mapping, deps=deps) if key == "current_focus" else mapping.get(key)
        if value is not None:
            summary_parts.append(f"{key}={deps.format_scalar(value)}")
    summary_line, inspection_summary_line = _preferred_summary_lines(mapping, deps=deps)
    if summary_line:
        summary_parts.append(f"summary={deps.format_scalar(summary_line)}")
    if inspection_summary_line and inspection_summary_line != summary_line:
        summary_parts.append(f"inspection_summary={deps.format_scalar(inspection_summary_line)}")
    return summary_parts


def append_event_stream_dashboard(
    lines: list[str],
    mapping: Mapping[str, Any],
    *,
    deps: OperationsFormattingDeps,
) -> None:
    dashboard = deps.coerce_mapping(mapping.get("dashboard"))
    if not dashboard:
        return

    dashboard_parts: list[str] = []
    for key in (
        "severity",
        "status",
        "attention_needed",
        "attention_reason",
        "current_focus",
        "required_action",
        "last_recovery_event",
        "last_recovery_reason",
        "last_recovery_note",
        "latest_source",
        "latest_event",
        "latest_reason",
    ):
        value = _resolved_focus(dashboard, deps=deps) if key == "current_focus" else dashboard.get(key)
        if value is not None:
            dashboard_parts.append(f"{key}={deps.format_scalar(value)}")
    dashboard_summary_line, dashboard_inspection_summary = _preferred_summary_lines(dashboard, deps=deps)
    if dashboard_summary_line:
        dashboard_parts.append(f"summary={deps.format_scalar(dashboard_summary_line)}")
    if dashboard_inspection_summary and dashboard_inspection_summary != dashboard_summary_line:
        dashboard_parts.append(f"inspection_summary={deps.format_scalar(dashboard_inspection_summary)}")
    if dashboard_parts:
        lines.append("  dashboard: " + " | ".join(dashboard_parts))


def _preferred_summary_lines(
    mapping: Mapping[str, Any],
    *,
    deps: OperationsFormattingDeps,
) -> tuple[Any, Any]:
    return deps.prefer_inspection_summary_for_generic_monitor(
        focus=_resolved_focus(mapping, deps=deps),
        summary_line=mapping.get("summary_line"),
        inspection_summary_line=mapping.get("inspection_summary_line"),
    )


__all__ = [
    "append_event_stream_dashboard",
    "event_stream_summary_parts",
]
