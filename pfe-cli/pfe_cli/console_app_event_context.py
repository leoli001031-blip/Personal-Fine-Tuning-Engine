"""Context assembly for the Rich event stream panel."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .console_app_data import (
    _dashboard_focus,
    _display_focus_name,
    _mapping,
    _sequence,
    _value,
)


@dataclass(frozen=True)
class EventPanelContext:
    stream: dict[str, Any]
    dashboard: dict[str, Any]
    overview: dict[str, Any]
    items: list[Any]
    stream_severity: str
    priority_value: str
    action_value: str
    latest_source: str
    latest_reason: str
    trigger_blocked_category: str
    focus_value: str
    normalized_focus: str
    inspection_summary: str


def build_event_panel_context(payload: Mapping[str, Any]) -> EventPanelContext:
    stream = _mapping(payload.get("operations_event_stream"))
    dashboard = _mapping(stream.get("dashboard"))
    alert_policy = _mapping(payload.get("operations_alert_policy"))
    overview = _mapping(payload.get("operations_overview"))
    console = _mapping(payload.get("operations_console"))
    focus_value = _dashboard_focus(dashboard)
    if focus_value.lower() in {"", "none", "idle", "stable"}:
        focus_value = _display_focus_name(_value(dashboard, "attention_reason", default="none"))
    inspection_summary = _value(
        stream,
        "inspection_summary_line",
        default=_value(
            dashboard,
            "inspection_summary_line",
            default=_value(overview, "inspection_summary_line", default=""),
        ),
    )
    return EventPanelContext(
        stream=stream,
        dashboard=dashboard,
        overview=overview,
        items=_sequence(stream.get("items")),
        stream_severity=_value(stream, "severity", default="stable"),
        priority_value=_value(alert_policy, "action_priority", default="p2"),
        action_value=_value(alert_policy, "required_action", "primary_action", default="observe_and_monitor"),
        latest_source=_value(stream, "latest_source", default="queue"),
        latest_reason=_value(stream, "latest_reason", "attention_reason", default=""),
        trigger_blocked_category=_value(
            console,
            "trigger_blocked_category",
            default=_value(overview, "trigger_blocked_category", default=""),
        ),
        focus_value=focus_value,
        normalized_focus=str(focus_value or "").strip().lower(),
        inspection_summary=inspection_summary,
    )


__all__ = ["EventPanelContext", "build_event_panel_context"]
