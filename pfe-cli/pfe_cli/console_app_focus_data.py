"""Focus normalization helpers for console rendering."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .console_app_data_core import _mapping


GENERIC_MONITOR_FOCUSES = {
    "candidate_idle",
    "queue_waiting_execution",
    "runner_active",
    "daemon_active",
    "candidate_monitoring",
    "queue_monitoring",
    "runner_monitoring",
    "daemon_monitoring",
}


def _prefer_inspection_summary_for_generic_monitor(
    *,
    focus: Any,
    summary_source: str,
    inspection_summary: str,
) -> str:
    focus_text = str(focus or "").strip().lower()
    if inspection_summary and focus_text in GENERIC_MONITOR_FOCUSES:
        return inspection_summary
    return summary_source


def _display_focus_name(focus: Any) -> str:
    normalized = str(focus or "").strip().lower()
    if normalized == "queue_backlog":
        return "queue_waiting_execution"
    return str(focus or "").strip()


def _dashboard_focus(dashboard: Mapping[str, Any] | None) -> str:
    dashboard_map = _mapping(dashboard)
    current_focus = str(dashboard_map.get("current_focus") or "").strip()
    if current_focus.lower() not in {"", "none", "idle", "stable"}:
        return _display_focus_name(current_focus)
    monitor_focus = str(dashboard_map.get("monitor_focus") or "").strip()
    if monitor_focus:
        return _display_focus_name(monitor_focus)
    return _display_focus_name(current_focus or "none")


def _payload_focus(payload: Mapping[str, Any] | None = None) -> str:
    payload_map = _mapping(payload)
    dashboard_focus = _dashboard_focus(payload_map.get("operations_dashboard"))
    if str(dashboard_focus).strip().lower() not in {"", "none", "idle", "stable"}:
        return dashboard_focus
    for raw_focus in (
        _mapping(payload_map.get("operations_console")).get("monitor_focus"),
        _mapping(payload_map.get("operations_overview")).get("monitor_focus"),
    ):
        focus_text = str(raw_focus or "").strip()
        if focus_text:
            return _display_focus_name(focus_text)
    return dashboard_focus


__all__ = [
    "GENERIC_MONITOR_FOCUSES",
    "_dashboard_focus",
    "_display_focus_name",
    "_payload_focus",
    "_prefer_inspection_summary_for_generic_monitor",
]
