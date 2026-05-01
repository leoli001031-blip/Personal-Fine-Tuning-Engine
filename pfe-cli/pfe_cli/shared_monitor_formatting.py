"""Shared operations monitor formatting helpers."""

from __future__ import annotations

from typing import Any


GENERIC_MONITOR_FOCUSES = {
    "candidate_idle",
    "queue_waiting_execution",
    "queue_backlog",
    "runner_active",
    "daemon_active",
    "candidate_monitoring",
    "queue_monitoring",
    "runner_monitoring",
    "daemon_monitoring",
}


def prefer_inspection_summary_for_generic_monitor(
    *,
    focus: Any,
    summary_line: Any,
    inspection_summary_line: Any,
) -> tuple[Any, Any]:
    focus_text = str(focus or "").strip().lower()
    if inspection_summary_line and focus_text in GENERIC_MONITOR_FOCUSES:
        return inspection_summary_line, inspection_summary_line
    return summary_line, inspection_summary_line


__all__ = ["GENERIC_MONITOR_FOCUSES", "prefer_inspection_summary_for_generic_monitor"]
