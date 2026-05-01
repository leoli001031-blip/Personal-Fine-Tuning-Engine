"""Operations dashboard and event helper functions for PipelineService."""

from __future__ import annotations

from collections.abc import Iterable
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


def generic_monitor_active(
    *,
    focus: Any,
    inspection_summary_line: Any,
    monitor_focuses: set[str] | frozenset[str] | None = None,
) -> bool:
    focus_text = str(focus or "").strip().lower()
    return bool(inspection_summary_line) and focus_text in (monitor_focuses or GENERIC_MONITOR_FOCUSES)


def prefer_inspection_summary_for_generic_monitor(
    *,
    focus: Any,
    summary_line: Any,
    inspection_summary_line: Any,
    monitor_focuses: set[str] | frozenset[str] | None = None,
) -> tuple[Any, Any]:
    if generic_monitor_active(
        focus=focus,
        inspection_summary_line=inspection_summary_line,
        monitor_focuses=monitor_focuses,
    ):
        return inspection_summary_line, inspection_summary_line
    return summary_line, inspection_summary_line


def operations_event_severity_rank(value: Any) -> int:
    severity = str(value or "info")
    if severity == "critical":
        return 4
    if severity == "warning":
        return 3
    if severity == "info":
        return 2
    return 1


def classify_operations_event(
    *,
    source: Any,
    event: Any,
    reason: Any,
    level: Any | None = None,
    status: Any | None = None,
    state: Any | None = None,
) -> dict[str, Any]:
    severity = "info"
    attention = False
    normalized_level = str(level or "").strip().lower()
    if normalized_level == "attention":
        severity = "info"
        attention = True
    elif normalized_level == "warning":
        severity = "warning"
        attention = True

    normalized_source = str(source or "operations")
    normalized_event = str(event or "")
    normalized_reason = str(reason or "")
    normalized_status = str(status or "")
    normalized_state = str(state or "")
    combined = " ".join(
        part
        for part in (
            normalized_event,
            normalized_reason,
            normalized_status,
            normalized_state,
        )
        if part
    ).lower()

    if "queue_pending_review" in combined:
        severity = "info"
        attention = True
    elif "queue_waiting_execution" in combined:
        severity = "info"
        attention = True
    elif "queue_processing_active" in combined:
        severity = "info"
        attention = False
    elif any(token in combined for token in ("awaiting_confirmation", "manual_review_required")):
        severity = "info"
        attention = True
    elif "candidate_ready_for_promotion" in combined:
        severity = "info"
        attention = True
    elif normalized_source == "daemon" and any(token in combined for token in ("stale", "expired", "failed", "error")):
        severity = "critical"
        attention = True
    elif normalized_source == "daemon" and any(
        token in combined for token in ("backoff", "capped", "blocked", "recover", "restart", "delayed")
    ):
        severity = "warning"
        attention = True
    elif normalized_source == "runner" and any(
        token in combined for token in ("stale", "blocked", "failed", "error", "stop_requested")
    ):
        severity = "warning"
        attention = True
    elif any(token in combined for token in ("stale", "expired", "backoff", "capped", "blocked", "failed", "error")):
        severity = "warning"
        attention = True
    elif normalized_source in {"runner", "daemon"} and normalized_event == "alert":
        severity = "warning"
        attention = True
    elif normalized_source == "queue" and normalized_state in {"awaiting_confirmation", "failed"}:
        severity = "warning" if normalized_state == "failed" else "info"
        attention = True

    return {
        "severity": severity,
        "attention": attention,
    }


def ordered_unique_actions(*action_groups: Iterable[Any]) -> list[str]:
    actions: list[str] = []
    for action_group in action_groups:
        for action_name in action_group:
            action_text = str(action_name or "").strip()
            if not action_text or action_text == "none" or action_text in actions:
                continue
            actions.append(action_text)
    return actions


__all__ = [
    "GENERIC_MONITOR_FOCUSES",
    "classify_operations_event",
    "generic_monitor_active",
    "operations_event_severity_rank",
    "ordered_unique_actions",
    "prefer_inspection_summary_for_generic_monitor",
]
