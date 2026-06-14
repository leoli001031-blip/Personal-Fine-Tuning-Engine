"""Build rendered text fields for operations console digest summaries."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .operations_console_digest_summary_fields import ConsoleDigestSummaryFields
from .operations_formatting_deps import OperationsFormattingDeps


def build_console_digest_summary_line(
    *,
    overview: Mapping[str, Any],
    candidate_section: Mapping[str, Any],
    queue_section: Mapping[str, Any],
    runner_section: Mapping[str, Any],
    daemon_section: Mapping[str, Any],
    runner_timeline_section: Mapping[str, Any],
    fields: ConsoleDigestSummaryFields,
    deps: OperationsFormattingDeps,
) -> Any:
    """Build the compact summary_line value."""
    summary_parts = []
    if overview.get("summary_line"):
        summary_parts.append(str(overview.get("summary_line")))
    elif candidate_section.get("current_stage"):
        summary_parts.append(f"candidate-stage={deps.format_scalar(candidate_section['current_stage'])}")
    if fields.current_focus is not None:
        summary_parts.append(f"current_focus={deps.format_scalar(fields.current_focus)}")
    if fields.required_action is not None:
        summary_parts.append(f"required_action={deps.format_scalar(fields.required_action)}")
    if queue_section.get("awaiting_confirmation_count"):
        summary_parts.append(f"awaiting-confirm={deps.format_scalar(queue_section['awaiting_confirmation_count'])}")
    if runner_section.get("lock_state"):
        summary_parts.append(f"runner-lock={deps.format_scalar(runner_section['lock_state'])}")
    if runner_timeline_section.get("last_event"):
        summary_parts.append(f"runner-timeline={deps.format_scalar(runner_timeline_section['last_event'])}")
    if runner_timeline_section.get("recent_anomaly_reason"):
        summary_parts.append(f"runner-anomaly={deps.format_scalar(runner_timeline_section['recent_anomaly_reason'])}")
    if daemon_section.get("recent_anomaly_reason"):
        summary_parts.append(f"daemon-anomaly={deps.format_scalar(daemon_section['recent_anomaly_reason'])}")
    if fields.last_recovery_event is not None:
        summary_parts.append(f"last_recovery_event={deps.format_scalar(fields.last_recovery_event)}")
    if fields.last_recovery_reason is not None:
        summary_parts.append(f"last_recovery_reason={deps.format_scalar(fields.last_recovery_reason)}")
    if fields.last_recovery_note is not None:
        summary_parts.append(f"last_recovery_note={deps.format_scalar(fields.last_recovery_note)}")
    return " | ".join(summary_parts)


def build_fallback_inspection_summary_line(
    *,
    fields: ConsoleDigestSummaryFields,
    deps: OperationsFormattingDeps,
) -> Any:
    """Build the fallback inspection summary value."""
    return " | ".join(
        part
        for part in (
            f"current_focus={deps.format_scalar(fields.current_focus)}" if fields.current_focus is not None else None,
            f"required_action={deps.format_scalar(fields.required_action)}" if fields.required_action is not None else None,
            f"last_recovery_event={deps.format_scalar(fields.last_recovery_event)}"
            if fields.last_recovery_event is not None
            else None,
            f"last_recovery_reason={deps.format_scalar(fields.last_recovery_reason)}"
            if fields.last_recovery_reason is not None
            else None,
            f"last_recovery_note={deps.format_scalar(fields.last_recovery_note)}"
            if fields.last_recovery_note is not None
            else None,
            f"next_actions={deps.format_scalar(fields.next_actions)}" if fields.next_actions else None,
        )
        if part is not None
    )


__all__ = ["build_console_digest_summary_line", "build_fallback_inspection_summary_line"]
