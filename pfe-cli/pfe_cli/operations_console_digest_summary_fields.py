"""Resolve derived fields for operations console digest summaries."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .operations_formatting_deps import OperationsFormattingDeps, resolved_first_focus as _resolved_first_focus


@dataclass(frozen=True)
class ConsoleDigestSummaryFields:
    current_focus: Any
    next_actions: list[str]
    required_action: Any
    inspection_summary_line: Any
    last_recovery_event: Any
    last_recovery_reason: Any
    last_recovery_note: Any


def resolve_console_digest_summary_fields(
    *,
    overview: Mapping[str, Any],
    dashboard_surface: Mapping[str, Any],
    alert_policy_surface: Mapping[str, Any],
    daemon_timeline: Mapping[str, Any],
    derived_next_actions: list[str],
    deps: OperationsFormattingDeps,
) -> ConsoleDigestSummaryFields:
    """Resolve summary inputs from operation surfaces."""
    current_focus = _resolved_first_focus(
        overview.get("current_focus"),
        overview.get("monitor_focus"),
        dashboard_surface.get("current_focus"),
        dashboard_surface.get("monitor_focus"),
        alert_policy_surface.get("current_focus"),
        overview.get("attention_reason"),
        overview.get("monitor_focus"),
        alert_policy_surface.get("required_action"),
        derived_next_actions[0] if derived_next_actions else None,
    )
    next_actions = (
        deps.coerce_sequence_of_scalars(overview.get("next_actions"))
        or deps.coerce_sequence_of_scalars(dashboard_surface.get("next_actions"))
        or deps.coerce_sequence_of_scalars(alert_policy_surface.get("next_actions"))
        or derived_next_actions
    )
    required_action = (
        overview.get("required_action")
        or alert_policy_surface.get("required_action")
        or dashboard_surface.get("required_action")
        or (derived_next_actions[0] if derived_next_actions else None)
    )
    inspection_summary_line = (
        overview.get("inspection_summary_line")
        or dashboard_surface.get("inspection_summary_line")
        or alert_policy_surface.get("inspection_summary_line")
    )
    last_recovery_event = (
        dashboard_surface.get("last_recovery_event")
        or alert_policy_surface.get("last_recovery_event")
        or daemon_timeline.get("last_recovery_event")
    )
    last_recovery_reason = (
        dashboard_surface.get("last_recovery_reason")
        or alert_policy_surface.get("last_recovery_reason")
        or daemon_timeline.get("last_recovery_reason")
    )
    last_recovery_note = (
        dashboard_surface.get("last_recovery_note")
        or alert_policy_surface.get("last_recovery_note")
        or daemon_timeline.get("last_recovery_note")
    )

    return ConsoleDigestSummaryFields(
        current_focus=current_focus,
        next_actions=next_actions,
        required_action=required_action,
        inspection_summary_line=inspection_summary_line,
        last_recovery_event=last_recovery_event,
        last_recovery_reason=last_recovery_reason,
        last_recovery_note=last_recovery_note,
    )


__all__ = ["ConsoleDigestSummaryFields", "resolve_console_digest_summary_fields"]
