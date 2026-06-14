"""Summary builders for derived operations console digests."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .operations_console_digest_summary_fields import resolve_console_digest_summary_fields
from .operations_console_digest_summary_text import (
    build_console_digest_summary_line,
    build_fallback_inspection_summary_line,
)
from .operations_formatting_deps import OperationsFormattingDeps


@dataclass(frozen=True)
class ConsoleDigestSummary:
    attention_needed: bool
    summary_line: Any
    inspection_summary_line: Any
    next_actions: list[str]
    current_focus: Any
    required_action: Any
    last_recovery_event: Any
    last_recovery_reason: Any
    last_recovery_note: Any


def build_console_digest_summary(
    *,
    overview: Mapping[str, Any],
    dashboard_surface: Mapping[str, Any],
    alert_policy_surface: Mapping[str, Any],
    daemon_timeline: Mapping[str, Any],
    candidate_section: Mapping[str, Any],
    queue_section: Mapping[str, Any],
    runner_section: Mapping[str, Any],
    daemon_section: Mapping[str, Any],
    runner_timeline_section: Mapping[str, Any],
    derived_next_actions: list[str],
    deps: OperationsFormattingDeps,
) -> ConsoleDigestSummary:
    fields = resolve_console_digest_summary_fields(
        overview=overview,
        dashboard_surface=dashboard_surface,
        alert_policy_surface=alert_policy_surface,
        daemon_timeline=daemon_timeline,
        derived_next_actions=derived_next_actions,
        deps=deps,
    )
    summary_line = build_console_digest_summary_line(
        overview=overview,
        candidate_section=candidate_section,
        queue_section=queue_section,
        runner_section=runner_section,
        daemon_section=daemon_section,
        runner_timeline_section=runner_timeline_section,
        fields=fields,
        deps=deps,
    )
    inspection_summary_line = fields.inspection_summary_line
    summary_line, inspection_summary_line = deps.prefer_inspection_summary_for_generic_monitor(
        focus=fields.current_focus,
        summary_line=summary_line,
        inspection_summary_line=inspection_summary_line,
    )

    attention_needed = overview.get("attention_needed")
    if attention_needed is None:
        attention_needed = bool(fields.next_actions)

    inspection_summary_line = inspection_summary_line or build_fallback_inspection_summary_line(fields=fields, deps=deps)

    return ConsoleDigestSummary(
        attention_needed=bool(attention_needed),
        summary_line=summary_line,
        inspection_summary_line=inspection_summary_line,
        next_actions=fields.next_actions,
        current_focus=fields.current_focus,
        required_action=fields.required_action,
        last_recovery_event=fields.last_recovery_event,
        last_recovery_reason=fields.last_recovery_reason,
        last_recovery_note=fields.last_recovery_note,
    )


__all__ = ["ConsoleDigestSummary", "build_console_digest_summary"]
