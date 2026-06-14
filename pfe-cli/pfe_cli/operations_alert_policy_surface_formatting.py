"""Operations alert-policy surface formatter."""

from __future__ import annotations

from typing import Any

from .operations_formatting_deps import OperationsFormattingDeps, resolved_focus as _resolved_focus


def format_operations_alert_policy(
    result: Any,
    *,
    deps: OperationsFormattingDeps,
) -> list[str] | None:
    mapping = deps.coerce_mapping(result)
    if not mapping:
        return None

    lines = ["operations alert policy:"]
    summary_parts: list[str] = []
    for key in (
        "severity",
        "required_action",
        "current_focus",
        "primary_action",
        "highest_priority_action",
        "action_priority",
        "escalation_mode",
        "requires_immediate_action",
        "requires_human_review",
        "auto_remediation_allowed",
        "remediation_mode",
        "operator_guidance",
        "active_recovery_hint",
        "latest_recovery",
        "escalated_reasons",
        "last_recovery_event",
        "last_recovery_reason",
        "last_recovery_note",
    ):
        value = _resolved_focus(mapping, deps=deps) if key == "current_focus" else mapping.get(key)
        if value is not None:
            summary_parts.append(f"{key}={deps.format_scalar(value)}")
    if summary_parts:
        lines.append("  " + " | ".join(summary_parts))
    next_actions = mapping.get("next_actions")
    if next_actions:
        lines.append("  next_actions=" + deps.format_scalar(next_actions))
    inspection_summary_line = mapping.get("inspection_summary_line")
    resolved_focus = _resolved_focus(mapping, deps=deps)
    summary_line = mapping.get("summary_line")
    summary_line, inspection_summary_line = deps.prefer_inspection_summary_for_generic_monitor(
        focus=resolved_focus,
        summary_line=summary_line,
        inspection_summary_line=inspection_summary_line,
    )
    if summary_line:
        lines.append("  summary=" + deps.format_scalar(summary_line))
    if inspection_summary_line and inspection_summary_line != summary_line:
        lines.append("  inspection_summary=" + deps.format_scalar(inspection_summary_line))
    return lines


__all__ = ["format_operations_alert_policy"]
