"""Operations dashboard and alert-policy surface formatters."""

from __future__ import annotations

from typing import Any

from .operations_alert_policy_surface_formatting import format_operations_alert_policy
from .operations_formatting_deps import OperationsFormattingDeps, resolved_focus as _resolved_focus


def format_operations_dashboard(
    result: Any,
    *,
    deps: OperationsFormattingDeps,
) -> list[str] | None:
    mapping = deps.coerce_mapping(result)
    if not mapping:
        return None

    lines = ["operations dashboard:"]
    resolved_focus = _resolved_focus(mapping, deps=deps)
    summary_line = mapping.get("summary_line")
    inspection_summary_line = mapping.get("inspection_summary_line")
    summary_line, inspection_summary_line = deps.prefer_inspection_summary_for_generic_monitor(
        focus=resolved_focus,
        summary_line=summary_line,
        inspection_summary_line=inspection_summary_line,
    )
    dashboard_digest = mapping.get("dashboard_digest")
    skip_dashboard_digest = bool(dashboard_digest) and dashboard_digest == summary_line

    summary_parts: list[str] = []
    for key in (
        "severity",
        "status",
        "attention_needed",
        "attention_reason",
        "highest_priority_action",
        "active_recovery_hint",
        "latest_recovery",
        "escalated_reasons",
        "remediation_mode",
        "operator_guidance",
        "auto_remediation_allowed",
        "requires_human_review",
        "requires_immediate_action",
        "current_focus",
        "candidate_stage",
        "queue_state",
        "runner_state",
        "daemon_health_state",
        "required_action",
        "last_recovery_event",
        "last_recovery_reason",
        "last_recovery_note",
        "latest_source",
        "latest_event",
        "latest_reason",
    ):
        value = _resolved_focus(mapping, deps=deps) if key == "current_focus" else mapping.get(key)
        if value is not None:
            summary_parts.append(f"{key}={deps.format_scalar(value)}")
    if dashboard_digest is not None and not skip_dashboard_digest:
        summary_parts.append(f"dashboard_digest={deps.format_scalar(dashboard_digest)}")
    if summary_parts:
        lines.append("  " + " | ".join(summary_parts))
    next_actions = mapping.get("next_actions")
    if next_actions:
        lines.append("  next_actions=" + deps.format_scalar(next_actions))
    if summary_line:
        lines.append("  summary=" + deps.format_scalar(summary_line))
    if inspection_summary_line and inspection_summary_line != summary_line:
        lines.append("  inspection_summary=" + deps.format_scalar(inspection_summary_line))
    return lines


__all__ = ["format_operations_alert_policy", "format_operations_dashboard"]
