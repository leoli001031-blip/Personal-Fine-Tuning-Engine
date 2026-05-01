"""Shared context helpers for operations attention formatting."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .operations_attention_alerts import (
    append_final_monitor_alert,
    append_generic_monitor_alert,
    append_structured_alert_reasons,
)


@dataclass(frozen=True)
class OperationsAttentionContext:
    overview: dict[str, Any]
    dashboard: dict[str, Any]
    alert_policy: dict[str, Any]
    structured_alerts: list[dict[str, Any]]
    resolved_focus: str | None
    required_action: Any
    inspection_summary_line: Any
    monitor_alert_emitted: bool = False


def build_attention_context(
    *,
    operations_alerts: Any | None,
    operations_overview: Mapping[str, Any] | None,
    operations_dashboard: Mapping[str, Any] | None,
    operations_alert_policy: Mapping[str, Any] | None,
    deps: Any,
) -> OperationsAttentionContext:
    overview = deps.coerce_mapping(operations_overview) or {}
    dashboard = deps.coerce_mapping(operations_dashboard) or {}
    alert_policy = deps.coerce_mapping(operations_alert_policy) or {}
    structured_alerts = deps.coerce_sequence_of_mappings(operations_alerts)
    resolved_focus = resolved_attention_focus(
        overview=overview,
        dashboard=dashboard,
        alert_policy=alert_policy,
    )
    required_action = (
        overview.get("required_action") or alert_policy.get("required_action") or dashboard.get("required_action")
    )
    inspection_summary_line = (
        overview.get("inspection_summary_line")
        or dashboard.get("inspection_summary_line")
        or alert_policy.get("inspection_summary_line")
    )
    return OperationsAttentionContext(
        overview=overview,
        dashboard=dashboard,
        alert_policy=alert_policy,
        structured_alerts=structured_alerts,
        resolved_focus=resolved_focus,
        required_action=required_action,
        inspection_summary_line=inspection_summary_line,
    )


def resolved_attention_focus(
    *,
    overview: Mapping[str, Any],
    dashboard: Mapping[str, Any],
    alert_policy: Mapping[str, Any],
) -> str | None:
    for candidate in (
        overview.get("current_focus"),
        overview.get("monitor_focus"),
        dashboard.get("current_focus"),
        dashboard.get("monitor_focus"),
        alert_policy.get("current_focus"),
    ):
        if candidate is None:
            continue
        text = str(candidate).strip()
        if text.lower() in {"", "none", "idle", "stable"}:
            continue
        return text
    return None


__all__ = [
    "OperationsAttentionContext",
    "append_final_monitor_alert",
    "append_generic_monitor_alert",
    "append_structured_alert_reasons",
    "build_attention_context",
    "resolved_attention_focus",
]
