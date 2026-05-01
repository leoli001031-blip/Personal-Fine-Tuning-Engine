"""Context assembly for the Rich operations summary panel."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .console_app_data import (
    _dashboard_focus,
    _mapping,
    _prefer_inspection_summary_for_generic_monitor,
    _resolved_queue_review_policy,
    _value,
)
from .console_app_operations_context_sources import (
    blocked_trigger_sources,
    runtime_stability_source,
    trigger_policy_gate_source,
    trigger_policy_source,
    trigger_threshold_source,
)


@dataclass(frozen=True)
class OperationsPanelContext:
    dashboard: dict[str, Any]
    alert_policy: dict[str, Any]
    overview: dict[str, Any]
    console: dict[str, Any]
    trigger_policy: dict[str, Any]
    trigger_policy_gate: dict[str, Any]
    trigger_threshold: dict[str, Any]
    runtime_stability: dict[str, Any]
    queue_review_policy: dict[str, Any]
    trigger_blocked_reason: str
    trigger_blocked_action: str
    trigger_blocked_category: str
    severity: str
    focus_value: str
    normalized_focus: str
    action_value: str
    priority_value: str
    summary_source: str
    guidance_source: str


def build_operations_panel_context(payload: Mapping[str, Any]) -> OperationsPanelContext:
    dashboard = _mapping(payload.get("operations_dashboard"))
    alert_policy = _mapping(payload.get("operations_alert_policy"))
    overview = _mapping(payload.get("operations_overview"))
    console = _mapping(payload.get("operations_console"))
    trigger = _mapping(payload.get("auto_train_trigger"))
    trigger_policy = trigger_policy_source(console=console, overview=overview, trigger=trigger)
    trigger_policy_gate = trigger_policy_gate_source(trigger_policy=trigger_policy, console=console)
    trigger_threshold = trigger_threshold_source(console=console, overview=overview, trigger=trigger)
    runtime_stability = runtime_stability_source(console=console, overview=overview)
    train_queue = _mapping(payload.get("train_queue"))
    trigger_blocked_reason, trigger_blocked_action, trigger_blocked_category = blocked_trigger_sources(
        console=console,
        overview=overview,
    )
    queue_review_policy = _resolved_queue_review_policy(
        console=console,
        overview=overview,
        train_queue=train_queue,
        trigger_policy=trigger_policy,
        trigger_blocked_reason=trigger_blocked_reason,
        trigger_blocked_action=trigger_blocked_action,
    )
    severity = _value(dashboard, "severity", default="stable")
    focus_value = _dashboard_focus(dashboard)
    normalized_focus = str(focus_value or "").strip().lower()
    summary_source = _prefer_inspection_summary_for_generic_monitor(
        focus=normalized_focus,
        summary_source=_value(overview, "summary_line", default="idle"),
        inspection_summary=_value(overview, "inspection_summary_line", default=""),
    )
    guidance_source = _value(alert_policy, "operator_guidance", default="observe the system")
    if normalized_focus in {
        "insufficient_new_signal_samples",
        "holdout_not_ready",
        "cooldown_active",
        "failure_backoff_active",
    }:
        guidance_source = _value(trigger_threshold, "summary_line", default=guidance_source)
    return OperationsPanelContext(
        dashboard=dashboard,
        alert_policy=alert_policy,
        overview=overview,
        console=console,
        trigger_policy=trigger_policy,
        trigger_policy_gate=trigger_policy_gate,
        trigger_threshold=trigger_threshold,
        runtime_stability=runtime_stability,
        queue_review_policy=queue_review_policy,
        trigger_blocked_reason=trigger_blocked_reason,
        trigger_blocked_action=trigger_blocked_action,
        trigger_blocked_category=trigger_blocked_category,
        severity=severity,
        focus_value=focus_value,
        normalized_focus=normalized_focus,
        action_value=_value(alert_policy, "required_action", "primary_action", default="observe_and_monitor"),
        priority_value=_value(alert_policy, "action_priority", default="p2"),
        summary_source=summary_source,
        guidance_source=guidance_source,
    )


__all__ = ["OperationsPanelContext", "build_operations_panel_context"]
