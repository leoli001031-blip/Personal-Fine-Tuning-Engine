"""Derived operations alert surface assembly."""

from __future__ import annotations

from typing import Any

from .operations_alert_surface_context import AlertSurfaceContext
from .operations_formatting_deps import OperationsFormattingDeps


def _derived_alerts(context: AlertSurfaceContext, *, deps: OperationsFormattingDeps) -> list[dict[str, Any]]:
    alerts = deps.coerce_sequence_of_mappings(context.overview.get("alerts"))
    if alerts:
        return alerts

    candidate_stage = deps.coerce_mapping(context.console.get("candidate"))
    queue_section = deps.coerce_mapping(context.console.get("queue"))
    runner_section = deps.coerce_mapping(context.console.get("runner"))
    if not (bool(context.overview.get("attention_needed")) or bool(context.console.get("attention_needed"))):
        return []

    return [
        {
            "reason": context.overview.get("attention_reason")
            or context.console.get("attention_reason")
            or "operations_attention",
            "detail": (
                context.overview.get("inspection_summary_line")
                or context.console.get("inspection_summary_line")
                or context.overview.get("summary_line")
                or context.console.get("summary_line")
            ),
            "candidate_stage": candidate_stage.get("current_stage") if candidate_stage else None,
            "queue_count": queue_section.get("count") if queue_section else None,
            "runner_lock_state": runner_section.get("lock_state") if runner_section else None,
        }
    ]


def _derived_next_actions(context: AlertSurfaceContext, *, deps: OperationsFormattingDeps) -> list[Any]:
    next_actions = deps.coerce_sequence_of_scalars(
        context.console.get("next_actions")
    ) or deps.coerce_sequence_of_scalars(context.overview.get("next_actions"))
    if next_actions:
        return next_actions
    if bool(context.overview.get("attention_needed")) or bool(context.console.get("attention_needed")):
        return deps.coerce_sequence_of_scalars(context.console.get("next_actions")) or []
    return []


def build_derived_alert_surface(
    context: AlertSurfaceContext,
    *,
    deps: OperationsFormattingDeps,
) -> dict[str, Any] | None:
    derived_alerts = _derived_alerts(context, deps=deps)
    derived_health = deps.coerce_mapping(context.overview.get("health")) or deps.coerce_mapping(context.console.get("health")) or {}
    derived_recovery = (
        deps.coerce_mapping(context.overview.get("recovery")) or deps.coerce_mapping(context.console.get("recovery")) or {}
    )
    derived_next_actions = _derived_next_actions(context, deps=deps)
    if not derived_alerts and not derived_health and not derived_recovery and not derived_next_actions:
        return None
    return {
        "attention_needed": bool(
            context.overview.get("attention_needed")
            if context.overview.get("attention_needed") is not None
            else context.console.get("attention_needed", False)
        ),
        "current_focus": context.current_focus,
        "required_action": context.required_action,
        "inspection_summary_line": context.inspection_summary_line,
        "alerts": derived_alerts,
        "health": derived_health,
        "recovery": derived_recovery,
        "next_actions": derived_next_actions,
        "summary_line": context.summary_line,
    }


__all__ = ["build_derived_alert_surface"]
