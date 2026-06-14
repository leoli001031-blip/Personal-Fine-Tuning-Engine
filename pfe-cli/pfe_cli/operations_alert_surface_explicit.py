"""Explicit operations alert surface assembly."""

from __future__ import annotations

from typing import Any

from .operations_alert_surface_context import AlertSurfaceContext
from .operations_formatting_deps import OperationsFormattingDeps


def build_explicit_alert_surface(
    context: AlertSurfaceContext,
    *,
    deps: OperationsFormattingDeps,
) -> dict[str, Any]:
    attention_needed = bool(context.health.get("status") == "attention" or context.alerts or context.next_actions)
    return {
        "attention_needed": attention_needed,
        "current_focus": context.current_focus,
        "required_action": context.required_action,
        "inspection_summary_line": context.inspection_summary_line,
        "alerts": context.alerts,
        "health": context.health,
        "recovery": context.recovery,
        "next_actions": context.next_actions,
        "summary_line": deps.format_scalar(context.summary_line),
    }


__all__ = ["build_explicit_alert_surface"]
