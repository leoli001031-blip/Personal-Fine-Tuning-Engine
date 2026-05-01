"""Alert appenders for operations attention formatting."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .operations_attention_context import OperationsAttentionContext


def append_structured_alert_reasons(alerts: list[str], context: "OperationsAttentionContext", *, deps: Any) -> None:
    for alert in context.structured_alerts:
        reason = alert.get("reason")
        if reason is not None:
            alerts.append(deps.format_scalar(reason))


def append_generic_monitor_alert(
    alerts: list[str],
    context: "OperationsAttentionContext",
    *,
    deps: Any,
) -> "OperationsAttentionContext":
    if not (
        context.resolved_focus
        and context.required_action is not None
        and str(context.resolved_focus).strip().lower() in deps.generic_monitor_focuses
        and not context.structured_alerts
    ):
        return context

    if context.inspection_summary_line:
        alerts.append("monitor " + deps.format_scalar(context.inspection_summary_line))
    else:
        parts = [
            f"current_focus={deps.format_scalar(context.resolved_focus)}",
            f"required_action={deps.format_scalar(context.required_action)}",
        ]
        alerts.append("monitor " + " | ".join(parts))
    return _with_monitor_alert_emitted(context)


def append_final_monitor_alert(alerts: list[str], context: "OperationsAttentionContext", *, deps: Any) -> None:
    if not context.resolved_focus or any("current_focus=" in alert for alert in alerts):
        return
    parts = [f"current_focus={deps.format_scalar(context.resolved_focus)}"]
    if context.required_action is not None:
        parts.append(f"required_action={deps.format_scalar(context.required_action)}")
    alerts.append("monitor " + " | ".join(parts))


def _with_monitor_alert_emitted(context: "OperationsAttentionContext") -> "OperationsAttentionContext":
    from .operations_attention_context import OperationsAttentionContext

    return OperationsAttentionContext(
        overview=context.overview,
        dashboard=context.dashboard,
        alert_policy=context.alert_policy,
        structured_alerts=context.structured_alerts,
        resolved_focus=context.resolved_focus,
        required_action=context.required_action,
        inspection_summary_line=context.inspection_summary_line,
        monitor_alert_emitted=True,
    )


__all__ = [
    "append_final_monitor_alert",
    "append_generic_monitor_alert",
    "append_structured_alert_reasons",
]
