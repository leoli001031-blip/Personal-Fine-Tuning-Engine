"""Operations attention digest formatting."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .operations_attention_adapters import append_adapter_export_attention
from .operations_attention_candidates import append_candidate_attention
from .operations_attention_context import (
    append_final_monitor_alert,
    append_generic_monitor_alert,
    append_structured_alert_reasons,
    build_attention_context,
)
from .operations_attention_queue import append_train_queue_attention
from .operations_formatting_deps import OperationsFormattingDeps


def format_ops_attention(
    *,
    operations_alerts: Any | None,
    operations_overview: Mapping[str, Any] | None,
    operations_dashboard: Mapping[str, Any] | None,
    operations_alert_policy: Mapping[str, Any] | None,
    candidate_summary: Mapping[str, Any] | None,
    train_queue: Mapping[str, Any] | None,
    latest_adapter_map: Mapping[str, Any] | None,
    recent_adapter_map: Mapping[str, Any] | None,
    deps: OperationsFormattingDeps,
) -> str | None:
    alerts: list[str] = []
    context = build_attention_context(
        operations_alerts=operations_alerts,
        operations_overview=operations_overview,
        operations_dashboard=operations_dashboard,
        operations_alert_policy=operations_alert_policy,
        deps=deps,
    )
    append_structured_alert_reasons(alerts, context, deps=deps)
    context = append_generic_monitor_alert(alerts, context, deps=deps)
    append_candidate_attention(alerts, candidate_summary=candidate_summary, context=context, deps=deps)
    append_train_queue_attention(alerts, train_queue=train_queue, context=context, deps=deps)
    append_adapter_export_attention(
        alerts,
        latest_adapter_map=latest_adapter_map,
        recent_adapter_map=recent_adapter_map,
        deps=deps,
    )
    append_final_monitor_alert(alerts, context, deps=deps)

    if not alerts:
        return "ops attention: clean"
    return "ops attention: " + " | ".join(alerts)
