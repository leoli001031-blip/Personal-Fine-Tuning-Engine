"""Operations alert surface builder."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .operations_alert_surface_context import build_alert_surface_context
from .operations_alert_surface_derived import build_derived_alert_surface
from .operations_alert_surface_explicit import build_explicit_alert_surface
from .operations_formatting_deps import OperationsFormattingDeps


def build_operations_alert_surface(
    *,
    operations_alerts: Any | None,
    operations_health: Any | None,
    operations_recovery: Any | None,
    operations_next_actions: Any | None,
    operations_dashboard: Mapping[str, Any] | None,
    operations_alert_policy: Mapping[str, Any] | None,
    operations_console: Mapping[str, Any] | None,
    operations_overview: Mapping[str, Any] | None,
    deps: OperationsFormattingDeps,
) -> dict[str, Any] | None:
    context = build_alert_surface_context(
        operations_alerts=operations_alerts,
        operations_health=operations_health,
        operations_recovery=operations_recovery,
        operations_next_actions=operations_next_actions,
        operations_dashboard=operations_dashboard,
        operations_alert_policy=operations_alert_policy,
        operations_console=operations_console,
        operations_overview=operations_overview,
        deps=deps,
    )
    if context.has_explicit_surface:
        return build_explicit_alert_surface(context, deps=deps)
    if not context.console and not context.overview:
        return None
    return build_derived_alert_surface(context, deps=deps)


__all__ = ["build_operations_alert_surface"]
