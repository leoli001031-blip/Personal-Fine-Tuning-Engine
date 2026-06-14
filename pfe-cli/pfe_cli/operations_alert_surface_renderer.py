"""Operations alert surface text renderer."""

from __future__ import annotations

from typing import Any

from .operations_alert_surface_builder import build_operations_alert_surface
from .operations_alert_surface_render_parts import (
    append_alert_health,
    append_alert_items,
    append_alert_next_actions,
    append_alert_recovery,
    append_alert_summary,
)
from .operations_formatting_deps import OperationsFormattingDeps


def format_operations_alert_surface(
    result: Any,
    *,
    deps: OperationsFormattingDeps,
) -> list[str] | None:
    mapping = deps.coerce_mapping(result)
    if mapping is None:
        return None

    alert_surface = build_operations_alert_surface(
        operations_alerts=mapping.pop("operations_alerts", None),
        operations_health=mapping.pop("operations_health", None),
        operations_recovery=mapping.pop("operations_recovery", None),
        operations_next_actions=mapping.pop("operations_next_actions", None),
        operations_dashboard=deps.coerce_mapping(mapping.get("operations_dashboard")),
        operations_alert_policy=deps.coerce_mapping(mapping.get("operations_alert_policy")),
        operations_console=deps.coerce_mapping(mapping.get("operations_console")),
        operations_overview=deps.coerce_mapping(mapping.get("operations_overview")),
        deps=deps,
    )
    if alert_surface is None:
        return None

    lines = ["operations alerts:"]
    append_alert_summary(lines, alert_surface, deps)
    append_alert_items(lines, alert_surface, deps)
    append_alert_health(lines, alert_surface, deps)
    append_alert_recovery(lines, alert_surface, deps)
    append_alert_next_actions(lines, alert_surface, deps)
    return lines


__all__ = ["format_operations_alert_surface"]
