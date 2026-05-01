"""Context assembly for console focus action decisions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .console_actions_deps import ConsoleActionsDeps


@dataclass(frozen=True)
class ConsoleFocusContext:
    operations_dashboard: Mapping[str, Any]
    operations_console: Mapping[str, Any]
    alert_policy: Mapping[str, Any]
    candidate_summary: Mapping[str, Any]
    current_focus: str
    required_action: str


def build_console_focus_context(
    payload: Mapping[str, Any] | None,
    *,
    deps: ConsoleActionsDeps,
) -> ConsoleFocusContext:
    source = payload or {}
    alert_policy = deps.coerce_mapping(source.get("operations_alert_policy")) or {}
    return ConsoleFocusContext(
        operations_dashboard=deps.coerce_mapping(source.get("operations_dashboard")) or {},
        operations_console=deps.coerce_mapping(source.get("operations_console")) or {},
        alert_policy=alert_policy,
        candidate_summary=deps.coerce_mapping(source.get("candidate_summary")) or {},
        current_focus=deps.console_dashboard_focus(payload),
        required_action=str(alert_policy.get("required_action") or ""),
    )


__all__ = ["ConsoleFocusContext", "build_console_focus_context"]
