"""Builder symbols for main operations surface compatibility."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .main_operations_surface_common import call
from .operations_formatting import build_operations_alert_surface, build_operations_console_digest


def make_operations_surface_builder_symbols(symbols: dict[str, Any]) -> dict[str, Any]:
    def _build_operations_console_digest(
        *,
        operations_console: Mapping[str, Any] | None,
        operations_overview: Mapping[str, Any] | None,
        operations_dashboard: Mapping[str, Any] | None = None,
        operations_alert_policy: Mapping[str, Any] | None = None,
        candidate_summary: Mapping[str, Any] | None,
        candidate_history: Mapping[str, Any] | None,
        candidate_timeline: Mapping[str, Any] | None,
        daemon_timeline: Mapping[str, Any] | None,
        runner_timeline: Mapping[str, Any] | None,
        train_queue: Mapping[str, Any] | None,
    ) -> dict[str, Any] | None:
        return build_operations_console_digest(
            operations_console=operations_console,
            operations_overview=operations_overview,
            operations_dashboard=operations_dashboard,
            operations_alert_policy=operations_alert_policy,
            candidate_summary=candidate_summary,
            candidate_history=candidate_history,
            candidate_timeline=candidate_timeline,
            daemon_timeline=daemon_timeline,
            runner_timeline=runner_timeline,
            train_queue=train_queue,
            deps=call(symbols, "_operations_formatting_deps"),
        )

    def _build_operations_alert_surface(
        *,
        operations_alerts: Any | None,
        operations_health: Any | None,
        operations_recovery: Any | None,
        operations_next_actions: Any | None,
        operations_dashboard: Mapping[str, Any] | None,
        operations_alert_policy: Mapping[str, Any] | None,
        operations_console: Mapping[str, Any] | None,
        operations_overview: Mapping[str, Any] | None,
    ) -> dict[str, Any] | None:
        return build_operations_alert_surface(
            operations_alerts=operations_alerts,
            operations_health=operations_health,
            operations_recovery=operations_recovery,
            operations_next_actions=operations_next_actions,
            operations_dashboard=operations_dashboard,
            operations_alert_policy=operations_alert_policy,
            operations_console=operations_console,
            operations_overview=operations_overview,
            deps=call(symbols, "_operations_formatting_deps"),
        )

    return {
        "_build_operations_console_digest": _build_operations_console_digest,
        "_build_operations_alert_surface": _build_operations_alert_surface,
    }


__all__ = ["make_operations_surface_builder_symbols"]
