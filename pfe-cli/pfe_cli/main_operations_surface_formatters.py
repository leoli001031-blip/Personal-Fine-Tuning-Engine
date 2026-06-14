"""Formatter symbols for main operations surface compatibility."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .main_deps import make_operations_formatting_deps
from .main_operations_surface_common import call
from .operations_formatting import (
    format_ops_attention,
    format_operations_alert_policy,
    format_operations_alert_surface,
    format_operations_console_digest,
    format_operations_dashboard,
    format_operations_event_stream,
    format_operations_timeline,
)


def make_operations_surface_formatter_symbols(symbols: dict[str, Any]) -> dict[str, Any]:
    def _operations_formatting_deps() -> Any:
        return make_operations_formatting_deps(symbols)

    def _format_operations_event_stream(result: Any) -> list[str] | None:
        return format_operations_event_stream(result, deps=call(symbols, "_operations_formatting_deps"))

    def _format_operations_dashboard(result: Any) -> list[str] | None:
        return format_operations_dashboard(result, deps=call(symbols, "_operations_formatting_deps"))

    def _format_operations_alert_policy(result: Any) -> list[str] | None:
        return format_operations_alert_policy(result, deps=call(symbols, "_operations_formatting_deps"))

    def _format_operations_timeline(result: Any) -> list[str] | None:
        return format_operations_timeline(result, deps=call(symbols, "_operations_formatting_deps"))

    def _format_ops_attention(
        *,
        operations_alerts: Any | None,
        operations_overview: Mapping[str, Any] | None,
        operations_dashboard: Mapping[str, Any] | None,
        operations_alert_policy: Mapping[str, Any] | None,
        candidate_summary: Mapping[str, Any] | None,
        train_queue: Mapping[str, Any] | None,
        latest_adapter_map: Mapping[str, Any] | None,
        recent_adapter_map: Mapping[str, Any] | None,
    ) -> str | None:
        return format_ops_attention(
            operations_alerts=operations_alerts,
            operations_overview=operations_overview,
            operations_dashboard=operations_dashboard,
            operations_alert_policy=operations_alert_policy,
            candidate_summary=candidate_summary,
            train_queue=train_queue,
            latest_adapter_map=latest_adapter_map,
            recent_adapter_map=recent_adapter_map,
            deps=call(symbols, "_operations_formatting_deps"),
        )

    def _format_operations_console_digest(result: Any) -> list[str] | None:
        return format_operations_console_digest(result, deps=call(symbols, "_operations_formatting_deps"))

    def _format_operations_alert_surface(result: Any) -> list[str] | None:
        return format_operations_alert_surface(result, deps=call(symbols, "_operations_formatting_deps"))

    return {
        "_operations_formatting_deps": _operations_formatting_deps,
        "_format_operations_event_stream": _format_operations_event_stream,
        "_format_operations_dashboard": _format_operations_dashboard,
        "_format_operations_alert_policy": _format_operations_alert_policy,
        "_format_operations_timeline": _format_operations_timeline,
        "_format_ops_attention": _format_ops_attention,
        "_format_operations_console_digest": _format_operations_console_digest,
        "_format_operations_alert_surface": _format_operations_alert_surface,
    }


__all__ = ["make_operations_surface_formatter_symbols"]
