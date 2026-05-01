"""Install legacy daemon formatting helpers onto the main module namespace."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .daemon_formatting import (
    format_daemon_alerts,
    format_daemon_health_status,
    format_daemon_heartbeat_status,
    format_daemon_lease_status,
    format_daemon_stale_check,
)
from .main_deps import make_daemon_formatting_deps


def _call(symbols: Mapping[str, Any], name: str, *args: Any, **kwargs: Any) -> Any:
    return symbols[name](*args, **kwargs)


def install_operations_daemon_compat(symbols: dict[str, Any]) -> None:
    def _daemon_formatting_deps() -> Any:
        return make_daemon_formatting_deps(symbols)

    def _format_daemon_health_status(result: Any) -> str:
        return format_daemon_health_status(result, deps=_call(symbols, "_daemon_formatting_deps"))

    def _format_daemon_heartbeat_status(result: Any) -> str:
        return format_daemon_heartbeat_status(result, deps=_call(symbols, "_daemon_formatting_deps"))

    def _format_daemon_lease_status(result: Any) -> str:
        return format_daemon_lease_status(result, deps=_call(symbols, "_daemon_formatting_deps"))

    def _format_daemon_stale_check(result: Any) -> str:
        return format_daemon_stale_check(result, deps=_call(symbols, "_daemon_formatting_deps"))

    def _format_daemon_alerts(result: Any) -> str:
        return format_daemon_alerts(result, deps=_call(symbols, "_daemon_formatting_deps"))

    symbols.update(
        {
            "_daemon_formatting_deps": _daemon_formatting_deps,
            "_format_daemon_health_status": _format_daemon_health_status,
            "_format_daemon_heartbeat_status": _format_daemon_heartbeat_status,
            "_format_daemon_lease_status": _format_daemon_lease_status,
            "_format_daemon_stale_check": _format_daemon_stale_check,
            "_format_daemon_alerts": _format_daemon_alerts,
        }
    )


__all__ = ["install_operations_daemon_compat"]
