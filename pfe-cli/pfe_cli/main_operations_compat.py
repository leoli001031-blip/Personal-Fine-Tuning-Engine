"""Install legacy operations formatting helpers onto the main module namespace."""

from __future__ import annotations

from typing import Any

from .main_operations_daemon_compat import install_operations_daemon_compat
from .main_operations_history_compat import install_operations_history_compat
from .main_operations_surface_compat import install_operations_surface_compat


def install_operations_format_compat(symbols: dict[str, Any]) -> None:
    install_operations_history_compat(symbols)
    install_operations_surface_compat(symbols)
    install_operations_daemon_compat(symbols)


__all__ = ["install_operations_format_compat"]
