"""Install legacy console helpers onto the main module namespace."""

from __future__ import annotations

from typing import Any

from .main_console_actions_compat import install_console_actions_compat
from .main_console_io_compat import install_console_io_compat
from .main_console_routing_compat import install_console_routing_compat
from .main_console_surface_compat import install_console_surface_compat


def install_console_compat(symbols: dict[str, Any]) -> None:
    install_console_io_compat(symbols)
    install_console_surface_compat(symbols)
    install_console_actions_compat(symbols)
    install_console_routing_compat(symbols)


__all__ = ["install_console_compat"]
