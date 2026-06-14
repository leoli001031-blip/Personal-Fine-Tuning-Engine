"""Install legacy operations surface helpers onto the main module namespace."""

from __future__ import annotations

from typing import Any

from .main_operations_surface_builders import make_operations_surface_builder_symbols
from .main_operations_surface_formatters import make_operations_surface_formatter_symbols


def install_operations_surface_compat(symbols: dict[str, Any]) -> None:
    symbols.update(make_operations_surface_formatter_symbols(symbols))
    symbols.update(make_operations_surface_builder_symbols(symbols))


__all__ = ["install_operations_surface_compat"]
