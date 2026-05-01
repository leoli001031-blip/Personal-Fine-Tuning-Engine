"""Install legacy status formatting helpers onto the main module namespace."""

from __future__ import annotations

from typing import Any

from .main_status_format_symbols import make_status_format_symbols
from .main_status_plan_symbols import make_status_plan_symbols


def install_status_format_compat(symbols: dict[str, Any]) -> None:
    symbols.update(make_status_plan_symbols(symbols))
    symbols.update(make_status_format_symbols(symbols))


__all__ = ["install_status_format_compat"]
