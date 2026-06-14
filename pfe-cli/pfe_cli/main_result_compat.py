"""Install legacy result and doctor formatting helpers onto the main module namespace."""

from __future__ import annotations

from typing import Any

from .main_result_doctor_symbols import make_result_doctor_symbols
from .main_result_legacy_symbols import make_result_legacy_symbols
from .main_result_snapshot_symbols import make_result_snapshot_symbols


def install_result_format_compat(symbols: dict[str, Any]) -> None:
    symbols.update(make_result_snapshot_symbols(symbols))
    symbols.update(make_result_legacy_symbols(symbols))
    symbols.update(make_result_doctor_symbols(symbols))


__all__ = ["install_result_format_compat"]
