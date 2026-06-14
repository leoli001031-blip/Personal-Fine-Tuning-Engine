"""Install legacy private formatting helpers onto the main module namespace."""

from __future__ import annotations

from typing import Any

from .main_operations_compat import install_operations_format_compat
from .main_preview_compat import install_preview_format_compat
from .main_result_compat import install_result_format_compat
from .main_status_compat import install_status_format_compat


def install_format_compat(symbols: dict[str, Any]) -> None:
    install_status_format_compat(symbols)
    install_operations_format_compat(symbols)
    install_result_format_compat(symbols)
    install_preview_format_compat(symbols)
