"""Install legacy serve and training preview helpers onto the main module namespace."""

from __future__ import annotations

from typing import Any

from .main_preview_serve_symbols import make_preview_serve_symbols
from .main_preview_training_symbols import make_preview_training_symbols


def install_preview_format_compat(symbols: dict[str, Any]) -> None:
    symbols.update(make_preview_serve_symbols(symbols))
    symbols.update(make_preview_training_symbols(symbols))


__all__ = ["install_preview_format_compat"]
