"""Install legacy state helpers onto the main module namespace."""

from __future__ import annotations

from typing import Any

from .main_state_daemon_symbols import make_state_daemon_symbols
from .main_state_paths_symbols import make_state_path_symbols
from .main_state_training_symbols import make_state_training_symbols


def install_state_compat(symbols: dict[str, Any]) -> None:
    symbols.update(make_state_path_symbols(symbols))
    symbols.update(make_state_daemon_symbols(symbols))
    symbols.update(make_state_training_symbols(symbols))


__all__ = ["install_state_compat"]
