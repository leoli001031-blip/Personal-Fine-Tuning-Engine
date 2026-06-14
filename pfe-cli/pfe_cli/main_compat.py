"""Install legacy private CLI helpers onto the main module namespace."""

from __future__ import annotations

from .main_command_compat import install_command_compat
from .main_console_compat import install_console_compat
from .main_state_compat import install_state_compat

__all__ = [
    "install_command_compat",
    "install_console_compat",
    "install_state_compat",
]
