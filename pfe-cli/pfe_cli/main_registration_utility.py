"""Utility command registration wiring."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

import typer

from .main_registration_common import call
from .utility_commands import UtilityCommandDeps, register_utility_commands


def register_main_utility_commands(app: typer.Typer, symbols: MutableMapping[str, Any]) -> None:
    register_utility_commands(
        app,
        UtilityCommandDeps(
            format_doctor=lambda **kwargs: call(symbols, "_format_doctor", **kwargs),
        ),
    )
