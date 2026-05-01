"""Typer commands for utility, profile, routing, and data surfaces."""

from __future__ import annotations

import typer

from .utility_basic_commands import register_basic_utility_commands
from .utility_command_deps import UtilityCommandDeps
from .utility_data_commands import register_data_commands
from .utility_profile_commands import register_profile_commands
from .utility_routing_commands import register_routing_commands


def register_utility_commands(app: typer.Typer, deps: UtilityCommandDeps) -> None:
    """Attach utility command groups to the root CLI app."""

    register_basic_utility_commands(app, deps)
    register_profile_commands(app)
    register_routing_commands(app)
    register_data_commands(app)


__all__ = ["UtilityCommandDeps", "register_utility_commands"]
