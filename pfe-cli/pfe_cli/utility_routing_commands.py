"""Scenario and route debug utility commands."""

from __future__ import annotations

import typer

from .utility_route_debug_commands import register_route_debug_commands
from .utility_scenario_commands import register_scenario_commands


def register_routing_commands(app: typer.Typer) -> None:
    scenario_app = typer.Typer(help="Manage scenario configurations for the current rule-based router.")
    app.add_typer(scenario_app, name="scenario")
    register_scenario_commands(scenario_app)

    route_app = typer.Typer(help="Test and debug the current keyword/rule-based scenario router.")
    app.add_typer(route_app, name="route")
    register_route_debug_commands(route_app)


__all__ = ["register_routing_commands"]
