"""Profile analysis utility command registration."""

from __future__ import annotations

import typer

from .utility_profile_analysis_commands import register_profile_analysis_commands
from .utility_profile_management_commands import register_profile_management_commands
from .utility_profile_show_commands import register_profile_show_commands
from .utility_profile_transfer_commands import register_profile_transfer_commands


def register_profile_commands(app: typer.Typer) -> None:
    profile_app = typer.Typer(
        help=(
            "Manage rule-based profile analysis snapshots. Runtime prompt injection still uses "
            "user_memory as the primary user-modeling path."
        )
    )
    app.add_typer(profile_app, name="profile")

    register_profile_show_commands(profile_app)
    register_profile_analysis_commands(profile_app)
    register_profile_transfer_commands(profile_app)
    register_profile_management_commands(profile_app)


__all__ = ["register_profile_commands"]
