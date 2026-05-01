"""Profile list and delete commands."""

from __future__ import annotations

import json
from typing import Optional

import typer


def register_profile_management_commands(profile_app: typer.Typer) -> None:
    @profile_app.command("list")
    def profile_list(
        json_output: bool = typer.Option(False, "--json", help="Emit JSON output."),
        workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
    ) -> None:
        """List all user profiles."""

        del workspace
        from pfe_core.profile_extractor import get_user_profile_store

        try:
            store = get_user_profile_store()
            profiles = store.list_profiles()

            if json_output:
                typer.echo(json.dumps({"profiles": profiles}, ensure_ascii=False, indent=2))
                return

            if not profiles:
                typer.echo("No profiles found.")
                return

            typer.echo("User Profiles:")
            for profile_id in profiles:
                typer.echo(f"  - {profile_id}")
        except Exception as exc:
            typer.secho(f"Error listing profiles: {exc}", err=True, fg=typer.colors.RED)
            raise typer.Exit(code=1)

    @profile_app.command("delete")
    def profile_delete(
        user_id: str = typer.Option("default", "--user-id", help="User identifier."),
        workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
        yes: bool = typer.Option(False, "--yes", help="Skip confirmation."),
    ) -> None:
        """Delete a user profile."""

        del workspace
        from pfe_core.profile_extractor import get_user_profile_store

        try:
            store = get_user_profile_store()

            if not yes:
                confirmed = typer.confirm(f"Are you sure you want to delete profile '{user_id}'?")
                if not confirmed:
                    typer.echo("Deletion cancelled.")
                    return

            success = store.delete_profile(user_id)

            if success:
                typer.echo(f"Profile deleted: {user_id}")
            else:
                typer.secho(f"Profile not found: {user_id}", err=True, fg=typer.colors.YELLOW)
        except Exception as exc:
            typer.secho(f"Error deleting profile: {exc}", err=True, fg=typer.colors.RED)
            raise typer.Exit(code=1)


__all__ = ["register_profile_management_commands"]
