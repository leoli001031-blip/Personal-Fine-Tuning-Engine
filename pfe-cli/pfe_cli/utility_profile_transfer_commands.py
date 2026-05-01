"""Profile import and export commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer


def register_profile_transfer_commands(profile_app: typer.Typer) -> None:
    @profile_app.command("export")
    def profile_export(
        user_id: str = typer.Option("default", "--user-id", help="User identifier."),
        output: str = typer.Option(..., "--output", help="Output file path."),
        workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
    ) -> None:
        """Export user profile to a file."""

        del workspace
        from pfe_core.profile_extractor import get_user_profile_store

        try:
            store = get_user_profile_store()
            profile_data = store.export_profile(user_id)

            if profile_data is None:
                typer.secho(f"Profile not found for user: {user_id}", err=True, fg=typer.colors.RED)
                raise typer.Exit(code=1)

            output_path = Path(output)
            output_path.write_text(json.dumps(profile_data, ensure_ascii=False, indent=2), encoding="utf-8")
            typer.echo(f"Profile exported to: {output}")
        except Exception as exc:
            typer.secho(f"Error exporting profile: {exc}", err=True, fg=typer.colors.RED)
            raise typer.Exit(code=1)

    @profile_app.command("import")
    def profile_import(
        user_id: str = typer.Option("default", "--user-id", help="User identifier."),
        input_file: str = typer.Option(..., "--input", help="Input file path."),
        workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
    ) -> None:
        """Import user profile from a file."""

        del workspace
        from pfe_core.profile_extractor import get_user_profile_store

        try:
            input_path = Path(input_file)
            if not input_path.exists():
                typer.secho(f"Input file not found: {input_file}", err=True, fg=typer.colors.RED)
                raise typer.Exit(code=1)

            profile_data = json.loads(input_path.read_text(encoding="utf-8"))

            store = get_user_profile_store()
            store.import_profile(user_id, profile_data)

            typer.echo(f"Profile imported for user: {user_id}")
        except Exception as exc:
            typer.secho(f"Error importing profile: {exc}", err=True, fg=typer.colors.RED)
            raise typer.Exit(code=1)


__all__ = ["register_profile_transfer_commands"]
