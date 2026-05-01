"""Auto-train trigger status command."""

from __future__ import annotations

import json
from typing import Optional

import typer

from . import formatters_matrix
from .operations_command_deps import OperationsCommandDeps, pipeline_service


def register_trigger_status_commands(*, trigger_app: typer.Typer, deps: OperationsCommandDeps) -> None:
    """Attach trigger status command."""

    @trigger_app.command("status")
    def trigger_status(
        workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
        json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON instead of formatted text."),
    ) -> None:
        """Show auto-train trigger status (thresholds, blocked reasons, queue state)."""

        service = pipeline_service(deps)
        if service is None:
            deps.run_placeholder("trigger status")
            return

        handler = deps.resolve_handler(service, "status")
        if handler is None:
            deps.run_placeholder("trigger status")
            return

        try:
            result = handler(workspace=workspace)
        except Exception as exc:
            friendly = deps.friendly_exception_message(exc)
            if friendly is not None:
                typer.secho(friendly, err=True, fg=typer.colors.RED)
                raise typer.Exit(code=1)
            raise

        mapping = deps.coerce_mapping(result) or {}
        trigger_data = deps.coerce_mapping(mapping.get("auto_train_trigger"))
        if trigger_data is None:
            typer.echo("No auto-train trigger status available.")
            return

        if json_output:
            typer.echo(json.dumps(trigger_data, ensure_ascii=False, indent=2, sort_keys=True))
            return

        typer.echo(formatters_matrix.format_status_matrix({"auto_train_trigger": trigger_data}, workspace=workspace))


__all__ = ["register_trigger_status_commands"]
