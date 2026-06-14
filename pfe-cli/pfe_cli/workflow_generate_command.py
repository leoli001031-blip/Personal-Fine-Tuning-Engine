"""Generate workflow command registration."""

from __future__ import annotations

from typing import Optional

import typer

from .workflow_command_deps import WorkflowCommandDeps


def register_generate_command(app: typer.Typer, deps: WorkflowCommandDeps) -> None:
    @app.command("generate")
    def generate(
        scenario: str = typer.Option(..., "--scenario", help="Target scenario, e.g. life-coach."),
        style: str = typer.Option(..., "--style", help="Desired response style."),
        num: int = typer.Option(200, "--num", min=1, help="Number of samples to generate."),
        output: Optional[str] = typer.Option(None, "--output", help="Optional output path for generated samples."),
        workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
    ) -> None:
        """Generate cold-start samples using the current bootstrap workflow."""

        service = deps.load_service("pfe_core.pipeline", "pfe_core.curator", "pfe_core.services.pipeline")
        if service is None:
            deps.run_placeholder("generate")
            return

        handler = deps.resolve_handler(service, "generate")
        if handler is None:
            deps.run_placeholder("generate")
            return

        deps.run_handler(
            "generate",
            handler,
            scenario=scenario,
            style=style,
            num_samples=num,
            output=output,
            workspace=workspace,
        )


__all__ = ["register_generate_command"]
