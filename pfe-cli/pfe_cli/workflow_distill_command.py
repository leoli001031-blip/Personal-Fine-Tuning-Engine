"""Distill workflow command registration."""

from __future__ import annotations

from typing import Optional

import typer

from .workflow_command_deps import WorkflowCommandDeps


def register_distill_command(app: typer.Typer, deps: WorkflowCommandDeps) -> None:
    @app.command("distill")
    def distill(
        teacher_model: str = typer.Option(..., "--teacher-model", help="Teacher model name or provider id."),
        scenario: str = typer.Option(..., "--scenario", help="Target scenario, e.g. life-coach."),
        style: str = typer.Option(..., "--style", help="Desired response style."),
        num: int = typer.Option(200, "--num", min=1, help="Number of samples to distill."),
        output: Optional[str] = typer.Option(None, "--output", help="Optional output path for distilled samples."),
        workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
        dry_run: bool = typer.Option(False, "--dry-run", help="Preview what would be distilled without executing."),
    ) -> None:
        """Run the current distillation workflow."""

        if dry_run:
            typer.echo(f"[dry-run] Would distill {num} samples with teacher={teacher_model}, scenario={scenario}, style={style}")
            return

        service = deps.load_service("pfe_core.curator", "pfe_core.pipeline", "pfe_core.services.pipeline")
        if service is None:
            deps.run_placeholder("distill")
            return

        handler = deps.resolve_handler(service, "distill", "run_distillation")
        if handler is None:
            deps.run_placeholder("distill")
            return

        deps.run_handler(
            "distill",
            handler,
            teacher_model=teacher_model,
            scenario=scenario,
            style=style,
            num_samples=num,
            output=output,
            workspace=workspace,
        )


__all__ = ["register_distill_command"]
