"""Status command wiring for runtime CLI surfaces."""

from __future__ import annotations

from typing import Optional

import typer

from .runtime_command_deps import RuntimeCommandDeps


def _teacher_distillation_status_block() -> str:
    """Render a small text block showing teacher distillation config status."""

    try:
        from pfe_core.config import PFEConfig

        cfg = PFEConfig.load()
        td = cfg.trainer.teacher_distillation
        enabled = "enabled" if td.enabled else "disabled"
        model = td.teacher_model or "(not set)"
        ratio = td.max_teacher_ratio
        threshold = td.similarity_threshold
        return (
            f"Teacher Distillation: {enabled}\n"
            f"  Model: {model}\n"
            f"  Max ratio: {ratio}\n"
            f"  Similarity threshold: {threshold}"
        )
    except Exception:
        return "Teacher Distillation: unavailable"


def register_status_command(app: typer.Typer, deps: RuntimeCommandDeps) -> None:
    @app.command("status")
    def status(
        workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
        json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON instead of formatted text."),
    ) -> None:
        """Show engine, adapter, and inference status."""

        service = deps.load_service("pfe_core.pipeline", "pfe_core.status", "pfe_server.app", "pfe_core.services.pipeline")
        if service is None:
            deps.run_placeholder("status")
            typer.echo("")
            typer.echo(_teacher_distillation_status_block())
            return

        handler = deps.resolve_handler(service, "status", "get_status")
        if handler is None:
            deps.run_placeholder("status")
            typer.echo("")
            typer.echo(_teacher_distillation_status_block())
            return

        if json_output:
            deps.run_handler_json("status", handler, workspace=workspace)
            return

        deps.run_handler("status", handler, formatter=lambda result: deps.format_status(result, workspace=workspace), workspace=workspace)
        typer.echo("")
        typer.echo(_teacher_distillation_status_block())
