"""Eval workflow command registration."""

from __future__ import annotations

from typing import Any, Optional

import typer

from .workflow_command_deps import WorkflowCommandDeps


def register_eval_command(app: typer.Typer, deps: WorkflowCommandDeps) -> None:
    @app.command("eval")
    def eval(
        base_model: str = typer.Option(..., "--base-model", help="Base model id or the special value 'base'."),
        adapter: str = typer.Option("latest", "--adapter", help="Adapter version to evaluate."),
        compare: Optional[str] = typer.Option(
            None,
            "--compare",
            help="Compare against another evaluated adapter version, e.g. --adapter v001 --compare v002.",
        ),
        num_samples: int = typer.Option(20, "--num-samples", min=1, help="Number of holdout/test samples."),
        workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
    ) -> None:
        """Evaluate an adapter using the future judge pipeline."""

        service = deps.load_service("pfe_core.evaluator", "pfe_core.pipeline", "pfe_core.services.pipeline")
        if service is None:
            deps.run_placeholder("eval")
            return

        handler, handler_kwargs = _resolve_eval_handler(
            deps,
            service=service,
            base_model=base_model,
            adapter=adapter,
            compare=compare,
            num_samples=num_samples,
            workspace=workspace,
        )
        if handler is None:
            return

        deps.run_handler(
            "eval",
            handler,
            formatter=lambda result: deps.format_eval_result(result, workspace=workspace or "user_default"),
            **handler_kwargs,
        )


def _resolve_eval_handler(
    deps: WorkflowCommandDeps,
    *,
    service: Any,
    base_model: str,
    adapter: str,
    compare: str | None,
    num_samples: int,
    workspace: str | None,
) -> tuple[Any | None, dict[str, Any]]:
    if compare:
        compare_handler = deps.resolve_handler(service, "compare_evaluations", "compare_eval_versions")
        if compare_handler is None:
            typer.secho("Compare-eval is unavailable because no compare handler is registered.", err=True, fg=typer.colors.RED)
            raise typer.Exit(code=1)
        return compare_handler, {
            "left_adapter": adapter,
            "right_adapter": compare,
            "workspace": workspace,
        }

    handler = deps.resolve_handler(service, "evaluate", "eval")
    if handler is None:
        deps.run_placeholder("eval")
        return None, {}
    return handler, {
        "base_model": base_model,
        "adapter": adapter,
        "num_samples": num_samples,
        "workspace": workspace,
    }


__all__ = ["register_eval_command"]
