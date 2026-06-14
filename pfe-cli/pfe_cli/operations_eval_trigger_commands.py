"""Auto-eval trigger command registration."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Optional

import typer

from .operations_command_deps import OperationsCommandDeps, pipeline_service, run_simple_status_command


def _format_eval_trigger_status_factory(deps: OperationsCommandDeps) -> Callable[[Any], str]:
    def _format_eval_trigger_status(result: Any) -> str:
        mapping = deps.coerce_mapping(result)
        if mapping is None:
            return deps.format_scalar(result)

        lines = ["Auto-eval trigger status"]
        enabled = mapping.get("enabled", False)
        lines.append(f"enabled: {enabled}")

        if enabled:
            auto_promote = mapping.get("auto_promote_after_eval", False)
            win_rate = mapping.get("win_rate_threshold", 0.6)
            lines.append(f"auto_promote_after_eval: {auto_promote}")
            lines.append(f"win_rate_threshold: {win_rate:.0%}")

        eval_config = deps.coerce_mapping(mapping.get("eval_config"))
        if eval_config:
            lines.append("eval_config:")
            for key, value in eval_config.items():
                lines.append(f"  {key}: {value}")

        promote_config = deps.coerce_mapping(mapping.get("promote_config"))
        if promote_config:
            lines.append("promote_config:")
            for key, value in promote_config.items():
                lines.append(f"  {key}: {value}")

        return "\n".join(lines)

    return _format_eval_trigger_status


def register_eval_trigger_commands(*, eval_trigger_app: typer.Typer, deps: OperationsCommandDeps) -> None:
    """Attach auto-eval trigger commands to the eval-trigger sub-app."""

    @eval_trigger_app.command("enable")
    def eval_trigger_enable(
        workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
    ) -> None:
        """Enable auto-eval trigger for the workspace."""

        run_simple_status_command(
            deps,
            command_name="eval-trigger enable",
            handler_name="enable_auto_eval_trigger",
            workspace=workspace,
        )

    @eval_trigger_app.command("disable")
    def eval_trigger_disable(
        workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
    ) -> None:
        """Disable auto-eval trigger for the workspace."""

        run_simple_status_command(
            deps,
            command_name="eval-trigger disable",
            handler_name="disable_auto_eval_trigger",
            workspace=workspace,
        )

    @eval_trigger_app.command("status")
    def eval_trigger_status(
        workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
    ) -> None:
        """Show auto-eval trigger status for the workspace."""

        service = pipeline_service(deps)
        if service is None:
            deps.run_placeholder("eval-trigger status")
            return

        handler = deps.resolve_handler(service, "get_auto_eval_trigger_status")
        if handler is None:
            deps.run_placeholder("eval-trigger status")
            return

        deps.run_handler(
            "eval-trigger status",
            handler,
            formatter=_format_eval_trigger_status_factory(deps),
            workspace=workspace,
        )
