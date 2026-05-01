"""Shared execution helper for daemon monitoring commands."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import typer

from .operations_command_deps import OperationsCommandDeps, pipeline_service


def run_daemon_monitor_handler(
    deps: OperationsCommandDeps,
    *,
    command_name: str,
    handler_names: tuple[str, ...],
    formatter: Callable[..., Any],
    json_output: bool,
    unavailable_message: str,
    **kwargs: Any,
) -> None:
    service = pipeline_service(deps)
    if service is not None:
        handler = deps.resolve_handler(service, *handler_names)
        if handler is not None:
            if json_output:
                deps.run_handler_json(command_name, handler, **kwargs)
                return
            deps.run_handler(command_name, handler, formatter=formatter, **kwargs)
            return

    typer.echo(unavailable_message)


__all__ = ["run_daemon_monitor_handler"]
