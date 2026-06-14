"""Workflow command registration wiring."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

import typer

from .main_registration_common import call
from .workflow_commands import WorkflowCommandDeps, register_workflow_commands


def register_main_workflow_commands(app: typer.Typer, symbols: MutableMapping[str, Any]) -> None:
    register_workflow_commands(
        app,
        WorkflowCommandDeps(
            load_service=lambda *module_names: call(symbols, "_load_service", *module_names),
            run_placeholder=lambda command_name: call(symbols, "_run_placeholder", command_name),
            resolve_handler=lambda service, *names: call(symbols, "_resolve_handler", service, *names),
            run_handler=lambda command_name, handler, **kwargs: call(
                symbols,
                "_run_handler",
                command_name,
                handler,
                **kwargs,
            ),
            format_eval_result=lambda result, *, workspace=None: call(
                symbols,
                "_format_eval_result",
                result,
                workspace=workspace,
            ),
        ),
    )
