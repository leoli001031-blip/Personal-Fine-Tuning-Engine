"""Runtime command registration wiring."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

import typer

from .main_registration_common import call
from .runtime_command_deps import RuntimeCommandDeps
from .runtime_commands import register_runtime_commands


def register_main_runtime_commands(app: typer.Typer, symbols: MutableMapping[str, Any]) -> None:
    register_runtime_commands(
        app,
        RuntimeCommandDeps(
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
            run_handler_json=lambda command_name, handler, **kwargs: call(
                symbols,
                "_run_handler_json",
                command_name,
                handler,
                **kwargs,
            ),
            friendly_exception_message=lambda exc: call(symbols, "_friendly_exception_message", exc),
            coerce_mapping=lambda result: call(symbols, "_coerce_mapping", result),
            format_serve_preview=lambda **kwargs: call(symbols, "_format_serve_preview", **kwargs),
            format_serve=lambda result: call(symbols, "_format_serve", result),
            format_status=lambda result, *, workspace=None: call(
                symbols,
                "_format_status",
                result,
                workspace=workspace,
            ),
            console_snapshot_payload=lambda handler, *, workspace: call(
                symbols,
                "_console_snapshot_payload",
                handler,
                workspace=workspace,
            ),
            console_shortcut_hint=lambda mode_name, payload=None: call(
                symbols,
                "_console_shortcut_hint",
                mode_name,
                payload,
            ),
            console_read_input=lambda prompt, **kwargs: call(symbols, "_console_read_input", prompt, **kwargs),
            console_command_output=lambda command, **kwargs: call(
                symbols,
                "_console_command_output",
                command,
                **kwargs,
            ),
            console_chat_text=lambda result: call(symbols, "_console_chat_text", result),
            append_console_line=lambda lines, **kwargs: call(symbols, "_append_console_line", lines, **kwargs),
            console_submit_feedback=lambda **kwargs: call(symbols, "_console_submit_feedback", **kwargs),
        ),
    )
