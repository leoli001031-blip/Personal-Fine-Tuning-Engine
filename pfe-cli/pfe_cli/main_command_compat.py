"""Install legacy command execution helpers onto the main module namespace."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from .command_execution import resolve_handler, run_handler, run_handler_json, run_placeholder
from .main_deps import make_command_execution_deps


def _call(symbols: Mapping[str, Any], name: str, *args: Any, **kwargs: Any) -> Any:
    return symbols[name](*args, **kwargs)


def install_command_compat(symbols: dict[str, Any]) -> None:
    def _command_execution_deps() -> Any:
        return make_command_execution_deps(symbols)

    def _run_handler(
        command_name: str,
        handler: Callable[..., Any],
        formatter: Callable[[Any], str] | None = None,
        on_result: Callable[[Any], None] | None = None,
        **kwargs: Any,
    ) -> None:
        return run_handler(
            command_name,
            handler,
            formatter=formatter,
            on_result=on_result,
            deps=_call(symbols, "_command_execution_deps"),
            **kwargs,
        )

    def _run_handler_json(command_name: str, handler: Callable[..., Any], **kwargs: Any) -> None:
        return run_handler_json(
            command_name,
            handler,
            deps=_call(symbols, "_command_execution_deps"),
            **kwargs,
        )

    symbols.update(
        {
            "_command_execution_deps": _command_execution_deps,
            "_run_handler": _run_handler,
            "_run_handler_json": _run_handler_json,
            "_run_placeholder": run_placeholder,
            "_resolve_handler": resolve_handler,
        }
    )
