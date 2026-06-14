"""Install legacy console IO helpers onto the main module namespace."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from .console_feedback import console_submit_feedback
from .console_io import (
    append_console_line,
    console_apply_edit,
    console_apply_history,
    console_chat_text,
    console_read_input,
    console_snapshot_payload,
)
from .main_deps import make_console_io_deps


def _call(symbols: Mapping[str, Any], name: str, *args: Any, **kwargs: Any) -> Any:
    return symbols[name](*args, **kwargs)


def install_console_io_compat(symbols: dict[str, Any]) -> None:
    def _console_io_deps() -> Any:
        return make_console_io_deps(symbols)

    def _console_chat_text(result: Any) -> str:
        return console_chat_text(result, deps=_call(symbols, "_console_io_deps"))

    def _console_snapshot_payload(handler: Callable[..., Any], *, workspace: str | None) -> dict[str, Any]:
        return console_snapshot_payload(handler, workspace=workspace, deps=_call(symbols, "_console_io_deps"))

    def _console_submit_feedback(
        workspace: str,
        session_id: str,
        request_id: str,
        user_message: str,
        assistant_message: str,
        response_time_seconds: float,
        adapter_version: str,
        action: str,
        edited_text: str | None = None,
    ) -> list[dict[str, Any]]:
        return console_submit_feedback(
            workspace=workspace,
            session_id=session_id,
            request_id=request_id,
            user_message=user_message,
            assistant_message=assistant_message,
            response_time_seconds=response_time_seconds,
            adapter_version=adapter_version,
            action=action,
            edited_text=edited_text,
        )

    symbols.update(
        {
            "_console_io_deps": _console_io_deps,
            "_console_chat_text": _console_chat_text,
            "_append_console_line": append_console_line,
            "_console_snapshot_payload": _console_snapshot_payload,
            "_console_apply_edit": console_apply_edit,
            "_console_apply_history": console_apply_history,
            "_console_read_input": console_read_input,
            "_console_submit_feedback": _console_submit_feedback,
        }
    )


__all__ = ["install_console_io_compat"]
