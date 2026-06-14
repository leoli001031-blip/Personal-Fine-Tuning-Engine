"""Shared dependency contract for runtime CLI commands."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RuntimeCommandDeps:
    """Runtime hooks supplied by the main CLI module."""

    load_service: Callable[..., Any | None]
    run_placeholder: Callable[[str], None]
    resolve_handler: Callable[..., Any | None]
    run_handler: Callable[..., None]
    run_handler_json: Callable[..., None]
    friendly_exception_message: Callable[[Exception], str | None]
    coerce_mapping: Callable[[Any], dict[str, Any] | None]
    format_serve_preview: Callable[..., str]
    format_serve: Callable[[Any], str]
    format_status: Callable[..., str]
    console_snapshot_payload: Callable[..., dict[str, Any]]
    console_shortcut_hint: Callable[..., str]
    console_read_input: Callable[..., Any]
    console_command_output: Callable[..., tuple[str | None, str, dict[str, Any] | None]]
    console_chat_text: Callable[[Any], str]
    append_console_line: Callable[..., None]
    console_submit_feedback: Callable[..., list[dict[str, Any]]]
