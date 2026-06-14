"""Compatibility exports for console input and transcript helpers."""

from __future__ import annotations

from .console_io_deps import ConsoleIODeps
from .console_io_editing import console_apply_edit, console_apply_history
from .console_io_input import console_read_input
from .console_io_text import append_console_line, console_chat_text, console_snapshot_payload

__all__ = [
    "ConsoleIODeps",
    "append_console_line",
    "console_apply_edit",
    "console_apply_history",
    "console_chat_text",
    "console_read_input",
    "console_snapshot_payload",
]
