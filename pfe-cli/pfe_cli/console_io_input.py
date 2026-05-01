"""TTY console input reader."""

from __future__ import annotations

import os
import select
import sys
from collections.abc import Callable, Sequence

import typer

from .console_io_editing import console_apply_edit
from .console_io_escape import apply_escape_sequence


def console_read_input(
    prompt_label: str,
    *,
    refresh_seconds: float,
    refresh_callback: Callable[[str, int], None],
    history: Sequence[str] | None = None,
) -> str:
    if not sys.stdin.isatty():
        return str(typer.prompt(prompt_label))

    import termios
    import tty

    fd = sys.stdin.fileno()
    previous_settings = termios.tcgetattr(fd)
    text = ""
    cursor = 0
    history_items = [str(item) for item in (history or []) if str(item).strip()]
    history_index: int | None = None
    history_draft = ""
    refresh_callback("", 0)
    try:
        tty.setraw(fd)
        while True:
            ready, _, _ = select.select([fd], [], [], max(refresh_seconds, 0.1))
            if not ready:
                refresh_callback(text, cursor)
                continue

            chunk = os.read(fd, 1)
            if not chunk:
                raise EOFError
            char = chunk.decode("utf-8", errors="ignore")

            if char in {"\r", "\n"}:
                return text
            if char == "\x03":
                raise KeyboardInterrupt
            if char == "\x04":
                if text:
                    continue
                raise EOFError
            if char == "\x01":
                text, cursor = console_apply_edit(text, cursor, "home")
                refresh_callback(text, cursor)
                continue
            if char == "\x05":
                text, cursor = console_apply_edit(text, cursor, "end")
                refresh_callback(text, cursor)
                continue
            if char == "\x15":
                text, cursor = console_apply_edit(text, cursor, "clear")
                history_index = None
                refresh_callback(text, cursor)
                continue
            if char == "\x17":
                text, cursor = console_apply_edit(text, cursor, "word_backspace")
                history_index = None
                refresh_callback(text, cursor)
                continue
            if char == "\x0b":
                text, cursor = console_apply_edit(text, cursor, "clear_to_end")
                refresh_callback(text, cursor)
                continue
            if char in {"\x7f", "\b"}:
                text, cursor = console_apply_edit(text, cursor, "backspace")
                history_index = None
                refresh_callback(text, cursor)
                continue
            if char == "\x1b":
                text, cursor, history_index, history_draft = apply_escape_sequence(
                    fd,
                    text=text,
                    cursor=cursor,
                    history_items=history_items,
                    history_index=history_index,
                    history_draft=history_draft,
                )
                refresh_callback(text, cursor)
                continue
            if char.isprintable():
                text, cursor = console_apply_edit(text, cursor, "insert", char)
                history_index = None
                refresh_callback(text, cursor)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, previous_settings)


__all__ = ["console_read_input"]
