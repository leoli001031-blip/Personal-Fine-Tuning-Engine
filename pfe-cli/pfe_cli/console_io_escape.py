"""ANSI escape sequence handling for the TTY console input reader."""

from __future__ import annotations

import os
import select
from collections.abc import Sequence

from .console_io_editing import console_apply_edit, console_apply_history


def _read_escape_sequence(fd: int) -> str:
    ready_more, _, _ = select.select([fd], [], [], 0.01)
    sequence = ""
    while ready_more:
        next_chunk = os.read(fd, 1)
        if not next_chunk:
            break
        sequence += next_chunk.decode("utf-8", errors="ignore")
        ready_more, _, _ = select.select([fd], [], [], 0.01)
        if sequence and sequence[-1].isalpha():
            break
        if sequence.endswith("~"):
            break
    return sequence


def apply_escape_sequence(
    fd: int,
    *,
    text: str,
    cursor: int,
    history_items: Sequence[str],
    history_index: int | None,
    history_draft: str,
) -> tuple[str, int, int | None, str]:
    sequence = _read_escape_sequence(fd)
    if sequence == "[A":
        return console_apply_history(
            text,
            cursor,
            history=history_items,
            history_index=history_index,
            history_draft=history_draft,
            event="up",
        )
    if sequence == "[B":
        return console_apply_history(
            text,
            cursor,
            history=history_items,
            history_index=history_index,
            history_draft=history_draft,
            event="down",
        )
    if sequence == "[C":
        text, cursor = console_apply_edit(text, cursor, "right")
    elif sequence == "[D":
        text, cursor = console_apply_edit(text, cursor, "left")
    elif sequence == "[H":
        text, cursor = console_apply_edit(text, cursor, "home")
    elif sequence == "[F":
        text, cursor = console_apply_edit(text, cursor, "end")
    elif sequence == "[3~":
        text, cursor = console_apply_edit(text, cursor, "delete")
    return text, cursor, history_index, history_draft


__all__ = ["apply_escape_sequence"]
