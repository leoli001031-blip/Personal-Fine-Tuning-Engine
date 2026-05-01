"""Console line editing and history navigation helpers."""

from __future__ import annotations

from collections.abc import Sequence


def console_apply_edit(text: str, cursor: int, event: str, value: str | None = None) -> tuple[str, int]:
    current = str(text or "")
    cursor = max(0, min(int(cursor), len(current)))
    if event == "insert" and value:
        updated = current[:cursor] + value + current[cursor:]
        return updated, cursor + len(value)
    if event == "left":
        return current, max(0, cursor - 1)
    if event == "right":
        return current, min(len(current), cursor + 1)
    if event == "home":
        return current, 0
    if event == "end":
        return current, len(current)
    if event == "backspace":
        if cursor <= 0:
            return current, cursor
        updated = current[: cursor - 1] + current[cursor:]
        return updated, cursor - 1
    if event == "delete":
        if cursor >= len(current):
            return current, cursor
        updated = current[:cursor] + current[cursor + 1 :]
        return updated, cursor
    if event == "clear":
        return "", 0
    if event == "clear_to_end":
        if cursor >= len(current):
            return current, cursor
        return current[:cursor], cursor
    if event == "word_backspace":
        if cursor <= 0:
            return current, cursor
        trimmed = current[:cursor].rstrip()
        suffix = current[cursor:]
        last_space = trimmed.rfind(" ")
        start = 0 if last_space < 0 else last_space + 1
        updated = current[:start] + suffix
        return updated, start
    return current, cursor


def console_apply_history(
    text: str,
    cursor: int,
    *,
    history: Sequence[str] | None,
    history_index: int | None,
    history_draft: str,
    event: str,
) -> tuple[str, int, int | None, str]:
    history_items = [str(item) for item in (history or []) if str(item).strip()]
    current = str(text or "")
    cursor = max(0, min(int(cursor), len(current)))
    if not history_items:
        return current, cursor, history_index, history_draft

    if event == "up":
        if history_index is None:
            history_draft = current
            history_index = len(history_items) - 1
        elif history_index > 0:
            history_index -= 1
        current = history_items[history_index]
        return current, len(current), history_index, history_draft

    if event == "down" and history_index is not None:
        if history_index < len(history_items) - 1:
            history_index += 1
            current = history_items[history_index]
        else:
            history_index = None
            current = history_draft
        return current, len(current), history_index, history_draft

    return current, cursor, history_index, history_draft


__all__ = ["console_apply_edit", "console_apply_history"]
