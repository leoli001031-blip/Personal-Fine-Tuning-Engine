"""Prompt state badges for the Rich operations console."""

from __future__ import annotations

from rich.text import Text

from .console_app_badges import _prompt_badge


def _prompt_state_badge(
    *,
    mode: str,
    input_active: bool,
    preview: str,
    feedback: str | None,
) -> Text:
    feedback_text = (feedback or "").strip().lower()
    if feedback_text.startswith("assistant generating"):
        return _prompt_badge("wait", "bold white on dark_blue")
    if input_active and preview:
        return _prompt_badge("compose", "bold black on yellow")
    if input_active:
        return _prompt_badge("ready", "bold white on dark_green" if mode == "chat" else "bold white on dark_blue")
    if mode == "command" and feedback_text.startswith("handled /"):
        return _prompt_badge("run", "bold white on dark_blue")
    return _prompt_badge("idle", "bold black on bright_white")


def _edit_state_badge(*, input_active: bool, raw_input: str, cursor_index: int) -> Text:
    if not input_active:
        return _prompt_badge("locked", "bold black on bright_white")
    if not raw_input:
        return _prompt_badge("blank", "bold white on dark_blue")
    if cursor_index != len(raw_input):
        return _prompt_badge("edit", "bold black on yellow")
    return _prompt_badge("typed", "bold white on dark_green")


def _activity_badge(*, mode: str, input_active: bool, feedback: str | None) -> Text:
    feedback_text = (feedback or "").strip().lower()
    if feedback_text.startswith("assistant generating"):
        return _prompt_badge("chat", "bold white on dark_blue")
    if feedback_text.startswith("running /") or (mode == "command" and feedback_text.startswith("handled /")):
        return _prompt_badge("exec", "bold white on dark_red")
    if input_active:
        return _prompt_badge("editable", "bold black on bright_white")
    return _prompt_badge("settled", "bold black on bright_white")


__all__ = ["_activity_badge", "_edit_state_badge", "_prompt_state_badge"]
