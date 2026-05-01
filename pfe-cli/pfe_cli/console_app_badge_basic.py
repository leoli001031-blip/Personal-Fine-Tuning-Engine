"""Basic Rich badge helpers for console rendering."""

from __future__ import annotations

from rich.text import Text

from .console_app_data import _display_focus_name


def _prompt_badge(label: str, style: str) -> Text:
    badge = Text()
    badge.append(" ")
    badge.append(label.upper(), style=style)
    badge.append(" ")
    return badge


def _ops_state_badge(state: str | None) -> Text:
    normalized = (state or "live").strip().lower()
    style = {
        "live": "bold white on dark_green",
        "cached": "bold black on bright_white",
        "syncing": "bold white on dark_blue",
    }.get(normalized, "bold white on dark_green")
    return _prompt_badge(normalized, style)


def _ops_badge(state: str | None, *, severity: str | None = None) -> Text:
    normalized = (state or "live").strip().lower()
    severity_name = (severity or "stable").strip().lower()
    if severity_name == "critical":
        return _prompt_badge(f"{normalized} !", "bold white on dark_red")
    if severity_name == "warning":
        return _prompt_badge(f"{normalized} !", "bold black on yellow")
    return _ops_state_badge(normalized)


def _focus_badge(focus: str | None, *, severity: str | None = None) -> Text:
    normalized = (_display_focus_name(focus) or "none").strip().lower()
    label = "focus" if normalized in {"none", "idle", "stable"} else normalized.replace("_", " ")
    severity_name = (severity or "stable").strip().lower()
    if severity_name == "critical":
        return _prompt_badge(label, "bold white on dark_red")
    if severity_name == "warning":
        return _prompt_badge(label, "bold black on yellow")
    return _prompt_badge(label, "bold black on bright_white")


def _action_badge(action: str | None, *, priority: str | None = None) -> Text:
    normalized = (action or "observe_and_monitor").strip().lower()
    label = "monitor" if normalized in {"observe_and_monitor", "none"} else normalized.replace("_", " ")
    priority_name = (priority or "p2").strip().lower()
    style = {
        "p0": "bold white on dark_red",
        "p1": "bold black on yellow",
        "p2": "bold white on dark_blue",
    }.get(priority_name, "bold black on bright_white")
    return _prompt_badge(label, style)


def _severity_badge(severity: str | None) -> Text:
    normalized = (severity or "stable").strip().lower()
    style = {
        "critical": "bold white on dark_red",
        "warning": "bold black on yellow",
        "info": "bold white on dark_blue",
        "stable": "bold black on bright_white",
    }.get(normalized, "bold black on bright_white")
    return _prompt_badge(normalized, style)


def _section_label(title: str, *, badge: Text | None = None) -> Text:
    label = Text()
    label.append(title.upper(), style="bold bright_white")
    if badge is not None:
        label.append(" ", style="dim")
        label.append_text(badge)
    return label


__all__ = [
    "_action_badge",
    "_focus_badge",
    "_ops_badge",
    "_ops_state_badge",
    "_prompt_badge",
    "_section_label",
    "_severity_badge",
]
