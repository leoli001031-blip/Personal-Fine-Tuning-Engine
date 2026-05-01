"""Runtime state badge primitives for console rendering."""

from __future__ import annotations

from rich.text import Text

from .console_app_badge_trigger import _state_badge


def _runtime_state_badge(kind: str, value: str | None) -> Text:
    normalized = (value or "n/a").strip().lower()
    if kind == "runner":
        if normalized == "stale":
            return _state_badge(normalized, style="bold black on yellow")
        if normalized == "active":
            return _state_badge(normalized, style="bold white on dark_blue")
        return _state_badge(normalized, style="bold black on bright_white")
    if kind == "health":
        if normalized in {"stale", "blocked"}:
            return _state_badge(normalized, style="bold white on dark_red")
        if normalized in {"recovering"}:
            return _state_badge(normalized, style="bold white on dark_blue")
        return _state_badge(normalized, style="bold black on bright_white")
    if kind in {"heartbeat", "lease"}:
        if normalized in {"stale", "expired"}:
            return _state_badge(normalized, style="bold white on dark_red")
        if normalized in {"delayed", "expiring"}:
            return _state_badge(normalized, style="bold black on yellow")
        return _state_badge(normalized, style="bold black on bright_white")
    if kind == "restart":
        if normalized in {"backoff", "capped"}:
            return _state_badge(normalized, style="bold black on yellow")
        return _state_badge(normalized, style="bold black on bright_white")
    if kind == "recover":
        if normalized not in {"none", "n/a"}:
            return _state_badge(normalized, style="bold white on dark_blue")
        return _state_badge(normalized, style="bold black on bright_white")
    return _state_badge(normalized, style="bold black on bright_white")


__all__ = ["_runtime_state_badge"]
