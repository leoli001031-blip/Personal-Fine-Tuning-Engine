"""Compact prompt digest helpers."""

from __future__ import annotations

from .console_app_data import _display_focus_name


def _prompt_feedback_digest(feedback: str | None = None) -> str:
    normalized = (feedback or "").strip()
    lower = normalized.lower()
    if not normalized:
        return "idle"
    if lower.startswith("assistant generating"):
        return "generating"
    if lower.startswith("running /"):
        return normalized.split(" ", 1)[1]
    if len(normalized) > 24:
        return normalized[:21] + "..."
    return normalized


def _prompt_ctx_digest(focus: str | None = None) -> str:
    normalized_focus = _display_focus_name(focus).strip().lower()
    if not normalized_focus or normalized_focus in {"none", "idle", "stable"}:
        return ""
    if len(normalized_focus) > 18:
        return normalized_focus[:15] + "..."
    return normalized_focus


def _prompt_hint_digest(shortcut_hint: str | None = None) -> str:
    normalized = (shortcut_hint or "").strip()
    if not normalized:
        return ""
    parts = [part.strip() for part in normalized.split(",") if part.strip()]
    if not parts:
        return ""
    compact = ",".join(parts[:2])
    if len(compact) > 18:
        return compact[:15] + "..."
    return compact


def _prompt_model_digest(value: str | None) -> str:
    text = (value or "local").strip()
    if len(text) > 12:
        return text[:9] + "..."
    return text


def _prompt_adapter_digest(value: str | None) -> str:
    text = (value or "latest").strip()
    if len(text) > 12:
        return text[:9] + "..."
    return text


__all__ = [
    "_prompt_adapter_digest",
    "_prompt_ctx_digest",
    "_prompt_feedback_digest",
    "_prompt_hint_digest",
    "_prompt_model_digest",
]
