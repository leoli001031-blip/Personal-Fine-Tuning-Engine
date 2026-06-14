"""Trigger category badges for console rendering."""

from __future__ import annotations

from rich.text import Text

from .console_app_badge_basic import _prompt_badge


def _state_badge(label: str | None, *, style: str) -> Text:
    return _prompt_badge((label or "n/a").replace("_", " "), style)


def _trigger_category_badge(category: str | None) -> Text:
    normalized = (category or "n/a").strip().lower()
    style = {
        "data": "bold white on dark_blue",
        "timing": "bold black on yellow",
        "recovery": "bold white on dark_red",
        "queue": "bold black on bright_white",
        "config": "bold white on dark_magenta",
    }.get(normalized, "bold black on bright_white")
    return _state_badge(normalized, style=style)


def _trigger_category_for_reason(reason: str | None, *, fallback: str | None = None) -> str | None:
    normalized = (reason or "").strip().lower()
    if normalized in {"insufficient_new_signal_samples", "holdout_not_ready"}:
        return "data"
    if normalized in {"cooldown_active", "wait_for_retrain_interval"}:
        return "timing"
    if normalized in {"failure_backoff_active", "wait_for_failure_backoff"}:
        return "recovery"
    if normalized in {
        "queue_pending_review",
        "queue_waiting_execution",
        "queue_processing_active",
        "review_required_before_execution",
    }:
        return "queue"
    if normalized in {"policy_requires_auto_evaluate", "trigger_disabled"}:
        return "config"
    normalized_fallback = (fallback or "").strip().lower()
    return normalized_fallback or None


__all__ = ["_state_badge", "_trigger_category_badge", "_trigger_category_for_reason"]
