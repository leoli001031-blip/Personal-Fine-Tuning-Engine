"""Console prompt context helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .console_app_badges import _trigger_category_for_reason
from .console_app_data import _mapping, _payload_focus, _value


def _prompt_context_focus(payload: Mapping[str, Any] | None = None) -> str:
    return _payload_focus(payload).strip().lower()


def _prompt_trigger_category(
    focus: str | None = None,
    *,
    payload: Mapping[str, Any] | None = None,
) -> str:
    overview = _mapping((payload or {}).get("operations_overview"))
    console = _mapping((payload or {}).get("operations_console"))
    fallback = _value(
        console,
        "trigger_blocked_category",
        default=_value(overview, "trigger_blocked_category", default=""),
    )
    return _trigger_category_for_reason(focus, fallback=fallback) or ""


__all__ = ["_prompt_context_focus", "_prompt_trigger_category"]
