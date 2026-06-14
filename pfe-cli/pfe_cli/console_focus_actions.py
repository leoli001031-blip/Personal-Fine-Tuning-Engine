"""Console focus-to-action decision helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .console_actions_deps import ConsoleActionsDeps
from .console_focus_context import build_console_focus_context
from .console_focus_fallback_actions import fallback_console_focus_actions
from .console_focus_summary_actions import mapped_console_focus_actions


def console_focus_actions(
    payload: Mapping[str, Any] | None = None,
    *,
    deps: ConsoleActionsDeps,
) -> dict[str, str | None]:
    context = build_console_focus_context(payload, deps=deps)
    mapped_actions = mapped_console_focus_actions(context, deps=deps)
    if mapped_actions is not None:
        return mapped_actions
    return fallback_console_focus_actions(context)


__all__ = ["console_focus_actions"]
