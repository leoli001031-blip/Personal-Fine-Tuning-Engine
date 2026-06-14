"""Prompt footer and sidebar digest rendering."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from rich.text import Text

from .console_app_prompt_footer_rendering import render_footer_digest
from .console_app_prompt_footer_snapshot import sidebar_snapshot_text
from .console_app_prompt_footer_state import build_footer_digest_state


def _sidebar_snapshot_text(*, ops_refresh_state: str | None, ops_age_seconds: float | None, refresh_seconds: float | None) -> Text:
    return sidebar_snapshot_text(
        ops_refresh_state=ops_refresh_state,
        ops_age_seconds=ops_age_seconds,
        refresh_seconds=refresh_seconds,
    )


def _footer_digest(
    payload: Mapping[str, Any],
    *,
    interactive: bool = False,
    mode: str = "chat",
    ops_refresh_state: str | None = None,
    ops_age_seconds: float | None = None,
) -> Text:
    state = build_footer_digest_state(
        payload,
        interactive=interactive,
        mode=mode,
        ops_refresh_state=ops_refresh_state,
    )
    return render_footer_digest(state, ops_age_seconds=ops_age_seconds)


__all__ = ["_footer_digest", "_sidebar_snapshot_text"]
