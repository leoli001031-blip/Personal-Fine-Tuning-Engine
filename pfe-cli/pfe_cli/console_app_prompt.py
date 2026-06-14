"""Compatibility exports for prompt and footer rendering."""

from __future__ import annotations

from .console_app_prompt_badges import _activity_badge, _edit_state_badge, _prompt_state_badge
from .console_app_prompt_footer import _footer_digest, _sidebar_snapshot_text
from .console_app_prompt_panel import _prompt_panel

__all__ = [
    "_activity_badge",
    "_edit_state_badge",
    "_footer_digest",
    "_prompt_panel",
    "_prompt_state_badge",
    "_sidebar_snapshot_text",
]
