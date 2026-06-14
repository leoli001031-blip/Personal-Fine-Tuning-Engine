"""Compatibility exports for Rich-based console surfaces."""

from __future__ import annotations

from .console_app_data import _compact_text
from .console_app_guidance import _payload_command_guidance
from .console_app_panels import (
    _chat_help_panel,
    _conversation_panel,
    _event_stream_panel,
    _operations_panel,
    _status_header,
)
from .console_app_prompt import _footer_digest, _prompt_panel, _sidebar_snapshot_text
from .console_app_prompt_rules import (
    _prompt_action_guidance,
    _prompt_context_focus,
    _prompt_ctx_digest,
    _prompt_feedback_digest,
    _prompt_hint_digest,
    _prompt_mode_help,
    _prompt_placeholder,
    _prompt_target_hint,
    _prompt_trigger_category,
)
from .console_app_renderable import build_console_renderable
from .console_app_snapshot import render_console_snapshot

__all__ = [
    "_chat_help_panel",
    "_compact_text",
    "_conversation_panel",
    "_event_stream_panel",
    "_footer_digest",
    "_operations_panel",
    "_payload_command_guidance",
    "_prompt_action_guidance",
    "_prompt_context_focus",
    "_prompt_ctx_digest",
    "_prompt_feedback_digest",
    "_prompt_hint_digest",
    "_prompt_mode_help",
    "_prompt_panel",
    "_prompt_placeholder",
    "_prompt_target_hint",
    "_prompt_trigger_category",
    "_sidebar_snapshot_text",
    "_status_header",
    "build_console_renderable",
    "render_console_snapshot",
]
