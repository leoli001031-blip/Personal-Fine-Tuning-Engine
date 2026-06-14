"""Prompt action and placeholder rules."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .console_app_data import _compact_text
from .console_app_guidance import (
    _focus_command_guidance,
    _payload_command_guidance,
    _prompt_action_token_from_label,
)
from .console_app_prompt_focus_targets import _focus_target


def _payload_action_token(payload: Mapping[str, Any] | None, focus: str | None) -> str | None:
    if payload is None:
        return None
    primary_label, _secondary_label = _payload_command_guidance(payload, focus)
    return _prompt_action_token_from_label(primary_label, focus=focus)


def _prompt_placeholder(mode: str, *, focus: str | None = None, payload: Mapping[str, Any] | None = None) -> str:
    normalized_focus = (focus or "none").strip().lower()
    if mode == "chat":
        return "Type message, /cmd, or /help"
    if _payload_action_token(payload, focus):
        return "Type /do or /see"
    target = _focus_target(normalized_focus)
    if target in {"trigger", "review", "promote", "process", "restart", "recover", "runtime", "candidate", "queue", "runner"}:
        return "Type /do or /see"
    if target == "daemon":
        return "Type /daemon or /daemon timeline"
    return "Type /status or /ops dashboard"


def _prompt_mode_help(
    mode: str,
    *,
    focus: str | None = None,
    payload: Mapping[str, Any] | None = None,
) -> str:
    normalized_focus = (focus or "none").strip().lower()
    if mode == "chat":
        return "reply"
    payload_hint = _payload_action_token(payload, focus)
    if payload_hint:
        return payload_hint
    target = _focus_target(normalized_focus)
    return target or "inspect"


def _prompt_target_hint(
    mode: str,
    *,
    focus: str | None = None,
    payload: Mapping[str, Any] | None = None,
) -> str:
    normalized_focus = (focus or "none").strip().lower()
    if mode == "chat":
        return "send"
    payload_target = _payload_action_token(payload, focus)
    if payload_target:
        return payload_target
    target = _focus_target(normalized_focus)
    return target or "status"


def _prompt_action_guidance(
    mode: str,
    *,
    focus: str | None = None,
    shortcut_hint: str | None = None,
    payload: Mapping[str, Any] | None = None,
) -> tuple[str, str]:
    normalized_focus = (focus or "none").strip().lower()
    if mode == "chat":
        return "send", "/cmd"
    if payload is not None:
        return _payload_command_guidance(payload, focus)
    if normalized_focus and normalized_focus not in {"none", "idle", "stable"}:
        return _focus_command_guidance(normalized_focus)
    normalized = (shortcut_hint or "").strip()
    parts = [part.strip() for part in normalized.split(",") if part.strip()]
    primary = parts[0] if parts else "/status"
    secondary = parts[1] if len(parts) > 1 else "/help"
    return _compact_text(primary, max_len=18), _compact_text(secondary, max_len=18)


__all__ = ["_prompt_action_guidance", "_prompt_mode_help", "_prompt_placeholder", "_prompt_target_hint"]
