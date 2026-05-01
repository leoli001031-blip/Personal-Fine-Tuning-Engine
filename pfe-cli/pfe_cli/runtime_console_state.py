"""Mutable state for the interactive runtime console."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4


@dataclass
class RuntimeConsoleState:
    """State carried across one interactive console session."""

    workspace: str | None
    model: str
    adapter: str
    temperature: float
    max_tokens: int | None
    real_local: bool
    refresh_seconds: float
    feedback: str = "interactive mode ready"
    transcript: list[dict[str, str]] = field(default_factory=list)
    chat_messages: list[dict[str, str]] = field(default_factory=list)
    input_history: list[str] = field(default_factory=list)
    session_id: str = field(default_factory=lambda: f"console-{uuid4().hex[:8]}")
    mode_name: str = "chat"
    last_interaction: dict[str, Any] | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    last_sidebar_refresh_at: float = 0.0
    ops_refresh_state: str = "live"


def remember_console_input(state: RuntimeConsoleState, message: str, *, limit: int = 50) -> None:
    state.input_history.append(message)
    if len(state.input_history) > limit:
        del state.input_history[:-limit]


def console_command_input(state: RuntimeConsoleState, message: str) -> str:
    if state.mode_name == "command" and not message.startswith("/"):
        return f"/{message}"
    return message


def set_real_local_env(enabled: bool, *, previous: str | None) -> None:
    if enabled:
        os.environ["PFE_ENABLE_REAL_LOCAL_INFERENCE"] = "1"
    elif previous is None:
        os.environ.pop("PFE_ENABLE_REAL_LOCAL_INFERENCE", None)
    else:
        os.environ["PFE_ENABLE_REAL_LOCAL_INFERENCE"] = previous


def restore_real_local_env(previous: str | None) -> None:
    if previous is None:
        os.environ.pop("PFE_ENABLE_REAL_LOCAL_INFERENCE", None)
    else:
        os.environ["PFE_ENABLE_REAL_LOCAL_INFERENCE"] = previous


__all__ = [
    "RuntimeConsoleState",
    "console_command_input",
    "remember_console_input",
    "restore_real_local_env",
    "set_real_local_env",
]
