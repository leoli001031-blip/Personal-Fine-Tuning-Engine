"""Dependency contract for console slash-command routing."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ConsoleRoutingDeps:
    """Runtime hooks supplied by the main CLI module."""

    coerce_mapping: Callable[[Any], dict[str, Any] | None]
    format_scalar: Callable[[Any], str]
    resolve_handler: Callable[..., Any | None]
    console_submit_feedback: Callable[..., list[dict[str, Any]]]
    console_help_text: Callable[[], str]
    console_status_compact_text: Callable[..., str]
    console_focus_actions: Callable[..., dict[str, str | None]]
    console_settings_text: Callable[..., str]
    format_status: Callable[..., str]
    format_operations_dashboard: Callable[[Any], list[str] | None]
    format_operations_alert_surface: Callable[[Any], list[str] | None]
    format_operations_alert_policy: Callable[[Any], list[str] | None]
    format_operations_event_stream: Callable[[Any], list[str] | None]
    format_operations_console_digest: Callable[[Any], list[str] | None]
    format_doctor: Callable[..., str]
    format_serve_preview: Callable[..., str]
    format_train_queue_daemon_status: Callable[[Any], str]
    format_worker_runner_status: Callable[[Any], str]
    format_eval_result: Callable[..., str]
    format_candidate_history: Callable[[Any], str]
    format_candidate_timeline: Callable[[Any], str]
    format_train_queue_history: Callable[[Any], str]
    format_runner_timeline_summary: Callable[[Any], str]
    format_worker_runner_history: Callable[[Any], str]
    format_train_queue_daemon_history: Callable[[Any], str]
    read_train_queue_daemon_state: Callable[[str | None], dict[str, Any] | None]
    format_daemon_timeline_summary: Callable[[Any], str]
    format_lifecycle_summary: Callable[[Any], list[str] | None]
    format_train_result: Callable[..., str]


__all__ = ["ConsoleRoutingDeps"]
