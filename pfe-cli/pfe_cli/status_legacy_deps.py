"""Dependency contract for legacy status formatting."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StatusLegacyFormattingDeps:
    """Runtime hooks supplied by the main CLI module."""

    build_plan_snapshots: Callable[..., Any]
    coerce_mapping: Callable[[Any], dict[str, Any] | None]
    coerce_sequence_of_mappings: Callable[[Any], list[dict[str, Any]]]
    coerce_sequence_of_scalars: Callable[[Any], list[str]]
    format_adapter_export_artifact_line: Callable[..., str | None]
    format_backend_dispatch: Callable[[Any], str | None]
    format_compare_evaluation: Callable[[Any], str | None]
    format_daemon_timeline_summary: Callable[[Any], str | None]
    format_export_write: Callable[[Any], str | None]
    format_operations_alert_policy: Callable[[Any], list[str] | None]
    format_operations_alert_surface: Callable[[Any], list[str] | None]
    format_operations_console_digest: Callable[[Any], list[str] | None]
    format_operations_dashboard: Callable[[Any], list[str] | None]
    format_operations_event_stream: Callable[[Any], list[str] | None]
    format_operations_timeline: Callable[[Any], list[str] | None]
    format_ops_attention: Callable[..., str | None]
    format_recent_training_snapshot: Callable[[Any], list[str] | None]
    format_runner_timeline_summary: Callable[[Any], str | None]
    format_scalar: Callable[[Any], str]
    format_trainer_summary: Callable[[Any], str | None]
    pick_first: Callable[..., Any]
    prefer_inspection_summary_for_generic_monitor: Callable[..., tuple[Any, Any]]
    read_cli_state: Callable[[str | None], dict[str, Any] | None]
    read_train_queue_daemon_state: Callable[[str | None], dict[str, Any] | None]


__all__ = ["StatusLegacyFormattingDeps"]
