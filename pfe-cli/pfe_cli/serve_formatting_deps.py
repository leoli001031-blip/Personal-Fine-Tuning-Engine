"""Dependency contract for serve formatting."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ServeFormattingDeps:
    """Runtime hooks supplied by the main CLI module."""

    build_plan_snapshots: Callable[[str | None, Mapping[str, Any] | None], dict[str, Any]]
    coerce_mapping: Callable[[Any], dict[str, Any] | None]
    format_adapter_snapshot_line: Callable[..., str | None]
    format_backend_dispatch: Callable[[Any], str | None]
    format_export_write: Callable[[Any], str | None]
    format_recent_training_snapshot: Callable[[Any], list[str] | None]
    format_scalar: Callable[[Any], str]
    format_status_legacy: Callable[[Any], str]
    format_trainer_summary: Callable[[Any], str | None]
    lookup_recent_adapter_snapshot: Callable[..., dict[str, Any] | None]
    optional_module_call: Callable[..., Any]
    read_cli_state: Callable[[str | None], dict[str, Any] | None]
    lookup_adapter_snapshot: Callable[..., dict[str, Any] | None]


__all__ = ["ServeFormattingDeps"]
