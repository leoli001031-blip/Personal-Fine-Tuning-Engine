"""Dependency contract for training preview formatting."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TrainingPreviewDeps:
    """Runtime hooks supplied by the main CLI module."""

    coerce_mapping: Callable[[Any], dict[str, Any] | None]
    format_adapter_snapshot_line: Callable[..., str | None]
    format_backend_dispatch: Callable[[Any], str | None]
    format_export_write: Callable[[Any], str | None]
    format_scalar: Callable[[Any], str]
    format_trainer_summary: Callable[[Any], str | None]
    lookup_adapter_snapshot: Callable[..., dict[str, Any] | None]
    optional_module_call: Callable[..., Any]
    pick_first: Callable[..., Any]


__all__ = ["TrainingPreviewDeps"]
