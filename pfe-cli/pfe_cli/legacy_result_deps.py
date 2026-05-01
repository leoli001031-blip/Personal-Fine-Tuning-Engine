"""Dependency contract for legacy result formatting."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LegacyResultFormattingDeps:
    """Runtime hooks supplied by the main CLI module."""

    coerce_mapping: Callable[[Any], dict[str, Any] | None]
    format_backend_dispatch: Callable[[Any], str | None]
    format_export_write: Callable[[Any], str | None]
    format_scalar: Callable[[Any], str]
    lookup_adapter_snapshot: Callable[..., dict[str, Any] | None]
    ordered_eval_scores: Callable[..., Any]
    pick_first: Callable[..., Any]


__all__ = ["LegacyResultFormattingDeps"]
