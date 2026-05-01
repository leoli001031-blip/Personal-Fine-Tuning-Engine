"""Dependency contract for console surface helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ConsoleSurfaceDeps:
    """Small adapter around shared CLI helper functions."""

    coerce_mapping: Callable[[Any], dict[str, Any] | None]
    format_scalar: Callable[[Any], str]
    yes_no: Callable[[Any], str]
    prefer_inspection_summary_for_generic_monitor: Callable[..., tuple[Any, Any]]


__all__ = ["ConsoleSurfaceDeps"]
