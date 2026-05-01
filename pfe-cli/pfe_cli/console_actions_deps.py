"""Dependency contract for console action helpers."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ConsoleActionsDeps:
    """Small adapter around shared CLI helper functions."""

    coerce_mapping: Callable[[Any], dict[str, Any] | None]
    console_dashboard_focus: Callable[[Mapping[str, Any] | None], str]


__all__ = ["ConsoleActionsDeps"]
