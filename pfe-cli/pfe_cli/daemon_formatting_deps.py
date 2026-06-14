"""Dependencies shared by daemon status formatters."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DaemonFormattingDeps:
    """Small adapter around shared CLI helper functions."""

    coerce_mapping: Callable[[Any], dict[str, Any] | None]
    format_scalar: Callable[[Any], str]


__all__ = ["DaemonFormattingDeps"]
