"""Dependency contract for utility commands."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class UtilityCommandDeps:
    """Runtime hooks supplied by the main CLI module."""

    format_doctor: Callable[..., str]


__all__ = ["UtilityCommandDeps"]
