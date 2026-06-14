"""Dependency contract for local CLI state helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CLIStateDeps:
    """Runtime hooks supplied by the main CLI module."""

    coerce_mapping: Callable[[Any], dict[str, Any] | None]
    optional_module_call: Callable[..., Any]
    pick_first: Callable[..., Any]


__all__ = ["CLIStateDeps"]
