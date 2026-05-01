"""Dependency hooks for adapter snapshot lookup helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AdapterSnapshotDeps:
    """Runtime hooks supplied by the main CLI module."""

    coerce_mapping: Callable[[Any], dict[str, Any] | None]
    optional_module_call: Callable[..., Any]
    pick_first: Callable[..., Any]


__all__ = ["AdapterSnapshotDeps"]
