"""Dependency contract for backend plan snapshots."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PlanSnapshotDeps:
    """Runtime hooks supplied by the main CLI module."""

    coerce_mapping: Callable[[Any], dict[str, Any] | None]
    format_plan_block: Callable[[str, Any, tuple[str, ...]], list[str]]
    load_latest_adapter_manifest: Callable[[str | None], dict[str, Any] | None]
    optional_module_call: Callable[..., Any]


__all__ = ["PlanSnapshotDeps"]
