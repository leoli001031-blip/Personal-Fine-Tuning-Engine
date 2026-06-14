"""Dependency contract for doctor formatting."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DoctorFormattingDeps:
    """Small adapter around shared CLI helper functions."""

    coerce_mapping: Callable[[Any], dict[str, Any] | None]
    format_scalar: Callable[[Any], str]
    pick_first: Callable[..., Any]
    load_latest_adapter_manifest: Callable[[str | None], dict[str, Any] | None]
    optional_module_call: Callable[..., Any | None]
    pfe_home: Callable[[str | None], Any]
    lookup_adapter_snapshot: Callable[..., dict[str, Any] | None]
    lookup_recent_adapter_snapshot: Callable[..., dict[str, Any] | None]
    format_adapter_export_artifact_line: Callable[[str, Any], str | None]


__all__ = ["DoctorFormattingDeps"]
