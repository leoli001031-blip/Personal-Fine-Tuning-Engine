"""Shared dependencies for operations formatting helpers."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class OperationsFormattingDeps:
    """Small adapter around shared CLI helper functions."""

    coerce_mapping: Callable[[Any], dict[str, Any] | None]
    coerce_sequence_of_mappings: Callable[[Any], list[dict[str, Any]]]
    coerce_sequence_of_scalars: Callable[[Any], list[str]]
    format_scalar: Callable[[Any], str]
    prefer_inspection_summary_for_generic_monitor: Callable[..., tuple[Any, Any]]
    generic_monitor_focuses: frozenset[str]


def resolved_focus(surface: Mapping[str, Any] | None, *, deps: OperationsFormattingDeps) -> Any:
    surface_map = deps.coerce_mapping(surface)
    if not surface_map:
        return None
    current_focus = surface_map.get("current_focus")
    current_focus_text = str(current_focus or "").strip().lower()
    if current_focus_text not in {"", "none", "idle", "stable"}:
        return current_focus
    monitor_focus = surface_map.get("monitor_focus")
    return monitor_focus if monitor_focus is not None else current_focus


def resolved_first_focus(*candidates: Any) -> Any:
    for candidate in candidates:
        if candidate is None:
            continue
        if str(candidate).strip().lower() in {"", "none", "idle", "stable"}:
            continue
        return candidate
    return next((candidate for candidate in candidates if candidate is not None), None)
