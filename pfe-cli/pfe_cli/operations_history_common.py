"""Shared dependencies and helpers for operations history formatting."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class OperationsHistoryFormattingDeps:
    """Small adapter around shared CLI helper functions."""

    coerce_mapping: Callable[[Any], dict[str, Any] | None]
    format_scalar: Callable[[Any], str]


def history_latest_timestamp(items: Any, *, deps: OperationsHistoryFormattingDeps) -> str | None:
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes, bytearray)):
        return None
    for item in reversed(items):
        item_map = deps.coerce_mapping(item)
        if item_map is None:
            continue
        timestamp = item_map.get("timestamp")
        if timestamp is not None:
            return deps.format_scalar(timestamp)
    return None


__all__ = ["OperationsHistoryFormattingDeps", "history_latest_timestamp"]
