"""Shared CLI coercion helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, is_dataclass
from typing import Any


def coerce_mapping(result: Any) -> dict[str, Any] | None:
    """Convert pydantic-style results into a plain mapping when possible."""

    if isinstance(result, dict):
        return dict(result)
    if is_dataclass(result) and not isinstance(result, type):
        try:
            dumped = asdict(result)
        except Exception:
            return None
        if isinstance(dumped, dict):
            return dict(dumped)
    model_dump = getattr(result, "model_dump", None)
    if callable(model_dump):
        try:
            dumped = model_dump()
        except Exception:
            return None
        if isinstance(dumped, dict):
            return dict(dumped)
    to_dict = getattr(result, "dict", None)
    if callable(to_dict):
        try:
            dumped = to_dict()
        except Exception:
            return None
        if isinstance(dumped, dict):
            return dict(dumped)
    return None


def coerce_sequence_of_mappings(result: Any) -> list[dict[str, Any]]:
    if not isinstance(result, Sequence) or isinstance(result, (str, bytes, bytearray)):
        return []
    items: list[dict[str, Any]] = []
    for item in result:
        mapping = coerce_mapping(item)
        if mapping is not None:
            items.append(mapping)
    return items


def coerce_sequence_of_scalars(result: Any) -> list[str]:
    if not isinstance(result, Sequence) or isinstance(result, (str, bytes, bytearray)):
        return []
    items: list[str] = []
    for item in result:
        if item is None:
            continue
        items.append(str(item))
    return items


__all__ = ["coerce_mapping", "coerce_sequence_of_mappings", "coerce_sequence_of_scalars"]
