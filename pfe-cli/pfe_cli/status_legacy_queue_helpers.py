"""Shared helpers for legacy train queue status formatting."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def append_scalar_parts(parts: list[str], mapping: Mapping[str, Any], keys: tuple[str, ...], *, deps: Any) -> None:
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            parts.append(f"{key}={deps.format_scalar(value)}")


def compact_item(mapping: Mapping[str, Any], keys: tuple[str, ...], *, deps: Any) -> str:
    return ",".join(
        part
        for part in (
            deps.format_scalar(mapping.get(key)) if mapping.get(key) is not None else ""
            for key in keys
        )
        if part
    )


__all__ = ["append_scalar_parts", "compact_item"]
