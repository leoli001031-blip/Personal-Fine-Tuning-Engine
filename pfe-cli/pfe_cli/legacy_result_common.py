"""Shared helpers for legacy result formatting."""

from __future__ import annotations

from typing import Any


def format_bytes_compact(value: Any) -> str | None:
    try:
        size = int(value)
    except Exception:
        return None
    units = ("B", "KB", "MB", "GB")
    scaled = float(size)
    unit = units[0]
    for candidate in units:
        unit = candidate
        if scaled < 1024.0 or candidate == units[-1]:
            break
        scaled /= 1024.0
    if unit == "B":
        return f"{int(scaled)}{unit}"
    return f"{scaled:.1f}{unit}"


__all__ = ["format_bytes_compact"]
