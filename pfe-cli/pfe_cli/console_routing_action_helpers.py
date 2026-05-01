"""Small helpers shared by console action routers."""

from __future__ import annotations


def command_suffix(normalized: str) -> str | None:
    parts = normalized.split(None, 1)
    return parts[1].strip() if len(parts) > 1 else None


def batch_limit(normalized: str, *, default: int = 5) -> int:
    parts = normalized.split(None, 1)
    if len(parts) <= 1:
        return default
    try:
        return int(parts[1].strip().split()[0])
    except ValueError:
        return default


__all__ = ["batch_limit", "command_suffix"]
