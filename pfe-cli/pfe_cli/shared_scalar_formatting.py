"""Shared scalar formatting helpers."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any


def format_scalar(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (str, int, float)):
        return str(value)
    if isinstance(value, Mapping):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return ", ".join(format_scalar(item) for item in value)
    return str(value)


def yes_no(value: Any) -> str:
    return "yes" if bool(value) else "no"


def pick_first(mapping: Mapping[str, Any] | None, *keys: str) -> Any:
    if not mapping:
        return None
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


__all__ = ["format_scalar", "pick_first", "yes_no"]
