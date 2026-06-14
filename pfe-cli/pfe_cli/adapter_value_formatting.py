"""Scalar value formatting for adapter CLI output."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any


def _format_value(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (str, int, float)):
        return str(value)
    if isinstance(value, Mapping):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return ", ".join(_format_value(item) for item in value)
    return str(value)


__all__ = ["_format_value"]
