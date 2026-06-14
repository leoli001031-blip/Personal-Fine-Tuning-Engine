"""Core data normalization helpers for console rendering."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, dict):
            return dict(dumped)
    to_dict = getattr(value, "dict", None)
    if callable(to_dict):
        dumped = to_dict()
        if isinstance(dumped, dict):
            return dict(dumped)
    return {}


def _sequence(value: Any) -> list[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    return []


def _yes_no(value: Any) -> str:
    if value is None:
        return "n/a"
    return "yes" if bool(value) else "no"


def _value(mapping: Mapping[str, Any], *keys: str, default: str = "n/a") -> str:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return str(mapping[key])
    return default


def _summary_field(summary_line: Any, field: str) -> str:
    text = str(summary_line or "")
    prefix = f"{field}="
    for part in text.split("|"):
        token = part.strip()
        if token.startswith(prefix):
            return token[len(prefix) :].strip()
    return ""


def _compact_text(value: Any, *, max_len: int = 28) -> str:
    text = str(value or "").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _timestamp_now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


__all__ = [
    "_compact_text",
    "_mapping",
    "_sequence",
    "_summary_field",
    "_timestamp_now",
    "_value",
    "_yes_no",
]
