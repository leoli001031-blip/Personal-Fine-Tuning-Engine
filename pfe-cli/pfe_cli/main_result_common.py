"""Shared helpers for main result compatibility installers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def call(symbols: Mapping[str, Any], name: str, *args: Any, **kwargs: Any) -> Any:
    return symbols[name](*args, **kwargs)


__all__ = ["call"]
