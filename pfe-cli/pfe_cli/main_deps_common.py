"""Shared helpers for main CLI dependency builders."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def symbol(symbols: Mapping[str, Any], name: str) -> Any:
    return symbols[name]


def call(symbols: Mapping[str, Any], name: str, *args: Any, **kwargs: Any) -> Any:
    return symbol(symbols, name)(*args, **kwargs)
