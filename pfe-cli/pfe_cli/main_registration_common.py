"""Shared helpers for main command registration wiring."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any


def call(symbols: MutableMapping[str, Any], name: str, *args: Any, **kwargs: Any) -> Any:
    return symbols[name](*args, **kwargs)
