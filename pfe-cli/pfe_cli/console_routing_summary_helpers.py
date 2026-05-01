"""Small helpers for console summary text renderers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .console_routing_deps import ConsoleRoutingDeps


def append_mapping_parts(
    parts: list[str],
    mapping: Mapping[str, Any],
    keys: Sequence[str],
    *,
    deps: ConsoleRoutingDeps,
    prefix: str = "",
) -> None:
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            parts.append(f"{prefix}{key}={deps.format_scalar(value)}")


def render_summary(title: str, parts: list[str], *, fallback: str) -> str:
    if not parts:
        parts.append(fallback)
    return "\n".join([title, "summary: " + " | ".join(parts)])


__all__ = ["append_mapping_parts", "render_summary"]
