"""Shared context object for console slash-command routing."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TypeAlias

from .console_routing_deps import ConsoleRoutingDeps

ConsoleCommandResult: TypeAlias = tuple[str | None, str, dict[str, Any] | None]


@dataclass(frozen=True)
class ConsoleRouteContext:
    """Inputs needed by a single console command routing attempt."""

    command: str
    normalized: str
    payload: Mapping[str, Any]
    workspace: str | None
    service: Any
    current_workspace: str | None
    mode: str
    model: str
    adapter: str
    temperature: float
    max_tokens: int | None
    real_local: bool
    refresh_seconds: float
    deps: ConsoleRoutingDeps
    last_interaction: dict[str, Any] | None = None


__all__ = ["ConsoleCommandResult", "ConsoleRouteContext"]
