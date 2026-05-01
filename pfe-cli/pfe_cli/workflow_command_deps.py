"""Dependency contract for workflow CLI commands."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class WorkflowCommandDeps:
    """Runtime hooks supplied by the main CLI module."""

    load_service: Callable[..., Any | None]
    run_placeholder: Callable[[str], None]
    resolve_handler: Callable[..., Any | None]
    run_handler: Callable[..., None]
    format_eval_result: Callable[..., str]


__all__ = ["WorkflowCommandDeps"]
