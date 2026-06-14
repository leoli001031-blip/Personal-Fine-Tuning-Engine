"""Status formatting helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from . import formatters_matrix


@dataclass(frozen=True)
class StatusFormattingDeps:
    """Runtime hooks supplied by the main CLI module."""

    coerce_mapping: Callable[[Any], dict[str, Any] | None]
    read_cli_state: Callable[[str | None], dict[str, Any] | None]


def format_status(
    result: Any,
    *,
    workspace: str | None = None,
    deps: StatusFormattingDeps,
) -> str:
    # Matrix theme - default style.
    mapping = deps.coerce_mapping(result)
    if mapping is not None:
        cached_state = deps.read_cli_state(workspace or mapping.get("workspace") or mapping.get("home"))
        if cached_state is not None:
            recent_training = deps.coerce_mapping(cached_state.get("recent_training"))
            if recent_training is not None:
                mapping = dict(mapping)
                mapping["recent_training_snapshot"] = recent_training
                recent_adapter = deps.coerce_mapping(mapping.get("recent_adapter")) or {}
                recent_adapter = dict(recent_adapter)
                for key in ("execution_backend", "executor_mode"):
                    if key in recent_training and recent_adapter.get(key) is None:
                        recent_adapter[key] = recent_training[key]
                mapping["recent_adapter"] = recent_adapter
                for key in (
                    "real_execution_summary",
                    "export_toolchain_summary",
                    "job_execution",
                    "export_execution",
                ):
                    if key in recent_training and mapping.get(key) is None:
                        mapping[key] = recent_training[key]
    return formatters_matrix.format_status_matrix(mapping or result, workspace=workspace)
