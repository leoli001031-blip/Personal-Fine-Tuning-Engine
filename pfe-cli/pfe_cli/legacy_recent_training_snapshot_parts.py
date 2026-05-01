"""Recent training snapshot sections for legacy result formatting."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .legacy_execution_result_formatting import (
    format_export_toolchain_summary,
    format_real_execution_summary,
)
from .legacy_result_deps import LegacyResultFormattingDeps


def recent_training_summary_line(mapping: Mapping[str, Any], *, deps: LegacyResultFormattingDeps) -> str | None:
    parts: list[str] = []
    version = deps.pick_first(mapping, "version")
    if version is not None:
        parts.append(f"version={deps.format_scalar(version)}")
    state = deps.pick_first(mapping, "state")
    if state is not None:
        parts.append(f"state={deps.format_scalar(state)}")
    execution_backend = deps.pick_first(mapping, "execution_backend")
    if execution_backend is not None:
        parts.append(f"execution_backend={deps.format_scalar(execution_backend)}")
    executor_mode = deps.pick_first(mapping, "executor_mode")
    if executor_mode is not None:
        parts.append(f"executor_mode={deps.format_scalar(executor_mode)}")
    if not parts:
        return None
    return "recent training: " + " | ".join(parts)


def recent_training_job_line(mapping: Mapping[str, Any], *, deps: LegacyResultFormattingDeps) -> str | None:
    return format_real_execution_summary(
        mapping.get("real_execution_summary") or mapping.get("job_execution_summary") or mapping.get("job_execution"),
        executor_mode=deps.pick_first(mapping, "executor_mode"),
        deps=deps,
    )


def recent_training_export_line(mapping: Mapping[str, Any], *, deps: LegacyResultFormattingDeps) -> str | None:
    return format_export_toolchain_summary(
        mapping.get("export_toolchain_summary") or mapping.get("export_execution"),
        deps=deps,
    )


__all__ = ["recent_training_export_line", "recent_training_job_line", "recent_training_summary_line"]
