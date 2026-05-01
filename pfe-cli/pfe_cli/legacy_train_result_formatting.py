"""Legacy train result formatter."""

from __future__ import annotations

from typing import Any

from .legacy_adapter_result_formatting import format_adapter_snapshot_line
from .legacy_context_result_formatting import format_incremental_context
from .legacy_execution_result_formatting import (
    format_export_execution_summary,
    format_job_execution_summary,
)
from .legacy_result_deps import LegacyResultFormattingDeps


def format_train_result_legacy(
    result: Any,
    *,
    workspace: str | None = None,
    deps: LegacyResultFormattingDeps,
) -> str:
    """Legacy plain text train formatter kept for compatibility checks."""

    mapping = deps.coerce_mapping(result)
    if mapping is None:
        return deps.format_scalar(result)

    lines = ["PFE train"]
    version = deps.pick_first(mapping, "version")
    adapter_path = deps.pick_first(mapping, "adapter_path")
    num_samples = deps.pick_first(mapping, "num_samples")
    if version is not None or adapter_path is not None or num_samples is not None:
        parts = []
        if version is not None:
            parts.append(f"version={deps.format_scalar(version)}")
        if adapter_path is not None:
            parts.append(f"adapter_path={deps.format_scalar(adapter_path)}")
        if num_samples is not None:
            parts.append(f"num_samples={deps.format_scalar(num_samples)}")
        lines.append(" | ".join(parts))

    incremental_line = format_incremental_context(mapping.get("incremental_context") or mapping, deps=deps)
    if incremental_line is not None:
        lines.append(incremental_line)

    training_snapshot = deps.lookup_adapter_snapshot(str(version) if version is not None else None, workspace=workspace)
    training_line = format_adapter_snapshot_line("recent training adapter", training_snapshot, include_latest=True, deps=deps)
    if training_line is not None:
        lines.append(training_line)

    latest_snapshot = deps.lookup_adapter_snapshot("latest", workspace=workspace)
    latest_line = format_adapter_snapshot_line("latest promoted", latest_snapshot, include_latest=True, deps=deps)
    if latest_line is not None:
        lines.append(latest_line)

    backend_dispatch = mapping.get("backend_plan") or mapping.get("backend_dispatch")
    job_execution = mapping.get("job_execution")
    if job_execution is not None:
        job_line = format_job_execution_summary(job_execution, deps=deps)
        if job_line is not None:
            lines.append(job_line)
    export_execution = mapping.get("export_execution")
    if export_execution is not None:
        export_exec_line = format_export_execution_summary(export_execution, deps=deps)
        if export_exec_line is not None:
            lines.append(export_exec_line)
    export_write = (
        mapping.get("export_write")
        or mapping.get("export_command_plan")
        or mapping.get("export_execution")
        or mapping.get("export_runtime")
    )
    if backend_dispatch is not None:
        dispatch_line = deps.format_backend_dispatch(backend_dispatch)
        if dispatch_line is not None:
            lines.append(dispatch_line)
    if export_write is not None:
        export_line = deps.format_export_write(export_write)
        if export_line is not None:
            lines.append(export_line)

    metrics = deps.coerce_mapping(mapping.get("metrics"))
    if metrics is not None:
        outcome = []
        if "num_fresh_samples" in metrics:
            outcome.append(f"fresh_samples={deps.format_scalar(metrics.get('num_fresh_samples'))}")
        if "num_replay_samples" in metrics:
            outcome.append(f"replay_samples={deps.format_scalar(metrics.get('num_replay_samples'))}")
        if "requires_export_step" in metrics:
            outcome.append(f"requires_export_step={deps.format_scalar(metrics.get('requires_export_step'))}")
        if outcome:
            lines.append("metrics: " + " | ".join(outcome))
    return "\n".join(lines)


__all__ = ["format_train_result_legacy"]
