"""Training preview text line rendering."""

from __future__ import annotations

from .training_preview_deps import TrainingPreviewDeps
from .training_preview_line_sections import (
    append_adapter_context,
    append_backend_dispatch,
    append_export_preview,
    append_job_execution,
    append_trainer_summary,
)
from .training_preview_plan import TrainingPreviewPlan


def build_training_preview_lines(
    *,
    method: str,
    epochs: int,
    train_type: str,
    workspace: str | None,
    snapshot_workspace: str | None,
    plan: TrainingPreviewPlan,
    deps: TrainingPreviewDeps,
) -> list[str]:
    lines = [
        "PFE train plan",
        f"request: method={method} | epochs={epochs} | train_type={train_type} | workspace={workspace or 'default'}",
    ]
    append_trainer_summary(lines, plan=plan, deps=deps)
    append_adapter_context(lines, snapshot_workspace=snapshot_workspace, deps=deps)
    append_job_execution(lines, plan=plan, deps=deps)
    append_backend_dispatch(lines, plan=plan, deps=deps)
    append_export_preview(lines, plan=plan, deps=deps)
    return lines


__all__ = ["build_training_preview_lines"]
