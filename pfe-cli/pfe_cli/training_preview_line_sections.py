"""Training preview line section renderers."""

from __future__ import annotations

from .training_preview_deps import TrainingPreviewDeps
from .training_preview_plan import TrainingPreviewPlan


def append_trainer_summary(
    lines: list[str],
    *,
    plan: TrainingPreviewPlan,
    deps: TrainingPreviewDeps,
) -> None:
    trainer_line = deps.format_trainer_summary(
        {
            "runtime": plan.runtime_mapping,
            "plans": plan.backend_plan_mapping,
        }
    )
    if trainer_line is not None:
        lines.append(trainer_line)


def append_adapter_context(
    lines: list[str],
    *,
    snapshot_workspace: str | None,
    deps: TrainingPreviewDeps,
) -> None:
    adapter_snapshot = deps.lookup_adapter_snapshot("latest", workspace=snapshot_workspace)
    adapter_line = deps.format_adapter_snapshot_line("latest promoted", adapter_snapshot, include_latest=True)
    if adapter_line is not None:
        lines.append(adapter_line)


def append_job_execution(lines: list[str], *, plan: TrainingPreviewPlan, deps: TrainingPreviewDeps) -> None:
    planned_executor_mode = (
        deps.pick_first(plan.dispatch_mapping, "executor_mode")
        or deps.pick_first(plan.dispatch_mapping, "execution_mode")
        or "fallback"
    )
    lines.append(
        "job-execution: "
        + " | ".join(
            [
                "status=planned",
                f"executor_mode={deps.format_scalar(planned_executor_mode)}",
                "execution_state=planned",
            ]
        )
    )


def append_backend_dispatch(lines: list[str], *, plan: TrainingPreviewPlan, deps: TrainingPreviewDeps) -> None:
    lines.append(
        deps.format_backend_dispatch(
            {
                **plan.backend_plan_mapping,
                **plan.dispatch_mapping,
                "execution_backend": plan.execution_backend,
                "execution_mode": plan.execution_mode,
                "runtime_device": deps.pick_first(plan.runtime_mapping, "runtime_device"),
                "requires_export_step": deps.pick_first(
                    plan.dispatch_mapping or plan.backend_plan_mapping,
                    "requires_export_step",
                ),
                "required_artifact_format": plan.export_artifact_format,
            }
        )
        or "backend-dispatch: n/a"
    )


def append_export_preview(lines: list[str], *, plan: TrainingPreviewPlan, deps: TrainingPreviewDeps) -> None:
    if plan.export_preview is not None:
        export_line = deps.format_export_write(plan.export_preview)
        if export_line is not None:
            lines.append(export_line)
        return
    if plan.execution_backend is None:
        return
    lines.append(
        "export-write: "
        + " | ".join(
            [
                f"gguf_export={_gguf_export_requirement(plan.export_artifact_format)}",
                "write_state=planned",
                f"target_artifact_format={deps.format_scalar(plan.export_artifact_format)}",
                f"execution_backend={deps.format_scalar(plan.execution_backend)}",
                f"execution_mode={deps.format_scalar(plan.execution_mode)}",
            ]
        )
    )


def _gguf_export_requirement(export_artifact_format: object) -> str:
    if str(export_artifact_format).lower() == "gguf_merged":
        return "required"
    return "not_required"


__all__ = [
    "append_adapter_context",
    "append_backend_dispatch",
    "append_export_preview",
    "append_job_execution",
    "append_trainer_summary",
]
