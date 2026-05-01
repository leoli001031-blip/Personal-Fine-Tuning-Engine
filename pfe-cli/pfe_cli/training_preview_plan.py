"""Backend and export planning for training previews."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .training_preview_backend_resolution import (
    backend_dispatch,
    execution_backend,
    execution_mode,
    executor_spec,
    target_inference_backend,
)
from .training_preview_deps import TrainingPreviewDeps
from .training_preview_export import build_training_export_preview


@dataclass(frozen=True)
class TrainingPreviewPlan:
    runtime_mapping: dict[str, Any]
    target_inference_backend: str
    backend_plan_mapping: dict[str, Any]
    dispatch_mapping: dict[str, Any]
    execution_backend: Any
    execution_mode: Any
    export_artifact_format: Any
    export_preview: Any


def build_training_preview_plan(
    *,
    method: str,
    epochs: int,
    base_model: str | None,
    train_type: str,
    workspace: str | None,
    backend_hint: str | None,
    deps: TrainingPreviewDeps,
) -> TrainingPreviewPlan:
    trainer_service = deps.optional_module_call("pfe_core.trainer", "service")
    runtime = deps.optional_module_call("pfe_core.trainer.runtime", "detect_trainer_runtime")
    runtime_mapping = deps.coerce_mapping(runtime) or {}
    target_backend = target_inference_backend(base_model)
    backend_plan = deps.optional_module_call(
        "pfe_core.trainer.runtime",
        "summarize_trainer_backend_plan",
        train_type=train_type,
        runtime=runtime_mapping or None,
        backend_hint=backend_hint,
        target_inference_backend=target_backend,
    )
    backend_dispatch_result = backend_dispatch(
        trainer_service=trainer_service,
        backend_plan=backend_plan,
        runtime_mapping=runtime_mapping,
        backend_hint=backend_hint,
        deps=deps,
    )
    resolved_executor_spec = executor_spec(
        trainer_service=trainer_service,
        backend_dispatch_result=backend_dispatch_result,
        backend_plan=backend_plan,
        runtime_mapping=runtime_mapping,
        backend_hint=backend_hint,
        deps=deps,
    )

    backend_plan_mapping = deps.coerce_mapping(backend_plan) or {}
    dispatch_mapping = deps.coerce_mapping(resolved_executor_spec) or deps.coerce_mapping(backend_dispatch_result) or {}
    selected_backend = execution_backend(
        dispatch_mapping=dispatch_mapping,
        backend_plan_mapping=backend_plan_mapping,
        fallback_backend=target_backend,
        deps=deps,
    )
    selected_mode = execution_mode(
        selected_backend=selected_backend,
        dispatch_mapping=dispatch_mapping,
        backend_plan_mapping=backend_plan_mapping,
        deps=deps,
    )
    export_artifact_format = deps.pick_first(
        dispatch_mapping or backend_plan_mapping,
        "export_format",
        "artifact_format",
    )
    export_preview = build_training_export_preview(
        method=method,
        epochs=epochs,
        base_model=base_model,
        train_type=train_type,
        workspace=workspace,
        execution_backend=selected_backend,
        target_inference_backend=target_backend,
        export_artifact_format=export_artifact_format,
        deps=deps,
    )
    return TrainingPreviewPlan(
        runtime_mapping=runtime_mapping,
        target_inference_backend=target_backend,
        backend_plan_mapping=backend_plan_mapping,
        dispatch_mapping=dispatch_mapping,
        execution_backend=selected_backend,
        execution_mode=selected_mode,
        export_artifact_format=export_artifact_format,
        export_preview=export_preview,
    )


__all__ = ["TrainingPreviewPlan", "build_training_preview_plan"]
