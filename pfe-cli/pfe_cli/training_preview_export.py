"""Export preview helpers for training preview planning."""

from __future__ import annotations

from typing import Any

from .training_preview_deps import TrainingPreviewDeps


def build_training_export_preview(
    *,
    method: str,
    epochs: int,
    base_model: str | None,
    train_type: str,
    workspace: str | None,
    execution_backend: Any,
    target_inference_backend: str,
    export_artifact_format: Any,
    deps: TrainingPreviewDeps,
) -> Any:
    return deps.optional_module_call(
        "pfe_core.inference.export_runtime",
        "build_export_runtime_spec",
        target_backend=execution_backend or target_inference_backend,
        source_artifact_format=export_artifact_format,
        workspace=workspace,
        source_model=base_model,
        training_run_id=None,
        num_samples=None,
        extra_metadata={
            "method": method,
            "epochs": epochs,
            "train_type": train_type,
        },
    )


__all__ = ["build_training_export_preview"]
