"""Top-level training preview renderer."""

from __future__ import annotations

from .training_preview_deps import TrainingPreviewDeps
from .training_preview_lines import build_training_preview_lines
from .training_preview_plan import build_training_preview_plan


def format_train_preview(
    *,
    method: str,
    epochs: int,
    base_model: str | None,
    train_type: str,
    workspace: str | None,
    snapshot_workspace: str | None = None,
    backend_hint: str | None,
    deps: TrainingPreviewDeps,
) -> str:
    """Render a compact training preflight summary without executing training."""

    plan = build_training_preview_plan(
        method=method,
        epochs=epochs,
        base_model=base_model,
        train_type=train_type,
        workspace=workspace,
        backend_hint=backend_hint,
        deps=deps,
    )
    return "\n".join(
        build_training_preview_lines(
            method=method,
            epochs=epochs,
            train_type=train_type,
            workspace=workspace,
            snapshot_workspace=snapshot_workspace,
            plan=plan,
            deps=deps,
        )
    )


__all__ = ["format_train_preview"]
