"""Backend and executor resolution helpers for training previews."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .training_preview_deps import TrainingPreviewDeps


def target_inference_backend(base_model: str | None) -> str:
    return "llama_cpp" if "llama" in str(base_model or "").lower() else "transformers"


def backend_dispatch(
    *,
    trainer_service: Any,
    backend_plan: Any,
    runtime_mapping: Mapping[str, Any],
    backend_hint: str | None,
    allow_mock_fallback: bool,
    deps: TrainingPreviewDeps,
) -> Any:
    if trainer_service is None or not hasattr(trainer_service, "_dispatch_training_backend"):
        return None
    try:
        return trainer_service._dispatch_training_backend(  # type: ignore[attr-defined]
            backend_plan=deps.coerce_mapping(backend_plan) or {},
            runtime=runtime_mapping,
            backend_hint=backend_hint,
            allow_mock_fallback=allow_mock_fallback,
        )
    except Exception:
        return None


def executor_spec(
    *,
    trainer_service: Any,
    backend_dispatch_result: Any,
    backend_plan: Any,
    runtime_mapping: Mapping[str, Any],
    backend_hint: str | None,
    allow_mock_fallback: bool,
    deps: TrainingPreviewDeps,
) -> Any:
    if trainer_service is None or not hasattr(trainer_service, "_resolve_training_executor"):
        return None
    try:
        return trainer_service._resolve_training_executor(  # type: ignore[attr-defined]
            backend_dispatch=backend_dispatch_result or deps.coerce_mapping(backend_plan) or {},
            runtime=runtime_mapping,
            backend_hint=backend_hint,
            allow_mock_fallback=allow_mock_fallback,
        )
    except Exception:
        return None


def execution_backend(
    *,
    dispatch_mapping: Mapping[str, Any],
    backend_plan_mapping: Mapping[str, Any],
    fallback_backend: str,
    deps: TrainingPreviewDeps,
) -> Any:
    return (
        deps.pick_first(dispatch_mapping, "execution_backend")
        or deps.pick_first(backend_plan_mapping, "recommended_backend", "selected_backend")
        or fallback_backend
    )


def execution_mode(
    *,
    selected_backend: Any,
    dispatch_mapping: Mapping[str, Any],
    backend_plan_mapping: Mapping[str, Any],
    deps: TrainingPreviewDeps,
) -> Any:
    resolved_mode = deps.pick_first(dispatch_mapping, "execution_mode", "executor_mode")
    if resolved_mode is not None:
        return resolved_mode
    reason = str(deps.pick_first(backend_plan_mapping, "reason") or "").lower()
    if "mock_local" in str(selected_backend).lower() or any(
        token in reason for token in ("fallback", "auto-selected", "dry-run")
    ):
        return "fallback"
    return "real"


__all__ = [
    "backend_dispatch",
    "execution_backend",
    "execution_mode",
    "executor_spec",
    "target_inference_backend",
]
