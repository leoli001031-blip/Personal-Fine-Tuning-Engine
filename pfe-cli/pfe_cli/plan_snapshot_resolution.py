"""Plan snapshot backend resolution helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .plan_snapshot_deps import PlanSnapshotDeps


def train_type_from_manifest(manifest: Mapping[str, Any] | None) -> str:
    train_type = "sft"
    if manifest:
        train_type = str((manifest.get("training_config") or {}).get("train_type", train_type))
    return train_type


def requested_backend_from_snapshot(
    *,
    manifest: Mapping[str, Any] | None,
    inference_status: Mapping[str, Any],
) -> Any:
    requested_backend = inference_status.get("requested_backend") if inference_status else None
    if requested_backend is None and manifest is not None:
        requested_backend = manifest.get("inference_backend") or (manifest.get("training_config") or {}).get("backend")
    return requested_backend or "auto"


def build_trainer_plan(
    *,
    train_type: str,
    requested_backend: Any,
    deps: PlanSnapshotDeps,
) -> Any:
    plan = deps.optional_module_call(
        "pfe_core.trainer.runtime",
        "summarize_trainer_backend_plan",
        train_type=train_type,
        target_inference_backend=requested_backend,
    )
    if plan is not None:
        return plan
    return {
        "selected_backend": "n/a",
        "requested_backend": requested_backend,
        "train_type": train_type,
    }


def build_inference_plan(
    *,
    manifest: Mapping[str, Any] | None,
    inference_status: Mapping[str, Any],
    requested_backend: Any,
    artifact_format: Any,
    deps: PlanSnapshotDeps,
) -> Any:
    plan = inference_status.get("backend_plan") or deps.optional_module_call(
        "pfe_core.inference.backends",
        "summarize_backend_plan",
        requested_backend=requested_backend,
        artifact_format=artifact_format,
        manifest=manifest,
    )
    if plan is not None:
        return plan
    return {
        "selected_backend": requested_backend,
        "requested_backend": requested_backend,
        "requires_export": False,
    }


def build_export_plan(
    *,
    manifest: Mapping[str, Any] | None,
    inference_status: Mapping[str, Any],
    selected_backend: Any,
    artifact_format: Any,
    workspace: str | None,
    status_mapping: Mapping[str, Any],
    deps: PlanSnapshotDeps,
) -> Any:
    plan = inference_status.get("export_plan") or deps.optional_module_call(
        "pfe_core.inference.export",
        "plan_export",
        target_backend=selected_backend,
        source_artifact_format=artifact_format,
        workspace=workspace or status_mapping.get("home"),
        adapter_dir=(manifest or {}).get("adapter_dir"),
        source_adapter_version=(manifest or {}).get("version"),
        source_model=(manifest or {}).get("base_model"),
        training_run_id=(manifest or {}).get("training_run_id"),
        num_samples=(manifest or {}).get("num_samples"),
    )
    if plan is not None:
        return plan
    return {
        "target_backend": selected_backend,
        "target_artifact_format": artifact_format,
        "required": False,
    }


__all__ = [
    "build_export_plan",
    "build_inference_plan",
    "build_trainer_plan",
    "requested_backend_from_snapshot",
    "train_type_from_manifest",
]
