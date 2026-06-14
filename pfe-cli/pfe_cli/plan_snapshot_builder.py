"""Backend plan snapshot assembly."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .plan_snapshot_deps import PlanSnapshotDeps
from .plan_snapshot_resolution import (
    build_export_plan,
    build_inference_plan,
    build_trainer_plan,
    requested_backend_from_snapshot,
    train_type_from_manifest,
)


def build_plan_snapshots(
    workspace: str | None,
    status_mapping: Mapping[str, Any] | None = None,
    *,
    deps: PlanSnapshotDeps,
) -> dict[str, Any]:
    """Assemble trainer/inference/export plan snapshots from local helpers."""

    manifest = deps.load_latest_adapter_manifest(workspace)
    status_mapping = dict(status_mapping or {})
    metadata = deps.coerce_mapping(status_mapping.get("metadata")) or {}
    inference_status = deps.coerce_mapping(metadata.get("inference")) or {}
    pipeline_status = deps.coerce_mapping(metadata.get("pipeline")) or {}

    train_type = train_type_from_manifest(manifest)
    requested_backend = requested_backend_from_snapshot(manifest=manifest, inference_status=inference_status)
    artifact_format = manifest.get("artifact_format") if manifest is not None else None
    trainer_plan = build_trainer_plan(train_type=train_type, requested_backend=requested_backend, deps=deps)
    inference_plan = build_inference_plan(
        manifest=manifest,
        inference_status=inference_status,
        requested_backend=requested_backend,
        artifact_format=artifact_format,
        deps=deps,
    )

    selected_backend = inference_plan.get("selected_backend") if isinstance(inference_plan, Mapping) else None
    selected_backend = selected_backend or requested_backend
    export_plan = build_export_plan(
        manifest=manifest,
        inference_status=inference_status,
        selected_backend=selected_backend,
        artifact_format=artifact_format,
        workspace=workspace,
        status_mapping=status_mapping,
        deps=deps,
    )

    result = {
        "trainer": trainer_plan,
        "inference": inference_plan,
        "export": export_plan,
    }
    if pipeline_status:
        result["pipeline"] = pipeline_status
    return result


__all__ = ["build_plan_snapshots"]
