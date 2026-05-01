"""Backend plan snapshot text formatting."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .plan_snapshot_deps import PlanSnapshotDeps


def format_plan_snapshot_lines(
    plan_snapshots: Mapping[str, Any],
    *,
    deps: PlanSnapshotDeps,
) -> list[str]:
    lines: list[str] = []
    if "trainer" in plan_snapshots:
        lines.extend(
            deps.format_plan_block(
                "trainer",
                plan_snapshots["trainer"],
                (
                    "selected_backend",
                    "requested_backend",
                    "train_type",
                    "requires_export_step",
                    "export_format",
                    "export_backend",
                ),
            )
        )
    if "inference" in plan_snapshots:
        lines.extend(
            deps.format_plan_block(
                "inference",
                plan_snapshots["inference"],
                (
                    "selected_backend",
                    "requested_backend",
                    "requires_export",
                    "required_artifact_format",
                    "preferred_device",
                    "reason",
                ),
            )
        )
    if "export" in plan_snapshots:
        lines.extend(
            deps.format_plan_block(
                "export",
                plan_snapshots["export"],
                (
                    "target_backend",
                    "target_artifact_format",
                    "required",
                    "artifact_name",
                    "artifact_directory",
                    "reason",
                ),
            )
        )
    return lines


__all__ = ["format_plan_snapshot_lines"]
