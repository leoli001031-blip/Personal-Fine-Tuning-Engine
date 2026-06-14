"""Trainer-specific plan formatting helpers."""

from __future__ import annotations

from typing import Any

from .shared_coercion_formatting import coerce_mapping, format_scalar, pick_first
from .shared_plan_blocks import plan_summary


def format_trainer_block(trainer: Any) -> list[str]:
    """Render trainer runtime plus per-train-type backend plans."""

    mapping = coerce_mapping(trainer)
    if mapping is None:
        return ["trainer plan:", f"  {format_scalar(trainer)}"]

    lines = ["trainer plan:"]
    runtime = coerce_mapping(mapping.get("runtime"))
    if runtime is not None:
        lines.append(
            "  runtime: "
            + plan_summary(
                runtime,
                ("runtime_device", "cpu_only", "mps_available", "cuda_available", "platform_name"),
            )
        )

    plans = coerce_mapping(mapping.get("plans"))
    if plans:
        for name in ("sft", "dpo"):
            if name in plans:
                lines.append(
                    f"  {name}: "
                    + plan_summary(
                        plans[name],
                        (
                            "selected_backend",
                            "requested_backend",
                            "train_type",
                            "requires_export_step",
                            "export_format",
                            "export_backend",
                            "reason",
                        ),
                    )
                )
        return lines

    lines.append(
        "  "
        + plan_summary(
            mapping,
            (
                "selected_backend",
                "requested_backend",
                "train_type",
                "requires_export_step",
                "export_format",
                "export_backend",
                "reason",
            ),
        )
    )
    return lines


def format_trainer_summary(trainer: Any) -> str | None:
    mapping = coerce_mapping(trainer)
    if mapping is None:
        return None

    runtime = coerce_mapping(mapping.get("runtime")) or coerce_mapping(mapping.get("runtime_summary"))
    plans = coerce_mapping(mapping.get("plans"))

    recommended_backend = pick_first(mapping, "recommended_backend", "selected_backend")
    requires_export_step = pick_first(mapping, "requires_export_step")
    if plans:
        for name in ("sft", "dpo"):
            subplan = coerce_mapping(plans.get(name))
            if subplan is None:
                continue
            if recommended_backend is None:
                recommended_backend = pick_first(subplan, "recommended_backend", "selected_backend")
            if requires_export_step is None:
                requires_export_step = pick_first(subplan, "requires_export_step")
            if recommended_backend is not None and requires_export_step is not None:
                break

    runtime_device = pick_first(runtime, "runtime_device")
    if recommended_backend is None and runtime_device is None and requires_export_step is None:
        return None

    parts = []
    if recommended_backend is not None:
        parts.append(f"recommended_backend={format_scalar(recommended_backend)}")
    if runtime_device is not None:
        parts.append(f"runtime_device={format_scalar(runtime_device)}")
    if requires_export_step is not None:
        parts.append(f"requires_export_step={format_scalar(requires_export_step)}")
    return "trainer: " + " | ".join(parts)


__all__ = ["format_trainer_block", "format_trainer_summary"]
