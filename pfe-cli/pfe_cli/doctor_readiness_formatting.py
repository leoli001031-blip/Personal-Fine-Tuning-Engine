"""Doctor blocked-capability and next-step formatting."""

from __future__ import annotations

from typing import Any

from .doctor_formatting_deps import DoctorFormattingDeps


def _format_doctor_blocked_capabilities(
    *,
    trainer_line: str | None,
    local_model_line: str | None,
    export_tool_line: str | None,
    latest_snapshot: Any,
    recent_snapshot: Any,
    deps: DoctorFormattingDeps,
) -> str:
    blocked: list[str] = []

    if local_model_line is None or "available=yes" not in local_model_line:
        blocked.extend(["train", "eval", "serve"])

    latest_snapshot_map = deps.coerce_mapping(latest_snapshot)
    recent_snapshot_map = deps.coerce_mapping(recent_snapshot)
    if latest_snapshot_map is None and recent_snapshot_map is None:
        blocked.append("eval")

    if export_tool_line is None or "allowed=yes" not in export_tool_line:
        blocked.append("export")

    if trainer_line is None or "ready=yes" not in trainer_line:
        blocked.append("train")

    unique_blocked = list(dict.fromkeys(blocked))
    if not unique_blocked:
        return "blocked capabilities: none"
    return "blocked capabilities: " + ", ".join(unique_blocked)


def _format_doctor_next_steps(
    *,
    trainer_line: str | None,
    local_model_line: str | None,
    export_tool_line: str | None,
    latest_snapshot: Any,
    recent_snapshot: Any,
    deps: DoctorFormattingDeps,
) -> str:
    steps: list[str] = []

    if local_model_line is None or "available=yes" not in local_model_line:
        steps.append("set a base_model in the latest adapter manifest or pass --base-model")

    if trainer_line is None or "ready=yes" not in trainer_line:
        steps.append("install torch, transformers, peft, accelerate, trl, and datasets")

    latest_snapshot_map = deps.coerce_mapping(latest_snapshot)
    recent_snapshot_map = deps.coerce_mapping(recent_snapshot)
    if latest_snapshot_map is None and recent_snapshot_map is None:
        steps.append("train an adapter to create a workspace snapshot")
    elif latest_snapshot_map is None and recent_snapshot_map is not None:
        steps.append("promote the recent adapter once it passes eval")

    if export_tool_line is None or "allowed=yes" not in export_tool_line:
        steps.append("put the llama.cpp export tool on PATH or configure its location")

    if not steps:
        steps.append("run pfe train or pfe eval as needed")

    return "next steps: " + "; ".join(steps)


__all__ = ["_format_doctor_blocked_capabilities", "_format_doctor_next_steps"]
