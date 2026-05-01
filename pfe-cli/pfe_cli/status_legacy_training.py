"""Legacy plain-text trainer, recent-training, and plan status formatting."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def append_legacy_trainer_and_plan_lines(
    lines: list[str],
    mapping: dict[str, Any],
    *,
    workspace: str | None,
    recent_adapter_version: Any,
    recent_adapter_state: Any,
    recent_adapter_map: Mapping[str, Any] | None,
    deps: Any,
) -> None:
    """Append trainer snapshot, recent training, and backend plan lines."""
    _build_plan_snapshots = deps.build_plan_snapshots
    _coerce_mapping = deps.coerce_mapping
    _format_backend_dispatch = deps.format_backend_dispatch
    _format_export_write = deps.format_export_write
    _format_recent_training_snapshot = deps.format_recent_training_snapshot
    _format_scalar = deps.format_scalar
    _format_trainer_summary = deps.format_trainer_summary
    _pick_first = deps.pick_first
    _read_cli_state = deps.read_cli_state

    trainer = mapping.pop("trainer", None)
    trainer_map = _coerce_mapping(trainer)
    last_run_map = _coerce_mapping(trainer_map.get("last_run")) if trainer_map is not None else None
    metadata = _coerce_mapping(mapping.pop("metadata", None))
    _coerce_mapping(mapping.pop("runtime", None))
    inference_runtime = None
    if metadata is not None:
        inference_runtime = _coerce_mapping(metadata.get("inference"))
    if inference_runtime is None:
        inference_runtime = _coerce_mapping(mapping.get("inference"))
    if inference_runtime is not None and "real_local_enabled" in inference_runtime:
        lines.append(f"real local inference: enabled={_format_scalar(inference_runtime.get('real_local_enabled'))}")
    plans = _coerce_mapping(mapping.pop("plans", None))
    if plans is None and metadata is not None:
        plans = _coerce_mapping(metadata.get("plans"))
    if trainer is None and metadata is not None:
        trainer = metadata.get("trainer")
    if trainer is None:
        trainer = mapping.get("trainer")
    if plans is None:
        plans = _build_plan_snapshots(
            workspace or mapping.get("workspace") or mapping.get("home"),
            {"metadata": metadata} if metadata else mapping,
        )

    recent_training_map = last_run_map or recent_adapter_map
    if recent_training_map is not None:
        recent_adapter_version = _pick_first(recent_training_map, "version") or recent_adapter_version
        recent_adapter_state = _pick_first(recent_training_map, "state") or recent_adapter_state

    trainer_line = _format_trainer_summary(trainer)
    if trainer_line is not None:
        lines.append(trainer_line)

    last_run = None
    if trainer_map is not None:
        last_run = _coerce_mapping(trainer_map.get("last_run"))
    if last_run is not None:
        recent_lines = _format_recent_training_snapshot(last_run)
        if recent_lines is not None:
            lines.extend(recent_lines)
    else:
        cached_state = _read_cli_state(workspace or mapping.get("workspace") or mapping.get("home"))
        if cached_state is not None:
            recent_snapshot = _coerce_mapping(cached_state.get("recent_training"))
            recent_lines = _format_recent_training_snapshot(recent_snapshot or cached_state)
            if recent_lines is not None:
                lines.extend(recent_lines)

    if plans:
        inference_plan = _coerce_mapping(plans.get("inference"))
        export_plan = _coerce_mapping(plans.get("export"))
        if inference_plan is not None:
            dispatch_line = _format_backend_dispatch(inference_plan)
            if dispatch_line is not None:
                lines.append(dispatch_line)
        if export_plan is not None:
            export_line = _format_export_write(export_plan)
            if export_line is not None:
                lines.append(export_line)


__all__ = ["append_legacy_trainer_and_plan_lines"]
