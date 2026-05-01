"""Local model probe formatting for doctor output."""

from __future__ import annotations

from .doctor_formatting_deps import DoctorFormattingDeps


def _format_doctor_local_model(
    workspace: str | None,
    base_model: str | None,
    deps: DoctorFormattingDeps,
) -> str | None:
    manifest = deps.load_latest_adapter_manifest(workspace)
    manifest_map = deps.coerce_mapping(manifest)
    requested_base_model = base_model
    if requested_base_model is None and manifest_map is not None:
        requested_base_model = deps.pick_first(manifest_map, "base_model")

    if requested_base_model is None:
        return "local model: available=no | requested_base_model=n/a | reason=no base model configured"

    local_source = deps.optional_module_call(
        "pfe_core.trainer.executors",
        "_resolve_real_local_model_source",
        {"base_model": requested_base_model},
    )
    local_source_map = deps.coerce_mapping(local_source)
    if local_source_map is None:
        local_source_map = {
            "available": False,
            "requested_base_model": requested_base_model,
            "source_kind": "unavailable",
            "source_path": None,
            "config_path": None,
            "load_mode": "unavailable",
            "reason": "local model probe unavailable",
        }

    parts: list[str] = []
    for key in ("available", "requested_base_model", "source_kind", "source_path", "config_path", "load_mode", "reason"):
        value = local_source_map.get(key)
        if value is not None:
            parts.append(f"{key}={deps.format_scalar(value)}")
    if not parts:
        return None
    return "local model: " + " | ".join(parts)


__all__ = ["_format_doctor_local_model"]
