"""Shared backend dispatch and export write formatting helpers."""

from __future__ import annotations

from typing import Any

from .shared_coercion_formatting import coerce_mapping, format_scalar, pick_first


def format_backend_dispatch(plan: Any) -> str | None:
    mapping = coerce_mapping(plan)
    if mapping is None:
        return None
    execution_backend = pick_first(mapping, "selected_backend", "recommended_backend", "execution_backend")
    requested_backend = pick_first(mapping, "requested_backend")
    reason = str(pick_first(mapping, "reason") or "").lower()
    execution_mode = pick_first(mapping, "execution_mode")
    if execution_mode is None:
        if "mock_local" in str(execution_backend or "").lower() or any(
            token in reason for token in ("fallback", "auto-selected", "dry-run")
        ):
            execution_mode = "fallback"
        else:
            execution_mode = "real"
    runtime_device = pick_first(mapping, "runtime_device", "preferred_device")
    requires_export_step = pick_first(mapping, "requires_export_step")
    required_artifact_format = pick_first(mapping, "required_artifact_format", "export_format")

    parts = []
    if execution_backend is not None:
        parts.append(f"execution_backend={format_scalar(execution_backend)}")
    if execution_mode is not None:
        parts.append(f"execution_mode={format_scalar(execution_mode)}")
    if runtime_device is not None:
        parts.append(f"runtime_device={format_scalar(runtime_device)}")
    if requires_export_step is not None:
        parts.append(f"requires_export_step={format_scalar(requires_export_step)}")
    if required_artifact_format is not None:
        parts.append(f"required_artifact_format={format_scalar(required_artifact_format)}")
    if requested_backend is not None and execution_backend != requested_backend:
        parts.append(f"requested_backend={format_scalar(requested_backend)}")
    return "backend-dispatch: " + " | ".join(parts) if parts else None


def format_export_write(plan: Any) -> str | None:
    mapping = coerce_mapping(plan)
    if mapping is None:
        return None
    target_artifact_format = pick_first(mapping, "target_artifact_format", "artifact_format")
    required = pick_first(mapping, "required")
    if required is None:
        required = str(target_artifact_format).lower() == "gguf_merged"
    gguf_export = "required" if str(target_artifact_format).lower() == "gguf_merged" or bool(required) else "not_required"
    parts = [f"gguf_export={gguf_export}"]
    if target_artifact_format is not None:
        parts.append(f"target_artifact_format={format_scalar(target_artifact_format)}")
    metadata = coerce_mapping(mapping.get("metadata"))
    execution_intent = pick_first(mapping, "execution_intent")
    if execution_intent is None:
        execution_intent = pick_first(metadata, "execution_intent")
    if execution_intent is not None:
        parts.append(f"execution_intent={format_scalar(execution_intent)}")
    execution_status = pick_first(mapping, "status", "execution_status")
    if execution_status is None:
        audit = coerce_mapping(mapping.get("audit"))
        execution_status = pick_first(audit, "status")
    dry_run = pick_first(mapping, "dry_run")
    if dry_run is not None:
        parts.append(f"dry_run={format_scalar(dry_run)}")
    output_dir = pick_first(mapping, "output_dir")
    if output_dir is not None:
        parts.append(f"output_dir={format_scalar(output_dir)}")
    artifact_name = pick_first(mapping, "artifact_name")
    if artifact_name is not None:
        parts.append(f"artifact_name={format_scalar(artifact_name)}")
    output_artifact_path = pick_first(mapping, "output_artifact_path", "artifact_path")
    if output_artifact_path is not None:
        parts.append(f"artifact_path={format_scalar(output_artifact_path)}")
    command = pick_first(mapping, "command")
    if command is not None:
        parts.append(f"command={format_scalar(command)}")
    write_state = pick_first(mapping, "write_state")
    if write_state is None:
        write_state = pick_first(metadata, "write_state")
    if write_state is None:
        write_state = "planned"
    if execution_status is not None and write_state in {"planned", "ready"}:
        if execution_status in {"success", "dry_run", "tool_missing", "failed", "not_required"}:
            write_state = str(execution_status)
    if command is not None or output_dir is not None:
        write_state = "ready" if write_state == "planned" else write_state
    if dry_run is False:
        write_state = "executing" if write_state in {"planned", "ready"} else write_state
    parts.insert(1, f"write_state={write_state}")
    if execution_status is not None:
        parts.append(f"execution_status={format_scalar(execution_status)}")
    return "export-write: " + " | ".join(parts)


__all__ = ["format_backend_dispatch", "format_export_write"]
