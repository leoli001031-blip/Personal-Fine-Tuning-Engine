"""Legacy export execution summary formatting."""

from __future__ import annotations

from typing import Any

from .legacy_result_deps import LegacyResultFormattingDeps


def format_export_execution_summary(
    export_execution: Any,
    *,
    deps: LegacyResultFormattingDeps,
) -> str | None:
    mapping = deps.coerce_mapping(export_execution)
    if mapping is None:
        return None

    parts: list[str] = []
    audit = deps.coerce_mapping(mapping.get("audit"))
    metadata = deps.coerce_mapping(mapping.get("metadata"))
    status = deps.pick_first(audit, "status", "execution_status") or deps.pick_first(
        mapping, "status", "execution_status"
    )
    execution_mode = deps.pick_first(metadata, "execution_mode") or deps.pick_first(mapping, "execution_mode")
    attempted = deps.pick_first(mapping, "attempted")
    success = deps.pick_first(mapping, "success")

    if status is not None:
        parts.append(f"status={deps.format_scalar(status)}")
    if execution_mode is not None:
        parts.append(f"execution_mode={deps.format_scalar(execution_mode)}")
    if attempted is not None:
        parts.append(f"attempted={deps.format_scalar(attempted)}")
    if success is not None:
        parts.append(f"success={deps.format_scalar(success)}")
    if not parts:
        return None
    return "export-execution: " + " | ".join(parts)


def format_export_toolchain_summary(
    export_execution: Any,
    *,
    deps: LegacyResultFormattingDeps,
) -> str | None:
    mapping = deps.coerce_mapping(export_execution)
    if mapping is None:
        return None

    parts: list[str] = []
    audit = deps.coerce_mapping(mapping.get("audit"))
    metadata = deps.coerce_mapping(mapping.get("metadata"))
    status = deps.pick_first(mapping, "summary", "status", "toolchain_status") or deps.pick_first(
        audit,
        "status",
        "execution_status",
    )
    execution_mode = deps.pick_first(mapping, "execution_mode") or deps.pick_first(metadata, "execution_mode")
    attempted = deps.pick_first(mapping, "attempted")
    success = deps.pick_first(mapping, "success")
    required = deps.pick_first(mapping, "required")
    artifact_valid = deps.pick_first(mapping, "output_artifact_valid")

    if status is not None:
        parts.append(f"status={deps.format_scalar(status)}")
    if execution_mode is not None:
        parts.append(f"execution_mode={deps.format_scalar(execution_mode)}")
    if attempted is not None:
        parts.append(f"attempted={deps.format_scalar(attempted)}")
    if success is not None:
        parts.append(f"success={deps.format_scalar(success)}")
    if required is not None:
        parts.append(f"required={deps.format_scalar(required)}")
    if artifact_valid is not None:
        parts.append(f"artifact_valid={deps.format_scalar(artifact_valid)}")
    if not parts:
        return None
    return "export-toolchain: " + " | ".join(parts)


__all__ = ["format_export_execution_summary", "format_export_toolchain_summary"]
