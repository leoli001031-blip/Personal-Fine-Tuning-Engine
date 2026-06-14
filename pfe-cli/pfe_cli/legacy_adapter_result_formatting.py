"""Legacy adapter result summary formatting helpers."""

from __future__ import annotations

from typing import Any

from .legacy_result_common import format_bytes_compact
from .legacy_result_deps import LegacyResultFormattingDeps


def format_adapter_snapshot_line(
    label: str,
    snapshot: Any,
    *,
    include_latest: bool = False,
    deps: LegacyResultFormattingDeps,
) -> str | None:
    mapping = deps.coerce_mapping(snapshot)
    if mapping is None:
        return None

    parts: list[str] = []
    version = deps.pick_first(mapping, "version")
    if version is not None:
        parts.append(f"version={deps.format_scalar(version)}")
    state = deps.pick_first(mapping, "state")
    if state is not None:
        parts.append(f"state={deps.format_scalar(state)}")
    if include_latest:
        latest = deps.pick_first(mapping, "latest")
        if latest is not None:
            parts.append(f"latest={deps.format_scalar(latest)}")
    samples = deps.pick_first(mapping, "num_samples", "samples")
    if samples is not None:
        parts.append(f"samples={deps.format_scalar(samples)}")
    artifact_format = deps.pick_first(mapping, "artifact_format", "format")
    if artifact_format is not None:
        parts.append(f"format={deps.format_scalar(artifact_format)}")
    if not parts:
        return None
    return f"{label}: " + " | ".join(parts)


def format_adapter_export_artifact_line(
    label: str,
    snapshot: Any,
    *,
    deps: LegacyResultFormattingDeps,
) -> str | None:
    mapping = deps.coerce_mapping(snapshot)
    if mapping is None:
        return None

    artifact_path = deps.pick_first(mapping, "export_artifact_path")
    artifact_valid = deps.pick_first(mapping, "export_artifact_valid")
    artifact_size = format_bytes_compact(deps.pick_first(mapping, "export_artifact_size_bytes"))
    export_status = deps.pick_first(mapping, "export_status")
    write_state = deps.pick_first(mapping, "export_write_state")
    artifact_exists = deps.pick_first(mapping, "export_artifact_exists")

    parts: list[str] = []
    if export_status is not None:
        parts.append(f"status={deps.format_scalar(export_status)}")
    if write_state is not None:
        parts.append(f"write_state={deps.format_scalar(write_state)}")
    if artifact_valid is not None:
        parts.append(f"valid={deps.format_scalar(artifact_valid)}")
    if artifact_exists is not None:
        parts.append(f"exists={deps.format_scalar(artifact_exists)}")
    if artifact_size is not None:
        parts.append(f"size={artifact_size}")
    if artifact_path:
        parts.append(f"path={deps.format_scalar(artifact_path)}")
    if not parts:
        return None
    return f"{label}: " + " | ".join(parts)


__all__ = ["format_adapter_export_artifact_line", "format_adapter_snapshot_line"]
