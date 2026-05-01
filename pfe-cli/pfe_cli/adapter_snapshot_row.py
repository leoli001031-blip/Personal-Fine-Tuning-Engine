"""Adapter snapshot construction from adapter store rows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .adapter_snapshot_deps import AdapterSnapshotDeps


def export_artifact_summary_from_manifest(path: Any, *, deps: AdapterSnapshotDeps) -> dict[str, Any]:
    if not path:
        return {}
    try:
        manifest_payload = json.loads(Path(str(path)).expanduser().read_text(encoding="utf-8"))
    except Exception:
        return {}
    manifest_metadata = deps.coerce_mapping(manifest_payload.get("metadata")) or {}
    manifest_export = deps.coerce_mapping(manifest_metadata.get("export"))
    return (
        deps.coerce_mapping(manifest_metadata.get("export_artifact_summary"))
        or deps.coerce_mapping(manifest_export.get("artifact") if manifest_export else None)
        or {}
    )


def snapshot_from_row(
    row_map: dict[str, Any],
    *,
    latest_version: Any,
    deps: AdapterSnapshotDeps,
) -> dict[str, Any]:
    row_version = str(row_map.get("version"))
    metadata = deps.coerce_mapping(row_map.get("metadata")) or {}
    export_execution = deps.coerce_mapping(metadata.get("export_execution")) or {}
    export_write = deps.coerce_mapping(metadata.get("export_write")) or {}
    export_metadata = deps.coerce_mapping(metadata.get("export"))
    export_artifact_summary = (
        deps.coerce_mapping(metadata.get("export_artifact_summary"))
        or deps.coerce_mapping(export_metadata.get("artifact") if export_metadata else None)
        or {}
    )
    if not export_artifact_summary:
        export_artifact_summary = export_artifact_summary_from_manifest(row_map.get("manifest_path"), deps=deps)
    output_artifact_validation = deps.coerce_mapping(export_execution.get("output_artifact_validation")) or {}
    export_artifact_path = (
        output_artifact_validation.get("path")
        or export_execution.get("output_artifact_path")
        or export_write.get("artifact_path")
        or export_artifact_summary.get("path")
        or row_map.get("artifact_path")
    )
    export_artifact_exists = False
    export_artifact_size_bytes = None
    if export_artifact_path:
        try:
            export_artifact_file = Path(str(export_artifact_path)).expanduser()
            export_artifact_exists = export_artifact_file.exists()
            if export_artifact_exists and export_artifact_file.is_file():
                export_artifact_size_bytes = export_artifact_file.stat().st_size
        except Exception:
            export_artifact_exists = False
    return {
        "version": row_map.get("version"),
        "state": row_map.get("state", row_map.get("status")),
        "latest": row_map.get("latest")
        if "latest" in row_map
        else (latest_version is not None and row_version == latest_version),
        "num_samples": row_map.get("num_samples", row_map.get("samples")),
        "artifact_format": row_map.get("artifact_format", row_map.get("format")),
        "export_status": export_execution.get("status")
        or deps.pick_first(deps.coerce_mapping(export_execution.get("audit")), "status")
        or export_artifact_summary.get("status"),
        "export_write_state": export_write.get("write_state")
        or deps.pick_first(deps.coerce_mapping(export_write.get("metadata")), "write_state")
        or export_artifact_summary.get("write_state"),
        "export_artifact_path": export_artifact_path,
        "export_artifact_valid": output_artifact_validation.get("valid", export_artifact_summary.get("valid")),
        "export_artifact_exists": export_artifact_exists,
        "export_artifact_size_bytes": (
            export_artifact_size_bytes
            if export_artifact_size_bytes is not None
            else export_artifact_summary.get("size_bytes")
        ),
    }


__all__ = ["export_artifact_summary_from_manifest", "snapshot_from_row"]
