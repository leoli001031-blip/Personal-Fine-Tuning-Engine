"""Adapter export attention fragments for operations attention formatting."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def append_adapter_export_attention(
    alerts: list[str],
    *,
    latest_adapter_map: Mapping[str, Any] | None,
    recent_adapter_map: Mapping[str, Any] | None,
    deps: Any,
) -> None:
    for label, snapshot in (("latest export", latest_adapter_map), ("recent export", recent_adapter_map)):
        if snapshot is None:
            continue
        export_valid = snapshot.get("export_artifact_valid")
        export_exists = snapshot.get("export_artifact_exists")
        export_path = snapshot.get("export_artifact_path")
        if export_valid is False or export_exists is False:
            export_parts = [
                f"valid={deps.format_scalar(export_valid)}",
                f"exists={deps.format_scalar(export_exists)}",
            ]
            if export_path is not None:
                export_parts.append(f"path={deps.format_scalar(export_path)}")
            alerts.append(f"{label} " + " | ".join(export_parts))


__all__ = ["append_adapter_export_attention"]
