"""Adapter-home formatting for doctor output."""

from __future__ import annotations

from .doctor_formatting_deps import DoctorFormattingDeps
from .doctor_snapshot_formatting import _format_doctor_snapshot_summary


def _format_doctor_adapter_home(workspace: str | None, deps: DoctorFormattingDeps) -> str:
    home = deps.pfe_home(workspace)
    latest_snapshot = deps.lookup_adapter_snapshot("latest", workspace=workspace)
    recent_snapshot = deps.lookup_recent_adapter_snapshot(workspace=workspace)
    parts = [
        f"adapter home: home={deps.format_scalar(home)} | "
        f"latest promoted={_format_doctor_snapshot_summary(latest_snapshot, include_latest=True, deps=deps)} "
        f"| recent training={_format_doctor_snapshot_summary(recent_snapshot, include_latest=True, deps=deps)}"
    ]
    latest_export_artifact_line = deps.format_adapter_export_artifact_line("latest export artifact", latest_snapshot)
    if latest_export_artifact_line is not None:
        parts.append(latest_export_artifact_line)
    recent_export_artifact_line = deps.format_adapter_export_artifact_line("recent export artifact", recent_snapshot)
    if recent_export_artifact_line is not None:
        parts.append(recent_export_artifact_line)
    return "\n".join(parts)


__all__ = ["_format_doctor_adapter_home"]
