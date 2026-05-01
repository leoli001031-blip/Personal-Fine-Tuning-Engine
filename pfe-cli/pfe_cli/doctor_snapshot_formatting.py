"""Adapter snapshot summary formatting for doctor output."""

from __future__ import annotations

from typing import Any

from .doctor_formatting_deps import DoctorFormattingDeps


def _format_doctor_snapshot_summary(
    snapshot: Any,
    *,
    include_latest: bool = False,
    deps: DoctorFormattingDeps,
) -> str:
    mapping = deps.coerce_mapping(snapshot)
    if mapping is None:
        return "n/a"

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
    return " | ".join(parts) if parts else "n/a"


__all__ = ["_format_doctor_snapshot_summary"]
