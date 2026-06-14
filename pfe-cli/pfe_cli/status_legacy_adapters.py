"""Legacy plain-text adapter status formatting."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def append_legacy_adapter_lines(
    lines: list[str],
    *,
    latest_adapter_version: Any,
    latest_adapter_state: Any,
    latest_adapter_map: Mapping[str, Any] | None,
    recent_adapter_version: Any,
    recent_adapter_state: Any,
    recent_adapter_map: Mapping[str, Any] | None,
    lifecycle: Mapping[str, Any] | None,
    deps: Any,
) -> None:
    """Append latest, recent, and lifecycle adapter lines."""
    _coerce_mapping = deps.coerce_mapping
    _format_adapter_export_artifact_line = deps.format_adapter_export_artifact_line
    _format_scalar = deps.format_scalar
    _pick_first = deps.pick_first

    latest_parts = []
    if latest_adapter_version is not None:
        latest_parts.append(f"version={_format_scalar(latest_adapter_version)}")
    if latest_adapter_state is not None:
        latest_parts.append(f"state={_format_scalar(latest_adapter_state)}")
    latest_samples = _pick_first(latest_adapter_map, "num_samples", "samples")
    if latest_samples is not None:
        latest_parts.append(f"samples={_format_scalar(latest_samples)}")
    latest_format = _pick_first(latest_adapter_map, "artifact_format", "format")
    if latest_format is not None:
        latest_parts.append(f"format={_format_scalar(latest_format)}")
    if latest_parts:
        lines.append("latest promoted: " + " | ".join(latest_parts))
    else:
        lines.append("latest promoted: none")
    latest_export_artifact_line = _format_adapter_export_artifact_line("latest export artifact", latest_adapter_map)
    if latest_export_artifact_line is not None:
        lines.append(latest_export_artifact_line)

    if recent_adapter_version is not None:
        recent_parts = [f"version={_format_scalar(recent_adapter_version)}"]
        if recent_adapter_state is not None:
            recent_parts.append(f"state={_format_scalar(recent_adapter_state)}")
        recent_samples = _pick_first(recent_adapter_map, "num_samples", "samples")
        if recent_samples is not None:
            recent_parts.append(f"samples={_format_scalar(recent_samples)}")
        recent_format = _pick_first(recent_adapter_map, "artifact_format", "format")
        if recent_format is not None:
            recent_parts.append(f"format={_format_scalar(recent_format)}")
        lines.append("recent training: " + " | ".join(recent_parts))
    recent_export_artifact_line = _format_adapter_export_artifact_line("recent export artifact", recent_adapter_map)
    if recent_export_artifact_line is not None:
        lines.append(recent_export_artifact_line)
    if lifecycle is not None:
        counts = _coerce_mapping(lifecycle.get("counts")) or {}
        if counts:
            ordered_states = ("pending_eval", "promoted", "failed_eval", "archived")
            summary = " | ".join(
                f"{state}={_format_scalar(counts.get(state, 0))}"
                for state in ordered_states
                if state in counts
            )
            if summary:
                lines.append(f"lifecycle: {summary}")


__all__ = ["append_legacy_adapter_lines"]
