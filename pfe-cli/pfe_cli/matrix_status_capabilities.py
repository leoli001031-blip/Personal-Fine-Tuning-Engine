"""Capability and user-modeling status sections for Matrix terminal output."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .matrix_formatting_common import _coerce_mapping
from .terminal_theme import draw_box, format_key_value


def append_capability_status_sections(lines: list[str], mapping: Mapping[str, Any]) -> None:
    """Append capability boundary and user-modeling status boxes."""
    metadata = _coerce_mapping(mapping.get("metadata"))
    capabilities = _coerce_mapping(mapping.get("capabilities"))
    if capabilities is None and metadata is not None:
        capabilities = _coerce_mapping(metadata.get("capabilities"))
    if capabilities:
        capability_content = []
        for key in ("train", "eval", "serve", "generate", "distill", "profile", "route"):
            item = _coerce_mapping(capabilities.get(key))
            if not item:
                continue
            tier = item.get("tier", "unknown")
            summary = item.get("summary", "")
            capability_content.append(format_key_value(key, f"{tier} | {summary}"))
        if capability_content:
            lines.append(draw_box("CAPABILITY BOUNDARIES", capability_content))
            lines.append("")

    user_modeling = _coerce_mapping(mapping.get("user_modeling"))
    if user_modeling is None and metadata is not None:
        user_modeling = _coerce_mapping(metadata.get("user_modeling"))
    if user_modeling:
        user_modeling_content = [
            format_key_value(
                "runtime",
                f"{user_modeling.get('primary_runtime_system', 'n/a')} | status={user_modeling.get('primary_runtime_status', 'unknown')}",
            ),
            format_key_value(
                "analysis",
                f"{user_modeling.get('secondary_analysis_system', 'n/a')} | status={user_modeling.get('secondary_runtime_status', 'unknown')}",
            ),
        ]
        summary = user_modeling.get("summary")
        if summary:
            user_modeling_content.append(format_key_value("summary", summary))
        lines.append(draw_box("USER MODELING", user_modeling_content))
        lines.append("")


__all__ = ["append_capability_status_sections"]
