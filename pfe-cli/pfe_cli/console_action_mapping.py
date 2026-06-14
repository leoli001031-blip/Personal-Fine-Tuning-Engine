"""Console action-name to slash-command mapping."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .console_action_mapping_data import ACTION_MAPPINGS
from .console_actions_deps import ConsoleActionsDeps


def action_mapping(action_name: str) -> dict[str, str | None] | None:
    normalized_action = str(action_name or "").strip().lower()
    if not normalized_action:
        return None
    mapping = ACTION_MAPPINGS.get(normalized_action)
    return dict(mapping) if mapping is not None else None


def action_values(summary: Mapping[str, Any]) -> list[str]:
    values: list[str] = []
    for raw_action in [
        summary.get("secondary_action"),
        *list(summary.get("secondary_actions") or []),
    ]:
        text = str(raw_action or "").strip()
        if text and text not in values:
            values.append(text)
    return values


def apply_secondary_action_values(
    mapping: dict[str, str | None],
    secondary_action_values: list[str],
) -> dict[str, str | None]:
    secondary_labels: list[str] = []
    secondary_exec: str | None = None
    for secondary_action in secondary_action_values[:2]:
        mapped_secondary = action_mapping(secondary_action)
        if mapped_secondary is None:
            continue
        label = str(mapped_secondary.get("primary_label") or "").strip()
        if label and label not in secondary_labels:
            secondary_labels.append(label)
        if secondary_exec is None:
            secondary_exec = mapped_secondary.get("primary_exec")
    if secondary_labels:
        mapping["secondary_label"] = " ".join(secondary_labels)
        mapping["secondary_exec"] = secondary_exec
    return mapping


def summary_mapping(summary: Mapping[str, Any] | None, *, deps: ConsoleActionsDeps) -> dict[str, str | None] | None:
    summary_map = deps.coerce_mapping(summary) or {}
    primary_action = str(summary_map.get("primary_action") or "").strip()
    if not primary_action:
        return None
    mapped_primary = action_mapping(primary_action)
    if mapped_primary is None:
        return None
    return apply_secondary_action_values(
        mapped_primary,
        [str(action or "") for action in list(summary_map.get("secondary_actions") or [])],
    )


__all__ = ["action_mapping", "action_values", "apply_secondary_action_values", "summary_mapping"]
