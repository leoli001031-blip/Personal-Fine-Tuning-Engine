"""Source resolution helpers for the operations panel context."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .console_app_data import _mapping, _value


def trigger_policy_source(
    *,
    console: Mapping[str, Any],
    overview: Mapping[str, Any],
    trigger: Mapping[str, Any],
) -> dict[str, Any]:
    return _mapping(console.get("auto_train_policy")) or _mapping(overview.get("auto_train_policy")) or _mapping(
        trigger.get("policy")
    )


def trigger_policy_gate_source(
    *,
    trigger_policy: Mapping[str, Any],
    console: Mapping[str, Any],
) -> dict[str, Any]:
    return _mapping(trigger_policy.get("gate_summary")) or _mapping(console.get("trigger_policy_gate_summary"))


def trigger_threshold_source(
    *,
    console: Mapping[str, Any],
    overview: Mapping[str, Any],
    trigger: Mapping[str, Any],
) -> dict[str, Any]:
    return (
        _mapping(console.get("trigger_threshold_summary"))
        or _mapping(overview.get("trigger_threshold_summary"))
        or _mapping(trigger.get("threshold_summary"))
    )


def runtime_stability_source(
    *,
    console: Mapping[str, Any],
    overview: Mapping[str, Any],
) -> dict[str, Any]:
    return _mapping(console.get("runtime_stability_summary")) or _mapping(overview.get("runtime_stability_summary"))


def blocked_trigger_sources(
    *,
    console: Mapping[str, Any],
    overview: Mapping[str, Any],
) -> tuple[str, str, str]:
    return (
        _value(console, "trigger_blocked_reason", default=_value(overview, "trigger_blocked_reason", default="ready")),
        _value(console, "trigger_blocked_action", default=_value(overview, "trigger_blocked_action", default="none")),
        _value(console, "trigger_blocked_category", default=_value(overview, "trigger_blocked_category", default="n/a")),
    )


__all__ = [
    "blocked_trigger_sources",
    "runtime_stability_source",
    "trigger_policy_gate_source",
    "trigger_policy_source",
    "trigger_threshold_source",
]
