"""Legacy auto-train trigger policy and gate formatting."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .status_legacy_auto_train_trigger_fields import POLICY_KEYS, THRESHOLD_KEYS, append_scalar_parts


def append_trigger_policy_lines(lines: list[str], auto_trigger: Mapping[str, Any], *, deps: Any) -> None:
    policy = deps.coerce_mapping(auto_trigger.get("policy"))
    if policy is None:
        return
    policy_parts: list[str] = []
    append_scalar_parts(policy_parts, policy, POLICY_KEYS, deps=deps)
    if policy_parts:
        lines.append("auto train trigger policy: " + " | ".join(policy_parts))


def append_trigger_threshold_lines(lines: list[str], auto_trigger: Mapping[str, Any], *, deps: Any) -> None:
    threshold_summary = deps.coerce_mapping(auto_trigger.get("threshold_summary"))
    if threshold_summary is None:
        return
    threshold_parts: list[str] = []
    append_scalar_parts(threshold_parts, threshold_summary, THRESHOLD_KEYS, deps=deps)
    if threshold_parts:
        lines.append("auto train trigger gate: " + " | ".join(threshold_parts))


__all__ = ["append_trigger_policy_lines", "append_trigger_threshold_lines"]
