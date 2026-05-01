"""Auto-train trigger status section for Matrix terminal output."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .matrix_formatting_common import _coerce_mapping, _format_scalar
from .terminal_theme import draw_box, format_key_value


def append_auto_train_trigger_section(lines: list[str], mapping: Mapping[str, Any]) -> None:
    """Append auto-train trigger status box."""
    auto_train_trigger = _coerce_mapping(mapping.get("auto_train_trigger"))
    if auto_train_trigger:
        trig = []
        enabled = auto_train_trigger.get("enabled", False)
        state = auto_train_trigger.get("state", "unknown")
        ready = auto_train_trigger.get("ready", False)
        trig.append(format_key_value("enabled", "yes" if enabled else "no"))
        trig.append(format_key_value("state", state))
        trig.append(format_key_value("ready", "yes" if ready else "no"))
        reason = auto_train_trigger.get("reason", "")
        if reason:
            trig.append(format_key_value("reason", reason))
        blocked_reasons = auto_train_trigger.get("blocked_reasons")
        if blocked_reasons:
            trig.append(format_key_value("blocked reasons", str(blocked_reasons)))
        min_new = auto_train_trigger.get("min_new_samples")
        if min_new is not None:
            trig.append(format_key_value("min new samples", min_new))
        preference_weight = auto_train_trigger.get("preference_reinforced_sample_weight")
        if preference_weight is not None:
            trig.append(format_key_value("preference reinforced sample weight", preference_weight))
        eligible = auto_train_trigger.get("eligible_signal_train_samples")
        if eligible is not None:
            trig.append(format_key_value("eligible signals", eligible))
        effective_eligible = auto_train_trigger.get("effective_eligible_train_samples")
        if effective_eligible is not None:
            trig.append(format_key_value("effective eligible train samples", effective_eligible))
        reinforced = auto_train_trigger.get("preference_reinforced_train_samples")
        if reinforced is not None:
            trig.append(format_key_value("preference reinforced train samples", reinforced))
        for key in (
            "min_trigger_interval_minutes",
            "failure_backoff_minutes",
            "queue_mode",
            "queue_dedup_scope",
            "queue_priority_policy",
            "queue_process_batch_size",
            "queue_process_until_idle_max",
            "queue_gate_reason",
            "queue_gate_action",
            "queue_review_mode",
        ):
            value = auto_train_trigger.get(key)
            if value is not None:
                trig.append(format_key_value(key.replace("_", " "), value))
        for key in (
            "blocked_primary_reason",
            "blocked_primary_action",
            "blocked_primary_category",
            "consecutive_failures",
            "recent_training_version",
        ):
            value = auto_train_trigger.get(key)
            if value is not None:
                trig.append(format_key_value(key.replace("_", " "), value))
        for key in ("holdout_ready", "interval_elapsed", "cooldown_elapsed", "failure_backoff_elapsed"):
            value = auto_train_trigger.get(key)
            if value is not None:
                trig.append(format_key_value(key.replace("_", " "), "yes" if value else "no"))
        policy = _coerce_mapping(auto_train_trigger.get("policy"))
        if policy:
            pol_parts = " | ".join(f"{k.replace('_', ' ')}={_format_scalar(v)}" for k, v in policy.items() if v is not None)
            if pol_parts:
                trig.append(format_key_value("policy", pol_parts))
        threshold = _coerce_mapping(auto_train_trigger.get("threshold_summary"))
        if threshold:
            ts_parts = " | ".join(f"{k.replace('_', ' ')}={_format_scalar(v)}" for k, v in threshold.items() if v is not None)
            if ts_parts:
                trig.append(format_key_value("thresholds", ts_parts))
        blocked_summary = auto_train_trigger.get("blocked_summary")
        if blocked_summary:
            trig.append(format_key_value("blocked summary", blocked_summary))
        last_result = _coerce_mapping(auto_train_trigger.get("last_result"))
        if last_result:
            lr_parts = " | ".join(f"{k.replace('_', ' ')}={_format_scalar(v)}" for k, v in last_result.items() if v is not None)
            if lr_parts:
                trig.append(format_key_value("last result", lr_parts))
        lines.append(draw_box("AUTO TRAIN TRIGGER", trig))
        lines.append("")


__all__ = ["append_auto_train_trigger_section"]
