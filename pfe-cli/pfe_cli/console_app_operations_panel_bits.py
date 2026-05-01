"""Compact policy and gate bit rendering for the operations panel."""

from __future__ import annotations

from .console_app_data import _compact_text, _value, _yes_no
from .console_app_operations_context import OperationsPanelContext


def _handling_bits(ctx: OperationsPanelContext) -> list[str]:
    handling_mode = _value(ctx.alert_policy, "escalation_mode", "remediation_mode", default="monitor")
    return [
        ctx.priority_value,
        handling_mode,
        f"h={_yes_no(ctx.alert_policy.get('requires_human_review', False))}",
        f"a={_yes_no(ctx.alert_policy.get('auto_remediation_allowed', False))}",
        f"now={_yes_no(ctx.alert_policy.get('requires_immediate_action', False))}",
    ]


def _policy_bits(ctx: OperationsPanelContext) -> list[str]:
    return [
        _value(ctx.trigger_policy, "queue_entry_mode", default="disabled"),
        _value(ctx.trigger_policy, "evaluation_mode", default="skip"),
        _value(ctx.trigger_policy, "promotion_mode", default="manual"),
        f"QRev={_value(ctx.queue_review_policy, 'review_mode', default='auto_queue')}",
        f"QNext={_compact_text(_value(ctx.queue_review_policy, 'next_action', default='await_signal_trigger'), max_len=22)}",
    ]


def _policy_gate_bits(ctx: OperationsPanelContext) -> list[str]:
    return [
        (
            f"e={_value(ctx.trigger_policy_gate, 'eval_num_samples', default='0')}:"
            f"{'on' if bool(ctx.trigger_policy_gate.get('auto_evaluate_enabled')) else 'off'}"
        ),
        f"p={'on' if bool(ctx.trigger_policy_gate.get('auto_promote_requested')) else 'off'}",
        _compact_text(_value(ctx.trigger_policy_gate, "promotion_requirement", default="manual"), max_len=10),
    ]


def _gate_bits(ctx: OperationsPanelContext) -> list[str]:
    return [
        (
            f"s={_value(ctx.trigger_threshold, 'eligible_signal_train_samples', default='0')}/"
            f"{_value(ctx.trigger_threshold, 'min_new_samples', default='0')}"
        ),
        (
            f"e={_value(ctx.trigger_threshold, 'effective_eligible_train_samples', default='0')}/"
            f"{_value(ctx.trigger_threshold, 'min_new_samples', default='0')}"
        ),
        f"r={_value(ctx.trigger_threshold, 'preference_reinforced_train_samples', default='0')}",
        f"h={_yes_no(ctx.trigger_threshold.get('holdout_ready')) if 'holdout_ready' in ctx.trigger_threshold else 'n/a'}",
        (
            f"i={_yes_no(ctx.trigger_threshold.get('interval_elapsed'))}"
            if "interval_elapsed" in ctx.trigger_threshold
            else "i=n/a"
        ),
        (
            f"cd={_yes_no(ctx.trigger_threshold.get('cooldown_elapsed'))}"
            if "cooldown_elapsed" in ctx.trigger_threshold
            else "cd=n/a"
        ),
        (
            f"bo={_yes_no(ctx.trigger_threshold.get('failure_backoff_elapsed'))}"
            if "failure_backoff_elapsed" in ctx.trigger_threshold
            else "bo=n/a"
        ),
    ]


def _review_bits(ctx: OperationsPanelContext) -> list[str]:
    return [
        _value(ctx.queue_review_policy, "review_mode", default="auto_queue"),
        _value(ctx.queue_review_policy, "queue_entry_mode", default="inline_execute"),
        _value(ctx.queue_review_policy, "next_action", default="await_signal_trigger"),
    ]


__all__ = [
    "_gate_bits",
    "_handling_bits",
    "_policy_bits",
    "_policy_gate_bits",
    "_review_bits",
]
