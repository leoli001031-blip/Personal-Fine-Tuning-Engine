"""Focus-to-action target rules for the prompt panel."""

from __future__ import annotations


_TRIGGER_FOCUSES = {
    "insufficient_new_signal_samples",
    "holdout_not_ready",
    "cooldown_active",
    "failure_backoff_active",
}


def _focus_target(normalized_focus: str) -> str:
    if normalized_focus.startswith("policy_") or normalized_focus.startswith("auto_train_policy"):
        return "trigger"
    if normalized_focus in _TRIGGER_FOCUSES:
        return "trigger"
    if "pending_review" in normalized_focus or "awaiting_confirmation" in normalized_focus:
        return "review"
    if "candidate_ready_for_promotion" in normalized_focus:
        return "promote"
    if normalized_focus in {"queue_waiting_execution", "queue_backlog"}:
        return "process"
    if normalized_focus.startswith("daemon") and "restart" in normalized_focus:
        return "restart"
    if normalized_focus.startswith("daemon") and ("heartbeat" in normalized_focus or "lease" in normalized_focus):
        return "recover"
    if normalized_focus == "daemon_active":
        return "runtime"
    if "runner" in normalized_focus and "stale" in normalized_focus:
        return "runtime"
    if normalized_focus.startswith("daemon"):
        return "daemon"
    if normalized_focus.startswith("candidate"):
        return "candidate"
    if normalized_focus.startswith("queue"):
        return "queue"
    if normalized_focus.startswith("runner"):
        return "runner"
    return ""


__all__ = ["_focus_target"]
