"""Fallback focus action decisions for console shortcuts."""

from __future__ import annotations

from .console_focus_context import ConsoleFocusContext


def _focus_action(
    primary_label: str,
    primary_exec: str | None,
    secondary_label: str,
    secondary_exec: str | None,
) -> dict[str, str | None]:
    return {
        "primary_label": primary_label,
        "primary_exec": primary_exec,
        "secondary_label": secondary_label,
        "secondary_exec": secondary_exec,
    }


def fallback_console_focus_actions(context: ConsoleFocusContext) -> dict[str, str | None]:
    current_focus = context.current_focus

    if current_focus.startswith("policy_") or current_focus.startswith("auto_train_policy"):
        return _focus_action("/policy", "policy", "/gate", "gate")
    if current_focus in {"insufficient_new_signal_samples", "holdout_not_ready"}:
        return _focus_action("/gate", "gate", "/trigger /policy", "trigger")
    if current_focus == "cooldown_active":
        return _focus_action("/trigger", "trigger", "/gate /policy", "gate")
    if current_focus == "failure_backoff_active":
        return _focus_action("/retry", "retry", "/trigger /gate", "trigger")
    if "pending_review" in current_focus or "awaiting_confirmation" in current_focus:
        return _focus_action("/approve or /reject", None, "/gate /trigger", "gate")
    if current_focus.startswith("daemon") and "restart" in current_focus:
        return _focus_action("/restart daemon", "restart daemon", "/runtime /alerts", "runtime")
    if current_focus.startswith("daemon") and ("heartbeat" in current_focus or "lease" in current_focus):
        return _focus_action("/recover daemon", "recover daemon", "/runtime /daemon", "runtime")
    if "runner" in current_focus and "stale" in current_focus:
        return _focus_action("/runtime /runner", "runtime", "/runner hist", "runner hist")
    if "candidate_ready_for_promotion" in current_focus:
        return _focus_action("/promote", "promote", "/candidate /cand sum", "candidate")
    if current_focus in {"queue_waiting_execution", "queue_backlog"}:
        return _focus_action("/process", "process", "/queue /qs", "queue")
    if current_focus.startswith("daemon"):
        return _focus_action("/recover daemon", "recover daemon", "/runtime /daemon", "runtime")
    if current_focus.startswith("candidate"):
        can_promote = bool(context.candidate_summary.get("candidate_can_promote"))
        can_archive = bool(context.candidate_summary.get("candidate_can_archive"))
        if can_promote:
            return _focus_action("/promote", "promote", "/candidate /cand sum", "candidate")
        if can_archive:
            return _focus_action("/archive", "archive", "/candidate /cand sum", "candidate")
        return _focus_action("/candidate", "candidate", "/cand sum", "cand sum")
    if current_focus.startswith("queue"):
        return _focus_action("/trigger /gate", "trigger", "/queue /qs", "queue")
    if current_focus.startswith("runner"):
        return _focus_action("/runtime /runner", "runtime", "/rs /runner hist", "rs")
    return _focus_action("/sum /dash", "sum", "/status /help", "status")


__all__ = ["fallback_console_focus_actions"]
