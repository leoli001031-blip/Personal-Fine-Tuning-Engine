"""Focus-based command guidance rules."""

from __future__ import annotations


def _focus_command_guidance(focus: str | None) -> tuple[str, str]:
    normalized = (focus or "none").strip().lower()
    if normalized.startswith("policy_") or normalized.startswith("auto_train_policy"):
        return ("/policy", "/gate")
    if normalized in {"insufficient_new_signal_samples", "holdout_not_ready"}:
        return ("/gate", "/trigger /policy")
    if normalized == "cooldown_active":
        return ("/trigger", "/gate /policy")
    if normalized == "failure_backoff_active":
        return ("/retry", "/trigger /gate")
    if "pending_review" in normalized or "awaiting_confirmation" in normalized:
        return ("/approve /reject", "/gate /trigger")
    if normalized.startswith("daemon") and "restart" in normalized:
        return ("/restart daemon", "/runtime /alerts")
    if normalized.startswith("daemon") and ("heartbeat" in normalized or "lease" in normalized):
        return ("/recover daemon", "/runtime /daemon")
    if "runner" in normalized and "stale" in normalized:
        return ("/runtime /runner", "/runner hist")
    if "candidate_ready_for_promotion" in normalized:
        return ("/promote", "/candidate /cand sum")
    if normalized in {"queue_waiting_execution", "queue_backlog"}:
        return ("/process", "/queue /qs")
    if normalized.startswith("daemon"):
        return ("/recover daemon", "/runtime /daemon")
    if normalized.startswith("candidate"):
        return ("/candidate", "/cand sum")
    if normalized.startswith("queue"):
        return ("/trigger /gate", "/queue /qs")
    if normalized.startswith("runner"):
        return ("/runtime /runner", "/rs /runner hist")
    return ("/sum /dash", "/status /help")


__all__ = ["_focus_command_guidance"]
