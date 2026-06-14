"""Action-based command guidance rules."""

from __future__ import annotations


def _action_command_guidance(action: str | None) -> tuple[str, str] | None:
    normalized = (action or "").strip().lower()
    if not normalized:
        return None
    if normalized == "promote_candidate":
        return ("/promote", "/candidate /cand sum")
    if normalized == "archive_candidate":
        return ("/archive", "/candidate /cand sum")
    if normalized == "inspect_candidate_status":
        return ("/candidate", "/cand tl")
    if normalized == "inspect_candidate_timeline":
        return ("/cand tl", "/candidate")
    if normalized == "process_next_queue_item":
        return ("/process", "/queue /qs")
    if normalized == "review_queue_confirmation":
        return ("/approve /reject", "/gate /trigger")
    if normalized == "recover_worker_daemon":
        return ("/recover daemon", "/runtime /daemon")
    if normalized == "inspect_daemon_restart_policy":
        return ("/runtime", "/alerts /daemon")
    if normalized == "inspect_runtime_stability":
        return ("/runtime", "/runner hist")
    if normalized == "inspect_worker_runner_history":
        return ("/runner hist", "/runtime")
    if normalized == "inspect_daemon_status":
        return ("/daemon", "/runtime")
    if normalized in {"inspect_daemon_heartbeat", "inspect_worker_stale_lock", "wait_for_runner_shutdown"}:
        return ("/runtime", "/runner /daemon")
    if normalized in {"enable_auto_evaluate", "inspect_auto_train_policy"}:
        return ("/policy", "/gate")
    if normalized == "inspect_auto_train_gate":
        return ("/gate", "/policy")
    if normalized == "inspect_auto_train_trigger":
        return ("/trigger", "/gate")
    if normalized == "wait_for_queue_completion":
        return ("/trigger", "/queue /qs")
    if normalized in {"collect_more_signal_samples", "collect_holdout_samples"}:
        return ("/gate", "/trigger /policy")
    if normalized == "wait_for_retrain_interval":
        return ("/trigger", "/gate /policy")
    if normalized == "wait_for_failure_backoff":
        return ("/retry", "/trigger /gate")
    return None


def _prompt_action_token_from_label(label: str | None, *, focus: str | None = None) -> str | None:
    normalized = str(label or "").strip().lower()
    normalized_focus = (focus or "none").strip().lower()
    if not normalized:
        return None
    if normalized.startswith("/approve"):
        return "review"
    if normalized.startswith("/process"):
        return "process"
    if normalized.startswith("/promote"):
        return "promote"
    if normalized.startswith("/archive"):
        return "archive"
    if normalized.startswith("/recover"):
        return "recover"
    if normalized.startswith("/restart"):
        return "restart"
    if normalized.startswith("/runtime"):
        return "runtime"
    if normalized.startswith("/candidate"):
        return "candidate"
    if normalized.startswith("/queue"):
        return "queue"
    if normalized.startswith("/runner"):
        return "runner"
    if normalized.startswith("/daemon"):
        return "daemon"
    if normalized.startswith("/policy"):
        return "policy" if "policy" in normalized_focus else "trigger"
    if normalized.startswith("/gate"):
        return (
            "trigger"
            if normalized_focus
            in {
                "insufficient_new_signal_samples",
                "holdout_not_ready",
                "cooldown_active",
                "failure_backoff_active",
                "policy_requires_auto_evaluate",
            }
            else "gate"
        )
    if normalized.startswith("/trigger"):
        return "trigger"
    if normalized.startswith("/retry"):
        return "retry"
    if normalized.startswith("/sum") or normalized.startswith("/status"):
        return "status"
    return None


__all__ = ["_action_command_guidance", "_prompt_action_token_from_label"]
