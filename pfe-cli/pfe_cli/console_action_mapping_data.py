"""Static console action-name mappings."""

from __future__ import annotations

ActionMapping = dict[str, str | None]

ACTION_MAPPINGS: dict[str, ActionMapping] = {
    "promote_candidate": {
        "primary_label": "/promote",
        "primary_exec": "promote",
        "secondary_label": "/candidate /cand sum",
        "secondary_exec": "candidate",
    },
    "archive_candidate": {
        "primary_label": "/archive",
        "primary_exec": "archive",
        "secondary_label": "/candidate /cand sum",
        "secondary_exec": "candidate",
    },
    "inspect_candidate_status": {
        "primary_label": "/candidate",
        "primary_exec": "candidate",
        "secondary_label": "/cand tl",
        "secondary_exec": "cand tl",
    },
    "inspect_candidate_timeline": {
        "primary_label": "/cand tl",
        "primary_exec": "cand tl",
        "secondary_label": "/candidate",
        "secondary_exec": "candidate",
    },
    "process_next_queue_item": {
        "primary_label": "/process",
        "primary_exec": "process",
        "secondary_label": "/queue /qs",
        "secondary_exec": "queue",
    },
    "review_queue_confirmation": {
        "primary_label": "/approve or /reject",
        "primary_exec": None,
        "secondary_label": "/gate /trigger",
        "secondary_exec": "gate",
    },
    "recover_worker_daemon": {
        "primary_label": "/recover daemon",
        "primary_exec": "recover daemon",
        "secondary_label": "/runtime /daemon",
        "secondary_exec": "runtime",
    },
    "inspect_daemon_restart_policy": {
        "primary_label": "/runtime",
        "primary_exec": "runtime",
        "secondary_label": "/alerts /daemon",
        "secondary_exec": "alerts",
    },
    "inspect_runtime_stability": {
        "primary_label": "/runtime",
        "primary_exec": "runtime",
        "secondary_label": "/runner hist",
        "secondary_exec": "runner hist",
    },
    "inspect_worker_runner_history": {
        "primary_label": "/runner hist",
        "primary_exec": "runner hist",
        "secondary_label": "/runtime",
        "secondary_exec": "runtime",
    },
    "inspect_daemon_status": {
        "primary_label": "/daemon",
        "primary_exec": "daemon",
        "secondary_label": "/runtime",
        "secondary_exec": "runtime",
    },
    "inspect_daemon_heartbeat": {
        "primary_label": "/runtime",
        "primary_exec": "runtime",
        "secondary_label": "/runner /daemon",
        "secondary_exec": "runtime",
    },
    "inspect_worker_stale_lock": {
        "primary_label": "/runtime",
        "primary_exec": "runtime",
        "secondary_label": "/runner /daemon",
        "secondary_exec": "runtime",
    },
    "wait_for_runner_shutdown": {
        "primary_label": "/runtime",
        "primary_exec": "runtime",
        "secondary_label": "/runner /daemon",
        "secondary_exec": "runtime",
    },
    "enable_auto_evaluate": {
        "primary_label": "/policy",
        "primary_exec": "policy",
        "secondary_label": "/gate",
        "secondary_exec": "gate",
    },
    "inspect_auto_train_policy": {
        "primary_label": "/policy",
        "primary_exec": "policy",
        "secondary_label": "/gate",
        "secondary_exec": "gate",
    },
    "inspect_auto_train_gate": {
        "primary_label": "/gate",
        "primary_exec": "gate",
        "secondary_label": "/policy",
        "secondary_exec": "policy",
    },
    "inspect_auto_train_trigger": {
        "primary_label": "/trigger",
        "primary_exec": "trigger",
        "secondary_label": "/gate",
        "secondary_exec": "gate",
    },
    "wait_for_queue_completion": {
        "primary_label": "/trigger",
        "primary_exec": "trigger",
        "secondary_label": "/queue /qs",
        "secondary_exec": "queue",
    },
    "collect_more_signal_samples": {
        "primary_label": "/gate",
        "primary_exec": "gate",
        "secondary_label": "/trigger /policy",
        "secondary_exec": "trigger",
    },
    "collect_holdout_samples": {
        "primary_label": "/gate",
        "primary_exec": "gate",
        "secondary_label": "/trigger /policy",
        "secondary_exec": "trigger",
    },
    "wait_for_retrain_interval": {
        "primary_label": "/trigger",
        "primary_exec": "trigger",
        "secondary_label": "/gate /policy",
        "secondary_exec": "gate",
    },
    "wait_for_failure_backoff": {
        "primary_label": "/retry",
        "primary_exec": "retry",
        "secondary_label": "/trigger /gate",
        "secondary_exec": "trigger",
    },
    "inspect_compare_evaluation": {
        "primary_label": "/candidate",
        "primary_exec": "candidate",
        "secondary_label": "/gate /trigger",
        "secondary_exec": "gate",
    },
    "inspect_candidate_gate": {
        "primary_label": "/gate",
        "primary_exec": "gate",
        "secondary_label": "/candidate /trigger",
        "secondary_exec": "candidate",
    },
    "rollback_candidate": {
        "primary_label": "/rollback",
        "primary_exec": "rollback",
        "secondary_label": "/candidate /archive",
        "secondary_exec": "candidate",
    },
    "evaluate": {
        "primary_label": "/eval",
        "primary_exec": "eval",
        "secondary_label": "/candidate /gate",
        "secondary_exec": "candidate",
    },
    "run_distillation": {
        "primary_label": "/distill",
        "primary_exec": "distill",
        "secondary_label": "/gate /trigger",
        "secondary_exec": "gate",
    },
    "force_recovery": {
        "primary_label": "/force-recovery",
        "primary_exec": "force-recovery",
        "secondary_label": "/runtime /recover daemon",
        "secondary_exec": "runtime",
    },
    "process_train_queue_batch": {
        "primary_label": "/batch",
        "primary_exec": "batch",
        "secondary_label": "/queue /process",
        "secondary_exec": "queue",
    },
    "process_train_queue_until_idle": {
        "primary_label": "/until-idle",
        "primary_exec": "until-idle",
        "secondary_label": "/queue /process",
        "secondary_exec": "queue",
    },
    "stop_train_queue_daemon": {
        "primary_label": "/stop daemon",
        "primary_exec": "stop daemon",
        "secondary_label": "/daemon /runtime",
        "secondary_exec": "daemon",
    },
    "start_train_queue_daemon": {
        "primary_label": "/start daemon",
        "primary_exec": "start daemon",
        "secondary_label": "/daemon /runtime",
        "secondary_exec": "daemon",
    },
    "stop_train_queue_worker_runner": {
        "primary_label": "/stop runner",
        "primary_exec": "stop runner",
        "secondary_label": "/runner /runtime",
        "secondary_exec": "runner",
    },
}

__all__ = ["ACTION_MAPPINGS", "ActionMapping"]
