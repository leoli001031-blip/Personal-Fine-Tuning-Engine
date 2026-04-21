"""Matrix Green Terminal formatters for PFE CLI output.

Transforms standard PFE CLI output into Terminal Hacker aesthetic.
"""

from __future__ import annotations

import json
from typing import Any, Sequence, Mapping
from dataclasses import asdict, is_dataclass

from .terminal_theme import (
    MatrixColors,
    Borders,
    draw_box,
    draw_header,
    draw_separator,
    draw_table,
    status_badge,
    progress_bar,
    format_key_value,
    STYLE_SUCCESS,
    STYLE_WARNING,
    STYLE_ERROR,
    STYLE_DIM,
)


def _coerce_mapping(result: Any) -> dict[str, Any] | None:
    """Convert pydantic-style results into a plain mapping when possible."""
    if isinstance(result, dict):
        return dict(result)
    if is_dataclass(result) and not isinstance(result, type):
        try:
            dumped = asdict(result)
        except Exception:
            return None
        if isinstance(dumped, dict):
            return dict(dumped)
    model_dump = getattr(result, "model_dump", None)
    if callable(model_dump):
        try:
            dumped = model_dump()
        except Exception:
            return None
        if isinstance(dumped, dict):
            return dict(dumped)
    to_dict = getattr(result, "dict", None)
    if callable(to_dict):
        try:
            dumped = to_dict()
        except Exception:
            return None
        if isinstance(dumped, dict):
            return dict(dumped)
    return None


def _coerce_sequence_of_mappings(result: Any) -> list[dict[str, Any]]:
    if not isinstance(result, Sequence) or isinstance(result, (str, bytes, bytearray)):
        return []
    items: list[dict[str, Any]] = []
    for item in result:
        mapping = _coerce_mapping(item)
        if mapping is not None:
            items.append(mapping)
    return items


def _coerce_sequence_of_scalars(result: Any) -> list[str]:
    if not isinstance(result, Sequence) or isinstance(result, (str, bytes, bytearray)):
        return []
    return [str(item) for item in result if item is not None]


def _format_scalar(value: Any) -> str:
    """Format a scalar value."""
    if value is None:
        return f"{MatrixColors.DIM}n/a{MatrixColors.RESET}"
    if isinstance(value, bool):
        return f"{MatrixColors.GREEN}yes{MatrixColors.RESET}" if value else f"{MatrixColors.GRAY}no{MatrixColors.RESET}"
    if isinstance(value, (str, int, float)):
        return f"{MatrixColors.WHITE}{value}{MatrixColors.RESET}"
    if isinstance(value, Mapping):
        return f"{MatrixColors.GRAY}{json.dumps(value, ensure_ascii=False, sort_keys=True)}{MatrixColors.RESET}"
    return f"{MatrixColors.WHITE}{str(value)}{MatrixColors.RESET}"


def _ordered_eval_scores(scores: Mapping[str, Any]) -> list[tuple[str, Any]]:
    """Return evaluation scores with personalization-oriented keys first."""
    preferred_order = (
        "style_preference_hit_rate",
        "style_match",
        "preference_alignment",
        "quality_preservation",
        "personality_consistency",
    )
    ordered: list[tuple[str, Any]] = []
    seen: set[str] = set()
    for key in preferred_order:
        if key in scores:
            ordered.append((key, scores[key]))
            seen.add(key)
    for key, value in scores.items():
        if key in seen:
            continue
        ordered.append((key, value))
    return ordered


def format_status_matrix(result: Any, *, workspace: str | None = None) -> str:
    """Format status output in Matrix Green terminal style."""
    lines = []

    # Header
    ws_text = workspace or "default"
    lines.append(draw_header(f"PFE STATUS // WORKSPACE: {ws_text}"))

    mapping = _coerce_mapping(result)
    if mapping is None:
        lines.append(f"{MatrixColors.RED}ERROR: Unable to parse status data{MatrixColors.RESET}")
        return "\n".join(lines)

    # Adapter Info Section
    adapter_content = []

    latest_adapter = _coerce_mapping(mapping.get("latest_adapter"))
    if latest_adapter:
        version = latest_adapter.get("version", "n/a")
        state = latest_adapter.get("state", "unknown")
        latest_parts = [f"{version} | state={state}"]
        for key in ("export_artifact_valid", "export_artifact_exists"):
            value = latest_adapter.get(key)
            if value is not None:
                latest_parts.append(f"{key}={value}")
        export_artifact_path = latest_adapter.get("export_artifact_path")
        if export_artifact_path is not None:
            latest_parts.append(f"export_artifact_path={export_artifact_path}")
        adapter_content.append(format_key_value("latest promoted", " | ".join(latest_parts)))

    recent_adapter = _coerce_mapping(mapping.get("recent_adapter"))
    if recent_adapter:
        version = recent_adapter.get("version", "n/a")
        state = recent_adapter.get("state", "unknown")
        recent_parts = [f"{version} | state={state}"]
        execution_backend = recent_adapter.get("execution_backend")
        if execution_backend is not None:
            recent_parts.append(f"execution_backend={execution_backend}")
        executor_mode = recent_adapter.get("executor_mode")
        if executor_mode is not None:
            recent_parts.append(f"executor_mode={executor_mode}")
        adapter_content.append(format_key_value("recent training", " | ".join(recent_parts)))

    # Signal Summary
    signal_summary = _coerce_mapping(mapping.get("signal_summary"))
    if signal_summary:
        total = signal_summary.get("total_signals", 0)
        processed = signal_summary.get("processed_signals", 0)
        adapter_content.append(format_key_value("signals", f"total={total} | processed={processed}"))

    # Sample Counts
    sample_counts = _coerce_mapping(mapping.get("sample_counts"))
    if sample_counts:
        train = sample_counts.get("train", 0)
        val = sample_counts.get("val", 0)
        test = sample_counts.get("test", 0)
        adapter_content.append(format_key_value("samples", f"train={train} | val={val} | test={test}"))

    lifecycle = _coerce_mapping(mapping.get("adapter_lifecycle"))
    if lifecycle:
        counts = _coerce_mapping(lifecycle.get("counts"))
        if counts:
            lifecycle_parts = []
            for key in ("promoted", "archived", "pending_eval", "training", "failed_eval"):
                val = counts.get(key)
                if val:
                    lifecycle_parts.append(f"{key}={val}")
            if lifecycle_parts:
                adapter_content.append(format_key_value("lifecycle counts", " | ".join(lifecycle_parts)))
        promoted_versions = lifecycle.get("promoted_versions")
        if promoted_versions:
            adapter_content.append(format_key_value("promoted versions", ", ".join(str(v) for v in promoted_versions)))
        archived_versions = lifecycle.get("archived_versions")
        if archived_versions:
            adapter_content.append(format_key_value("archived versions", ", ".join(str(v) for v in archived_versions)))

    if adapter_content:
        lines.append(draw_box("ADAPTER LIFECYCLE", adapter_content))
        lines.append("")

    # Capability Boundaries
    metadata = _coerce_mapping(mapping.get("metadata"))
    capabilities = _coerce_mapping(mapping.get("capabilities"))
    if capabilities is None and metadata is not None:
        capabilities = _coerce_mapping(metadata.get("capabilities"))
    if capabilities:
        capability_content = []
        for key in ("train", "eval", "serve", "generate", "distill", "profile", "route"):
            item = _coerce_mapping(capabilities.get(key))
            if not item:
                continue
            tier = item.get("tier", "unknown")
            summary = item.get("summary", "")
            capability_content.append(format_key_value(key, f"{tier} | {summary}"))
        if capability_content:
            lines.append(draw_box("CAPABILITY BOUNDARIES", capability_content))
            lines.append("")

    user_modeling = _coerce_mapping(mapping.get("user_modeling"))
    if user_modeling is None and metadata is not None:
        user_modeling = _coerce_mapping(metadata.get("user_modeling"))
    if user_modeling:
        user_modeling_content = [
            format_key_value("runtime", f"{user_modeling.get('primary_runtime_system', 'n/a')} | status={user_modeling.get('primary_runtime_status', 'unknown')}"),
            format_key_value("analysis", f"{user_modeling.get('secondary_analysis_system', 'n/a')} | status={user_modeling.get('secondary_runtime_status', 'unknown')}"),
        ]
        summary = user_modeling.get("summary")
        if summary:
            user_modeling_content.append(format_key_value("summary", summary))
        lines.append(draw_box("USER MODELING", user_modeling_content))
        lines.append("")

    # Signal Readiness
    signal_readiness = _coerce_mapping(mapping.get("signal_readiness"))
    if signal_readiness is None:
        signal_summary = _coerce_mapping(mapping.get("signal_summary"))
        if signal_summary:
            signal_readiness = {
                "state": signal_summary.get("state", "unknown"),
                "total_ready_signals": signal_summary.get("event_chain_complete_count", 0),
                "event_chain_ready": signal_summary.get("event_chain_ready", False),
                "readiness_reason": signal_summary.get("quality_filter_state", ""),
                "signal_quality_filter": {
                    "total_signals": signal_summary.get("processed_count", 0),
                    "passed_signals": signal_summary.get("quality_filtered_count", 0),
                    "rejected_signals": signal_summary.get("quality_filtered_count", 0),
                    "rejection_reasons": signal_summary.get("quality_filter_reasons", {}),
                },
            }
    if signal_readiness:
        sig_content = []
        state = signal_readiness.get("state", "unknown")
        sig_content.append(format_key_value("state", status_badge(state)))
        total_ready = signal_readiness.get("total_ready_signals", 0)
        sig_content.append(format_key_value("ready signals", total_ready))
        readiness_reason = signal_readiness.get("readiness_reason", "")
        if readiness_reason:
            sig_content.append(format_key_value("reason", readiness_reason))
        event_chain_ready = signal_readiness.get("event_chain_ready")
        if event_chain_ready is not None:
            sig_content.append(format_key_value("event chain ready", "yes" if event_chain_ready else "no"))
        signal_quality = _coerce_mapping(signal_readiness.get("signal_quality_filter"))
        if signal_quality:
            q_total = signal_quality.get("total_signals", 0)
            q_passed = signal_quality.get("passed_signals", 0)
            q_rejected = signal_quality.get("rejected_signals", 0)
            sig_content.append(format_key_value("quality filter", f"total={q_total} | passed={q_passed} | rejected={q_rejected}"))
            q_reasons = _coerce_mapping(signal_quality.get("rejection_reasons"))
            if q_reasons:
                for reason, count in q_reasons.items():
                    sig_content.append(format_key_value(f"  {reason.replace('_', ' ')}", count))
        lines.append(draw_box("SIGNAL READINESS", sig_content))
        lines.append("")

    # Training Status Section
    current_training = _coerce_mapping(mapping.get("current_training"))
    if current_training:
        training_content = []
        status = current_training.get("status", "idle")
        version = current_training.get("version", "n/a")

        training_content.append(format_key_value("status", status_badge(status)))
        training_content.append(format_key_value("version", version))

        epochs = current_training.get("epochs", 0)
        current_epoch = current_training.get("current_epoch", 0)
        if epochs > 0:
            training_content.append(format_key_value("progress", progress_bar(current_epoch, epochs)))

        lines.append(draw_box("TRAINING STATUS", training_content))
        lines.append("")

    # Candidate Summary
    candidate_summary = _coerce_mapping(mapping.get("candidate_summary"))
    if candidate_summary:
        cands = []
        for key in (
            "candidate_version",
            "candidate_state",
            "candidate_can_promote",
            "candidate_can_archive",
            "pending_eval_count",
            "candidate_needs_promotion",
            "promotion_gate_status",
            "promotion_gate_reason",
            "promotion_gate_action",
            "promotion_compare_comparison",
            "promotion_compare_recommendation",
            "promotion_compare_winner",
            "promotion_compare_overall_delta",
            "promotion_compare_style_preference_hit_rate_delta",
            "promotion_compare_personalization_summary",
            "promotion_compare_quality_summary",
            "promotion_compare_summary_line",
        ):
            value = candidate_summary.get(key)
            if value is not None:
                cands.append(format_key_value(key.replace("_", " "), value))
        if cands:
            lines.append(draw_box("CANDIDATE SUMMARY", cands))
            lines.append("")

    # Candidate Action
    candidate_action = _coerce_mapping(mapping.get("candidate_action"))
    if candidate_action:
        action_lines = []
        for key in ("action", "status", "reason", "required_action", "operator_note", "candidate_version", "promoted_version", "archived_version"):
            value = candidate_action.get(key)
            if value is not None:
                action_lines.append(format_key_value(key.replace("_", " "), value))
        if action_lines:
            lines.append(draw_box("CANDIDATE ACTION", action_lines))
            lines.append("")

    # Candidate History
    candidate_history = _coerce_mapping(mapping.get("candidate_history"))
    if candidate_history:
        hist_lines = []
        for key in ("count", "last_action", "last_status", "last_reason", "last_note", "last_candidate_version"):
            value = candidate_history.get(key)
            if value is not None:
                hist_lines.append(format_key_value(key.replace("_", " "), value))
        if hist_lines:
            lines.append(draw_box("CANDIDATE HISTORY", hist_lines))
            lines.append("")

    # Auto Train Trigger
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
        for key in ("min_trigger_interval_minutes", "failure_backoff_minutes", "queue_mode", "queue_dedup_scope", "queue_priority_policy", "queue_process_batch_size", "queue_process_until_idle_max", "queue_gate_reason", "queue_gate_action", "queue_review_mode"):
            value = auto_train_trigger.get(key)
            if value is not None:
                trig.append(format_key_value(key.replace("_", " "), value))
        for key in ("blocked_primary_reason", "blocked_primary_action", "blocked_primary_category", "consecutive_failures", "recent_training_version"):
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
    auto_trigger_action = _coerce_mapping(mapping.get("auto_train_trigger_action"))
    if auto_trigger_action:
        action_lines = []
        for key in (
            "action", "status", "reason", "triggered", "queue_job_id",
            "confirmation_reason", "approval_reason", "rejection_reason",
            "processed_count", "completed_count", "failed_count", "limit",
            "max_iterations", "max_cycles", "loop_cycles", "idle_rounds",
            "remaining_queued", "drained", "stopped_reason", "triggered_version",
            "promoted_version",
        ):
            value = auto_trigger_action.get(key)
            if value is not None:
                action_lines.append(format_key_value(key.replace("_", " "), value))
        if action_lines:
            lines.append(draw_box("AUTO TRAIN ACTION", action_lines))
            lines.append("")

    # Train Queue
    train_queue = _coerce_mapping(mapping.get("train_queue"))
    if train_queue:
        q_content = []
        count = train_queue.get("count", 0)
        max_priority = train_queue.get("max_priority")
        q_content.append(format_key_value("count", count))
        if max_priority is not None:
            q_content.append(format_key_value("max priority", max_priority))
        counts = _coerce_mapping(train_queue.get("counts"))
        if counts:
            q_content.append(format_key_value("states", ",".join(f"{n}:{counts.get(n)}" for n in sorted(counts))))
        current = _coerce_mapping(train_queue.get("current"))
        if current:
            q_content.append(format_key_value("current", f"{current.get('job_id','')} | {current.get('state','')}"))
        last_item = _coerce_mapping(train_queue.get("last_item"))
        if last_item:
            q_content.append(format_key_value("last", f"{last_item.get('job_id','')} | {last_item.get('state','')} | {last_item.get('adapter_version','')}"))
        policy_summary = _coerce_mapping(train_queue.get("policy_summary"))
        if policy_summary:
            ps = " | ".join(f"{k.replace('_', ' ')}={_format_scalar(v)}" for k, v in policy_summary.items() if v is not None)
            if ps:
                q_content.append(format_key_value("policy", ps))
        confirmation_summary = _coerce_mapping(train_queue.get("confirmation_summary"))
        if confirmation_summary:
            cs = " | ".join(f"{k.replace('_', ' ')}={_format_scalar(v)}" for k, v in confirmation_summary.items() if v is not None)
            if cs:
                q_content.append(format_key_value("confirmation", cs))
        worker_runner = _coerce_mapping(train_queue.get("worker_runner"))
        if worker_runner:
            wr_keys = ["active", "lock_state", "stop_requested", "processed_count", "failed_count", "loop_cycles", "last_action", "last_event"]
            wr_parts = " | ".join(
                f"{k.replace('_', ' ')}={_format_scalar(worker_runner.get(k))}"
                for k in wr_keys
                if worker_runner.get(k) is not None
            )
            if wr_parts:
                q_content.append(format_key_value("worker runner", wr_parts))
        if q_content:
            lines.append(draw_box("TRAIN QUEUE", q_content))
            lines.append("")

    # Real Execution Summary
    real_execution_summary = _coerce_mapping(mapping.get("real_execution_summary"))
    if real_execution_summary is None:
        real_execution_summary = _coerce_mapping(mapping.get("job_execution"))
    if real_execution_summary is None:
        recent_training_snapshot = _coerce_mapping(mapping.get("recent_training_snapshot"))
        if recent_training_snapshot:
            real_execution_summary = _coerce_mapping(
                recent_training_snapshot.get("real_execution_summary")
                or recent_training_snapshot.get("job_execution_summary")
                or recent_training_snapshot.get("job_execution")
            )
    if real_execution_summary:
        re_content = []
        for key in ("status", "state", "kind", "executor_mode", "execution_mode", "attempted", "success", "available", "runner_status"):
            value = real_execution_summary.get(key)
            if value is not None:
                re_content.append(format_key_value(key.replace("_", " "), value))
        audit = _coerce_mapping(real_execution_summary.get("audit"))
        if audit:
            for key in ("runner_status", "status", "execution_status"):
                value = audit.get(key)
                if value is not None:
                    re_content.append(format_key_value(f"audit {key.replace('_', ' ')}", value))
        if re_content:
            lines.append(draw_box("REAL EXECUTION", re_content))
            lines.append("")

    # Export Toolchain Summary
    export_toolchain_summary = _coerce_mapping(mapping.get("export_toolchain_summary"))
    if export_toolchain_summary is None:
        export_toolchain_summary = _coerce_mapping(mapping.get("export_execution"))
    if export_toolchain_summary is None:
        recent_training_snapshot = _coerce_mapping(mapping.get("recent_training_snapshot"))
        if recent_training_snapshot:
            export_toolchain_summary = _coerce_mapping(recent_training_snapshot.get("export_execution") or recent_training_snapshot.get("export_toolchain_summary"))
    if export_toolchain_summary:
        et_content = []
        for key in ("status", "summary", "toolchain_status", "execution_mode", "attempted", "success", "required", "output_artifact_valid"):
            value = export_toolchain_summary.get(key)
            if value is not None:
                et_content.append(format_key_value(key.replace("_", " "), value))
        metadata = _coerce_mapping(export_toolchain_summary.get("metadata"))
        if metadata:
            for key in ("execution_mode", "status"):
                value = metadata.get(key)
                if value is not None:
                    et_content.append(format_key_value(f"meta {key}", value))
        audit = _coerce_mapping(export_toolchain_summary.get("audit"))
        if audit:
            for key in ("status", "execution_status"):
                value = audit.get(key)
                if value is not None:
                    et_content.append(format_key_value(f"audit {key}", value))
        if et_content:
            lines.append(draw_box("EXPORT TOOLCHAIN", et_content))
            lines.append("")

    # System Health Section
    system_health = _coerce_mapping(mapping.get("system_health"))
    if system_health:
        health_content = []

        daemon_active = system_health.get("daemon_active", False)
        runner_active = system_health.get("runner_active", False)

        daemon_status = f"{MatrixColors.GREEN}ONLINE{MatrixColors.RESET}" if daemon_active else f"{MatrixColors.RED}OFFLINE{MatrixColors.RESET}"
        runner_status = f"{MatrixColors.GREEN}ONLINE{MatrixColors.RESET}" if runner_active else f"{MatrixColors.RED}OFFLINE{MatrixColors.RESET}"

        health_content.append(format_key_value("daemon", daemon_status))
        health_content.append(format_key_value("runner", runner_status))

        queue_pending = system_health.get("queue_pending_jobs", 0)
        queue_failed = system_health.get("queue_failed_jobs", 0)
        health_content.append(format_key_value("queue pending", queue_pending))
        health_content.append(format_key_value("queue failed", queue_failed if queue_failed == 0 else f"{MatrixColors.RED}{queue_failed}{MatrixColors.RESET}"))

        lines.append(draw_box("SYSTEM HEALTH", health_content))
        lines.append("")

    # Operations Overview
    operations = _coerce_mapping(mapping.get("operations_overview"))
    if operations:
        ops_content = []

        attention = operations.get("attention_needed", False)
        if attention:
            reason = operations.get("attention_reason", "unknown")
            ops_content.append(f"{MatrixColors.AMBER}⚠ ATTENTION REQUIRED: {reason}{MatrixColors.RESET}")

        trigger_state = operations.get("trigger_state", "unknown")
        ops_content.append(format_key_value("trigger state", status_badge(trigger_state)))

        summary = operations.get("summary_line", "")
        if summary:
            ops_content.append(format_key_value("summary", summary))

        for key in ("current_focus", "monitor_focus", "required_action", "candidate_version", "candidate_state", "queue_count", "awaiting_confirmation_count", "runner_active", "runner_lock_state", "runner_last_event"):
            value = operations.get(key)
            if value is not None:
                ops_content.append(format_key_value(key.replace("_", " "), value))

        auto_train_blocker = _coerce_mapping(operations.get("auto_train_blocker"))
        if auto_train_blocker:
            block = []
            for k, v in auto_train_blocker.items():
                if v is not None:
                    block.append(f"{k}={_format_scalar(v)}")
            if block:
                ops_content.append(format_key_value("blocker", " | ".join(block)))

        lines.append(draw_box("OPERATIONS", ops_content))
        lines.append("")

    # Operations Dashboard
    operations_dashboard = _coerce_mapping(mapping.get("operations_dashboard"))
    if operations_dashboard:
        dash_content = []
        for key, value in operations_dashboard.items():
            if value is not None and key not in ("dashboard",):
                dash_content.append(format_key_value(key.replace("_", " "), value))
        nested_dashboard = _coerce_mapping(operations_dashboard.get("dashboard"))
        if nested_dashboard:
            for key, value in nested_dashboard.items():
                if value is not None:
                    dash_content.append(format_key_value(f"dashboard {key.replace('_', ' ')}".strip(), value))
        if dash_content:
            lines.append(draw_box("OPERATIONS DASHBOARD", dash_content))
            lines.append("")

    # Operations Alert Policy
    operations_alert_policy = _coerce_mapping(mapping.get("operations_alert_policy"))
    if operations_alert_policy:
        policy_content = []
        for key, value in operations_alert_policy.items():
            if value is not None:
                policy_content.append(format_key_value(key.replace("_", " "), value))
        if policy_content:
            lines.append(draw_box("OPERATIONS ALERT POLICY", policy_content))
            lines.append("")

    # Operations Console
    operations_console = _coerce_mapping(mapping.get("operations_console"))
    if operations_console:
        console_content = []
        attention_needed = operations_console.get("attention_needed")
        if attention_needed is not None:
            console_content.append(format_key_value("attention needed", "yes" if attention_needed else "no"))
        attention_reason = operations_console.get("attention_reason")
        if attention_reason:
            console_content.append(format_key_value("attention reason", attention_reason))
        summary_line = operations_console.get("summary_line")
        if summary_line:
            console_content.append(format_key_value("summary", summary_line))
        next_actions = operations_console.get("next_actions")
        if next_actions:
            console_content.append(format_key_value("next actions", ", ".join(str(a) for a in next_actions)))
        candidate = _coerce_mapping(operations_console.get("candidate"))
        if candidate:
            cands = []
            for key, value in candidate.items():
                if value is not None:
                    cands.append(f"{key.replace('_', ' ')}={_format_scalar(value)}")
            if cands:
                console_content.append(format_key_value("candidate", " | ".join(cands)))
        queue = _coerce_mapping(operations_console.get("queue"))
        if queue:
            qs = []
            for key, value in queue.items():
                if value is not None:
                    qs.append(f"{key.replace('_', ' ')}={_format_scalar(value)}")
            if qs:
                console_content.append(format_key_value("queue", " | ".join(qs)))
        runner = _coerce_mapping(operations_console.get("runner"))
        if runner:
            rs = []
            for key, value in runner.items():
                if value is not None:
                    rs.append(f"{key.replace('_', ' ')}={_format_scalar(value)}")
            if rs:
                console_content.append(format_key_value("runner", " | ".join(rs)))
        timelines = _coerce_mapping(operations_console.get("timelines"))
        if timelines:
            ts = []
            for key, value in timelines.items():
                if value is not None:
                    ts.append(f"{key.replace('_', ' ')}={_format_scalar(value)}")
            if ts:
                console_content.append(format_key_value("timelines", " | ".join(ts)))
        if console_content:
            lines.append(draw_box("OPERATIONS CONSOLE", console_content))
            lines.append("")

    # Operations Alerts
    operations_alerts = _coerce_sequence_of_mappings(mapping.get("operations_alerts"))
    if operations_alerts:
        alert_content = []
        for alert in operations_alerts:
            severity = alert.get("severity", "info")
            message = alert.get("message", alert.get("alert", "unknown alert"))
            badge = status_badge(severity) if severity in ("info", "warning", "error", "critical") else f"{MatrixColors.GRAY}[ {severity.upper()} ]{MatrixColors.RESET}"
            alert_content.append(f"  {badge} {message}")
        if alert_content:
            lines.append(draw_box("OPERATIONS ALERTS", alert_content))
            lines.append("")

    # Operations Event Stream
    operations_event_stream = _coerce_mapping(mapping.get("operations_event_stream"))
    if operations_event_stream is None:
        operations_console = _coerce_mapping(mapping.get("operations_console"))
        if operations_console:
            operations_event_stream = _coerce_mapping(operations_console.get("event_stream"))
    if operations_event_stream:
        es_content = []
        for key, value in operations_event_stream.items():
            if value is not None and key != "dashboard" and not isinstance(value, (list, tuple)):
                es_content.append(format_key_value(key.replace("_", " "), value))
        nested_dashboard = _coerce_mapping(operations_event_stream.get("dashboard"))
        if nested_dashboard:
            for key, value in nested_dashboard.items():
                if value is not None:
                    es_content.append(format_key_value(f"dashboard {key.replace('_', ' ')}".strip(), value))
        if es_content:
            lines.append(draw_box("OPERATIONS EVENT STREAM", es_content))
            lines.append("")

    # Runner Timeline
    runner_timeline = _coerce_mapping(mapping.get("runner_timeline"))
    if runner_timeline is None:
        operations_console = _coerce_mapping(mapping.get("operations_console"))
        if operations_console:
            runner_timeline = _coerce_mapping(operations_console.get("runner_timeline"))
    if runner_timeline:
        rt_content = []
        count = runner_timeline.get("count", 0)
        rt_content.append(format_key_value("count", count))
        last_event = runner_timeline.get("last_event")
        if last_event:
            rt_content.append(format_key_value("last event", last_event))
        last_reason = runner_timeline.get("last_reason")
        if last_reason:
            rt_content.append(format_key_value("last reason", last_reason))
        for key in ("takeover_event_count", "last_takeover_event", "last_takeover_reason", "recent_anomaly_reason"):
            value = runner_timeline.get(key)
            if value is not None:
                rt_content.append(format_key_value(key.replace("_", " "), value))
        for key in ("current_active", "current_stop_requested"):
            value = runner_timeline.get(key)
            if value is not None:
                rt_content.append(format_key_value(key.replace("_", " "), "yes" if value else "no"))
        current_lock_state = runner_timeline.get("current_lock_state")
        if current_lock_state is not None:
            rt_content.append(format_key_value("current lock state", current_lock_state))
        events = _coerce_sequence_of_mappings(runner_timeline.get("events") or runner_timeline.get("recent_events"))
        if events:
            for ev in events[:5]:
                ev_parts = " | ".join(f"{k.replace('_', ' ')}={_format_scalar(v)}" for k, v in ev.items() if v is not None)
                if ev_parts:
                    rt_content.append(f"  {MatrixColors.GREEN_DIM}>{MatrixColors.RESET} {ev_parts}")
        takeover_events = _coerce_sequence_of_mappings(runner_timeline.get("recent_takeover_events"))
        if takeover_events:
            for ev in takeover_events[:3]:
                ev_parts = " | ".join(f"{k.replace('_', ' ')}={_format_scalar(v)}" for k, v in ev.items() if v is not None)
                if ev_parts:
                    rt_content.append(f"  {MatrixColors.AMBER}>{MatrixColors.RESET} {ev_parts}")
        if rt_content:
            lines.append(draw_box("RUNNER TIMELINE", rt_content))
            lines.append("")

    # Daemon Timeline
    daemon_timeline = _coerce_mapping(mapping.get("daemon_timeline"))
    if daemon_timeline is None:
        operations_console = _coerce_mapping(mapping.get("operations_console"))
        if operations_console:
            daemon_timeline = _coerce_mapping(operations_console.get("daemon_timeline"))
    if daemon_timeline:
        dt_content = []
        count = daemon_timeline.get("count", 0)
        dt_content.append(format_key_value("count", count))
        recovery_event_count = daemon_timeline.get("recovery_event_count")
        if recovery_event_count is not None:
            dt_content.append(format_key_value("recovery event count", recovery_event_count))
        last_event = daemon_timeline.get("last_event")
        if last_event:
            dt_content.append(format_key_value("last event", last_event))
        last_reason = daemon_timeline.get("last_reason")
        if last_reason:
            dt_content.append(format_key_value("last reason", last_reason))
        for key in ("last_recovery_event", "last_recovery_reason", "last_recovery_note", "recent_anomaly_reason"):
            value = daemon_timeline.get(key)
            if value is not None:
                dt_content.append(format_key_value(key.replace("_", " "), value))
        latest_timestamp = daemon_timeline.get("latest_timestamp")
        if latest_timestamp is not None:
            dt_content.append(format_key_value("latest timestamp", latest_timestamp))
        events = _coerce_sequence_of_mappings(daemon_timeline.get("events"))
        if events:
            for ev in events[:5]:
                ev_parts = " | ".join(f"{k.replace('_', ' ')}={_format_scalar(v)}" for k, v in ev.items() if v is not None)
                if ev_parts:
                    dt_content.append(f"  {MatrixColors.GREEN_DIM}>{MatrixColors.RESET} {ev_parts}")
        recovery_events = _coerce_sequence_of_mappings(daemon_timeline.get("recent_recovery_events"))
        if recovery_events:
            dt_content.append(f"{MatrixColors.GREEN_BRIGHT}recent recovery events:{MatrixColors.RESET}")
            for ev in recovery_events[:5]:
                ev_parts = " | ".join(f"{k.replace('_', ' ')}={_format_scalar(v)}" for k, v in ev.items() if v is not None)
                if ev_parts:
                    dt_content.append(f"  {MatrixColors.AMBER}>{MatrixColors.RESET} {ev_parts}")
        if dt_content:
            lines.append(draw_box("DAEMON TIMELINE", dt_content))
            lines.append("")

    # Operations Health
    operations_health = _coerce_mapping(mapping.get("operations_health"))
    if operations_health:
        oh_content = []
        for key, value in operations_health.items():
            if value is not None:
                oh_content.append(format_key_value(key.replace("_", " "), value))
        if oh_content:
            lines.append(draw_box("OPERATIONS HEALTH", oh_content))
            lines.append("")

    # Operations Recovery
    operations_recovery = _coerce_mapping(mapping.get("operations_recovery"))
    if operations_recovery:
        or_content = []
        for key, value in operations_recovery.items():
            if value is not None:
                or_content.append(format_key_value(key.replace("_", " "), value))
        if or_content:
            lines.append(draw_box("OPERATIONS RECOVERY", or_content))
            lines.append("")

    # Operations Next Actions
    operations_next_actions = _coerce_sequence_of_scalars(mapping.get("operations_next_actions"))
    if operations_next_actions:
        lines.append(draw_box("NEXT ACTIONS", [", ".join(str(a) for a in operations_next_actions)]))
        lines.append("")

    # Footer
    lines.append(draw_separator())
    lines.append(f"{MatrixColors.GREEN_DIM}> PFE v2.0 // Matrix Terminal Interface{MatrixColors.RESET}")

    return "\n".join(lines)


def format_train_result_matrix(result: Any, *, workspace: str | None = None) -> str:
    """Format train result in Matrix Green terminal style."""
    lines = []

    lines.append(draw_header("TRAINING COMPLETE"))

    mapping = _coerce_mapping(result)
    if mapping is None:
        lines.append(f"{MatrixColors.RED}ERROR: Unable to parse training result{MatrixColors.RESET}")
        return "\n".join(lines)

    content = []

    version = mapping.get("version", "n/a")
    content.append(format_key_value("version", version))

    adapter_path = mapping.get("adapter_path", "n/a")
    content.append(format_key_value("path", f"{MatrixColors.GRAY}{adapter_path}{MatrixColors.RESET}"))

    num_samples = mapping.get("num_samples", 0)
    content.append(format_key_value("samples", num_samples))

    # Backend info
    backend_plan = _coerce_mapping(mapping.get("backend_plan"))
    if backend_plan:
        backend = backend_plan.get("selected_backend", "unknown")
        device = backend_plan.get("runtime_device", "unknown")
        content.append(format_key_value("backend", f"{backend} | device={device}"))

    # Export info
    export_runtime = _coerce_mapping(mapping.get("export_runtime"))
    if export_runtime:
        required = export_runtime.get("required", False)
        format_type = export_runtime.get("target_artifact_format", "unknown")
        if required:
            content.append(format_key_value("export", f"{MatrixColors.AMBER}REQUIRED{MatrixColors.RESET} | format={format_type}"))
        else:
            content.append(format_key_value("export", f"{MatrixColors.GREEN}NOT REQUIRED{MatrixColors.RESET}"))

    # Metrics
    metrics = _coerce_mapping(mapping.get("metrics"))
    if metrics:
        fresh = metrics.get("num_fresh_samples", 0)
        replay = metrics.get("num_replay_samples", 0)
        content.append(format_key_value("metrics", f"fresh={fresh} | replay={replay}"))

    lines.append(draw_box("TRAINING RESULT", content))
    lines.append("")

    # Success message
    lines.append(f"{MatrixColors.GREEN}    [✓] Training job completed successfully{MatrixColors.RESET}")
    lines.append("")

    return "\n".join(lines)


def format_serve_preview_matrix(
    port: int,
    host: str,
    adapter: str,
    workspace: str | None,
    api_key: str | None,
    real_local: bool,
    recent_training: dict[str, Any] | None = None,
    latest_training: dict[str, Any] | None = None,
) -> str:
    """Format serve preview in Matrix Green terminal style."""
    lines = []

    lines.append(draw_header("SERVE PREVIEW"))

    content = []
    content.append(format_key_value("host", host))
    content.append(format_key_value("port", port))
    content.append(format_key_value("adapter", adapter))
    content.append(format_key_value("workspace", workspace or "default"))
    content.append(format_key_value("api_key", f"{MatrixColors.GREEN}SET{MatrixColors.RESET}" if api_key else f"{MatrixColors.GRAY}UNSET{MatrixColors.RESET}"))
    content.append(format_key_value("mode", f"{MatrixColors.GREEN}REAL{MatrixColors.RESET}" if real_local else f"{MatrixColors.GRAY}MOCK{MatrixColors.RESET}"))

    lines.append(draw_box("SERVER CONFIGURATION", content))
    lines.append("")

    if latest_training:
        lt_content = []
        version = latest_training.get("version")
        state = latest_training.get("state")
        if version is not None:
            lt_content.append(format_key_value("version", version))
        if state is not None:
            lt_content.append(format_key_value("state", state))
        if lt_content:
            lines.append(draw_box("LATEST PROMOTED", lt_content))
            lines.append("")

    if recent_training:
        rt_content = []
        version = recent_training.get("version")
        state = recent_training.get("state")
        if version is not None:
            rt_content.append(format_key_value("version", version))
        if state is not None:
            rt_content.append(format_key_value("state", state))
        execution_backend = recent_training.get("execution_backend")
        if execution_backend is not None:
            rt_content.append(format_key_value("execution backend", execution_backend))
        executor_mode = recent_training.get("executor_mode")
        if executor_mode is not None:
            rt_content.append(format_key_value("executor mode", executor_mode))
        job_execution = _coerce_mapping(recent_training.get("real_execution_summary") or recent_training.get("job_execution"))
        if job_execution:
            for key in ("status", "state", "executor_mode", "execution_mode", "attempted", "success", "runner_status", "kind"):
                value = job_execution.get(key)
                if value is not None:
                    rt_content.append(format_key_value(key.replace("_", " "), value))
        export_execution = _coerce_mapping(recent_training.get("export_execution") or recent_training.get("export_toolchain_summary"))
        if export_execution:
            for key in ("status", "state", "execution_mode", "attempted", "success"):
                value = export_execution.get(key)
                if value is not None:
                    rt_content.append(format_key_value(key.replace("_", " "), value))
        if rt_content:
            lines.append(draw_box("RECENT TRAINING", rt_content))
            lines.append("")

    return "\n".join(lines)


def format_serve_matrix(result: Any) -> str:
    """Format serve result in Matrix Green terminal style."""
    mapping = _coerce_mapping(result)
    if mapping and "ready_message" in mapping:
        return f"{MatrixColors.GREEN}    [✓] {mapping['ready_message']}{MatrixColors.RESET}"
    return _format_scalar(result)


def format_eval_result_matrix(result: Any, *, workspace: str | None = None) -> str:
    """Format eval result in Matrix Green terminal style."""
    lines = []

    lines.append(draw_header("EVALUATION RESULT"))

    mapping = _coerce_mapping(result)
    if mapping is None:
        try:
            # Try parsing as JSON string
            mapping = json.loads(result)
        except Exception:
            lines.append(f"{MatrixColors.RED}ERROR: Unable to parse evaluation result{MatrixColors.RESET}")
            return "\n".join(lines)

    content = []

    adapter_version = mapping.get("adapter_version", "n/a")
    content.append(format_key_value("adapter", adapter_version))

    base_model = mapping.get("base_model", "n/a")
    content.append(format_key_value("base model", base_model))

    num_samples = mapping.get("num_test_samples", 0)
    content.append(format_key_value("test samples", num_samples))

    recommendation = mapping.get("recommendation", "unknown")
    comparison = mapping.get("comparison", "unknown")

    rec_color = MatrixColors.GREEN if recommendation == "deploy" else MatrixColors.AMBER if recommendation == "review" else MatrixColors.RED
    content.append(format_key_value("recommendation", f"{rec_color}{recommendation.upper()}{MatrixColors.RESET}"))
    content.append(format_key_value("comparison", comparison))

    # Scores
    scores = _coerce_mapping(mapping.get("scores"))
    if scores:
        content.append("")
        content.append(f"{MatrixColors.GREEN_BRIGHT}SCORES:{MatrixColors.RESET}")
        for key, value in _ordered_eval_scores(scores):
            bar_width = int(value * 20)  # Scale to 20 chars
            bar = "█" * bar_width + "░" * (20 - bar_width)
            content.append(f"  {key:25} [{MatrixColors.GREEN}{bar}{MatrixColors.RESET}] {value:.2f}")

    lines.append(draw_box("EVALUATION METRICS", content))
    lines.append("")

    return "\n".join(lines)


def format_adapter_list_matrix(versions: list[dict[str, Any]], *, limit: int = 10) -> str:
    """Format adapter list in Matrix Green terminal style."""
    lines = []

    lines.append(draw_header("ADAPTER VERSIONS"))

    if not versions:
        lines.append(f"{MatrixColors.GRAY}    No adapters found{MatrixColors.RESET}")
        return "\n".join(lines)

    # Find latest
    latest_version = None
    for v in versions:
        if v.get("latest") or v.get("state") == "promoted":
            latest_version = v.get("version")
            break

    if latest_version:
        lines.append(f"{MatrixColors.GREEN}    CURRENT LATEST: {latest_version}{MatrixColors.RESET}")
        lines.append("")

    # Table data
    headers = ["VERSION", "STATE", "SAMPLES", "FORMAT"]
    rows = []

    for v in versions[:limit]:
        version = v.get("version", "n/a")
        state = v.get("state", "unknown")
        samples = str(v.get("num_training_samples", 0))
        fmt = v.get("artifact_format", "unknown")

        # Highlight latest
        if version == latest_version:
            version = f"{MatrixColors.GREEN}*{MatrixColors.RESET} {version}"

        rows.append([version, state, samples, fmt])

    lines.append(draw_table(headers, rows))
    lines.append("")

    lines.append(f"{MatrixColors.GREEN_DIM}    Showing {min(limit, len(versions))} of {len(versions)} versions{MatrixColors.RESET}")

    return "\n".join(lines)


__all__ = [
    "format_status_matrix",
    "format_train_result_matrix",
    "format_serve_preview_matrix",
    "format_serve_matrix",
    "format_eval_result_matrix",
    "format_adapter_list_matrix",
]
