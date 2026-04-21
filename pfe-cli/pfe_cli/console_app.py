"""Rich-based console surfaces for PFE operations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

from rich.box import HEAVY
from rich.console import Console, Group, RenderableType
from rich.layout import Layout
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, dict):
            return dict(dumped)
    to_dict = getattr(value, "dict", None)
    if callable(to_dict):
        dumped = to_dict()
        if isinstance(dumped, dict):
            return dict(dumped)
    return {}


def _sequence(value: Any) -> list[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    return []


def _yes_no(value: Any) -> str:
    if value is None:
        return "n/a"
    return "yes" if bool(value) else "no"


def _value(mapping: Mapping[str, Any], *keys: str, default: str = "n/a") -> str:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return str(mapping[key])
    return default


def _summary_field(summary_line: Any, field: str) -> str:
    text = str(summary_line or "")
    prefix = f"{field}="
    for part in text.split("|"):
        token = part.strip()
        if token.startswith(prefix):
            return token[len(prefix) :].strip()
    return ""


def _resolved_queue_review_policy(
    *,
    console: Mapping[str, Any],
    overview: Mapping[str, Any],
    train_queue: Mapping[str, Any],
    trigger_policy: Mapping[str, Any],
    trigger_blocked_reason: Any,
    trigger_blocked_action: Any,
) -> dict[str, Any]:
    queue_review_policy = (
        _mapping(console.get("queue_review_policy"))
        or _mapping(overview.get("queue_review_policy"))
        or _mapping(_mapping(train_queue).get("review_policy_summary"))
    )
    if queue_review_policy:
        return queue_review_policy

    review_mode = (
        _summary_field(overview.get("trigger_policy_summary"), "review")
        or _value(trigger_policy, "review_mode", default="")
    )
    queue_entry_mode = _value(trigger_policy, "queue_entry_mode", default="")
    next_action = ""
    blocked_reason = str(trigger_blocked_reason or "").strip().lower()
    blocked_action_text = str(trigger_blocked_action or "").strip()

    if blocked_reason == "queue_pending_review":
        next_action = blocked_action_text or "review_queue_confirmation"
    elif blocked_reason == "queue_waiting_execution":
        next_action = blocked_action_text or "process_next_queue_item"
    elif blocked_reason:
        next_action = blocked_action_text or blocked_reason
    elif queue_entry_mode == "awaiting_confirmation":
        next_action = "review_queue_confirmation"
    else:
        next_action = "await_signal_trigger"

    if not review_mode:
        review_mode = "manual_review" if queue_entry_mode == "awaiting_confirmation" else "auto_queue"
    if not queue_entry_mode:
        queue_entry_mode = "awaiting_confirmation" if review_mode == "manual_review" else "inline_execute"

    return {
        "review_mode": review_mode,
        "queue_entry_mode": queue_entry_mode,
        "next_action": next_action,
    }


def _compact_text(value: Any, *, max_len: int = 28) -> str:
    text = str(value or "").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


_GENERIC_MONITOR_FOCUSES = {
    "candidate_idle",
    "queue_waiting_execution",
    "runner_active",
    "daemon_active",
    "candidate_monitoring",
    "queue_monitoring",
    "runner_monitoring",
    "daemon_monitoring",
}


def _prefer_inspection_summary_for_generic_monitor(
    *,
    focus: Any,
    summary_source: str,
    inspection_summary: str,
) -> str:
    focus_text = str(focus or "").strip().lower()
    if inspection_summary and focus_text in _GENERIC_MONITOR_FOCUSES:
        return inspection_summary
    return summary_source


def _display_focus_name(focus: Any) -> str:
    normalized = str(focus or "").strip().lower()
    if normalized == "queue_backlog":
        return "queue_waiting_execution"
    return str(focus or "").strip()


def _dashboard_focus(dashboard: Mapping[str, Any] | None) -> str:
    dashboard_map = _mapping(dashboard)
    current_focus = str(dashboard_map.get("current_focus") or "").strip()
    if current_focus.lower() not in {"", "none", "idle", "stable"}:
        return _display_focus_name(current_focus)
    monitor_focus = str(dashboard_map.get("monitor_focus") or "").strip()
    if monitor_focus:
        return _display_focus_name(monitor_focus)
    return _display_focus_name(current_focus or "none")


def _payload_focus(payload: Mapping[str, Any] | None = None) -> str:
    payload_map = _mapping(payload)
    dashboard_focus = _dashboard_focus(payload_map.get("operations_dashboard"))
    if str(dashboard_focus).strip().lower() not in {"", "none", "idle", "stable"}:
        return dashboard_focus
    for raw_focus in (
        _mapping(payload_map.get("operations_console")).get("monitor_focus"),
        _mapping(payload_map.get("operations_overview")).get("monitor_focus"),
    ):
        focus_text = str(raw_focus or "").strip()
        if focus_text:
            return _display_focus_name(focus_text)
    return dashboard_focus


def _timestamp_now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _status_header(payload: Mapping[str, Any], *, workspace: str | None = None) -> RenderableType:
    latest = _mapping(payload.get("latest_adapter"))
    dashboard = _mapping(payload.get("operations_dashboard"))
    metadata = _mapping(payload.get("metadata"))
    inference = _mapping(metadata.get("inference"))
    plans = _mapping(payload.get("plans"))
    inference_plan = _mapping(plans.get("inference"))
    mode = "strict_local" if payload.get("strict_local", True) else _value(payload, "mode", default="unknown")
    inference_backend = _value(
        payload,
        "inference_backend",
        default=_value(inference, "selected_backend", "provider", default=_value(inference_plan, "selected_backend", default="unknown")),
    )
    workspace_name = str(workspace or payload.get("workspace") or "user_default")
    latest_value = latest.get("version") and f"{latest.get('version')} ({latest.get('state', 'unknown')})" or "none"
    severity = _value(dashboard, "severity", default="stable")
    line_one = Text()
    line_one.append("ws=", style="bold")
    line_one.append(workspace_name, style="white")
    line_one.append(" · ", style="dim")
    line_one.append("md=", style="bold")
    line_one.append(mode, style="white")
    line_one.append(" · ", style="dim")
    line_one.append("infer=", style="bold")
    line_one.append(inference_backend, style="white")

    line_two = Text()
    line_two.append("lat=", style="bold")
    line_two.append(latest_value, style="white")
    line_two.append(" · ", style="dim")
    line_two.append("sev=", style="bold")
    line_two.append(severity, style="white")
    line_two.append(" ", style="dim")
    line_two.append_text(_severity_badge(severity))
    line_two.append(" · ", style="dim")
    line_two.append("upd=", style="bold")
    line_two.append(datetime.now().strftime("%H:%M:%S"), style="white")
    return Panel(Group(line_one, line_two), title="PFE Console", border_style="cyan")


def _operations_panel(payload: Mapping[str, Any]) -> RenderableType:
    dashboard = _mapping(payload.get("operations_dashboard"))
    alert_policy = _mapping(payload.get("operations_alert_policy"))
    overview = _mapping(payload.get("operations_overview"))
    console = _mapping(payload.get("operations_console"))
    trigger = _mapping(payload.get("auto_train_trigger"))
    trigger_policy = _mapping(console.get("auto_train_policy")) or _mapping(overview.get("auto_train_policy")) or _mapping(trigger.get("policy"))
    trigger_policy_gate = _mapping(trigger_policy.get("gate_summary")) or _mapping(console.get("trigger_policy_gate_summary"))
    trigger_threshold = _mapping(console.get("trigger_threshold_summary")) or _mapping(overview.get("trigger_threshold_summary")) or _mapping(trigger.get("threshold_summary"))
    runtime_stability = _mapping(console.get("runtime_stability_summary")) or _mapping(overview.get("runtime_stability_summary"))
    train_queue = _mapping(payload.get("train_queue"))
    trigger_blocked_reason = _value(
        console,
        "trigger_blocked_reason",
        default=_value(overview, "trigger_blocked_reason", default="ready"),
    )
    trigger_blocked_action = _value(
        console,
        "trigger_blocked_action",
        default=_value(overview, "trigger_blocked_action", default="none"),
    )
    trigger_blocked_category = _value(
        console,
        "trigger_blocked_category",
        default=_value(overview, "trigger_blocked_category", default="n/a"),
    )
    queue_review_policy = _resolved_queue_review_policy(
        console=console,
        overview=overview,
        train_queue=train_queue,
        trigger_policy=trigger_policy,
        trigger_blocked_reason=trigger_blocked_reason,
        trigger_blocked_action=trigger_blocked_action,
    )
    severity = _value(dashboard, "severity", default="stable")
    focus_value = _dashboard_focus(dashboard)
    action_value = _value(alert_policy, "required_action", "primary_action", default="observe_and_monitor")
    priority_value = _value(alert_policy, "action_priority", default="p2")
    table = Table.grid(padding=(0, 1))
    table.add_column(style="bold")
    table.add_column()
    focus_text = Text(f"{_compact_text(focus_value, max_len=20)} ")
    focus_text.append_text(_focus_badge(focus_value, severity=severity))
    action_text = Text(f"{_compact_text(action_value, max_len=20)} ")
    action_text.append_text(_action_badge(action_value, priority=priority_value))
    handling_mode = _value(alert_policy, "escalation_mode", "remediation_mode", default="monitor")
    handling_bits = [
        priority_value,
        handling_mode,
        f"h={_yes_no(alert_policy.get('requires_human_review', False))}",
        f"a={_yes_no(alert_policy.get('auto_remediation_allowed', False))}",
        f"now={_yes_no(alert_policy.get('requires_immediate_action', False))}",
    ]
    policy_bits = [
        _value(trigger_policy, "queue_entry_mode", default="disabled"),
        _value(trigger_policy, "evaluation_mode", default="skip"),
        _value(trigger_policy, "promotion_mode", default="manual"),
        f"QRev={_value(queue_review_policy, 'review_mode', default='auto_queue')}",
        f"QNext={_compact_text(_value(queue_review_policy, 'next_action', default='await_signal_trigger'), max_len=22)}",
    ]
    policy_gate_bits = [
        f"e={_value(trigger_policy_gate, 'eval_num_samples', default='0')}:{'on' if bool(trigger_policy_gate.get('auto_evaluate_enabled')) else 'off'}",
        f"p={'on' if bool(trigger_policy_gate.get('auto_promote_requested')) else 'off'}",
        _compact_text(_value(trigger_policy_gate, 'promotion_requirement', default='manual'), max_len=10),
    ]
    gate_bits = [
        f"s={_value(trigger_threshold, 'eligible_signal_train_samples', default='0')}/{_value(trigger_threshold, 'min_new_samples', default='0')}",
        f"e={_value(trigger_threshold, 'effective_eligible_train_samples', default='0')}/{_value(trigger_threshold, 'min_new_samples', default='0')}",
        f"r={_value(trigger_threshold, 'preference_reinforced_train_samples', default='0')}",
        f"h={_yes_no(trigger_threshold.get('holdout_ready')) if 'holdout_ready' in trigger_threshold else 'n/a'}",
        f"i={_yes_no(trigger_threshold.get('interval_elapsed')) if 'interval_elapsed' in trigger_threshold else 'n/a'}",
        f"cd={_yes_no(trigger_threshold.get('cooldown_elapsed')) if 'cooldown_elapsed' in trigger_threshold else 'n/a'}",
        f"bo={_yes_no(trigger_threshold.get('failure_backoff_elapsed')) if 'failure_backoff_elapsed' in trigger_threshold else 'n/a'}",
    ]
    trigger_text = Text()
    trigger_text.append_text(_trigger_category_badge(trigger_blocked_category))
    trigger_text.append(" | ", style="dim")
    trigger_text.append(_compact_text(trigger_blocked_reason, max_len=18), style="white")
    trigger_text.append(" | ", style="dim")
    trigger_text.append(_compact_text(trigger_blocked_action, max_len=22), style="white")
    review_bits = [
        _value(queue_review_policy, "review_mode", default="auto_queue"),
        _value(queue_review_policy, "queue_entry_mode", default="inline_execute"),
        _value(queue_review_policy, "next_action", default="await_signal_trigger"),
    ]
    stability_text = _runtime_stability_text(runtime_stability, severity=severity)
    normalized_focus = str(focus_value or "").strip().lower()
    summary_source = _value(overview, "summary_line", default="idle")
    summary_source = _prefer_inspection_summary_for_generic_monitor(
        focus=normalized_focus,
        summary_source=summary_source,
        inspection_summary=_value(overview, "inspection_summary_line", default=""),
    )
    table.add_row("F", focus_text)
    table.add_row("A", action_text)
    table.add_row("Pol", " | ".join(policy_bits))
    table.add_row("PGate", " | ".join(policy_gate_bits))
    table.add_row("Gate", " | ".join(gate_bits))
    table.add_row("Trig", trigger_text)
    table.add_row("QRev", " | ".join(review_bits))
    table.add_row("Stab", stability_text)
    table.add_row("Handle", _handle_text(handling_bits, alert_policy))
    table.add_row("Sum", _compact_text(summary_source, max_len=30))
    guidance_source = _value(alert_policy, "operator_guidance", default="observe the system")
    if normalized_focus in {
        "insufficient_new_signal_samples",
        "holdout_not_ready",
        "cooldown_active",
        "failure_backoff_active",
    }:
        guidance_source = _value(trigger_threshold, "summary_line", default=guidance_source)
    guidance = _compact_text(guidance_source, max_len=20)
    primary_cmd, secondary_cmd = _payload_command_guidance(payload, focus_value)
    do_text = Text()
    do_text.append_text(_action_badge(action_value, priority=priority_value))
    do_text.append(" ", style="dim")
    do_text.append(_compact_text(primary_cmd, max_len=20), style="bold cyan")
    see_text = Text(_compact_text(secondary_cmd, max_len=22), style="dim")
    table.add_row("Do", do_text)
    table.add_row("See", see_text)
    table.add_row("G", guidance)
    return Panel(table, border_style="yellow")


def _event_stream_panel(payload: Mapping[str, Any]) -> RenderableType:
    stream = _mapping(payload.get("operations_event_stream"))
    dashboard = _mapping(stream.get("dashboard"))
    alert_policy = _mapping(payload.get("operations_alert_policy"))
    overview = _mapping(payload.get("operations_overview"))
    console = _mapping(payload.get("operations_console"))
    items = _sequence(stream.get("items"))
    stream_severity = _value(stream, "severity", default="stable")
    priority_value = _value(alert_policy, "action_priority", default="p2")
    action_value = _value(alert_policy, "required_action", "primary_action", default="observe_and_monitor")
    latest_source = _value(stream, "latest_source", default="queue")
    latest_reason = _value(stream, "latest_reason", "attention_reason", default="")
    trigger_blocked_category = _value(
        console,
        "trigger_blocked_category",
        default=_value(overview, "trigger_blocked_category", default=""),
    )
    table = Table.grid(padding=(0, 1))
    table.add_column(style="bold")
    table.add_column()
    severity_text = Text(f"{stream_severity} ")
    severity_text.append_text(_severity_badge(stream_severity))
    focus_value = _dashboard_focus(dashboard)
    if focus_value.lower() in {"", "none", "idle", "stable"}:
        focus_value = _display_focus_name(_value(dashboard, "attention_reason", default="none"))
    normalized_focus = str(focus_value or "").strip().lower()
    generic_monitor_focuses = {
        "candidate_idle",
        "queue_waiting_execution",
        "runner_active",
        "daemon_active",
        "candidate_monitoring",
        "queue_monitoring",
        "runner_monitoring",
        "daemon_monitoring",
    }
    focus_text = Text(f"{focus_value} ")
    focus_text.append_text(_focus_badge(focus_value, severity=stream_severity))
    trigger_focus_category = _trigger_category_for_reason(focus_value, fallback=trigger_blocked_category)
    if trigger_focus_category and latest_source in {"trigger", "ops", "queue"}:
        focus_text.append(" ", style="dim")
        focus_text.append_text(_trigger_category_badge(trigger_focus_category))
    table.add_row("Sev", severity_text)
    status_text = Text()
    status_text.append(_value(stream, "status", default="healthy"), style="white")
    status_text.append(" ", style="dim")
    status_text.append(f"attn={_yes_no(stream.get('attention_needed'))}", style="dim")
    if action_value not in {"", "observe_and_monitor", "none"}:
        status_text.append(" ", style="dim")
        status_text.append_text(_action_badge(action_value, priority=priority_value))
    source_text = Text()
    source_text.append(latest_source, style="white")
    source_text.append(" ", style="dim")
    source_text.append(f"alerts={_value(stream, 'alert_count', default='0')}", style="dim")
    trigger_source_category = _trigger_category_for_reason(focus_value or latest_reason, fallback=trigger_blocked_category)
    if trigger_source_category and latest_source in {"trigger", "ops", "queue"}:
        source_text.append(" ", style="dim")
        source_text.append_text(_trigger_category_badge(trigger_source_category))
    runtime_badges = _event_runtime_badges(source=latest_source, reason=focus_value or latest_reason)
    if runtime_badges:
        source_text.append(" ", style="dim")
        source_text.append_text(runtime_badges[0])
    table.add_row("St", status_text)
    table.add_row("Src", source_text)
    table.add_row("F", focus_text)
    inspection_summary = _value(
        stream,
        "inspection_summary_line",
        default=_value(
            dashboard,
            "inspection_summary_line",
            default=_value(overview, "inspection_summary_line", default=""),
        ),
    )
    if _prefer_inspection_summary_for_generic_monitor(
        focus=normalized_focus,
        summary_source="",
        inspection_summary=inspection_summary,
    ):
        if inspection_summary:
            table.add_row("I", _compact_text(inspection_summary, max_len=42))

    recent_lines: list[Text] = []
    for item in items[:4]:
        mapping = _mapping(item)
        source = _value(mapping, "source", default="ops")
        event = _value(mapping, "event", default="none")
        severity = _value(mapping, "severity", default="stable")
        reason = _value(mapping, "reason", default="none")
        line = Text("- ", style="dim")
        line.append(_compact_text(source, max_len=8), style="bold")
        line.append(":", style="dim")
        line.append(_compact_text(event, max_len=10), style="white")
        line.append(" ", style="dim")
        line.append_text(_severity_badge(severity))
        line.append(" ", style="dim")
        line.append_text(_focus_badge(reason, severity=severity))
        trigger_item_category = _trigger_category_for_reason(
            reason,
            fallback=trigger_blocked_category if source in {"trigger", "ops", "queue"} else "",
        )
        if trigger_item_category and source in {"trigger", "ops", "queue"}:
            line.append(" ", style="dim")
            line.append_text(_trigger_category_badge(trigger_item_category))
        for badge in _event_runtime_badges(source=source, reason=reason):
            line.append(" ", style="dim")
            line.append_text(badge)
        recent_lines.append(line)
    if not recent_lines:
        recent_lines.append(Text("- none", style="dim"))
    content = Group(table, Text("R", style="bold dim"), *recent_lines)
    return Panel(content, border_style="magenta")


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


def _payload_command_guidance(payload: Mapping[str, Any], focus: str | None = None) -> tuple[str, str]:
    normalized = (focus or _payload_focus(payload) or "none").strip().lower()
    operations_dashboard = _mapping(payload.get("operations_dashboard"))
    operations_console = _mapping(payload.get("operations_console"))
    alert_policy = _mapping(payload.get("operations_alert_policy"))
    required_action = _value(alert_policy, "required_action", "primary_action", default="")
    secondary_action_values: list[str] = []
    for raw_action in [
        alert_policy.get("secondary_action"),
        *list(alert_policy.get("secondary_actions") or []),
    ]:
        text = str(raw_action or "").strip()
        if text and text not in secondary_action_values:
            secondary_action_values.append(text)
    required_guidance = _action_command_guidance(required_action)
    if required_guidance is not None:
        secondary_guidance_values: list[str] = []
        for secondary_action in secondary_action_values[:2]:
            secondary_guidance = _action_command_guidance(secondary_action)
            if secondary_guidance is None:
                continue
            label = str(secondary_guidance[0] or "").strip()
            if label and label not in secondary_guidance_values:
                secondary_guidance_values.append(label)
        if secondary_guidance_values:
            return (required_guidance[0], " ".join(secondary_guidance_values))
        return required_guidance

    def _summary_guidance(summary: Mapping[str, Any] | None) -> tuple[str, str] | None:
        summary_map = _mapping(summary)
        primary_action = _value(summary_map, "primary_action", default="")
        primary_guidance = _action_command_guidance(primary_action)
        if primary_guidance is None:
            return None
        secondary_guidance_values: list[str] = []
        for secondary_action in list(summary_map.get("secondary_actions") or [])[:2]:
            secondary_guidance = _action_command_guidance(str(secondary_action or ""))
            if secondary_guidance is None:
                continue
            label = str(secondary_guidance[0] or "").strip()
            if label and label not in secondary_guidance_values:
                secondary_guidance_values.append(label)
        if secondary_guidance_values:
            return (primary_guidance[0], " ".join(secondary_guidance_values))
        return primary_guidance

    if normalized.startswith("candidate"):
        candidate_summary_guidance = _summary_guidance(
            operations_dashboard.get("candidate_action_summary") or operations_console.get("candidate_action_summary")
        )
        if candidate_summary_guidance is not None:
            return candidate_summary_guidance
    if normalized.startswith("queue"):
        queue_summary_guidance = _summary_guidance(
            operations_dashboard.get("queue_action_summary") or operations_console.get("queue_action_summary")
        )
        if queue_summary_guidance is not None:
            return queue_summary_guidance
    if normalized.startswith("runner") or normalized.startswith("daemon"):
        runtime_summary_guidance = _summary_guidance(
            operations_dashboard.get("runtime_action_summary") or operations_console.get("runtime_action_summary")
        )
        runtime_summary = _mapping(
            operations_dashboard.get("runtime_action_summary") or operations_console.get("runtime_action_summary")
        )
        if (
            runtime_summary_guidance is not None
            and _value(runtime_summary, "primary_action", default="") == "inspect_runtime_stability"
        ):
            return runtime_summary_guidance

    candidate_summary = _mapping(payload.get("candidate_summary"))
    if normalized.startswith("candidate") and normalized != "candidate_ready_for_promotion":
        can_promote = bool(candidate_summary.get("candidate_can_promote"))
        can_archive = bool(candidate_summary.get("candidate_can_archive"))
        if can_promote:
            return ("/promote", "/candidate /cand sum")
        if can_archive:
            return ("/archive", "/candidate /cand sum")
    return _focus_command_guidance(focus)


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
        return "trigger" if normalized_focus in {
            "insufficient_new_signal_samples",
            "holdout_not_ready",
            "cooldown_active",
            "failure_backoff_active",
            "policy_requires_auto_evaluate",
        } else "gate"
    if normalized.startswith("/trigger"):
        return "trigger"
    if normalized.startswith("/retry"):
        return "retry"
    if normalized.startswith("/sum") or normalized.startswith("/status"):
        return "status"
    return None


def _chat_help_panel(payload: Mapping[str, Any], *, interactive: bool = False) -> RenderableType:
    latest = _mapping(payload.get("latest_adapter"))
    queue = _mapping(payload.get("train_queue"))
    dashboard = _mapping(payload.get("operations_dashboard"))
    console = _mapping(payload.get("operations_console"))
    overview = _mapping(payload.get("operations_overview"))
    focus = _payload_focus(payload)
    primary_cmd, secondary_cmd = _payload_command_guidance(payload, focus)
    normalized_focus = str(focus or "").strip().lower()
    trigger_threshold = _mapping(console.get("trigger_threshold_summary")) or _mapping(overview.get("trigger_threshold_summary"))
    trigger_blocked_summary = _value(
        console,
        "trigger_blocked_summary",
        default=_value(overview, "trigger_blocked_summary", default=""),
    )
    # Check if trigger threshold is met for showing trigger-train action
    trigger_ready = False
    min_samples = trigger_threshold.get("min_new_samples")
    eligible_samples = trigger_threshold.get("effective_eligible_train_samples")
    if eligible_samples is None:
        eligible_samples = trigger_threshold.get("eligible_signal_train_samples")
    if min_samples is not None and eligible_samples is not None:
        try:
            trigger_ready = float(eligible_samples) >= float(min_samples)
        except (ValueError, TypeError):
            trigger_ready = False
    header_parts = [f"lat={latest.get('version') or 'none'}", f"q={queue.get('count', 0)}"]
    if normalized_focus in {
        "insufficient_new_signal_samples",
        "holdout_not_ready",
        "cooldown_active",
        "failure_backoff_active",
    }:
        header_parts.append(
            "gate="
            + _compact_text(
                _value(trigger_threshold, "summary_line", default=trigger_blocked_summary or "trigger gate active"),
                max_len=28,
            )
        )
    elif trigger_ready and normalized_focus in {
        "policy_requires_auto_evaluate",
        "trigger_ready",
        "awaiting_signal_trigger",
    }:
        header_parts.append(f"ready={eligible_samples}/{min_samples}")
    lines = [" | ".join(header_parts), f"do: {primary_cmd}", f"see: {secondary_cmd}"]
    # Show trigger-train action when threshold is met
    if trigger_ready and normalized_focus in {
        "policy_requires_auto_evaluate",
        "trigger_ready",
        "awaiting_signal_trigger",
    }:
        lines.append("action: /do trigger-train")
    if interactive:
        lines.extend(
            [
                "chat=text | cmd=/do /see",
                "slash=/help /cmd /quit",
            ]
        )
    return Panel(Text("\n".join(lines)), border_style="green")


def _conversation_panel(session_messages: Sequence[Mapping[str, Any]] | None = None) -> RenderableType:
    items = list(session_messages or [])
    lines: list[str] = []
    for item in items[-8:]:
        role = str(item.get("role") or "system")
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        label = "user" if role == "user" else "assistant" if role == "assistant" else role
        lines.append(f"{label}> {content}")
    if not lines:
        lines.append("No conversation yet.")
    return Panel(Text("\n\n".join(lines)), title="Conversation", border_style="blue")


def _prompt_badge(label: str, style: str) -> Text:
    badge = Text()
    badge.append(" ")
    badge.append(label.upper(), style=style)
    badge.append(" ")
    return badge


def _ops_state_badge(state: str | None) -> Text:
    normalized = (state or "live").strip().lower()
    style = {
        "live": "bold white on dark_green",
        "cached": "bold black on bright_white",
        "syncing": "bold white on dark_blue",
    }.get(normalized, "bold white on dark_green")
    return _prompt_badge(normalized, style)


def _ops_badge(state: str | None, *, severity: str | None = None) -> Text:
    normalized = (state or "live").strip().lower()
    severity_name = (severity or "stable").strip().lower()
    if severity_name == "critical":
        return _prompt_badge(f"{normalized} !", "bold white on dark_red")
    if severity_name == "warning":
        return _prompt_badge(f"{normalized} !", "bold black on yellow")
    return _ops_state_badge(normalized)


def _focus_badge(focus: str | None, *, severity: str | None = None) -> Text:
    normalized = (_display_focus_name(focus) or "none").strip().lower()
    label = "focus" if normalized in {"none", "idle", "stable"} else normalized.replace("_", " ")
    severity_name = (severity or "stable").strip().lower()
    if severity_name == "critical":
        return _prompt_badge(label, "bold white on dark_red")
    if severity_name == "warning":
        return _prompt_badge(label, "bold black on yellow")
    return _prompt_badge(label, "bold black on bright_white")


def _action_badge(action: str | None, *, priority: str | None = None) -> Text:
    normalized = (action or "observe_and_monitor").strip().lower()
    label = "monitor" if normalized in {"observe_and_monitor", "none"} else normalized.replace("_", " ")
    priority_name = (priority or "p2").strip().lower()
    style = {
        "p0": "bold white on dark_red",
        "p1": "bold black on yellow",
        "p2": "bold white on dark_blue",
    }.get(priority_name, "bold black on bright_white")
    return _prompt_badge(label, style)


def _state_badge(label: str | None, *, style: str) -> Text:
    return _prompt_badge((label or "n/a").replace("_", " "), style)


def _runtime_state_badge(kind: str, value: str | None) -> Text:
    normalized = (value or "n/a").strip().lower()
    if kind == "runner":
        if normalized == "stale":
            return _state_badge(normalized, style="bold black on yellow")
        if normalized == "active":
            return _state_badge(normalized, style="bold white on dark_blue")
        return _state_badge(normalized, style="bold black on bright_white")
    if kind == "health":
        if normalized in {"stale", "blocked"}:
            return _state_badge(normalized, style="bold white on dark_red")
        if normalized in {"recovering"}:
            return _state_badge(normalized, style="bold white on dark_blue")
        return _state_badge(normalized, style="bold black on bright_white")
    if kind in {"heartbeat", "lease"}:
        if normalized in {"stale", "expired"}:
            return _state_badge(normalized, style="bold white on dark_red")
        if normalized in {"delayed", "expiring"}:
            return _state_badge(normalized, style="bold black on yellow")
        return _state_badge(normalized, style="bold black on bright_white")
    if kind == "restart":
        if normalized in {"backoff", "capped"}:
            return _state_badge(normalized, style="bold black on yellow")
        return _state_badge(normalized, style="bold black on bright_white")
    if kind == "recover":
        if normalized not in {"none", "n/a"}:
            return _state_badge(normalized, style="bold white on dark_blue")
        return _state_badge(normalized, style="bold black on bright_white")
    return _state_badge(normalized, style="bold black on bright_white")


def _trigger_category_badge(category: str | None) -> Text:
    normalized = (category or "n/a").strip().lower()
    style = {
        "data": "bold white on dark_blue",
        "timing": "bold black on yellow",
        "recovery": "bold white on dark_red",
        "queue": "bold black on bright_white",
        "config": "bold white on dark_magenta",
    }.get(normalized, "bold black on bright_white")
    return _state_badge(normalized, style=style)


def _trigger_category_for_reason(reason: str | None, *, fallback: str | None = None) -> str | None:
    normalized = (reason or "").strip().lower()
    if normalized in {"insufficient_new_signal_samples", "holdout_not_ready"}:
        return "data"
    if normalized in {"cooldown_active", "wait_for_retrain_interval"}:
        return "timing"
    if normalized in {"failure_backoff_active", "wait_for_failure_backoff"}:
        return "recovery"
    if normalized in {
        "queue_pending_review",
        "queue_waiting_execution",
        "queue_processing_active",
        "review_required_before_execution",
    }:
        return "queue"
    if normalized in {"policy_requires_auto_evaluate", "trigger_disabled"}:
        return "config"
    normalized_fallback = (fallback or "").strip().lower()
    return normalized_fallback or None


def _runtime_stability_text(runtime_stability: Mapping[str, Any], *, severity: str | None = None) -> Text:
    runner_state = _value(
        runtime_stability,
        "runner_lock_state",
        default=("active" if runtime_stability.get("runner_active") else "idle"),
    )
    health_state = _value(runtime_stability, "daemon_health_state", default="stopped")
    heartbeat_state = _value(runtime_stability, "daemon_heartbeat_state", default="idle")
    lease_state = _value(runtime_stability, "daemon_lease_state", default="idle")
    restart_state = _value(runtime_stability, "daemon_restart_policy_state", default="ready")
    recovery_state = _value(runtime_stability, "daemon_recovery_action", default="none")
    text = Text()
    text.append("R ", style="bold")
    text.append_text(_runtime_state_badge("runner", runner_state))
    text.append(" ", style="dim")
    text.append("D ", style="bold")
    text.append_text(_runtime_state_badge("health", health_state))
    if heartbeat_state not in {"idle", "fresh", "n/a"}:
        text.append(" hb ", style="dim")
        text.append_text(_runtime_state_badge("heartbeat", heartbeat_state))
    if lease_state not in {"idle", "valid", "n/a"}:
        text.append(" lease ", style="dim")
        text.append_text(_runtime_state_badge("lease", lease_state))
    if restart_state not in {"ready", "n/a"}:
        text.append(" rs ", style="dim")
        text.append_text(_runtime_state_badge("restart", restart_state))
    if recovery_state not in {"none", "n/a"}:
        text.append(" rec ", style="dim")
        text.append_text(_runtime_state_badge("recover", recovery_state))
    return text


def _handle_text(bits: Sequence[str], alert_policy: Mapping[str, Any]) -> Text:
    text = Text()
    if bits:
        text.append(_compact_text(bits[0], max_len=6), style="bold")
    if len(bits) > 1:
        text.append(" ", style="dim")
        text.append(_compact_text(bits[1], max_len=10), style="white")
    text.append(" ", style="dim")
    text.append(f"h={_yes_no(alert_policy.get('requires_human_review', False))}", style="dim")
    text.append(" ", style="dim")
    text.append(f"a={_yes_no(alert_policy.get('auto_remediation_allowed', False))}", style="dim")
    text.append(" ", style="dim")
    text.append(f"now={_yes_no(alert_policy.get('requires_immediate_action', False))}", style="dim")
    return text


def _runtime_focus_badges(*, focus: str | None, action: str | None, priority: str | None) -> list[Text]:
    normalized_focus = (focus or "").strip().lower()
    badges: list[Text] = []
    if not normalized_focus:
        return badges
    if normalized_focus in {"daemon_stale", "daemon_blocked"}:
        badges.append(_runtime_state_badge("health", normalized_focus.removeprefix("daemon_")))
    elif normalized_focus.startswith("daemon_restart_"):
        badges.append(_runtime_state_badge("restart", normalized_focus.removeprefix("daemon_restart_")))
    elif normalized_focus.startswith("daemon_heartbeat_"):
        badges.append(_runtime_state_badge("heartbeat", normalized_focus.removeprefix("daemon_heartbeat_")))
    elif normalized_focus.startswith("daemon_lease_"):
        badges.append(_runtime_state_badge("lease", normalized_focus.removeprefix("daemon_lease_")))
    elif normalized_focus == "stale_runner_lock":
        badges.append(_runtime_state_badge("runner", "stale"))
    elif normalized_focus == "runner_stop_requested":
        badges.append(_runtime_state_badge("runner", "active"))
    if action and action not in {"observe_and_monitor", "none"}:
        badges.append(_action_badge(action, priority=priority))
    return badges


def _event_runtime_badges(*, source: str | None, reason: str | None) -> list[Text]:
    normalized_source = (source or "").strip().lower()
    normalized_reason = (reason or "").strip().lower()
    badges: list[Text] = []
    if normalized_reason in {"daemon_stale", "daemon_blocked"}:
        state = "stale" if normalized_reason.endswith("stale") else "blocked"
        badges.append(_runtime_state_badge("health", state))
        badges.append(_action_badge("recover_worker_daemon", priority="p0"))
        return badges
    if normalized_reason.startswith("daemon_heartbeat_"):
        badges.append(_runtime_state_badge("heartbeat", normalized_reason.removeprefix("daemon_heartbeat_")))
        badges.append(_action_badge("inspect_daemon_heartbeat", priority="p1"))
        return badges
    if normalized_reason.startswith("daemon_lease_"):
        lease_state = normalized_reason.removeprefix("daemon_lease_")
        badges.append(_runtime_state_badge("lease", lease_state))
        badges.append(
            _action_badge(
                "recover_worker_daemon" if lease_state == "expired" else "inspect_daemon_heartbeat",
                priority="p1",
            )
        )
        return badges
    if normalized_reason.startswith("daemon_restart_"):
        badges.append(_runtime_state_badge("restart", normalized_reason.removeprefix("daemon_restart_")))
        badges.append(_action_badge("inspect_daemon_restart_policy", priority="p1"))
        return badges
    if normalized_reason == "stale_runner_lock":
        badges.append(_runtime_state_badge("runner", "stale"))
        badges.append(_action_badge("inspect_worker_stale_lock", priority="p1"))
        return badges
    if normalized_reason == "runner_stop_requested":
        badges.append(_runtime_state_badge("runner", "active"))
        badges.append(_action_badge("wait_for_runner_shutdown", priority="p2"))
        return badges
    if normalized_source == "daemon":
        if "restart" in normalized_reason:
            badges.append(_action_badge("inspect_daemon_restart_policy", priority="p1"))
        elif "heartbeat" in normalized_reason:
            badges.append(_action_badge("inspect_daemon_heartbeat", priority="p1"))
        elif "lease" in normalized_reason or "recover" in normalized_reason:
            badges.append(_action_badge("recover_worker_daemon", priority="p1"))
    return badges


def _severity_badge(severity: str | None) -> Text:
    normalized = (severity or "stable").strip().lower()
    style = {
        "critical": "bold white on dark_red",
        "warning": "bold black on yellow",
        "info": "bold white on dark_blue",
        "stable": "bold black on bright_white",
    }.get(normalized, "bold black on bright_white")
    return _prompt_badge(normalized, style)


def _section_label(title: str, *, badge: Text | None = None) -> Text:
    label = Text()
    label.append(title.upper(), style="bold bright_white")
    if badge is not None:
        label.append(" ", style="dim")
        label.append_text(badge)
    return label


def _prompt_context_focus(payload: Mapping[str, Any] | None = None) -> str:
    return _payload_focus(payload).strip().lower()


def _prompt_placeholder(mode: str, *, focus: str | None = None, payload: Mapping[str, Any] | None = None) -> str:
    normalized_focus = (focus or "none").strip().lower()
    if mode == "chat":
        return "Type message, /cmd, or /help"
    if payload is not None:
        primary_label, _secondary_label = _payload_command_guidance(payload, focus)
        payload_target = _prompt_action_token_from_label(primary_label, focus=focus)
        if payload_target:
            return "Type /do or /see"
    if normalized_focus.startswith("policy_") or normalized_focus.startswith("auto_train_policy"):
        return "Type /do or /see"
    if normalized_focus in {
        "insufficient_new_signal_samples",
        "holdout_not_ready",
        "cooldown_active",
        "failure_backoff_active",
    }:
        return "Type /do or /see"
    if "pending_review" in normalized_focus or "awaiting_confirmation" in normalized_focus:
        return "Type /do or /see"
    if normalized_focus.startswith("daemon") and "restart" in normalized_focus:
        return "Type /do or /see"
    if normalized_focus.startswith("daemon") and ("heartbeat" in normalized_focus or "lease" in normalized_focus):
        return "Type /do or /see"
    if normalized_focus == "daemon_active":
        return "Type /do or /see"
    if "runner" in normalized_focus and "stale" in normalized_focus:
        return "Type /do or /see"
    if normalized_focus.startswith("daemon"):
        return "Type /daemon or /daemon timeline"
    if normalized_focus.startswith("candidate"):
        return "Type /do or /see"
    if normalized_focus.startswith("queue"):
        return "Type /do or /see"
    if normalized_focus.startswith("runner"):
        return "Type /do or /see"
    return "Type /status or /ops dashboard"


def _prompt_mode_help(
    mode: str,
    *,
    focus: str | None = None,
    payload: Mapping[str, Any] | None = None,
) -> str:
    normalized_focus = (focus or "none").strip().lower()
    if mode == "chat":
        return "reply"
    if payload is not None:
        primary_label, _secondary_label = _payload_command_guidance(payload, focus)
        payload_hint = _prompt_action_token_from_label(primary_label, focus=focus)
        if payload_hint:
            return payload_hint
    if normalized_focus.startswith("policy_") or normalized_focus.startswith("auto_train_policy"):
        return "trigger"
    if normalized_focus in {
        "insufficient_new_signal_samples",
        "holdout_not_ready",
        "cooldown_active",
        "failure_backoff_active",
    }:
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
    return "inspect"


def _prompt_target_hint(
    mode: str,
    *,
    focus: str | None = None,
    payload: Mapping[str, Any] | None = None,
) -> str:
    normalized_focus = (focus or "none").strip().lower()
    if mode == "chat":
        return "send"
    if payload is not None:
        primary_label, _secondary_label = _payload_command_guidance(payload, focus)
        payload_target = _prompt_action_token_from_label(primary_label, focus=focus)
        if payload_target:
            return payload_target
    if normalized_focus.startswith("policy_") or normalized_focus.startswith("auto_train_policy"):
        return "trigger"
    if normalized_focus in {
        "insufficient_new_signal_samples",
        "holdout_not_ready",
        "cooldown_active",
        "failure_backoff_active",
    }:
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
    return "status"


def _prompt_feedback_digest(feedback: str | None = None) -> str:
    normalized = (feedback or "").strip()
    lower = normalized.lower()
    if not normalized:
        return "idle"
    if lower.startswith("assistant generating"):
        return "generating"
    if lower.startswith("running /"):
        return normalized.split(" ", 1)[1]
    if len(normalized) > 24:
        return normalized[:21] + "..."
    return normalized


def _prompt_ctx_digest(focus: str | None = None) -> str:
    normalized_focus = _display_focus_name(focus).strip().lower()
    if not normalized_focus or normalized_focus in {"none", "idle", "stable"}:
        return ""
    if len(normalized_focus) > 18:
        return normalized_focus[:15] + "..."
    return normalized_focus


def _prompt_hint_digest(shortcut_hint: str | None = None) -> str:
    normalized = (shortcut_hint or "").strip()
    if not normalized:
        return ""
    parts = [part.strip() for part in normalized.split(",") if part.strip()]
    if not parts:
        return ""
    compact = ",".join(parts[:2])
    if len(compact) > 18:
        return compact[:15] + "..."
    return compact


def _prompt_trigger_category(
    focus: str | None = None,
    *,
    payload: Mapping[str, Any] | None = None,
) -> str:
    overview = _mapping((payload or {}).get("operations_overview"))
    console = _mapping((payload or {}).get("operations_console"))
    fallback = _value(
        console,
        "trigger_blocked_category",
        default=_value(overview, "trigger_blocked_category", default=""),
    )
    return _trigger_category_for_reason(focus, fallback=fallback) or ""


def _prompt_action_guidance(
    mode: str,
    *,
    focus: str | None = None,
    shortcut_hint: str | None = None,
    payload: Mapping[str, Any] | None = None,
) -> tuple[str, str]:
    normalized_focus = (focus or "none").strip().lower()
    if mode == "chat":
        return "send", "/cmd"
    if payload is not None:
        return _payload_command_guidance(payload, focus)
    if normalized_focus and normalized_focus not in {"none", "idle", "stable"}:
        return _focus_command_guidance(normalized_focus)
    normalized = (shortcut_hint or "").strip()
    parts = [part.strip() for part in normalized.split(",") if part.strip()]
    primary = parts[0] if parts else "/status"
    secondary = parts[1] if len(parts) > 1 else "/help"
    return _compact_text(primary, max_len=18), _compact_text(secondary, max_len=18)


def _prompt_model_digest(value: str | None) -> str:
    text = (value or "local").strip()
    if len(text) > 12:
        return text[:9] + "..."
    return text


def _prompt_adapter_digest(value: str | None) -> str:
    text = (value or "latest").strip()
    if len(text) > 12:
        return text[:9] + "..."
    return text


def _prompt_state_badge(
    *,
    mode: str,
    input_active: bool,
    preview: str,
    feedback: str | None,
) -> Text:
    feedback_text = (feedback or "").strip().lower()
    if feedback_text.startswith("assistant generating"):
        return _prompt_badge("wait", "bold white on dark_blue")
    if input_active and preview:
        return _prompt_badge("compose", "bold black on yellow")
    if input_active:
        return _prompt_badge("ready", "bold white on dark_green" if mode == "chat" else "bold white on dark_blue")
    if mode == "command" and feedback_text.startswith("handled /"):
        return _prompt_badge("run", "bold white on dark_blue")
    return _prompt_badge("idle", "bold black on bright_white")


def _edit_state_badge(*, input_active: bool, raw_input: str, cursor_index: int) -> Text:
    if not input_active:
        return _prompt_badge("locked", "bold black on bright_white")
    if not raw_input:
        return _prompt_badge("blank", "bold white on dark_blue")
    if cursor_index != len(raw_input):
        return _prompt_badge("edit", "bold black on yellow")
    return _prompt_badge("typed", "bold white on dark_green")


def _activity_badge(*, mode: str, input_active: bool, feedback: str | None) -> Text:
    feedback_text = (feedback or "").strip().lower()
    if feedback_text.startswith("assistant generating"):
        return _prompt_badge("chat", "bold white on dark_blue")
    if feedback_text.startswith("running /") or (mode == "command" and feedback_text.startswith("handled /")):
        return _prompt_badge("exec", "bold white on dark_red")
    if input_active:
        return _prompt_badge("editable", "bold black on bright_white")
    return _prompt_badge("settled", "bold black on bright_white")


def _sidebar_snapshot_text(*, ops_refresh_state: str | None, ops_age_seconds: float | None, refresh_seconds: float | None) -> Text:
    line = Text()
    line.append("snap=", style="dim")
    line.append_text(_ops_badge(ops_refresh_state))
    if ops_age_seconds is not None:
        line.append(" · ", style="dim")
        line.append(f"age={ops_age_seconds:.1f}s", style="dim")
    if refresh_seconds is not None:
        line.append(" · ", style="dim")
        line.append(f"cadence={refresh_seconds:.1f}s", style="dim")
    return line


def _footer_digest(
    payload: Mapping[str, Any],
    *,
    interactive: bool = False,
    mode: str = "chat",
    ops_refresh_state: str | None = None,
    ops_age_seconds: float | None = None,
) -> Text:
    dashboard = _mapping(payload.get("operations_dashboard"))
    alert_policy = _mapping(payload.get("operations_alert_policy"))
    resolved_focus = _dashboard_focus(dashboard)
    focus = _compact_text(resolved_focus, max_len=16)
    action = _compact_text(_value(alert_policy, "required_action", "primary_action", default="observe_and_monitor"), max_len=22)
    full_focus = resolved_focus
    full_action = _value(alert_policy, "required_action", "primary_action", default="observe_and_monitor")
    action_priority = _value(alert_policy, "action_priority", default="p2")
    handling = _value(alert_policy, "remediation_mode", "escalation_mode", default="monitor")
    status_badge = _ops_badge(ops_refresh_state, severity=_value(dashboard, "severity", default="stable"))
    if not interactive:
        help_hint = "console,status,doctor"
    elif mode == "chat":
        help_hint = "Enter,/help,/cmd,/quit"
    else:
        help_hint = "/status,/candidate,/daemon,/chat"
    if "," in help_hint:
        help_hint = ",".join(part.strip() for part in help_hint.split(",")[:2])
    primary_action_hint, secondary_action_hint = _prompt_action_guidance(
        mode,
        focus=full_focus,
        shortcut_hint=help_hint,
        payload=payload,
    )
    runtime_badges = _runtime_focus_badges(focus=full_focus, action=full_action, priority=action_priority)
    runtime_mode = bool(runtime_badges)

    line = Text()
    line.append("f=", style="dim")
    if runtime_mode:
        line.append(full_focus, style="white")
        line.append(" ", style="dim")
        line.append_text(_focus_badge(full_focus, severity=_value(dashboard, "severity", default="stable")))
    else:
        line.append(_compact_text(focus, max_len=16), style="white")
        line.append(" ", style="dim")
        line.append_text(_focus_badge(focus, severity=_value(dashboard, "severity", default="stable")))
    line.append(" · ", style="dim")
    line.append("a=", style="dim")
    if runtime_mode:
        line.append(full_action, style="white")
        line.append(" ", style="dim")
        line.append_text(_action_badge(full_action, priority=_value(alert_policy, "action_priority", default="p2")))
    else:
        line.append(action, style="white")
        line.append(" ", style="dim")
        line.append_text(_action_badge(action, priority=_value(alert_policy, "action_priority", default="p2")))
    if runtime_mode:
        return line
    else:
        line.append(" · ", style="dim")
        line.append("h=", style="dim")
        line.append(handling, style="cyan")
        trigger_category = _prompt_trigger_category(full_focus, payload=payload)
        if trigger_category:
            line.append(" · ", style="dim")
            line.append("tg=", style="dim")
            line.append_text(_trigger_category_badge(trigger_category))
        for badge in runtime_badges[:2]:
            line.append(" · ", style="dim")
            line.append_text(badge)
        line.append(" · ", style="dim")
        line.append("o=", style="dim")
        line.append_text(status_badge)
        line.append(" · ", style="dim")
        line.append("d=", style="dim")
        line.append(primary_action_hint, style="bold cyan")
        line.append(" · ", style="dim")
        line.append("s=", style="dim")
        line.append(secondary_action_hint, style="dim")
        if ops_age_seconds is not None:
            line.append(" · ", style="dim")
            line.append(f"t={ops_age_seconds:.1f}s", style="dim")
    return line


def _prompt_panel(
    *,
    feedback: str | None = None,
    mode: str = "chat",
    prompt_label: str = "chat>",
    model: str | None = None,
    adapter: str | None = None,
    real_local: bool = False,
    refresh_seconds: float | None = None,
    input_active: bool = False,
    input_text: str | None = None,
    input_cursor: int | None = None,
    shortcut_hint: str | None = None,
    ops_refresh_state: str | None = None,
    ops_age_seconds: float | None = None,
    focus: str | None = None,
    payload: Mapping[str, Any] | None = None,
) -> RenderableType:
    preview = (input_text or "").strip()
    raw_input = str(input_text or "")
    cursor_index = max(0, min(int(input_cursor if input_cursor is not None else len(raw_input)), len(raw_input)))
    if len(preview) > 36:
        preview = preview[:33] + "..."
    if input_active and not preview:
        placeholder = _prompt_placeholder(mode, focus=focus, payload=payload)
    else:
        placeholder = ""
    recent_action = (feedback or "").strip()
    if len(recent_action) > 42:
        recent_action = recent_action[:39] + "..."

    bar = Text()
    bar.append("> ", style="bold cyan")
    bar.append(prompt_label, style="bold cyan")
    bar.append(" ")
    if input_active and raw_input:
        before = raw_input[:cursor_index]
        current = raw_input[cursor_index : cursor_index + 1] or " "
        after = raw_input[cursor_index + 1 :] if cursor_index < len(raw_input) else ""
        if before:
            bar.append(escape(before), style="white")
        bar.append(current, style="black on white")
        if after:
            bar.append(escape(after), style="white")
    elif preview:
        bar.append(escape(preview), style="white")
    elif placeholder:
        bar.append(placeholder, style="dim italic")
    else:
        bar.append("...", style="dim")
    bar.append("  ", style="dim")
    bar.append_text(_prompt_badge(mode, "bold white on dark_green" if mode == "chat" else "bold white on dark_blue"))
    bar.append(" ")
    bar.append_text(_prompt_badge("real-local", "bold white on dark_magenta") if real_local else _prompt_badge("local", "bold black on bright_white"))
    bar.append(" ")
    bar.append_text(
        _prompt_state_badge(
            mode=mode,
            input_active=input_active,
            preview=preview,
            feedback=feedback,
        )
    )
    bar.append(" ")
    bar.append_text(_edit_state_badge(input_active=input_active, raw_input=raw_input, cursor_index=cursor_index))
    bar.append(" ")
    bar.append_text(_activity_badge(mode=mode, input_active=input_active, feedback=feedback))
    bar.append("  ", style="dim")
    bar.append(f"m={_prompt_model_digest(model)}", style="dim")
    bar.append(" ", style="dim")
    bar.append(f"a={_prompt_adapter_digest(adapter)}", style="dim")
    if refresh_seconds is not None:
        bar.append(" ", style="dim")
        bar.append(f"r={refresh_seconds:.1f}s", style="dim")
    if input_active:
        bar.append(" ", style="dim")
        bar.append(f"c={cursor_index}", style="dim")
        bar.append("/", style="dim")
        bar.append(str(len(raw_input)), style="dim")

    effective_hint = shortcut_hint
    if effective_hint is None:
        effective_hint = "Enter,/help,^C" if mode == "chat" else "/status,/candidate,/daemon,/chat"

    bar.append(" ", style="dim")
    bar.append("o=", style="dim")
    bar.append(_prompt_target_hint(mode, focus=focus, payload=payload), style="bold cyan")

    ctx_digest = _prompt_ctx_digest(focus)
    if ctx_digest:
        bar.append(" ", style="dim")
        bar.append("x=", style="dim")
        bar.append(ctx_digest, style="bold white")

    trigger_category = _prompt_trigger_category(focus, payload=payload)
    if trigger_category:
        bar.append(" ", style="dim")
        bar.append("tg=", style="dim")
        bar.append_text(_trigger_category_badge(trigger_category))

    primary_action_hint, secondary_action_hint = _prompt_action_guidance(
        mode,
        focus=focus,
        shortcut_hint=effective_hint,
        payload=payload,
    )
    if primary_action_hint:
        bar.append(" ", style="dim")
        bar.append("d=", style="dim")
        bar.append(escape(primary_action_hint), style="bold cyan")
    if secondary_action_hint:
        bar.append(" ", style="dim")
        bar.append("s=", style="dim")
        bar.append(escape(secondary_action_hint), style="dim")

    recent_digest = _prompt_feedback_digest(recent_action)
    if recent_digest and recent_digest != "idle":
        bar.append(" ", style="dim")
        bar.append(f"l={escape(recent_digest)}", style="bold yellow")

    mode_help = _prompt_mode_help(mode, focus=focus, payload=payload)
    bar.append(" ", style="dim")
    bar.append("?=", style="dim")
    bar.append(mode_help, style="bold cyan")

    if ops_refresh_state:
        bar.append(" ", style="dim")
        bar.append("o=", style="dim")
        bar.append_text(_ops_badge(ops_refresh_state))
    alert_policy = _mapping((payload or {}).get("operations_alert_policy"))
    runtime_badges = _runtime_focus_badges(
        focus=focus,
        action=_value(alert_policy, "required_action", "primary_action", default="observe_and_monitor"),
        priority=_value(alert_policy, "action_priority", default="p2"),
    )
    for badge in runtime_badges[:2]:
        bar.append(" ", style="dim")
        bar.append_text(badge)
    return Panel(bar, border_style="white", box=HEAVY)


def build_console_renderable(
    payload: Mapping[str, Any],
    *,
    workspace: str | None = None,
    session_messages: Sequence[Mapping[str, Any]] | None = None,
    interactive: bool = False,
    feedback: str | None = None,
    mode: str = "chat",
    prompt_label: str = "chat>",
    model: str | None = None,
    adapter: str | None = None,
    real_local: bool = False,
    refresh_seconds: float | None = None,
    input_active: bool = False,
    input_text: str | None = None,
    input_cursor: int | None = None,
    shortcut_hint: str | None = None,
    ops_refresh_state: str | None = None,
    ops_age_seconds: float | None = None,
) -> RenderableType:
    header = _status_header(payload, workspace=workspace)
    dashboard = _mapping(payload.get("operations_dashboard"))
    severity = _value(dashboard, "severity", default="stable")
    focus = _prompt_context_focus(payload)
    sidebar_header = Text("Operations Sidebar ", style="bold")
    sidebar_header.append_text(_ops_badge(ops_refresh_state, severity=severity))
    if ops_age_seconds is not None:
        sidebar_header.append(f" {ops_age_seconds:.1f}s", style="dim")
    event_stream = _event_stream_panel(payload)
    event_stream_title = Text("Operations Event Stream ", style="bold")
    event_stream_title.append_text(_ops_badge(ops_refresh_state, severity=severity))
    event_stream.title = event_stream_title
    help_panel = _chat_help_panel(payload, interactive=interactive)
    help_title = Text("Help ", style="bold")
    help_title.append_text(_ops_state_badge(ops_refresh_state))
    help_panel.title = help_title
    sidebar = Group(
        _section_label("Snapshot", badge=_ops_badge(ops_refresh_state, severity=severity)),
        _sidebar_snapshot_text(
            ops_refresh_state=ops_refresh_state,
            ops_age_seconds=ops_age_seconds,
            refresh_seconds=refresh_seconds,
        ),
        _section_label("Operations", badge=_ops_badge(ops_refresh_state, severity=severity)),
        _operations_panel(payload),
        _section_label("Event Stream", badge=_severity_badge(stream_severity := _value(_mapping(payload.get("operations_event_stream")), "severity", default="stable"))),
        event_stream,
        _section_label("Help", badge=_prompt_badge("shortcuts", "bold black on bright_white")),
        help_panel,
    )
    body = Layout(name="body")
    body.split_row(
        Layout(_conversation_panel(session_messages), name="transcript", ratio=3),
        Layout(Panel(sidebar, title=sidebar_header, border_style="bright_black"), name="sidebar", size=54),
    )
    lower = _prompt_panel(
        feedback=feedback,
        mode=mode,
        prompt_label=prompt_label,
        model=model,
        adapter=adapter,
        real_local=real_local,
        refresh_seconds=refresh_seconds,
        input_active=input_active,
        input_text=input_text,
        input_cursor=input_cursor,
        shortcut_hint=shortcut_hint,
        ops_refresh_state=ops_refresh_state,
        ops_age_seconds=ops_age_seconds,
        focus=focus,
        payload=payload,
    )
    footer = _footer_digest(
        payload,
        interactive=interactive,
        mode=mode,
        ops_refresh_state=ops_refresh_state,
        ops_age_seconds=ops_age_seconds,
    )
    layout = Layout(name="root")
    layout.split_column(
        Layout(header, name="header", size=5),
        Layout(body, name="main", ratio=1),
        Layout(lower, name="prompt", size=3),
        Layout(footer, name="footer", size=1),
    )
    return layout


def render_console_snapshot(
    payload: Mapping[str, Any],
    *,
    workspace: str | None = None,
    console: Console | None = None,
    clear: bool = False,
    session_messages: Sequence[Mapping[str, Any]] | None = None,
    interactive: bool = False,
    feedback: str | None = None,
    mode: str = "chat",
    prompt_label: str = "chat>",
    model: str | None = None,
    adapter: str | None = None,
    real_local: bool = False,
    refresh_seconds: float | None = None,
    input_active: bool = False,
    input_text: str | None = None,
    input_cursor: int | None = None,
    shortcut_hint: str | None = None,
    ops_refresh_state: str | None = None,
    ops_age_seconds: float | None = None,
) -> None:
    target = console or Console()
    if clear:
        target.clear()
    target.print(
        build_console_renderable(
            payload,
            workspace=workspace,
            session_messages=session_messages,
            interactive=interactive,
            feedback=feedback,
            mode=mode,
            prompt_label=prompt_label,
            model=model,
            adapter=adapter,
            real_local=real_local,
            refresh_seconds=refresh_seconds,
            input_active=input_active,
            input_text=input_text,
            input_cursor=input_cursor,
            shortcut_hint=shortcut_hint,
            ops_refresh_state=ops_refresh_state,
            ops_age_seconds=ops_age_seconds,
        )
    )
