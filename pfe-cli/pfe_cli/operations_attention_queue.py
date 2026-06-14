"""Queue and runner attention fragments for operations attention formatting."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .operations_attention_context import OperationsAttentionContext


def append_train_queue_attention(
    alerts: list[str],
    *,
    train_queue: Mapping[str, Any] | None,
    context: OperationsAttentionContext,
    deps: Any,
) -> None:
    if train_queue is None:
        return
    _append_queue_counts(alerts, train_queue=train_queue, context=context, deps=deps)
    _append_confirmation_summary(alerts, train_queue=train_queue, deps=deps)
    _append_worker_runner(alerts, train_queue=train_queue, context=context, deps=deps)


def _append_queue_counts(
    alerts: list[str],
    *,
    train_queue: Mapping[str, Any],
    context: OperationsAttentionContext,
    deps: Any,
) -> None:
    counts = deps.coerce_mapping(train_queue.get("counts")) or {}
    queued_count = counts.get("queued")
    if not queued_count or (
        context.monitor_alert_emitted and str(context.resolved_focus).strip().lower().startswith("queue")
    ):
        return

    current_item = deps.coerce_mapping(train_queue.get("current"))
    if current_item is not None and current_item.get("state") == "awaiting_confirmation":
        queue_parts = [f"awaiting_confirmation={deps.format_scalar(queued_count)}"]
    else:
        queue_parts = [f"queued={deps.format_scalar(queued_count)}"]
    alerts.append("queue " + " | ".join(queue_parts))


def _append_confirmation_summary(alerts: list[str], *, train_queue: Mapping[str, Any], deps: Any) -> None:
    confirmation_summary = deps.coerce_mapping(train_queue.get("confirmation_summary"))
    if confirmation_summary is None:
        return
    awaiting_confirmation_count = confirmation_summary.get("awaiting_confirmation_count")
    if not awaiting_confirmation_count:
        return
    next_job_id = confirmation_summary.get("next_job_id")
    queue_parts = [f"awaiting_confirmation={deps.format_scalar(awaiting_confirmation_count)}"]
    if next_job_id is not None:
        queue_parts.append(f"next_job_id={deps.format_scalar(next_job_id)}")
    alerts.append("confirmation " + " | ".join(queue_parts))


def _append_worker_runner(
    alerts: list[str],
    *,
    train_queue: Mapping[str, Any],
    context: OperationsAttentionContext,
    deps: Any,
) -> None:
    worker_runner = deps.coerce_mapping(train_queue.get("worker_runner"))
    if worker_runner is None:
        return
    lock_state = worker_runner.get("lock_state")
    active = worker_runner.get("active")
    stop_requested = worker_runner.get("stop_requested")
    if not (lock_state in {"active", "stale"} or active or stop_requested) or (
        context.monitor_alert_emitted
        and str(context.resolved_focus).strip().lower().startswith(("runner", "daemon"))
    ):
        return

    runner_parts: list[str] = []
    if lock_state is not None:
        runner_parts.append(f"lock_state={deps.format_scalar(lock_state)}")
    if active is not None:
        runner_parts.append(f"active={deps.format_scalar(active)}")
    if stop_requested is not None:
        runner_parts.append(f"stop_requested={deps.format_scalar(stop_requested)}")
    lease_expires_at = worker_runner.get("lease_expires_at")
    if lease_expires_at is not None:
        runner_parts.append(f"lease_expires_at={deps.format_scalar(lease_expires_at)}")
    alerts.append("worker runner " + " | ".join(runner_parts))


__all__ = ["append_train_queue_attention"]
