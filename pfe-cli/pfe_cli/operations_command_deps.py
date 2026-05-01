"""Shared dependencies and helpers for operations CLI commands."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class OperationsCommandDeps:
    """Runtime hooks supplied by the main CLI module."""

    load_service: Callable[..., Any | None]
    run_placeholder: Callable[[str], None]
    resolve_handler: Callable[..., Any | None]
    run_handler: Callable[..., None]
    run_handler_json: Callable[..., None]
    friendly_exception_message: Callable[[Exception], str | None]
    coerce_mapping: Callable[[Any], dict[str, Any] | None]
    format_scalar: Callable[[Any], str]
    format_status: Callable[..., str]
    format_worker_runner_history: Callable[[Any], str]
    format_train_queue_history: Callable[[Any], str]
    format_train_queue_daemon_status: Callable[[Any], str]
    format_train_queue_daemon_history: Callable[[Any], str]
    format_daemon_health_status: Callable[[Any], str]
    format_daemon_heartbeat_status: Callable[[Any], str]
    format_daemon_lease_status: Callable[[Any], str]
    format_daemon_stale_check: Callable[[Any], str]
    format_daemon_alerts: Callable[[Any], str]
    format_candidate_history: Callable[[Any], str]
    format_candidate_timeline: Callable[[Any], str]
    read_train_queue_daemon_state: Callable[..., dict[str, Any] | None]
    update_train_queue_daemon_state: Callable[..., dict[str, Any]]
    daemon_recovery_payload: Callable[..., dict[str, Any]]


def pipeline_service(deps: OperationsCommandDeps) -> Any | None:
    return deps.load_service("pfe_core.pipeline", "pfe_core.services.pipeline")


def status_formatter(deps: OperationsCommandDeps, workspace: str | None) -> Callable[[Any], str]:
    return lambda result: deps.format_status(result, workspace=workspace)


def run_simple_status_command(
    deps: OperationsCommandDeps,
    *,
    command_name: str,
    handler_name: str,
    workspace: str | None,
    **kwargs: Any,
) -> None:
    service = pipeline_service(deps)
    if service is None:
        deps.run_placeholder(command_name)
        return

    handler = deps.resolve_handler(service, handler_name)
    if handler is None:
        deps.run_placeholder(command_name)
        return

    deps.run_handler(
        command_name,
        handler,
        formatter=status_formatter(deps, workspace),
        workspace=workspace,
        **kwargs,
    )


def handler_accepts_note(handler: Any) -> bool:
    try:
        signature = inspect.signature(handler)
    except (TypeError, ValueError):
        return False
    return "note" in signature.parameters
