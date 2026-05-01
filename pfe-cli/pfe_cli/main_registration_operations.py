"""Operations command registration wiring."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

import typer

from .main_registration_common import call
from .operations_commands import OperationsCommandDeps, register_operations_commands


def register_main_operations_commands(
    *,
    trigger_app: typer.Typer,
    daemon_app: typer.Typer,
    candidate_app: typer.Typer,
    eval_trigger_app: typer.Typer,
    collect_app: typer.Typer,
    symbols: MutableMapping[str, Any],
) -> None:
    register_operations_commands(
        trigger_app=trigger_app,
        daemon_app=daemon_app,
        candidate_app=candidate_app,
        eval_trigger_app=eval_trigger_app,
        collect_app=collect_app,
        deps=OperationsCommandDeps(
            load_service=lambda *module_names: call(symbols, "_load_service", *module_names),
            run_placeholder=lambda command_name: call(symbols, "_run_placeholder", command_name),
            resolve_handler=lambda service, *names: call(symbols, "_resolve_handler", service, *names),
            run_handler=lambda command_name, handler, **kwargs: call(
                symbols,
                "_run_handler",
                command_name,
                handler,
                **kwargs,
            ),
            run_handler_json=lambda command_name, handler, **kwargs: call(
                symbols,
                "_run_handler_json",
                command_name,
                handler,
                **kwargs,
            ),
            friendly_exception_message=lambda exc: call(symbols, "_friendly_exception_message", exc),
            coerce_mapping=lambda result: call(symbols, "_coerce_mapping", result),
            format_scalar=lambda value: call(symbols, "_format_scalar", value),
            format_status=lambda result, *, workspace=None: call(
                symbols,
                "_format_status",
                result,
                workspace=workspace,
            ),
            format_worker_runner_history=lambda result: call(symbols, "_format_worker_runner_history", result),
            format_train_queue_history=lambda result: call(symbols, "_format_train_queue_history", result),
            format_train_queue_daemon_status=lambda result: call(symbols, "_format_train_queue_daemon_status", result),
            format_train_queue_daemon_history=lambda result: call(
                symbols,
                "_format_train_queue_daemon_history",
                result,
            ),
            format_daemon_health_status=lambda result: call(symbols, "_format_daemon_health_status", result),
            format_daemon_heartbeat_status=lambda result: call(symbols, "_format_daemon_heartbeat_status", result),
            format_daemon_lease_status=lambda result: call(symbols, "_format_daemon_lease_status", result),
            format_daemon_stale_check=lambda result: call(symbols, "_format_daemon_stale_check", result),
            format_daemon_alerts=lambda result: call(symbols, "_format_daemon_alerts", result),
            format_candidate_history=lambda result: call(symbols, "_format_candidate_history", result),
            format_candidate_timeline=lambda result: call(symbols, "_format_candidate_timeline", result),
            read_train_queue_daemon_state=lambda workspace=None: call(
                symbols,
                "_read_train_queue_daemon_state",
                workspace,
            ),
            update_train_queue_daemon_state=lambda **kwargs: call(
                symbols,
                "_update_train_queue_daemon_state",
                **kwargs,
            ),
            daemon_recovery_payload=lambda **kwargs: call(symbols, "_daemon_recovery_payload", **kwargs),
        ),
    )
