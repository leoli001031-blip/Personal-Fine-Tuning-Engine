"""Dependency builders for console compatibility helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .console_actions import ConsoleActionsDeps
from .console_io import ConsoleIODeps
from .console_routing import ConsoleRoutingDeps
from .console_surface import ConsoleSurfaceDeps
from .main_deps_common import call, symbol


def make_console_io_deps(symbols: Mapping[str, Any]) -> ConsoleIODeps:
    return ConsoleIODeps(
        coerce_mapping=symbol(symbols, "_coerce_mapping"),
        format_scalar=symbol(symbols, "_format_scalar"),
    )


def make_console_surface_deps(symbols: Mapping[str, Any]) -> ConsoleSurfaceDeps:
    return ConsoleSurfaceDeps(
        coerce_mapping=symbol(symbols, "_coerce_mapping"),
        format_scalar=symbol(symbols, "_format_scalar"),
        yes_no=symbol(symbols, "_yes_no"),
        prefer_inspection_summary_for_generic_monitor=symbol(
            symbols,
            "_prefer_inspection_summary_for_generic_monitor",
        ),
    )


def make_console_actions_deps(symbols: Mapping[str, Any]) -> ConsoleActionsDeps:
    return ConsoleActionsDeps(
        coerce_mapping=symbol(symbols, "_coerce_mapping"),
        console_dashboard_focus=symbol(symbols, "_console_dashboard_focus"),
    )


def make_console_routing_deps(symbols: Mapping[str, Any]) -> ConsoleRoutingDeps:
    return ConsoleRoutingDeps(
        coerce_mapping=lambda result: call(symbols, "_coerce_mapping", result),
        format_scalar=lambda value: call(symbols, "_format_scalar", value),
        resolve_handler=lambda service, *names: call(symbols, "_resolve_handler", service, *names),
        console_submit_feedback=lambda **kwargs: call(symbols, "_console_submit_feedback", **kwargs),
        console_help_text=lambda: call(symbols, "_console_help_text"),
        console_status_compact_text=lambda payload, *, workspace=None: call(
            symbols,
            "_console_status_compact_text",
            payload,
            workspace=workspace,
        ),
        console_focus_actions=lambda payload=None: call(symbols, "_console_focus_actions", payload),
        console_settings_text=lambda **kwargs: call(symbols, "_console_settings_text", **kwargs),
        format_status=lambda result, *, workspace=None: call(
            symbols,
            "_format_status",
            result,
            workspace=workspace,
        ),
        format_operations_dashboard=lambda result: call(symbols, "_format_operations_dashboard", result),
        format_operations_alert_surface=lambda result: call(symbols, "_format_operations_alert_surface", result),
        format_operations_alert_policy=lambda result: call(symbols, "_format_operations_alert_policy", result),
        format_operations_event_stream=lambda result: call(symbols, "_format_operations_event_stream", result),
        format_operations_console_digest=lambda result: call(symbols, "_format_operations_console_digest", result),
        format_doctor=lambda **kwargs: call(symbols, "_format_doctor", **kwargs),
        format_serve_preview=lambda **kwargs: call(symbols, "_format_serve_preview", **kwargs),
        format_train_queue_daemon_status=lambda result: call(symbols, "_format_train_queue_daemon_status", result),
        format_worker_runner_status=lambda result: call(symbols, "_format_worker_runner_status", result),
        format_eval_result=lambda result, *, workspace=None: call(
            symbols,
            "_format_eval_result",
            result,
            workspace=workspace,
        ),
        format_candidate_history=lambda result: call(symbols, "_format_candidate_history", result),
        format_candidate_timeline=lambda result: call(symbols, "_format_candidate_timeline", result),
        format_train_queue_history=lambda result: call(symbols, "_format_train_queue_history", result),
        format_runner_timeline_summary=lambda result: call(symbols, "_format_runner_timeline_summary", result),
        format_worker_runner_history=lambda result: call(symbols, "_format_worker_runner_history", result),
        format_train_queue_daemon_history=lambda result: call(symbols, "_format_train_queue_daemon_history", result),
        read_train_queue_daemon_state=lambda workspace: call(symbols, "_read_train_queue_daemon_state", workspace),
        format_daemon_timeline_summary=lambda result: call(symbols, "_format_daemon_timeline_summary", result),
        format_lifecycle_summary=lambda result: call(symbols, "_format_lifecycle_summary", result),
        format_train_result=lambda result, *, workspace=None: call(
            symbols,
            "_format_train_result",
            result,
            workspace=workspace,
        ),
    )
