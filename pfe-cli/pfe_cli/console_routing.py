"""Console slash-command routing helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .console_routing_actions import route_console_action_command
from .console_routing_context import ConsoleCommandResult, ConsoleRouteContext
from .console_routing_core_commands import route_console_core_command
from .console_routing_focus_commands import route_console_focus_command
from .console_routing_settings import route_console_settings_command
from .console_routing_views import route_console_view_command
from .console_routing_deps import ConsoleRoutingDeps
from .console_routing_summaries import (
    console_candidate_summary_text,
    console_daemon_summary_text,
    console_gate_summary_text,
    console_queue_summary_text,
    console_runner_summary_text,
    console_runtime_summary_text,
    console_trigger_summary_text,
)


def console_command_output(
    command: str,
    *,
    payload: Mapping[str, Any],
    workspace: str | None,
    service: Any,
    current_workspace: str | None,
    mode: str,
    model: str,
    adapter: str,
    temperature: float,
    max_tokens: int | None,
    real_local: bool,
    refresh_seconds: float,
    deps: ConsoleRoutingDeps,
    last_interaction: dict[str, Any] | None = None,
) -> tuple[str | None, str, dict[str, Any] | None]:
    normalized = command.strip().lower()
    ctx = ConsoleRouteContext(
        command=command,
        normalized=normalized,
        payload=payload,
        workspace=workspace,
        service=service,
        current_workspace=current_workspace,
        mode=mode,
        model=model,
        adapter=adapter,
        temperature=temperature,
        max_tokens=max_tokens,
        real_local=real_local,
        refresh_seconds=refresh_seconds,
        deps=deps,
        last_interaction=last_interaction,
    )

    for router in (
        route_console_core_command,
        route_console_action_command,
        lambda route_ctx: route_console_focus_command(
            route_ctx,
            reroute=lambda next_command: _reroute_console_command(route_ctx, next_command),
        ),
        route_console_settings_command,
        route_console_view_command,
    ):
        result = router(ctx)
        if result is not None:
            return result

    return f"Unknown command: /{command}. Try /help.", "unknown", None


def _reroute_console_command(ctx: ConsoleRouteContext, command: str) -> ConsoleCommandResult:
    return console_command_output(
        command,
        payload=ctx.payload,
        workspace=ctx.workspace,
        service=ctx.service,
        current_workspace=ctx.current_workspace,
        mode=ctx.mode,
        model=ctx.model,
        adapter=ctx.adapter,
        temperature=ctx.temperature,
        max_tokens=ctx.max_tokens,
        real_local=ctx.real_local,
        refresh_seconds=ctx.refresh_seconds,
        deps=ctx.deps,
    )


__all__ = [
    "ConsoleRoutingDeps",
    "console_candidate_summary_text",
    "console_command_output",
    "console_daemon_summary_text",
    "console_gate_summary_text",
    "console_queue_summary_text",
    "console_runner_summary_text",
    "console_runtime_summary_text",
    "console_trigger_summary_text",
]
