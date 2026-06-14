"""Action-oriented console slash-command routing."""

from __future__ import annotations

from .console_routing_candidate_actions import route_console_candidate_action
from .console_routing_context import ConsoleCommandResult, ConsoleRouteContext
from .console_routing_daemon_actions import route_console_daemon_action
from .console_routing_pipeline_actions import route_console_pipeline_action
from .console_routing_queue_actions import (
    route_console_queue_action,
    route_console_queue_batch_action,
)
from .console_routing_trigger_actions import route_console_trigger_action


def route_console_action_command(ctx: ConsoleRouteContext) -> ConsoleCommandResult | None:
    """Route mutating and service-triggering console commands."""
    for router in (
        route_console_candidate_action,
        route_console_queue_action,
        route_console_trigger_action,
        route_console_daemon_action,
        route_console_queue_batch_action,
        route_console_pipeline_action,
    ):
        result = router(ctx)
        if result is not None:
            return result

    return None


__all__ = ["route_console_action_command"]
