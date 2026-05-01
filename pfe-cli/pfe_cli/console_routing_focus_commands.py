"""Focus-aware console slash-command routing."""

from __future__ import annotations

from collections.abc import Callable

from .console_routing_context import ConsoleCommandResult, ConsoleRouteContext


def route_console_focus_command(
    ctx: ConsoleRouteContext,
    *,
    reroute: Callable[[str], ConsoleCommandResult],
) -> ConsoleCommandResult | None:
    if ctx.normalized == "do":
        focus_actions = ctx.deps.console_focus_actions(ctx.payload)
        primary_exec = focus_actions.get("primary_exec")
        primary_label = str(focus_actions.get("primary_label") or "/status")
        if not primary_exec:
            return f"Primary action requires a review choice. Use {primary_label}.", "do-ambiguous", None
        return reroute(str(primary_exec))

    if ctx.normalized == "see":
        focus_actions = ctx.deps.console_focus_actions(ctx.payload)
        secondary_exec = focus_actions.get("secondary_exec")
        secondary_label = str(focus_actions.get("secondary_label") or "/status")
        if not secondary_exec:
            return f"No secondary view is available. Try {secondary_label}.", "see-unavailable", None
        return reroute(str(secondary_exec))

    return None


__all__ = ["route_console_focus_command"]
