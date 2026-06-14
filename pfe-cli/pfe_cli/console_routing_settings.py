"""Settings and utility console slash-command routing."""

from __future__ import annotations

from .console_routing_context import ConsoleCommandResult, ConsoleRouteContext


def route_console_settings_command(ctx: ConsoleRouteContext) -> ConsoleCommandResult | None:
    """Route settings, utility, and local preview commands."""
    normalized = ctx.normalized
    deps = ctx.deps

    if normalized == "doctor":
        doctor_model = None if ctx.model in {"local", "base", "local-default"} else ctx.model
        return deps.format_doctor(workspace=ctx.current_workspace, base_model=doctor_model), "doctor", None
    if normalized == "serve":
        return (
            deps.format_serve_preview(
                port=8921,
                host="127.0.0.1",
                adapter=ctx.adapter,
                workspace=ctx.current_workspace,
                api_key=None,
                real_local=ctx.real_local,
            ),
            "serve",
            None,
        )
    if normalized == "clear":
        return "", "clear", None
    if normalized == "settings":
        return (
            deps.console_settings_text(
                workspace=ctx.current_workspace,
                mode=ctx.mode,
                model=ctx.model,
                adapter=ctx.adapter,
                temperature=ctx.temperature,
                max_tokens=ctx.max_tokens,
                real_local=ctx.real_local,
                refresh_seconds=ctx.refresh_seconds,
            ),
            "settings",
            None,
        )
    if normalized.startswith("workspace "):
        selected_workspace = ctx.command.split(" ", 1)[1].strip()
        if not selected_workspace:
            return "Usage: /workspace <name>", "unknown", None
        return f"workspace set to {selected_workspace}", "set-workspace", {"workspace": selected_workspace}
    if normalized.startswith("model "):
        selected_model = ctx.command.split(" ", 1)[1].strip()
        if not selected_model:
            return "Usage: /model <id>", "unknown", None
        return f"model set to {selected_model}", "set-model", {"model": selected_model}
    if normalized.startswith("adapter "):
        selected_adapter = ctx.command.split(" ", 1)[1].strip()
        if not selected_adapter:
            return "Usage: /adapter <version>", "unknown", None
        return f"adapter set to {selected_adapter}", "set-adapter", {"adapter": selected_adapter}
    if normalized.startswith("temperature "):
        value = normalized.split(" ", 1)[1].strip()
        try:
            selected_temperature = float(value)
        except ValueError:
            return "Usage: /temperature <value>", "unknown", None
        if selected_temperature < 0.0 or selected_temperature > 2.0:
            return "Temperature must be between 0.0 and 2.0.", "unknown", None
        return f"temperature set to {selected_temperature:.2f}", "set-temperature", {"temperature": selected_temperature}
    if normalized.startswith("max-tokens "):
        value = normalized.split(" ", 1)[1].strip().lower()
        if value in {"auto", "none", "default"}:
            return "max tokens set to auto", "set-max-tokens", {"max_tokens": None}
        try:
            selected_max_tokens = int(value)
        except ValueError:
            return "Usage: /max-tokens <n|auto>", "unknown", None
        if selected_max_tokens < 1:
            return "Max tokens must be at least 1.", "unknown", None
        return f"max tokens set to {selected_max_tokens}", "set-max-tokens", {"max_tokens": selected_max_tokens}
    if normalized.startswith("real-local "):
        selected_state = normalized.split(" ", 1)[1].strip()
        if selected_state in {"on", "true", "1", "yes"}:
            return "real-local enabled", "set-real-local", {"real_local": True}
        if selected_state in {"off", "false", "0", "no"}:
            return "real-local disabled", "set-real-local", {"real_local": False}
        return "Usage: /real-local on|off", "unknown", None
    if normalized.startswith("refresh "):
        value = normalized.split(" ", 1)[1].strip()
        try:
            refresh_value = float(value)
        except ValueError:
            return "Usage: /refresh <seconds>", "unknown", None
        if refresh_value < 0.1:
            return "Refresh must be at least 0.1 seconds.", "unknown", None
        return f"refresh set to {refresh_value:.1f}s", "set-refresh", {"refresh_seconds": refresh_value}

    return None


__all__ = ["route_console_settings_command"]
