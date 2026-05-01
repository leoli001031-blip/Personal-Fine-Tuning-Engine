"""Install legacy console routing helpers onto the main module namespace."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .console_routing import (
    console_candidate_summary_text,
    console_command_output,
    console_daemon_summary_text,
    console_gate_summary_text,
    console_queue_summary_text,
    console_runner_summary_text,
    console_runtime_summary_text,
    console_trigger_summary_text,
)
from .main_deps import make_console_routing_deps


def _call(symbols: Mapping[str, Any], name: str, *args: Any, **kwargs: Any) -> Any:
    return symbols[name](*args, **kwargs)


def install_console_routing_compat(symbols: dict[str, Any]) -> None:
    def _console_routing_deps() -> Any:
        return make_console_routing_deps(symbols)

    def _console_candidate_summary_text(
        payload: Mapping[str, Any],
        timeline: Mapping[str, Any] | None = None,
    ) -> str:
        return console_candidate_summary_text(payload, timeline=timeline, deps=_call(symbols, "_console_routing_deps"))

    def _console_queue_summary_text(
        payload: Mapping[str, Any],
        history: Mapping[str, Any] | None = None,
    ) -> str:
        return console_queue_summary_text(payload, history=history, deps=_call(symbols, "_console_routing_deps"))

    def _console_runner_summary_text(
        payload: Mapping[str, Any],
        history: Mapping[str, Any] | None = None,
    ) -> str:
        return console_runner_summary_text(payload, history=history, deps=_call(symbols, "_console_routing_deps"))

    def _console_daemon_summary_text(result: Any) -> str:
        return console_daemon_summary_text(result, deps=_call(symbols, "_console_routing_deps"))

    def _console_trigger_summary_text(payload: Mapping[str, Any]) -> str:
        return console_trigger_summary_text(payload, deps=_call(symbols, "_console_routing_deps"))

    def _console_gate_summary_text(payload: Mapping[str, Any]) -> str:
        return console_gate_summary_text(payload, deps=_call(symbols, "_console_routing_deps"))

    def _console_runtime_summary_text(payload: Mapping[str, Any]) -> str:
        return console_runtime_summary_text(payload, deps=_call(symbols, "_console_routing_deps"))

    def _console_command_output(
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
        last_interaction: dict[str, Any] | None = None,
    ) -> tuple[str | None, str, dict[str, Any] | None]:
        return console_command_output(
            command,
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
            last_interaction=last_interaction,
            deps=_call(symbols, "_console_routing_deps"),
        )

    symbols.update(
        {
            "_console_routing_deps": _console_routing_deps,
            "_console_candidate_summary_text": _console_candidate_summary_text,
            "_console_queue_summary_text": _console_queue_summary_text,
            "_console_runner_summary_text": _console_runner_summary_text,
            "_console_daemon_summary_text": _console_daemon_summary_text,
            "_console_trigger_summary_text": _console_trigger_summary_text,
            "_console_gate_summary_text": _console_gate_summary_text,
            "_console_runtime_summary_text": _console_runtime_summary_text,
            "_console_command_output": _console_command_output,
        }
    )


__all__ = ["install_console_routing_compat"]
