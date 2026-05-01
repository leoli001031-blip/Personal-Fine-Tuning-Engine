"""Shared command execution helper facade."""

from __future__ import annotations

from .command_handler_execution import (
    CommandExecutionDeps,
    friendly_exception_message,
    run_handler,
    run_handler_json,
    run_placeholder,
)
from .command_service_resolution import load_service, optional_module_call, resolve_handler


__all__ = [
    "CommandExecutionDeps",
    "friendly_exception_message",
    "load_service",
    "optional_module_call",
    "resolve_handler",
    "run_handler",
    "run_handler_json",
    "run_placeholder",
]
