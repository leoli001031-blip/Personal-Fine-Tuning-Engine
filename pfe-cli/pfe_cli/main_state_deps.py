"""Dependency builders for CLI state and command execution compatibility."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .cli_state_helpers import CLIStateDeps
from .command_execution import CommandExecutionDeps
from .main_deps_common import symbol


def make_cli_state_deps(symbols: Mapping[str, Any]) -> CLIStateDeps:
    return CLIStateDeps(
        coerce_mapping=symbol(symbols, "_coerce_mapping"),
        optional_module_call=symbol(symbols, "_optional_module_call"),
        pick_first=symbol(symbols, "_pick_first"),
    )


def make_command_execution_deps(symbols: Mapping[str, Any]) -> CommandExecutionDeps:
    return CommandExecutionDeps(
        coerce_mapping=symbol(symbols, "_coerce_mapping"),
        friendly_exception_message=symbol(symbols, "_friendly_exception_message"),
    )
