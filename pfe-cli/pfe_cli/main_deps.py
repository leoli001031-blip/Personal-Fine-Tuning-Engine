"""Dependency builders for the main CLI compatibility layer."""

from __future__ import annotations

from .main_console_deps import (
    make_console_actions_deps,
    make_console_io_deps,
    make_console_routing_deps,
    make_console_surface_deps,
)
from .main_operations_deps import (
    make_daemon_formatting_deps,
    make_operations_formatting_deps,
    make_operations_history_formatting_deps,
)
from .main_preview_deps import make_serve_formatting_deps, make_training_preview_deps
from .main_result_deps import (
    make_adapter_snapshot_deps,
    make_doctor_formatting_deps,
    make_legacy_result_deps,
)
from .main_state_deps import make_cli_state_deps, make_command_execution_deps
from .main_status_deps import (
    make_plan_snapshot_deps,
    make_status_formatting_deps,
    make_status_legacy_formatting_deps,
)

__all__ = [
    "make_adapter_snapshot_deps",
    "make_cli_state_deps",
    "make_command_execution_deps",
    "make_console_actions_deps",
    "make_console_io_deps",
    "make_console_routing_deps",
    "make_console_surface_deps",
    "make_daemon_formatting_deps",
    "make_doctor_formatting_deps",
    "make_legacy_result_deps",
    "make_operations_formatting_deps",
    "make_operations_history_formatting_deps",
    "make_plan_snapshot_deps",
    "make_serve_formatting_deps",
    "make_status_formatting_deps",
    "make_status_legacy_formatting_deps",
    "make_training_preview_deps",
]
