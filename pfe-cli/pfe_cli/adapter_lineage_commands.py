"""Adapter lineage command group registration."""

from __future__ import annotations

import typer

from .adapter_compare_commands import register_adapter_compare_command
from .adapter_history_commands import register_adapter_history_command
from .adapter_lineage_view_commands import register_adapter_lineage_view_command


def register_adapter_lineage_commands(adapter_app: typer.Typer) -> None:
    register_adapter_lineage_view_command(adapter_app)
    register_adapter_history_command(adapter_app)
    register_adapter_compare_command(adapter_app)


__all__ = ["register_adapter_lineage_commands"]
