"""Adapter management command registration for the PFE CLI."""

from __future__ import annotations

from dataclasses import dataclass

import typer

from .adapter_basic_commands import register_adapter_basic_commands
from .adapter_lifecycle_formatting import _echo_result, _format_lifecycle_summary
from .adapter_lineage_commands import register_adapter_lineage_commands
from .adapter_store_helpers import _call_store, _load_adapter_store

__all__ = [
    "AdapterCommandContext",
    "_call_store",
    "_echo_result",
    "_format_lifecycle_summary",
    "_load_adapter_store",
    "adapter_app",
]


@dataclass(frozen=True)
class AdapterCommandContext:
    """Shared CLI context for adapter operations."""

    workspace: str | None = None
    config_path: str | None = None


adapter_app = typer.Typer(
    help=(
        "Adapter lifecycle management in strict_local mode. "
        "Only 'promote' updates latest; OpenAI compatibility applies to inference only."
    ),
    no_args_is_help=True,
)

register_adapter_basic_commands(adapter_app)
register_adapter_lineage_commands(adapter_app)
