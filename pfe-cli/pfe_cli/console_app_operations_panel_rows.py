"""Row construction for the Rich operations summary panel."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from rich.table import Table

from .console_app_badges import _handle_text, _runtime_stability_text
from .console_app_data import _compact_text
from .console_app_operations_context import OperationsPanelContext
from .console_app_operations_panel_bits import (
    _gate_bits,
    _handling_bits,
    _policy_bits,
    _policy_gate_bits,
    _review_bits,
)
from .console_app_operations_panel_text import _action_command_text, _action_text, _focus_text, _trigger_text


def build_operations_panel_table(payload: Mapping[str, Any], ctx: OperationsPanelContext) -> Table:
    table = Table.grid(padding=(0, 1))
    table.add_column(style="bold")
    table.add_column()
    table.add_row("F", _focus_text(ctx))
    table.add_row("A", _action_text(ctx))
    table.add_row("Pol", " | ".join(_policy_bits(ctx)))
    table.add_row("PGate", " | ".join(_policy_gate_bits(ctx)))
    table.add_row("Gate", " | ".join(_gate_bits(ctx)))
    table.add_row("Trig", _trigger_text(ctx))
    table.add_row("QRev", " | ".join(_review_bits(ctx)))
    table.add_row("Stab", _runtime_stability_text(ctx.runtime_stability, severity=ctx.severity))
    table.add_row("Handle", _handle_text(_handling_bits(ctx), ctx.alert_policy))
    table.add_row("Sum", _compact_text(ctx.summary_source, max_len=30))
    do_text, see_text = _action_command_text(payload, ctx)
    table.add_row("Do", do_text)
    table.add_row("See", see_text)
    table.add_row("G", _compact_text(ctx.guidance_source, max_len=20))
    return table


__all__ = ["build_operations_panel_table"]
