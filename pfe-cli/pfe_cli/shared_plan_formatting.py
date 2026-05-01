"""Shared plan and trainer formatting helper facade."""

from __future__ import annotations

from .shared_plan_blocks import format_compact_plan_line, format_plan_block, plan_summary
from .shared_trainer_formatting import format_trainer_block, format_trainer_summary


__all__ = [
    "format_compact_plan_line",
    "format_plan_block",
    "format_trainer_block",
    "format_trainer_summary",
    "plan_summary",
]
