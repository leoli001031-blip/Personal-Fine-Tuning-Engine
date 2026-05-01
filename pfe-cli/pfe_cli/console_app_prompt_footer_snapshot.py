"""Sidebar snapshot rendering for the Rich operations console."""

from __future__ import annotations

from rich.text import Text

from .console_app_badges import _ops_badge


def sidebar_snapshot_text(
    *,
    ops_refresh_state: str | None,
    ops_age_seconds: float | None,
    refresh_seconds: float | None,
) -> Text:
    line = Text()
    line.append("snap=", style="dim")
    line.append_text(_ops_badge(ops_refresh_state))
    if ops_age_seconds is not None:
        line.append(" · ", style="dim")
        line.append(f"age={ops_age_seconds:.1f}s", style="dim")
    if refresh_seconds is not None:
        line.append(" · ", style="dim")
        line.append(f"cadence={refresh_seconds:.1f}s", style="dim")
    return line


__all__ = ["sidebar_snapshot_text"]
