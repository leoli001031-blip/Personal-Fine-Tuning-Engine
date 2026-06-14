"""Status header panel for the Rich operations console."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any

from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.text import Text

from .console_app_badges import _severity_badge
from .console_app_data import _mapping, _value


def _status_header(payload: Mapping[str, Any], *, workspace: str | None = None) -> RenderableType:
    latest = _mapping(payload.get("latest_adapter"))
    dashboard = _mapping(payload.get("operations_dashboard"))
    metadata = _mapping(payload.get("metadata"))
    inference = _mapping(metadata.get("inference"))
    plans = _mapping(payload.get("plans"))
    inference_plan = _mapping(plans.get("inference"))
    mode = "strict_local" if payload.get("strict_local", True) else _value(payload, "mode", default="unknown")
    inference_backend = _value(
        payload,
        "inference_backend",
        default=_value(inference, "selected_backend", "provider", default=_value(inference_plan, "selected_backend", default="unknown")),
    )
    workspace_name = str(workspace or payload.get("workspace") or "user_default")
    latest_value = latest.get("version") and f"{latest.get('version')} ({latest.get('state', 'unknown')})" or "none"
    severity = _value(dashboard, "severity", default="stable")
    line_one = Text()
    line_one.append("ws=", style="bold")
    line_one.append(workspace_name, style="white")
    line_one.append(" · ", style="dim")
    line_one.append("md=", style="bold")
    line_one.append(mode, style="white")
    line_one.append(" · ", style="dim")
    line_one.append("infer=", style="bold")
    line_one.append(inference_backend, style="white")

    line_two = Text()
    line_two.append("lat=", style="bold")
    line_two.append(latest_value, style="white")
    line_two.append(" · ", style="dim")
    line_two.append("sev=", style="bold")
    line_two.append(severity, style="white")
    line_two.append(" ", style="dim")
    line_two.append_text(_severity_badge(severity))
    line_two.append(" · ", style="dim")
    line_two.append("upd=", style="bold")
    line_two.append(datetime.now().strftime("%H:%M:%S"), style="white")
    return Panel(Group(line_one, line_two), title="PFE Console", border_style="cyan")


__all__ = ["_status_header"]
