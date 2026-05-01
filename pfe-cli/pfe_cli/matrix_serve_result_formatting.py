"""Matrix terminal formatters for serve results."""

from __future__ import annotations

from typing import Any

from .matrix_formatting_common import _coerce_mapping, _format_scalar
from .terminal_theme import MatrixColors, draw_box, draw_header, format_key_value


def format_serve_preview_matrix(
    port: int,
    host: str,
    adapter: str,
    workspace: str | None,
    api_key: str | None,
    real_local: bool,
    recent_training: dict[str, Any] | None = None,
    latest_training: dict[str, Any] | None = None,
) -> str:
    """Format serve preview in Matrix Green terminal style."""
    lines = []

    lines.append(draw_header("SERVE PREVIEW"))

    content = []
    content.append(format_key_value("host", host))
    content.append(format_key_value("port", port))
    content.append(format_key_value("adapter", adapter))
    content.append(format_key_value("workspace", workspace or "default"))
    content.append(format_key_value("api_key", f"{MatrixColors.GREEN}SET{MatrixColors.RESET}" if api_key else f"{MatrixColors.GRAY}UNSET{MatrixColors.RESET}"))
    content.append(format_key_value("mode", f"{MatrixColors.GREEN}REAL{MatrixColors.RESET}" if real_local else f"{MatrixColors.GRAY}MOCK{MatrixColors.RESET}"))

    lines.append(draw_box("SERVER CONFIGURATION", content))
    lines.append("")

    if latest_training:
        lt_content = []
        version = latest_training.get("version")
        state = latest_training.get("state")
        if version is not None:
            lt_content.append(format_key_value("version", version))
        if state is not None:
            lt_content.append(format_key_value("state", state))
        if lt_content:
            lines.append(draw_box("LATEST PROMOTED", lt_content))
            lines.append("")

    if recent_training:
        rt_content = []
        version = recent_training.get("version")
        state = recent_training.get("state")
        if version is not None:
            rt_content.append(format_key_value("version", version))
        if state is not None:
            rt_content.append(format_key_value("state", state))
        execution_backend = recent_training.get("execution_backend")
        if execution_backend is not None:
            rt_content.append(format_key_value("execution backend", execution_backend))
        executor_mode = recent_training.get("executor_mode")
        if executor_mode is not None:
            rt_content.append(format_key_value("executor mode", executor_mode))
        job_execution = _coerce_mapping(recent_training.get("real_execution_summary") or recent_training.get("job_execution"))
        if job_execution:
            for key in ("status", "state", "executor_mode", "execution_mode", "attempted", "success", "runner_status", "kind"):
                value = job_execution.get(key)
                if value is not None:
                    rt_content.append(format_key_value(key.replace("_", " "), value))
        export_execution = _coerce_mapping(recent_training.get("export_execution") or recent_training.get("export_toolchain_summary"))
        if export_execution:
            for key in ("status", "state", "execution_mode", "attempted", "success"):
                value = export_execution.get(key)
                if value is not None:
                    rt_content.append(format_key_value(key.replace("_", " "), value))
        if rt_content:
            lines.append(draw_box("RECENT TRAINING", rt_content))
            lines.append("")

    return "\n".join(lines)


def format_serve_matrix(result: Any) -> str:
    """Format serve result in Matrix Green terminal style."""
    mapping = _coerce_mapping(result)
    if mapping and "ready_message" in mapping:
        return f"{MatrixColors.GREEN}    [✓] {mapping['ready_message']}{MatrixColors.RESET}"
    return _format_scalar(result)


__all__ = ["format_serve_matrix", "format_serve_preview_matrix"]
