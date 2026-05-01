"""Execution section boxes for Matrix terminal status output."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .matrix_formatting_common import _coerce_mapping
from .terminal_theme import MatrixColors, draw_box, format_key_value


def append_real_execution_box(lines: list[str], mapping: Mapping[str, Any]) -> None:
    real_execution_summary = _real_execution_summary(mapping)
    if not real_execution_summary:
        return

    content: list[str] = []
    for key in (
        "status",
        "state",
        "kind",
        "executor_mode",
        "execution_mode",
        "attempted",
        "success",
        "available",
        "runner_status",
    ):
        value = real_execution_summary.get(key)
        if value is not None:
            content.append(format_key_value(key.replace("_", " "), value))
    audit = _coerce_mapping(real_execution_summary.get("audit"))
    if audit:
        for key in ("runner_status", "status", "execution_status"):
            value = audit.get(key)
            if value is not None:
                content.append(format_key_value(f"audit {key.replace('_', ' ')}", value))
    if content:
        lines.append(draw_box("REAL EXECUTION", content))
        lines.append("")


def append_export_toolchain_box(lines: list[str], mapping: Mapping[str, Any]) -> None:
    export_toolchain_summary = _export_toolchain_summary(mapping)
    if not export_toolchain_summary:
        return

    content: list[str] = []
    for key in (
        "status",
        "summary",
        "toolchain_status",
        "execution_mode",
        "attempted",
        "success",
        "required",
        "output_artifact_valid",
    ):
        value = export_toolchain_summary.get(key)
        if value is not None:
            content.append(format_key_value(key.replace("_", " "), value))
    metadata = _coerce_mapping(export_toolchain_summary.get("metadata"))
    if metadata:
        for key in ("execution_mode", "status"):
            value = metadata.get(key)
            if value is not None:
                content.append(format_key_value(f"meta {key}", value))
    audit = _coerce_mapping(export_toolchain_summary.get("audit"))
    if audit:
        for key in ("status", "execution_status"):
            value = audit.get(key)
            if value is not None:
                content.append(format_key_value(f"audit {key}", value))
    if content:
        lines.append(draw_box("EXPORT TOOLCHAIN", content))
        lines.append("")


def append_system_health_box(lines: list[str], mapping: Mapping[str, Any]) -> None:
    system_health = _coerce_mapping(mapping.get("system_health"))
    if not system_health:
        return

    health_content = [
        format_key_value("daemon", _online_status(bool(system_health.get("daemon_active", False)))),
        format_key_value("runner", _online_status(bool(system_health.get("runner_active", False)))),
        format_key_value("queue pending", system_health.get("queue_pending_jobs", 0)),
    ]
    queue_failed = system_health.get("queue_failed_jobs", 0)
    health_content.append(
        format_key_value(
            "queue failed",
            queue_failed if queue_failed == 0 else f"{MatrixColors.RED}{queue_failed}{MatrixColors.RESET}",
        )
    )
    lines.append(draw_box("SYSTEM HEALTH", health_content))
    lines.append("")


def _real_execution_summary(mapping: Mapping[str, Any]) -> dict[str, Any] | None:
    real_execution_summary = _coerce_mapping(mapping.get("real_execution_summary"))
    if real_execution_summary is None:
        real_execution_summary = _coerce_mapping(mapping.get("job_execution"))
    if real_execution_summary is None:
        recent_training_snapshot = _coerce_mapping(mapping.get("recent_training_snapshot"))
        if recent_training_snapshot:
            real_execution_summary = _coerce_mapping(
                recent_training_snapshot.get("real_execution_summary")
                or recent_training_snapshot.get("job_execution_summary")
                or recent_training_snapshot.get("job_execution")
            )
    return real_execution_summary


def _export_toolchain_summary(mapping: Mapping[str, Any]) -> dict[str, Any] | None:
    export_toolchain_summary = _coerce_mapping(mapping.get("export_toolchain_summary"))
    if export_toolchain_summary is None:
        export_toolchain_summary = _coerce_mapping(mapping.get("export_execution"))
    if export_toolchain_summary is None:
        recent_training_snapshot = _coerce_mapping(mapping.get("recent_training_snapshot"))
        if recent_training_snapshot:
            export_toolchain_summary = _coerce_mapping(
                recent_training_snapshot.get("export_execution")
                or recent_training_snapshot.get("export_toolchain_summary")
            )
    return export_toolchain_summary


def _online_status(active: bool) -> str:
    color = MatrixColors.GREEN if active else MatrixColors.RED
    label = "ONLINE" if active else "OFFLINE"
    return f"{color}{label}{MatrixColors.RESET}"


__all__ = ["append_export_toolchain_box", "append_real_execution_box", "append_system_health_box"]
