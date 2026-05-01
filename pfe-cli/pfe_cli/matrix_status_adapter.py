"""Adapter lifecycle status section for Matrix terminal output."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .matrix_formatting_common import _coerce_mapping
from .terminal_theme import draw_box, format_key_value


def append_adapter_lifecycle_section(lines: list[str], mapping: Mapping[str, Any]) -> None:
    """Append adapter lifecycle status box."""
    adapter_content = []

    latest_adapter = _coerce_mapping(mapping.get("latest_adapter"))
    if latest_adapter:
        version = latest_adapter.get("version", "n/a")
        state = latest_adapter.get("state", "unknown")
        latest_parts = [f"{version} | state={state}"]
        for key in ("export_artifact_valid", "export_artifact_exists"):
            value = latest_adapter.get(key)
            if value is not None:
                latest_parts.append(f"{key}={value}")
        export_artifact_path = latest_adapter.get("export_artifact_path")
        if export_artifact_path is not None:
            latest_parts.append(f"export_artifact_path={export_artifact_path}")
        adapter_content.append(format_key_value("latest promoted", " | ".join(latest_parts)))

    recent_adapter = _coerce_mapping(mapping.get("recent_adapter"))
    if recent_adapter:
        version = recent_adapter.get("version", "n/a")
        state = recent_adapter.get("state", "unknown")
        recent_parts = [f"{version} | state={state}"]
        execution_backend = recent_adapter.get("execution_backend")
        if execution_backend is not None:
            recent_parts.append(f"execution_backend={execution_backend}")
        executor_mode = recent_adapter.get("executor_mode")
        if executor_mode is not None:
            recent_parts.append(f"executor_mode={executor_mode}")
        adapter_content.append(format_key_value("recent training", " | ".join(recent_parts)))

    signal_summary = _coerce_mapping(mapping.get("signal_summary"))
    if signal_summary:
        total = signal_summary.get("total_signals", 0)
        processed = signal_summary.get("processed_signals", 0)
        adapter_content.append(format_key_value("signals", f"total={total} | processed={processed}"))

    sample_counts = _coerce_mapping(mapping.get("sample_counts"))
    if sample_counts:
        train = sample_counts.get("train", 0)
        val = sample_counts.get("val", 0)
        test = sample_counts.get("test", 0)
        adapter_content.append(format_key_value("samples", f"train={train} | val={val} | test={test}"))

    lifecycle = _coerce_mapping(mapping.get("adapter_lifecycle"))
    if lifecycle:
        counts = _coerce_mapping(lifecycle.get("counts"))
        if counts:
            lifecycle_parts = []
            for key in ("promoted", "archived", "pending_eval", "training", "failed_eval"):
                val = counts.get(key)
                if val:
                    lifecycle_parts.append(f"{key}={val}")
            if lifecycle_parts:
                adapter_content.append(format_key_value("lifecycle counts", " | ".join(lifecycle_parts)))
        promoted_versions = lifecycle.get("promoted_versions")
        if promoted_versions:
            adapter_content.append(format_key_value("promoted versions", ", ".join(str(v) for v in promoted_versions)))
        archived_versions = lifecycle.get("archived_versions")
        if archived_versions:
            adapter_content.append(format_key_value("archived versions", ", ".join(str(v) for v in archived_versions)))

    if adapter_content:
        lines.append(draw_box("ADAPTER LIFECYCLE", adapter_content))
        lines.append("")


__all__ = ["append_adapter_lifecycle_section"]
