"""Matrix terminal formatter for train results."""

from __future__ import annotations

from typing import Any

from .matrix_formatting_common import _coerce_mapping
from .terminal_theme import MatrixColors, draw_box, draw_header, format_key_value


def format_train_result_matrix(result: Any, *, workspace: str | None = None) -> str:
    """Format train result in Matrix Green terminal style."""
    lines = []

    lines.append(draw_header("TRAINING COMPLETE"))

    mapping = _coerce_mapping(result)
    if mapping is None:
        lines.append(f"{MatrixColors.RED}ERROR: Unable to parse training result{MatrixColors.RESET}")
        return "\n".join(lines)

    content = []

    version = mapping.get("version", "n/a")
    content.append(format_key_value("version", version))

    adapter_path = mapping.get("adapter_path", "n/a")
    content.append(format_key_value("path", f"{MatrixColors.GRAY}{adapter_path}{MatrixColors.RESET}"))

    num_samples = mapping.get("num_samples", 0)
    content.append(format_key_value("samples", num_samples))

    backend_plan = _coerce_mapping(mapping.get("backend_plan"))
    backend_dispatch = _coerce_mapping(mapping.get("backend_dispatch"))
    if backend_plan or backend_dispatch or mapping.get("execution_backend"):
        backend = (
            mapping.get("execution_backend")
            or (backend_dispatch or {}).get("execution_backend")
            or (backend_plan or {}).get("recommended_backend")
            or (backend_plan or {}).get("selected_backend")
            or "unknown"
        )
        device = (backend_plan or {}).get("runtime_device") or mapping.get("runtime_device") or "unknown"
        content.append(format_key_value("backend", f"{backend} | device={device}"))

    job_execution = _coerce_mapping(mapping.get("job_execution"))
    job_audit = _coerce_mapping(job_execution.get("audit")) if job_execution else None
    if job_execution:
        runner_status = (
            job_execution.get("runner_status")
            or job_execution.get("status")
            or (job_audit or {}).get("runner_status")
            or (job_audit or {}).get("status")
        )
        if runner_status:
            content.append(format_key_value("execution", runner_status))

    export_runtime = _coerce_mapping(mapping.get("export_runtime"))
    if export_runtime:
        required = export_runtime.get("required", False)
        format_type = export_runtime.get("target_artifact_format", "unknown")
        if required:
            content.append(format_key_value("export", f"{MatrixColors.AMBER}REQUIRED{MatrixColors.RESET} | format={format_type}"))
        else:
            content.append(format_key_value("export", f"{MatrixColors.GREEN}NOT REQUIRED{MatrixColors.RESET}"))

    metrics = _coerce_mapping(mapping.get("metrics"))
    if metrics:
        fresh = metrics.get("num_fresh_samples", 0)
        replay = metrics.get("num_replay_samples", 0)
        content.append(format_key_value("metrics", f"fresh={fresh} | replay={replay}"))

    lines.append(draw_box("TRAINING RESULT", content))
    lines.append("")

    lines.append(f"{MatrixColors.GREEN}    [✓] Training job completed successfully{MatrixColors.RESET}")
    lines.append("")

    return "\n".join(lines)


__all__ = ["format_train_result_matrix"]
