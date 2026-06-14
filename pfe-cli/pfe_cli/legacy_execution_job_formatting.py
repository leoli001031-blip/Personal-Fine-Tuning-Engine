"""Legacy job and real execution summary formatting."""

from __future__ import annotations

from typing import Any

from .legacy_result_deps import LegacyResultFormattingDeps


def format_job_execution_summary(
    job_execution: Any,
    *,
    deps: LegacyResultFormattingDeps,
) -> str | None:
    mapping = deps.coerce_mapping(job_execution)
    if mapping is None:
        return None

    parts: list[str] = []
    status = deps.pick_first(mapping, "status")
    executor_mode = deps.pick_first(mapping, "executor_mode")
    metadata = deps.coerce_mapping(mapping.get("metadata"))
    execution_state = deps.pick_first(metadata, "execution_state")
    runner_status = deps.pick_first(deps.coerce_mapping(mapping.get("audit")), "runner_status")

    if status is not None:
        parts.append(f"status={deps.format_scalar(status)}")
    if executor_mode is not None:
        parts.append(f"executor_mode={deps.format_scalar(executor_mode)}")
    if execution_state is not None:
        parts.append(f"execution_state={deps.format_scalar(execution_state)}")
    if runner_status is not None and runner_status != status:
        parts.append(f"runner_status={deps.format_scalar(runner_status)}")
    if not parts:
        return None
    return "job-execution: " + " | ".join(parts)


def format_real_execution_summary(
    job_execution: Any,
    *,
    executor_mode: str | None = None,
    deps: LegacyResultFormattingDeps,
) -> str | None:
    mapping = deps.coerce_mapping(job_execution)
    if mapping is None:
        return None

    parts: list[str] = []
    status = deps.pick_first(mapping, "state", "status")
    kind = deps.pick_first(mapping, "kind")
    executor_mode = deps.pick_first(mapping, "execution_mode", "executor_mode") or executor_mode
    attempted = deps.pick_first(mapping, "attempted")
    success = deps.pick_first(mapping, "success")
    available = deps.pick_first(mapping, "available")
    audit = deps.coerce_mapping(mapping.get("audit"))
    runner_status = deps.pick_first(mapping, "runner_status") or deps.pick_first(audit, "runner_status")

    if status is not None:
        parts.append(f"status={deps.format_scalar(status)}")
    if kind is not None:
        parts.append(f"kind={deps.format_scalar(kind)}")
    if executor_mode is not None:
        parts.append(f"executor_mode={deps.format_scalar(executor_mode)}")
    if attempted is not None:
        parts.append(f"attempted={deps.format_scalar(attempted)}")
    if success is not None:
        parts.append(f"success={deps.format_scalar(success)}")
    if available is not None:
        parts.append(f"available={deps.format_scalar(available)}")
    if runner_status is not None and runner_status != status:
        parts.append(f"runner_status={deps.format_scalar(runner_status)}")
    if not parts:
        return None
    return "real-execution: " + " | ".join(parts)


__all__ = ["format_job_execution_summary", "format_real_execution_summary"]
