"""Recent training CLI state helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .cli_state_deps import CLIStateDeps
from .cli_state_user_helpers import write_cli_state


def record_train_cli_state(result: Any, *, workspace: str | None = None, deps: CLIStateDeps) -> None:
    mapping = deps.coerce_mapping(result)
    if mapping is None:
        return

    payload = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "workspace": workspace or "default",
        "recent_training": {
            "version": mapping.get("version"),
            "state": "pending_eval",
            "execution_backend": mapping.get("execution_backend"),
            "executor_mode": deps.pick_first(deps.coerce_mapping(mapping.get("executor_spec")), "executor_mode")
            or deps.pick_first(deps.coerce_mapping(mapping.get("backend_dispatch")), "executor_mode")
            or deps.pick_first(deps.coerce_mapping(mapping.get("job_execution")), "executor_mode")
            or "fallback",
            "job_execution": mapping.get("job_execution"),
            "job_execution_summary": mapping.get("job_execution_summary"),
            "real_execution_summary": mapping.get("real_execution_summary"),
            "export_execution": mapping.get("export_execution"),
            "export_toolchain_summary": mapping.get("export_toolchain_summary"),
        },
    }
    write_cli_state(workspace, payload, deps=deps)


__all__ = ["record_train_cli_state"]
