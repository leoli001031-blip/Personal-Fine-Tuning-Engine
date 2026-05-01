"""Read and write the general CLI state file."""

from __future__ import annotations

import json
from typing import Any

from .cli_state_deps import CLIStateDeps
from .cli_state_paths import cli_state_path


def read_cli_state(workspace: str | None = None, *, deps: CLIStateDeps) -> dict[str, Any] | None:
    path = cli_state_path(workspace, deps=deps)
    try:
        raw = path.read_text(encoding="utf-8")
    except Exception:
        return None
    try:
        payload = json.loads(raw)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def write_cli_state(workspace: str | None, payload: dict[str, Any], *, deps: CLIStateDeps) -> None:
    path = cli_state_path(workspace, deps=deps)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    except Exception:
        return


__all__ = ["read_cli_state", "write_cli_state"]
