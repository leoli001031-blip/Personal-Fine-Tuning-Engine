"""Train queue daemon state file persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .cli_state_deps import CLIStateDeps
from .cli_state_paths import pfe_home


def train_queue_daemon_state_path(workspace: str | None = None, *, deps: CLIStateDeps) -> Path:
    workspace_name = str(workspace or "user_default")
    safe_workspace = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in workspace_name
    )
    return pfe_home(workspace=workspace, deps=deps) / "data" / f"train_queue_daemon_{safe_workspace}.json"


def read_train_queue_daemon_state(workspace: str | None = None, *, deps: CLIStateDeps) -> dict[str, Any] | None:
    path = train_queue_daemon_state_path(workspace, deps=deps)
    try:
        raw = path.read_text(encoding="utf-8")
    except Exception:
        return None
    try:
        payload = json.loads(raw)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def write_train_queue_daemon_state(workspace: str | None, payload: dict[str, Any], *, deps: CLIStateDeps) -> None:
    path = train_queue_daemon_state_path(workspace, deps=deps)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    except Exception:
        return


__all__ = [
    "read_train_queue_daemon_state",
    "train_queue_daemon_state_path",
    "write_train_queue_daemon_state",
]
