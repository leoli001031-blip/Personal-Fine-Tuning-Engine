"""Filesystem paths for local CLI state."""

from __future__ import annotations

import os
from pathlib import Path

from .cli_state_deps import CLIStateDeps


def pfe_home(workspace: str | None = None, *, deps: CLIStateDeps) -> Path:
    del workspace
    helper = deps.optional_module_call("pfe_core.storage", "resolve_home")
    if isinstance(helper, Path):
        return helper
    override = os.environ.get("PFE_HOME")
    if override:
        return Path(override).expanduser()
    for candidate_root in (Path.cwd(), *Path.cwd().parents):
        candidate = candidate_root / ".pfe"
        if candidate.is_dir():
            return candidate
    return Path.home() / ".pfe"


def cli_state_path(workspace: str | None = None, *, deps: CLIStateDeps) -> Path:
    return pfe_home(workspace=workspace, deps=deps) / "cli_state.json"


__all__ = ["cli_state_path", "pfe_home"]
