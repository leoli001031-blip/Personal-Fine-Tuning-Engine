"""Path and root state symbols for main state compatibility."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .cli_state_helpers import cli_state_path, pfe_home, read_cli_state, write_cli_state
from .main_deps import make_cli_state_deps
from .main_state_common import call


def make_state_path_symbols(symbols: dict[str, Any]) -> dict[str, Any]:
    def _cli_state_deps() -> Any:
        return make_cli_state_deps(symbols)

    def _pfe_home(workspace: str | None = None) -> Path:
        return pfe_home(workspace=workspace, deps=call(symbols, "_cli_state_deps"))

    def _cli_state_path(workspace: str | None = None) -> Path:
        return cli_state_path(workspace=workspace, deps=call(symbols, "_cli_state_deps"))

    def _read_cli_state(workspace: str | None = None) -> dict[str, Any] | None:
        return read_cli_state(workspace=workspace, deps=call(symbols, "_cli_state_deps"))

    def _write_cli_state(workspace: str | None, payload: dict[str, Any]) -> None:
        write_cli_state(workspace, payload, deps=call(symbols, "_cli_state_deps"))

    return {
        "_cli_state_deps": _cli_state_deps,
        "_pfe_home": _pfe_home,
        "_cli_state_path": _cli_state_path,
        "_read_cli_state": _read_cli_state,
        "_write_cli_state": _write_cli_state,
    }


__all__ = ["make_state_path_symbols"]
