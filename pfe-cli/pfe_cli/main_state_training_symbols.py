"""Training state symbols for main state compatibility."""

from __future__ import annotations

from typing import Any

from .cli_state_helpers import record_train_cli_state
from .main_state_common import call


def make_state_training_symbols(symbols: dict[str, Any]) -> dict[str, Any]:
    def _record_train_cli_state(result: Any, *, workspace: str | None = None) -> None:
        record_train_cli_state(result, workspace=workspace, deps=call(symbols, "_cli_state_deps"))

    return {"_record_train_cli_state": _record_train_cli_state}


__all__ = ["make_state_training_symbols"]
