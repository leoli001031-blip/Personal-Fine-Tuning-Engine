"""Training preview symbols for the main compatibility namespace."""

from __future__ import annotations

from typing import Any

from .main_deps import make_training_preview_deps
from .main_preview_common import call
from .training_preview_formatting import format_train_preview


def make_preview_training_symbols(symbols: dict[str, Any]) -> dict[str, Any]:
    def _training_preview_deps() -> Any:
        return make_training_preview_deps(symbols)

    def _format_train_preview(
        *,
        method: str,
        epochs: int,
        base_model: str | None,
        train_type: str,
        workspace: str | None,
        snapshot_workspace: str | None = None,
        backend_hint: str | None,
        dry_run: bool = False,
        real_local: bool = False,
    ) -> str:
        return format_train_preview(
            method=method,
            epochs=epochs,
            base_model=base_model,
            train_type=train_type,
            workspace=workspace,
            snapshot_workspace=snapshot_workspace,
            backend_hint=backend_hint,
            dry_run=dry_run,
            real_local=real_local,
            deps=call(symbols, "_training_preview_deps"),
        )

    return {
        "_training_preview_deps": _training_preview_deps,
        "_format_train_preview": _format_train_preview,
    }


__all__ = ["make_preview_training_symbols"]
