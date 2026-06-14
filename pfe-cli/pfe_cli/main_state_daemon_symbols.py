"""Daemon state symbols for main state compatibility."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .cli_state_helpers import (
    daemon_recovery_payload,
    read_train_queue_daemon_state,
    record_train_queue_daemon_history,
    train_queue_daemon_state_path,
    update_train_queue_daemon_state,
    write_train_queue_daemon_state,
)
from .main_state_common import call


def make_state_daemon_symbols(symbols: dict[str, Any]) -> dict[str, Any]:
    def _train_queue_daemon_state_path(workspace: str | None = None) -> Path:
        return train_queue_daemon_state_path(workspace=workspace, deps=call(symbols, "_cli_state_deps"))

    def _read_train_queue_daemon_state(workspace: str | None = None) -> dict[str, Any] | None:
        return read_train_queue_daemon_state(workspace=workspace, deps=call(symbols, "_cli_state_deps"))

    def _write_train_queue_daemon_state(workspace: str | None, payload: dict[str, Any]) -> None:
        write_train_queue_daemon_state(workspace, payload, deps=call(symbols, "_cli_state_deps"))

    def _record_train_queue_daemon_history(
        *,
        workspace: str | None = None,
        event: str,
        reason: str | None = None,
        note: str | None = None,
    ) -> dict[str, Any]:
        return record_train_queue_daemon_history(
            workspace=workspace,
            event=event,
            reason=reason,
            note=note,
            deps=call(symbols, "_cli_state_deps"),
        )

    def _update_train_queue_daemon_state(
        *,
        workspace: str | None = None,
        desired_state: str,
        event: str,
        reason: str | None = None,
        note: str | None = None,
        observed_state: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return update_train_queue_daemon_state(
            workspace=workspace,
            desired_state=desired_state,
            event=event,
            reason=reason,
            note=note,
            observed_state=observed_state,
            extra=extra,
            deps=call(symbols, "_cli_state_deps"),
        )

    def _daemon_recovery_payload(
        *,
        workspace: str | None = None,
        action: str,
        note: str | None = None,
        reason: str | None = None,
    ) -> dict[str, Any]:
        return daemon_recovery_payload(
            workspace=workspace,
            action=action,
            note=note,
            reason=reason,
            deps=call(symbols, "_cli_state_deps"),
        )

    return {
        "_train_queue_daemon_state_path": _train_queue_daemon_state_path,
        "_read_train_queue_daemon_state": _read_train_queue_daemon_state,
        "_write_train_queue_daemon_state": _write_train_queue_daemon_state,
        "_record_train_queue_daemon_history": _record_train_queue_daemon_history,
        "_update_train_queue_daemon_state": _update_train_queue_daemon_state,
        "_daemon_recovery_payload": _daemon_recovery_payload,
    }


__all__ = ["make_state_daemon_symbols"]
