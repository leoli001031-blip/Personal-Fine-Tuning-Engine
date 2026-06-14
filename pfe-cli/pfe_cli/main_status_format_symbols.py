"""Status formatting symbols for the main compatibility namespace."""

from __future__ import annotations

from typing import Any

from .main_deps import make_status_formatting_deps, make_status_legacy_formatting_deps
from .main_status_common import call
from .status_formatting import format_status
from .status_legacy_formatting import format_status_legacy


def make_status_format_symbols(symbols: dict[str, Any]) -> dict[str, Any]:
    def _status_formatting_deps() -> Any:
        return make_status_formatting_deps(symbols)

    def _format_status(result: Any, *, workspace: str | None = None) -> str:
        return format_status(result, workspace=workspace, deps=call(symbols, "_status_formatting_deps"))

    def _status_legacy_formatting_deps() -> Any:
        return make_status_legacy_formatting_deps(symbols)

    def _format_status_legacy(result: Any, *, workspace: str | None = None) -> str:
        return format_status_legacy(result, workspace=workspace, deps=call(symbols, "_status_legacy_formatting_deps"))

    return {
        "_status_formatting_deps": _status_formatting_deps,
        "_format_status": _format_status,
        "_status_legacy_formatting_deps": _status_legacy_formatting_deps,
        "_format_status_legacy": _format_status_legacy,
    }


__all__ = ["make_status_format_symbols"]
