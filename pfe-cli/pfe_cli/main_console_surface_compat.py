"""Install legacy console surface helpers onto the main module namespace."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .console_surface import (
    console_dashboard_focus,
    console_help_text,
    console_settings_text,
    console_status_compact_text,
)
from .main_deps import make_console_surface_deps


def _call(symbols: Mapping[str, Any], name: str, *args: Any, **kwargs: Any) -> Any:
    return symbols[name](*args, **kwargs)


def install_console_surface_compat(symbols: dict[str, Any]) -> None:
    def _console_surface_deps() -> Any:
        return make_console_surface_deps(symbols)

    def _console_dashboard_focus(payload: Mapping[str, Any] | None = None) -> str:
        return console_dashboard_focus(payload, deps=_call(symbols, "_console_surface_deps"))

    def _console_settings_text(
        *,
        workspace: str | None,
        mode: str,
        model: str,
        adapter: str,
        temperature: float,
        max_tokens: int | None,
        real_local: bool,
        refresh_seconds: float,
    ) -> str:
        return console_settings_text(
            workspace=workspace,
            mode=mode,
            model=model,
            adapter=adapter,
            temperature=temperature,
            max_tokens=max_tokens,
            real_local=real_local,
            refresh_seconds=refresh_seconds,
            deps=_call(symbols, "_console_surface_deps"),
        )

    def _console_status_compact_text(payload: Mapping[str, Any], *, workspace: str | None = None) -> str:
        return console_status_compact_text(
            payload,
            workspace=workspace,
            deps=_call(symbols, "_console_surface_deps"),
        )

    symbols.update(
        {
            "_console_surface_deps": _console_surface_deps,
            "_console_help_text": console_help_text,
            "_console_dashboard_focus": _console_dashboard_focus,
            "_console_settings_text": _console_settings_text,
            "_console_status_compact_text": _console_status_compact_text,
        }
    )


__all__ = ["install_console_surface_compat"]
