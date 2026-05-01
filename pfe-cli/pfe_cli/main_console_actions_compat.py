"""Install legacy console action helpers onto the main module namespace."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .console_actions import console_focus_actions, console_shortcut_hint
from .main_deps import make_console_actions_deps


def _call(symbols: Mapping[str, Any], name: str, *args: Any, **kwargs: Any) -> Any:
    return symbols[name](*args, **kwargs)


def install_console_actions_compat(symbols: dict[str, Any]) -> None:
    def _console_actions_deps() -> Any:
        return make_console_actions_deps(symbols)

    def _console_focus_actions(payload: Mapping[str, Any] | None = None) -> dict[str, str | None]:
        return console_focus_actions(payload, deps=_call(symbols, "_console_actions_deps"))

    def _console_shortcut_hint(mode_name: str, payload: Mapping[str, Any] | None = None) -> str:
        return console_shortcut_hint(mode_name, payload, deps=_call(symbols, "_console_actions_deps"))

    symbols.update(
        {
            "_console_actions_deps": _console_actions_deps,
            "_console_focus_actions": _console_focus_actions,
            "_console_shortcut_hint": _console_shortcut_hint,
        }
    )


__all__ = ["install_console_actions_compat"]
