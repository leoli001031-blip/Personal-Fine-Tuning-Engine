"""Serve preview symbols for the main compatibility namespace."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .main_deps import make_serve_formatting_deps
from .main_preview_common import call
from .serve_formatting import (
    extract_launch_mode,
    format_serve_legacy,
    format_serve_preview,
    format_serve_preview_legacy,
    serve_preview_launch_mode,
    serve_preview_runtime_mapping,
)


def make_preview_serve_symbols(symbols: dict[str, Any]) -> dict[str, Any]:
    def _extract_launch_mode(preview_mapping: Mapping[str, Any] | None) -> str | None:
        preview = dict(preview_mapping) if preview_mapping is not None else None
        return extract_launch_mode(preview, deps=call(symbols, "_serve_formatting_deps"))

    def _serve_preview_runtime_mapping(preview: Any) -> dict[str, Any] | None:
        return serve_preview_runtime_mapping(preview, deps=call(symbols, "_serve_formatting_deps"))

    def _serve_preview_launch_mode(preview: Any) -> str | None:
        return serve_preview_launch_mode(preview, deps=call(symbols, "_serve_formatting_deps"))

    def _format_serve_legacy(result: Any) -> str:
        return format_serve_legacy(result, deps=call(symbols, "_serve_formatting_deps"))

    def _serve_formatting_deps() -> Any:
        return make_serve_formatting_deps(symbols)

    def _format_serve_preview(
        *,
        port: int,
        host: str,
        adapter: str,
        workspace: str | None,
        api_key: str | None,
        real_local: bool,
    ) -> str:
        return format_serve_preview(
            port=port,
            host=host,
            adapter=adapter,
            workspace=workspace,
            api_key=api_key,
            real_local=real_local,
            deps=call(symbols, "_serve_formatting_deps"),
        )

    def _format_serve_preview_legacy(
        *,
        port: int,
        host: str,
        adapter: str,
        workspace: str | None,
        api_key: str | None,
        real_local: bool,
    ) -> str:
        return format_serve_preview_legacy(
            port=port,
            host=host,
            adapter=adapter,
            workspace=workspace,
            api_key=api_key,
            real_local=real_local,
            deps=call(symbols, "_serve_formatting_deps"),
        )

    return {
        "_extract_launch_mode": _extract_launch_mode,
        "_serve_preview_runtime_mapping": _serve_preview_runtime_mapping,
        "_serve_preview_launch_mode": _serve_preview_launch_mode,
        "_format_serve_legacy": _format_serve_legacy,
        "_serve_formatting_deps": _serve_formatting_deps,
        "_format_serve_preview": _format_serve_preview,
        "_format_serve_preview_legacy": _format_serve_preview_legacy,
    }


__all__ = ["make_preview_serve_symbols"]
