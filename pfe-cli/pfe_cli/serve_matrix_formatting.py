"""Matrix serve output formatting adapters."""

from __future__ import annotations

from typing import Any

from . import formatters_matrix
from .serve_formatting_deps import ServeFormattingDeps


def format_serve(result: Any) -> str:
    return formatters_matrix.format_serve_matrix(result)


def format_serve_preview(
    *,
    port: int,
    host: str,
    adapter: str,
    workspace: str | None,
    api_key: str | None,
    real_local: bool,
    deps: ServeFormattingDeps,
) -> str:
    """Return a readable preflight summary for serve() without mutating runtime state."""

    cached_state = deps.read_cli_state(workspace)
    recent_training = None
    if cached_state is not None:
        recent_training = deps.coerce_mapping(cached_state.get("recent_training"))
    latest_snapshot = deps.lookup_adapter_snapshot("latest", workspace=workspace)
    latest_training = deps.coerce_mapping(latest_snapshot)
    return formatters_matrix.format_serve_preview_matrix(
        port=port,
        host=host,
        adapter=adapter,
        workspace=workspace,
        api_key=api_key,
        real_local=real_local,
        recent_training=recent_training,
        latest_training=latest_training,
    )


__all__ = ["format_serve", "format_serve_preview"]
