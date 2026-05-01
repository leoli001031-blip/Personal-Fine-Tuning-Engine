"""Console settings text rendering."""

from __future__ import annotations

from .console_surface_deps import ConsoleSurfaceDeps


def console_settings_text(
    *,
    workspace: str | None,
    mode: str,
    model: str,
    adapter: str,
    temperature: float,
    max_tokens: int | None,
    real_local: bool,
    refresh_seconds: float,
    deps: ConsoleSurfaceDeps,
) -> str:
    return "\n".join(
        [
            "PFE console session settings:",
            f"workspace={workspace or 'user_default'}",
            f"mode={mode}",
            f"model={model}",
            f"adapter={adapter}",
            f"temperature={temperature:.2f}",
            f"max_tokens={max_tokens if max_tokens is not None else 'auto'}",
            f"real_local={deps.yes_no(real_local)}",
            f"refresh_seconds={refresh_seconds:.1f}",
        ]
    )


__all__ = ["console_settings_text"]
