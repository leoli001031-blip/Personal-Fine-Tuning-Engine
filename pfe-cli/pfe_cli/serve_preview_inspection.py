"""Helpers for extracting serve preview runtime information."""

from __future__ import annotations

from typing import Any

from .serve_formatting_deps import ServeFormattingDeps


def extract_launch_mode(
    preview_mapping: dict[str, Any] | None,
    *,
    deps: ServeFormattingDeps,
) -> str | None:
    runtime = deps.coerce_mapping((preview_mapping or {}).get("runtime")) if preview_mapping else None
    if runtime is None:
        return None
    runner = deps.coerce_mapping(runtime.get("runner"))
    launch_mode = runtime.get("launch_mode")
    if launch_mode is not None:
        return str(launch_mode)
    if runner is not None:
        launch_mode = runner.get("kind")
        if launch_mode is not None:
            return str(launch_mode)
    if runtime.get("dry_run") is True:
        return "dry_run"
    if runtime.get("uvicorn_available") is True:
        return "uvicorn.run"
    return None


def serve_preview_runtime_mapping(
    preview: Any,
    *,
    deps: ServeFormattingDeps,
) -> dict[str, Any] | None:
    preview_mapping = deps.coerce_mapping(preview)
    if preview_mapping is not None:
        runtime = deps.coerce_mapping(preview_mapping.get("runtime"))
        if runtime is not None:
            return runtime

    runtime_attr = getattr(preview, "runtime", None)
    runtime = deps.coerce_mapping(runtime_attr)
    if runtime is not None:
        return runtime
    return None


def serve_preview_launch_mode(
    preview: Any,
    *,
    deps: ServeFormattingDeps,
) -> str | None:
    preview_mapping = deps.coerce_mapping(preview)
    launch_mode = extract_launch_mode(preview_mapping, deps=deps)
    if launch_mode is not None:
        return launch_mode
    runtime_attr = getattr(preview, "runtime", None)
    if runtime_attr is not None:
        launch_mode = getattr(runtime_attr, "launch_mode", None)
        if launch_mode is not None:
            return str(launch_mode)
        runner = getattr(runtime_attr, "runner", None)
        runner_map = deps.coerce_mapping(runner)
        if runner_map is not None:
            kind = runner_map.get("kind")
            if kind is not None:
                return str(kind)
        dry_run = getattr(runtime_attr, "dry_run", None)
        if dry_run is True:
            return "dry_run"
        uvicorn_available = getattr(runtime_attr, "uvicorn_available", None)
        if uvicorn_available is True:
            return "uvicorn.run"
    return None


__all__ = [
    "extract_launch_mode",
    "serve_preview_launch_mode",
    "serve_preview_runtime_mapping",
]
