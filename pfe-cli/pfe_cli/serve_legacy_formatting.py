"""Legacy serve output formatting."""

from __future__ import annotations

from typing import Any

from .serve_formatting_deps import ServeFormattingDeps
from .serve_preview_inspection import serve_preview_launch_mode, serve_preview_runtime_mapping


def format_serve_legacy(result: Any, *, deps: ServeFormattingDeps) -> str:
    """Legacy plain text formatter kept for compatibility checks."""

    mapping = deps.coerce_mapping(result)
    if mapping is None:
        return deps.format_scalar(result)

    for key in ("message", "ready_message", "ready", "detail", "status"):
        value = mapping.get(key)
        if isinstance(value, str) and value.strip():
            return value

    return deps.format_status_legacy(mapping)


def format_serve_preview_legacy(
    *,
    port: int,
    host: str,
    adapter: str,
    workspace: str | None,
    api_key: str | None,
    real_local: bool,
    deps: ServeFormattingDeps,
) -> str:
    """Legacy plain text formatter kept for compatibility checks."""

    preview = deps.optional_module_call(
        "pfe_server.app",
        "build_serve_plan",
        port=port,
        host=host,
        adapter=adapter,
        api_key=api_key,
        workspace=workspace,
        dry_run=True,
    )
    lines = ["PFE serve plan"]
    lines.append(f"request: host={host} | port={port} | adapter={adapter} | workspace={workspace or 'default'}")
    lines.append(f"api key: {'set' if api_key else 'unset'}")
    lines.append(f"real local inference: {'enabled' if real_local else 'disabled'}")
    if preview is not None:
        preview_mapping = deps.coerce_mapping(preview)
        runtime = serve_preview_runtime_mapping(preview, deps=deps)
        if runtime is not None:
            lines.append(
                "runtime: "
                + " | ".join(
                    part
                    for part in (
                        f"provider={deps.format_scalar(runtime.get('provider'))}",
                        f"dry_run={deps.format_scalar(runtime.get('dry_run'))}",
                        f"uvicorn_available={deps.format_scalar(runtime.get('uvicorn_available'))}",
                        f"app_target={deps.format_scalar(runtime.get('app_target'))}",
                    )
                    if part is not None
                )
            )
            launch_mode = serve_preview_launch_mode(preview, deps=deps)
            if launch_mode is not None:
                lines.append(f"server launch_mode: {deps.format_scalar(launch_mode)}")
            command = runtime.get("command")
            if not command and hasattr(preview, "command"):
                try:
                    command = list(getattr(preview, "command"))
                except Exception:
                    command = getattr(preview, "command")
            if command:
                lines.append(f"command: {deps.format_scalar(command)}")
        plan_snapshots = deps.build_plan_snapshots(workspace, {})
        trainer_line = deps.format_trainer_summary(plan_snapshots.get("trainer"))
        if trainer_line is not None:
            lines.append(trainer_line)
        inference_plan = deps.coerce_mapping(plan_snapshots.get("inference"))
        export_plan = deps.coerce_mapping(plan_snapshots.get("export"))
        if inference_plan is not None:
            dispatch_line = deps.format_backend_dispatch(inference_plan)
            if dispatch_line is not None:
                lines.append(dispatch_line)
        if export_plan is not None:
            export_line = deps.format_export_write(export_plan)
            if export_line is not None:
                lines.append(export_line)
        latest_snapshot = deps.lookup_adapter_snapshot("latest", workspace=workspace)
        latest_line = deps.format_adapter_snapshot_line("latest promoted", latest_snapshot, include_latest=True)
        if latest_line is not None:
            lines.append(latest_line)
        cached_state = deps.read_cli_state(workspace)
        recent_snapshot = None
        if cached_state is not None:
            recent_snapshot = deps.coerce_mapping(cached_state.get("recent_training"))
        if recent_snapshot is None:
            recent_snapshot = deps.lookup_recent_adapter_snapshot(workspace=workspace)
        recent_lines = deps.format_recent_training_snapshot(recent_snapshot or cached_state)
        if recent_lines is not None:
            lines.extend(recent_lines)
        if preview_mapping and preview_mapping.get("uvicorn_module"):
            lines.append(f"uvicorn module: {deps.format_scalar(preview_mapping.get('uvicorn_module'))}")
    return "\n".join(lines)


__all__ = ["format_serve_legacy", "format_serve_preview_legacy"]
