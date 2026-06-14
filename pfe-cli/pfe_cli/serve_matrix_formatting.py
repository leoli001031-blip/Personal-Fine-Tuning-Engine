"""Matrix serve output formatting adapters."""

from __future__ import annotations

from typing import Any

from . import formatters_matrix
from .serve_formatting_deps import ServeFormattingDeps


def format_serve(result: Any) -> str:
    return formatters_matrix.format_serve_matrix(result)


def _serve_preview_guidance(
    *,
    port: int,
    adapter: str,
    workspace: str | None,
    real_local: bool,
    preview: Any,
    latest_training: dict[str, Any] | None,
    recent_training: dict[str, Any] | None,
    deps: ServeFormattingDeps,
) -> list[str]:
    guidance: list[str] = []
    preview_mapping = deps.coerce_mapping(preview)
    runtime = deps.coerce_mapping(preview_mapping.get("runtime")) if preview_mapping is not None else None
    dry_run = bool(runtime.get("dry_run")) if runtime is not None else True
    command = runtime.get("command") if runtime is not None else None

    if dry_run:
        guidance.append(f"next: preview only; start the server with pfe serve --port {port} --live")
    elif command:
        guidance.append(f"launch command: {deps.format_scalar(command)}")

    if adapter == "latest" and latest_training is None:
        workspace_hint = f" --workspace {workspace}" if workspace else ""
        guidance.append(
            "adapter: no latest promoted snapshot found; "
            f"train/evaluate/promote an adapter or pass --adapter <version>{workspace_hint}"
        )

    recent_backend = (recent_training or {}).get("execution_backend")
    recent_mode = (recent_training or {}).get("executor_mode")
    if recent_backend == "mock_local" or recent_mode in {"fallback", "phase0_mock"}:
        guidance.append(
            "training backend: recent snapshot used mock/fallback execution; "
            "run real local training before expecting personalized local inference"
        )

    if not real_local:
        guidance.append(
            "real local inference: disabled by default; "
            "add --real-local only after doctor resolves a base model"
        )

    return guidance


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
    text = formatters_matrix.format_serve_preview_matrix(
        port=port,
        host=host,
        adapter=adapter,
        workspace=workspace,
        api_key=api_key,
        real_local=real_local,
        recent_training=recent_training,
        latest_training=latest_training,
    )
    guidance = _serve_preview_guidance(
        port=port,
        adapter=adapter,
        workspace=workspace,
        real_local=real_local,
        preview=preview,
        latest_training=latest_training,
        recent_training=recent_training,
        deps=deps,
    )
    if guidance:
        return text + "\n\n" + "\n".join(guidance)
    return text


__all__ = ["format_serve", "format_serve_preview"]
