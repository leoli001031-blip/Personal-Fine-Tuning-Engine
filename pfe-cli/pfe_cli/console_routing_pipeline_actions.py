"""Training, evaluation, and generation console action routing."""

from __future__ import annotations

from .console_routing_context import ConsoleCommandResult, ConsoleRouteContext


def route_console_pipeline_action(ctx: ConsoleRouteContext) -> ConsoleCommandResult | None:
    normalized = ctx.normalized
    deps = ctx.deps
    service = ctx.service
    workspace = ctx.workspace

    if normalized in {"train", "trigger-train"}:
        handler = deps.resolve_handler(service, "retry_auto_train_trigger")
        if handler is None:
            return "Train trigger is unavailable.", "train-unavailable", None
        result = handler(workspace=workspace)
        return deps.format_status(result, workspace=workspace), "train-trigger", None
    if normalized in {"eval", "evaluate"}:
        handler = deps.resolve_handler(service, "evaluate", "eval")
        if handler is None:
            return "Evaluation is unavailable.", "eval-unavailable", None
        result = handler(workspace=workspace)
        return deps.format_eval_result(result, workspace=workspace), "eval-trigger", None
    if normalized in {"distill", "distill run"}:
        handler = deps.resolve_handler(service, "run_distillation", "distill")
        if handler is None:
            return "Distillation is unavailable.", "distill-unavailable", None
        result = handler()
        return deps.format_status(result, workspace=workspace), "distill", None
    if normalized in {"force-recovery", "force recovery"}:
        handler = deps.resolve_handler(service, "force_recovery")
        if handler is None:
            return "Force recovery is unavailable.", "force-recovery-unavailable", None
        result = handler(workspace=workspace, reason="console-request")
        return deps.format_status(result, workspace=workspace), "force-recovery", None
    if normalized.startswith("generate "):
        handler = deps.resolve_handler(service, "generate")
        if handler is None:
            return "Generate is unavailable.", "generate-unavailable", None
        parts = normalized.split(None, 2)
        if len(parts) < 2:
            return "Usage: /generate <scenario> [style]", "generate", None
        scenario = parts[1]
        style = parts[2] if len(parts) > 2 else "default"
        result = handler(scenario=scenario, style=style, num_samples=10, workspace=workspace)
        return deps.format_status(result, workspace=workspace), "generate", None
    if normalized in {"dpo", "dpo train"}:
        handler = deps.resolve_handler(service, "train_dpo")
        if handler is None:
            return "DPO training is unavailable.", "dpo-unavailable", None
        result = handler(workspace=workspace)
        return deps.format_train_result(result, workspace=workspace or "user_default"), "dpo", None

    return None


__all__ = ["route_console_pipeline_action"]
