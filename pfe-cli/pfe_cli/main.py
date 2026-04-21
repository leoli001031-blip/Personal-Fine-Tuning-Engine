"""Typer entrypoint for the PFE CLI."""

from __future__ import annotations

import importlib
import inspect
import json
import os
import select
import threading
import time
import sys
from uuid import uuid4
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable, Optional

import typer

from .adapter_commands import adapter_app, _format_lifecycle_summary
from . import formatters_matrix
from .terminal_theme import MatrixColors


@dataclass(frozen=True)
class CLIContext:
    """Common CLI settings shared across commands."""

    workspace: str | None = None
    config_path: str | None = None


app = typer.Typer(
    help=(
        "PFE command line interface. Default mode is strict_local. "
        "OpenAI compatibility covers inference only; personalized loops need Signal SDK or /pfe/signal. "
        "Current capability boundary: train/eval/serve are the core loop, while generate/distill/profile/route are still rule-based or bootstrap-oriented surfaces."
    ),
    add_completion=False,
    no_args_is_help=True,
)
app.add_typer(adapter_app, name="adapter")
trigger_app = typer.Typer(help="Manage auto-train trigger state and manual recovery.")
app.add_typer(trigger_app, name="trigger")
daemon_app = typer.Typer(help="Manage the background train queue daemon lifecycle.")
app.add_typer(daemon_app, name="daemon")
candidate_app = typer.Typer(help="Manage the current candidate adapter lifecycle.")

eval_trigger_app = typer.Typer(help="Manage auto-evaluation trigger after training.")
app.add_typer(eval_trigger_app, name="eval-trigger")
app.add_typer(candidate_app, name="candidate")
collect_app = typer.Typer(help="Manage signal collection state.")
app.add_typer(collect_app, name="collect")


def _load_service(*module_names: str) -> Any | None:
    """Resolve a future high-level service object if the mainline has provided it."""

    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue

        for attr_name in (
            "service",
            "pipeline",
            "trainer",
            "evaluator",
            "server",
            "inference",
            "generate",
            "distill",
            "train",
            "evaluate",
            "serve",
            "status",
        ):
            candidate = getattr(module, attr_name, None)
            if candidate is not None:
                return candidate
    return None


def _friendly_exception_message(exc: Exception) -> str | None:
    """Return a concise message for known domain errors."""

    name = exc.__class__.__name__.lower()
    if "trainingerror" in name:
        return f"Training failed: {exc}"
    if "adaptererror" in name:
        return f"Adapter error: {exc}"
    if "evaluationerror" in name:
        return f"Evaluation failed: {exc}"
    if "servererror" in name:
        return f"Server error: {exc}"
    if "pipelineerror" in name:
        return f"Pipeline error: {exc}"
    return None


def _coerce_mapping(result: Any) -> dict[str, Any] | None:
    """Convert pydantic-style results into a plain mapping when possible."""

    if isinstance(result, dict):
        return dict(result)
    if is_dataclass(result) and not isinstance(result, type):
        try:
            dumped = asdict(result)
        except Exception:
            return None
        if isinstance(dumped, dict):
            return dict(dumped)
    model_dump = getattr(result, "model_dump", None)
    if callable(model_dump):
        try:
            dumped = model_dump()
        except Exception:
            return None
        if isinstance(dumped, dict):
            return dict(dumped)
    to_dict = getattr(result, "dict", None)
    if callable(to_dict):
        try:
            dumped = to_dict()
        except Exception:
            return None
        if isinstance(dumped, dict):
            return dict(dumped)
    return None


def _coerce_sequence_of_mappings(result: Any) -> list[dict[str, Any]]:
    if not isinstance(result, Sequence) or isinstance(result, (str, bytes, bytearray)):
        return []
    items: list[dict[str, Any]] = []
    for item in result:
        mapping = _coerce_mapping(item)
        if mapping is not None:
            items.append(mapping)
    return items


def _coerce_sequence_of_scalars(result: Any) -> list[str]:
    if not isinstance(result, Sequence) or isinstance(result, (str, bytes, bytearray)):
        return []
    items: list[str] = []
    for item in result:
        if item is None:
            continue
        items.append(str(item))
    return items


def _ordered_eval_scores(scores: Mapping[str, Any]) -> list[tuple[str, Any]]:
    """Return eval scores with personalization metrics highlighted first."""
    preferred_order = (
        "style_preference_hit_rate",
        "style_match",
        "preference_alignment",
        "quality_preservation",
        "personality_consistency",
    )
    ordered: list[tuple[str, Any]] = []
    seen: set[str] = set()
    for key in preferred_order:
        if key in scores:
            ordered.append((key, scores[key]))
            seen.add(key)
    for key, value in scores.items():
        if key in seen:
            continue
        ordered.append((key, value))
    return ordered


_GENERIC_MONITOR_FOCUSES = {
    "candidate_idle",
    "queue_waiting_execution",
    "queue_backlog",
    "runner_active",
    "daemon_active",
    "candidate_monitoring",
    "queue_monitoring",
    "runner_monitoring",
    "daemon_monitoring",
}


def _prefer_inspection_summary_for_generic_monitor(
    *,
    focus: Any,
    summary_line: Any,
    inspection_summary_line: Any,
) -> tuple[Any, Any]:
    focus_text = str(focus or "").strip().lower()
    if inspection_summary_line and focus_text in _GENERIC_MONITOR_FOCUSES:
        return inspection_summary_line, inspection_summary_line
    return summary_line, inspection_summary_line


def _optional_module_call(module_name: str, attr_name: str, *args: Any, **kwargs: Any) -> Any | None:
    """Call a helper from an optional module without hard-failing CLI formatting."""

    try:
        module = importlib.import_module(module_name)
    except Exception:
        return None
    candidate = getattr(module, attr_name, None)
    if not callable(candidate):
        return None
    try:
        return candidate(*args, **kwargs)
    except Exception:
        return None


def _pfe_home(workspace: str | None = None) -> Path:
    del workspace
    helper = _optional_module_call("pfe_core.storage", "resolve_home")
    if isinstance(helper, Path):
        return helper
    override = os.environ.get("PFE_HOME")
    if override:
        return Path(override).expanduser()
    for candidate_root in (Path.cwd(), *Path.cwd().parents):
        candidate = candidate_root / ".pfe"
        if candidate.is_dir():
            return candidate
    return Path.home() / ".pfe"


def _cli_state_path(workspace: str | None = None) -> Path:
    return _pfe_home(workspace=workspace) / "cli_state.json"


def _read_cli_state(workspace: str | None = None) -> dict[str, Any] | None:
    path = _cli_state_path(workspace)
    try:
        raw = path.read_text(encoding="utf-8")
    except Exception:
        return None
    try:
        payload = json.loads(raw)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _write_cli_state(workspace: str | None, payload: dict[str, Any]) -> None:
    path = _cli_state_path(workspace)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    except Exception:
        return


def _train_queue_daemon_state_path(workspace: str | None = None) -> Path:
    workspace_name = str(workspace or "user_default")
    safe_workspace = "".join(character if character.isalnum() or character in {"-", "_"} else "_" for character in workspace_name)
    return _pfe_home(workspace=workspace) / "data" / f"train_queue_daemon_{safe_workspace}.json"


def _read_train_queue_daemon_state(workspace: str | None = None) -> dict[str, Any] | None:
    path = _train_queue_daemon_state_path(workspace)
    try:
        raw = path.read_text(encoding="utf-8")
    except Exception:
        return None
    try:
        payload = json.loads(raw)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _write_train_queue_daemon_state(workspace: str | None, payload: dict[str, Any]) -> None:
    path = _train_queue_daemon_state_path(workspace)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    except Exception:
        return


def _record_train_queue_daemon_history(
    *,
    workspace: str | None = None,
    event: str,
    reason: str | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    payload = _read_train_queue_daemon_state(workspace) or {}
    history = list(payload.get("history") or [])
    entry: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": str(event),
    }
    if reason is not None:
        entry["reason"] = str(reason)
    if note is not None:
        entry["note"] = str(note)
    history.append(entry)

    payload.update(
        {
            "workspace": workspace or "user_default",
            "history": history[-20:],
            "history_count": len(history),
            "last_event": str(event),
            "last_reason": str(reason) if reason is not None else payload.get("last_reason"),
            "last_requested_at": entry["timestamp"],
            "last_requested_by": "pfe-cli",
        }
    )
    _write_train_queue_daemon_state(workspace, payload)
    return payload


def _update_train_queue_daemon_state(
    *,
    workspace: str | None = None,
    desired_state: str,
    event: str,
    reason: str | None = None,
    note: str | None = None,
    observed_state: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = _record_train_queue_daemon_history(
        workspace=workspace,
        event=event,
        reason=reason,
        note=note,
    )
    payload.update(
        {
            "desired_state": desired_state,
            "requested_action": event.replace("_requested", ""),
            "command_status": "requested",
            "observed_state": observed_state or payload.get("observed_state") or "unknown",
            "state_path": str(_train_queue_daemon_state_path(workspace)),
            "active": desired_state == "running",
        }
    )
    if extra:
        payload.update(dict(extra))
    _write_train_queue_daemon_state(workspace, payload)
    return payload


def _format_scalar(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (str, int, float)):
        return str(value)
    if isinstance(value, Mapping):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return ", ".join(_format_scalar(item) for item in value)
    return str(value)


def _yes_no(value: Any) -> str:
    return "yes" if bool(value) else "no"


def _plan_summary(plan: Any, fields: Sequence[str]) -> str:
    mapping = _coerce_mapping(plan)
    if mapping is None:
        return _format_scalar(plan)

    parts: list[str] = []
    for field in fields:
        if field in mapping:
            parts.append(f"{field.replace('_', ' ')}={_format_scalar(mapping[field])}")
    if not parts:
        return _format_scalar(mapping)
    return " | ".join(parts)


def _format_plan_block(title: str, plan: Any, fields: Sequence[str]) -> list[str]:
    lines = [f"{title} plan:"]
    mapping = _coerce_mapping(plan)
    if mapping is None:
        lines.append(f"  {_format_scalar(plan)}")
        return lines

    lines.append(f"  {_plan_summary(mapping, fields)}")
    notes = mapping.get("notes")
    if notes:
        lines.append("  notes:")
        for note in notes if isinstance(notes, Sequence) and not isinstance(notes, (str, bytes, bytearray)) else [notes]:
            lines.append(f"    - {_format_scalar(note)}")
    return lines


def _format_trainer_block(trainer: Any) -> list[str]:
    """Render trainer runtime plus per-train-type backend plans."""

    mapping = _coerce_mapping(trainer)
    if mapping is None:
        return ["trainer plan:", f"  {_format_scalar(trainer)}"]

    lines = ["trainer plan:"]
    runtime = _coerce_mapping(mapping.get("runtime"))
    if runtime is not None:
        lines.append(
            "  runtime: "
            + _plan_summary(
                runtime,
                ("runtime_device", "cpu_only", "mps_available", "cuda_available", "platform_name"),
            )
        )

    plans = _coerce_mapping(mapping.get("plans"))
    if plans:
        for name in ("sft", "dpo"):
            if name in plans:
                lines.append(
                    f"  {name}: "
                    + _plan_summary(
                        plans[name],
                        (
                            "selected_backend",
                            "requested_backend",
                            "train_type",
                            "requires_export_step",
                            "export_format",
                            "export_backend",
                            "reason",
                        ),
                    )
                )
        return lines

    lines.append(
        "  "
        + _plan_summary(
            mapping,
            (
                "selected_backend",
                "requested_backend",
                "train_type",
                "requires_export_step",
                "export_format",
                "export_backend",
                "reason",
            ),
        )
    )
    return lines


def _pick_first(mapping: Mapping[str, Any] | None, *keys: str) -> Any:
    if not mapping:
        return None
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _format_compact_plan_line(label: str, plan: Any, fields: Sequence[str]) -> str:
    mapping = _coerce_mapping(plan)
    if mapping is None:
        return f"{label}: {_format_scalar(plan)}"
    parts = [f"{field}={_format_scalar(mapping.get(field))}" for field in fields if mapping.get(field) is not None]
    if not parts:
        return f"{label}: {_format_scalar(mapping)}"
    return f"{label}: " + " | ".join(parts)


def _format_trainer_summary(trainer: Any) -> str | None:
    mapping = _coerce_mapping(trainer)
    if mapping is None:
        return None

    runtime = _coerce_mapping(mapping.get("runtime")) or _coerce_mapping(mapping.get("runtime_summary"))
    plans = _coerce_mapping(mapping.get("plans"))

    recommended_backend = _pick_first(mapping, "recommended_backend", "selected_backend")
    requires_export_step = _pick_first(mapping, "requires_export_step")
    if plans:
        for name in ("sft", "dpo"):
            subplan = _coerce_mapping(plans.get(name))
            if subplan is None:
                continue
            if recommended_backend is None:
                recommended_backend = _pick_first(subplan, "recommended_backend", "selected_backend")
            if requires_export_step is None:
                requires_export_step = _pick_first(subplan, "requires_export_step")
            if recommended_backend is not None and requires_export_step is not None:
                break

    runtime_device = _pick_first(runtime, "runtime_device")
    if recommended_backend is None and runtime_device is None and requires_export_step is None:
        return None

    parts = []
    if recommended_backend is not None:
        parts.append(f"recommended_backend={_format_scalar(recommended_backend)}")
    if runtime_device is not None:
        parts.append(f"runtime_device={_format_scalar(runtime_device)}")
    if requires_export_step is not None:
        parts.append(f"requires_export_step={_format_scalar(requires_export_step)}")
    return "trainer: " + " | ".join(parts)


def _format_backend_dispatch(plan: Any) -> str | None:
    mapping = _coerce_mapping(plan)
    if mapping is None:
        return None
    execution_backend = _pick_first(mapping, "selected_backend", "recommended_backend", "execution_backend")
    requested_backend = _pick_first(mapping, "requested_backend")
    reason = str(_pick_first(mapping, "reason") or "").lower()
    execution_mode = _pick_first(mapping, "execution_mode")
    if execution_mode is None:
        if "mock_local" in str(execution_backend or "").lower() or any(
            token in reason for token in ("fallback", "auto-selected", "dry-run")
        ):
            execution_mode = "fallback"
        else:
            execution_mode = "real"
    runtime_device = _pick_first(mapping, "runtime_device", "preferred_device")
    requires_export_step = _pick_first(mapping, "requires_export_step")
    required_artifact_format = _pick_first(mapping, "required_artifact_format", "export_format")

    parts = []
    if execution_backend is not None:
        parts.append(f"execution_backend={_format_scalar(execution_backend)}")
    if execution_mode is not None:
        parts.append(f"execution_mode={_format_scalar(execution_mode)}")
    if runtime_device is not None:
        parts.append(f"runtime_device={_format_scalar(runtime_device)}")
    if requires_export_step is not None:
        parts.append(f"requires_export_step={_format_scalar(requires_export_step)}")
    if required_artifact_format is not None:
        parts.append(f"required_artifact_format={_format_scalar(required_artifact_format)}")
    if requested_backend is not None and execution_backend != requested_backend:
        parts.append(f"requested_backend={_format_scalar(requested_backend)}")
    return "backend-dispatch: " + " | ".join(parts) if parts else None


def _format_export_write(plan: Any) -> str | None:
    mapping = _coerce_mapping(plan)
    if mapping is None:
        return None
    target_artifact_format = _pick_first(mapping, "target_artifact_format", "artifact_format")
    required = _pick_first(mapping, "required")
    if required is None:
        required = str(target_artifact_format).lower() == "gguf_merged"
    gguf_export = "required" if str(target_artifact_format).lower() == "gguf_merged" or bool(required) else "not_required"
    parts = [f"gguf_export={gguf_export}"]
    if target_artifact_format is not None:
        parts.append(f"target_artifact_format={_format_scalar(target_artifact_format)}")
    execution_status = _pick_first(mapping, "status", "execution_status")
    if execution_status is None:
        audit = _coerce_mapping(mapping.get("audit"))
        execution_status = _pick_first(audit, "status")
    dry_run = _pick_first(mapping, "dry_run")
    if dry_run is not None:
        parts.append(f"dry_run={_format_scalar(dry_run)}")
    output_dir = _pick_first(mapping, "output_dir")
    if output_dir is not None:
        parts.append(f"output_dir={_format_scalar(output_dir)}")
    artifact_name = _pick_first(mapping, "artifact_name")
    if artifact_name is not None:
        parts.append(f"artifact_name={_format_scalar(artifact_name)}")
    output_artifact_path = _pick_first(mapping, "output_artifact_path", "artifact_path")
    if output_artifact_path is not None:
        parts.append(f"artifact_path={_format_scalar(output_artifact_path)}")
    command = _pick_first(mapping, "command")
    if command is not None:
        parts.append(f"command={_format_scalar(command)}")
    write_state = _pick_first(mapping, "write_state")
    if write_state is None:
        metadata = _coerce_mapping(mapping.get("metadata"))
        write_state = _pick_first(metadata, "write_state")
    if write_state is None:
        write_state = "planned"
    if execution_status is not None and write_state in {"planned", "ready"}:
        if execution_status in {"success", "dry_run", "tool_missing", "failed", "not_required"}:
            write_state = str(execution_status)
    if command is not None or output_dir is not None:
        write_state = "ready" if write_state == "planned" else write_state
    if dry_run is False:
        write_state = "executing" if write_state in {"planned", "ready"} else write_state
    parts.insert(1, f"write_state={write_state}")
    if execution_status is not None:
        parts.append(f"execution_status={_format_scalar(execution_status)}")
    return "export-write: " + " | ".join(parts)


def _format_train_result(result: Any, *, workspace: str | None = None) -> str:
    # Matrix theme - default style
    return formatters_matrix.format_train_result_matrix(result, workspace=workspace)

def _format_train_result_legacy(result: Any, *, workspace: str | None = None) -> str:
    """Legacy plain text formatter (kept for reference)."""
    mapping = _coerce_mapping(result)
    if mapping is None:
        return _format_scalar(result)

    lines = ["PFE train"]
    version = _pick_first(mapping, "version")
    adapter_path = _pick_first(mapping, "adapter_path")
    num_samples = _pick_first(mapping, "num_samples")
    if version is not None or adapter_path is not None or num_samples is not None:
        parts = []
        if version is not None:
            parts.append(f"version={_format_scalar(version)}")
        if adapter_path is not None:
            parts.append(f"adapter_path={_format_scalar(adapter_path)}")
        if num_samples is not None:
            parts.append(f"num_samples={_format_scalar(num_samples)}")
        lines.append(" | ".join(parts))

    incremental_line = _format_incremental_context(mapping.get("incremental_context") or mapping)
    if incremental_line is not None:
        lines.append(incremental_line)

    training_snapshot = _lookup_adapter_snapshot(str(version) if version is not None else None, workspace=workspace)
    training_line = _format_adapter_snapshot_line("recent training adapter", training_snapshot, include_latest=True)
    if training_line is not None:
        lines.append(training_line)

    latest_snapshot = _lookup_adapter_snapshot("latest", workspace=workspace)
    latest_line = _format_adapter_snapshot_line("latest promoted", latest_snapshot, include_latest=True)
    if latest_line is not None:
        lines.append(latest_line)

    backend_dispatch = mapping.get("backend_plan") or mapping.get("backend_dispatch")
    job_execution = mapping.get("job_execution")
    if job_execution is not None:
        job_line = _format_job_execution_summary(job_execution)
        if job_line is not None:
            lines.append(job_line)
    export_execution = mapping.get("export_execution")
    if export_execution is not None:
        export_exec_line = _format_export_execution_summary(export_execution)
        if export_exec_line is not None:
            lines.append(export_exec_line)
    export_write = (
        mapping.get("export_write")
        or mapping.get("export_command_plan")
        or mapping.get("export_execution")
        or mapping.get("export_runtime")
    )
    if backend_dispatch is not None:
        dispatch_line = _format_backend_dispatch(backend_dispatch)
        if dispatch_line is not None:
            lines.append(dispatch_line)
    if export_write is not None:
        export_line = _format_export_write(export_write)
        if export_line is not None:
            lines.append(export_line)

    metrics = _coerce_mapping(mapping.get("metrics"))
    if metrics is not None:
        outcome = []
        if "num_fresh_samples" in metrics:
            outcome.append(f"fresh_samples={_format_scalar(metrics.get('num_fresh_samples'))}")
        if "num_replay_samples" in metrics:
            outcome.append(f"replay_samples={_format_scalar(metrics.get('num_replay_samples'))}")
        if "requires_export_step" in metrics:
            outcome.append(f"requires_export_step={_format_scalar(metrics.get('requires_export_step'))}")
        if outcome:
            lines.append("metrics: " + " | ".join(outcome))
    return "\n".join(lines)


def _extract_launch_mode(preview_mapping: Mapping[str, Any] | None) -> str | None:
    runtime = _coerce_mapping((preview_mapping or {}).get("runtime")) if preview_mapping else None
    if runtime is None:
        return None
    runner = _coerce_mapping(runtime.get("runner"))
    launch_mode = _pick_first(runtime, "launch_mode")
    if launch_mode is not None:
        return str(launch_mode)
    if runner is not None:
        launch_mode = _pick_first(runner, "kind")
        if launch_mode is not None:
            return str(launch_mode)
    if runtime.get("dry_run") is True:
        return "dry_run"
    if runtime.get("uvicorn_available") is True:
        return "uvicorn.run"
    return None


def _serve_preview_runtime_mapping(preview: Any) -> dict[str, Any] | None:
    preview_mapping = _coerce_mapping(preview)
    if preview_mapping is not None:
        runtime = _coerce_mapping(preview_mapping.get("runtime"))
        if runtime is not None:
            return runtime

    runtime_attr = getattr(preview, "runtime", None)
    runtime = _coerce_mapping(runtime_attr)
    if runtime is not None:
        return runtime
    return None


def _serve_preview_launch_mode(preview: Any) -> str | None:
    preview_mapping = _coerce_mapping(preview)
    launch_mode = _extract_launch_mode(preview_mapping)
    if launch_mode is not None:
        return launch_mode
    runtime_attr = getattr(preview, "runtime", None)
    if runtime_attr is not None:
        launch_mode = getattr(runtime_attr, "launch_mode", None)
        if launch_mode is not None:
            return str(launch_mode)
        runner = getattr(runtime_attr, "runner", None)
        runner_map = _coerce_mapping(runner)
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


def _load_latest_adapter_manifest(workspace: str | None) -> dict[str, Any] | None:
    """Read the latest adapter manifest for a workspace when the store is available."""

    try:
        module = importlib.import_module("pfe_core.adapter_store.store")
    except Exception:
        return None

    store_cls = getattr(module, "AdapterStore", None)
    if store_cls is None:
        return None
    try:
        store = store_cls(workspace=workspace or "user_default")
        latest_version = store.current_latest_version()
        if not latest_version:
            return None
        read_manifest = getattr(store, "_read_manifest", None)
        if not callable(read_manifest):
            return None
        manifest = read_manifest(latest_version)
        return manifest if isinstance(manifest, dict) else None
    except Exception:
        return None


def _build_plan_snapshots(workspace: str | None, status_mapping: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Assemble trainer/inference/export plan snapshots from local helpers."""

    manifest = _load_latest_adapter_manifest(workspace)
    status_mapping = dict(status_mapping or {})
    metadata = _coerce_mapping(status_mapping.get("metadata")) or {}
    inference_status = _coerce_mapping(metadata.get("inference")) or {}
    pipeline_status = _coerce_mapping(metadata.get("pipeline")) or {}

    train_type = "sft"
    if manifest:
        train_type = str((manifest.get("training_config") or {}).get("train_type", train_type))

    requested_backend = None
    if inference_status:
        requested_backend = inference_status.get("requested_backend")
    if requested_backend is None and manifest is not None:
        requested_backend = manifest.get("inference_backend") or (manifest.get("training_config") or {}).get("backend")
    requested_backend = requested_backend or "auto"

    artifact_format = None
    if manifest is not None:
        artifact_format = manifest.get("artifact_format")

    trainer_plan = _optional_module_call(
        "pfe_core.trainer.runtime",
        "summarize_trainer_backend_plan",
        train_type=train_type,
        target_inference_backend=requested_backend,
    )
    if trainer_plan is None:
        trainer_plan = {
            "selected_backend": "n/a",
            "requested_backend": requested_backend,
            "train_type": train_type,
        }

    inference_plan = inference_status.get("backend_plan") or _optional_module_call(
        "pfe_core.inference.backends",
        "summarize_backend_plan",
        requested_backend=requested_backend,
        artifact_format=artifact_format,
        manifest=manifest,
    )
    if inference_plan is None:
        inference_plan = {
            "selected_backend": requested_backend,
            "requested_backend": requested_backend,
            "requires_export": False,
        }

    selected_backend = None
    if isinstance(inference_plan, Mapping):
        selected_backend = inference_plan.get("selected_backend")
    selected_backend = selected_backend or requested_backend

    export_plan = inference_status.get("export_plan") or _optional_module_call(
        "pfe_core.inference.export",
        "plan_export",
        target_backend=selected_backend,
        source_artifact_format=artifact_format,
        workspace=workspace or status_mapping.get("home"),
        adapter_dir=(manifest or {}).get("adapter_dir"),
        source_adapter_version=(manifest or {}).get("version"),
        source_model=(manifest or {}).get("base_model"),
        training_run_id=(manifest or {}).get("training_run_id"),
        num_samples=(manifest or {}).get("num_samples"),
    )
    if export_plan is None:
        export_plan = {
            "target_backend": selected_backend,
            "target_artifact_format": artifact_format,
            "required": False,
        }

    result = {
        "trainer": trainer_plan,
        "inference": inference_plan,
        "export": export_plan,
    }
    if pipeline_status:
        result["pipeline"] = pipeline_status
    return result


def _format_plan_snapshot_lines(plan_snapshots: Mapping[str, Any]) -> list[str]:
    lines: list[str] = []
    if "trainer" in plan_snapshots:
        lines.extend(
            _format_plan_block(
                "trainer",
                plan_snapshots["trainer"],
                ("selected_backend", "requested_backend", "train_type", "requires_export_step", "export_format", "export_backend"),
            )
        )
    if "inference" in plan_snapshots:
        lines.extend(
            _format_plan_block(
                "inference",
                plan_snapshots["inference"],
                ("selected_backend", "requested_backend", "requires_export", "required_artifact_format", "preferred_device", "reason"),
            )
        )
    if "export" in plan_snapshots:
        lines.extend(
            _format_plan_block(
                "export",
                plan_snapshots["export"],
                ("target_backend", "target_artifact_format", "required", "artifact_name", "artifact_directory", "reason"),
            )
        )
    return lines


def _format_status(result: Any, *, workspace: str | None = None) -> str:
    # Matrix theme - default style
    mapping = _coerce_mapping(result)
    if mapping is not None:
        cached_state = _read_cli_state(workspace or mapping.get("workspace") or mapping.get("home"))
        if cached_state is not None:
            recent_training = _coerce_mapping(cached_state.get("recent_training"))
            if recent_training is not None:
                mapping = dict(mapping)
                mapping["recent_training_snapshot"] = recent_training
                recent_adapter = _coerce_mapping(mapping.get("recent_adapter")) or {}
                recent_adapter = dict(recent_adapter)
                for key in ("execution_backend", "executor_mode"):
                    if key in recent_training and recent_adapter.get(key) is None:
                        recent_adapter[key] = recent_training[key]
                mapping["recent_adapter"] = recent_adapter
                for key in ("real_execution_summary", "export_toolchain_summary", "job_execution", "export_execution"):
                    if key in recent_training and mapping.get(key) is None:
                        mapping[key] = recent_training[key]
    return formatters_matrix.format_status_matrix(mapping or result, workspace=workspace)

def _format_status_legacy(result: Any, *, workspace: str | None = None) -> str:
    """Legacy plain text formatter (kept for reference)."""
    mapping = _coerce_mapping(result)
    if mapping is None:
        return _format_scalar(result)

    lines: list[str] = ["PFE status"]
    latest_adapter_version = mapping.pop("latest_adapter_version", None)
    latest_adapter = mapping.pop("latest_adapter", None)
    latest_adapter_map = _coerce_mapping(latest_adapter)
    if latest_adapter_version is None and latest_adapter_map is not None:
        latest_adapter_version = latest_adapter_map.get("version")
    latest_adapter_state = None
    if latest_adapter_map is not None:
        latest_adapter_state = latest_adapter_map.get("state")

    recent_adapter_version = mapping.pop("recent_adapter_version", None)
    recent_adapter = mapping.pop("recent_adapter", None)
    recent_adapter_map = _coerce_mapping(recent_adapter)
    if recent_adapter_version is None and recent_adapter_map is not None:
        recent_adapter_version = recent_adapter_map.get("version")
    recent_adapter_state = None
    if recent_adapter_map is not None:
        recent_adapter_state = recent_adapter_map.get("state")

    lifecycle = _coerce_mapping(mapping.pop("adapter_lifecycle", None))
    candidate_summary = _coerce_mapping(mapping.pop("candidate_summary", None))
    compare_evaluation = _coerce_mapping(mapping.pop("compare_evaluation", None))
    candidate_history = _coerce_mapping(mapping.pop("candidate_history", None))
    candidate_timeline = _coerce_mapping(mapping.pop("candidate_timeline", None))
    operations_console = _coerce_mapping(mapping.pop("operations_console", None))
    daemon_timeline = _coerce_mapping(mapping.pop("daemon_timeline", None))
    if daemon_timeline is None and operations_console is not None:
        daemon_timeline = _coerce_mapping(operations_console.get("daemon_timeline"))
    runner_timeline = _coerce_mapping(mapping.pop("runner_timeline", None))
    if runner_timeline is None and operations_console is not None:
        runner_timeline = _coerce_mapping(operations_console.get("runner_timeline"))
    train_queue = _coerce_mapping(mapping.pop("train_queue", None))
    operations_overview = _coerce_mapping(mapping.pop("operations_overview", None))
    operations_alerts = _coerce_sequence_of_mappings(mapping.pop("operations_alerts", None))
    operations_health = _coerce_mapping(mapping.pop("operations_health", None))
    operations_recovery = _coerce_mapping(mapping.pop("operations_recovery", None))
    operations_next_actions = _coerce_sequence_of_scalars(mapping.pop("operations_next_actions", None))
    operations_dashboard = _coerce_mapping(mapping.pop("operations_dashboard", None))
    operations_alert_policy = _coerce_mapping(mapping.pop("operations_alert_policy", None))
    operations_event_stream = _coerce_mapping(mapping.pop("operations_event_stream", None))
    operations_timeline = _coerce_mapping(mapping.pop("operations_timeline", None))
    if operations_timeline is None and operations_console is not None:
        operations_timeline = _coerce_mapping(operations_console.get("timelines"))
    if operations_event_stream is None and operations_console is not None:
        operations_event_stream = _coerce_mapping(operations_console.get("event_stream"))

    headline_keys = (
        "home",
        "strict_local",
        "provider",
        "signal_count",
        "adapter_versions",
        "workspace",
    )

    for key in headline_keys:
        if key in mapping:
            lines.append(f"{key.replace('_', ' ')}: {_format_scalar(mapping.pop(key))}")

    latest_parts = []
    if latest_adapter_version is not None:
        latest_parts.append(f"version={_format_scalar(latest_adapter_version)}")
    if latest_adapter_state is not None:
        latest_parts.append(f"state={_format_scalar(latest_adapter_state)}")
    latest_samples = _pick_first(latest_adapter_map, "num_samples", "samples")
    if latest_samples is not None:
        latest_parts.append(f"samples={_format_scalar(latest_samples)}")
    latest_format = _pick_first(latest_adapter_map, "artifact_format", "format")
    if latest_format is not None:
        latest_parts.append(f"format={_format_scalar(latest_format)}")
    if latest_parts:
        lines.append("latest promoted: " + " | ".join(latest_parts))
    else:
        lines.append("latest promoted: none")
    latest_export_artifact_line = _format_adapter_export_artifact_line("latest export artifact", latest_adapter_map)
    if latest_export_artifact_line is not None:
        lines.append(latest_export_artifact_line)

    if recent_adapter_version is not None:
        recent_parts = [f"version={_format_scalar(recent_adapter_version)}"]
        if recent_adapter_state is not None:
            recent_parts.append(f"state={_format_scalar(recent_adapter_state)}")
        recent_samples = _pick_first(recent_adapter_map, "num_samples", "samples")
        if recent_samples is not None:
            recent_parts.append(f"samples={_format_scalar(recent_samples)}")
        recent_format = _pick_first(recent_adapter_map, "artifact_format", "format")
        if recent_format is not None:
            recent_parts.append(f"format={_format_scalar(recent_format)}")
        lines.append("recent training: " + " | ".join(recent_parts))
    recent_export_artifact_line = _format_adapter_export_artifact_line("recent export artifact", recent_adapter_map)
    if recent_export_artifact_line is not None:
        lines.append(recent_export_artifact_line)
    if lifecycle is not None:
        counts = _coerce_mapping(lifecycle.get("counts")) or {}
        if counts:
            ordered_states = ("pending_eval", "promoted", "failed_eval", "archived")
            summary = " | ".join(
                f"{state}={_format_scalar(counts.get(state, 0))}"
                for state in ordered_states
                if state in counts
            )
            if summary:
                lines.append(f"lifecycle: {summary}")
    if candidate_summary is not None:
        candidate_parts: list[str] = []
        for key in (
            "candidate_version",
            "candidate_state",
            "candidate_can_promote",
            "candidate_can_archive",
            "pending_eval_count",
            "training_count",
            "failed_eval_count",
            "candidate_needs_promotion",
            "promotion_gate_status",
            "promotion_gate_reason",
            "promotion_gate_action",
            "promotion_compare_comparison",
            "promotion_compare_recommendation",
            "promotion_compare_winner",
            "promotion_compare_left_adapter",
            "promotion_compare_right_adapter",
            "promotion_compare_overall_delta",
            "promotion_compare_details_count",
            "promotion_compare_personalization_delta",
            "promotion_compare_quality_delta",
            "promotion_compare_style_preference_hit_rate_delta",
            "promotion_compare_personalization_summary",
            "promotion_compare_quality_summary",
            "promotion_compare_summary_line",
        ):
            value = candidate_summary.get(key)
            if value is not None:
                candidate_parts.append(f"{key}={_format_scalar(value)}")
        if candidate_parts:
            lines.append("candidate summary: " + " | ".join(candidate_parts))
    compare_line = _format_compare_evaluation(compare_evaluation)
    if compare_line is not None:
        lines.append(compare_line)
    candidate_action = _coerce_mapping(mapping.pop("candidate_action", None))
    if candidate_action is not None:
        action_parts: list[str] = []
        for key in ("action", "status", "reason", "required_action", "operator_note", "candidate_version", "promoted_version", "archived_version"):
            value = candidate_action.get(key)
            if value is not None:
                action_parts.append(f"{key}={_format_scalar(value)}")
        if action_parts:
            lines.append("candidate action: " + " | ".join(action_parts))
    candidate_history = _coerce_mapping(mapping.pop("candidate_history", None))
    if candidate_history is not None:
        history_parts: list[str] = []
        for key in ("count", "last_action", "last_status", "last_reason", "last_note", "last_candidate_version"):
            value = candidate_history.get(key)
            if value is not None:
                history_parts.append(f"{key}={_format_scalar(value)}")
        if history_parts:
            lines.append("candidate history: " + " | ".join(history_parts))
    if operations_overview is not None:
        overview_parts: list[str] = []
        overview_focus = None
        for candidate in (
            operations_overview.get("current_focus"),
            operations_overview.get("monitor_focus"),
            _coerce_mapping(operations_dashboard).get("monitor_focus") if operations_dashboard is not None else None,
            operations_overview.get("attention_reason"),
        ):
            if candidate is None:
                continue
            if str(candidate).strip().lower() in {"", "none", "idle", "stable"}:
                continue
            overview_focus = candidate
            break
        overview_required_action = (
            operations_overview.get("required_action")
            or (_coerce_mapping(operations_alert_policy).get("required_action") if operations_alert_policy is not None else None)
            or (_coerce_mapping(operations_dashboard).get("required_action") if operations_dashboard is not None else None)
        )
        for key in (
            "attention_needed",
            "attention_reason",
            "trigger_state",
            "trigger_ready",
            "candidate_version",
            "candidate_state",
            "candidate_needs_promotion",
            "queue_count",
            "awaiting_confirmation_count",
            "runner_active",
            "runner_lock_state",
            "runner_last_event",
        ):
            value = operations_overview.get(key)
            if value is not None:
                overview_parts.append(f"{key}={_format_scalar(value)}")
        if overview_focus is not None:
            overview_parts.append(f"monitor_focus={_format_scalar(overview_focus)}")
        if overview_required_action is not None:
            overview_parts.append(f"required_action={_format_scalar(overview_required_action)}")
        summary_line = operations_overview.get("summary_line")
        inspection_summary_line = operations_overview.get("inspection_summary_line")
        summary_line, inspection_summary_line = _prefer_inspection_summary_for_generic_monitor(
            focus=overview_focus,
            summary_line=summary_line,
            inspection_summary_line=inspection_summary_line,
        )
        if summary_line:
            overview_parts.append(f"summary={_format_scalar(summary_line)}")
        if inspection_summary_line and inspection_summary_line != summary_line:
            overview_parts.append(f"inspection_summary={_format_scalar(inspection_summary_line)}")
        if overview_parts:
            lines.append("operations overview: " + " | ".join(overview_parts))
        auto_train_blocker = _coerce_mapping(operations_overview.get("auto_train_blocker"))
        if auto_train_blocker is not None:
            blocker_parts: list[str] = []
            for key in ("source", "reason", "action", "category", "summary"):
                value = auto_train_blocker.get(key)
                if value is not None:
                    blocker_parts.append(f"{key}={_format_scalar(value)}")
            secondary_reasons = _coerce_sequence_of_scalars(auto_train_blocker.get("secondary_reasons"))
            secondary_actions = _coerce_sequence_of_scalars(auto_train_blocker.get("secondary_actions"))
            if secondary_reasons:
                blocker_parts.append(f"secondary_reasons={_format_scalar(secondary_reasons)}")
            if secondary_actions:
                blocker_parts.append(f"secondary_actions={_format_scalar(secondary_actions)}")
            if blocker_parts:
                lines.append("auto train blocker: " + " | ".join(blocker_parts))
    operations_alert_lines = _format_operations_alert_surface(
        {
            "operations_alerts": operations_alerts,
            "operations_health": operations_health,
            "operations_recovery": operations_recovery,
            "operations_next_actions": operations_next_actions,
            "operations_dashboard": operations_dashboard,
            "operations_alert_policy": operations_alert_policy,
            "operations_console": operations_console,
            "operations_overview": operations_overview,
        }
    )
    if operations_alert_lines is not None:
        lines.extend(operations_alert_lines)
    operations_console_lines = _format_operations_console_digest(
        {
            "operations_console": operations_console,
            "operations_overview": operations_overview,
            "candidate_summary": candidate_summary,
            "candidate_history": candidate_history,
            "candidate_timeline": candidate_timeline,
            "daemon_timeline": daemon_timeline,
            "runner_timeline": runner_timeline,
            "train_queue": train_queue,
        }
    )
    if operations_console_lines is not None:
        lines.extend(operations_console_lines)
    operations_dashboard_lines = _format_operations_dashboard(operations_dashboard) if operations_dashboard is not None else None
    if operations_dashboard_lines is not None:
        lines.extend(operations_dashboard_lines)
    operations_alert_policy_for_display = _coerce_mapping(operations_alert_policy)
    dashboard_monitor_focus = _coerce_mapping(operations_dashboard).get("monitor_focus") if operations_dashboard is not None else None
    if operations_alert_policy_for_display is not None:
        policy_focus = str(operations_alert_policy_for_display.get("current_focus") or "").strip().lower()
        if policy_focus in {"", "none", "idle", "stable"} and dashboard_monitor_focus is not None:
            operations_alert_policy_for_display["current_focus"] = dashboard_monitor_focus
    operations_alert_policy_lines = (
        _format_operations_alert_policy(operations_alert_policy_for_display)
        if operations_alert_policy_for_display is not None
        else None
    )
    if operations_alert_policy_lines is not None:
        lines.extend(operations_alert_policy_lines)
    operations_event_stream_lines = _format_operations_event_stream(operations_event_stream) if operations_event_stream is not None else None
    if operations_event_stream_lines is not None:
        lines.extend(operations_event_stream_lines)
    operations_timeline_lines = _format_operations_timeline(operations_timeline) if operations_timeline is not None else None
    if operations_timeline_lines is not None:
        lines.extend(operations_timeline_lines)
    runner_timeline_lines = _format_runner_timeline_summary(runner_timeline) if runner_timeline is not None else None
    if runner_timeline_lines is not None:
        lines.extend(runner_timeline_lines.splitlines())
    daemon_timeline_lines = _format_daemon_timeline_summary(daemon_timeline) if daemon_timeline is not None else None
    if daemon_timeline_lines is not None:
        lines.extend(daemon_timeline_lines.splitlines())
    if train_queue is not None:
        queue_parts: list[str] = []
        for key in ("count", "max_priority"):
            value = train_queue.get(key)
            if value is not None:
                queue_parts.append(f"{key}={_format_scalar(value)}")
        counts = _coerce_mapping(train_queue.get("counts")) or {}
        if counts:
            queue_parts.append(
                "states="
                + ",".join(f"{name}:{_format_scalar(counts.get(name))}" for name in sorted(counts))
            )
        current_item = _coerce_mapping(train_queue.get("current"))
        if current_item:
            queue_parts.append(
                "current="
                + ",".join(
                    part
                    for part in (
                        _format_scalar(current_item.get("job_id")) if current_item.get("job_id") is not None else "",
                        _format_scalar(current_item.get("state")) if current_item.get("state") is not None else "",
                    )
                    if part
                )
            )
        last_item = _coerce_mapping(train_queue.get("last_item"))
        if last_item:
            queue_parts.append(
                "last="
                + ",".join(
                    part
                    for part in (
                        _format_scalar(last_item.get("job_id")) if last_item.get("job_id") is not None else "",
                        _format_scalar(last_item.get("state")) if last_item.get("state") is not None else "",
                        _format_scalar(last_item.get("adapter_version")) if last_item.get("adapter_version") is not None else "",
                    )
                    if part
                )
            )
        if queue_parts:
            lines.append("train queue: " + " | ".join(queue_parts))
        policy_summary = _coerce_mapping(train_queue.get("policy_summary"))
        if policy_summary:
            policy_parts: list[str] = []
            for key in ("current_priority_source", "current_dedup_scope"):
                value = policy_summary.get(key)
                if value is not None:
                    policy_parts.append(f"{key}={_format_scalar(value)}")
            for key in ("dedup_scopes", "priority_sources"):
                value = policy_summary.get(key)
                if value:
                    policy_parts.append(f"{key}={_format_scalar(value)}")
            if policy_parts:
                lines.append("queue policy: " + " | ".join(policy_parts))
        confirmation_summary = _coerce_mapping(train_queue.get("confirmation_summary"))
        if confirmation_summary:
            confirmation_parts: list[str] = []
            for key in ("confirmation_required_count", "awaiting_confirmation_count", "next_job_id", "next_confirmation_reason"):
                value = confirmation_summary.get(key)
                if value is not None:
                    confirmation_parts.append(f"{key}={_format_scalar(value)}")
            if confirmation_parts:
                lines.append("queue confirmation: " + " | ".join(confirmation_parts))
        review_summary = _coerce_mapping(train_queue.get("review_summary"))
        if review_summary:
            review_parts: list[str] = []
            for key in (
                "reviewed_transition_count",
                "approved_transition_count",
                "rejected_transition_count",
                "last_review_event",
                "last_review_reason",
                "last_review_note",
                "next_job_id",
                "next_confirmation_reason",
            ):
                value = review_summary.get(key)
                if value is not None:
                    review_parts.append(f"{key}={_format_scalar(value)}")
            if review_parts:
                lines.append("queue review: " + " | ".join(review_parts))
        review_policy_summary = _coerce_mapping(train_queue.get("review_policy_summary"))
        if review_policy_summary:
            review_policy_parts: list[str] = []
            for key in (
                "review_mode",
                "queue_entry_mode",
                "review_required_by_policy",
                "review_required_now",
                "next_action",
                "review_reason",
            ):
                value = review_policy_summary.get(key)
                if value is not None:
                    review_policy_parts.append(f"{key}={_format_scalar(value)}")
            if review_policy_parts:
                lines.append("queue review policy: " + " | ".join(review_policy_parts))
        worker_runner = _coerce_mapping(train_queue.get("worker_runner"))
        if worker_runner:
            worker_parts: list[str] = []
            for key in (
                "active",
                "lock_state",
                "stop_requested",
                "processed_count",
                "failed_count",
                "loop_cycles",
                "stopped_reason",
                "max_seconds",
                "idle_sleep_seconds",
                "stale_after_seconds",
                "lease_expires_at",
            ):
                value = worker_runner.get(key)
                if value is not None:
                    worker_parts.append(f"{key}={_format_scalar(value)}")
            if worker_parts:
                lines.append("queue worker runner: " + " | ".join(worker_parts))
    daemon_state = _coerce_mapping(mapping.pop("daemon", None))
    if daemon_state is None and isinstance(train_queue, Mapping):
        daemon_state = _coerce_mapping(train_queue.get("daemon"))
    if daemon_state is None:
        daemon_state = _read_train_queue_daemon_state(workspace)
    if daemon_state is not None:
        daemon_parts: list[str] = []
        for key in ("desired_state", "requested_action", "command_status", "active", "observed_state", "lock_state"):
            value = daemon_state.get(key)
            if value is not None:
                daemon_parts.append(f"{key}={_format_scalar(value)}")
        state_path = daemon_state.get("state_path")
        if state_path is not None:
            daemon_parts.append(f"state_path={_format_scalar(state_path)}")
        history_count = daemon_state.get("history_count")
        if history_count is not None:
            daemon_parts.append(f"history_count={_format_scalar(history_count)}")
        last_requested_at = daemon_state.get("last_requested_at")
        if last_requested_at is not None:
            daemon_parts.append(f"last_requested_at={_format_scalar(last_requested_at)}")
        if daemon_parts:
            lines.append("queue daemon: " + " | ".join(daemon_parts))
        history_summary = _coerce_mapping(train_queue.get("history_summary"))
        if history_summary:
            history_parts: list[str] = []
            transition_count = history_summary.get("transition_count")
            if transition_count is not None:
                history_parts.append(f"transition_count={_format_scalar(transition_count)}")
            last_transition = _coerce_mapping(history_summary.get("last_transition"))
            if last_transition:
                transition_text = ",".join(
                    part
                    for part in (
                        _format_scalar(last_transition.get("job_id")) if last_transition.get("job_id") is not None else "",
                        _format_scalar(last_transition.get("event")) if last_transition.get("event") is not None else "",
                        _format_scalar(last_transition.get("state")) if last_transition.get("state") is not None else "",
                    )
                    if part
                )
                if transition_text:
                    history_parts.append(f"last_transition={transition_text}")
            last_reason = history_summary.get("last_reason")
            if last_reason is not None:
                history_parts.append(f"last_reason={_format_scalar(last_reason)}")
            if history_parts:
                lines.append("queue history: " + " | ".join(history_parts))
    ops_attention = _format_ops_attention(
        operations_alerts=operations_alerts,
        operations_overview=operations_overview,
        operations_dashboard=operations_dashboard,
        operations_alert_policy=operations_alert_policy,
        candidate_summary=candidate_summary,
        train_queue=train_queue,
        latest_adapter_map=latest_adapter_map,
        recent_adapter_map=recent_adapter_map,
    )
    if ops_attention is not None:
        lines.append(ops_attention)
    serve_state = _coerce_mapping(mapping.pop("serve", None))
    if serve_state is not None:
        adapter_resolution_state = serve_state.get("adapter_resolution_state")
        using_promoted_adapter = serve_state.get("using_promoted_adapter")
        if adapter_resolution_state or using_promoted_adapter is not None:
            serve_parts = []
            if using_promoted_adapter is not None:
                serve_parts.append(f"using_promoted_adapter={_format_scalar(using_promoted_adapter)}")
            if adapter_resolution_state is not None:
                serve_parts.append(f"adapter_resolution_state={_format_scalar(adapter_resolution_state)}")
            fallback_reason = serve_state.get("fallback_reason")
            if fallback_reason:
                serve_parts.append(f"reason={_format_scalar(fallback_reason)}")
            if serve_parts:
                lines.append("serve target: " + " | ".join(serve_parts))

    sample_counts = mapping.pop("sample_counts", None)
    if sample_counts is not None:
        sample_map = _coerce_mapping(sample_counts) or {}
        sample_summary = " | ".join(
            f"{split}={_format_scalar(sample_map.get(split))}"
            for split in ("train", "val", "test")
            if split in sample_map
        )
        if sample_summary:
            lines.append(f"sample counts: {sample_summary}")

    signal_summary = _coerce_mapping(mapping.pop("signal_summary", None))
    signal_sample_counts = _coerce_mapping(mapping.pop("signal_sample_counts", None))
    signal_sample_details = mapping.pop("signal_sample_details", None)
    signal_quality_summary = _coerce_mapping(mapping.pop("signal_quality_summary", None))
    signal_count_value = mapping.pop("signal_count", None)
    signal_sample_count_value = mapping.pop("signal_sample_count", None)
    if signal_summary is not None or signal_sample_counts is not None or signal_count_value is not None:
        signal_parts: list[str] = []
        if signal_summary is not None:
            state = signal_summary.get("state")
            if state is not None:
                signal_parts.append(f"state={_format_scalar(state)}")
            collection_enabled = signal_summary.get("collection_enabled")
            if collection_enabled is not None:
                signal_parts.append(f"collection_enabled={_format_scalar(collection_enabled)}")
            event_chain_ready = signal_summary.get("event_chain_ready")
            if event_chain_ready is not None:
                signal_parts.append(f"event_chain_ready={_format_scalar(event_chain_ready)}")
            event_chain_count = signal_summary.get("event_chain_complete_count")
            if event_chain_count is not None:
                signal_parts.append(f"event_chain_complete_count={_format_scalar(event_chain_count)}")
            processed_count = signal_summary.get("processed_count")
            if processed_count is not None:
                signal_parts.append(f"processed_count={_format_scalar(processed_count)}")
            latest_signal_id = signal_summary.get("latest_signal_id")
            if latest_signal_id:
                signal_parts.append(f"latest_signal_id={_format_scalar(latest_signal_id)}")
            quality_filter_state = signal_summary.get("quality_filter_state")
            if quality_filter_state is not None:
                signal_parts.append(f"quality_filter_state={_format_scalar(quality_filter_state)}")
            quality_filtered_count = signal_summary.get("quality_filtered_count")
            if quality_filtered_count is not None:
                signal_parts.append(f"quality_filtered_count={_format_scalar(quality_filtered_count)}")
        elif signal_count_value is not None:
            signal_parts.append(f"count={_format_scalar(signal_count_value)}")
        if signal_sample_count_value is not None:
            signal_parts.append(f"samples={_format_scalar(signal_sample_count_value)}")
        if signal_parts:
            lines.append("signal readiness: " + " | ".join(signal_parts))
    if signal_sample_counts is not None:
        sample_map = signal_sample_counts
        sample_summary = " | ".join(
            f"{split}={_format_scalar(sample_map.get(split))}"
            for split in ("train", "val", "test")
            if split in sample_map
        )
        if sample_summary:
            lines.append(f"signal samples: {sample_summary}")
    if signal_sample_details:
        detail_lines: list[str] = []
        detail_values = signal_sample_details if isinstance(signal_sample_details, Sequence) and not isinstance(signal_sample_details, (str, bytes, bytearray)) else [signal_sample_details]
        for detail in detail_values[:3]:
            detail_map = _coerce_mapping(detail) or {}
            parts = []
            for key in ("sample_id", "sample_type", "dataset_split", "source_adapter_version"):
                value = detail_map.get(key)
                if value is not None:
                    parts.append(f"{key.replace('_', ' ')}={_format_scalar(value)}")
            source_event_ids = detail_map.get("source_event_ids")
            if source_event_ids:
                parts.append(f"source_event_ids={_format_scalar(source_event_ids)}")
            if parts:
                detail_lines.append(" | ".join(parts))
        if detail_lines:
            lines.append("signal sample details:")
            for detail_line in detail_lines:
                lines.append(f"  - {detail_line}")

    if signal_quality_summary is not None:
        quality_parts: list[str] = []
        for key in ("evaluated_count", "passed_count", "filtered_count", "minimum_confidence"):
            value = signal_quality_summary.get(key)
            if value is not None:
                quality_parts.append(f"{key}={_format_scalar(value)}")
        filtered_reasons = signal_quality_summary.get("filtered_reasons")
        if filtered_reasons:
            if isinstance(filtered_reasons, dict):
                reason_text = ", ".join(f"{key}:{value}" for key, value in filtered_reasons.items())
                quality_parts.append(f"filtered_reasons={reason_text}")
            else:
                quality_parts.append(f"filtered_reasons={_format_scalar(filtered_reasons)}")
        if quality_parts:
            lines.append("signal quality filter: " + " | ".join(quality_parts))

    auto_trigger_action = _coerce_mapping(mapping.pop("auto_train_trigger_action", None))
    if auto_trigger_action is not None:
        action_parts: list[str] = []
        for key in (
            "action",
            "status",
            "reason",
            "triggered",
            "queue_job_id",
            "confirmation_reason",
            "approval_reason",
            "rejection_reason",
            "operator_note",
            "processed_count",
            "completed_count",
            "failed_count",
            "limit",
            "max_iterations",
            "max_cycles",
            "loop_cycles",
            "idle_rounds",
            "poll_interval_seconds",
            "remaining_queued",
            "drained",
            "stopped_reason",
            "triggered_version",
            "promoted_version",
        ):
            value = auto_trigger_action.get(key)
            if value is not None:
                action_parts.append(f"{key}={_format_scalar(value)}")
        if action_parts:
            lines.append("auto train action: " + " | ".join(action_parts))

    auto_trigger = _coerce_mapping(mapping.pop("auto_train_trigger", None))
    if auto_trigger is not None:
        trigger_parts: list[str] = []
        enabled = auto_trigger.get("enabled")
        state = auto_trigger.get("state")
        ready = auto_trigger.get("ready")
        reason = auto_trigger.get("reason")
        if enabled is not None:
            trigger_parts.append(f"enabled={_format_scalar(enabled)}")
        if state is not None:
            trigger_parts.append(f"state={_format_scalar(state)}")
        if ready is not None:
            trigger_parts.append(f"ready={_format_scalar(ready)}")
        if reason:
            trigger_parts.append(f"reason={_format_scalar(reason)}")
        for key in (
            "min_new_samples",
            "max_interval_days",
            "min_trigger_interval_minutes",
            "failure_backoff_minutes",
            "queue_mode",
            "queue_dedup_scope",
            "queue_priority_policy",
            "queue_process_batch_size",
            "queue_process_until_idle_max",
            "queue_worker_max_cycles",
            "queue_worker_idle_rounds",
            "queue_worker_poll_seconds",
            "require_queue_confirmation",
            "preference_reinforced_sample_weight",
            "effective_eligible_train_samples",
            "preference_reinforced_train_samples",
            "eligible_signal_train_samples",
            "effective_signal_train_samples",
            "preference_reinforced_signal_train_samples",
            "holdout_ready",
            "interval_elapsed",
            "queue_gate_reason",
            "queue_gate_action",
            "queue_review_mode",
            "blocked_primary_reason",
            "blocked_primary_action",
            "blocked_primary_category",
            "cooldown_elapsed",
            "cooldown_remaining_minutes",
            "failure_backoff_elapsed",
            "failure_backoff_remaining_minutes",
            "days_since_last_training",
            "consecutive_failures",
            "recent_training_version",
        ):
            value = auto_trigger.get(key)
            if value is not None:
                trigger_parts.append(f"{key}={_format_scalar(value)}")
        blocked_reasons = auto_trigger.get("blocked_reasons")
        if blocked_reasons:
            if isinstance(blocked_reasons, Sequence) and not isinstance(blocked_reasons, (str, bytes, bytearray)):
                trigger_parts.append(f"blocked_reasons={list(blocked_reasons)!r}")
            else:
                trigger_parts.append(f"blocked_reasons={_format_scalar(blocked_reasons)}")
        if trigger_parts:
            lines.append("auto train trigger: " + " | ".join(trigger_parts))

        policy = _coerce_mapping(auto_trigger.get("policy"))
        policy_parts: list[str] = []
        if policy is not None:
            for key in (
                "execution_mode",
                "queue_entry_mode",
                "review_mode",
                "evaluation_mode",
                "promotion_mode",
                "stop_stage",
                "evaluation_gate_reason",
                "evaluation_gate_action",
                "promote_gate_reason",
                "promote_gate_action",
                "promotion_requirement",
            ):
                value = policy.get(key)
                if value is not None:
                    policy_parts.append(f"{key}={_format_scalar(value)}")
        if policy_parts:
            lines.append("auto train trigger policy: " + " | ".join(policy_parts))

        threshold_summary = _coerce_mapping(auto_trigger.get("threshold_summary"))
        if threshold_summary is not None:
            threshold_parts: list[str] = []
            for key in (
                "min_new_samples",
                "effective_eligible_train_samples",
                "preference_reinforced_train_samples",
                "eligible_signal_train_samples",
                "effective_signal_train_samples",
                "preference_reinforced_signal_train_samples",
                "remaining_signal_samples",
                "remaining_effective_train_samples",
                "holdout_required",
                "holdout_ready",
                "max_interval_days",
                "days_since_last_training",
                "interval_elapsed",
                "min_trigger_interval_minutes",
                "cooldown_elapsed",
                "cooldown_remaining_minutes",
                "failure_backoff_minutes",
                "failure_backoff_elapsed",
                "failure_backoff_remaining_minutes",
                "preference_reinforced_sample_weight",
            ):
                value = threshold_summary.get(key)
                if value is not None:
                    threshold_parts.append(f"{key}={_format_scalar(value)}")
            if threshold_parts:
                lines.append("auto train trigger gate: " + " | ".join(threshold_parts))

        blocked_summary = auto_trigger.get("blocked_summary")
        if blocked_summary:
            lines.append(f"auto train trigger blocked summary: {_format_scalar(blocked_summary)}")

        summary = auto_trigger.get("last_result_summary")
        if summary:
            lines.append(f"auto train trigger summary: {_format_scalar(summary)}")

        last_result = _coerce_mapping(auto_trigger.get("last_result"))
        if last_result is not None:
            last_parts: list[str] = []
            for key in (
                "triggered",
                "state",
                "reason",
                "error_stage",
                "triggered_version",
                "triggered_state",
                "triggered_num_fresh_samples",
                "triggered_num_replay_samples",
                "eval_recommendation",
                "eval_comparison",
                "promoted_version",
            ):
                value = last_result.get(key)
                if value is not None:
                    last_parts.append(f"{key}={_format_scalar(value)}")
            if last_parts:
                lines.append("auto train trigger last result: " + " | ".join(last_parts))

    trainer = mapping.pop("trainer", None)
    trainer_map = _coerce_mapping(trainer)
    last_run_map = _coerce_mapping(trainer_map.get("last_run")) if trainer_map is not None else None
    metadata = _coerce_mapping(mapping.pop("metadata", None))
    runtime = _coerce_mapping(mapping.pop("runtime", None))
    inference_runtime = None
    if metadata is not None:
        inference_runtime = _coerce_mapping(metadata.get("inference"))
    if inference_runtime is None:
        inference_runtime = _coerce_mapping(mapping.get("inference"))
    if inference_runtime is not None and "real_local_enabled" in inference_runtime:
        lines.append(f"real local inference: enabled={_format_scalar(inference_runtime.get('real_local_enabled'))}")
    plans = _coerce_mapping(mapping.pop("plans", None))
    if plans is None and metadata is not None:
        plans = _coerce_mapping(metadata.get("plans"))
    if trainer is None and metadata is not None:
        trainer = metadata.get("trainer")
    if trainer is None:
        trainer = mapping.get("trainer")
    if plans is None:
        plans = _build_plan_snapshots(workspace or mapping.get("workspace") or mapping.get("home"), {"metadata": metadata} if metadata else mapping)

    recent_training_map = last_run_map or recent_adapter_map
    if recent_training_map is not None:
        recent_adapter_version = _pick_first(recent_training_map, "version") or recent_adapter_version
        recent_adapter_state = _pick_first(recent_training_map, "state") or recent_adapter_state

    trainer_line = _format_trainer_summary(trainer)
    if trainer_line is not None:
        lines.append(trainer_line)

    last_run = None
    if trainer_map is not None:
        last_run = _coerce_mapping(trainer_map.get("last_run"))
    if last_run is not None:
        recent_lines = _format_recent_training_snapshot(last_run)
        if recent_lines is not None:
            lines.extend(recent_lines)
    else:
        cached_state = _read_cli_state(workspace or mapping.get("workspace") or mapping.get("home"))
        if cached_state is not None:
            recent_snapshot = _coerce_mapping(cached_state.get("recent_training"))
            recent_lines = _format_recent_training_snapshot(recent_snapshot or cached_state)
            if recent_lines is not None:
                lines.extend(recent_lines)

    if plans:
        inference_plan = _coerce_mapping(plans.get("inference"))
        export_plan = _coerce_mapping(plans.get("export"))
        if inference_plan is not None:
            dispatch_line = _format_backend_dispatch(inference_plan)
            if dispatch_line is not None:
                lines.append(dispatch_line)
        if export_plan is not None:
            export_line = _format_export_write(export_plan)
            if export_line is not None:
                lines.append(export_line)

    return "\n".join(lines)


def _format_serve(result: Any) -> str:
    # Matrix theme - default style
    return formatters_matrix.format_serve_matrix(result)

def _format_serve_legacy(result: Any) -> str:
    """Legacy plain text formatter (kept for reference)."""
    mapping = _coerce_mapping(result)
    if mapping is None:
        return _format_scalar(result)

    for key in ("message", "ready_message", "ready", "detail", "status"):
        value = mapping.get(key)
        if isinstance(value, str) and value.strip():
            return value

    return _format_status_legacy(mapping)


def _format_candidate_history(result: Any) -> str:
    mapping = _coerce_mapping(result)
    if mapping is None:
        return _format_scalar(result)

    lines = ["PFE candidate history"]
    summary_parts: list[str] = []
    for key in ("workspace", "count", "last_action", "last_status", "last_reason", "last_note", "last_candidate_version"):
        value = mapping.get(key)
        if value is not None:
            summary_parts.append(f"{key}={_format_scalar(value)}")
    if summary_parts:
        lines.append("summary: " + " | ".join(summary_parts))

    items = list(mapping.get("items") or [])
    latest_timestamp = _history_latest_timestamp(items)
    if latest_timestamp is not None:
        lines.append(f"latest timestamp: {latest_timestamp}")
    if items:
        lines.append("items:")
        for item in items:
            if not isinstance(item, Mapping):
                lines.append(f"  - {_format_scalar(item)}")
                continue
            parts: list[str] = []
            for key in ("timestamp", "action", "status", "reason", "operator_note", "candidate_version", "promoted_version", "archived_version"):
                value = item.get(key)
                if value is not None:
                    parts.append(f"{key}={_format_scalar(value)}")
            lines.append("  - " + " | ".join(parts))
    else:
        lines.append("items: none")
    return "\n".join(lines)


def _candidate_timeline_stage(item: Mapping[str, Any] | None) -> str | None:
    if item is None:
        return None

    stage = item.get("stage")
    if stage is not None:
        return _format_scalar(stage)

    action = str(item.get("action") or "")
    status = str(item.get("status") or "")
    if action == "promote_candidate" and status == "completed":
        return "promoted"
    if action == "archive_candidate" and status == "completed":
        return "archived"
    if status == "blocked":
        return "blocked"
    if status == "noop":
        return "noop"
    return "candidate_action"


def _format_candidate_timeline_item(item: Any, *, index: int) -> str:
    if not isinstance(item, Mapping):
        return f"  - {index}. {_format_scalar(item)}"

    action = str(item.get("action") or "")
    status = str(item.get("status") or "")
    label = item.get("label") or f"{action}:{status}"

    parts: list[str] = []
    for key, value in (
        ("timestamp", item.get("timestamp")),
        ("stage", item.get("stage")),
        ("label", label if label else None),
        ("action", action if action else None),
        ("status", status if status else None),
        ("reason", item.get("reason")),
        ("operator_note", item.get("operator_note")),
        ("candidate_version", item.get("candidate_version")),
        ("promoted_version", item.get("promoted_version")),
        ("archived_version", item.get("archived_version")),
    ):
        if value is not None:
            parts.append(f"{key}={_format_scalar(value)}")

    if not parts:
        return f"  - {index}. {_format_scalar(item)}"
    return f"  - {index}. " + " | ".join(parts)


def _format_candidate_timeline(result: Any) -> str:
    mapping = _coerce_mapping(result)
    if mapping is None:
        return _format_scalar(result)

    lines = ["PFE candidate timeline"]
    summary_parts: list[str] = []
    for key in ("workspace", "count", "limit", "current_stage", "transition_count", "last_reason", "last_candidate_version"):
        value = mapping.get(key)
        if value is not None:
            summary_parts.append(f"{key}={_format_scalar(value)}")
    if summary_parts:
        lines.append("summary: " + " | ".join(summary_parts))

    items = list(mapping.get("items") or [])
    latest_timestamp = _history_latest_timestamp(items)
    if latest_timestamp is not None:
        lines.append(f"latest timestamp: {latest_timestamp}")
    if items:
        lines.append("timeline:")
        for index, item in enumerate(items, 1):
            if isinstance(item, Mapping) and "stage" not in item:
                item = {**dict(item), "stage": _candidate_timeline_stage(item)}
            lines.append(_format_candidate_timeline_item(item, index=index))
    else:
        lines.append("timeline: none")
    return "\n".join(lines)


def _format_train_queue_history(result: Any) -> str:
    mapping = _coerce_mapping(result)
    if mapping is None:
        return _format_scalar(result)

    lines = ["PFE train queue history"]
    summary_parts: list[str] = []
    for key in ("workspace", "job_id", "state", "count", "history_count"):
        value = mapping.get(key)
        if value is not None:
            summary_parts.append(f"{key}={_format_scalar(value)}")
    if summary_parts:
        lines.append("summary: " + " | ".join(summary_parts))

    available_job_ids = list(mapping.get("available_job_ids") or [])
    if available_job_ids:
        lines.append("available jobs: " + ", ".join(str(item) for item in available_job_ids))

    history_summary = _coerce_mapping(mapping.get("history_summary")) or {}
    if history_summary:
        history_summary_parts: list[str] = []
        if history_summary.get("transition_count") is not None:
            history_summary_parts.append(f"transition_count={_format_scalar(history_summary.get('transition_count'))}")
        if history_summary.get("last_reason") is not None:
            history_summary_parts.append(f"last_reason={_format_scalar(history_summary.get('last_reason'))}")
        last_transition = _coerce_mapping(history_summary.get("last_transition")) or {}
        if last_transition.get("event") is not None:
            history_summary_parts.append(f"last_event={_format_scalar(last_transition.get('event'))}")
        if history_summary_parts:
            lines.append("history summary: " + " | ".join(history_summary_parts))

    history = list(mapping.get("history") or [])
    latest_timestamp = _history_latest_timestamp(history)
    if latest_timestamp is not None:
        lines.append(f"latest timestamp: {latest_timestamp}")
    if history:
        lines.append("history:")
        for item in history:
            if not isinstance(item, Mapping):
                lines.append(f"  - {_format_scalar(item)}")
                continue
            parts: list[str] = []
            for key in ("timestamp", "event", "state", "reason", "note"):
                value = item.get(key)
                if value is not None:
                    parts.append(f"{key}={_format_scalar(value)}")
            lines.append("  - " + " | ".join(parts))
    else:
        lines.append("history: none")
    return "\n".join(lines)


def _format_worker_runner_history(result: Any) -> str:
    mapping = _coerce_mapping(result)
    if mapping is None:
        return _format_scalar(result)

    lines = ["PFE worker runner history"]
    summary_parts: list[str] = []
    for key in ("workspace", "count", "last_event", "last_reason"):
        value = mapping.get(key)
        if value is not None:
            summary_parts.append(f"{key}={_format_scalar(value)}")
    if summary_parts:
        lines.append("summary: " + " | ".join(summary_parts))

    items = list(mapping.get("items") or [])
    latest_timestamp = _history_latest_timestamp(items)
    if latest_timestamp is not None:
        lines.append(f"latest timestamp: {latest_timestamp}")
    if items:
        lines.append("items:")
        for item in items:
            if not isinstance(item, Mapping):
                lines.append(f"  - {_format_scalar(item)}")
                continue
            parts: list[str] = []
            for key in ("timestamp", "event", "reason"):
                value = item.get(key)
                if value is not None:
                    parts.append(f"{key}={_format_scalar(value)}")
            metadata = _coerce_mapping(item.get("metadata")) or {}
            for key in ("pid", "takeover", "previous_pid", "processed_count", "failed_count"):
                value = metadata.get(key)
                if value is not None:
                    parts.append(f"{key}={_format_scalar(value)}")
            lines.append("  - " + " | ".join(parts))
        else:
            lines.append("items: none")
    return "\n".join(lines)


def _format_train_queue_daemon_status(result: Any) -> str:
    mapping = _coerce_mapping(result)
    if mapping is None:
        return _format_scalar(result)

    lines = ["PFE worker daemon"]
    summary_parts: list[str] = []
    for key in ("workspace", "desired_state", "requested_action", "command_status", "last_event", "last_reason"):
        value = mapping.get(key)
        if value is not None:
            summary_parts.append(f"{key}={_format_scalar(value)}")
    if summary_parts:
        lines.append("summary: " + " | ".join(summary_parts))

    state_parts: list[str] = []
    for key in (
        "active",
        "observed_state",
        "lock_state",
        "recovery_state",
        "restart_attempts",
        "auto_restart_enabled",
        "auto_recover_enabled",
        "heartbeat_interval_seconds",
        "lease_timeout_seconds",
        "next_restart_after",
        "last_requested_by",
        "last_requested_at",
        "history_count",
        "auto_recovery_count",
    ):
        value = mapping.get(key)
        if value is not None:
            state_parts.append(f"{key}={_format_scalar(value)}")
    if state_parts:
        lines.append("state: " + " | ".join(state_parts))

    state_path = mapping.get("state_path")
    if state_path is not None:
        lines.append(f"state path: {_format_scalar(state_path)}")

    history = list(mapping.get("history") or [])
    if history:
        lines.append("history:")
        for item in history:
            if not isinstance(item, Mapping):
                lines.append(f"  - {_format_scalar(item)}")
                continue
            parts: list[str] = []
            for key in ("timestamp", "event", "reason", "note"):
                value = item.get(key)
                if value is not None:
                    parts.append(f"{key}={_format_scalar(value)}")
            lines.append("  - " + " | ".join(parts))
    else:
        lines.append("history: none")
    return "\n".join(lines)


def _format_daemon_timeline_summary(result: Any) -> str:
    mapping = _coerce_mapping(result)
    if mapping is None:
        return _format_scalar(result)

    summary_parts: list[str] = []
    for key in (
        "count",
        "recovery_event_count",
        "last_event",
        "last_reason",
        "last_recovery_event",
        "last_recovery_reason",
        "last_recovery_note",
        "recent_anomaly_event",
        "recent_anomaly_reason",
        "latest_timestamp",
    ):
        value = mapping.get(key)
        if value is not None:
            summary_parts.append(f"{key}={_format_scalar(value)}")
    lines = ["daemon timeline: " + " | ".join(summary_parts) if summary_parts else "daemon timeline:"]

    recent_recovery_events = list(mapping.get("recent_recovery_events") or [])
    if recent_recovery_events:
        lines.append("  recent recovery events:")
        for item in recent_recovery_events[:3]:
            if not isinstance(item, Mapping):
                lines.append(f"    - {_format_scalar(item)}")
                continue
            parts: list[str] = []
            for key in ("timestamp", "event", "reason", "note"):
                value = item.get(key)
                if value is not None:
                    parts.append(f"{key}={_format_scalar(value)}")
            lines.append("    - " + " | ".join(parts) if parts else "    - " + _format_scalar(item))
    return "\n".join(lines)


def _format_runner_timeline_summary(result: Any) -> str:
    mapping = _coerce_mapping(result)
    if mapping is None:
        return _format_scalar(result)

    summary_parts: list[str] = []
    for key in (
        "count",
        "last_event",
        "last_reason",
        "takeover_event_count",
        "last_takeover_event",
        "last_takeover_reason",
        "current_active",
        "current_lock_state",
        "current_stop_requested",
        "current_lease_expires_at",
        "recent_anomaly_reason",
        "latest_timestamp",
    ):
        value = mapping.get(key)
        if value is not None:
            summary_parts.append(f"{key}={_format_scalar(value)}")
    lines = ["runner timeline: " + " | ".join(summary_parts) if summary_parts else "runner timeline:"]

    recent_events = list(mapping.get("recent_events") or [])
    if recent_events:
        lines.append("  recent events:")
        for item in recent_events[:3]:
            if not isinstance(item, Mapping):
                lines.append(f"    - {_format_scalar(item)}")
                continue
            parts: list[str] = []
            for key in ("timestamp", "event", "reason"):
                value = item.get(key)
                if value is not None:
                    parts.append(f"{key}={_format_scalar(value)}")
            lines.append("    - " + " | ".join(parts) if parts else "    - " + _format_scalar(item))
    recent_takeover_events = list(mapping.get("recent_takeover_events") or [])
    if recent_takeover_events:
        lines.append("  recent takeover events:")
        for item in recent_takeover_events[:3]:
            if not isinstance(item, Mapping):
                lines.append(f"    - {_format_scalar(item)}")
                continue
            parts: list[str] = []
            for key in ("timestamp", "event", "reason", "note"):
                value = item.get(key)
                if value is not None:
                    parts.append(f"{key}={_format_scalar(value)}")
            lines.append("    - " + " | ".join(parts) if parts else "    - " + _format_scalar(item))
    return "\n".join(lines)


def _format_operations_event_stream(result: Any) -> list[str] | None:
    mapping = _coerce_mapping(result)
    if not mapping:
        return None

    def _resolved_focus(surface: Mapping[str, Any] | None) -> Any:
        surface_map = _coerce_mapping(surface)
        if not surface_map:
            return None
        current_focus = surface_map.get("current_focus")
        current_focus_text = str(current_focus or "").strip().lower()
        if current_focus_text not in {"", "none", "idle", "stable"}:
            return current_focus
        monitor_focus = surface_map.get("monitor_focus")
        return monitor_focus if monitor_focus is not None else current_focus

    lines = ["operations event stream:"]
    summary_parts: list[str] = []
    for key in (
        "count",
        "severity",
        "status",
        "attention_needed",
        "attention_reason",
        "attention_source",
        "current_focus",
        "required_action",
        "last_recovery_event",
        "last_recovery_reason",
        "last_recovery_note",
        "highest_priority_action",
        "active_recovery_hint",
        "latest_recovery",
        "latest_source",
        "latest_event",
        "latest_reason",
        "latest_timestamp",
        "alert_count",
        "escalated_reasons",
    ):
        value = _resolved_focus(mapping) if key == "current_focus" else mapping.get(key)
        if value is not None:
            summary_parts.append(f"{key}={_format_scalar(value)}")
    resolved_focus = _resolved_focus(mapping)
    summary_line = mapping.get("summary_line")
    inspection_summary_line = mapping.get("inspection_summary_line")
    summary_line, inspection_summary_line = _prefer_inspection_summary_for_generic_monitor(
        focus=resolved_focus,
        summary_line=summary_line,
        inspection_summary_line=inspection_summary_line,
    )
    if summary_line:
        summary_parts.append(f"summary={_format_scalar(summary_line)}")
    if inspection_summary_line and inspection_summary_line != summary_line:
        summary_parts.append(f"inspection_summary={_format_scalar(inspection_summary_line)}")
    if summary_parts:
        lines.append("  " + " | ".join(summary_parts))
    next_actions = mapping.get("next_actions")
    if next_actions:
        lines.append("  next_actions=" + _format_scalar(next_actions))
    dashboard = _coerce_mapping(mapping.get("dashboard"))
    if dashboard:
        dashboard_parts: list[str] = []
        for key in (
            "severity",
            "status",
            "attention_needed",
            "attention_reason",
            "current_focus",
            "required_action",
            "last_recovery_event",
            "last_recovery_reason",
            "last_recovery_note",
            "latest_source",
            "latest_event",
            "latest_reason",
        ):
            value = _resolved_focus(dashboard) if key == "current_focus" else dashboard.get(key)
            if value is not None:
                dashboard_parts.append(f"{key}={_format_scalar(value)}")
        dashboard_resolved_focus = _resolved_focus(dashboard)
        dashboard_summary_line = dashboard.get("summary_line")
        dashboard_inspection_summary = dashboard.get("inspection_summary_line")
        dashboard_summary_line, dashboard_inspection_summary = _prefer_inspection_summary_for_generic_monitor(
            focus=dashboard_resolved_focus,
            summary_line=dashboard_summary_line,
            inspection_summary_line=dashboard_inspection_summary,
        )
        if dashboard_summary_line:
            dashboard_parts.append(f"summary={_format_scalar(dashboard_summary_line)}")
        if dashboard_inspection_summary and dashboard_inspection_summary != dashboard_summary_line:
            dashboard_parts.append(f"inspection_summary={_format_scalar(dashboard_inspection_summary)}")
        if dashboard_parts:
            lines.append("  dashboard: " + " | ".join(dashboard_parts))

    items = list(mapping.get("items") or [])
    if items:
        lines.append("  recent events:")
        for item in items[:5]:
            if not isinstance(item, Mapping):
                lines.append(f"    - {_format_scalar(item)}")
                continue
            parts: list[str] = []
            for key in ("timestamp", "source", "event", "reason", "severity", "attention", "status", "version", "job_id", "note", "message"):
                value = item.get(key)
                if value is not None:
                    parts.append(f"{key}={_format_scalar(value)}")
            lines.append("    - " + " | ".join(parts))
    return lines


def _format_operations_dashboard(result: Any) -> list[str] | None:
    mapping = _coerce_mapping(result)
    if not mapping:
        return None

    def _resolved_focus(surface: Mapping[str, Any] | None) -> Any:
        surface_map = _coerce_mapping(surface)
        if not surface_map:
            return None
        current_focus = surface_map.get("current_focus")
        current_focus_text = str(current_focus or "").strip().lower()
        if current_focus_text not in {"", "none", "idle", "stable"}:
            return current_focus
        monitor_focus = surface_map.get("monitor_focus")
        return monitor_focus if monitor_focus is not None else current_focus

    lines = ["operations dashboard:"]
    resolved_focus = _resolved_focus(mapping)
    summary_line = mapping.get("summary_line")
    inspection_summary_line = mapping.get("inspection_summary_line")
    summary_line, inspection_summary_line = _prefer_inspection_summary_for_generic_monitor(
        focus=resolved_focus,
        summary_line=summary_line,
        inspection_summary_line=inspection_summary_line,
    )
    dashboard_digest = mapping.get("dashboard_digest")
    skip_dashboard_digest = bool(dashboard_digest) and dashboard_digest == summary_line

    summary_parts: list[str] = []
    for key in (
        "severity",
        "status",
        "attention_needed",
        "attention_reason",
        "highest_priority_action",
        "active_recovery_hint",
        "latest_recovery",
        "escalated_reasons",
        "remediation_mode",
        "operator_guidance",
        "auto_remediation_allowed",
        "requires_human_review",
        "requires_immediate_action",
        "current_focus",
        "candidate_stage",
        "queue_state",
        "runner_state",
        "daemon_health_state",
        "required_action",
        "last_recovery_event",
        "last_recovery_reason",
        "last_recovery_note",
        "latest_source",
        "latest_event",
        "latest_reason",
    ):
        value = _resolved_focus(mapping) if key == "current_focus" else mapping.get(key)
        if value is not None:
            summary_parts.append(f"{key}={_format_scalar(value)}")
    if dashboard_digest is not None and not skip_dashboard_digest:
        summary_parts.append(f"dashboard_digest={_format_scalar(dashboard_digest)}")
    if summary_parts:
        lines.append("  " + " | ".join(summary_parts))
    next_actions = mapping.get("next_actions")
    if next_actions:
        lines.append("  next_actions=" + _format_scalar(next_actions))
    if summary_line:
        lines.append("  summary=" + _format_scalar(summary_line))
    if inspection_summary_line and inspection_summary_line != summary_line:
        lines.append("  inspection_summary=" + _format_scalar(inspection_summary_line))
    return lines


def _format_operations_alert_policy(result: Any) -> list[str] | None:
    mapping = _coerce_mapping(result)
    if not mapping:
        return None

    def _resolved_focus(surface: Mapping[str, Any] | None) -> Any:
        surface_map = _coerce_mapping(surface)
        if not surface_map:
            return None
        current_focus = surface_map.get("current_focus")
        current_focus_text = str(current_focus or "").strip().lower()
        if current_focus_text not in {"", "none", "idle", "stable"}:
            return current_focus
        monitor_focus = surface_map.get("monitor_focus")
        return monitor_focus if monitor_focus is not None else current_focus

    lines = ["operations alert policy:"]
    summary_parts: list[str] = []
    for key in (
        "severity",
        "required_action",
        "current_focus",
        "primary_action",
        "highest_priority_action",
        "action_priority",
        "escalation_mode",
        "requires_immediate_action",
        "requires_human_review",
        "auto_remediation_allowed",
        "remediation_mode",
        "operator_guidance",
        "active_recovery_hint",
        "latest_recovery",
        "escalated_reasons",
        "last_recovery_event",
        "last_recovery_reason",
        "last_recovery_note",
    ):
        value = _resolved_focus(mapping) if key == "current_focus" else mapping.get(key)
        if value is not None:
            summary_parts.append(f"{key}={_format_scalar(value)}")
    if summary_parts:
        lines.append("  " + " | ".join(summary_parts))
    next_actions = mapping.get("next_actions")
    if next_actions:
        lines.append("  next_actions=" + _format_scalar(next_actions))
    inspection_summary_line = mapping.get("inspection_summary_line")
    resolved_focus = _resolved_focus(mapping)
    summary_line = mapping.get("summary_line")
    summary_line, inspection_summary_line = _prefer_inspection_summary_for_generic_monitor(
        focus=resolved_focus,
        summary_line=summary_line,
        inspection_summary_line=inspection_summary_line,
    )
    if summary_line:
        lines.append("  summary=" + _format_scalar(summary_line))
    if inspection_summary_line and inspection_summary_line != summary_line:
        lines.append("  inspection_summary=" + _format_scalar(inspection_summary_line))
    return lines


def _format_operations_timeline(result: Any) -> list[str] | None:
    mapping = _coerce_mapping(result)
    if not mapping:
        return None

    lines = ["operations timeline:"]
    summary_parts: list[str] = []
    summary_line = mapping.get("summary_line")
    if summary_line:
        summary_parts.append(f"summary={_format_scalar(summary_line)}")
    if summary_parts:
        lines.append("  " + " | ".join(summary_parts))

    for label in ("candidate", "queue", "runner", "daemon"):
        section = _coerce_mapping(mapping.get(label))
        if not section:
            continue
        section_parts: list[str] = []
        if label == "candidate":
            for key in ("current_stage", "last_candidate_version", "last_reason", "latest_timestamp", "transition_count"):
                value = section.get(key)
                if value is not None:
                    section_parts.append(f"{key}={_format_scalar(value)}")
        elif label == "queue":
            for key in ("count", "last_transition", "last_reason", "latest_timestamp", "transition_count"):
                value = section.get(key)
                if value is not None:
                    section_parts.append(f"{key}={_format_scalar(value)}")
        else:
            for key in (
                "count",
                "last_event",
                "last_reason",
                "takeover_event_count",
                "last_takeover_event",
                "last_takeover_reason",
                "recovery_event_count",
                "last_recovery_event",
                "last_recovery_reason",
                "last_recovery_note",
                "recent_anomaly_reason",
                "latest_timestamp",
            ):
                value = section.get(key)
                if value is not None:
                    section_parts.append(f"{key}={_format_scalar(value)}")
        if section_parts:
            lines.append(f"  {label}: " + " | ".join(section_parts))
    return lines


def _format_train_queue_daemon_history(result: Any) -> str:
    mapping = _coerce_mapping(result)
    if mapping is None:
        return _format_scalar(result)

    lines = ["PFE worker daemon history"]
    summary_parts: list[str] = []
    for key in ("workspace", "count", "last_event", "last_reason"):
        value = mapping.get(key)
        if value is not None:
            summary_parts.append(f"{key}={_format_scalar(value)}")
    if summary_parts:
        lines.append("summary: " + " | ".join(summary_parts))

    items = list(mapping.get("items") or [])
    latest_timestamp = _history_latest_timestamp(items)
    if latest_timestamp is not None:
        lines.append(f"latest timestamp: {latest_timestamp}")
    if items:
        lines.append("items:")
        for item in items:
            if not isinstance(item, Mapping):
                lines.append(f"  - {_format_scalar(item)}")
                continue
            parts: list[str] = []
            for key in ("timestamp", "event", "reason", "note"):
                value = item.get(key)
                if value is not None:
                    parts.append(f"{key}={_format_scalar(value)}")
            lines.append("  - " + " | ".join(parts))
    else:
        lines.append("items: none")
    return "\n".join(lines)


def _format_daemon_health_status(result: Any) -> str:
    """Format daemon health status for CLI output."""
    mapping = _coerce_mapping(result)
    if mapping is None:
        return _format_scalar(result)

    lines = ["PFE daemon health status"]

    # Overall health
    overall = mapping.get("overall_health", "unknown")
    lines.append(f"overall: {overall}")

    # Issues
    issues = list(mapping.get("issues") or [])
    if issues:
        lines.append("issues:")
        for issue in issues:
            component = issue.get("component", "unknown")
            state = issue.get("state", "unknown")
            message = issue.get("message", "")
            lines.append(f"  - [{component}] {state}: {message}")
    else:
        lines.append("issues: none")

    # Daemon details
    daemon = mapping.get("daemon") or {}
    daemon_parts: list[str] = []
    for key in ("health_state", "lock_state", "heartbeat_state", "lease_state", "restart_policy_state"):
        value = daemon.get(key)
        if value is not None:
            daemon_parts.append(f"{key}={value}")
    if daemon_parts:
        lines.append("daemon: " + " | ".join(daemon_parts))

    # Runner details
    runner = mapping.get("runner") or {}
    runner_parts: list[str] = []
    for key in ("lock_state", "active", "lease_expires_at"):
        value = runner.get(key)
        if value is not None:
            runner_parts.append(f"{key}={_format_scalar(value)}")
    if runner_parts:
        lines.append("runner: " + " | ".join(runner_parts))

    # Reliability summary
    reliability = mapping.get("reliability") or {}
    if reliability:
        rel_parts: list[str] = []
        for key in ("active_runners", "stalled_jobs", "expired_leases"):
            value = reliability.get(key)
            if value is not None:
                rel_parts.append(f"{key}={value}")
        if rel_parts:
            lines.append("reliability: " + " | ".join(rel_parts))

        # Alerts summary
        alerts = reliability.get("alerts_summary") or {}
        if alerts:
            total = alerts.get("total_active", 0)
            critical = alerts.get("critical_count", 0)
            error = alerts.get("error_count", 0)
            warning = alerts.get("warning_count", 0)
            if total > 0:
                lines.append(f"alerts: total={total} critical={critical} error={error} warning={warning}")

    checked_at = mapping.get("checked_at")
    if checked_at:
        lines.append(f"checked_at: {checked_at}")

    return "\n".join(lines)


def _format_daemon_heartbeat_status(result: Any) -> str:
    """Format daemon heartbeat status for CLI output."""
    mapping = _coerce_mapping(result)
    if mapping is None:
        return _format_scalar(result)

    lines = ["PFE daemon heartbeat status"]

    # Daemon heartbeat
    daemon = mapping.get("daemon") or {}
    daemon_parts: list[str] = []
    for key in ("heartbeat_state", "heartbeat_age_seconds", "lease_state"):
        value = daemon.get(key)
        if value is not None:
            daemon_parts.append(f"{key}={_format_scalar(value)}")
    if daemon_parts:
        lines.append("daemon: " + " | ".join(daemon_parts))

    last_hb = daemon.get("last_heartbeat_at")
    if last_hb:
        lines.append(f"  last_heartbeat: {last_hb}")
    lease_expires = daemon.get("lease_expires_at")
    if lease_expires:
        lines.append(f"  lease_expires: {lease_expires}")

    # Runner heartbeat
    runner = mapping.get("runner") or {}
    runner_parts: list[str] = []
    for key in ("heartbeat_age_seconds", "stale_after_seconds"):
        value = runner.get(key)
        if value is not None:
            runner_parts.append(f"{key}={_format_scalar(value)}")
    if runner_parts:
        lines.append("runner: " + " | ".join(runner_parts))

    runner_hb = runner.get("last_heartbeat_at")
    if runner_hb:
        lines.append(f"  last_heartbeat: {runner_hb}")
    runner_lease = runner.get("lease_expires_at")
    if runner_lease:
        lines.append(f"  lease_expires: {runner_lease}")

    checked_at = mapping.get("checked_at")
    if checked_at:
        lines.append(f"checked_at: {checked_at}")

    return "\n".join(lines)


def _format_daemon_lease_status(result: Any) -> str:
    """Format daemon lease status for CLI output."""
    mapping = _coerce_mapping(result)
    if mapping is None:
        return _format_scalar(result)

    lines = ["PFE daemon lease status"]

    # Daemon lease
    daemon_lease = mapping.get("daemon_lease") or {}
    daemon_parts: list[str] = []
    for key in ("lease_state", "heartbeat_state"):
        value = daemon_lease.get(key)
        if value is not None:
            daemon_parts.append(f"{key}={value}")
    if daemon_parts:
        lines.append("daemon: " + " | ".join(daemon_parts))

    expires = daemon_lease.get("lease_expires_at")
    if expires:
        lines.append(f"  lease_expires: {expires}")

    # Runner lease
    runner_lease = mapping.get("runner_lease") or {}
    runner_parts: list[str] = []
    for key in ("lock_state", "stale_after_seconds"):
        value = runner_lease.get(key)
        if value is not None:
            runner_parts.append(f"{key}={_format_scalar(value)}")
    if runner_parts:
        lines.append("runner: " + " | ".join(runner_parts))

    runner_expires = runner_lease.get("lease_expires_at")
    if runner_expires:
        lines.append(f"  lease_expires: {runner_expires}")

    # Expired leases
    expired_count = mapping.get("expired_leases_count", 0)
    lines.append(f"expired_leases: {expired_count}")

    expired = list(mapping.get("expired_leases") or [])
    if expired:
        lines.append("recent expired:")
        for lease in expired[:5]:
            lid = lease.get("lease_id", "unknown")[:8]
            job = lease.get("job_id", "unknown")[:8]
            state = lease.get("state", "unknown")
            lines.append(f"  - {lid}... job={job}... state={state}")

    checked_at = mapping.get("checked_at")
    if checked_at:
        lines.append(f"checked_at: {checked_at}")

    return "\n".join(lines)


def _format_daemon_stale_check(result: Any) -> str:
    """Format daemon stale check results for CLI output."""
    mapping = _coerce_mapping(result)
    if mapping is None:
        return _format_scalar(result)

    lines = ["PFE daemon stale check"]

    takeover = mapping.get("takeover_requested", False)
    lines.append(f"takeover_requested: {takeover}")

    # Daemon status
    daemon = mapping.get("daemon") or {}
    daemon_stale = daemon.get("is_stale", False)
    daemon_parts: list[str] = [f"is_stale={daemon_stale}"]
    for key in ("lock_state", "heartbeat_state", "can_recover"):
        value = daemon.get(key)
        if value is not None:
            daemon_parts.append(f"{key}={value}")
    lines.append("daemon: " + " | ".join(daemon_parts))

    # Runner status
    runner = mapping.get("runner") or {}
    runner_stale = runner.get("is_stale", False)
    runner_parts: list[str] = [f"is_stale={runner_stale}"]
    for key in ("lock_state", "active"):
        value = runner.get(key)
        if value is not None:
            runner_parts.append(f"{key}={value}")
    lines.append("runner: " + " | ".join(runner_parts))

    # Actions taken
    actions = list(mapping.get("actions_taken") or [])
    if actions:
        lines.append("actions_taken:")
        for action in actions:
            component = action.get("component", "unknown")
            act = action.get("action", "unknown")
            result_status = action.get("result", "")
            note = action.get("note", "")
            if result_status:
                lines.append(f"  - [{component}] {act}: {result_status}")
            elif note:
                lines.append(f"  - [{component}] {act}: {note}")
            else:
                lines.append(f"  - [{component}] {act}")

    checked_at = mapping.get("checked_at")
    if checked_at:
        lines.append(f"checked_at: {checked_at}")

    return "\n".join(lines)


def _format_daemon_alerts(result: Any) -> str:
    """Format daemon alerts for CLI output."""
    mapping = _coerce_mapping(result)
    if mapping is None:
        return _format_scalar(result)

    lines = ["PFE daemon alerts"]

    # Filters
    filters = mapping.get("filters") or {}
    filter_parts: list[str] = []
    for key in ("level", "scope"):
        value = filters.get(key)
        if value:
            filter_parts.append(f"{key}={value}")
    if filter_parts:
        lines.append("filters: " + " | ".join(filter_parts))

    # Count
    count = mapping.get("count", 0)
    lines.append(f"count: {count}")

    # Alerts
    alerts = list(mapping.get("alerts") or [])
    if alerts:
        lines.append("alerts:")
        for alert in alerts:
            level = alert.get("level", "unknown")
            scope = alert.get("scope", "unknown")
            reason = alert.get("reason", "unknown")
            message = alert.get("message", "")
            timestamp = alert.get("timestamp", "")
            alert_id = alert.get("alert_id", "")[:8]

            lines.append(f"  - [{level}] {scope}: {reason}")
            if message:
                lines.append(f"    message: {message}")
            if timestamp:
                lines.append(f"    timestamp: {timestamp}")
    else:
        lines.append("alerts: none")

    # Summary
    summary = mapping.get("summary") or {}
    if summary:
        total = summary.get("total_active", 0)
        critical = summary.get("critical_count", 0)
        error = summary.get("error_count", 0)
        warning = summary.get("warning_count", 0)
        lines.append(f"summary: total_active={total} critical={critical} error={error} warning={warning}")

    checked_at = mapping.get("checked_at")
    if checked_at:
        lines.append(f"checked_at: {checked_at}")

    return "\n".join(lines)


def _daemon_recovery_payload(
    *,
    workspace: str | None = None,
    action: str,
    note: str | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    state = _read_train_queue_daemon_state(workspace) or {}
    history = list(state.get("history") or [])
    timestamp = datetime.now(timezone.utc).isoformat()
    restart_attempts = int(state.get("restart_attempts", 0) or 0)
    if action == "restart":
        restart_attempts += 1
    recovery_state = "restarting" if action == "restart" else "recovering"
    state.update(
        {
            "workspace": workspace or "user_default",
            "desired_state": "running",
            "requested_action": action,
            "command_status": "requested",
            "active": True,
            "observed_state": recovery_state,
            "recovery_state": recovery_state,
            "auto_restart_enabled": True,
            "restart_attempts": restart_attempts,
            "restart_backoff_seconds": float(state.get("restart_backoff_seconds", 30.0) or 30.0),
            "next_restart_after": timestamp,
            "last_requested_at": timestamp,
            "last_requested_by": "pfe-cli",
            "last_reason": reason or "cli_requested",
            "last_recovery_reason": reason or "cli_requested",
        }
    )
    history.append(
        {
            "timestamp": timestamp,
            "event": f"{action}_requested",
            "reason": reason or "cli_requested",
            "note": note,
        }
    )
    state["history"] = history[-20:]
    state["history_count"] = len(history)
    _write_train_queue_daemon_state(workspace, state)
    return state


def _format_ops_attention(
    *,
    operations_alerts: Any | None,
    operations_overview: Mapping[str, Any] | None,
    operations_dashboard: Mapping[str, Any] | None,
    operations_alert_policy: Mapping[str, Any] | None,
    candidate_summary: Mapping[str, Any] | None,
    train_queue: Mapping[str, Any] | None,
    latest_adapter_map: Mapping[str, Any] | None,
    recent_adapter_map: Mapping[str, Any] | None,
) -> str | None:
    alerts: list[str] = []
    overview = _coerce_mapping(operations_overview) or {}
    dashboard = _coerce_mapping(operations_dashboard) or {}
    alert_policy = _coerce_mapping(operations_alert_policy) or {}

    def _resolved_focus() -> str | None:
        for candidate in (
            overview.get("current_focus"),
            overview.get("monitor_focus"),
            dashboard.get("current_focus"),
            dashboard.get("monitor_focus"),
            alert_policy.get("current_focus"),
        ):
            if candidate is None:
                continue
            text = str(candidate).strip()
            if text.lower() in {"", "none", "idle", "stable"}:
                continue
            return text
        return None

    structured_alerts = _coerce_sequence_of_mappings(operations_alerts)
    for alert in structured_alerts:
        reason = alert.get("reason")
        if reason is not None:
            alerts.append(_format_scalar(reason))

    resolved_focus = _resolved_focus()
    required_action = (
        overview.get("required_action") or alert_policy.get("required_action") or dashboard.get("required_action")
    )
    inspection_summary_line = (
        overview.get("inspection_summary_line")
        or dashboard.get("inspection_summary_line")
        or alert_policy.get("inspection_summary_line")
    )
    monitor_alert_emitted = False
    if (
        resolved_focus
        and required_action is not None
        and str(resolved_focus).strip().lower() in _GENERIC_MONITOR_FOCUSES
        and not structured_alerts
    ):
        if inspection_summary_line:
            alerts.append("monitor " + _format_scalar(inspection_summary_line))
        else:
            parts = [
                f"current_focus={_format_scalar(resolved_focus)}",
                f"required_action={_format_scalar(required_action)}",
            ]
            alerts.append("monitor " + " | ".join(parts))
        monitor_alert_emitted = True

    if candidate_summary is not None:
        candidate_version = candidate_summary.get("candidate_version")
        candidate_state = candidate_summary.get("candidate_state")
        needs_promotion = candidate_summary.get("candidate_needs_promotion")
        if needs_promotion:
            parts = []
            if candidate_version is not None:
                parts.append(f"version={_format_scalar(candidate_version)}")
            if candidate_state is not None:
                parts.append(f"state={_format_scalar(candidate_state)}")
            if parts:
                alerts.append("candidate_needs_promotion " + " | ".join(parts))
            else:
                alerts.append("candidate_needs_promotion")
        elif candidate_state in {"training", "pending_eval", "failed_eval"} and not (
            monitor_alert_emitted and str(resolved_focus).strip().lower().startswith("candidate")
        ):
            parts = []
            if candidate_version is not None:
                parts.append(f"version={_format_scalar(candidate_version)}")
            parts.append(f"state={_format_scalar(candidate_state)}")
            alerts.append("candidate " + " | ".join(parts))

    if train_queue is not None:
        counts = _coerce_mapping(train_queue.get("counts")) or {}
        queued_count = counts.get("queued")
        if queued_count and not (monitor_alert_emitted and str(resolved_focus).strip().lower().startswith("queue")):
            current_item = _coerce_mapping(train_queue.get("current"))
            if current_item is not None and current_item.get("state") == "awaiting_confirmation":
                queue_parts = [f"awaiting_confirmation={_format_scalar(queued_count)}"]
            else:
                queue_parts = [f"queued={_format_scalar(queued_count)}"]
            alerts.append("queue " + " | ".join(queue_parts))

        confirmation_summary = _coerce_mapping(train_queue.get("confirmation_summary"))
        if confirmation_summary is not None:
            awaiting_confirmation_count = confirmation_summary.get("awaiting_confirmation_count")
            if awaiting_confirmation_count:
                next_job_id = confirmation_summary.get("next_job_id")
                queue_parts = [f"awaiting_confirmation={_format_scalar(awaiting_confirmation_count)}"]
                if next_job_id is not None:
                    queue_parts.append(f"next_job_id={_format_scalar(next_job_id)}")
                alerts.append("confirmation " + " | ".join(queue_parts))

        worker_runner = _coerce_mapping(train_queue.get("worker_runner"))
        if worker_runner is not None:
            lock_state = worker_runner.get("lock_state")
            active = worker_runner.get("active")
            stop_requested = worker_runner.get("stop_requested")
            if (lock_state in {"active", "stale"} or active or stop_requested) and not (
                monitor_alert_emitted and str(resolved_focus).strip().lower().startswith(("runner", "daemon"))
            ):
                runner_parts: list[str] = []
                if lock_state is not None:
                    runner_parts.append(f"lock_state={_format_scalar(lock_state)}")
                if active is not None:
                    runner_parts.append(f"active={_format_scalar(active)}")
                if stop_requested is not None:
                    runner_parts.append(f"stop_requested={_format_scalar(stop_requested)}")
                lease_expires_at = worker_runner.get("lease_expires_at")
                if lease_expires_at is not None:
                    runner_parts.append(f"lease_expires_at={_format_scalar(lease_expires_at)}")
                alerts.append("worker runner " + " | ".join(runner_parts))

    for label, snapshot in (("latest export", latest_adapter_map), ("recent export", recent_adapter_map)):
        if snapshot is None:
            continue
        export_valid = snapshot.get("export_artifact_valid")
        export_exists = snapshot.get("export_artifact_exists")
        export_path = snapshot.get("export_artifact_path")
        if export_valid is False or export_exists is False:
            export_parts = [f"valid={_format_scalar(export_valid)}", f"exists={_format_scalar(export_exists)}"]
            if export_path is not None:
                export_parts.append(f"path={_format_scalar(export_path)}")
            alerts.append(f"{label} " + " | ".join(export_parts))

    if resolved_focus and all("current_focus=" not in alert for alert in alerts):
        parts = [f"current_focus={_format_scalar(resolved_focus)}"]
        if required_action is not None:
            parts.append(f"required_action={_format_scalar(required_action)}")
        alerts.append("monitor " + " | ".join(parts))

    if not alerts:
        return "ops attention: clean"
    return "ops attention: " + " | ".join(alerts)


def _build_operations_console_digest(
    *,
    operations_console: Mapping[str, Any] | None,
    operations_overview: Mapping[str, Any] | None,
    operations_dashboard: Mapping[str, Any] | None = None,
    operations_alert_policy: Mapping[str, Any] | None = None,
    candidate_summary: Mapping[str, Any] | None,
    candidate_history: Mapping[str, Any] | None,
    candidate_timeline: Mapping[str, Any] | None,
    daemon_timeline: Mapping[str, Any] | None,
    runner_timeline: Mapping[str, Any] | None,
    train_queue: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    console = _coerce_mapping(operations_console) or {}
    daemon_timeline = _coerce_mapping(daemon_timeline) or _coerce_mapping(console.get("daemon_timeline")) or {}
    runner_timeline = _coerce_mapping(runner_timeline) or _coerce_mapping(console.get("runner_timeline")) or {}
    if console:
        if daemon_timeline and "daemon" not in console:
            console = dict(console)
            console["daemon"] = {
                "count": daemon_timeline.get("count"),
                "recovery_event_count": daemon_timeline.get("recovery_event_count"),
                "last_event": daemon_timeline.get("last_event"),
                "last_reason": daemon_timeline.get("last_reason"),
                "last_recovery_event": daemon_timeline.get("last_recovery_event"),
                "last_recovery_reason": daemon_timeline.get("last_recovery_reason"),
                "last_recovery_note": daemon_timeline.get("last_recovery_note"),
                "latest_timestamp": daemon_timeline.get("latest_timestamp"),
            }
        if runner_timeline and "runner_timeline" not in console:
            console = dict(console)
            console["runner_timeline"] = {
                "count": runner_timeline.get("count"),
                "last_event": runner_timeline.get("last_event"),
                "last_reason": runner_timeline.get("last_reason"),
                "current_active": runner_timeline.get("current_active"),
                "current_lock_state": runner_timeline.get("current_lock_state"),
                "current_stop_requested": runner_timeline.get("current_stop_requested"),
                "current_lease_expires_at": runner_timeline.get("current_lease_expires_at"),
                "latest_timestamp": runner_timeline.get("latest_timestamp"),
            }
        return console

    overview = _coerce_mapping(operations_overview) or {}
    dashboard_surface = _coerce_mapping(operations_dashboard) or {}
    alert_policy_surface = _coerce_mapping(operations_alert_policy) or {}
    candidate_summary = _coerce_mapping(candidate_summary) or {}
    candidate_history = _coerce_mapping(candidate_history) or {}
    candidate_timeline = _coerce_mapping(candidate_timeline) or {}
    train_queue = _coerce_mapping(train_queue) or {}

    if not overview and not candidate_summary and not candidate_history and not candidate_timeline and not daemon_timeline and not runner_timeline and not train_queue:
        return None

    queue_history = _coerce_mapping(train_queue.get("history_summary")) or {}
    queue_review = _coerce_mapping(train_queue.get("review_summary")) or {}
    queue_confirm = _coerce_mapping(train_queue.get("confirmation_summary")) or {}
    worker = _coerce_mapping(train_queue.get("worker_runner")) or {}

    derived_next_actions: list[str] = []
    attention_reason = overview.get("attention_reason")
    candidate_state = candidate_summary.get("candidate_state")
    candidate_needs_promotion = bool(candidate_summary.get("candidate_needs_promotion"))
    if attention_reason == "awaiting_confirmation":
        derived_next_actions.append("review_queue_confirmation")
    elif attention_reason == "candidate_ready_for_promotion" or candidate_needs_promotion:
        derived_next_actions.append("review_candidate_promotion")
    elif attention_reason:
        derived_next_actions.append(str(attention_reason))

    if str(worker.get("lock_state") or "") == "stale":
        derived_next_actions.append("inspect_worker_stale_lock")
    if bool(worker.get("active")) and bool(worker.get("stop_requested")):
        derived_next_actions.append("wait_for_runner_shutdown")
    if int(queue_confirm.get("awaiting_confirmation_count", 0) or 0) > 0 and "review_queue_confirmation" not in derived_next_actions:
        derived_next_actions.append("review_queue_confirmation")

    candidate_section = {
        "current_stage": candidate_timeline.get("current_stage") or candidate_state,
        "last_candidate_version": candidate_timeline.get("last_candidate_version")
        or candidate_history.get("last_candidate_version")
        or candidate_summary.get("candidate_version"),
        "last_reason": candidate_timeline.get("last_reason") or candidate_history.get("last_reason"),
        "latest_timestamp": candidate_timeline.get("latest_timestamp") or candidate_history.get("latest_timestamp"),
        "transition_count": candidate_timeline.get("transition_count") or candidate_history.get("count"),
        "history_count": candidate_history.get("count") or candidate_timeline.get("history_count"),
    }
    queue_section = {
        "count": train_queue.get("count"),
        "awaiting_confirmation_count": queue_confirm.get("awaiting_confirmation_count"),
        "next_confirmation_reason": queue_confirm.get("next_confirmation_reason"),
        "last_transition": queue_history.get("last_transition"),
        "last_reason": queue_history.get("last_reason"),
        "reviewed_transition_count": queue_review.get("reviewed_transition_count"),
        "last_review_event": queue_review.get("last_review_event"),
        "last_review_note": queue_review.get("last_review_note"),
    }
    runner_section = {
        "active": worker.get("active"),
        "lock_state": worker.get("lock_state"),
        "last_event": worker.get("last_event"),
        "last_event_reason": worker.get("last_event_reason"),
        "lease_expires_at": worker.get("lease_expires_at"),
        "history_count": worker.get("history_count"),
    }
    daemon_section = {
        "count": daemon_timeline.get("count"),
        "recovery_event_count": daemon_timeline.get("recovery_event_count"),
        "last_event": daemon_timeline.get("last_event"),
        "last_reason": daemon_timeline.get("last_reason"),
        "last_recovery_event": daemon_timeline.get("last_recovery_event"),
        "last_recovery_reason": daemon_timeline.get("last_recovery_reason"),
        "last_recovery_note": daemon_timeline.get("last_recovery_note"),
        "recent_anomaly_reason": daemon_timeline.get("recent_anomaly_reason"),
        "latest_timestamp": daemon_timeline.get("latest_timestamp"),
    }
    runner_timeline_section = {
        "count": runner_timeline.get("count"),
        "last_event": runner_timeline.get("last_event"),
        "last_reason": runner_timeline.get("last_reason"),
        "takeover_event_count": runner_timeline.get("takeover_event_count"),
        "last_takeover_event": runner_timeline.get("last_takeover_event"),
        "last_takeover_reason": runner_timeline.get("last_takeover_reason"),
        "recent_anomaly_reason": runner_timeline.get("recent_anomaly_reason"),
        "latest_timestamp": runner_timeline.get("latest_timestamp"),
    }

    def _resolved_focus(*candidates: Any) -> Any:
        for candidate in candidates:
            if candidate is None:
                continue
            if str(candidate).strip().lower() in {"", "none", "idle", "stable"}:
                continue
            return candidate
        return next((candidate for candidate in candidates if candidate is not None), None)

    current_focus = _resolved_focus(
        overview.get("current_focus"),
        overview.get("monitor_focus"),
        dashboard_surface.get("current_focus"),
        dashboard_surface.get("monitor_focus"),
        alert_policy_surface.get("current_focus"),
        overview.get("attention_reason"),
        overview.get("monitor_focus"),
        alert_policy_surface.get("required_action"),
        derived_next_actions[0] if derived_next_actions else None,
    )
    next_actions = (
        _coerce_sequence_of_scalars(overview.get("next_actions"))
        or _coerce_sequence_of_scalars(dashboard_surface.get("next_actions"))
        or _coerce_sequence_of_scalars(alert_policy_surface.get("next_actions"))
        or derived_next_actions
    )
    required_action = (
        overview.get("required_action")
        or alert_policy_surface.get("required_action")
        or dashboard_surface.get("required_action")
        or (derived_next_actions[0] if derived_next_actions else None)
    )
    inspection_summary_line = (
        overview.get("inspection_summary_line")
        or dashboard_surface.get("inspection_summary_line")
        or alert_policy_surface.get("inspection_summary_line")
    )
    last_recovery_event = (
        dashboard_surface.get("last_recovery_event")
        or alert_policy_surface.get("last_recovery_event")
        or daemon_timeline.get("last_recovery_event")
    )
    last_recovery_reason = (
        dashboard_surface.get("last_recovery_reason")
        or alert_policy_surface.get("last_recovery_reason")
        or daemon_timeline.get("last_recovery_reason")
    )
    last_recovery_note = (
        dashboard_surface.get("last_recovery_note")
        or alert_policy_surface.get("last_recovery_note")
        or daemon_timeline.get("last_recovery_note")
    )

    summary_parts = []
    if overview.get("summary_line"):
        summary_parts.append(str(overview.get("summary_line")))
    elif candidate_section.get("current_stage"):
        summary_parts.append(f"candidate-stage={_format_scalar(candidate_section['current_stage'])}")
    if current_focus is not None:
        summary_parts.append(f"current_focus={_format_scalar(current_focus)}")
    if required_action is not None:
        summary_parts.append(f"required_action={_format_scalar(required_action)}")
    if queue_section.get("awaiting_confirmation_count"):
        summary_parts.append(f"awaiting-confirm={_format_scalar(queue_section['awaiting_confirmation_count'])}")
    if runner_section.get("lock_state"):
        summary_parts.append(f"runner-lock={_format_scalar(runner_section['lock_state'])}")
    if runner_timeline_section.get("last_event"):
        summary_parts.append(f"runner-timeline={_format_scalar(runner_timeline_section['last_event'])}")
    if runner_timeline_section.get("recent_anomaly_reason"):
        summary_parts.append(f"runner-anomaly={_format_scalar(runner_timeline_section['recent_anomaly_reason'])}")
    if daemon_section.get("recent_anomaly_reason"):
        summary_parts.append(f"daemon-anomaly={_format_scalar(daemon_section['recent_anomaly_reason'])}")
    if last_recovery_event is not None:
        summary_parts.append(f"last_recovery_event={_format_scalar(last_recovery_event)}")
    if last_recovery_reason is not None:
        summary_parts.append(f"last_recovery_reason={_format_scalar(last_recovery_reason)}")
    if last_recovery_note is not None:
        summary_parts.append(f"last_recovery_note={_format_scalar(last_recovery_note)}")
    summary_line = " | ".join(summary_parts)
    summary_line, inspection_summary_line = _prefer_inspection_summary_for_generic_monitor(
        focus=current_focus,
        summary_line=summary_line,
        inspection_summary_line=inspection_summary_line,
    )

    attention_needed = overview.get("attention_needed")
    if attention_needed is None:
        attention_needed = bool(next_actions)

    return {
        "attention_needed": bool(attention_needed),
        "attention_reason": attention_reason,
        "summary_line": summary_line,
        "inspection_summary_line": inspection_summary_line
        or " | ".join(
            part
            for part in (
                f"current_focus={_format_scalar(current_focus)}" if current_focus is not None else None,
                f"required_action={_format_scalar(required_action)}" if required_action is not None else None,
                f"last_recovery_event={_format_scalar(last_recovery_event)}" if last_recovery_event is not None else None,
                f"last_recovery_reason={_format_scalar(last_recovery_reason)}" if last_recovery_reason is not None else None,
                f"last_recovery_note={_format_scalar(last_recovery_note)}" if last_recovery_note is not None else None,
                f"next_actions={_format_scalar(next_actions)}" if next_actions else None,
            )
            if part is not None
        ),
        "next_actions": next_actions,
        "current_focus": current_focus,
        "required_action": required_action,
        "last_recovery_event": last_recovery_event,
        "last_recovery_reason": last_recovery_reason,
        "last_recovery_note": last_recovery_note,
        "candidate": candidate_section,
        "queue": queue_section,
        "runner": runner_section,
        "daemon": daemon_section,
        "runner_timeline": runner_timeline_section,
    }


def _build_operations_alert_surface(
    *,
    operations_alerts: Any | None,
    operations_health: Any | None,
    operations_recovery: Any | None,
    operations_next_actions: Any | None,
    operations_dashboard: Mapping[str, Any] | None,
    operations_alert_policy: Mapping[str, Any] | None,
    operations_console: Mapping[str, Any] | None,
    operations_overview: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    alerts = _coerce_sequence_of_mappings(operations_alerts)
    health = _coerce_mapping(operations_health) or {}
    recovery = _coerce_mapping(operations_recovery) or {}
    next_actions = _coerce_sequence_of_scalars(operations_next_actions)
    dashboard = _coerce_mapping(operations_dashboard) or {}
    alert_policy = _coerce_mapping(operations_alert_policy) or {}
    overview = _coerce_mapping(operations_overview) or {}
    console = _coerce_mapping(operations_console) or {}

    def _resolved_focus(*candidates: Any) -> Any:
        for candidate in candidates:
            if candidate is None:
                continue
            if str(candidate).strip().lower() in {"", "none", "idle", "stable"}:
                continue
            return candidate
        return next((candidate for candidate in candidates if candidate is not None), None)

    current_focus = _resolved_focus(
        overview.get("current_focus"),
        overview.get("monitor_focus"),
        dashboard.get("current_focus"),
        dashboard.get("monitor_focus"),
        alert_policy.get("current_focus"),
        console.get("current_focus"),
        overview.get("attention_reason"),
        overview.get("monitor_focus"),
    )
    required_action = (
        overview.get("required_action")
        or alert_policy.get("required_action")
        or dashboard.get("required_action")
        or console.get("required_action")
    )
    inspection_summary_line = (
        overview.get("inspection_summary_line")
        or dashboard.get("inspection_summary_line")
        or alert_policy.get("inspection_summary_line")
        or console.get("inspection_summary_line")
    )
    summary_line = (
        health.get("summary_line")
        or recovery.get("summary_line")
        or console.get("summary_line")
        or overview.get("summary_line")
    )
    summary_line, inspection_summary_line = _prefer_inspection_summary_for_generic_monitor(
        focus=current_focus,
        summary_line=summary_line,
        inspection_summary_line=inspection_summary_line,
    )

    if alerts or health or recovery or next_actions:
        attention_needed = bool(health.get("status") == "attention" or alerts or next_actions)
        return {
            "attention_needed": attention_needed,
            "current_focus": current_focus,
            "required_action": required_action,
            "inspection_summary_line": inspection_summary_line,
            "alerts": alerts,
            "health": health,
            "recovery": recovery,
            "next_actions": next_actions,
            "summary_line": _format_scalar(summary_line),
        }

    if not console and not overview:
        return None

    derived_alerts = _coerce_sequence_of_mappings(overview.get("alerts"))
    if not derived_alerts:
        candidate_stage = _coerce_mapping(console.get("candidate"))
        queue_section = _coerce_mapping(console.get("queue"))
        runner_section = _coerce_mapping(console.get("runner"))
        derived_alerts = []
        if bool(overview.get("attention_needed")) or bool(console.get("attention_needed")):
            derived_alerts.append(
                {
                    "reason": overview.get("attention_reason") or console.get("attention_reason") or "operations_attention",
                    "detail": (
                        overview.get("inspection_summary_line")
                        or console.get("inspection_summary_line")
                        or overview.get("summary_line")
                        or console.get("summary_line")
                    ),
                    "candidate_stage": candidate_stage.get("current_stage") if candidate_stage else None,
                    "queue_count": queue_section.get("count") if queue_section else None,
                    "runner_lock_state": runner_section.get("lock_state") if runner_section else None,
                }
            )
    derived_health = _coerce_mapping(overview.get("health")) or _coerce_mapping(console.get("health")) or {}
    derived_recovery = _coerce_mapping(overview.get("recovery")) or _coerce_mapping(console.get("recovery")) or {}
    derived_next_actions = _coerce_sequence_of_scalars(console.get("next_actions")) or _coerce_sequence_of_scalars(
        overview.get("next_actions")
    )
    if not derived_next_actions:
        derived_next_actions = []
        if bool(overview.get("attention_needed")) or bool(console.get("attention_needed")):
            derived_next_actions.extend(_coerce_sequence_of_scalars(console.get("next_actions")) or [])
    if not derived_alerts and not derived_health and not derived_recovery and not derived_next_actions:
        return None
    return {
        "attention_needed": bool(overview.get("attention_needed") if overview.get("attention_needed") is not None else console.get("attention_needed", False)),
        "current_focus": current_focus,
        "required_action": required_action,
        "inspection_summary_line": inspection_summary_line,
        "alerts": derived_alerts,
        "health": derived_health,
        "recovery": derived_recovery,
        "next_actions": derived_next_actions,
        "summary_line": summary_line,
    }


def _format_operations_console_digest(result: Any) -> list[str] | None:
    mapping = _coerce_mapping(result)
    if mapping is None:
        return None

    operations_console_mapping = _coerce_mapping(mapping.pop("operations_console", None))
    daemon_timeline = _coerce_mapping(mapping.get("daemon_timeline"))
    runner_timeline = _coerce_mapping(mapping.get("runner_timeline"))
    if daemon_timeline is None and operations_console_mapping is not None:
        daemon_timeline = _coerce_mapping(operations_console_mapping.get("daemon_timeline"))
    if runner_timeline is None and operations_console_mapping is not None:
        runner_timeline = _coerce_mapping(operations_console_mapping.get("runner_timeline"))
    console = _build_operations_console_digest(
        operations_console=(
            {**operations_console_mapping, "daemon_timeline": daemon_timeline, "runner_timeline": runner_timeline}
            if operations_console_mapping is not None and (daemon_timeline is not None or runner_timeline is not None)
            else operations_console_mapping
        ),
        operations_overview=_coerce_mapping(mapping.get("operations_overview")),
        operations_dashboard=_coerce_mapping(mapping.get("operations_dashboard")),
        operations_alert_policy=_coerce_mapping(mapping.get("operations_alert_policy")),
        candidate_summary=_coerce_mapping(mapping.get("candidate_summary")),
        candidate_history=_coerce_mapping(mapping.get("candidate_history")),
        candidate_timeline=_coerce_mapping(mapping.get("candidate_timeline")),
        daemon_timeline=daemon_timeline,
        runner_timeline=runner_timeline,
        train_queue=_coerce_mapping(mapping.get("train_queue")),
    )
    if console is None:
        return None

    lines = ["operations console digest:"]
    digest_parts: list[str] = []
    attention_needed = console.get("attention_needed")
    if attention_needed is not None:
        digest_parts.append(f"attention_needed={_format_scalar(attention_needed)}")
    attention_reason = console.get("attention_reason")
    if attention_reason is not None:
        digest_parts.append(f"attention_reason={_format_scalar(attention_reason)}")
    current_focus = console.get("current_focus")
    if current_focus is not None:
        digest_parts.append(f"current_focus={_format_scalar(current_focus)}")
    required_action = console.get("required_action")
    if required_action is not None:
        digest_parts.append(f"required_action={_format_scalar(required_action)}")
    summary_line = console.get("summary_line")
    if summary_line:
        digest_parts.append(f"summary={_format_scalar(summary_line)}")
    inspection_summary_line = console.get("inspection_summary_line")
    if inspection_summary_line and inspection_summary_line != summary_line:
        digest_parts.append(f"inspection_summary={_format_scalar(inspection_summary_line)}")
    next_actions = console.get("next_actions")
    if next_actions:
        digest_parts.append(f"next_actions={_format_scalar(next_actions)}")
    last_recovery_event = console.get("last_recovery_event")
    if last_recovery_event is not None:
        digest_parts.append(f"last_recovery_event={_format_scalar(last_recovery_event)}")
    if digest_parts:
        lines.append("  " + " | ".join(digest_parts))

    for label in ("candidate", "queue", "runner", "runner_timeline", "daemon"):
        section = _coerce_mapping(console.get(label))
        if not section:
            continue
        section_parts: list[str] = []
        if label == "candidate":
            for key in ("current_stage", "last_candidate_version", "last_reason", "latest_timestamp", "transition_count", "history_count"):
                value = section.get(key)
                if value is not None:
                    section_parts.append(f"{key}={_format_scalar(value)}")
        elif label == "queue":
            for key in (
                "count",
                "awaiting_confirmation_count",
                "next_confirmation_reason",
                "last_reason",
                "reviewed_transition_count",
                "last_review_event",
                "last_review_note",
            ):
                value = section.get(key)
                if value is not None:
                    section_parts.append(f"{key}={_format_scalar(value)}")
            last_transition = _coerce_mapping(section.get("last_transition"))
            if last_transition:
                transition_parts: list[str] = []
                for key in ("job_id", "event", "state"):
                    value = last_transition.get(key)
                    if value is not None:
                        transition_parts.append(_format_scalar(value))
                if transition_parts:
                    section_parts.append("last_transition=" + ",".join(transition_parts))
        elif label in {"runner", "daemon"}:
            for key in (
                "active",
                "lock_state",
                "health_state",
                "lease_state",
                "heartbeat_state",
                "restart_policy_state",
                "recovery_action",
                "last_event",
                "last_event_reason",
                "lease_expires_at",
                "history_count",
                "recovery_needed",
                "can_recover",
                "recovery_reason",
                "recovery_state",
                "recovery_event_count",
                "last_recovery_event",
                "last_recovery_reason",
                "last_recovery_note",
                "recent_anomaly_reason",
            ):
                value = section.get(key)
                if value is not None:
                    section_parts.append(f"{key}={_format_scalar(value)}")
        elif label == "runner_timeline":
            for key in (
                "count",
                "last_event",
                "last_reason",
                "takeover_event_count",
                "last_takeover_event",
                "last_takeover_reason",
                "recent_anomaly_reason",
                "latest_timestamp",
            ):
                value = section.get(key)
                if value is not None:
                    section_parts.append(f"{key}={_format_scalar(value)}")
        daemon_timeline = _coerce_mapping(console.get("daemon_timeline"))
        runner_timeline = _coerce_mapping(console.get("runner_timeline"))
        if label == "daemon" and daemon_timeline:
            for key in ("count", "last_event", "last_reason", "latest_timestamp"):
                value = daemon_timeline.get(key)
                if value is not None:
                    section_parts.append(f"{key}={_format_scalar(value)}")
        if label == "runner_timeline" and runner_timeline:
            for key in ("count", "last_event", "last_reason", "takeover_event_count", "last_takeover_event", "last_takeover_reason", "recent_anomaly_reason", "latest_timestamp"):
                value = runner_timeline.get(key)
                if value is not None and f"{key}={_format_scalar(value)}" not in section_parts:
                    section_parts.append(f"{key}={_format_scalar(value)}")
        if section_parts:
            lines.append(f"  operations console {label}: " + " | ".join(section_parts))

    timelines = _coerce_mapping(console.get("timelines"))
    if timelines:
        timeline_summary = timelines.get("summary_line")
        if timeline_summary:
            lines.append(f"  operations console timelines: {_format_scalar(timeline_summary)}")

    return lines


def _format_operations_alert_surface(result: Any) -> list[str] | None:
    mapping = _coerce_mapping(result)
    if mapping is None:
        return None

    alert_surface = _build_operations_alert_surface(
        operations_alerts=mapping.pop("operations_alerts", None),
        operations_health=mapping.pop("operations_health", None),
        operations_recovery=mapping.pop("operations_recovery", None),
        operations_next_actions=mapping.pop("operations_next_actions", None),
        operations_dashboard=_coerce_mapping(mapping.get("operations_dashboard")),
        operations_alert_policy=_coerce_mapping(mapping.get("operations_alert_policy")),
        operations_console=_coerce_mapping(mapping.get("operations_console")),
        operations_overview=_coerce_mapping(mapping.get("operations_overview")),
    )
    if alert_surface is None:
        return None

    lines = ["operations alerts:"]
    alert_parts: list[str] = []
    if alert_surface.get("attention_needed") is not None:
        alert_parts.append(f"attention_needed={_format_scalar(alert_surface.get('attention_needed'))}")
    if alert_surface.get("current_focus") is not None:
        alert_parts.append(f"current_focus={_format_scalar(alert_surface.get('current_focus'))}")
    if alert_surface.get("required_action") is not None:
        alert_parts.append(f"required_action={_format_scalar(alert_surface.get('required_action'))}")
    summary_line = alert_surface.get("summary_line")
    if summary_line:
        alert_parts.append(f"summary={_format_scalar(summary_line)}")
    inspection_summary_line = alert_surface.get("inspection_summary_line")
    if inspection_summary_line and inspection_summary_line != summary_line:
        alert_parts.append(f"inspection_summary={_format_scalar(inspection_summary_line)}")
    if alert_parts:
        lines.append("  " + " | ".join(alert_parts))

    alerts = _coerce_sequence_of_mappings(alert_surface.get("alerts"))
    if alerts:
        lines.append("  alerts:")
        for alert in alerts:
            parts: list[str] = []
            for key in ("reason", "detail", "candidate_stage", "queue_count", "runner_lock_state", "severity"):
                value = alert.get(key)
                if value is not None:
                    parts.append(f"{key}={_format_scalar(value)}")
            lines.append("    - " + " | ".join(parts) if parts else "    - " + _format_scalar(alert))

    health = _coerce_mapping(alert_surface.get("health")) or {}
    if health:
        parts: list[str] = []
        for key in (
            "status",
            "daemon_lock_state",
            "health_state",
            "daemon_health_state",
            "lease_state",
            "daemon_lease_state",
            "heartbeat_state",
            "daemon_heartbeat_state",
            "restart_policy_state",
            "daemon_restart_policy_state",
            "recovery_action",
            "daemon_recovery_action",
            "runner_lock_state",
            "candidate_state",
            "queue_state",
        ):
            value = health.get(key)
            if value is not None:
                parts.append(f"{key}={_format_scalar(value)}")
        if parts:
            lines.append("  health: " + " | ".join(parts))

    recovery = _coerce_mapping(alert_surface.get("recovery")) or {}
    if recovery:
        parts = []
        for key in (
            "daemon_recovery_needed",
            "daemon_recovery_reason",
            "daemon_recovery_state",
            "daemon_recovery_action",
            "recovery_needed",
            "recovery_reason",
        ):
            value = recovery.get(key)
            if value is not None:
                parts.append(f"{key}={_format_scalar(value)}")
        if parts:
            lines.append("  recovery: " + " | ".join(parts))

    next_actions = _coerce_sequence_of_scalars(alert_surface.get("next_actions"))
    if next_actions:
        lines.append("  next actions: " + ", ".join(_format_scalar(action) for action in next_actions))

    return lines


def _history_latest_timestamp(items: Any) -> str | None:
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes, bytearray)):
        return None
    for item in reversed(items):
        item_map = _coerce_mapping(item)
        if item_map is None:
            continue
        timestamp = item_map.get("timestamp")
        if timestamp is not None:
            return _format_scalar(timestamp)
    return None


def _lookup_adapter_snapshot(version: str | None, *, workspace: str | None = None) -> dict[str, Any] | None:
    if version is None:
        return None

    store = _optional_module_call("pfe_core.adapter_store.store", "create_adapter_store", workspace=workspace)
    if store is None:
        return None

    list_records = getattr(store, "list_version_records", None)
    if not callable(list_records):
        return None

    try:
        rows = list_records(limit=100)
    except Exception:
        return None
    if not isinstance(rows, Sequence):
        return None

    latest_version = None
    if str(version) == "latest":
        current_latest = getattr(store, "current_latest_version", None)
        if callable(current_latest):
            try:
                latest_version = current_latest()
            except Exception:
                latest_version = None
    target_version = latest_version or str(version)

    for row in rows:
        row_map = _coerce_mapping(row)
        if row_map is None:
            continue
        if str(row_map.get("version")) == target_version:
            row_version = str(row_map.get("version"))
            metadata = _coerce_mapping(row_map.get("metadata")) or {}
            export_execution = _coerce_mapping(metadata.get("export_execution")) or {}
            export_write = _coerce_mapping(metadata.get("export_write")) or {}
            export_artifact_summary = _coerce_mapping(metadata.get("export_artifact_summary")) or _coerce_mapping(
                _coerce_mapping(metadata.get("export")).get("artifact") if _coerce_mapping(metadata.get("export")) else None
            ) or {}
            if not export_artifact_summary:
                manifest_path = row_map.get("manifest_path")
                if manifest_path:
                    try:
                        manifest_payload = json.loads(Path(str(manifest_path)).expanduser().read_text(encoding="utf-8"))
                    except Exception:
                        manifest_payload = {}
                    manifest_metadata = _coerce_mapping(manifest_payload.get("metadata")) or {}
                    export_artifact_summary = _coerce_mapping(manifest_metadata.get("export_artifact_summary")) or _coerce_mapping(
                        _coerce_mapping(manifest_metadata.get("export")).get("artifact") if _coerce_mapping(manifest_metadata.get("export")) else None
                    ) or {}
            output_artifact_validation = _coerce_mapping(export_execution.get("output_artifact_validation")) or {}
            export_artifact_path = (
                output_artifact_validation.get("path")
                or export_execution.get("output_artifact_path")
                or export_write.get("artifact_path")
                or export_artifact_summary.get("path")
                or row_map.get("artifact_path")
            )
            export_artifact_exists = False
            export_artifact_size_bytes = None
            if export_artifact_path:
                try:
                    export_artifact_file = Path(str(export_artifact_path)).expanduser()
                    export_artifact_exists = export_artifact_file.exists()
                    if export_artifact_exists and export_artifact_file.is_file():
                        export_artifact_size_bytes = export_artifact_file.stat().st_size
                except Exception:
                    export_artifact_exists = False
            return {
                "version": row_map.get("version"),
                "state": row_map.get("state", row_map.get("status")),
                "latest": row_map.get("latest") if "latest" in row_map else (latest_version is not None and row_version == latest_version),
                "num_samples": row_map.get("num_samples", row_map.get("samples")),
                "artifact_format": row_map.get("artifact_format", row_map.get("format")),
                "export_status": export_execution.get("status") or _pick_first(_coerce_mapping(export_execution.get("audit")), "status") or export_artifact_summary.get("status"),
                "export_write_state": export_write.get("write_state") or _pick_first(_coerce_mapping(export_write.get("metadata")), "write_state") or export_artifact_summary.get("write_state"),
                "export_artifact_path": export_artifact_path,
                "export_artifact_valid": output_artifact_validation.get("valid", export_artifact_summary.get("valid")),
                "export_artifact_exists": export_artifact_exists,
                "export_artifact_size_bytes": export_artifact_size_bytes if export_artifact_size_bytes is not None else export_artifact_summary.get("size_bytes"),
            }
    return None


def _lookup_recent_adapter_snapshot(*, workspace: str | None = None) -> dict[str, Any] | None:
    store = _optional_module_call("pfe_core.adapter_store.store", "create_adapter_store", workspace=workspace)
    if store is None:
        return None

    list_records = getattr(store, "list_version_records", None)
    if not callable(list_records):
        return None

    try:
        rows = list_records(limit=1)
    except Exception:
        return None
    if not isinstance(rows, Sequence) or not rows:
        return None

    row_map = _coerce_mapping(rows[0])
    if row_map is None:
        return None

    latest_version = None
    current_latest = getattr(store, "current_latest_version", None)
    if callable(current_latest):
        try:
            latest_version = current_latest()
        except Exception:
            latest_version = None

    row_version = str(row_map.get("version"))
    metadata = _coerce_mapping(row_map.get("metadata")) or {}
    export_execution = _coerce_mapping(metadata.get("export_execution")) or {}
    export_write = _coerce_mapping(metadata.get("export_write")) or {}
    export_artifact_summary = _coerce_mapping(metadata.get("export_artifact_summary")) or _coerce_mapping(
        _coerce_mapping(metadata.get("export")).get("artifact") if _coerce_mapping(metadata.get("export")) else None
    ) or {}
    if not export_artifact_summary:
        manifest_path = row_map.get("manifest_path")
        if manifest_path:
            try:
                manifest_payload = json.loads(Path(str(manifest_path)).expanduser().read_text(encoding="utf-8"))
            except Exception:
                manifest_payload = {}
            manifest_metadata = _coerce_mapping(manifest_payload.get("metadata")) or {}
            export_artifact_summary = _coerce_mapping(manifest_metadata.get("export_artifact_summary")) or _coerce_mapping(
                _coerce_mapping(manifest_metadata.get("export")).get("artifact") if _coerce_mapping(manifest_metadata.get("export")) else None
            ) or {}
    output_artifact_validation = _coerce_mapping(export_execution.get("output_artifact_validation")) or {}
    export_artifact_path = (
        output_artifact_validation.get("path")
        or export_execution.get("output_artifact_path")
        or export_write.get("artifact_path")
        or export_artifact_summary.get("path")
        or row_map.get("artifact_path")
    )
    export_artifact_exists = False
    export_artifact_size_bytes = None
    if export_artifact_path:
        try:
            export_artifact_file = Path(str(export_artifact_path)).expanduser()
            export_artifact_exists = export_artifact_file.exists()
            if export_artifact_exists and export_artifact_file.is_file():
                export_artifact_size_bytes = export_artifact_file.stat().st_size
        except Exception:
            export_artifact_exists = False
    return {
        "version": row_map.get("version"),
        "state": row_map.get("state", row_map.get("status")),
        "latest": latest_version is not None and row_version == str(latest_version),
        "num_samples": row_map.get("num_samples", row_map.get("samples")),
        "artifact_format": row_map.get("artifact_format", row_map.get("format")),
        "export_status": export_execution.get("status") or _pick_first(_coerce_mapping(export_execution.get("audit")), "status") or export_artifact_summary.get("status"),
        "export_write_state": export_write.get("write_state") or _pick_first(_coerce_mapping(export_write.get("metadata")), "write_state") or export_artifact_summary.get("write_state"),
        "export_artifact_path": export_artifact_path,
        "export_artifact_valid": output_artifact_validation.get("valid", export_artifact_summary.get("valid")),
        "export_artifact_exists": export_artifact_exists,
        "export_artifact_size_bytes": export_artifact_size_bytes if export_artifact_size_bytes is not None else export_artifact_summary.get("size_bytes"),
    }


def _format_bytes_compact(value: Any) -> str | None:
    try:
        size = int(value)
    except Exception:
        return None
    units = ("B", "KB", "MB", "GB")
    scaled = float(size)
    unit = units[0]
    for candidate in units:
        unit = candidate
        if scaled < 1024.0 or candidate == units[-1]:
            break
        scaled /= 1024.0
    if unit == "B":
        return f"{int(scaled)}{unit}"
    return f"{scaled:.1f}{unit}"


def _format_adapter_snapshot_line(
    label: str,
    snapshot: Any,
    *,
    include_latest: bool = False,
) -> str | None:
    mapping = _coerce_mapping(snapshot)
    if mapping is None:
        return None

    parts: list[str] = []
    version = _pick_first(mapping, "version")
    if version is not None:
        parts.append(f"version={_format_scalar(version)}")
    state = _pick_first(mapping, "state")
    if state is not None:
        parts.append(f"state={_format_scalar(state)}")
    if include_latest:
        latest = _pick_first(mapping, "latest")
        if latest is not None:
            parts.append(f"latest={_format_scalar(latest)}")
    samples = _pick_first(mapping, "num_samples", "samples")
    if samples is not None:
        parts.append(f"samples={_format_scalar(samples)}")
    artifact_format = _pick_first(mapping, "artifact_format", "format")
    if artifact_format is not None:
        parts.append(f"format={_format_scalar(artifact_format)}")
    if not parts:
        return None
    return f"{label}: " + " | ".join(parts)


def _format_adapter_export_artifact_line(label: str, snapshot: Any) -> str | None:
    mapping = _coerce_mapping(snapshot)
    if mapping is None:
        return None

    artifact_path = _pick_first(mapping, "export_artifact_path")
    artifact_valid = _pick_first(mapping, "export_artifact_valid")
    artifact_size = _format_bytes_compact(_pick_first(mapping, "export_artifact_size_bytes"))
    export_status = _pick_first(mapping, "export_status")
    write_state = _pick_first(mapping, "export_write_state")
    artifact_exists = _pick_first(mapping, "export_artifact_exists")

    parts: list[str] = []
    if export_status is not None:
        parts.append(f"status={_format_scalar(export_status)}")
    if write_state is not None:
        parts.append(f"write_state={_format_scalar(write_state)}")
    if artifact_valid is not None:
        parts.append(f"valid={_format_scalar(artifact_valid)}")
    if artifact_exists is not None:
        parts.append(f"exists={_format_scalar(artifact_exists)}")
    if artifact_size is not None:
        parts.append(f"size={artifact_size}")
    if artifact_path:
        parts.append(f"path={_format_scalar(artifact_path)}")
    if not parts:
        return None
    return f"{label}: " + " | ".join(parts)


def _format_job_execution_summary(job_execution: Any) -> str | None:
    mapping = _coerce_mapping(job_execution)
    if mapping is None:
        return None

    parts: list[str] = []
    status = _pick_first(mapping, "status")
    executor_mode = _pick_first(mapping, "executor_mode")
    metadata = _coerce_mapping(mapping.get("metadata"))
    execution_state = _pick_first(metadata, "execution_state")
    runner_status = _pick_first(_coerce_mapping(mapping.get("audit")), "runner_status")

    if status is not None:
        parts.append(f"status={_format_scalar(status)}")
    if executor_mode is not None:
        parts.append(f"executor_mode={_format_scalar(executor_mode)}")
    if execution_state is not None:
        parts.append(f"execution_state={_format_scalar(execution_state)}")
    if runner_status is not None and runner_status != status:
        parts.append(f"runner_status={_format_scalar(runner_status)}")
    if not parts:
        return None
    return "job-execution: " + " | ".join(parts)


def _format_real_execution_summary(job_execution: Any, *, executor_mode: str | None = None) -> str | None:
    mapping = _coerce_mapping(job_execution)
    if mapping is None:
        return None

    parts: list[str] = []
    status = _pick_first(mapping, "state", "status")
    kind = _pick_first(mapping, "kind")
    executor_mode = _pick_first(mapping, "execution_mode", "executor_mode") or executor_mode
    attempted = _pick_first(mapping, "attempted")
    success = _pick_first(mapping, "success")
    available = _pick_first(mapping, "available")
    audit = _coerce_mapping(mapping.get("audit"))
    runner_status = _pick_first(mapping, "runner_status") or _pick_first(audit, "runner_status")

    if status is not None:
        parts.append(f"status={_format_scalar(status)}")
    if kind is not None:
        parts.append(f"kind={_format_scalar(kind)}")
    if executor_mode is not None:
        parts.append(f"executor_mode={_format_scalar(executor_mode)}")
    if attempted is not None:
        parts.append(f"attempted={_format_scalar(attempted)}")
    if success is not None:
        parts.append(f"success={_format_scalar(success)}")
    if available is not None:
        parts.append(f"available={_format_scalar(available)}")
    if runner_status is not None and runner_status != status:
        parts.append(f"runner_status={_format_scalar(runner_status)}")
    if not parts:
        return None
    return "real-execution: " + " | ".join(parts)


def _format_export_execution_summary(export_execution: Any) -> str | None:
    mapping = _coerce_mapping(export_execution)
    if mapping is None:
        return None

    parts: list[str] = []
    audit = _coerce_mapping(mapping.get("audit"))
    metadata = _coerce_mapping(mapping.get("metadata"))
    status = _pick_first(audit, "status", "execution_status") or _pick_first(mapping, "status", "execution_status")
    execution_mode = _pick_first(metadata, "execution_mode") or _pick_first(mapping, "execution_mode")
    attempted = _pick_first(mapping, "attempted")
    success = _pick_first(mapping, "success")

    if status is not None:
        parts.append(f"status={_format_scalar(status)}")
    if execution_mode is not None:
        parts.append(f"execution_mode={_format_scalar(execution_mode)}")
    if attempted is not None:
        parts.append(f"attempted={_format_scalar(attempted)}")
    if success is not None:
        parts.append(f"success={_format_scalar(success)}")
    if not parts:
        return None
    return "export-execution: " + " | ".join(parts)


def _format_export_toolchain_summary(export_execution: Any) -> str | None:
    mapping = _coerce_mapping(export_execution)
    if mapping is None:
        return None

    parts: list[str] = []
    audit = _coerce_mapping(mapping.get("audit"))
    metadata = _coerce_mapping(mapping.get("metadata"))
    status = _pick_first(mapping, "summary", "status", "toolchain_status") or _pick_first(
        audit,
        "status",
        "execution_status",
    )
    execution_mode = _pick_first(mapping, "execution_mode") or _pick_first(metadata, "execution_mode")
    attempted = _pick_first(mapping, "attempted")
    success = _pick_first(mapping, "success")
    required = _pick_first(mapping, "required")
    artifact_valid = _pick_first(mapping, "output_artifact_valid")

    if status is not None:
        parts.append(f"status={_format_scalar(status)}")
    if execution_mode is not None:
        parts.append(f"execution_mode={_format_scalar(execution_mode)}")
    if attempted is not None:
        parts.append(f"attempted={_format_scalar(attempted)}")
    if success is not None:
        parts.append(f"success={_format_scalar(success)}")
    if required is not None:
        parts.append(f"required={_format_scalar(required)}")
    if artifact_valid is not None:
        parts.append(f"artifact_valid={_format_scalar(artifact_valid)}")
    if not parts:
        return None
    return "export-toolchain: " + " | ".join(parts)


def _format_incremental_context(context: Any) -> str | None:
    mapping = _coerce_mapping(context)
    if mapping is None:
        return None

    parts: list[str] = []
    for key in (
        "requested_base_adapter",
        "parent_adapter_version",
        "parent_base_model",
        "parent_adapter_path",
        "source_adapter_version",
        "source_adapter_path",
        "source_model",
    ):
        value = mapping.get(key)
        if value is not None:
            parts.append(f"{key}={_format_scalar(value)}")
    if not parts:
        return None
    return "incremental: " + " | ".join(parts)


def _format_compare_evaluation(compare_evaluation: Any) -> str | None:
    mapping = _coerce_mapping(compare_evaluation)
    if mapping is None:
        return None

    parts: list[str] = []
    for key in (
        "left_adapter",
        "right_adapter",
        "comparison",
        "winner",
        "recommendation",
        "overall_delta",
        "style_preference_hit_rate_delta",
        "personalization_summary",
        "quality_summary",
        "summary_line",
    ):
        value = mapping.get(key)
        if value is not None:
            parts.append(f"{key}={_format_scalar(value)}")
    if not parts:
        return None
    return "promotion compare: " + " | ".join(parts)


def _format_recent_training_snapshot(snapshot: Any) -> list[str] | None:
    mapping = _coerce_mapping(snapshot)
    if mapping is None:
        return None

    lines: list[str] = []
    parts: list[str] = []
    version = _pick_first(mapping, "version")
    if version is not None:
        parts.append(f"version={_format_scalar(version)}")
    state = _pick_first(mapping, "state")
    if state is not None:
        parts.append(f"state={_format_scalar(state)}")
    execution_backend = _pick_first(mapping, "execution_backend")
    if execution_backend is not None:
        parts.append(f"execution_backend={_format_scalar(execution_backend)}")
    executor_mode = _pick_first(mapping, "executor_mode")
    if executor_mode is not None:
        parts.append(f"executor_mode={_format_scalar(executor_mode)}")
    if parts:
        lines.append("recent training: " + " | ".join(parts))

    incremental_line = _format_incremental_context(mapping.get("incremental_context") or mapping)
    if incremental_line is not None:
        lines.append(incremental_line)

    job_line = _format_real_execution_summary(
        mapping.get("real_execution_summary") or mapping.get("job_execution_summary") or mapping.get("job_execution"),
        executor_mode=_pick_first(mapping, "executor_mode"),
    )
    if job_line is not None:
        lines.append(job_line)

    export_exec_line = _format_export_toolchain_summary(
        mapping.get("export_toolchain_summary") or mapping.get("export_execution")
    )
    if export_exec_line is not None:
        lines.append(export_exec_line)
    return lines or None


def _format_doctor_trainer_deps(runtime: Any) -> str | None:
    mapping = _coerce_mapping(runtime)
    if mapping is None:
        return None

    installed_packages = _coerce_mapping(mapping.get("installed_packages")) or {}
    if not installed_packages:
        return None

    required_packages = ("torch", "transformers", "peft", "accelerate", "trl", "datasets")
    optional_packages = ("unsloth", "mlx", "mlx_lm")
    ready = all(bool(installed_packages.get(name, False)) for name in required_packages)

    parts = [f"ready={_format_scalar(ready)}"]
    missing = [name for name in required_packages if not installed_packages.get(name, False)]
    if missing:
        parts.append(f"missing={_format_scalar(missing)}")
    for name in (*required_packages, *optional_packages):
        if name in installed_packages:
            parts.append(f"{name}={_format_scalar(installed_packages.get(name))}")
    python_version = _pick_first(mapping, "python_version")
    if python_version is not None:
        parts.append(f"python_version={_format_scalar(python_version)}")
    requires_python = _format_doctor_package_mapping(
        mapping.get("requires_python"),
        preferred_order=(*required_packages, *optional_packages),
    )
    if requires_python is not None:
        parts.append(f"requires_python={requires_python}")
    python_supported = _format_doctor_package_mapping(
        mapping.get("python_supported"),
        preferred_order=(*required_packages, *optional_packages),
    )
    if python_supported is not None:
        parts.append(f"python_supported={python_supported}")
    runtime_device = _pick_first(mapping, "runtime_device")
    if runtime_device is not None:
        parts.append(f"runtime_device={_format_scalar(runtime_device)}")
    return "trainer deps: " + " | ".join(parts)


def _format_doctor_snapshot_summary(snapshot: Any, *, include_latest: bool = False) -> str:
    mapping = _coerce_mapping(snapshot)
    if mapping is None:
        return "n/a"

    parts: list[str] = []
    version = _pick_first(mapping, "version")
    if version is not None:
        parts.append(f"version={_format_scalar(version)}")
    state = _pick_first(mapping, "state")
    if state is not None:
        parts.append(f"state={_format_scalar(state)}")
    if include_latest:
        latest = _pick_first(mapping, "latest")
        if latest is not None:
            parts.append(f"latest={_format_scalar(latest)}")
    samples = _pick_first(mapping, "num_samples", "samples")
    if samples is not None:
        parts.append(f"samples={_format_scalar(samples)}")
    artifact_format = _pick_first(mapping, "artifact_format", "format")
    if artifact_format is not None:
        parts.append(f"format={_format_scalar(artifact_format)}")
    return " | ".join(parts) if parts else "n/a"


def _format_doctor_local_model(workspace: str | None = None, base_model: str | None = None) -> str | None:
    manifest = _load_latest_adapter_manifest(workspace)
    manifest_map = _coerce_mapping(manifest)
    requested_base_model = base_model
    if requested_base_model is None and manifest_map is not None:
        requested_base_model = _pick_first(manifest_map, "base_model")

    if requested_base_model is None:
        return "local model: available=no | requested_base_model=n/a | reason=no base model configured"

    local_source = _optional_module_call(
        "pfe_core.trainer.executors",
        "_resolve_real_local_model_source",
        {"base_model": requested_base_model},
    )
    local_source_map = _coerce_mapping(local_source)
    if local_source_map is None:
        local_source_map = {
            "available": False,
            "requested_base_model": requested_base_model,
            "source_kind": "unavailable",
            "source_path": None,
            "config_path": None,
            "load_mode": "unavailable",
            "reason": "local model probe unavailable",
        }

    parts: list[str] = []
    for key in ("available", "requested_base_model", "source_kind", "source_path", "config_path", "load_mode", "reason"):
        value = local_source_map.get(key)
        if value is not None:
            parts.append(f"{key}={_format_scalar(value)}")
    if not parts:
        return None
    return "local model: " + " | ".join(parts)


def _format_doctor_export_tool() -> str | None:
    resolution = _optional_module_call("pfe_core.inference.export_runtime", "resolve_llama_cpp_export_tool_path")
    validation = None
    if resolution is not None:
        validation = _optional_module_call(
            "pfe_core.inference.export_runtime",
            "validate_llama_cpp_export_toolchain",
            resolution,
        )

    mapping = _coerce_mapping(validation) or _coerce_mapping(resolution)
    if mapping is None:
        return "llama.cpp export tool: status=n/a | allowed=n/a | reason=probe unavailable"

    parts: list[str] = []
    if "status" in mapping:
        parts.append(f"status={_format_scalar(mapping.get('status'))}")
    if "allowed" in mapping:
        parts.append(f"allowed={_format_scalar(mapping.get('allowed'))}")
    if "resolved_path" in mapping:
        parts.append(f"resolved_path={_format_scalar(mapping.get('resolved_path'))}")
    if "reason" in mapping:
        parts.append(f"reason={_format_scalar(mapping.get('reason'))}")
    if not parts:
        return None
    return "llama.cpp export tool: " + " | ".join(parts)


def _format_doctor_package_mapping(mapping: Any, *, preferred_order: Sequence[str]) -> str | None:
    coerced = _coerce_mapping(mapping)
    if coerced is None:
        return None

    parts: list[str] = []
    seen: set[str] = set()
    for name in preferred_order:
        if name in coerced:
            parts.append(f"{name}={_format_scalar(coerced.get(name))}")
            seen.add(name)
    for name in sorted(coerced):
        if name not in seen:
            parts.append(f"{name}={_format_scalar(coerced.get(name))}")
    return ", ".join(parts) if parts else None


def _format_doctor_blocked_capabilities(
    *,
    trainer_line: str | None,
    local_model_line: str | None,
    export_tool_line: str | None,
    latest_snapshot: Any,
    recent_snapshot: Any,
) -> str:
    blocked: list[str] = []

    if local_model_line is None or "available=yes" not in local_model_line:
        blocked.extend(["train", "eval", "serve"])

    latest_snapshot_map = _coerce_mapping(latest_snapshot)
    recent_snapshot_map = _coerce_mapping(recent_snapshot)
    if latest_snapshot_map is None and recent_snapshot_map is None:
        blocked.append("eval")

    if export_tool_line is None or "allowed=yes" not in export_tool_line:
        blocked.append("export")

    if trainer_line is None or "ready=yes" not in trainer_line:
        blocked.append("train")

    unique_blocked = list(dict.fromkeys(blocked))
    if not unique_blocked:
        return "blocked capabilities: none"
    return "blocked capabilities: " + ", ".join(unique_blocked)


def _format_doctor_next_steps(
    *,
    trainer_line: str | None,
    local_model_line: str | None,
    export_tool_line: str | None,
    latest_snapshot: Any,
    recent_snapshot: Any,
) -> str:
    steps: list[str] = []

    if local_model_line is None or "available=yes" not in local_model_line:
        steps.append("set a base_model in the latest adapter manifest or pass --base-model")

    if trainer_line is None or "ready=yes" not in trainer_line:
        steps.append("install torch, transformers, peft, accelerate, trl, and datasets")

    latest_snapshot_map = _coerce_mapping(latest_snapshot)
    recent_snapshot_map = _coerce_mapping(recent_snapshot)
    if latest_snapshot_map is None and recent_snapshot_map is None:
        steps.append("train an adapter to create a workspace snapshot")
    elif latest_snapshot_map is None and recent_snapshot_map is not None:
        steps.append("promote the recent adapter once it passes eval")

    if export_tool_line is None or "allowed=yes" not in export_tool_line:
        steps.append("put the llama.cpp export tool on PATH or configure its location")

    if not steps:
        steps.append("run pfe train or pfe eval as needed")

    return "next steps: " + "; ".join(steps)


def _format_doctor_adapter_home(workspace: str | None = None) -> str:
    home = _pfe_home(workspace)
    latest_snapshot = _lookup_adapter_snapshot("latest", workspace=workspace)
    recent_snapshot = _lookup_recent_adapter_snapshot(workspace=workspace)
    parts = [
        f"adapter home: home={_format_scalar(home)} | latest promoted={_format_doctor_snapshot_summary(latest_snapshot, include_latest=True)} "
        f"| recent training={_format_doctor_snapshot_summary(recent_snapshot, include_latest=True)}"
    ]
    latest_export_artifact_line = _format_adapter_export_artifact_line("latest export artifact", latest_snapshot)
    if latest_export_artifact_line is not None:
        parts.append(latest_export_artifact_line)
    recent_export_artifact_line = _format_adapter_export_artifact_line("recent export artifact", recent_snapshot)
    if recent_export_artifact_line is not None:
        parts.append(recent_export_artifact_line)
    return "\n".join(parts)


def _format_doctor_pii_compliance(workspace: str | None = None) -> str | None:
    """Check PII compliance for training samples."""
    try:
        from pfe_core.data_policy import audit_pii_exposure
        from pfe_core.storage import list_samples
    except Exception:
        return None
    try:
        samples = list_samples(limit=100)
        report = audit_pii_exposure(samples)
        if report.pii_detected_count == 0:
            return "pii compliance: clean"
        return (
            f"pii compliance: detected={report.pii_detected_count} "
            f"severity={report.severity} types={sorted(report.pii_types_found.keys())}"
        )
    except Exception:
        return None


def _format_doctor_training_audit(workspace: str | None = None) -> str | None:
    """Check training prohibition audit status."""
    try:
        from pfe_core.trainer.training_auditor import TrainingAuditor
        from pfe_core.storage import list_samples
    except Exception:
        return None
    try:
        samples = list_samples(limit=100)
        auditor = TrainingAuditor()
        report = auditor.audit(samples)
        if report.severity == "low" and not report.blocked:
            return "training audit: clean"
        return (
            f"training audit: severity={report.severity} blocked={report.blocked} "
            f"reasons={report.reasons}"
        )
    except Exception:
        return None


def _format_doctor_signal_chain_integrity(workspace: str | None = None) -> str | None:
    """Check signal chain integrity via observability traces."""
    try:
        from pfe_core.observability.trace import TraceStore
    except Exception:
        return None
    try:
        store = TraceStore()
        recent_ids = store.list_recent_signal_ids(limit=5)
        if not recent_ids:
            return "signal chain: no recent traces"
        complete_count = 0
        for sid in recent_ids:
            trace = store.load_signal_trace(sid)
            if trace is not None and trace.nodes:
                node_names = {n.node for n in trace.nodes}
                if "collect" in node_names:
                    complete_count += 1
        return f"signal chain: recent={len(recent_ids)} traced={complete_count}"
    except Exception:
        return None


def _format_doctor(*, workspace: str | None = None, base_model: str | None = None) -> str:
    lines = ["PFE doctor"]

    runtime = _optional_module_call("pfe_core.trainer.runtime", "detect_trainer_runtime")
    trainer_line = _format_doctor_trainer_deps(runtime)
    if trainer_line is not None:
        lines.append(trainer_line)

    local_model_line = _format_doctor_local_model(workspace, base_model)
    if local_model_line is not None:
        lines.append(local_model_line)

    export_tool_line = _format_doctor_export_tool()
    if export_tool_line is not None:
        lines.append(export_tool_line)

    latest_snapshot = _lookup_adapter_snapshot("latest", workspace=workspace)
    recent_snapshot = _lookup_recent_adapter_snapshot(workspace=workspace)
    lines.append(_format_doctor_blocked_capabilities(
        trainer_line=trainer_line,
        local_model_line=local_model_line,
        export_tool_line=export_tool_line,
        latest_snapshot=latest_snapshot,
        recent_snapshot=recent_snapshot,
    ))
    lines.append(
        _format_doctor_next_steps(
            trainer_line=trainer_line,
            local_model_line=local_model_line,
            export_tool_line=export_tool_line,
            latest_snapshot=latest_snapshot,
            recent_snapshot=recent_snapshot,
        )
    )
    lines.append(
        "capability boundaries: "
        "train/core | eval/core | serve/core | generate/heuristic | distill/heuristic | profile/heuristic | route/heuristic"
    )
    lines.append("user modeling: runtime=user_memory | analysis=user_profile")
    lines.append(_format_doctor_adapter_home(workspace))

    # P2-G: Privacy / security / observability checks
    pii_line = _format_doctor_pii_compliance(workspace=workspace)
    if pii_line is not None:
        lines.append(pii_line)

    training_audit_line = _format_doctor_training_audit(workspace=workspace)
    if training_audit_line is not None:
        lines.append(training_audit_line)

    signal_chain_line = _format_doctor_signal_chain_integrity(workspace=workspace)
    if signal_chain_line is not None:
        lines.append(signal_chain_line)

    return "\n".join(lines)


def _format_eval_result(result: Any, *, workspace: str | None = None) -> str:
    # Matrix theme - default style
    return formatters_matrix.format_eval_result_matrix(result, workspace=workspace)

def _format_eval_result_legacy(result: Any, *, workspace: str | None = None) -> str:
    """Legacy plain text formatter (kept for reference)."""
    mapping = _coerce_mapping(result)
    if mapping is None and isinstance(result, str):
        try:
            loaded = json.loads(result)
        except Exception:
            loaded = None
        mapping = _coerce_mapping(loaded)
        if mapping is None:
            return result.strip() if result.strip() else _format_scalar(result)
    if mapping is None:
        return _format_scalar(result)

    lines = ["PFE eval"]
    left_adapter = _pick_first(mapping, "left_adapter", "left_adapter_version")
    right_adapter = _pick_first(mapping, "right_adapter", "right_adapter_version")
    winner = _pick_first(mapping, "winner")
    overall_delta = _pick_first(mapping, "overall_delta")
    if left_adapter is not None or right_adapter is not None:
        parts = []
        if left_adapter is not None:
            parts.append(f"left_adapter={_format_scalar(left_adapter)}")
        if right_adapter is not None:
            parts.append(f"right_adapter={_format_scalar(right_adapter)}")
        comparison = _pick_first(mapping, "comparison")
        if comparison is not None:
            parts.append(f"comparison={_format_scalar(comparison)}")
        if winner is not None:
            parts.append(f"winner={_format_scalar(winner)}")
        recommendation = _pick_first(mapping, "recommendation")
        if recommendation is not None:
            parts.append(f"recommendation={_format_scalar(recommendation)}")
        if overall_delta is not None:
            parts.append(f"overall_delta={_format_scalar(overall_delta)}")
        personalization_summary = _pick_first(mapping, "personalization_summary")
        if personalization_summary:
            parts.append(f"personalization_summary={_format_scalar(personalization_summary)}")
        quality_summary = _pick_first(mapping, "quality_summary")
        if quality_summary:
            parts.append(f"quality_summary={_format_scalar(quality_summary)}")
        summary_line = _pick_first(mapping, "summary_line")
        if summary_line:
            parts.append(f"summary_line={_format_scalar(summary_line)}")
        if parts:
            lines.append("compare: " + " | ".join(parts))
    version = _pick_first(mapping, "adapter_version", "version")
    base_model = _pick_first(mapping, "base_model")
    num_test_samples = _pick_first(mapping, "num_test_samples", "num_samples")
    if version is not None or base_model is not None or num_test_samples is not None:
        parts = []
        if version is not None:
            parts.append(f"adapter_version={_format_scalar(version)}")
        if base_model is not None:
            parts.append(f"base_model={_format_scalar(base_model)}")
        if num_test_samples is not None:
            parts.append(f"num_test_samples={_format_scalar(num_test_samples)}")
        lines.append(" | ".join(parts))

    adapter_snapshot = _lookup_adapter_snapshot(str(version) if version is not None else None, workspace=workspace)
    adapter_line = _format_adapter_snapshot_line("evaluated adapter", adapter_snapshot, include_latest=True)
    if adapter_line is not None:
        lines.append(adapter_line)

    recommendation = _pick_first(mapping, "recommendation")
    comparison = _pick_first(mapping, "comparison")
    if recommendation is not None or comparison is not None:
        parts = []
        if recommendation is not None:
            parts.append(f"recommendation={_format_scalar(recommendation)}")
        if comparison is not None:
            parts.append(f"comparison={_format_scalar(comparison)}")
        lines.append("result: " + " | ".join(parts))

    personalization_summary = _pick_first(mapping, "personalization_summary")
    quality_summary = _pick_first(mapping, "quality_summary")
    summary_line = _pick_first(mapping, "summary_line")
    if personalization_summary or quality_summary or summary_line:
        parts = []
        if personalization_summary:
            parts.append(f"personalization_summary={_format_scalar(personalization_summary)}")
        if quality_summary:
            parts.append(f"quality_summary={_format_scalar(quality_summary)}")
        if summary_line:
            parts.append(f"summary_line={_format_scalar(summary_line)}")
        lines.append("compare detail: " + " | ".join(parts))

    scores = _coerce_mapping(mapping.get("scores"))
    if not scores:
        scores = _coerce_mapping(mapping.get("score_deltas"))
    if scores:
        score_parts = []
        ordered_scores = _ordered_eval_scores(scores)
        if ordered_scores:
            for key, value in ordered_scores:
                score_parts.append(f"{key}={_format_scalar(value)}")
        if score_parts:
            label = "score_deltas" if mapping.get("score_deltas") is not None and mapping.get("scores") is None else "scores"
            lines.append(f"{label}: " + " | ".join(score_parts))

    details = mapping.get("details")
    if isinstance(details, Sequence) and not isinstance(details, (str, bytes, bytearray)):
        lines.append(f"details: {_format_scalar(len(details))} item(s)")
    return "\n".join(lines)


def _run_handler(
    command_name: str,
    handler: Callable[..., Any],
    formatter: Callable[[Any], str] | None = None,
    on_result: Callable[[Any], None] | None = None,
    **kwargs: Any,
) -> None:
    """Execute a handler with short domain-error messages and full propagation for unknown bugs."""

    try:
        result = handler(**kwargs)
    except typer.Exit:
        raise
    except Exception as exc:
        friendly = _friendly_exception_message(exc)
        if friendly is not None:
            typer.secho(friendly, err=True, fg=typer.colors.RED)
            raise typer.Exit(code=1)
        raise

    if result is not None:
        if on_result is not None:
            try:
                on_result(result)
            except Exception:
                pass
        typer.echo(formatter(result) if formatter is not None else result)


def _run_handler_json(command_name: str, handler: Callable[..., Any], **kwargs: Any) -> None:
    try:
        result = handler(**kwargs)
    except typer.Exit:
        raise
    except Exception as exc:
        friendly = _friendly_exception_message(exc)
        if friendly is not None:
            typer.secho(friendly, err=True, fg=typer.colors.RED)
            raise typer.Exit(code=1)
        raise

    mapping = _coerce_mapping(result)
    payload: Any = mapping if mapping is not None else result
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def _console_chat_text(result: Any) -> str:
    mapping = _coerce_mapping(result)
    if not mapping:
        return _format_scalar(result)
    choices = list(mapping.get("choices") or [])
    for choice in choices:
        if not isinstance(choice, Mapping):
            continue
        message = choice.get("message")
        if isinstance(message, Mapping):
            content = message.get("content")
            if content:
                return str(content)
    return ""


def _append_console_line(lines: list[dict[str, str]], *, role: str, content: str, limit: int = 10) -> None:
    text = str(content or "").strip()
    if not text:
        return
    lines.append({"role": role, "content": text})
    if len(lines) > limit:
        del lines[:-limit]


def _console_snapshot_payload(handler: Callable[..., Any], *, workspace: str | None) -> dict[str, Any]:
    result = handler(workspace=workspace)
    mapping = _coerce_mapping(result)
    return mapping if mapping is not None else {"status_result": str(result)}


def _console_apply_edit(text: str, cursor: int, event: str, value: str | None = None) -> tuple[str, int]:
    current = str(text or "")
    cursor = max(0, min(int(cursor), len(current)))
    if event == "insert" and value:
        updated = current[:cursor] + value + current[cursor:]
        return updated, cursor + len(value)
    if event == "left":
        return current, max(0, cursor - 1)
    if event == "right":
        return current, min(len(current), cursor + 1)
    if event == "home":
        return current, 0
    if event == "end":
        return current, len(current)
    if event == "backspace":
        if cursor <= 0:
            return current, cursor
        updated = current[: cursor - 1] + current[cursor:]
        return updated, cursor - 1
    if event == "delete":
        if cursor >= len(current):
            return current, cursor
        updated = current[:cursor] + current[cursor + 1 :]
        return updated, cursor
    if event == "clear":
        return "", 0
    if event == "clear_to_end":
        if cursor >= len(current):
            return current, cursor
        return current[:cursor], cursor
    if event == "word_backspace":
        if cursor <= 0:
            return current, cursor
        trimmed = current[:cursor].rstrip()
        suffix = current[cursor:]
        last_space = trimmed.rfind(" ")
        start = 0 if last_space < 0 else last_space + 1
        updated = current[:start] + suffix
        return updated, start
    return current, cursor


def _console_apply_history(
    text: str,
    cursor: int,
    *,
    history: Sequence[str] | None,
    history_index: int | None,
    history_draft: str,
    event: str,
) -> tuple[str, int, int | None, str]:
    history_items = [str(item) for item in (history or []) if str(item).strip()]
    current = str(text or "")
    cursor = max(0, min(int(cursor), len(current)))
    if not history_items:
        return current, cursor, history_index, history_draft

    if event == "up":
        if history_index is None:
            history_draft = current
            history_index = len(history_items) - 1
        elif history_index > 0:
            history_index -= 1
        current = history_items[history_index]
        return current, len(current), history_index, history_draft

    if event == "down" and history_index is not None:
        if history_index < len(history_items) - 1:
            history_index += 1
            current = history_items[history_index]
        else:
            history_index = None
            current = history_draft
        return current, len(current), history_index, history_draft

    return current, cursor, history_index, history_draft


def _console_read_input(
    prompt_label: str,
    *,
    refresh_seconds: float,
    refresh_callback: Callable[[str, int], None],
    history: Sequence[str] | None = None,
) -> str:
    if not sys.stdin.isatty():
        return str(typer.prompt(prompt_label))

    import termios
    import tty

    fd = sys.stdin.fileno()
    previous_settings = termios.tcgetattr(fd)
    text = ""
    cursor = 0
    history_items = [str(item) for item in (history or []) if str(item).strip()]
    history_index: int | None = None
    history_draft = ""
    refresh_callback("", 0)
    try:
        tty.setraw(fd)
        while True:
            ready, _, _ = select.select([fd], [], [], max(refresh_seconds, 0.1))
            if not ready:
                refresh_callback(text, cursor)
                continue

            chunk = os.read(fd, 1)
            if not chunk:
                raise EOFError
            char = chunk.decode("utf-8", errors="ignore")

            if char in {"\r", "\n"}:
                return text
            if char == "\x03":
                raise KeyboardInterrupt
            if char == "\x04":
                if text:
                    continue
                raise EOFError
            if char == "\x01":
                text, cursor = _console_apply_edit(text, cursor, "home")
                refresh_callback(text, cursor)
                continue
            if char == "\x05":
                text, cursor = _console_apply_edit(text, cursor, "end")
                refresh_callback(text, cursor)
                continue
            if char == "\x15":
                text, cursor = _console_apply_edit(text, cursor, "clear")
                history_index = None
                refresh_callback(text, cursor)
                continue
            if char == "\x17":
                text, cursor = _console_apply_edit(text, cursor, "word_backspace")
                history_index = None
                refresh_callback(text, cursor)
                continue
            if char == "\x0b":
                text, cursor = _console_apply_edit(text, cursor, "clear_to_end")
                refresh_callback(text, cursor)
                continue
            if char in {"\x7f", "\b"}:
                text, cursor = _console_apply_edit(text, cursor, "backspace")
                history_index = None
                refresh_callback(text, cursor)
                continue
            if char == "\x1b":
                ready_more, _, _ = select.select([fd], [], [], 0.01)
                sequence = ""
                while ready_more:
                    next_chunk = os.read(fd, 1)
                    if not next_chunk:
                        break
                    sequence += next_chunk.decode("utf-8", errors="ignore")
                    ready_more, _, _ = select.select([fd], [], [], 0.01)
                    if sequence and sequence[-1].isalpha():
                        break
                    if sequence.endswith("~"):
                        break
                if sequence == "[A":
                    text, cursor, history_index, history_draft = _console_apply_history(
                        text,
                        cursor,
                        history=history_items,
                        history_index=history_index,
                        history_draft=history_draft,
                        event="up",
                    )
                elif sequence == "[B":
                    text, cursor, history_index, history_draft = _console_apply_history(
                        text,
                        cursor,
                        history=history_items,
                        history_index=history_index,
                        history_draft=history_draft,
                        event="down",
                    )
                elif sequence == "[C":
                    text, cursor = _console_apply_edit(text, cursor, "right")
                elif sequence == "[D":
                    text, cursor = _console_apply_edit(text, cursor, "left")
                elif sequence == "[H":
                    text, cursor = _console_apply_edit(text, cursor, "home")
                elif sequence == "[F":
                    text, cursor = _console_apply_edit(text, cursor, "end")
                elif sequence == "[3~":
                    text, cursor = _console_apply_edit(text, cursor, "delete")
                refresh_callback(text, cursor)
                continue
            if char.isprintable():
                text, cursor = _console_apply_edit(text, cursor, "insert", char)
                history_index = None
                refresh_callback(text, cursor)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, previous_settings)


def _console_help_text() -> str:
    return "\n".join(
        [
            "PFE Console slash commands:",
            "/help - show this help",
            "/status - show the full status formatter",
            "/status compact - show a shorter status digest",
            "/sum - alias for /status compact",
            "/ops - show the operations console digest",
            "/os - alias for /ops",
            "/ops dashboard - show the operations dashboard digest",
            "/dash - alias for /ops dashboard",
            "/ops dash - alias for /ops dashboard",
            "/ops alerts - show the operations alerts digest",
            "/alerts - alias for /ops alerts",
            "/ops policy - show the operations alert policy digest",
            "/policy - alias for /ops policy",
            "/ops pol - alias for /ops policy",
            "/trigger - show the auto-train trigger summary",
            "/trig - alias for /trigger summary",
            "/gate - show the auto-train gate summary",
            "/runtime - show the runtime stability summary",
            "/rt - alias for /runtime summary",
            "/event stream - show the operations event stream digest",
            "/event - alias for /event stream",
            "/do - run the primary focus action when it is unambiguous",
            "/see - open the primary focus inspection view",
            "/promote [note] - promote the current candidate when available",
            "/archive [note] - archive the current candidate when available",
            "/approve [note] - approve the next queued review item",
            "/reject [note] - reject the next queued review item",
            "/approve <id> [note] - approve a specific queue item by id",
            "/reject <id> [note] - reject a specific queue item by id",
            "/process - process the next queued train item",
            "/batch - process the next train queue batch",
            "/until-idle - process train queue until idle",
            "/retry - retry the current auto-train trigger evaluation",
            "/trigger-train - alias for /retry, trigger training when threshold is met",
            "/train - alias for /trigger-train, trigger training when threshold is met",
            "/dpo - run DPO training from signals",
            "/generate <scenario> [style] - generate cold-start samples",
            "/rollback [version] - rollback to a previous adapter version",
            "/reset - reset the auto-train trigger state",
            "/recover daemon - recover the worker daemon",
            "/force-recovery - force runtime recovery",
            "/restart daemon - restart the worker daemon",
            "/stop daemon - stop the worker daemon",
            "/start daemon - start the worker daemon",
            "/stop runner - stop the worker runner",
            "/doctor - show doctor output for this session",
            "/eval - trigger evaluation",
            "/distill - run distillation",
            "/serve - show serve preview for this session",
            "/candidate - show candidate timeline",
            "/cand - alias for /candidate summary",
            "/cand sum - alias for /candidate summary",
            "/cand tl - alias for /candidate timeline",
            "/cand hist - alias for /candidate history",
            "/candidate timeline - show candidate timeline explicitly",
            "/candidate history - show candidate history",
            "/candidate summary - show the candidate summary",
            "/queue - show the train queue summary",
            "/qs - alias for /queue summary",
            "/queue summary - show the train queue summary",
            "/queue sum - alias for /queue summary",
            "/queue hist - alias for /queue history",
            "/queue history - show train queue history",
            "/runner - show the worker runner summary",
            "/rs - alias for /runner summary",
            "/runner summary - show the worker runner summary",
            "/runner sum - alias for /runner summary",
            "/runner tl - alias for /runner timeline",
            "/runner timeline - show the worker runner timeline",
            "/runner hist - alias for /runner history",
            "/daemon - show worker daemon status",
            "/ds - alias for /daemon summary",
            "/daemon sum - alias for /daemon summary",
            "/daemon tl - alias for /daemon timeline",
            "/daemon history - show worker daemon history",
            "/daemon hist - alias for /daemon history",
            "/daemon summary - show the daemon summary",
            "/daemon timeline - show the worker daemon timeline",
            "/runner history - show worker runner history",
            "/settings - show current console session settings",
            "/workspace <name> - switch the workspace for this session",
            "/model <id> - change the chat model for this session",
            "/adapter <version> - change the adapter for this session",
            "/list - list adapter versions",
            "/temperature <value> - change chat temperature for this session",
            "/max-tokens <n|auto> - change chat max tokens for this session",
            "/real-local on|off - toggle real local inference for chat",
            "/refresh <seconds> - change sidebar refresh cadence",
            "/chat - switch to chat mode",
            "/cmd - switch to command mode",
            "/mode chat|command - switch input mode explicitly",
            "/like - accept the previous assistant response",
            "/dislike - reject the previous assistant response",
            "/fix <text> - submit edited version of the previous response",
            "/again - request regeneration of the previous response",
            "/clear - clear the local console transcript",
            "/quit - exit interactive mode",
            "",
            "Plain text input uses the local chat_completion pipeline.",
        ]
    )


def _console_dashboard_focus(payload: Mapping[str, Any] | None = None) -> str:
    operations_dashboard = _coerce_mapping((payload or {}).get("operations_dashboard")) or {}
    operations_console = _coerce_mapping((payload or {}).get("operations_console")) or {}
    operations_overview = _coerce_mapping((payload or {}).get("operations_overview")) or {}
    current_focus = str(operations_dashboard.get("current_focus") or "").strip().lower()
    if current_focus not in {"", "none", "idle", "stable"}:
        return current_focus
    for raw_focus in (
        operations_dashboard.get("monitor_focus"),
        operations_console.get("monitor_focus"),
        operations_overview.get("monitor_focus"),
    ):
        monitor_focus = str(raw_focus or "").strip().lower()
        if monitor_focus:
            return monitor_focus
    return current_focus or "none"


def _console_focus_actions(payload: Mapping[str, Any] | None = None) -> dict[str, str | None]:
    operations_dashboard = _coerce_mapping((payload or {}).get("operations_dashboard")) or {}
    operations_console = _coerce_mapping((payload or {}).get("operations_console")) or {}
    alert_policy = _coerce_mapping((payload or {}).get("operations_alert_policy")) or {}
    candidate_summary = _coerce_mapping((payload or {}).get("candidate_summary")) or {}
    current_focus = _console_dashboard_focus(payload)
    required_action = str(alert_policy.get("required_action") or "")
    secondary_action_values: list[str] = []
    for raw_action in [
        alert_policy.get("secondary_action"),
        *list(alert_policy.get("secondary_actions") or []),
    ]:
        text = str(raw_action or "").strip()
        if text and text not in secondary_action_values:
            secondary_action_values.append(text)

    def _action_mapping(action_name: str) -> dict[str, str | None] | None:
        normalized_action = str(action_name or "").strip().lower()
        if not normalized_action:
            return None
        if normalized_action == "promote_candidate":
            return {
                "primary_label": "/promote",
                "primary_exec": "promote",
                "secondary_label": "/candidate /cand sum",
                "secondary_exec": "candidate",
            }
        if normalized_action == "archive_candidate":
            return {
                "primary_label": "/archive",
                "primary_exec": "archive",
                "secondary_label": "/candidate /cand sum",
                "secondary_exec": "candidate",
            }
        if normalized_action == "inspect_candidate_status":
            return {
                "primary_label": "/candidate",
                "primary_exec": "candidate",
                "secondary_label": "/cand tl",
                "secondary_exec": "cand tl",
            }
        if normalized_action == "inspect_candidate_timeline":
            return {
                "primary_label": "/cand tl",
                "primary_exec": "cand tl",
                "secondary_label": "/candidate",
                "secondary_exec": "candidate",
            }
        if normalized_action == "process_next_queue_item":
            return {
                "primary_label": "/process",
                "primary_exec": "process",
                "secondary_label": "/queue /qs",
                "secondary_exec": "queue",
            }
        if normalized_action == "review_queue_confirmation":
            return {
                "primary_label": "/approve or /reject",
                "primary_exec": None,
                "secondary_label": "/gate /trigger",
                "secondary_exec": "gate",
            }
        if normalized_action == "recover_worker_daemon":
            return {
                "primary_label": "/recover daemon",
                "primary_exec": "recover daemon",
                "secondary_label": "/runtime /daemon",
                "secondary_exec": "runtime",
            }
        if normalized_action == "inspect_daemon_restart_policy":
            return {
                "primary_label": "/runtime",
                "primary_exec": "runtime",
                "secondary_label": "/alerts /daemon",
                "secondary_exec": "alerts",
            }
        if normalized_action == "inspect_runtime_stability":
            return {
                "primary_label": "/runtime",
                "primary_exec": "runtime",
                "secondary_label": "/runner hist",
                "secondary_exec": "runner hist",
            }
        if normalized_action == "inspect_worker_runner_history":
            return {
                "primary_label": "/runner hist",
                "primary_exec": "runner hist",
                "secondary_label": "/runtime",
                "secondary_exec": "runtime",
            }
        if normalized_action == "inspect_daemon_status":
            return {
                "primary_label": "/daemon",
                "primary_exec": "daemon",
                "secondary_label": "/runtime",
                "secondary_exec": "runtime",
            }
        if normalized_action in {"inspect_daemon_heartbeat", "inspect_worker_stale_lock", "wait_for_runner_shutdown"}:
            return {
                "primary_label": "/runtime",
                "primary_exec": "runtime",
                "secondary_label": "/runner /daemon",
                "secondary_exec": "runtime",
            }
        if normalized_action in {"enable_auto_evaluate", "inspect_auto_train_policy"}:
            return {
                "primary_label": "/policy",
                "primary_exec": "policy",
                "secondary_label": "/gate",
                "secondary_exec": "gate",
            }
        if normalized_action == "inspect_auto_train_gate":
            return {
                "primary_label": "/gate",
                "primary_exec": "gate",
                "secondary_label": "/policy",
                "secondary_exec": "policy",
            }
        if normalized_action == "inspect_auto_train_trigger":
            return {
                "primary_label": "/trigger",
                "primary_exec": "trigger",
                "secondary_label": "/gate",
                "secondary_exec": "gate",
            }
        if normalized_action == "wait_for_queue_completion":
            return {
                "primary_label": "/trigger",
                "primary_exec": "trigger",
                "secondary_label": "/queue /qs",
                "secondary_exec": "queue",
            }
        if normalized_action in {"collect_more_signal_samples", "collect_holdout_samples"}:
            return {
                "primary_label": "/gate",
                "primary_exec": "gate",
                "secondary_label": "/trigger /policy",
                "secondary_exec": "trigger",
            }
        if normalized_action == "wait_for_retrain_interval":
            return {
                "primary_label": "/trigger",
                "primary_exec": "trigger",
                "secondary_label": "/gate /policy",
                "secondary_exec": "gate",
            }
        if normalized_action == "wait_for_failure_backoff":
            return {
                "primary_label": "/retry",
                "primary_exec": "retry",
                "secondary_label": "/trigger /gate",
                "secondary_exec": "trigger",
            }
        if normalized_action == "inspect_compare_evaluation":
            return {
                "primary_label": "/candidate",
                "primary_exec": "candidate",
                "secondary_label": "/gate /trigger",
                "secondary_exec": "gate",
            }
        if normalized_action == "inspect_candidate_gate":
            return {
                "primary_label": "/gate",
                "primary_exec": "gate",
                "secondary_label": "/candidate /trigger",
                "secondary_exec": "candidate",
            }
        if normalized_action == "rollback_candidate":
            return {
                "primary_label": "/rollback",
                "primary_exec": "rollback",
                "secondary_label": "/candidate /archive",
                "secondary_exec": "candidate",
            }
        if normalized_action == "evaluate":
            return {
                "primary_label": "/eval",
                "primary_exec": "eval",
                "secondary_label": "/candidate /gate",
                "secondary_exec": "candidate",
            }
        if normalized_action == "run_distillation":
            return {
                "primary_label": "/distill",
                "primary_exec": "distill",
                "secondary_label": "/gate /trigger",
                "secondary_exec": "gate",
            }
        if normalized_action == "force_recovery":
            return {
                "primary_label": "/force-recovery",
                "primary_exec": "force-recovery",
                "secondary_label": "/runtime /recover daemon",
                "secondary_exec": "runtime",
            }
        if normalized_action == "process_train_queue_batch":
            return {
                "primary_label": "/batch",
                "primary_exec": "batch",
                "secondary_label": "/queue /process",
                "secondary_exec": "queue",
            }
        if normalized_action == "process_train_queue_until_idle":
            return {
                "primary_label": "/until-idle",
                "primary_exec": "until-idle",
                "secondary_label": "/queue /process",
                "secondary_exec": "queue",
            }
        if normalized_action == "stop_train_queue_daemon":
            return {
                "primary_label": "/stop daemon",
                "primary_exec": "stop daemon",
                "secondary_label": "/daemon /runtime",
                "secondary_exec": "daemon",
            }
        if normalized_action == "start_train_queue_daemon":
            return {
                "primary_label": "/start daemon",
                "primary_exec": "start daemon",
                "secondary_label": "/daemon /runtime",
                "secondary_exec": "daemon",
            }
        if normalized_action == "stop_train_queue_worker_runner":
            return {
                "primary_label": "/stop runner",
                "primary_exec": "stop runner",
                "secondary_label": "/runner /runtime",
                "secondary_exec": "runner",
            }
        return None

    mapped_required = _action_mapping(required_action)
    if mapped_required is not None:
        secondary_labels: list[str] = []
        secondary_exec: str | None = None
        for secondary_action in secondary_action_values[:2]:
            mapped_secondary = _action_mapping(secondary_action)
            if mapped_secondary is None:
                continue
            label = str(mapped_secondary.get("primary_label") or "").strip()
            if label and label not in secondary_labels:
                secondary_labels.append(label)
            if secondary_exec is None:
                secondary_exec = mapped_secondary.get("primary_exec")
        if secondary_labels:
            mapped_required["secondary_label"] = " ".join(secondary_labels)
            mapped_required["secondary_exec"] = secondary_exec
        return mapped_required

    def _summary_mapping(summary: Mapping[str, Any] | None) -> dict[str, str | None] | None:
        summary_map = _coerce_mapping(summary) or {}
        primary_action = str(summary_map.get("primary_action") or "").strip()
        if not primary_action:
            return None
        mapped_primary = _action_mapping(primary_action)
        if mapped_primary is None:
            return None
        secondary_labels: list[str] = []
        secondary_exec: str | None = None
        for secondary_action in list(summary_map.get("secondary_actions") or [])[:2]:
            mapped_secondary = _action_mapping(str(secondary_action or ""))
            if mapped_secondary is None:
                continue
            label = str(mapped_secondary.get("primary_label") or "").strip()
            if label and label not in secondary_labels:
                secondary_labels.append(label)
            if secondary_exec is None:
                secondary_exec = mapped_secondary.get("primary_exec")
        if secondary_labels:
            mapped_primary["secondary_label"] = " ".join(secondary_labels)
            mapped_primary["secondary_exec"] = secondary_exec
        return mapped_primary

    candidate_action_summary = _coerce_mapping(operations_dashboard.get("candidate_action_summary")) or _coerce_mapping(
        operations_console.get("candidate_action_summary")
    )
    queue_action_summary = _coerce_mapping(operations_dashboard.get("queue_action_summary")) or _coerce_mapping(
        operations_console.get("queue_action_summary")
    )
    runtime_action_summary = _coerce_mapping(operations_dashboard.get("runtime_action_summary")) or _coerce_mapping(
        operations_console.get("runtime_action_summary")
    )

    if current_focus.startswith("candidate"):
        mapped_candidate_summary = _summary_mapping(candidate_action_summary)
        if mapped_candidate_summary is not None:
            return mapped_candidate_summary
    if current_focus.startswith("queue"):
        mapped_queue_summary = _summary_mapping(queue_action_summary)
        if mapped_queue_summary is not None:
            return mapped_queue_summary
    if current_focus.startswith("runner") or current_focus.startswith("daemon"):
        mapped_runtime_summary = _summary_mapping(runtime_action_summary)
        if (
            mapped_runtime_summary is not None
            and str(runtime_action_summary.get("primary_action") or "").strip() == "inspect_runtime_stability"
        ):
            return mapped_runtime_summary

    if current_focus.startswith("policy_") or current_focus.startswith("auto_train_policy"):
        return {
            "primary_label": "/policy",
            "primary_exec": "policy",
            "secondary_label": "/gate",
            "secondary_exec": "gate",
        }
    if current_focus in {"insufficient_new_signal_samples", "holdout_not_ready"}:
        return {
            "primary_label": "/gate",
            "primary_exec": "gate",
            "secondary_label": "/trigger /policy",
            "secondary_exec": "trigger",
        }
    if current_focus == "cooldown_active":
        return {
            "primary_label": "/trigger",
            "primary_exec": "trigger",
            "secondary_label": "/gate /policy",
            "secondary_exec": "gate",
        }
    if current_focus == "failure_backoff_active":
        return {
            "primary_label": "/retry",
            "primary_exec": "retry",
            "secondary_label": "/trigger /gate",
            "secondary_exec": "trigger",
        }
    if "pending_review" in current_focus or "awaiting_confirmation" in current_focus:
        return {
            "primary_label": "/approve or /reject",
            "primary_exec": None,
            "secondary_label": "/gate /trigger",
            "secondary_exec": "gate",
        }
    if current_focus.startswith("daemon") and "restart" in current_focus:
        return {
            "primary_label": "/restart daemon",
            "primary_exec": "restart daemon",
            "secondary_label": "/runtime /alerts",
            "secondary_exec": "runtime",
        }
    if current_focus.startswith("daemon") and ("heartbeat" in current_focus or "lease" in current_focus):
        return {
            "primary_label": "/recover daemon",
            "primary_exec": "recover daemon",
            "secondary_label": "/runtime /daemon",
            "secondary_exec": "runtime",
        }
    if "runner" in current_focus and "stale" in current_focus:
        return {
            "primary_label": "/runtime /runner",
            "primary_exec": "runtime",
            "secondary_label": "/runner hist",
            "secondary_exec": "runner hist",
        }
    if "candidate_ready_for_promotion" in current_focus:
        return {
            "primary_label": "/promote",
            "primary_exec": "promote",
            "secondary_label": "/candidate /cand sum",
            "secondary_exec": "candidate",
        }
    if current_focus in {"queue_waiting_execution", "queue_backlog"}:
        return {
            "primary_label": "/process",
            "primary_exec": "process",
            "secondary_label": "/queue /qs",
            "secondary_exec": "queue",
        }
    if current_focus.startswith("daemon"):
        return {
            "primary_label": "/recover daemon",
            "primary_exec": "recover daemon",
            "secondary_label": "/runtime /daemon",
            "secondary_exec": "runtime",
        }
    if current_focus.startswith("candidate"):
        can_promote = bool(candidate_summary.get("candidate_can_promote"))
        can_archive = bool(candidate_summary.get("candidate_can_archive"))
        if can_promote:
            return {
                "primary_label": "/promote",
                "primary_exec": "promote",
                "secondary_label": "/candidate /cand sum",
                "secondary_exec": "candidate",
            }
        if can_archive:
            return {
                "primary_label": "/archive",
                "primary_exec": "archive",
                "secondary_label": "/candidate /cand sum",
                "secondary_exec": "candidate",
            }
        return {
            "primary_label": "/candidate",
            "primary_exec": "candidate",
            "secondary_label": "/cand sum",
            "secondary_exec": "cand sum",
        }
    if current_focus.startswith("queue"):
        return {
            "primary_label": "/trigger /gate",
            "primary_exec": "trigger",
            "secondary_label": "/queue /qs",
            "secondary_exec": "queue",
        }
    if current_focus.startswith("runner"):
        return {
            "primary_label": "/runtime /runner",
            "primary_exec": "runtime",
            "secondary_label": "/rs /runner hist",
            "secondary_exec": "rs",
        }
    return {
        "primary_label": "/sum /dash",
        "primary_exec": "sum",
        "secondary_label": "/status /help",
        "secondary_exec": "status",
    }


def _console_shortcut_hint(mode_name: str, payload: Mapping[str, Any] | None = None) -> str:
    operations_dashboard = _coerce_mapping((payload or {}).get("operations_dashboard")) or {}
    attention_needed = bool(operations_dashboard.get("attention_needed"))
    current_focus = _console_dashboard_focus(payload)
    if mode_name == "chat":
        return "Enter,/do,/see" if attention_needed else "Enter,/help,^C"

    def _primary_shortcut_token() -> str | None:
        focus_actions = _console_focus_actions(payload)
        primary_label = str(focus_actions.get("primary_label") or "").strip()
        if not primary_label.startswith("/"):
            return None
        return primary_label.split(" or ")[0].strip()

    if current_focus.startswith("policy_") or current_focus.startswith("auto_train_policy"):
        return "/do,/see,/policy,/chat"
    if current_focus in {"insufficient_new_signal_samples", "holdout_not_ready"}:
        return "/do,/see,/gate,/chat"
    if current_focus == "cooldown_active":
        return "/do,/see,/trigger,/chat"
    if current_focus == "failure_backoff_active":
        return "/do,/see,/retry,/chat"
    if "pending_review" in current_focus or "awaiting_confirmation" in current_focus:
        return "/do,/see,/approve,/chat"
    if current_focus.startswith("daemon") and "restart" in current_focus:
        return "/do,/see,/alerts,/chat"
    if current_focus.startswith("daemon") and ("heartbeat" in current_focus or "lease" in current_focus):
        return "/do,/see,/daemon,/chat"
    if "runner" in current_focus and "stale" in current_focus:
        return "/do,/see,/runner,/chat"
    if "candidate_ready_for_promotion" in current_focus:
        return "/do,/see,/archive,/chat"
    if current_focus in {"queue_waiting_execution", "queue_backlog"}:
        return "/do,/see,/process,/chat"
    if current_focus in {"candidate_idle", "runner_active", "daemon_active"}:
        primary_shortcut = _primary_shortcut_token()
        if primary_shortcut:
            return f"/do,/see,{primary_shortcut},/chat"
    if current_focus.startswith("daemon"):
        primary_shortcut = _primary_shortcut_token()
        if primary_shortcut == "/runtime":
            return "/do,/see,/runtime,/chat"
        return "/do,/see,/daemon,/chat"
    if current_focus.startswith("candidate"):
        primary_shortcut = _primary_shortcut_token()
        if primary_shortcut:
            return f"/do,/see,{primary_shortcut},/chat"
        candidate_summary = _coerce_mapping((payload or {}).get("candidate_summary")) or {}
        if bool(candidate_summary.get("candidate_can_promote")) or bool(candidate_summary.get("candidate_can_archive")):
            return "/do,/see,/archive,/chat"
        return "/do,/see,/candidate,/chat"
    if current_focus.startswith("queue"):
        primary_shortcut = _primary_shortcut_token()
        if primary_shortcut:
            return f"/do,/see,{primary_shortcut},/chat"
        return "/do,/see,/queue,/chat"
    if current_focus.startswith("runner"):
        primary_shortcut = _primary_shortcut_token()
        if primary_shortcut == "/runtime":
            return "/do,/see,/runtime,/chat"
        return "/do,/see,/runner,/chat"
    return "/status,/candidate,/daemon,/chat"


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
    return "\n".join(
        [
            "PFE console session settings:",
            f"workspace={workspace or 'user_default'}",
            f"mode={mode}",
            f"model={model}",
            f"adapter={adapter}",
            f"temperature={temperature:.2f}",
            f"max_tokens={max_tokens if max_tokens is not None else 'auto'}",
            f"real_local={_yes_no(real_local)}",
            f"refresh_seconds={refresh_seconds:.1f}",
        ]
    )


def _console_status_compact_text(payload: Mapping[str, Any], *, workspace: str | None = None) -> str:
    mapping = _coerce_mapping(payload) or {}
    latest_adapter = _coerce_mapping(mapping.get("latest_adapter")) or {}
    operations_overview = _coerce_mapping(mapping.get("operations_overview")) or {}
    operations_console = _coerce_mapping(mapping.get("operations_console")) or {}
    operations_dashboard = _coerce_mapping(mapping.get("operations_dashboard")) or {}
    operations_alert_policy = _coerce_mapping(mapping.get("operations_alert_policy")) or {}
    train_queue = _coerce_mapping(mapping.get("train_queue")) or {}
    resolved_focus = _console_dashboard_focus(mapping)
    resolved_action = (
        operations_alert_policy.get("required_action")
        or operations_dashboard.get("required_action")
        or operations_console.get("required_action")
        or operations_overview.get("required_action")
        or "observe_and_monitor"
    )
    summary_line = (
        operations_overview.get("summary_line")
        or operations_console.get("summary_line")
        or operations_dashboard.get("summary_line")
        or operations_alert_policy.get("summary_line")
    )
    inspection_summary_line = (
        operations_overview.get("inspection_summary_line")
        or operations_console.get("inspection_summary_line")
        or operations_dashboard.get("inspection_summary_line")
        or operations_alert_policy.get("inspection_summary_line")
    )
    summary_line, inspection_summary_line = _prefer_inspection_summary_for_generic_monitor(
        focus=resolved_focus,
        summary_line=summary_line,
        inspection_summary_line=inspection_summary_line,
    )

    parts: list[str] = [
        f"workspace={_format_scalar(workspace or mapping.get('workspace') or 'user_default')}",
        f"latest={_format_scalar(latest_adapter.get('version') or 'none')}",
        f"severity={_format_scalar(operations_dashboard.get('severity') or 'stable')}",
        f"focus={_format_scalar(resolved_focus)}",
        f"action={_format_scalar(resolved_action)}",
        f"queue={_format_scalar(train_queue.get('count', 0))}",
    ]
    if summary_line:
        parts.append(f"summary={_format_scalar(summary_line)}")
    return "\n".join(["PFE status compact", "summary: " + " | ".join(parts)])


def _console_candidate_summary_text(payload: Mapping[str, Any], timeline: Mapping[str, Any] | None = None) -> str:
    mapping = _coerce_mapping(payload) or {}
    candidate_summary = _coerce_mapping(mapping.get("candidate_summary")) or {}
    candidate_timeline = _coerce_mapping(timeline) or _coerce_mapping(mapping.get("candidate_timeline")) or {}

    parts: list[str] = []
    for key in (
        "candidate_version",
        "candidate_state",
        "candidate_can_promote",
        "candidate_can_archive",
        "pending_eval_count",
        "training_count",
        "failed_eval_count",
        "candidate_needs_promotion",
        "promotion_compare_comparison",
        "promotion_compare_recommendation",
        "promotion_compare_winner",
        "promotion_compare_left_adapter",
        "promotion_compare_right_adapter",
        "promotion_compare_overall_delta",
        "promotion_compare_style_preference_hit_rate_delta",
        "promotion_compare_personalization_delta",
        "promotion_compare_quality_delta",
        "promotion_compare_personalization_summary",
        "promotion_compare_quality_summary",
        "promotion_compare_summary_line",
    ):
        value = candidate_summary.get(key)
        if value is not None:
            parts.append(f"{key}={_format_scalar(value)}")
    for key in ("current_stage", "transition_count", "last_reason", "last_candidate_version"):
        value = candidate_timeline.get(key)
        if value is not None:
            parts.append(f"{key}={_format_scalar(value)}")
    if not parts:
        parts.append("state=idle")
    return "\n".join(["PFE candidate summary", "summary: " + " | ".join(parts)])


def _console_queue_summary_text(payload: Mapping[str, Any], history: Mapping[str, Any] | None = None) -> str:
    mapping = _coerce_mapping(payload) or {}
    queue_summary = _coerce_mapping(mapping.get("train_queue")) or {}
    queue_history = _coerce_mapping(history) or _coerce_mapping(mapping.get("train_queue")) or {}

    parts: list[str] = []
    for key in (
        "count",
        "max_priority",
        "current_job_id",
        "awaiting_confirmation_count",
        "reviewed_transition_count",
        "approved_transition_count",
        "rejected_transition_count",
    ):
        value = queue_summary.get(key)
        if value is not None:
            parts.append(f"{key}={_format_scalar(value)}")
    history_summary = _coerce_mapping(queue_history.get("history_summary")) or {}
    for key in ("transition_count", "last_reason"):
        value = history_summary.get(key)
        if value is not None:
            parts.append(f"{key}={_format_scalar(value)}")
    last_transition = _coerce_mapping(history_summary.get("last_transition")) or {}
    if last_transition.get("event") is not None:
        parts.append(f"last_event={_format_scalar(last_transition.get('event'))}")
    if not parts:
        parts.append("count=0")
    return "\n".join(["PFE train queue summary", "summary: " + " | ".join(parts)])


def _console_runner_summary_text(payload: Mapping[str, Any], history: Mapping[str, Any] | None = None) -> str:
    mapping = _coerce_mapping(payload) or {}
    runner_summary = _coerce_mapping(mapping.get("train_queue_worker_runner")) or _coerce_mapping(mapping.get("worker_runner")) or {}
    runner_history = _coerce_mapping(history) or _coerce_mapping(mapping.get("runner_timeline")) or {}

    parts: list[str] = []
    for key in ("active", "lock_state", "stop_requested", "processed_count", "failed_count", "loop_cycles"):
        value = runner_summary.get(key)
        if value is not None:
            parts.append(f"{key}={_format_scalar(value)}")
    for key in ("count", "last_event", "last_reason", "takeover_event_count", "current_lock_state"):
        value = runner_history.get(key)
        if value is not None:
            parts.append(f"{key}={_format_scalar(value)}")
    if not parts:
        parts.append("state=idle")
    return "\n".join(["PFE worker runner summary", "summary: " + " | ".join(parts)])


def _console_daemon_summary_text(result: Any) -> str:
    mapping = _coerce_mapping(result) or {}
    parts: list[str] = []
    for key in (
        "workspace",
        "desired_state",
        "observed_state",
        "command_status",
        "lock_state",
        "health_state",
        "lease_state",
        "heartbeat_state",
        "recovery_action",
    ):
        value = mapping.get(key)
        if value is not None:
            parts.append(f"{key}={_format_scalar(value)}")
    if not parts:
        parts.append("state=unknown")
    return "\n".join(["PFE worker daemon summary", "summary: " + " | ".join(parts)])


def _console_trigger_summary_text(payload: Mapping[str, Any]) -> str:
    mapping = _coerce_mapping(payload) or {}
    trigger = _coerce_mapping(mapping.get("auto_train_trigger")) or {}
    parts: list[str] = []
    for key in (
        "enabled",
        "state",
        "ready",
        "reason",
        "blocked_primary_reason",
        "blocked_primary_action",
        "blocked_primary_category",
        "queue_gate_reason",
        "queue_gate_action",
        "queue_review_mode",
    ):
        value = trigger.get(key)
        if value is not None:
            parts.append(f"{key}={_format_scalar(value)}")
    blocked_summary = trigger.get("blocked_summary")
    if blocked_summary:
        parts.append(f"blocked_summary={_format_scalar(blocked_summary)}")
    if not parts:
        parts.append("state=idle")
    return "\n".join(["PFE auto-train trigger summary", "summary: " + " | ".join(parts)])


def _console_gate_summary_text(payload: Mapping[str, Any]) -> str:
    mapping = _coerce_mapping(payload) or {}
    trigger = _coerce_mapping(mapping.get("auto_train_trigger")) or {}
    policy = _coerce_mapping(trigger.get("policy")) or {}
    threshold = _coerce_mapping(trigger.get("threshold_summary")) or {}
    train_queue = _coerce_mapping(mapping.get("train_queue")) or {}
    review_policy = _coerce_mapping(train_queue.get("review_policy_summary")) or {}

    parts: list[str] = []
    for key in ("queue_entry_mode", "review_mode", "evaluation_mode", "promotion_mode", "stop_stage"):
        value = policy.get(key)
        if value is not None:
            parts.append(f"{key}={_format_scalar(value)}")
    for key in (
        "eligible_signal_train_samples",
        "effective_eligible_train_samples",
        "preference_reinforced_train_samples",
        "min_new_samples",
        "holdout_ready",
        "interval_elapsed",
        "cooldown_elapsed",
        "failure_backoff_elapsed",
    ):
        value = threshold.get(key)
        if value is not None:
            parts.append(f"{key}={_format_scalar(value)}")
    for key in ("review_mode", "review_required_now", "next_action", "review_reason"):
        value = review_policy.get(key)
        if value is not None:
            parts.append(f"queue_{key}={_format_scalar(value)}")
    if not parts:
        parts.append("state=idle")
    return "\n".join(["PFE gate summary", "summary: " + " | ".join(parts)])


def _console_runtime_summary_text(payload: Mapping[str, Any]) -> str:
    mapping = _coerce_mapping(payload) or {}
    console = _coerce_mapping(mapping.get("operations_console")) or {}
    operations_overview = _coerce_mapping(mapping.get("operations_overview")) or {}
    runtime = _coerce_mapping(console.get("runtime_stability_summary")) or _coerce_mapping(
        operations_overview.get("runtime_stability_summary")
    ) or {}
    alert_policy = _coerce_mapping(mapping.get("operations_alert_policy")) or {}

    parts: list[str] = []
    for key in (
        "runner_active",
        "runner_lock_state",
        "runner_stop_requested",
        "daemon_health_state",
        "daemon_heartbeat_state",
        "daemon_lease_state",
        "daemon_restart_policy_state",
        "daemon_recovery_action",
    ):
        value = runtime.get(key)
        if value is not None:
            parts.append(f"{key}={_format_scalar(value)}")
    for key in ("required_action", "action_priority", "remediation_mode", "operator_guidance"):
        value = alert_policy.get(key)
        if value is not None:
            parts.append(f"{key}={_format_scalar(value)}")
    if not parts:
        parts.append("state=idle")
    return "\n".join(["PFE runtime stability summary", "summary: " + " | ".join(parts)])


def _console_command_output(
    command: str,
    *,
    payload: Mapping[str, Any],
    workspace: str | None,
    service: Any,
    current_workspace: str | None,
    mode: str,
    model: str,
    adapter: str,
    temperature: float,
    max_tokens: int | None,
    real_local: bool,
    refresh_seconds: float,
    last_interaction: dict[str, Any] | None = None,
) -> tuple[str | None, str, dict[str, Any] | None]:
    normalized = command.strip().lower()
    if normalized in {"quit", "exit"}:
        return None, "quit", None
    if normalized == "help":
        return _console_help_text(), "help", None
    if normalized == "chat":
        return None, "mode:chat", None
    if normalized in {"cmd", "command"}:
        return None, "mode:command", None
    if normalized.startswith("mode "):
        selected_mode = normalized.split(" ", 1)[1].strip()
        if selected_mode in {"chat", "command"}:
            return None, f"mode:{selected_mode}", None
        return "Unknown mode. Use /mode chat or /mode command.", "unknown", None
    if normalized in {"like", "good", "yes"}:
        if last_interaction is None:
            return "No previous interaction to accept. Start a chat first.", "like", None
        _console_submit_feedback(
            workspace=workspace or current_workspace or "user_default",
            session_id=last_interaction.get("session_id", ""),
            request_id=last_interaction.get("request_id", ""),
            user_message=last_interaction.get("user_message", ""),
            assistant_message=last_interaction.get("assistant_message", ""),
            response_time_seconds=last_interaction.get("response_time_seconds", 0.0),
            adapter_version=last_interaction.get("adapter_version", adapter),
            action="continue",
        )
        return "Accepted previous assistant response.", "like", None
    if normalized in {"dislike", "bad", "no"}:
        if last_interaction is None:
            return "No previous interaction to reject. Start a chat first.", "dislike", None
        _console_submit_feedback(
            workspace=workspace or current_workspace or "user_default",
            session_id=last_interaction.get("session_id", ""),
            request_id=last_interaction.get("request_id", ""),
            user_message=last_interaction.get("user_message", ""),
            assistant_message=last_interaction.get("assistant_message", ""),
            response_time_seconds=last_interaction.get("response_time_seconds", 0.0),
            adapter_version=last_interaction.get("adapter_version", adapter),
            action="delete",
        )
        return "Rejected previous assistant response.", "dislike", None
    if normalized in {"again", "redo"}:
        if last_interaction is None:
            return "No previous interaction to regenerate. Start a chat first.", "again", None
        _console_submit_feedback(
            workspace=workspace or current_workspace or "user_default",
            session_id=last_interaction.get("session_id", ""),
            request_id=last_interaction.get("request_id", ""),
            user_message=last_interaction.get("user_message", ""),
            assistant_message=last_interaction.get("assistant_message", ""),
            response_time_seconds=last_interaction.get("response_time_seconds", 0.0),
            adapter_version=last_interaction.get("adapter_version", adapter),
            action="regenerate",
        )
        return "Requested regeneration for previous response.", "again", {"regenerate": True}
    if normalized.startswith("fix "):
        if last_interaction is None:
            return "No previous interaction to edit. Start a chat first.", "fix", None
        edited_text = normalized[4:].strip()
        if not edited_text:
            return "Usage: /fix <corrected text>", "fix", None
        _console_submit_feedback(
            workspace=workspace or current_workspace or "user_default",
            session_id=last_interaction.get("session_id", ""),
            request_id=last_interaction.get("request_id", ""),
            user_message=last_interaction.get("user_message", ""),
            assistant_message=last_interaction.get("assistant_message", ""),
            response_time_seconds=last_interaction.get("response_time_seconds", 0.0),
            adapter_version=last_interaction.get("adapter_version", adapter),
            action="edit",
            edited_text=edited_text,
        )
        return f"Submitted edit: {edited_text}", "fix", {"edited_text": edited_text}
    if normalized in {"status compact", "status summary", "sum"}:
        return _console_status_compact_text(payload, workspace=current_workspace or workspace), "status-compact", None
    if normalized == "status":
        return _format_status(dict(payload), workspace=workspace), "status", None
    if normalized in {"ops dashboard", "ops summary", "ops dash", "dash"}:
        dashboard_lines = _format_operations_dashboard(payload.get("operations_dashboard")) or ["operations dashboard: none"]
        return "\n".join(dashboard_lines), "ops-dashboard", None
    if normalized in {"ops alerts", "alerts"}:
        alert_lines = _format_operations_alert_surface(
            {
                "operations_alerts": payload.get("operations_alerts"),
                "operations_health": payload.get("operations_health"),
                "operations_recovery": payload.get("operations_recovery"),
                "operations_next_actions": payload.get("operations_next_actions"),
                "operations_dashboard": payload.get("operations_dashboard"),
                "operations_alert_policy": payload.get("operations_alert_policy"),
                "operations_console": payload.get("operations_console"),
                "operations_overview": payload.get("operations_overview"),
            }
        ) or ["operations alerts: none"]
        return "\n".join(alert_lines), "ops-alerts", None
    if normalized in {"ops policy", "ops pol", "alert policy", "policy"}:
        policy_lines = _format_operations_alert_policy(payload.get("operations_alert_policy")) or ["operations alert policy: none"]
        return "\n".join(policy_lines), "ops-policy", None
    if normalized in {"trigger", "trigger summary", "trig"}:
        return _console_trigger_summary_text(payload), "trigger-summary", None
    if normalized in {"gate", "gate summary", "gates"}:
        return _console_gate_summary_text(payload), "gate-summary", None
    if normalized in {"runtime", "runtime summary", "stability", "rt"}:
        return _console_runtime_summary_text(payload), "runtime-summary", None
    if normalized in {"event stream", "ops event-stream", "ops events", "event"}:
        stream_lines = _format_operations_event_stream(payload.get("operations_event_stream")) or ["operations event stream: none"]
        return "\n".join(stream_lines), "event-stream", None
    if normalized.startswith("promote"):
        handler = _resolve_handler(service, "promote_candidate")
        if handler is None:
            return "Candidate promote is unavailable.", "candidate-promote-unavailable", None
        parts = normalized.split(None, 1)
        note = None
        if len(parts) > 1:
            note = parts[1].strip()
        result = handler(workspace=workspace, note=note)
        return _format_status(result, workspace=workspace), "candidate-promote", None
    if normalized.startswith("archive"):
        handler = _resolve_handler(service, "archive_candidate")
        if handler is None:
            return "Candidate archive is unavailable.", "candidate-archive-unavailable", None
        parts = normalized.split(None, 1)
        note = None
        if len(parts) > 1:
            note = parts[1].strip()
        result = handler(workspace=workspace, note=note)
        return _format_status(result, workspace=workspace), "candidate-archive", None
    if normalized.startswith("rollback"):
        handler = _resolve_handler(service, "rollback_candidate", "rollback_adapter")
        if handler is None:
            return "Rollback action is unavailable.", "rollback-unavailable", None
        parts = normalized.split(None, 1)
        version = None
        if len(parts) > 1:
            version = parts[1].strip()
        result = handler(workspace=workspace, version=version)
        return _format_status(result, workspace=workspace), "candidate-rollback", None
    if normalized == "do":
        focus_actions = _console_focus_actions(payload)
        primary_exec = focus_actions.get("primary_exec")
        primary_label = str(focus_actions.get("primary_label") or "/status")
        if not primary_exec:
            return f"Primary action requires a review choice. Use {primary_label}.", "do-ambiguous", None
        return _console_command_output(
            str(primary_exec),
            payload=payload,
            workspace=workspace,
            service=service,
            current_workspace=current_workspace,
            mode=mode,
            model=model,
            adapter=adapter,
            temperature=temperature,
            max_tokens=max_tokens,
            real_local=real_local,
            refresh_seconds=refresh_seconds,
        )
    if normalized == "see":
        focus_actions = _console_focus_actions(payload)
        secondary_exec = focus_actions.get("secondary_exec")
        secondary_label = str(focus_actions.get("secondary_label") or "/status")
        if not secondary_exec:
            return f"No secondary view is available. Try {secondary_label}.", "see-unavailable", None
        return _console_command_output(
            str(secondary_exec),
            payload=payload,
            workspace=workspace,
            service=service,
            current_workspace=current_workspace,
            mode=mode,
            model=model,
            adapter=adapter,
            temperature=temperature,
            max_tokens=max_tokens,
            real_local=real_local,
            refresh_seconds=refresh_seconds,
        )
    if normalized.startswith("approve"):
        handler = _resolve_handler(service, "approve_next_train_queue", "approve_train_queue_item")
        if handler is None:
            return "Approve action is unavailable.", "approve-unavailable", None
        parts = normalized.split(None, 1)
        note = None
        if len(parts) > 1:
            note = parts[1].strip()
        result = handler(workspace=workspace, note=note)
        return _format_status(result, workspace=workspace), "approve-next", None
    if normalized.startswith("reject"):
        handler = _resolve_handler(service, "reject_next_train_queue", "reject_train_queue_item")
        if handler is None:
            return "Reject action is unavailable.", "reject-unavailable", None
        parts = normalized.split(None, 1)
        note = None
        if len(parts) > 1:
            note = parts[1].strip()
        result = handler(workspace=workspace, note=note)
        return _format_status(result, workspace=workspace), "reject-next", None
    if normalized in {"process", "process next", "next"}:
        handler = _resolve_handler(service, "process_next_train_queue")
        if handler is None:
            return "Queue processing is unavailable.", "process-unavailable", None
        result = handler(workspace=workspace)
        return _format_status(result, workspace=workspace), "process-next", None
    if normalized in {"retry", "trigger-train", "trigger train"}:
        handler = _resolve_handler(service, "retry_auto_train_trigger")
        if handler is None:
            return "Retry action is unavailable.", "retry-unavailable", None
        result = handler(workspace=workspace)
        return _format_status(result, workspace=workspace), "retry-trigger", None
    if normalized == "reset":
        handler = _resolve_handler(service, "reset_auto_train_trigger")
        if handler is None:
            return "Reset action is unavailable.", "reset-unavailable", None
        result = handler(workspace=workspace)
        return _format_status(result, workspace=workspace), "reset-trigger", None
    if normalized in {"recover daemon", "daemon recover"}:
        handler = _resolve_handler(service, "recover_train_queue_daemon", "daemon_recover")
        if handler is None:
            return "Daemon recovery is unavailable.", "daemon-recover-unavailable", None
        result = handler(workspace=workspace, note=None)
        return _format_train_queue_daemon_status(result), "daemon-recover", None
    if normalized in {"restart daemon", "daemon restart"}:
        handler = _resolve_handler(service, "restart_train_queue_daemon", "daemon_restart")
        if handler is None:
            return "Daemon restart is unavailable.", "daemon-restart-unavailable", None
        result = handler(workspace=workspace, note=None)
        return _format_train_queue_daemon_status(result), "daemon-restart", None
    if normalized in {"stop daemon", "daemon stop"}:
        handler = _resolve_handler(service, "stop_train_queue_daemon", "daemon_stop")
        if handler is None:
            return "Daemon stop is unavailable.", "daemon-stop-unavailable", None
        result = handler(workspace=workspace, note=None)
        return _format_train_queue_daemon_status(result), "daemon-stop", None
    if normalized in {"start daemon", "daemon start"}:
        handler = _resolve_handler(service, "start_train_queue_daemon", "daemon_start")
        if handler is None:
            return "Daemon start is unavailable.", "daemon-start-unavailable", None
        result = handler(workspace=workspace, note=None)
        return _format_train_queue_daemon_status(result), "daemon-start", None
    if normalized in {"stop runner", "runner stop"}:
        handler = _resolve_handler(service, "stop_train_queue_worker_runner", "runner_stop")
        if handler is None:
            return "Runner stop is unavailable.", "runner-stop-unavailable", None
        result = handler(workspace=workspace)
        return _format_worker_runner_status(result), "runner-stop", None
    if normalized in {"batch", "process batch"}:
        handler = _resolve_handler(service, "process_train_queue_batch")
        if handler is None:
            return "Queue batch processing is unavailable.", "batch-unavailable", None
        parts = normalized.split(None, 1)
        limit = 5
        if len(parts) > 1:
            try:
                limit = int(parts[1].strip().split()[0])
            except ValueError:
                pass
        result = handler(limit=limit)
        return _format_status(result, workspace=workspace), "process-batch", None
    if normalized in {"until-idle", "process until-idle", "process idle"}:
        handler = _resolve_handler(service, "process_train_queue_until_idle")
        if handler is None:
            return "Queue until-idle processing is unavailable.", "until-idle-unavailable", None
        result = handler()
        return _format_status(result, workspace=workspace), "process-until-idle", None
    if normalized in {"train", "trigger-train"}:
        handler = _resolve_handler(service, "retry_auto_train_trigger")
        if handler is None:
            return "Train trigger is unavailable.", "train-unavailable", None
        result = handler(workspace=workspace)
        return _format_status(result, workspace=workspace), "train-trigger", None
    if normalized in {"eval", "evaluate"}:
        handler = _resolve_handler(service, "evaluate", "eval")
        if handler is None:
            return "Evaluation is unavailable.", "eval-unavailable", None
        result = handler(workspace=workspace)
        return _format_eval_result(result, workspace=workspace), "eval-trigger", None
    if normalized in {"distill", "distill run"}:
        handler = _resolve_handler(service, "run_distillation", "distill")
        if handler is None:
            return "Distillation is unavailable.", "distill-unavailable", None
        result = handler()
        return _format_status(result, workspace=workspace), "distill", None
    if normalized in {"force-recovery", "force recovery"}:
        handler = _resolve_handler(service, "force_recovery")
        if handler is None:
            return "Force recovery is unavailable.", "force-recovery-unavailable", None
        result = handler(workspace=workspace, reason="console-request")
        return _format_status(result, workspace=workspace), "force-recovery", None
    if normalized in {"ops", "os"}:
        ops_lines = _format_operations_console_digest(dict(payload)) or ["operations console digest: none"]
        return "\n".join(ops_lines), "ops", None
    if normalized == "doctor":
        doctor_model = None if model in {"local", "base", "local-default"} else model
        return _format_doctor(workspace=current_workspace, base_model=doctor_model), "doctor", None
    if normalized == "serve":
        return (
            _format_serve_preview(
                port=8921,
                host="127.0.0.1",
                adapter=adapter,
                workspace=current_workspace,
                api_key=None,
                real_local=real_local,
            ),
            "serve",
            None,
        )
    if normalized == "clear":
        return "", "clear", None
    if normalized == "settings":
        return (
            _console_settings_text(
                workspace=current_workspace,
                mode=mode,
                model=model,
                adapter=adapter,
                temperature=temperature,
                max_tokens=max_tokens,
                real_local=real_local,
                refresh_seconds=refresh_seconds,
            ),
            "settings",
            None,
        )
    if normalized.startswith("workspace "):
        selected_workspace = command.split(" ", 1)[1].strip()
        if not selected_workspace:
            return "Usage: /workspace <name>", "unknown", None
        return f"workspace set to {selected_workspace}", "set-workspace", {"workspace": selected_workspace}
    if normalized.startswith("model "):
        selected_model = command.split(" ", 1)[1].strip()
        if not selected_model:
            return "Usage: /model <id>", "unknown", None
        return f"model set to {selected_model}", "set-model", {"model": selected_model}
    if normalized.startswith("adapter "):
        selected_adapter = command.split(" ", 1)[1].strip()
        if not selected_adapter:
            return "Usage: /adapter <version>", "unknown", None
        return f"adapter set to {selected_adapter}", "set-adapter", {"adapter": selected_adapter}
    if normalized.startswith("temperature "):
        value = normalized.split(" ", 1)[1].strip()
        try:
            selected_temperature = float(value)
        except ValueError:
            return "Usage: /temperature <value>", "unknown", None
        if selected_temperature < 0.0 or selected_temperature > 2.0:
            return "Temperature must be between 0.0 and 2.0.", "unknown", None
        return f"temperature set to {selected_temperature:.2f}", "set-temperature", {"temperature": selected_temperature}
    if normalized.startswith("max-tokens "):
        value = normalized.split(" ", 1)[1].strip().lower()
        if value in {"auto", "none", "default"}:
            return "max tokens set to auto", "set-max-tokens", {"max_tokens": None}
        try:
            selected_max_tokens = int(value)
        except ValueError:
            return "Usage: /max-tokens <n|auto>", "unknown", None
        if selected_max_tokens < 1:
            return "Max tokens must be at least 1.", "unknown", None
        return f"max tokens set to {selected_max_tokens}", "set-max-tokens", {"max_tokens": selected_max_tokens}
    if normalized.startswith("real-local "):
        selected_state = normalized.split(" ", 1)[1].strip()
        if selected_state in {"on", "true", "1", "yes"}:
            return "real-local enabled", "set-real-local", {"real_local": True}
        if selected_state in {"off", "false", "0", "no"}:
            return "real-local disabled", "set-real-local", {"real_local": False}
        return "Usage: /real-local on|off", "unknown", None
    if normalized.startswith("refresh "):
        value = normalized.split(" ", 1)[1].strip()
        try:
            refresh_value = float(value)
        except ValueError:
            return "Usage: /refresh <seconds>", "unknown", None
        if refresh_value < 0.1:
            return "Refresh must be at least 0.1 seconds.", "unknown", None
        return f"refresh set to {refresh_value:.1f}s", "set-refresh", {"refresh_seconds": refresh_value}
    if normalized in {"candidate history", "candidate-history", "cand hist"}:
        handler = _resolve_handler(service, "candidate_history")
        if handler is not None:
            result = handler(workspace=workspace, limit=10)
            return _format_candidate_history(result), "candidate-history", None
        history = payload.get("candidate_history") or {}
        return _format_candidate_history(history), "candidate-history", None
    if normalized in {"candidate summary", "candidate compact", "cand", "cand sum"}:
        handler = _resolve_handler(service, "candidate_timeline")
        timeline_result = handler(workspace=workspace, limit=5) if handler is not None else None
        return _console_candidate_summary_text(payload, timeline=timeline_result), "candidate-summary", None
    if normalized in {"candidate", "candidate timeline", "candidate-timeline", "cand tl"}:
        handler = _resolve_handler(service, "candidate_timeline")
        if handler is not None:
            result = handler(workspace=workspace, limit=5)
            return _format_candidate_timeline(result), "candidate", None
        timeline = payload.get("candidate_timeline") or payload.get("operations_console", {}).get("candidate")
        return _format_candidate_timeline(timeline or {}), "candidate", None
    if normalized in {"queue", "queue summary", "queue compact", "queue sum", "qs"}:
        handler = _resolve_handler(service, "train_queue_history")
        result = handler(workspace=workspace, limit=5) if handler is not None else None
        return _console_queue_summary_text(payload, history=_coerce_mapping(result) if result is not None else None), "queue-summary", None
    if normalized in {"queue history", "queue-history", "queue hist"}:
        handler = _resolve_handler(service, "train_queue_history")
        if handler is not None:
            result = handler(workspace=workspace, limit=10)
            return _format_train_queue_history(result), "queue-history", None
        history = payload.get("train_queue") or {}
        return _format_train_queue_history(history), "queue-history", None
    if normalized in {"runner", "runner summary", "runner compact", "runner sum", "rs"}:
        handler = _resolve_handler(service, "train_queue_worker_runner_history")
        result = handler(workspace=workspace, limit=5) if handler is not None else None
        return _console_runner_summary_text(payload, history=_coerce_mapping(result) if result is not None else None), "runner-summary", None
    if normalized in {"runner timeline", "runner-timeline", "runner tl"}:
        timeline = payload.get("runner_timeline") or payload.get("operations_console", {}).get("runner_timeline") or {}
        return _format_runner_timeline_summary(timeline), "runner-timeline", None
    if normalized in {"runner history", "runner-history", "runner hist"}:
        handler = _resolve_handler(service, "train_queue_worker_runner_history")
        if handler is not None:
            result = handler(workspace=workspace, limit=10)
            return _format_worker_runner_history(result), "runner-history", None
        history = payload.get("runner_timeline") or {}
        return _format_runner_timeline_summary(history), "runner-history", None
    if normalized in {"daemon history", "daemon-history", "daemon hist"}:
        handler = _resolve_handler(service, "train_queue_daemon_history", "daemon_history")
        if handler is not None:
            result = handler(workspace=workspace, limit=10)
            return _format_train_queue_daemon_history(result), "daemon-history", None
        return (
            _format_train_queue_daemon_history(
                _read_train_queue_daemon_state(workspace) or {"workspace": workspace or "user_default", "history": []}
            ),
            "daemon-history",
            None,
        )
    if normalized in {"daemon summary", "daemon compact", "daemon sum", "ds"}:
        handler = _resolve_handler(service, "train_queue_daemon_status", "daemon_status", "get_daemon_status")
        if handler is not None:
            result = handler(workspace=workspace)
            return _console_daemon_summary_text(result), "daemon-summary", None
        return (
            _console_daemon_summary_text(
                _read_train_queue_daemon_state(workspace) or {"workspace": workspace or "user_default", "command_status": "absent"}
            ),
            "daemon-summary",
            None,
        )
    if normalized in {"daemon timeline", "daemon-timeline", "daemon tl"}:
        timeline = payload.get("daemon_timeline") or payload.get("operations_console", {}).get("daemon_timeline") or {}
        return _format_daemon_timeline_summary(timeline), "daemon-timeline", None
    if normalized == "daemon":
        handler = _resolve_handler(service, "train_queue_daemon_status", "daemon_status", "get_daemon_status")
        if handler is not None:
            result = handler(workspace=workspace)
            return _format_train_queue_daemon_status(result), "daemon", None
        return (
            _format_train_queue_daemon_status(
                _read_train_queue_daemon_state(workspace) or {"workspace": workspace or "user_default", "command_status": "absent"}
            ),
            "daemon",
            None,
        )
    if normalized in {"list", "adapter list", "adapters"}:
        handler = _resolve_handler(service, "list_versions")
        if handler is not None:
            result = handler(workspace=workspace, limit=20)
            lines = _format_lifecycle_summary(result)
            return "\n".join(lines or ["No adapters found."]), "adapter-list", None
        return "Adapter listing is unavailable.", "adapter-list-unavailable", None
    if normalized.startswith("generate "):
        handler = _resolve_handler(service, "generate")
        if handler is None:
            return "Generate is unavailable.", "generate-unavailable", None
        parts = normalized.split(None, 2)
        if len(parts) < 2:
            return "Usage: /generate <scenario> [style]", "generate", None
        scenario = parts[1]
        style = parts[2] if len(parts) > 2 else "default"
        result = handler(scenario=scenario, style=style, num_samples=10, workspace=workspace)
        return _format_status(result, workspace=workspace), "generate", None
    if normalized in {"dpo", "dpo train"}:
        handler = _resolve_handler(service, "train_dpo")
        if handler is None:
            return "DPO training is unavailable.", "dpo-unavailable", None
        result = handler(workspace=workspace)
        return _format_train_result(result, workspace=workspace or "user_default"), "dpo", None
    return f"Unknown command: /{command}. Try /help.", "unknown", None


@trigger_app.command("reset")
def trigger_reset(
    workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
) -> None:
    """Clear persisted auto-train cooldown/backoff state for the workspace."""

    service = _load_service("pfe_core.pipeline", "pfe_core.services.pipeline")
    if service is None:
        _run_placeholder("trigger reset")
        return

    handler = _resolve_handler(service, "reset_auto_train_trigger")
    if handler is None:
        _run_placeholder("trigger reset")
        return

    _run_handler(
        "trigger reset",
        handler,
        formatter=lambda result: _format_status(result, workspace=workspace),
        workspace=workspace,
    )


@trigger_app.command("retry")
def trigger_retry(
    workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
) -> None:
    """Re-check the auto-train trigger and run it if all current gates pass."""

    service = _load_service("pfe_core.pipeline", "pfe_core.services.pipeline")
    if service is None:
        _run_placeholder("trigger retry")
        return

    handler = _resolve_handler(service, "retry_auto_train_trigger")
    if handler is None:
        _run_placeholder("trigger retry")
        return

    _run_handler(
        "trigger retry",
        handler,
        formatter=lambda result: _format_status(result, workspace=workspace),
        workspace=workspace,
    )


@trigger_app.command("enable")
def trigger_enable(
    workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
) -> None:
    """Enable auto-train trigger for the workspace."""

    service = _load_service("pfe_core.pipeline", "pfe_core.services.pipeline")
    if service is None:
        _run_placeholder("trigger enable")
        return

    handler = _resolve_handler(service, "enable_auto_train_trigger")
    if handler is None:
        _run_placeholder("trigger enable")
        return

    _run_handler(
        "trigger enable",
        handler,
        formatter=lambda result: _format_status(result, workspace=workspace),
        workspace=workspace,
    )


@trigger_app.command("disable")
def trigger_disable(
    workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
) -> None:
    """Disable auto-train trigger for the workspace."""

    service = _load_service("pfe_core.pipeline", "pfe_core.services.pipeline")
    if service is None:
        _run_placeholder("trigger disable")
        return

    handler = _resolve_handler(service, "disable_auto_train_trigger")
    if handler is None:
        _run_placeholder("trigger disable")
        return

    _run_handler(
        "trigger disable",
        handler,
        formatter=lambda result: _format_status(result, workspace=workspace),
        workspace=workspace,
    )


@trigger_app.command("process-next")
def trigger_process_next(
    workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
) -> None:
    """Process the next queued auto-train item when queue mode is deferred."""

    service = _load_service("pfe_core.pipeline", "pfe_core.services.pipeline")
    if service is None:
        _run_placeholder("trigger process-next")
        return

    handler = _resolve_handler(service, "process_next_train_queue")
    if handler is None:
        _run_placeholder("trigger process-next")
        return

    _run_handler(
        "trigger process-next",
        handler,
        formatter=lambda result: _format_status(result, workspace=workspace),
        workspace=workspace,
    )


@trigger_app.command("process-batch")
def trigger_process_batch(
    limit: int = typer.Option(5, "--limit", min=1, help="Maximum queued items to process in one batch."),
    workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
) -> None:
    """Process up to N queued auto-train items when queue mode is deferred."""

    service = _load_service("pfe_core.pipeline", "pfe_core.services.pipeline")
    if service is None:
        _run_placeholder("trigger process-batch")
        return

    handler = _resolve_handler(service, "process_train_queue_batch")
    if handler is None:
        _run_placeholder("trigger process-batch")
        return

    _run_handler(
        "trigger process-batch",
        handler,
        formatter=lambda result: _format_status(result, workspace=workspace),
        workspace=workspace,
        limit=limit,
    )


@trigger_app.command("process-until-idle")
def trigger_process_until_idle(
    max_iterations: int = typer.Option(10, "--max-iterations", min=1, help="Maximum queued items to process before stopping."),
    workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
) -> None:
    """Process queued auto-train items until the queue drains or the iteration cap is reached."""

    service = _load_service("pfe_core.pipeline", "pfe_core.services.pipeline")
    if service is None:
        _run_placeholder("trigger process-until-idle")
        return

    handler = _resolve_handler(service, "process_train_queue_until_idle")
    if handler is None:
        _run_placeholder("trigger process-until-idle")
        return

    _run_handler(
        "trigger process-until-idle",
        handler,
        formatter=lambda result: _format_status(result, workspace=workspace),
        workspace=workspace,
        max_iterations=max_iterations,
    )


@trigger_app.command("run-worker-loop")
def trigger_run_worker_loop(
    max_cycles: int = typer.Option(10, "--max-cycles", min=1, help="Maximum worker loop cycles before stopping."),
    idle_rounds: int = typer.Option(1, "--idle-rounds", min=1, help="Stop after this many idle polling rounds."),
    poll_interval_seconds: float = typer.Option(0.0, "--poll-interval-seconds", min=0.0, help="Sleep between loop cycles in seconds."),
    workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
) -> None:
    """Run the train queue worker loop for a bounded number of cycles."""

    service = _load_service("pfe_core.pipeline", "pfe_core.services.pipeline")
    if service is None:
        _run_placeholder("trigger run-worker-loop")
        return

    handler = _resolve_handler(service, "run_train_queue_worker_loop")
    if handler is None:
        _run_placeholder("trigger run-worker-loop")
        return

    _run_handler(
        "trigger run-worker-loop",
        handler,
        formatter=lambda result: _format_status(result, workspace=workspace),
        workspace=workspace,
        max_cycles=max_cycles,
        idle_rounds=idle_rounds,
        poll_interval_seconds=poll_interval_seconds,
    )


@trigger_app.command("run-worker-runner")
def trigger_run_worker_runner(
    max_seconds: float = typer.Option(30.0, "--max-seconds", min=0.1, help="Maximum runner duration in seconds."),
    idle_sleep_seconds: float = typer.Option(1.0, "--idle-sleep-seconds", min=0.0, help="Sleep duration between idle polls."),
    workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
) -> None:
    """Run the long-poll train queue worker runner for a bounded duration."""

    service = _load_service("pfe_core.pipeline", "pfe_core.services.pipeline")
    if service is None:
        _run_placeholder("trigger run-worker-runner")
        return

    handler = _resolve_handler(service, "run_train_queue_worker_runner")
    if handler is None:
        _run_placeholder("trigger run-worker-runner")
        return

    _run_handler(
        "trigger run-worker-runner",
        handler,
        formatter=lambda result: _format_status(result, workspace=workspace),
        workspace=workspace,
        max_seconds=max_seconds,
        idle_sleep_seconds=idle_sleep_seconds,
    )


@trigger_app.command("stop-worker-runner")
def trigger_stop_worker_runner(
    workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
) -> None:
    """Request the long-poll train queue worker runner to stop."""

    service = _load_service("pfe_core.pipeline", "pfe_core.services.pipeline")
    if service is None:
        _run_placeholder("trigger stop-worker-runner")
        return

    handler = _resolve_handler(service, "stop_train_queue_worker_runner")
    if handler is None:
        _run_placeholder("trigger stop-worker-runner")
        return

    _run_handler(
        "trigger stop-worker-runner",
        handler,
        formatter=lambda result: _format_status(result, workspace=workspace),
        workspace=workspace,
    )


@trigger_app.command("worker-runner-history")
def trigger_worker_runner_history(
    limit: int = typer.Option(10, "--limit", min=1, help="Maximum worker runner history entries to show."),
    workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
) -> None:
    """Show worker runner lifecycle history for the workspace."""

    service = _load_service("pfe_core.pipeline", "pfe_core.services.pipeline")
    if service is None:
        _run_placeholder("trigger worker-runner-history")
        return

    handler = _resolve_handler(service, "train_queue_worker_runner_history")
    if handler is None:
        _run_placeholder("trigger worker-runner-history")
        return

    _run_handler(
        "trigger worker-runner-history",
        handler,
        formatter=_format_worker_runner_history,
        workspace=workspace,
        limit=limit,
    )


@trigger_app.command("approve-next")
def trigger_approve_next(
    note: Optional[str] = typer.Option(None, "--note", help="Optional approval note for audit trail."),
    workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
) -> None:
    """Approve the next queued item that is waiting for manual confirmation."""

    service = _load_service("pfe_core.pipeline", "pfe_core.services.pipeline")
    if service is None:
        _run_placeholder("trigger approve-next")
        return

    handler = _resolve_handler(service, "approve_next_train_queue")
    if handler is None:
        _run_placeholder("trigger approve-next")
        return

    _run_handler(
        "trigger approve-next",
        handler,
        formatter=lambda result: _format_status(result, workspace=workspace),
        workspace=workspace,
        note=note,
    )


@trigger_app.command("reject-next")
def trigger_reject_next(
    note: Optional[str] = typer.Option(None, "--note", help="Optional rejection note for audit trail."),
    workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
) -> None:
    """Reject the next queued item that is waiting for manual confirmation."""

    service = _load_service("pfe_core.pipeline", "pfe_core.services.pipeline")
    if service is None:
        _run_placeholder("trigger reject-next")
        return

    handler = _resolve_handler(service, "reject_next_train_queue")
    if handler is None:
        _run_placeholder("trigger reject-next")
        return

    _run_handler(
        "trigger reject-next",
        handler,
        formatter=lambda result: _format_status(result, workspace=workspace),
        workspace=workspace,
        note=note,
    )


@trigger_app.command("queue-history")
def trigger_queue_history(
    job_id: Optional[str] = typer.Option(None, "--job-id", help="Specific queue job id."),
    limit: int = typer.Option(10, "--limit", min=1, help="Maximum history entries to show."),
    workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
) -> None:
    """Show train queue history for the latest job or a specific queue job."""

    service = _load_service("pfe_core.pipeline", "pfe_core.services.pipeline")
    if service is None:
        _run_placeholder("trigger queue-history")
        return

    handler = _resolve_handler(service, "train_queue_history")
    if handler is None:
        _run_placeholder("trigger queue-history")
        return

    _run_handler(
        "trigger queue-history",
        handler,
        formatter=_format_train_queue_history,
        workspace=workspace,
        job_id=job_id,
        limit=limit,
    )


@trigger_app.command("status")
def trigger_status(
    workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON instead of formatted text."),
) -> None:
    """Show auto-train trigger status (thresholds, blocked reasons, queue state)."""

    service = _load_service("pfe_core.pipeline", "pfe_core.services.pipeline")
    if service is None:
        _run_placeholder("trigger status")
        return

    handler = _resolve_handler(service, "status")
    if handler is None:
        _run_placeholder("trigger status")
        return

    try:
        result = handler(workspace=workspace)
    except Exception as exc:
        friendly = _friendly_exception_message(exc)
        if friendly is not None:
            typer.secho(friendly, err=True, fg=typer.colors.RED)
            raise typer.Exit(code=1)
        raise

    mapping = _coerce_mapping(result) or {}
    trigger_data = _coerce_mapping(mapping.get("auto_train_trigger"))
    if trigger_data is None:
        typer.echo("No auto-train trigger status available.")
        return

    if json_output:
        typer.echo(json.dumps(trigger_data, ensure_ascii=False, indent=2, sort_keys=True))
        return

    from . import formatters_matrix
    typer.echo(formatters_matrix.format_status_matrix({"auto_train_trigger": trigger_data}, workspace=workspace))


@daemon_app.command("status")
def daemon_status(
    workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
) -> None:
    """Show the current background worker daemon control state."""

    service = _load_service("pfe_core.pipeline", "pfe_core.services.pipeline")
    if service is not None:
        handler = _resolve_handler(service, "train_queue_daemon_status", "daemon_status", "get_daemon_status")
        if handler is not None:
            _run_handler(
                "daemon status",
                handler,
                formatter=_format_train_queue_daemon_status,
                workspace=workspace,
            )
            return

    typer.echo(_format_train_queue_daemon_status(_read_train_queue_daemon_state(workspace) or {"workspace": workspace or "user_default", "command_status": "absent"}))


@daemon_app.command("start")
def daemon_start(
    workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
) -> None:
    """Request the background worker daemon to start."""

    service = _load_service("pfe_core.pipeline", "pfe_core.services.pipeline")
    if service is not None:
        handler = _resolve_handler(service, "start_train_queue_daemon", "run_train_queue_daemon", "daemon_start")
        if handler is not None:
            _run_handler(
                "daemon start",
                handler,
                formatter=_format_train_queue_daemon_status,
                workspace=workspace,
            )
            return

    typer.echo(
        _format_train_queue_daemon_status(
            _update_train_queue_daemon_state(
                workspace=workspace,
                desired_state="running",
                event="start_requested",
                reason="cli_requested",
            )
        )
    )


@daemon_app.command("stop")
def daemon_stop(
    workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
) -> None:
    """Request the background worker daemon to stop."""

    service = _load_service("pfe_core.pipeline", "pfe_core.services.pipeline")
    if service is not None:
        handler = _resolve_handler(service, "stop_train_queue_daemon", "daemon_stop")
        if handler is not None:
            _run_handler(
                "daemon stop",
                handler,
                formatter=_format_train_queue_daemon_status,
                workspace=workspace,
            )
            return

    typer.echo(
        _format_train_queue_daemon_status(
            _update_train_queue_daemon_state(
                workspace=workspace,
                desired_state="stopped",
                event="stop_requested",
                reason="cli_requested",
            )
        )
    )


@daemon_app.command("recover")
def daemon_recover(
    workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
    note: Optional[str] = typer.Option(None, "--note", help="Optional recovery note."),
) -> None:
    """Request daemon recovery with local restart policy bookkeeping."""

    service = _load_service("pfe_core.pipeline", "pfe_core.services.pipeline")
    if service is not None:
        handler = _resolve_handler(service, "recover_train_queue_daemon", "daemon_recover")
        if handler is not None:
            handler_kwargs: dict[str, Any] = {"workspace": workspace}
            try:
                signature = inspect.signature(handler)
            except (TypeError, ValueError):
                signature = None
            if signature is not None and "note" in signature.parameters:
                handler_kwargs["note"] = note
            _run_handler(
                "daemon recover",
                handler,
                formatter=_format_train_queue_daemon_status,
                **handler_kwargs,
            )
            return

    typer.echo(
        _format_train_queue_daemon_status(
            _daemon_recovery_payload(
                workspace=workspace,
                action="recover",
                note=note,
                reason="cli_requested",
            )
        )
    )


@daemon_app.command("restart")
def daemon_restart(
    workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
    note: Optional[str] = typer.Option(None, "--note", help="Optional restart note."),
) -> None:
    """Request daemon restart with local retry bookkeeping."""

    service = _load_service("pfe_core.pipeline", "pfe_core.services.pipeline")
    if service is not None:
        handler = _resolve_handler(service, "restart_train_queue_daemon", "daemon_restart")
        if handler is not None:
            handler_kwargs: dict[str, Any] = {"workspace": workspace}
            try:
                signature = inspect.signature(handler)
            except (TypeError, ValueError):
                signature = None
            if signature is not None and "note" in signature.parameters:
                handler_kwargs["note"] = note
            _run_handler(
                "daemon restart",
                handler,
                formatter=_format_train_queue_daemon_status,
                **handler_kwargs,
            )
            return

    typer.echo(
        _format_train_queue_daemon_status(
            _daemon_recovery_payload(
                workspace=workspace,
                action="restart",
                note=note,
                reason="cli_requested",
            )
        )
    )


@daemon_app.command("history")
def daemon_history(
    limit: int = typer.Option(10, "--limit", min=1, help="Maximum daemon history entries to show."),
    workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
) -> None:
    """Show the background worker daemon control history."""

    service = _load_service("pfe_core.pipeline", "pfe_core.services.pipeline")
    if service is not None:
        handler = _resolve_handler(service, "train_queue_daemon_history", "daemon_history")
        if handler is not None:
            _run_handler(
                "daemon history",
                handler,
                formatter=_format_train_queue_daemon_history,
                workspace=workspace,
                limit=limit,
            )
            return

    state = _read_train_queue_daemon_state(workspace) or {"workspace": workspace or "user_default"}
    history = list(state.get("history") or [])
    payload = {
        "workspace": state.get("workspace") or workspace or "user_default",
        "count": len(history),
        "last_event": state.get("last_event"),
        "last_reason": state.get("last_reason"),
        "items": history[-max(1, int(limit or 10)) :],
    }
    typer.echo(_format_train_queue_daemon_history(payload))


@daemon_app.command("health")
def daemon_health(
    workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Show comprehensive health status for daemon and runner."""

    service = _load_service("pfe_core.pipeline", "pfe_core.services.pipeline")
    if service is not None:
        handler = _resolve_handler(service, "get_health_status", "get_daemon_health_status")
        if handler is not None:
            if json_output:
                _run_handler_json("daemon health", handler, workspace=workspace)
                return
            _run_handler(
                "daemon health",
                handler,
                formatter=_format_daemon_health_status,
                workspace=workspace,
            )
            return

    typer.echo("Health status check not available.")


@daemon_app.command("heartbeat")
def daemon_heartbeat(
    workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Show heartbeat status for daemon and runner."""

    service = _load_service("pfe_core.pipeline", "pfe_core.services.pipeline")
    if service is not None:
        handler = _resolve_handler(service, "get_heartbeat_status", "get_daemon_heartbeat_status")
        if handler is not None:
            if json_output:
                _run_handler_json("daemon heartbeat", handler, workspace=workspace)
                return
            _run_handler(
                "daemon heartbeat",
                handler,
                formatter=_format_daemon_heartbeat_status,
                workspace=workspace,
            )
            return

    typer.echo("Heartbeat status check not available.")


@daemon_app.command("lease")
def daemon_lease(
    workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Show lease status for task execution."""

    service = _load_service("pfe_core.pipeline", "pfe_core.services.pipeline")
    if service is not None:
        handler = _resolve_handler(service, "get_lease_status", "get_runner_lease_status")
        if handler is not None:
            if json_output:
                _run_handler_json("daemon lease", handler, workspace=workspace)
                return
            _run_handler(
                "daemon lease",
                handler,
                formatter=_format_daemon_lease_status,
                workspace=workspace,
            )
            return

    typer.echo("Lease status check not available.")


@daemon_app.command("check-stale")
def daemon_check_stale(
    workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
    takeover: bool = typer.Option(False, "--takeover", help="Attempt to take over stale locks."),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Check if daemon or runner is stale and optionally trigger takeover."""

    service = _load_service("pfe_core.pipeline", "pfe_core.services.pipeline")
    if service is not None:
        handler = _resolve_handler(service, "check_stale_status", "check_daemon_stale")
        if handler is not None:
            if json_output:
                _run_handler_json("daemon check-stale", handler, workspace=workspace, takeover=takeover)
                return
            _run_handler(
                "daemon check-stale",
                handler,
                formatter=_format_daemon_stale_check,
                workspace=workspace,
                takeover=takeover,
            )
            return

    typer.echo("Stale check not available.")


@daemon_app.command("force-recovery")
def daemon_force_recovery(
    workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
    reason: Optional[str] = typer.Option(None, "--reason", help="Optional reason for forced recovery."),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Force daemon recovery with reset restart policy (bypasses backoff)."""

    service = _load_service("pfe_core.pipeline", "pfe_core.services.pipeline")
    if service is not None:
        handler = _resolve_handler(service, "force_recovery", "force_daemon_recovery")
        if handler is not None:
            if json_output:
                _run_handler_json("daemon force-recovery", handler, workspace=workspace, reason=reason)
                return
            _run_handler(
                "daemon force-recovery",
                handler,
                formatter=_format_train_queue_daemon_status,
                workspace=workspace,
                reason=reason,
            )
            return

    typer.echo("Force recovery not available.")


@daemon_app.command("alerts")
def daemon_alerts(
    workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
    level: Optional[str] = typer.Option(None, "--level", help="Filter by level (critical, error, warning, attention, info)."),
    scope: Optional[str] = typer.Option(None, "--scope", help="Filter by scope (daemon, runner, task, system)."),
    limit: int = typer.Option(10, "--limit", min=1, help="Maximum alerts to show."),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Show reliability alerts for monitoring."""

    service = _load_service("pfe_core.pipeline", "pfe_core.services.pipeline")
    if service is not None:
        handler = _resolve_handler(service, "get_reliability_alerts")
        if handler is not None:
            if json_output:
                _run_handler_json("daemon alerts", handler, workspace=workspace, level=level, scope=scope, limit=limit)
                return
            _run_handler(
                "daemon alerts",
                handler,
                formatter=_format_daemon_alerts,
                workspace=workspace,
                level=level,
                scope=scope,
                limit=limit,
            )
            return

    typer.echo("Alerts check not available.")


@candidate_app.command("promote")
def candidate_promote(
    note: Optional[str] = typer.Option(None, "--note", help="Optional promotion note for audit trail."),
    workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
) -> None:
    """Promote the current candidate adapter to latest promoted."""

    service = _load_service("pfe_core.pipeline", "pfe_core.services.pipeline")
    if service is None:
        _run_placeholder("candidate promote")
        return

    handler = _resolve_handler(service, "promote_candidate")
    if handler is None:
        _run_placeholder("candidate promote")
        return

    _run_handler(
        "candidate promote",
        handler,
        formatter=lambda result: _format_status(result, workspace=workspace),
        workspace=workspace,
        note=note,
    )


@candidate_app.command("archive")
def candidate_archive(
    note: Optional[str] = typer.Option(None, "--note", help="Optional archive note for audit trail."),
    workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
) -> None:
    """Archive the current candidate adapter without changing latest promoted."""

    service = _load_service("pfe_core.pipeline", "pfe_core.services.pipeline")
    if service is None:
        _run_placeholder("candidate archive")
        return

    handler = _resolve_handler(service, "archive_candidate")
    if handler is None:
        _run_placeholder("candidate archive")
        return

    _run_handler(
        "candidate archive",
        handler,
        formatter=lambda result: _format_status(result, workspace=workspace),
        workspace=workspace,
        note=note,
    )




@eval_trigger_app.command("enable")
def eval_trigger_enable(
    workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
) -> None:
    """Enable auto-eval trigger for the workspace."""

    service = _load_service("pfe_core.pipeline", "pfe_core.services.pipeline")
    if service is None:
        _run_placeholder("eval-trigger enable")
        return

    handler = _resolve_handler(service, "enable_auto_eval_trigger")
    if handler is None:
        _run_placeholder("eval-trigger enable")
        return

    _run_handler(
        "eval-trigger enable",
        handler,
        formatter=lambda result: _format_status(result, workspace=workspace),
        workspace=workspace,
    )


@eval_trigger_app.command("disable")
def eval_trigger_disable(
    workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
) -> None:
    """Disable auto-eval trigger for the workspace."""

    service = _load_service("pfe_core.pipeline", "pfe_core.services.pipeline")
    if service is None:
        _run_placeholder("eval-trigger disable")
        return

    handler = _resolve_handler(service, "disable_auto_eval_trigger")
    if handler is None:
        _run_placeholder("eval-trigger disable")
        return

    _run_handler(
        "eval-trigger disable",
        handler,
        formatter=lambda result: _format_status(result, workspace=workspace),
        workspace=workspace,
    )


@eval_trigger_app.command("status")
def eval_trigger_status(
    workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
) -> None:
    """Show auto-eval trigger status for the workspace."""

    service = _load_service("pfe_core.pipeline", "pfe_core.services.pipeline")
    if service is None:
        _run_placeholder("eval-trigger status")
        return

    handler = _resolve_handler(service, "get_auto_eval_trigger_status")
    if handler is None:
        _run_placeholder("eval-trigger status")
        return

    def _format_eval_trigger_status(result: Any) -> str:
        mapping = _coerce_mapping(result)
        if mapping is None:
            return _format_scalar(result)

        lines = ["Auto-eval trigger status"]
        enabled = mapping.get("enabled", False)
        lines.append(f"enabled: {enabled}")

        if enabled:
            auto_promote = mapping.get("auto_promote_after_eval", False)
            win_rate = mapping.get("win_rate_threshold", 0.6)
            lines.append(f"auto_promote_after_eval: {auto_promote}")
            lines.append(f"win_rate_threshold: {win_rate:.0%}")

        eval_config = _coerce_mapping(mapping.get("eval_config"))
        if eval_config:
            lines.append("eval_config:")
            for key, value in eval_config.items():
                lines.append(f"  {key}: {value}")

        promote_config = _coerce_mapping(mapping.get("promote_config"))
        if promote_config:
            lines.append("promote_config:")
            for key, value in promote_config.items():
                lines.append(f"  {key}: {value}")

        return "\n".join(lines)

    _run_handler(
        "eval-trigger status",
        handler,
        formatter=_format_eval_trigger_status,
        workspace=workspace,
    )

@candidate_app.command("history")
def candidate_history(
    limit: int = typer.Option(10, "--limit", min=1, help="Maximum candidate history entries to show."),
    workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
) -> None:
    """Show candidate lifecycle history for the workspace."""

    service = _load_service("pfe_core.pipeline", "pfe_core.services.pipeline")
    if service is None:
        _run_placeholder("candidate history")
        return

    handler = _resolve_handler(service, "candidate_history")
    if handler is None:
        _run_placeholder("candidate history")
        return

    _run_handler(
        "candidate history",
        handler,
        formatter=_format_candidate_history,
        workspace=workspace,
        limit=limit,
    )


@candidate_app.command("timeline")
def candidate_timeline(
    limit: int = typer.Option(10, "--limit", min=1, help="Maximum candidate timeline entries to show."),
    workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
) -> None:
    """Show candidate lifecycle timeline for the workspace."""

    service = _load_service("pfe_core.pipeline", "pfe_core.services.pipeline")
    if service is None:
        _run_placeholder("candidate timeline")
        return

    handler = _resolve_handler(service, "candidate_timeline")
    if handler is None:
        _run_placeholder("candidate timeline")
        return

    _run_handler(
        "candidate timeline",
        handler,
        formatter=_format_candidate_timeline,
        workspace=workspace,
        limit=limit,
    )


@collect_app.command("start")
def collect_start(
    workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
) -> None:
    """Enable signal collection for the current workspace."""

    service = _load_service("pfe_core.pipeline", "pfe_core.services.pipeline")
    if service is None:
        _run_placeholder("collect start")
        return

    handler = _resolve_handler(service, "start_signal_collection")
    if handler is None:
        _run_placeholder("collect start")
        return

    _run_handler(
        "collect start",
        handler,
        formatter=lambda result: _format_status(result, workspace=workspace),
        workspace=workspace,
    )


@collect_app.command("stop")
def collect_stop(
    workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
) -> None:
    """Disable signal collection for the current workspace."""

    service = _load_service("pfe_core.pipeline", "pfe_core.services.pipeline")
    if service is None:
        _run_placeholder("collect stop")
        return

    handler = _resolve_handler(service, "stop_signal_collection")
    if handler is None:
        _run_placeholder("collect stop")
        return

    _run_handler(
        "collect stop",
        handler,
        formatter=lambda result: _format_status(result, workspace=workspace),
        workspace=workspace,
    )


@collect_app.command("status")
def collect_status(
    workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
) -> None:
    """Show signal collection statistics."""

    from pfe_core.collector import ChatCollector, CollectorConfig
    from pfe_core.config import PFEConfig

    config = PFEConfig.load()
    collector_config = config.collector if hasattr(config, 'collector') else CollectorConfig()
    home = str(config.home) if hasattr(config, 'home') else None

    collector = ChatCollector(
        workspace=workspace or "user_default",
        config=collector_config,
        home=home
    )

    stats = collector.get_stats()

    typer.echo("Signal Collection Status")
    typer.echo("=" * 40)
    typer.echo(f"Enabled: {stats['config']['enabled']}")
    typer.echo(f"Total Interactions: {stats['total_interactions']}")
    typer.echo(f"Total Signals: {stats['total_signals']}")
    typer.echo("\nSignals by Type:")
    for signal_type, count in stats['signals_by_type'].items():
        typer.echo(f"  {signal_type}: {count}")
    typer.echo("\nThresholds:")
    typer.echo(f"  Accept: {stats['config']['accept_threshold']}")
    typer.echo(f"  Edit: {stats['config']['edit_threshold']}")


@collect_app.command("review")
def collect_review(
    workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
    signal_type: Optional[str] = typer.Option(None, "--type", help="Filter by signal type (accept, edit, reject, regenerate)."),
    min_confidence: float = typer.Option(0.0, "--min-confidence", help="Minimum confidence threshold."),
    max_confidence: float = typer.Option(1.0, "--max-confidence", help="Maximum confidence threshold."),
    limit: int = typer.Option(20, "--limit", help="Maximum number of signals to display."),
) -> None:
    """Review collected signals for manual verification."""

    from pfe_core.collector import ChatCollector, CollectorConfig
    from pfe_core.config import PFEConfig

    config = PFEConfig.load()
    collector_config = config.collector if hasattr(config, 'collector') else CollectorConfig()
    home = str(config.home) if hasattr(config, 'home') else None

    collector = ChatCollector(
        workspace=workspace or "user_default",
        config=collector_config,
        home=home
    )

    signals = collector.get_signals_for_review(
        signal_type=signal_type,
        min_confidence=min_confidence,
        max_confidence=max_confidence,
        limit=limit
    )

    if not signals:
        typer.echo("No signals found matching the criteria.")
        return

    typer.echo(f"Collected Signals (showing {len(signals)})")
    typer.echo("=" * 60)

    for i, signal in enumerate(signals, 1):
        typer.echo(f"\n[{i}] Signal ID: {signal.signal_id}")
        typer.echo(f"    Type: {signal.signal_type}")
        typer.echo(f"    Confidence: {signal.confidence:.2f}")
        typer.echo(f"    Rule: {signal.extraction_rule}")
        typer.echo(f"    Session: {signal.session_id}")
        if signal.edit_distance is not None:
            typer.echo(f"    Edit Distance: {signal.edit_distance}")
        if signal.response_time_seconds is not None:
            typer.echo(f"    Response Time: {signal.response_time_seconds:.1f}s")
        typer.echo(f"    Context: {signal.context[:100]}..." if len(signal.context) > 100 else f"    Context: {signal.context}")


def _record_train_cli_state(result: Any, *, workspace: str | None = None) -> None:
    mapping = _coerce_mapping(result)
    if mapping is None:
        return

    payload = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "workspace": workspace or "default",
        "recent_training": {
            "version": mapping.get("version"),
            "state": "pending_eval",
            "execution_backend": mapping.get("execution_backend"),
            "executor_mode": _pick_first(_coerce_mapping(mapping.get("executor_spec")), "executor_mode")
            or _pick_first(_coerce_mapping(mapping.get("backend_dispatch")), "executor_mode")
            or _pick_first(_coerce_mapping(mapping.get("job_execution")), "executor_mode")
            or "fallback",
            "job_execution": mapping.get("job_execution"),
            "job_execution_summary": mapping.get("job_execution_summary"),
            "real_execution_summary": mapping.get("real_execution_summary"),
            "export_execution": mapping.get("export_execution"),
            "export_toolchain_summary": mapping.get("export_toolchain_summary"),
        },
    }
    _write_cli_state(workspace, payload)


def _format_serve_preview(
    *,
    port: int,
    host: str,
    adapter: str,
    workspace: str | None,
    api_key: str | None,
    real_local: bool,
) -> str:
    """Return a readable preflight summary for serve() without mutating runtime state."""

    # Matrix theme - default style
    cached_state = _read_cli_state(workspace)
    recent_training = None
    if cached_state is not None:
        recent_training = _coerce_mapping(cached_state.get("recent_training"))
    latest_snapshot = _lookup_adapter_snapshot("latest", workspace=workspace)
    latest_training = _coerce_mapping(latest_snapshot)
    return formatters_matrix.format_serve_preview_matrix(
        port=port, host=host, adapter=adapter, workspace=workspace,
        api_key=api_key, real_local=real_local,
        recent_training=recent_training,
        latest_training=latest_training,
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
    """Legacy plain text formatter (kept for reference)."""
    preview = _optional_module_call(
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
        preview_mapping = _coerce_mapping(preview)
        runtime = _serve_preview_runtime_mapping(preview)
        if runtime is not None:
            lines.append(
                "runtime: "
                + " | ".join(
                    part
                    for part in (
                        f"provider={_format_scalar(runtime.get('provider'))}",
                        f"dry_run={_format_scalar(runtime.get('dry_run'))}",
                        f"uvicorn_available={_format_scalar(runtime.get('uvicorn_available'))}",
                        f"app_target={_format_scalar(runtime.get('app_target'))}",
                    )
                    if part is not None
                )
            )
            launch_mode = _serve_preview_launch_mode(preview)
            if launch_mode is not None:
                lines.append(f"server launch_mode: {_format_scalar(launch_mode)}")
            command = runtime.get("command")
            if not command and hasattr(preview, "command"):
                try:
                    command = list(getattr(preview, "command"))
                except Exception:
                    command = getattr(preview, "command")
            if command:
                lines.append(f"command: {_format_scalar(command)}")
        plan_snapshots = _build_plan_snapshots(workspace, {})
        trainer_line = _format_trainer_summary(plan_snapshots.get("trainer"))
        if trainer_line is not None:
            lines.append(trainer_line)
        inference_plan = _coerce_mapping(plan_snapshots.get("inference"))
        export_plan = _coerce_mapping(plan_snapshots.get("export"))
        if inference_plan is not None:
            dispatch_line = _format_backend_dispatch(inference_plan)
            if dispatch_line is not None:
                lines.append(dispatch_line)
        if export_plan is not None:
            export_line = _format_export_write(export_plan)
            if export_line is not None:
                lines.append(export_line)
        latest_snapshot = _lookup_adapter_snapshot("latest", workspace=workspace)
        latest_line = _format_adapter_snapshot_line("latest promoted", latest_snapshot, include_latest=True)
        if latest_line is not None:
            lines.append(latest_line)
        cached_state = _read_cli_state(workspace)
        recent_snapshot = None
        if cached_state is not None:
            recent_snapshot = _coerce_mapping(cached_state.get("recent_training"))
        if recent_snapshot is None:
            recent_snapshot = _lookup_recent_adapter_snapshot(workspace=workspace)
        recent_lines = _format_recent_training_snapshot(recent_snapshot or cached_state)
        if recent_lines is not None:
            lines.extend(recent_lines)
        if preview_mapping and preview_mapping.get("uvicorn_module"):
            lines.append(f"uvicorn module: {_format_scalar(preview_mapping.get('uvicorn_module'))}")
    return "\n".join(lines)


def _format_train_preview(
    *,
    method: str,
    epochs: int,
    base_model: str | None,
    train_type: str,
    workspace: str | None,
    snapshot_workspace: str | None = None,
    backend_hint: str | None,
) -> str:
    """Render a compact training preflight summary without executing training."""

    trainer_service = _optional_module_call("pfe_core.trainer", "service")
    runtime = _optional_module_call("pfe_core.trainer.runtime", "detect_trainer_runtime")
    runtime_mapping = _coerce_mapping(runtime) or {}
    target_inference_backend = "llama_cpp" if "llama" in str(base_model or "").lower() else "transformers"
    backend_plan = _optional_module_call(
        "pfe_core.trainer.runtime",
        "summarize_trainer_backend_plan",
        train_type=train_type,
        runtime=runtime_mapping or None,
        target_inference_backend=target_inference_backend,
    )

    backend_dispatch = None
    if trainer_service is not None and hasattr(trainer_service, "_dispatch_training_backend"):
        try:
            backend_dispatch = trainer_service._dispatch_training_backend(  # type: ignore[attr-defined]
                backend_plan=_coerce_mapping(backend_plan) or {},
                runtime=runtime_mapping,
                backend_hint=backend_hint,
                allow_mock_fallback=True,
            )
        except Exception:
            backend_dispatch = None

    executor_spec = None
    if trainer_service is not None and hasattr(trainer_service, "_resolve_training_executor"):
        try:
            executor_spec = trainer_service._resolve_training_executor(  # type: ignore[attr-defined]
                backend_dispatch=backend_dispatch or _coerce_mapping(backend_plan) or {},
                runtime=runtime_mapping,
                backend_hint=backend_hint,
                allow_mock_fallback=True,
            )
        except Exception:
            executor_spec = None

    backend_plan_mapping = _coerce_mapping(backend_plan) or {}
    dispatch_mapping = _coerce_mapping(executor_spec) or _coerce_mapping(backend_dispatch) or {}
    execution_backend = _pick_first(dispatch_mapping, "execution_backend")
    if execution_backend is None:
        execution_backend = _pick_first(backend_plan_mapping, "recommended_backend", "selected_backend") or target_inference_backend
    execution_mode = _pick_first(dispatch_mapping, "execution_mode", "executor_mode")
    if execution_mode is None:
        reason = str(_pick_first(backend_plan_mapping, "reason") or "").lower()
        if "mock_local" in str(execution_backend).lower() or any(token in reason for token in ("fallback", "auto-selected", "dry-run")):
            execution_mode = "fallback"
        else:
            execution_mode = "real"
    export_artifact_format = _pick_first(
        dispatch_mapping or backend_plan_mapping,
        "export_format",
        "artifact_format",
    )
    export_preview = _optional_module_call(
        "pfe_core.inference.export_runtime",
        "build_export_runtime_spec",
        target_backend=execution_backend or target_inference_backend,
        source_artifact_format=export_artifact_format,
        workspace=workspace,
        source_model=base_model,
        training_run_id=None,
        num_samples=None,
        extra_metadata={
            "method": method,
            "epochs": epochs,
            "train_type": train_type,
        },
    )

    lines = [
        "PFE train plan",
        f"request: method={method} | epochs={epochs} | train_type={train_type} | workspace={workspace or 'default'}",
    ]
    trainer_line = _format_trainer_summary(
        {
            "runtime": runtime_mapping,
            "plans": backend_plan_mapping,
        }
    )
    if trainer_line is not None:
        lines.append(trainer_line)
    adapter_snapshot = _lookup_adapter_snapshot("latest", workspace=snapshot_workspace)
    adapter_line = _format_adapter_snapshot_line("latest promoted", adapter_snapshot, include_latest=True)
    if adapter_line is not None:
        lines.append(adapter_line)
    planned_executor_mode = _pick_first(dispatch_mapping, "executor_mode") or _pick_first(dispatch_mapping, "execution_mode") or "fallback"
    lines.append(
        "job-execution: "
        + " | ".join(
            [
                "status=planned",
                f"executor_mode={_format_scalar(planned_executor_mode)}",
                "execution_state=planned",
            ]
        )
    )
    lines.append(
        _format_backend_dispatch(
            {
                **backend_plan_mapping,
                **dispatch_mapping,
                "execution_backend": execution_backend,
                "execution_mode": execution_mode,
                "runtime_device": _pick_first(runtime_mapping, "runtime_device"),
                "requires_export_step": _pick_first(dispatch_mapping or backend_plan_mapping, "requires_export_step"),
                "required_artifact_format": export_artifact_format,
            }
        )
        or "backend-dispatch: n/a"
    )
    if export_preview is not None:
        export_line = _format_export_write(export_preview)
        if export_line is not None:
            lines.append(export_line)
    elif execution_backend is not None:
        lines.append(
            "export-write: "
            + " | ".join(
                [
                    f"gguf_export={'required' if str(export_artifact_format).lower() == 'gguf_merged' else 'not_required'}",
                    "write_state=planned",
                    f"target_artifact_format={_format_scalar(export_artifact_format)}",
                    f"execution_backend={_format_scalar(execution_backend)}",
                    f"execution_mode={_format_scalar(execution_mode)}",
                ]
            )
        )
    return "\n".join(lines)


def _run_placeholder(command_name: str) -> None:
    typer.echo(
        f"[pfe] {command_name}: command is not available in the current environment. "
        "Some PFE surfaces are still bootstrap-oriented or require optional services to be resolved."
    )


def _resolve_handler(service: Any, *names: str) -> Any | None:
    for name in names:
        candidate = getattr(service, name, None)
        if candidate is not None:
            return candidate
    if callable(service):
        return service
    return None


@app.command("generate")
def generate(
    scenario: str = typer.Option(..., "--scenario", help="Target scenario, e.g. life-coach."),
    style: str = typer.Option(..., "--style", help="Desired response style."),
    num: int = typer.Option(200, "--num", min=1, help="Number of samples to generate."),
    output: Optional[str] = typer.Option(None, "--output", help="Optional output path for generated samples."),
    workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
) -> None:
    """Generate cold-start samples using the current bootstrap workflow. This is useful for seeding data, but it should still be treated as a heuristic/template-backed capability."""

    service = _load_service("pfe_core.pipeline", "pfe_core.curator", "pfe_core.services.pipeline")
    if service is None:
        _run_placeholder("generate")
        return

    handler = _resolve_handler(service, "generate")
    if handler is None:
        _run_placeholder("generate")
        return

    _run_handler(
        "generate",
        handler,
        scenario=scenario,
        style=style,
        num_samples=num,
        output=output,
        workspace=workspace,
    )


@app.command("distill")
def distill(
    teacher_model: str = typer.Option(..., "--teacher-model", help="Teacher model name or provider id."),
    scenario: str = typer.Option(..., "--scenario", help="Target scenario, e.g. life-coach."),
    style: str = typer.Option(..., "--style", help="Desired response style."),
    num: int = typer.Option(200, "--num", min=1, help="Number of samples to distill."),
    output: Optional[str] = typer.Option(None, "--output", help="Optional output path for distilled samples."),
    workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview what would be distilled without executing."),
) -> None:
    """Run the current distillation workflow. This path can still fall back to template or synthetic generation unless a real teacher path is configured."""

    if dry_run:
        typer.echo(f"[dry-run] Would distill {num} samples with teacher={teacher_model}, scenario={scenario}, style={style}")
        return

    service = _load_service("pfe_core.curator", "pfe_core.pipeline", "pfe_core.services.pipeline")
    if service is None:
        _run_placeholder("distill")
        return

    handler = _resolve_handler(service, "distill", "run_distillation")
    if handler is None:
        _run_placeholder("distill")
        return

    _run_handler(
        "distill",
        handler,
        teacher_model=teacher_model,
        scenario=scenario,
        style=style,
        num_samples=num,
        output=output,
        workspace=workspace,
    )


@app.command("train")
def train(
    method: str = typer.Option("qlora", "--method", help="Training method, e.g. lora or qlora."),
    epochs: int = typer.Option(3, "--epochs", min=1, help="Training epochs."),
    base_model: Optional[str] = typer.Option(None, "--base-model", help="Base model id or local path."),
    incremental: bool = typer.Option(False, "--incremental", help="Continue training from an existing adapter."),
    base_adapter: Optional[str] = typer.Option(None, "--base-adapter", help="Parent adapter version or path for incremental training."),
    train_type: str = typer.Option("sft", "--train-type", help="Training type, e.g. sft or dpo."),
    workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
) -> None:
    """Train an adapter. The future trainer service decides backend selection and artifact export."""

    service = _load_service("pfe_core.trainer", "pfe_core.pipeline", "pfe_core.services.pipeline")
    if service is None:
        _run_placeholder("train")
        return

    handler = _resolve_handler(service, "train_result", "train")
    handler_kwargs: dict[str, Any] = {
        "method": method,
        "epochs": epochs,
        "base_model": base_model,
        "train_type": train_type,
        "workspace": workspace,
    }
    if incremental:
        incremental_handler = _resolve_handler(service, "train_incremental")
        if incremental_handler is None:
            typer.secho("Incremental training is unavailable because no train_incremental handler is registered.", err=True, fg=typer.colors.RED)
            raise typer.Exit(code=1)
        if not base_adapter:
            typer.secho("Incremental training requires --base-adapter.", err=True, fg=typer.colors.RED)
            raise typer.Exit(code=1)
        handler = incremental_handler
        handler_kwargs = {
            "base_adapter": base_adapter,
            "method": method,
            "epochs": epochs,
            "train_type": train_type,
            "workspace": workspace,
        }
    if handler is None:
        _run_placeholder("train")
        return

    typer.echo(
        _format_train_preview(
            method=method,
            epochs=epochs,
            base_model=base_model,
            train_type=train_type,
            workspace=workspace,
            snapshot_workspace=workspace or "user_default",
            backend_hint=None,
        )
    )
    _run_handler(
        "train",
        handler,
        formatter=lambda result: _format_train_result(result, workspace=workspace or "user_default"),
        on_result=lambda result: _record_train_cli_state(result, workspace=workspace or "user_default"),
        **handler_kwargs,
    )


@app.command("dpo")
def dpo_train(
    method: str = typer.Option("qlora", "--method", help="Training method, e.g. lora or qlora."),
    epochs: int = typer.Option(3, "--epochs", min=1, help="Training epochs."),
    base_model: Optional[str] = typer.Option(None, "--base-model", help="Base model id or local path."),
    base_adapter: Optional[str] = typer.Option(None, "--base-adapter", help="Parent SFT adapter for incremental DPO training."),
    min_confidence: float = typer.Option(0.4, "--min-confidence", min=0.0, max=1.0, help="Minimum signal confidence for DPO pairs."),
    workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
    preview: bool = typer.Option(False, "--preview", help="Preview DPO dataset without training."),
    train: bool = typer.Option(False, "--train", help="Execute real DPO training (default is preview mode)."),
) -> None:
    """Train using Direct Preference Optimization (DPO).

    Builds preference pairs from accepted/rejected/edited signals and trains
the model to prefer user-approved responses over rejected ones.

    Examples:
        pfe dpo --preview                    # Preview available DPO pairs
        pfe dpo --train                      # Run DPO training
        pfe dpo --train --base-adapter v001  # Run DPO on top of SFT adapter v001
    """
    service = _load_service("pfe_core.trainer", "pfe_core.pipeline", "pfe_core.services.pipeline")
    if service is None:
        _run_placeholder("dpo")
        return

    # Preview mode - just show dataset stats (default if --train not specified)
    if preview or not train:
        preview_handler = _resolve_handler(service, "build_dpo_dataset")
        if preview_handler:
            result = preview_handler(workspace=workspace, min_confidence=min_confidence)
            typer.echo("DPO Dataset Preview:")
            typer.echo(f"  Estimated pairs: {result.get('num_pairs', 0)}")
            typer.echo(f"  Min confidence: {result.get('min_confidence', min_confidence)}")
            typer.echo(f"  Signal statistics: {result.get('statistics', {})}")
            if not train:
                typer.echo("\nUse --train to execute DPO training.")
                return
        else:
            typer.secho("DPO preview not available.", err=True, fg=typer.colors.YELLOW)
            if not train:
                return

    # Training mode
    handler = _resolve_handler(service, "train_dpo")
    if handler is None:
        typer.secho("DPO training is unavailable. Ensure pfe_core is installed with trl support.", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1)

    handler_kwargs: dict[str, Any] = {
        "method": method,
        "epochs": epochs,
        "base_model": base_model,
        "workspace": workspace,
        "min_confidence": min_confidence,
    }

    if base_adapter:
        handler_kwargs["base_adapter_path"] = base_adapter

    typer.echo(f"Starting DPO training (method={method}, epochs={epochs}, min_confidence={min_confidence})...")
    if base_adapter:
        typer.echo(f"  Base SFT adapter: {base_adapter}")
    _run_handler(
        "dpo",
        handler,
        formatter=lambda result: _format_train_result(result, workspace=workspace or "user_default"),
        on_result=lambda result: _record_train_cli_state(result, workspace=workspace or "user_default"),
        **handler_kwargs,
    )


@app.command("eval")
def eval(
    base_model: str = typer.Option(..., "--base-model", help="Base model id or the special value 'base'."),
    adapter: str = typer.Option("latest", "--adapter", help="Adapter version to evaluate."),
    compare: Optional[str] = typer.Option(None, "--compare", help="Compare against another evaluated adapter version, e.g. --adapter v001 --compare v002."),
    num_samples: int = typer.Option(20, "--num-samples", min=1, help="Number of holdout/test samples."),
    workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
) -> None:
    """Evaluate an adapter using the future judge pipeline. Training split must not be used here."""

    service = _load_service("pfe_core.evaluator", "pfe_core.pipeline", "pfe_core.services.pipeline")
    if service is None:
        _run_placeholder("eval")
        return

    handler_kwargs: dict[str, Any]
    if compare:
        compare_handler = _resolve_handler(service, "compare_evaluations", "compare_eval_versions")
        if compare_handler is None:
            typer.secho("Compare-eval is unavailable because no compare handler is registered.", err=True, fg=typer.colors.RED)
            raise typer.Exit(code=1)
        handler = compare_handler
        handler_kwargs = {
            "left_adapter": adapter,
            "right_adapter": compare,
            "workspace": workspace,
        }
    else:
        handler = _resolve_handler(service, "evaluate", "eval")
        if handler is None:
            _run_placeholder("eval")
            return
        handler_kwargs = {
            "base_model": base_model,
            "adapter": adapter,
            "num_samples": num_samples,
            "workspace": workspace,
        }

    _run_handler(
        "eval",
        handler,
        formatter=lambda result: _format_eval_result(result, workspace=workspace or "user_default"),
        **handler_kwargs,
    )


@app.command("serve")
def serve(
    port: int = typer.Option(8921, "--port", min=1, max=65535, help="Port to bind."),
    host: str = typer.Option("127.0.0.1", "--host", help="Bind host, default strict_local loopback."),
    adapter: str = typer.Option("latest", "--adapter", help="Adapter version to load at startup."),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="Optional API key for remote access."),
    workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
    live: bool = typer.Option(False, "--live", help="Actually launch the local server instead of previewing the serve plan."),
    real_local: bool = typer.Option(False, "--real-local", help="Explicitly allow real local model loading for chat inference."),
) -> None:
    """Start the OpenAI-compatible inference server. This does not create the personalized loop by itself."""

    service = _load_service("pfe_server.app", "pfe_server", "pfe_core.inference", "pfe_core.pipeline")
    if service is None:
        _run_placeholder("serve")
        return

    previous_real_local = os.environ.get("PFE_ENABLE_REAL_LOCAL_INFERENCE")
    if real_local:
        os.environ["PFE_ENABLE_REAL_LOCAL_INFERENCE"] = "1"
    typer.echo(
        _format_serve_preview(
            port=port,
            host=host,
            adapter=adapter,
            workspace=workspace,
            api_key=api_key,
            real_local=real_local,
        )
    )
    handler = _resolve_handler(service, "serve", "run", "start")
    if handler is None:
        _run_placeholder("serve")
        return

    try:
        _run_handler(
            "serve",
            handler,
            formatter=_format_serve,
            port=port,
            host=host,
            adapter=adapter,
            api_key=api_key,
            workspace=workspace,
            dry_run=not live,
        )
    finally:
        if previous_real_local is None:
            os.environ.pop("PFE_ENABLE_REAL_LOCAL_INFERENCE", None)
        else:
            os.environ["PFE_ENABLE_REAL_LOCAL_INFERENCE"] = previous_real_local


def _console_submit_feedback(
    workspace: str,
    session_id: str,
    request_id: str,
    user_message: str,
    assistant_message: str,
    response_time_seconds: float,
    adapter_version: str,
    action: str,
    edited_text: str | None = None,
) -> list[dict[str, Any]]:
    """Submit feedback via ChatCollector for console chat interactions."""
    try:
        from pfe_core.collector import ChatCollector, CollectorConfig
        from pfe_core.models import ChatInteraction
        from pfe_core.config import PFEConfig

        config = PFEConfig.load()
        collector_config = config.collector if hasattr(config, "collector") else CollectorConfig()
        home = str(config.home) if hasattr(config, "home") else None

        collector = ChatCollector(
            workspace=workspace,
            config=collector_config,
            home=home,
        )

        interaction = ChatInteraction(
            session_id=session_id,
            request_id=request_id,
            user_message=user_message,
            assistant_message=assistant_message,
            adapter_version=adapter_version,
            response_time_seconds=response_time_seconds,
        )

        next_message = None
        if action == "continue":
            next_message = ""

        signals = collector.on_interaction(
            interaction=interaction,
            next_user_message=next_message,
            edited_text=edited_text,
            action=action,  # type: ignore[arg-type]
        )
        return [s.to_dict() for s in signals]
    except Exception:
        # Feedback collection is best-effort; don't fail the console
        return []


@app.command("console")
def console(
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Workspace label."),
    interactive: bool = typer.Option(False, "--interactive", help="Open a simple prompt loop on top of the console snapshot."),
    model: str = typer.Option("local", "--model", help="Chat model id or special local alias for interactive mode."),
    adapter: str = typer.Option("latest", "--adapter", help="Adapter version used for interactive chat mode."),
    temperature: float = typer.Option(0.7, "--temperature", min=0.0, max=2.0, help="Chat temperature for interactive mode."),
    max_tokens: Optional[int] = typer.Option(None, "--max-tokens", min=1, help="Optional max tokens for interactive chat mode."),
    real_local: bool = typer.Option(False, "--real-local", help="Allow real local inference during interactive chat."),
    watch: bool = typer.Option(False, "--watch", help="Refresh the console snapshot repeatedly."),
    refresh_seconds: float = typer.Option(2.0, "--refresh-seconds", min=0.1, help="Refresh interval when --watch is enabled."),
    cycles: int = typer.Option(1, "--cycles", min=1, help="Number of render cycles. Use 1 for a single snapshot."),
) -> None:
    """Render a Rich-based PFE operations console with optional prompt mode."""

    # Matrix theme boot banner - default style
    from .pixel_logo import render_boot_banner
    typer.echo(render_boot_banner())
    typer.echo(f"{formatters_matrix.MatrixColors.GREEN}  [■] Initializing console interface...{formatters_matrix.MatrixColors.RESET}")
    typer.echo(f"{formatters_matrix.MatrixColors.GREEN}  [■] Loading Rich console components...{formatters_matrix.MatrixColors.RESET}")
    typer.echo(f"{formatters_matrix.MatrixColors.GREEN}  [■] Establishing service connections...{formatters_matrix.MatrixColors.RESET}")
    typer.echo("")
    typer.echo(f"{formatters_matrix.MatrixColors.GREEN_BRIGHT}{formatters_matrix.MatrixColors.BOLD}  >> ENTERING MATRIX CONSOLE MODE <<{formatters_matrix.MatrixColors.RESET}")
    typer.echo("")

    service = _load_service("pfe_core.pipeline", "pfe_core.status", "pfe_server.app", "pfe_core.services.pipeline")
    if service is None:
        _run_placeholder("console")
        return

    handler = _resolve_handler(service, "status", "get_status")
    if handler is None:
        _run_placeholder("console")
        return

    from rich.console import Console as RichConsole
    from rich.live import Live

    from .console_app import build_console_renderable, render_console_snapshot

    if interactive:
        console_ui = RichConsole()
        chat_handler = _resolve_handler(service, "chat_completion")
        feedback = "interactive mode ready"
        transcript: list[dict[str, str]] = []
        chat_messages: list[dict[str, str]] = []
        input_history: list[str] = []
        session_id = f"console-{uuid4().hex[:8]}"
        mode_name = "chat"
        last_interaction: dict[str, Any] | None = None

        previous_real_local = os.environ.get("PFE_ENABLE_REAL_LOCAL_INFERENCE")
        if real_local:
            os.environ["PFE_ENABLE_REAL_LOCAL_INFERENCE"] = "1"
        try:
            with Live(console=console_ui, auto_refresh=False, screen=False) as live:
                payload = _console_snapshot_payload(handler, workspace=workspace)
                last_sidebar_refresh_at = time.monotonic()
                ops_refresh_state = "live"

                def _refresh_console(
                    input_text: str = "",
                    *,
                    input_cursor: int | None = None,
                    current_feedback: str | None = None,
                    force_sidebar: bool = False,
                ) -> None:
                    nonlocal payload, last_sidebar_refresh_at, ops_refresh_state
                    now = time.monotonic()
                    should_refresh_sidebar = force_sidebar or (now - last_sidebar_refresh_at >= refresh_seconds)
                    if should_refresh_sidebar:
                        live.update(
                            build_console_renderable(
                                payload,
                                workspace=workspace,
                                session_messages=transcript,
                                interactive=True,
                                feedback=current_feedback if current_feedback is not None else feedback,
                                mode=mode_name,
                                prompt_label="cmd>" if mode_name == "command" else "chat>",
                                model=model,
                                adapter=adapter,
                                real_local=real_local,
                                refresh_seconds=refresh_seconds,
                                input_active=True,
                                input_text=input_text,
                                input_cursor=input_cursor,
                                shortcut_hint=_console_shortcut_hint(mode_name, payload),
                                ops_refresh_state="syncing",
                                ops_age_seconds=max(0.0, now - last_sidebar_refresh_at),
                            )
                        )
                        live.refresh()
                        payload = _console_snapshot_payload(handler, workspace=workspace)
                        last_sidebar_refresh_at = now
                        ops_refresh_state = "live"
                    else:
                        ops_refresh_state = "cached"
                    ops_age_seconds = max(0.0, now - last_sidebar_refresh_at)
                    prompt_label = "cmd>" if mode_name == "command" else "chat>"
                    live.update(
                        build_console_renderable(
                            payload,
                            workspace=workspace,
                            session_messages=transcript,
                            interactive=True,
                            feedback=current_feedback if current_feedback is not None else feedback,
                            mode=mode_name,
                            prompt_label=prompt_label,
                            model=model,
                            adapter=adapter,
                            real_local=real_local,
                            refresh_seconds=refresh_seconds,
                            input_active=True,
                            input_text=input_text,
                            input_cursor=input_cursor,
                            shortcut_hint=_console_shortcut_hint(mode_name, payload),
                            ops_refresh_state=ops_refresh_state,
                            ops_age_seconds=ops_age_seconds,
                        )
                    )
                    live.refresh()

                while True:
                    prompt_label = "cmd>" if mode_name == "command" else "chat>"
                    _refresh_console("", input_cursor=0, current_feedback=feedback, force_sidebar=True)
                    try:
                        user_text = _console_read_input(
                            prompt_label,
                            refresh_seconds=refresh_seconds,
                            refresh_callback=lambda current, cursor: _refresh_console(
                                current,
                                input_cursor=cursor,
                                current_feedback=feedback,
                            ),
                            history=input_history,
                        )
                    except (typer.Abort, EOFError, KeyboardInterrupt):
                        typer.echo("Exiting PFE Console.")
                        break

                    console_ui.print("")
                    message = str(user_text or "").strip()
                    if not message:
                        feedback = "empty input"
                        continue
                    input_history.append(message)
                    if len(input_history) > 50:
                        del input_history[:-50]
                    command_input = message
                    if mode_name == "command" and not message.startswith("/"):
                        command_input = f"/{message}"
                    regenerate_mode = False
                    if command_input.startswith("/"):
                        feedback = f"running /{command_input[1:].split()[0]}"
                        _refresh_console(
                            message,
                            input_cursor=len(message),
                            current_feedback=feedback,
                            force_sidebar=False,
                        )
                        command_output, action, updates = _console_command_output(
                            command_input[1:],
                            payload=payload,
                            workspace=workspace,
                            service=service,
                            current_workspace=workspace,
                            mode=mode_name,
                            model=model,
                            adapter=adapter,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            real_local=real_local,
                            refresh_seconds=refresh_seconds,
                            last_interaction=last_interaction,
                        )
                        if action == "quit":
                            typer.echo("Exiting PFE Console.")
                            break
                        elif action == "clear":
                            transcript.clear()
                            chat_messages.clear()
                            feedback = "console transcript cleared"
                            continue
                        elif action == "mode:chat":
                            mode_name = "chat"
                            feedback = "switched to chat mode"
                            continue
                        elif action == "mode:command":
                            mode_name = "command"
                            feedback = "switched to command mode"
                            continue
                        elif action == "fix" or (updates and "edited_text" in updates):
                            edited_text = (updates or {}).get("edited_text", "")
                            if edited_text:
                                if chat_messages and chat_messages[-1]["role"] == "assistant":
                                    chat_messages[-1]["content"] = edited_text
                                if transcript and transcript[-1].get("role") == "assistant":
                                    transcript[-1]["content"] = edited_text
                                if last_interaction is not None:
                                    last_interaction["assistant_message"] = edited_text
                            if command_output:
                                _append_console_line(transcript, role="system", content=command_output)
                            feedback = "response edited"
                            continue
                        elif action == "again" or (updates and updates.get("regenerate")):
                            if chat_messages and chat_messages[-1]["role"] == "assistant":
                                chat_messages.pop()
                            if last_interaction is not None:
                                message = last_interaction.get("user_message", "")
                                regenerate_mode = True
                            if command_output:
                                _append_console_line(transcript, role="system", content=command_output)
                            if not regenerate_mode:
                                feedback = "nothing to regenerate"
                                continue
                        elif updates:
                            if "workspace" in updates:
                                workspace = str(updates["workspace"])
                            if "model" in updates:
                                model = str(updates["model"])
                            if "adapter" in updates:
                                adapter = str(updates["adapter"])
                            if "temperature" in updates:
                                temperature = float(updates["temperature"])
                            if "max_tokens" in updates:
                                max_tokens = updates["max_tokens"]
                            if "real_local" in updates:
                                real_local = bool(updates["real_local"])
                                if real_local:
                                    os.environ["PFE_ENABLE_REAL_LOCAL_INFERENCE"] = "1"
                                elif previous_real_local is None:
                                    os.environ.pop("PFE_ENABLE_REAL_LOCAL_INFERENCE", None)
                                else:
                                    os.environ["PFE_ENABLE_REAL_LOCAL_INFERENCE"] = previous_real_local
                            if "refresh_seconds" in updates:
                                refresh_seconds = float(updates["refresh_seconds"])
                            payload = _console_snapshot_payload(handler, workspace=workspace)
                            last_sidebar_refresh_at = time.monotonic()
                            ops_refresh_state = "live"
                            if command_output:
                                _append_console_line(transcript, role="system", content=command_output)
                            feedback = f"handled /{action}"
                            continue

                        if not regenerate_mode:
                            if command_output:
                                _append_console_line(transcript, role="system", content=command_output)
                            feedback = f"handled /{action}"
                            continue

                    if chat_handler is None:
                        _append_console_line(transcript, role="user", content=message)
                        _append_console_line(
                            transcript,
                            role="assistant",
                            content="Interactive chat is unavailable because no chat_completion handler is registered.",
                        )
                        feedback = "chat handler unavailable"
                        continue

                    if not regenerate_mode:
                        # Submit implicit accept for previous interaction before new message
                        if mode_name == "chat" and last_interaction is not None:
                            _console_submit_feedback(
                                workspace=workspace,
                                session_id=last_interaction.get("session_id", session_id),
                                request_id=last_interaction.get("request_id", ""),
                                user_message=last_interaction.get("user_message", ""),
                                assistant_message=last_interaction.get("assistant_message", ""),
                                response_time_seconds=last_interaction.get("response_time_seconds", 0.0),
                                adapter_version=last_interaction.get("adapter_version", adapter),
                                action="continue",
                            )
                        _append_console_line(transcript, role="user", content=message)
                        chat_messages.append({"role": "user", "content": message})
                    else:
                        # Regenerate mode: ensure user message is in chat_messages after popping assistant
                        if not chat_messages or chat_messages[-1]["role"] != "user":
                            chat_messages.append({"role": "user", "content": message})
                    effective_max_tokens = max_tokens or 96
                    feedback = "assistant generating..."
                    started_at = time.monotonic()
                    response_holder: dict[str, Any] = {}

                    def _run_chat() -> None:
                        try:
                            response_holder["result"] = chat_handler(
                                messages=chat_messages,
                                model=model,
                                adapter_version=adapter,
                                temperature=temperature,
                                max_tokens=effective_max_tokens,
                                metadata={"enable_real_local": True} if real_local else {},
                                session_id=session_id,
                                workspace=workspace,
                            )
                        except Exception as exc:  # pragma: no cover - surfaced in main thread
                            response_holder["error"] = exc

                    worker = threading.Thread(target=_run_chat, daemon=True)
                    worker.start()
                    wait_seconds = min(max(refresh_seconds / 2.0, 0.2), 0.5)

                    while worker.is_alive():
                        now = time.monotonic()
                        should_refresh_sidebar = now - last_sidebar_refresh_at >= refresh_seconds
                        if should_refresh_sidebar:
                            live.update(
                                build_console_renderable(
                                    payload,
                                    workspace=workspace,
                                    session_messages=transcript,
                                    interactive=True,
                                    feedback=(
                                        f"assistant generating... {time.monotonic() - started_at:.1f}s | "
                                        f"mode={mode_name} | model={model} | adapter={adapter}"
                                    ),
                                    mode=mode_name,
                                    prompt_label=prompt_label,
                                    model=model,
                                    adapter=adapter,
                                    real_local=real_local,
                                    refresh_seconds=refresh_seconds,
                                    input_active=False,
                                    input_text=message,
                                    input_cursor=len(message),
                                    shortcut_hint="wait,^C",
                                    ops_refresh_state="syncing",
                                    ops_age_seconds=max(0.0, now - last_sidebar_refresh_at),
                                )
                            )
                            live.refresh()
                            refreshed_payload = _console_snapshot_payload(handler, workspace=workspace)
                            payload = refreshed_payload
                            last_sidebar_refresh_at = now
                            ops_refresh_state = "live"
                        else:
                            refreshed_payload = payload
                            ops_refresh_state = "cached"
                        ops_age_seconds = max(0.0, now - last_sidebar_refresh_at)
                        elapsed = time.monotonic() - started_at
                        refresh_feedback = (
                            f"assistant generating... {elapsed:.1f}s | "
                            f"mode={mode_name} | model={model} | adapter={adapter}"
                        )
                        live.update(
                            build_console_renderable(
                                refreshed_payload,
                                workspace=workspace,
                                session_messages=transcript,
                                interactive=True,
                                feedback=refresh_feedback,
                                mode=mode_name,
                                prompt_label=prompt_label,
                                model=model,
                                adapter=adapter,
                                real_local=real_local,
                                refresh_seconds=refresh_seconds,
                                input_active=False,
                                input_text=message,
                                input_cursor=len(message),
                                shortcut_hint="wait,^C",
                                ops_refresh_state=ops_refresh_state,
                                ops_age_seconds=ops_age_seconds,
                            )
                        )
                        live.refresh()
                        worker.join(timeout=wait_seconds)

                    if "error" in response_holder:
                        raise response_holder["error"]
                    chat_response = response_holder.get("result")
                    assistant_text = _console_chat_text(chat_response) or "(empty response)"
                    chat_messages.append({"role": "assistant", "content": assistant_text})
                    _append_console_line(transcript, role="assistant", content=assistant_text)
                    latency_seconds = time.monotonic() - started_at
                    last_interaction = {
                        "session_id": session_id,
                        "request_id": f"req-{uuid4().hex[:12]}",
                        "user_message": message,
                        "assistant_message": assistant_text,
                        "response_time_seconds": latency_seconds,
                        "adapter_version": adapter,
                    }
                    feedback = f"assistant replied ({len(assistant_text)} chars in {latency_seconds:.1f}s)"
                    live.update(
                        build_console_renderable(
                            _console_snapshot_payload(handler, workspace=workspace),
                            workspace=workspace,
                            session_messages=transcript,
                            interactive=True,
                            feedback=feedback,
                            mode=mode_name,
                            prompt_label=prompt_label,
                            model=model,
                            adapter=adapter,
                            real_local=real_local,
                            refresh_seconds=refresh_seconds,
                            input_active=False,
                            input_cursor=0,
                            shortcut_hint=_console_shortcut_hint(mode_name, payload),
                            ops_refresh_state="live",
                            ops_age_seconds=0.0,
                        )
                    )
                    last_sidebar_refresh_at = time.monotonic()
                    live.refresh()
        finally:
            if previous_real_local is None:
                os.environ.pop("PFE_ENABLE_REAL_LOCAL_INFERENCE", None)
            else:
                os.environ["PFE_ENABLE_REAL_LOCAL_INFERENCE"] = previous_real_local
        return

    run_cycles = cycles if watch else 1
    for index in range(run_cycles):
        try:
            result = handler(workspace=workspace)
        except typer.Exit:
            raise
        except Exception as exc:
            friendly = _friendly_exception_message(exc)
            if friendly is not None:
                typer.secho(friendly, err=True, fg=typer.colors.RED)
                raise typer.Exit(code=1)
            raise
        mapping = _coerce_mapping(result)
        payload = mapping if mapping is not None else {"status_result": str(result)}
        render_console_snapshot(payload, workspace=workspace, clear=index > 0)
        if watch and index < run_cycles - 1:
            time.sleep(refresh_seconds)


def _teacher_distillation_status_block() -> str:
    """Render a small text block showing teacher distillation config status."""
    try:
        from pfe_core.config import PFEConfig
        cfg = PFEConfig.load()
        td = cfg.trainer.teacher_distillation
        enabled = "enabled" if td.enabled else "disabled"
        model = td.teacher_model or "(not set)"
        ratio = td.max_teacher_ratio
        threshold = td.similarity_threshold
        return (
            f"Teacher Distillation: {enabled}\n"
            f"  Model: {model}\n"
            f"  Max ratio: {ratio}\n"
            f"  Similarity threshold: {threshold}"
        )
    except Exception:
        return "Teacher Distillation: unavailable"


@app.command("status")
def status(
    workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON instead of formatted text."),
) -> None:
    """Show engine, adapter, and inference status."""

    service = _load_service("pfe_core.pipeline", "pfe_core.status", "pfe_server.app", "pfe_core.services.pipeline")
    if service is None:
        _run_placeholder("status")
        typer.echo("")
        typer.echo(_teacher_distillation_status_block())
        return

    handler = _resolve_handler(service, "status", "get_status")
    if handler is None:
        _run_placeholder("status")
        typer.echo("")
        typer.echo(_teacher_distillation_status_block())
        return

    if json_output:
        _run_handler_json("status", handler, workspace=workspace)
        return

    _run_handler("status", handler, formatter=lambda result: _format_status(result, workspace=workspace), workspace=workspace)
    typer.echo("")
    typer.echo(_teacher_distillation_status_block())


@app.command("doctor")
def doctor(
    workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
    base_model: Optional[str] = typer.Option(None, "--base-model", help="Override base model path or model id for local model checks."),
) -> None:
    """Show strict_local readiness signals for trainer, model, export, and adapter state."""

    typer.echo(_format_doctor(workspace=workspace, base_model=base_model))


@app.command("dashboard")
def dashboard(
    port: int = typer.Option(8921, "--port", min=1, max=65535, help="Server port to connect to."),
    host: str = typer.Option("127.0.0.1", "--host", help="Server host."),
    open_browser: bool = typer.Option(True, "--open/--no-open", help="Open dashboard in browser."),
) -> None:
    """Launch the PFE observability dashboard in a web browser."""
    import webbrowser
    from urllib.parse import urljoin

    dashboard_url = f"http://{host}:{port}/dashboard"

    typer.echo(f"PFE Observability Dashboard")
    typer.echo(f"URL: {dashboard_url}")

    if open_browser:
        typer.echo("Opening browser...")
        webbrowser.open(dashboard_url)
    else:
        typer.echo("Use --open to launch browser automatically.")


@app.command("boot")
def boot() -> None:
    """Display PFE boot sequence with ZC logo.

    Shows the pixel art ZC logo, system initialization sequence,
    and available commands matrix.
    """
    import time
    from .pixel_logo import render_boot_banner, render_loading_sequence, render_commands_matrix

    # Matrix style boot sequence - default
    typer.echo(render_boot_banner(version="2.0.0"))

    steps = [
        "Loading adapter store...",
        "Initializing trainer service...",
        "Mounting signal collector...",
        "Establishing daemon connection...",
        "Calibrating neural weights...",
    ]

    for i, step in enumerate(steps, 1):
        typer.echo(f"{formatters_matrix.MatrixColors.GREEN}  {render_loading_sequence(i, len(steps))}{formatters_matrix.MatrixColors.RESET} {step}")
        time.sleep(0.15)

    typer.echo("")
    typer.echo(f"{formatters_matrix.MatrixColors.GREEN_BRIGHT}{formatters_matrix.MatrixColors.BOLD}  >> ALL SYSTEMS OPERATIONAL <<{formatters_matrix.MatrixColors.RESET}")
    typer.echo("")

    # Show commands matrix
    typer.echo(render_commands_matrix())


# Profile commands
profile_app = typer.Typer(help="Manage rule-based profile analysis snapshots. Runtime prompt injection still uses user_memory as the primary user-modeling path.")
app.add_typer(profile_app, name="profile")


@profile_app.command("show")
def profile_show(
    user_id: str = typer.Option("default", "--user-id", help="User identifier."),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON output."),
    workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
) -> None:
    """Show the current profile analysis snapshot for a user."""
    from pfe_core.profile_extractor import get_user_profile_store

    try:
        store = get_user_profile_store()
        profile = store.get_profile(user_id)

        if json_output:
            typer.echo(json.dumps(profile.to_dict(), ensure_ascii=False, indent=2))
            return

        lines = [f"User Profile: {user_id}", ""]

        # Style preferences
        if profile.style_preferences:
            lines.append("Style Preferences:")
            for key, pref in sorted(
                profile.style_preferences.items(),
                key=lambda x: x[1].score * x[1].confidence,
                reverse=True
            )[:5]:
                lines.append(f"  - {key}: {pref.score:.2f} (confidence: {pref.confidence:.2f}, freq: {pref.frequency})")
            lines.append("")

        # Domain preferences
        if profile.domain_preferences:
            lines.append("Domain Preferences:")
            for key, pref in sorted(
                profile.domain_preferences.items(),
                key=lambda x: x[1].score * x[1].confidence,
                reverse=True
            )[:5]:
                lines.append(f"  - {key}: {pref.score:.2f} (confidence: {pref.confidence:.2f}, freq: {pref.frequency})")
            lines.append("")

        # Interaction patterns
        if profile.interaction_patterns:
            lines.append("Interaction Patterns:")
            for key, pref in sorted(
                profile.interaction_patterns.items(),
                key=lambda x: x[1].score * x[1].confidence,
                reverse=True
            )[:5]:
                lines.append(f"  - {key}: {pref.score:.2f} (confidence: {pref.confidence:.2f}, freq: {pref.frequency})")
            lines.append("")

        # Summary
        if profile.profile_summary:
            lines.append(f"Profile Summary: {profile.profile_summary}")
        if profile.dominant_style:
            lines.append(f"Dominant Style: {profile.dominant_style}")
        if profile.dominant_domains:
            lines.append(f"Dominant Domains: {', '.join(profile.dominant_domains)}")

        lines.append(f"\nAnalysis Count: {profile.analysis_count}")
        if profile.last_analysis_at:
            lines.append(f"Last Analysis: {profile.last_analysis_at.isoformat()}")

        typer.echo("\n".join(lines))

    except Exception as exc:
        typer.secho(f"Error loading profile: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1)


@profile_app.command("analyze")
def profile_analyze(
    user_id: str = typer.Option("default", "--user-id", help="User identifier."),
    incremental: bool = typer.Option(True, "--incremental/--full", help="Incremental or full analysis."),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON output."),
    workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
) -> None:
    """Manually trigger rule-based profile analysis from stored signals."""
    from pfe_core.profile_extractor import extract_profile_for_user

    try:
        result = extract_profile_for_user(
            user_id=user_id,
            signals=None,  # Load from storage
            incremental=incremental,
        )

        if json_output:
            typer.echo(json.dumps(result, ensure_ascii=False, indent=2))
            return

        typer.echo(f"Profile analysis completed for user: {user_id}")
        typer.echo(f"Signals analyzed: {result.get('signals_analyzed', 0)}")

        if result.get('domains_found'):
            typer.echo(f"Domains found: {', '.join(result['domains_found'])}")
        if result.get('styles_found'):
            typer.echo(f"Styles found: {', '.join(result['styles_found'])}")
        if result.get('patterns_found'):
            typer.echo(f"Patterns found: {', '.join(result['patterns_found'])}")

        if result.get('profile_summary'):
            typer.echo(f"\nProfile Summary: {result['profile_summary']}")

    except Exception as exc:
        typer.secho(f"Error analyzing profile: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1)


@profile_app.command("extract")
def profile_extract(
    user_id: str = typer.Option("default", "--user-id", help="User identifier."),
    use_llm: bool = typer.Option(False, "--use-llm", help="Use LLM for structured extraction."),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON output."),
    workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
) -> None:
    """Extract or re-extract profile summary for a user."""
    from pfe_core.profile_extractor import ProfileExtractor

    try:
        extractor = ProfileExtractor(user_id=user_id)
        summary = extractor.generate_profile_summary(use_llm=use_llm)
        profile = extractor.profile

        if json_output:
            typer.echo(json.dumps(profile.to_dict(), ensure_ascii=False, indent=2))
            return

        typer.echo(f"Profile extraction completed for user: {user_id}")
        typer.echo(f"Extracted by: {profile.extracted_by}")
        if profile.llm_extracted_at:
            typer.echo(f"LLM extracted at: {profile.llm_extracted_at.isoformat()}")
        typer.echo(f"Summary: {summary}")
    except Exception as exc:
        typer.secho(f"Error extracting profile: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1)


@profile_app.command("drift")
def profile_drift(
    user_id: str = typer.Option("default", "--user-id", help="User identifier."),
    threshold: float = typer.Option(0.3, "--threshold", help="Drift detection threshold."),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON output."),
    workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
) -> None:
    """Show preference drift detection report for a user."""
    from pfe_core.profile_extractor import get_user_profile_store

    try:
        store = get_user_profile_store()
        profile = store.get_profile(user_id)
        alerts = profile.detect_preference_drift(threshold=threshold)

        if json_output:
            typer.echo(json.dumps({"alerts": alerts}, ensure_ascii=False, indent=2))
            return

        typer.echo(f"Drift Report for user: {user_id}")
        if not alerts:
            typer.echo("No significant preference drift detected.")
            return

        typer.echo(f"Detected {len(alerts)} drift alert(s) (threshold={threshold}):")
        for alert in alerts:
            direction_icon = "+" if alert["drift_direction"] == "increase" else "-"
            typer.echo(
                f"  - {alert['preference_key']}: {alert['old_avg']:.2f} -> {alert['new_avg']:.2f} "
                f"({direction_icon}, severity={alert['severity']})"
            )
    except Exception as exc:
        typer.secho(f"Error detecting drift: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1)


@profile_app.command("export")
def profile_export(
    user_id: str = typer.Option("default", "--user-id", help="User identifier."),
    output: str = typer.Option(..., "--output", help="Output file path."),
    workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
) -> None:
    """Export user profile to a file."""
    from pfe_core.profile_extractor import get_user_profile_store

    try:
        store = get_user_profile_store()
        profile_data = store.export_profile(user_id)

        if profile_data is None:
            typer.secho(f"Profile not found for user: {user_id}", err=True, fg=typer.colors.RED)
            raise typer.Exit(code=1)

        output_path = Path(output)
        output_path.write_text(
            json.dumps(profile_data, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        typer.echo(f"Profile exported to: {output}")

    except Exception as exc:
        typer.secho(f"Error exporting profile: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1)


@profile_app.command("import")
def profile_import(
    user_id: str = typer.Option("default", "--user-id", help="User identifier."),
    input_file: str = typer.Option(..., "--input", help="Input file path."),
    workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
) -> None:
    """Import user profile from a file."""
    from pfe_core.profile_extractor import get_user_profile_store

    try:
        input_path = Path(input_file)
        if not input_path.exists():
            typer.secho(f"Input file not found: {input_file}", err=True, fg=typer.colors.RED)
            raise typer.Exit(code=1)

        profile_data = json.loads(input_path.read_text(encoding="utf-8"))

        store = get_user_profile_store()
        store.import_profile(user_id, profile_data)

        typer.echo(f"Profile imported for user: {user_id}")

    except Exception as exc:
        typer.secho(f"Error importing profile: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1)


@profile_app.command("list")
def profile_list(
    json_output: bool = typer.Option(False, "--json", help="Emit JSON output."),
    workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
) -> None:
    """List all user profiles."""
    from pfe_core.profile_extractor import get_user_profile_store

    try:
        store = get_user_profile_store()
        profiles = store.list_profiles()

        if json_output:
            typer.echo(json.dumps({"profiles": profiles}, ensure_ascii=False, indent=2))
            return

        if not profiles:
            typer.echo("No profiles found.")
            return

        typer.echo("User Profiles:")
        for pid in profiles:
            typer.echo(f"  - {pid}")

    except Exception as exc:
        typer.secho(f"Error listing profiles: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1)


@profile_app.command("delete")
def profile_delete(
    user_id: str = typer.Option("default", "--user-id", help="User identifier."),
    workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
    yes: bool = typer.Option(False, "--yes", help="Skip confirmation."),
) -> None:
    """Delete a user profile."""
    from pfe_core.profile_extractor import get_user_profile_store

    try:
        store = get_user_profile_store()

        if not yes:
            confirmed = typer.confirm(f"Are you sure you want to delete profile '{user_id}'?")
            if not confirmed:
                typer.echo("Deletion cancelled.")
                return

        success = store.delete_profile(user_id)

        if success:
            typer.echo(f"Profile deleted: {user_id}")
        else:
            typer.secho(f"Profile not found: {user_id}", err=True, fg=typer.colors.YELLOW)

    except Exception as exc:
        typer.secho(f"Error deleting profile: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1)


# Scenario routing commands
scenario_app = typer.Typer(help="Manage scenario configurations for the current rule-based router.")
app.add_typer(scenario_app, name="scenario")
route_app = typer.Typer(help="Test and debug the current keyword/rule-based scenario router.")
app.add_typer(route_app, name="route")

# Data management commands (PII check, etc.)
data_app = typer.Typer(help="Data management and privacy compliance.")
app.add_typer(data_app, name="data")


@scenario_app.command("list")
def scenario_list(
    json_output: bool = typer.Option(False, "--json", help="Emit JSON output."),
) -> None:
    """List all available scenarios."""
    from pfe_core.router import create_router
    from pfe_core.config import PFEConfig

    config = PFEConfig.load()
    router = create_router(config=config)
    scenarios = router.list_scenarios()

    if json_output:
        typer.echo(json.dumps(scenarios, ensure_ascii=False, indent=2))
        return

    if not scenarios:
        typer.echo("No scenarios configured.")
        return

    lines = ["Available scenarios:", ""]
    for s in scenarios:
        lines.append(f"  {s['scenario_id']}: {s['name']}")
        lines.append(f"    Description: {s['description']}")
        lines.append(f"    Adapter: {s['adapter_version']}")
        lines.append(f"    Keywords: {s['keyword_count']} | Examples: {s['example_count']} | Priority: {s['priority']}")
        lines.append("")
    typer.echo("\n".join(lines))


@scenario_app.command("create")
def scenario_create(
    name: str = typer.Argument(..., help="Scenario ID (e.g., 'coding', 'writing')."),
    adapter: str = typer.Option("latest", "--adapter", help="Adapter version to bind to this scenario."),
    description: str = typer.Option("", "--description", help="Scenario description."),
    keywords: str = typer.Option("", "--keywords", help="Comma-separated trigger keywords."),
    priority: int = typer.Option(0, "--priority", help="Scenario priority (higher = preferred)."),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON output."),
) -> None:
    """Create a new custom scenario."""
    from pfe_core.router import create_router
    from pfe_core.scenarios import create_custom_scenario
    from pfe_core.config import PFEConfig

    config = PFEConfig.load()
    router = create_router(config=config)

    keyword_list = [k.strip() for k in keywords.split(",") if k.strip()]
    scenario = create_custom_scenario(
        scenario_id=name,
        name=name.replace("_", " ").title(),
        description=description or f"Custom scenario: {name}",
        adapter_version=adapter,
        trigger_keywords=keyword_list,
        priority=priority,
    )
    router.add_scenario(scenario)

    result = {
        "created": True,
        "scenario": scenario.to_dict(),
    }

    if json_output:
        typer.echo(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        typer.echo(f"Created scenario '{name}' with adapter '{adapter}'.")
        if keyword_list:
            typer.echo(f"  Keywords: {', '.join(keyword_list)}")


@scenario_app.command("bind")
def scenario_bind(
    scenario: str = typer.Argument(..., help="Scenario ID to bind."),
    adapter: str = typer.Option(..., "--adapter", help="Adapter version to bind."),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON output."),
) -> None:
    """Bind a scenario to a specific adapter version."""
    from pfe_core.router import create_router
    from pfe_core.config import PFEConfig

    config = PFEConfig.load()
    router = create_router(config=config)

    success = router.bind_scenario_to_adapter(scenario, adapter)

    result = {
        "bound": success,
        "scenario_id": scenario,
        "adapter_version": adapter,
    }

    if json_output:
        typer.echo(json.dumps(result, ensure_ascii=False, indent=2))
    elif success:
        typer.echo(f"Bound scenario '{scenario}' to adapter '{adapter}'.")
    else:
        typer.echo(f"Failed to bind scenario '{scenario}'. Scenario not found.", err=True)
        raise typer.Exit(code=1)


@route_app.command("test")
def route_test(
    text: str = typer.Argument(..., help="Input text to test routing."),
    strategy: str = typer.Option("keyword", "--strategy", help="Routing strategy: keyword or hybrid."),
    show_scores: bool = typer.Option(False, "--show-scores", help="Show detailed scores for all scenarios."),
    json_output: bool = typer.Option(False, "--json", help="Emit detailed JSON output."),
) -> None:
    """Test scenario routing for a given input text. Supports keyword and hybrid strategies."""
    from pfe_core.router import create_router
    from pfe_core.config import PFEConfig

    config = PFEConfig.load()
    # Temporarily override strategy for this test
    original_strategy = config.router.strategy
    config.router.strategy = strategy  # type: ignore[misc]
    router = create_router(config=config)
    result = router.test_route(text)
    config.router.strategy = original_strategy  # type: ignore[misc]

    if json_output:
        typer.echo(json.dumps(result, ensure_ascii=False, indent=2))
        return

    # Format human-readable output
    primary = result["primary_route"]
    classification = result["classification"]

    typer.echo(f"Input: {text}")
    typer.echo(f"Strategy: {result.get('strategy', strategy)}")
    typer.echo(f"Primary Intent: {classification['primary_intent']} (confidence: {classification['confidence']:.2f})")
    typer.echo(f"Selected Scenario: {primary['scenario_id']}")
    typer.echo(f"Adapter Version: {primary['adapter_version']}")
    typer.echo(f"Routing Confidence: {primary['confidence']:.2f}")
    if primary['fallback']:
        typer.echo("Note: Using fallback routing (low confidence)")
    typer.echo(f"Reasoning: {primary['reasoning']}")

    if show_scores and result["all_routes"]:
        typer.echo("\nAll scenario scores:")
        for route in result["all_routes"]:
            typer.echo(f"  {route['scenario_id']}: {route['score']:.3f}")
    elif result["all_routes"]:
        typer.echo("\nTop scenario scores:")
        for route in result["all_routes"][:5]:
            typer.echo(f"  {route['scenario_id']}: {route['score']:.3f}")


@data_app.command("pii-check")
def data_pii_check(
    fix: bool = typer.Option(False, "--fix", help="Auto-anonymize detected PII."),
    sample_file: Optional[str] = typer.Option(None, "--file", help="Specific sample file to check."),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON output."),
) -> None:
    """Check training samples for PII (Personally Identifiable Information)."""
    from pfe_core.pii_detector import PIIDetector
    from pfe_core.anonymizer import Anonymizer, AnonymizationConfig
    from pfe_core.config import PFEConfig
    from pathlib import Path
    import json as json_module

    config = PFEConfig.load()

    # Initialize detector
    detector = PIIDetector(sensitivity="medium")
    anonymizer = None
    if fix:
        anon_config = AnonymizationConfig(strategy="mask")
        anonymizer = Anonymizer(anon_config)

    # Find samples to check
    samples_to_check = []
    if sample_file:
        file_path = Path(sample_file)
        if file_path.exists():
            samples_to_check.append(file_path)
    else:
        # Find all sample files in workspace
        samples_dir = Path(config.workspace) / "training_samples"
        if samples_dir.exists():
            samples_to_check = list(samples_dir.glob("**/*.jsonl"))

    if not samples_to_check:
        typer.echo("No sample files found to check.", err=True)
        raise typer.Exit(code=1)

    results = []
    total_files = 0
    total_samples = 0
    samples_with_pii = 0
    total_findings = 0

    for file_path in samples_to_check:
        total_files += 1
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    sample = json_module.loads(line.strip())
                    total_samples += 1

                    # Check all text fields
                    fields_to_check = ["instruction", "input", "output", "conversation"]
                    file_findings = []

                    for field in fields_to_check:
                        if field in sample and isinstance(sample[field], str):
                            detection = detector.detect(sample[field])
                            if detection.has_pii:
                                file_findings.append({
                                    "field": field,
                                    "types": [t.value for t in detection.pii_types_found],
                                    "count": len(detection.findings),
                                })
                                total_findings += len(detection.findings)

                                if fix and anonymizer:
                                    sample[field] = anonymizer.anonymize(sample[field], detection)

                    if file_findings:
                        samples_with_pii += 1
                        results.append({
                            "file": str(file_path),
                            "sample_index": total_samples - 1,
                            "findings": file_findings,
                        })
                except json_module.JSONDecodeError:
                    continue

    summary = {
        "total_files": total_files,
        "total_samples": total_samples,
        "samples_with_pii": samples_with_pii,
        "total_findings": total_findings,
        "files_checked": [str(p) for p in samples_to_check],
    }

    if json_output:
        typer.echo(json_module.dumps({
            "summary": summary,
            "results": results,
        }, ensure_ascii=False, indent=2))
    else:
        typer.echo(f"PII Check Results:")
        typer.echo(f"  Files checked: {total_files}")
        typer.echo(f"  Total samples: {total_samples}")
        typer.echo(f"  Samples with PII: {samples_with_pii}")
        typer.echo(f"  Total PII findings: {total_findings}")

        if results and not fix:
            typer.echo("\nRun with --fix to auto-anonymize detected PII.")
        elif fix:
            typer.echo("\nAnonymization applied to detected PII.")


def main() -> None:
    """Console script entrypoint."""

    app()


if __name__ == "__main__":
    main()
