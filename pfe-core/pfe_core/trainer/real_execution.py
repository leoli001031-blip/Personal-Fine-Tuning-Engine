"""Safety boundary for real trainer execution."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

from .executors import materialize_training_job_bundle, run_materialized_training_job_bundle

REAL_TRAINING_ENV = "PFE_REAL_TRAINING"
TRAINING_SUBPROCESS_ENV = "PFE_TRAINING_SUBPROCESS"
TRAINING_SUBPROCESS_MARKER = "_pfe_training_subprocess"

REAL_TRAINING_BACKENDS = frozenset({"mlx", "peft", "unsloth", "dpo"})
SUBPROCESS_ISOLATED_BACKENDS = frozenset({"mlx", "peft", "unsloth", "dpo"})


def is_real_training_allowed(*, dry_run: bool, environ: Mapping[str, str] | None = None) -> bool:
    """Return whether the current process may launch real training."""
    if dry_run:
        return True
    env = environ or os.environ
    return str(env.get(REAL_TRAINING_ENV, "")).lower() in {"1", "true", "yes"}


def real_training_disabled_result(*, backend: str, dry_run: bool) -> dict[str, Any]:
    return {
        "status": "blocked",
        "reason": "Real training execution disabled. Set PFE_REAL_TRAINING=1 to enable.",
        "backend": backend,
        "dry_run": dry_run,
    }


def is_training_subprocess(
    job_spec: Mapping[str, Any],
    *,
    environ: Mapping[str, str] | None = None,
) -> bool:
    env = environ or os.environ
    return bool(job_spec.get(TRAINING_SUBPROCESS_MARKER)) or str(env.get(TRAINING_SUBPROCESS_ENV, "")) == "1"


def mark_training_subprocess(job_spec: Mapping[str, Any]) -> dict[str, Any]:
    marked = dict(job_spec)
    marked[TRAINING_SUBPROCESS_MARKER] = True
    return marked


def should_isolate_backend(
    *,
    backend: str,
    dry_run: bool,
    job_spec: Mapping[str, Any],
    environ: Mapping[str, str] | None = None,
) -> bool:
    return (
        backend in SUBPROCESS_ISOLATED_BACKENDS
        and not dry_run
        and not is_training_subprocess(job_spec, environ=environ)
    )


def run_training_preflight(job_spec: Mapping[str, Any], *, backend: str) -> dict[str, Any]:
    """Run parent-process checks that are safe before launching the trainer subprocess."""
    try:
        from .preflight import TrainingPreflight

        preflight_job = {**dict(job_spec), "backend": backend}
        result = TrainingPreflight(preflight_job).check()
        return {**result, "ready": bool(result.get("ready") or result.get("status") == "ok")}
    except Exception as exc:
        return {
            "ready": False,
            "status": "blocked",
            "stage": "preflight",
            "backend": backend,
            "reasons": [f"preflight_error:{type(exc).__name__}"],
            "error": str(exc),
        }


def preflight_blocked_result(*, backend: str, preflight: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "backend": backend,
        "dry_run": False,
        "status": "blocked",
        "reason": "preflight failed",
        "preflight": dict(preflight),
    }


def _materialization_output_dir(job_spec: Mapping[str, Any]) -> Path:
    raw_output_dir = job_spec.get("output_dir") or job_spec.get("workspace") or "."
    return Path(str(raw_output_dir)).expanduser().resolve()


def run_backend_in_subprocess(
    job_spec: Mapping[str, Any],
    *,
    backend: str,
    dry_run: bool,
) -> dict[str, Any]:
    """Materialize and run a real trainer backend in an isolated Python process."""
    output_dir = _materialization_output_dir(job_spec)
    output_dir.mkdir(parents=True, exist_ok=True)

    execution_plan = {
        "job_spec": mark_training_subprocess(job_spec),
        "backend": backend,
        "execution_backend": backend,
        "execution_executor": backend,
        "ready": True,
    }

    materialized = materialize_training_job_bundle(
        execution_plan=execution_plan,
        output_dir=output_dir,
    )

    if not materialized.ready:
        return {
            "backend": backend,
            "dry_run": dry_run,
            "status": "blocked",
            "reason": "materialization not ready",
            "materialization": materialized.to_dict(),
        }

    run_result = run_materialized_training_job_bundle(
        materialized,
        force_dry_run=dry_run,
        timeout_seconds=job_spec.get("timeout_seconds"),
    )

    return {
        "backend": backend,
        "dry_run": dry_run,
        "status": "completed" if run_result.success else "failed",
        "failure_category": run_result.failure_category,
        "returncode": run_result.returncode,
        "stdout_log": run_result.stdout_log,
        "stderr_log": run_result.stderr_log,
        "diagnostics": run_result.diagnostics,
        "runner_result": run_result.runner_result,
        "materialization": run_result.materialization,
    }
