#!/usr/bin/env python3
"""Validate CUDA real-training backends through PFE's isolated runtime path."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
for package_dir in ("pfe-core", "pfe-cli", "pfe-server"):
    package_path = str(REPO_ROOT / package_dir)
    if package_path not in sys.path:
        sys.path.insert(0, package_path)


DEFAULT_PEFT_MODEL = "sshleifer/tiny-gpt2"
DEFAULT_UNSLOTH_MODEL = "unsloth/tinyllama-bnb-4bit"


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _backend_list(value: str) -> list[str]:
    if value == "all":
        return ["peft", "unsloth"]
    return [value]


def _default_model(backend: str, *, peft_model: str | None, unsloth_model: str | None) -> str:
    if backend == "unsloth":
        return unsloth_model or DEFAULT_UNSLOTH_MODEL
    return peft_model or DEFAULT_PEFT_MODEL


def build_job_spec(
    *,
    backend: str,
    base_model: str,
    output_dir: Path,
    epochs: int,
    max_seq_length: int,
    learning_rate: float,
    timeout_seconds: int,
) -> dict[str, Any]:
    """Build a minimal SFT job that uses the same shape as service-created jobs."""

    backend_output = output_dir / backend
    return {
        "backend": backend,
        "execution_backend": backend,
        "execution_executor": backend,
        "executor_mode": "real_import",
        "ready": True,
        "real_local": True,
        "real_training_enabled": True,
        "base_model": base_model,
        "output_dir": str(backend_output),
        "timeout_seconds": timeout_seconds,
        "training_examples": [
            {
                "sample_id": f"{backend}-cuda-smoke-1",
                "instruction": "Say ping.",
                "chosen": "pong",
                "output": "pong",
                "rejected": None,
                "sample_type": "sft",
            }
        ],
        "recipe": {
            "training": {
                "method": "qlora",
                "train_type": "sft",
                "base_model": base_model,
                "epochs": epochs,
                "max_seq_length": max_seq_length,
                "learning_rate": learning_rate,
                "batch_size": 1,
                "output_dir": str(backend_output / f"{backend}_output"),
            },
            "peft": {
                "lora_config": {
                    "r": 2,
                    "lora_alpha": 4,
                    "lora_dropout": 0.0,
                }
            },
        },
        "audit": {
            "import_probe": {
                "ready": True,
                "missing_modules": [],
            }
        },
    }


def summarize_result(result: Mapping[str, Any], *, backend: str, output_dir: Path) -> dict[str, Any]:
    runner = result.get("runner_result")
    runner_result = runner if isinstance(runner, Mapping) else {}
    diagnostics = result.get("diagnostics")
    diagnostics_map = diagnostics if isinstance(diagnostics, Mapping) else {}
    real_execution = runner_result.get("real_execution")
    real_execution_map = real_execution if isinstance(real_execution, Mapping) else {}
    nested_result = runner_result.get("result")
    nested_result_map = nested_result if isinstance(nested_result, Mapping) else {}
    returncode = result.get("returncode")
    if returncode is None:
        returncode = diagnostics_map.get("returncode")

    return {
        "backend": backend,
        "output_dir": str(output_dir),
        "status": result.get("status"),
        "dry_run": result.get("dry_run"),
        "returncode": returncode,
        "failure_category": result.get("failure_category") or diagnostics_map.get("failure_category"),
        "signal_name": diagnostics_map.get("signal_name"),
        "runner_status": runner_result.get("status") or diagnostics_map.get("runner_status"),
        "runner_execution_mode": runner_result.get("execution_mode"),
        "real_execution_kind": real_execution_map.get("kind"),
        "real_execution_success": real_execution_map.get("success"),
        "adapter_path": real_execution_map.get("artifact_dir") or nested_result_map.get("adapter_path"),
        "result_json": str(output_dir / backend / "training_job_result.json"),
        "diagnostics_json": str(output_dir / backend / "diagnostics.json"),
        "stdout_log": result.get("stdout_log") or diagnostics_map.get("stdout_log"),
        "stderr_log": result.get("stderr_log") or diagnostics_map.get("stderr_log"),
    }


def run_backend(
    *,
    backend: str,
    base_model: str,
    output_dir: Path,
    epochs: int,
    max_seq_length: int,
    learning_rate: float,
    timeout_seconds: int,
    dry_run: bool,
) -> dict[str, Any]:
    from pfe_core.trainer.runtime_job import dispatch_training_job

    job_spec = build_job_spec(
        backend=backend,
        base_model=base_model,
        output_dir=output_dir,
        epochs=epochs,
        max_seq_length=max_seq_length,
        learning_rate=learning_rate,
        timeout_seconds=timeout_seconds,
    )
    result = dispatch_training_job(job_spec, dry_run=dry_run)
    return {
        "job_spec": job_spec,
        "result": result,
        "summary": summarize_result(result, backend=backend, output_dir=output_dir),
    }


def write_summary(output_dir: Path, payload: Mapping[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "cuda_real_training_summary.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run minimal PEFT/Unsloth CUDA real-training validation through PFE isolation.",
    )
    parser.add_argument("--backend", choices=("peft", "unsloth", "all"), default="peft")
    parser.add_argument("--base-model", default=None, help="Override base model for the selected backend.")
    parser.add_argument("--peft-base-model", default=None, help=f"PEFT model id/path. Default: {DEFAULT_PEFT_MODEL}")
    parser.add_argument("--unsloth-base-model", default=None, help=f"Unsloth model id/path. Default: {DEFAULT_UNSLOTH_MODEL}")
    parser.add_argument("--output-dir", default=None, help="Directory for job materialization and diagnostics.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-seq-length", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--run", action="store_true", help="Actually launch real training. Omit for dry-run planning.")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    output_dir = Path(args.output_dir or tempfile.mkdtemp(prefix=f"pfe-cuda-verify-{_timestamp()}-", dir="/tmp")).expanduser().resolve()
    dry_run = not args.run
    if args.run:
        os.environ["PFE_REAL_TRAINING"] = "1"

    results = []
    for backend in _backend_list(args.backend):
        base_model = args.base_model or _default_model(
            backend,
            peft_model=args.peft_base_model,
            unsloth_model=args.unsloth_base_model,
        )
        results.append(
            run_backend(
                backend=backend,
                base_model=base_model,
                output_dir=output_dir,
                epochs=args.epochs,
                max_seq_length=args.max_seq_length,
                learning_rate=args.learning_rate,
                timeout_seconds=args.timeout_seconds,
                dry_run=dry_run,
            )
        )

    payload = {
        "dry_run": dry_run,
        "output_dir": str(output_dir),
        "backends": [item["summary"] for item in results],
        "results": results,
    }
    summary_path = write_summary(output_dir, payload)
    payload["summary_path"] = str(summary_path)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    failed = [
        item
        for item in payload["backends"]
        if not dry_run and item.get("status") != "completed"
    ]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
