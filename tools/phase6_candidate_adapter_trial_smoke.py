#!/usr/bin/env python3
"""Run the Phase6 candidate adapter trial smoke."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from first_run_smoke import _default_python, _repo_root, _require
from phase4_real_train_smoke import _read_manifest, _run_cli_checked
from pfe_core.phase6_candidate_adapter_trial import (
    PHASE6_RECOMMENDED_MODEL,
    qwen36_mlx_preflight,
    run_phase6_candidate_adapter_trial,
)


def _allocate_workdir(args: argparse.Namespace) -> tuple[Path, bool]:
    if args.workdir is not None:
        workdir = args.workdir.expanduser().resolve()
        if workdir.exists():
            shutil.rmtree(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        return workdir, False
    return Path(tempfile.mkdtemp(prefix="pfe-phase6-candidate-trial-")), not args.keep_workdir


def _latest_adapter_version(workdir: Path, workspace: str) -> str:
    adapter_root = workdir / ".pfe" / "adapters" / workspace
    versions = sorted(path.name for path in adapter_root.iterdir() if path.is_dir()) if adapter_root.is_dir() else []
    versions = [version for version in versions if len(version) == 12 and version[8] == "-"]
    if not versions:
        raise AssertionError(f"no adapter version directories were created under {adapter_root}")
    return versions[-1]


def _maybe_run_mlx_training(args: argparse.Namespace, workdir: Path, preflight: dict[str, Any]) -> dict[str, Any]:
    if not args.run_real_training:
        return {
            "real_training": "not_started",
            "mock_fallback": False,
            "skip_reason": "pass --run-real-training after preflight is ready",
        }
    if not preflight.get("ready_for_real_training"):
        return {
            "real_training": "blocked",
            "mock_fallback": False,
            "skip_reason": "phase6_qwen36_mlx_preflight_blocked",
            "blocked_by": preflight.get("blocked_by") or [preflight.get("status")],
        }

    base_model_value = str(args.model_path or args.model_id)
    init_output = _run_cli_checked(
        args,
        workdir,
        [
            "init",
            "--workspace",
            args.workspace,
            "--base-model",
            base_model_value,
            "--home",
            ".pfe",
        ],
    )
    _require(init_output, "PFE workspace initialized", label="init output")
    train_output = _run_cli_checked(
        args,
        workdir,
        [
            "train",
            "--workspace",
            args.workspace,
            "--base-model",
            base_model_value,
            "--backend",
            "mlx",
            "--real-local",
            "--train-type",
            "sft",
            "--epochs",
            str(args.epochs),
        ],
    )
    _require(train_output, "TRAINING COMPLETE", label="train output")
    version = _latest_adapter_version(workdir, args.workspace)
    manifest = _read_manifest(workdir, args.workspace, version)
    real_execution = dict(manifest.get("real_execution") or {})
    return {
        "real_training": "completed",
        "mock_fallback": False,
        "base_model": base_model_value,
        "adapter_version": version,
        "adapter_state": manifest.get("state"),
        "manifest_path": str(workdir / ".pfe" / "adapters" / args.workspace / version / "adapter_manifest.json"),
        "real_execution_summary": real_execution,
        "training_command": "pfe train --backend mlx --real-local --train-type sft",
    }


def _build_parser() -> argparse.ArgumentParser:
    repo_root = _repo_root()
    parser = argparse.ArgumentParser(
        description=(
            "Run Phase6 Candidate Adapter Trial Mode: Phase5 real domain loop -> "
            "signal/provenance candidate samples -> Qwen3.6/MLX preflight -> eval/decision."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--python", default=_default_python(repo_root))
    parser.add_argument("--workspace", default="phase6_candidate_trial")
    parser.add_argument("--source-limit", type=int, default=10)
    parser.add_argument("--candidate-limit", type=int, default=60)
    parser.add_argument("--holdout-count", type=int, default=16)
    parser.add_argument("--model-id", default=PHASE6_RECOMMENDED_MODEL)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--require-local-model", action="store_true")
    parser.add_argument("--allow-remote-download", action="store_true")
    parser.add_argument("--run-real-training", action="store_true")
    parser.add_argument("--real-model-calls", action="store_true")
    parser.add_argument("--strict-real", action="store_true", help="Return non-zero unless real MLX training completes.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--workdir", type=Path, default=None)
    parser.add_argument("--keep-workdir", action="store_true")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    args.repo_root = args.repo_root.resolve()

    workdir, cleanup_workdir = _allocate_workdir(args)
    previous_home = os.environ.get("PFE_HOME")
    os.environ["PFE_HOME"] = str(workdir / ".pfe")
    try:
        preflight = qwen36_mlx_preflight(
            model_id=args.model_id,
            model_path=args.model_path,
            require_local_model=args.require_local_model,
            allow_remote_download=args.allow_remote_download,
        )
        training = _maybe_run_mlx_training(args, workdir, preflight)
        result = run_phase6_candidate_adapter_trial(
            home=workdir / ".pfe",
            workspace=args.workspace,
            model_id=args.model_id,
            source_limit=args.source_limit,
            candidate_limit=args.candidate_limit,
            holdout_count=args.holdout_count,
            require_local_model=args.require_local_model,
            allow_remote_download=args.allow_remote_download,
            model_path=args.model_path,
            training=training,
            real_model_calls=args.real_model_calls and training.get("real_training") == "completed",
        )
        payload = {
            "ok": True,
            "workspace": args.workspace,
            "workdir": str(workdir),
            "workdir_retained": not cleanup_workdir,
            **result,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        if args.strict_real and training.get("real_training") != "completed":
            return 2
        return 0
    finally:
        if previous_home is None:
            os.environ.pop("PFE_HOME", None)
        else:
            os.environ["PFE_HOME"] = previous_home
        if cleanup_workdir:
            shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
