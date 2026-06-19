#!/usr/bin/env python3
"""Run the Phase5 real-domain loop smoke with optional real tiny training."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from first_run_smoke import _default_python, _repo_root, _require
from phase4_real_train_smoke import _read_manifest, _resolve_train_model, _run_cli_checked
from pfe_core.phase5_real_domain_loop import run_phase5_domain_loop
from real_local_happy_path_smoke import _missing_modules, _verify_real_execution_manifest


def _allocate_workdir(args: argparse.Namespace) -> tuple[Path, bool]:
    if args.workdir is not None:
        workdir = args.workdir.expanduser().resolve()
        if workdir.exists():
            shutil.rmtree(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        return workdir, False
    return Path(tempfile.mkdtemp(prefix="pfe-phase5-domain-loop-")), not args.keep_workdir


def _latest_adapter_version(workdir: Path, workspace: str) -> str:
    adapter_root = workdir / ".pfe" / "adapters" / workspace
    versions = sorted(path.name for path in adapter_root.iterdir() if path.is_dir()) if adapter_root.is_dir() else []
    versions = [version for version in versions if len(version) == 12 and version[8] == "-"]
    if not versions:
        raise AssertionError(f"no adapter version directories were created under {adapter_root}")
    return versions[-1]


def _maybe_run_real_training(args: argparse.Namespace, workdir: Path) -> dict[str, Any]:
    train_model, prepared_tiny_model, skip_reason = _resolve_train_model(args)
    if train_model is None:
        return {
            "real_training": "skipped",
            "skip_reason": skip_reason,
            "mock_fallback": False,
        }
    missing = _missing_modules()
    if missing:
        return {
            "real_training": "skipped",
            "skip_reason": "configured local model path requires training runtime modules",
            "missing_modules": missing,
            "base_model": str(train_model),
            "mock_fallback": False,
        }

    base_model_value = str(train_model)
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
            "peft",
            "--real-local",
            "--epochs",
            "1",
        ],
    )
    _require(train_output, "TRAINING COMPLETE", label="train output")
    version = _latest_adapter_version(workdir, args.workspace)
    manifest = _read_manifest(workdir, args.workspace, version)
    real_summary = _verify_real_execution_manifest(manifest, base_model=train_model)
    return {
        "real_training": "completed",
        "base_model": base_model_value,
        "prepared_tiny_model": prepared_tiny_model or None,
        "adapter_version": version,
        "adapter_state": manifest.get("state"),
        "manifest_path": str(workdir / ".pfe" / "adapters" / args.workspace / version / "adapter_manifest.json"),
        "real_execution_summary": real_summary,
        "training_command": "pfe train --backend peft --real-local --epochs 1",
    }


def _build_parser() -> argparse.ArgumentParser:
    repo_root = _repo_root()
    parser = argparse.ArgumentParser(
        description=(
            "Run Phase5 real-domain loop proof: Common Paper sources -> Phase4 candidates -> "
            "samples DB -> holdout eval -> loop correction signals -> optional real PEFT training."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--python", default=_default_python(repo_root))
    parser.add_argument("--workspace", default="phase5_domain_loop")
    parser.add_argument("--source-limit", type=int, default=10)
    parser.add_argument("--candidate-limit", type=int, default=60)
    parser.add_argument("--holdout-count", type=int, default=16)
    parser.add_argument("--base-model", default=None, help="Local model/config directory. Defaults to PFE_PHASE4_REAL_TRAIN_MODEL.")
    parser.add_argument("--prepare-tiny-model", action="store_true", help="Create/reuse the repo tiny HF model and use it for real training.")
    parser.add_argument("--tiny-model-dir", type=Path, default=Path.home() / ".cache" / "pfe" / "release-models" / "tiny-gpt2-local")
    parser.add_argument("--strict-real", action="store_true", help="Return non-zero when real training is skipped.")
    parser.add_argument("--timeout", type=int, default=180)
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
        loop = run_phase5_domain_loop(
            home=workdir / ".pfe",
            workspace=args.workspace,
            source_limit=args.source_limit,
            candidate_limit=args.candidate_limit,
            holdout_count=args.holdout_count,
        )
        training = _maybe_run_real_training(args, workdir)
        result = {
            "ok": True,
            "workspace": args.workspace,
            "workdir": str(workdir),
            "workdir_retained": not cleanup_workdir,
            **loop,
            "real_training": training,
        }
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
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
