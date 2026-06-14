#!/usr/bin/env python3
"""Smoke-test the dependency-safe real-local readiness path."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

from first_run_smoke import _default_python, _repo_root, _require, _run_cli, _strip_ansi


def _write_minimal_local_model(workdir: Path) -> str:
    model_dir = workdir / "models" / "local-base"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["GPT2LMHeadModel"],
                "model_type": "gpt2",
                "vocab_size": 32,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return "./models/local-base"


def _run_smoke(args: argparse.Namespace, workdir: Path) -> dict[str, str]:
    workspace = args.workspace
    base_model = _write_minimal_local_model(workdir)

    def run(command_args: list[str]) -> str:
        output = _run_cli(
            python=args.python,
            repo_root=args.repo_root,
            cwd=workdir,
            args=command_args,
            timeout=args.timeout,
        )
        return _strip_ansi(output)

    init_output = run(
        [
            "init",
            "--workspace",
            workspace,
            "--base-model",
            base_model,
            "--home",
            ".pfe",
        ]
    )
    _require(init_output, "PFE workspace initialized", label="init output")
    _require(init_output, f"base model:  {base_model}", label="init output")

    doctor_output = run(["doctor", "--workspace", workspace])
    _require(doctor_output, "local model: available=yes", label="doctor output")
    _require(doctor_output, f"requested_base_model={base_model}", label="doctor output")
    _require(doctor_output, "source_kind=path", label="doctor output")

    next_output = run(["next", "--workspace", workspace])
    _require(next_output, "state: collect_feedback", label="next output")

    generate_output = run(
        [
            "generate",
            "--scenario",
            "life-coach",
            "--style",
            "warm",
            "--num",
            str(args.num_samples),
            "--workspace",
            workspace,
        ]
    )
    _require(generate_output, f"Saved {args.num_samples} distilled sample(s)", label="generate output")

    train_preview_output = run(
        [
            "train",
            "--workspace",
            workspace,
            "--backend",
            "peft",
            "--real-local",
            "--preview",
            "--epochs",
            "1",
        ]
    )
    _require(train_preview_output, "PFE train plan", label="train preview output")
    _require(train_preview_output, "execution_intent=real_local", label="train preview output")
    _require(train_preview_output, "backend-dispatch:", label="train preview output")
    _require(train_preview_output, "execution_backend=peft", label="train preview output")

    serve_preview_output = run(
        [
            "serve",
            "--workspace",
            workspace,
            "--host",
            "127.0.0.1",
            "--port",
            str(args.port),
            "--real-local",
        ]
    )
    _require(serve_preview_output, "[ SERVE PREVIEW ]", label="serve preview output")
    _require(serve_preview_output, "mode", label="serve preview output")
    _require(serve_preview_output, "REAL", label="serve preview output")

    console_output = run(["console", "--workspace", workspace, "--cycles", "1", "--real-local"])
    _require(console_output, "ENTERING MATRIX CONSOLE MODE", label="console output")
    _require(console_output, workspace, label="console output")

    return {
        "base_model": base_model,
        "config_path": str(workdir / ".pfe" / "config.toml"),
        "workspace": workspace,
    }


def main() -> int:
    repo_root = _repo_root()
    parser = argparse.ArgumentParser(
        description=(
            "Smoke-test the real-local readiness path without model downloads or heavy training dependencies. "
            "This validates local model discovery, train --preview --real-local, serve --real-local preview, "
            "and console snapshot wiring."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--python", default=_default_python(repo_root))
    parser.add_argument("--workspace", default="real_local_ready")
    parser.add_argument("--port", type=int, default=8921)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--workdir", type=Path, default=None)
    parser.add_argument("--keep-workdir", action="store_true")
    args = parser.parse_args()
    args.repo_root = args.repo_root.resolve()

    if args.num_samples < 4:
        raise SystemExit("--num-samples must be at least 4 so train/val/test splits are populated")

    tempdir = None
    if args.workdir is None:
        tempdir = tempfile.TemporaryDirectory(prefix="pfe-real-local-readiness-")
        workdir = Path(tempdir.name)
    else:
        workdir = args.workdir.resolve()
        if workdir.exists():
            shutil.rmtree(workdir)
        workdir.mkdir(parents=True, exist_ok=True)

    print(f"workdir: {workdir}")
    print(f"python:  {args.python}")
    print()
    try:
        summary = _run_smoke(args, workdir)
        print("REAL-LOCAL READINESS SMOKE PASSED")
        print(f"workspace:  {summary['workspace']}")
        print(f"base_model: {summary['base_model']}")
        print(f"config:     {summary['config_path']}")
        return 0
    finally:
        if tempdir is not None and not args.keep_workdir:
            tempdir.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
