#!/usr/bin/env python3
"""Run the opt-in real local model happy-path smoke."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

from first_run_smoke import ADAPTER_VERSION_RE, _default_python, _repo_root, _require, _run_cli, _strip_ansi


REQUIRED_REAL_PEFT_MODULES = ("torch", "transformers", "peft", "accelerate")


def _missing_modules() -> list[str]:
    return [name for name in REQUIRED_REAL_PEFT_MODULES if importlib.util.find_spec(name) is None]


def _resolve_base_model(args: argparse.Namespace) -> Path | None:
    raw = args.base_model or os.environ.get("PFE_REAL_LOCAL_MODEL")
    if not raw:
        return None
    path = Path(str(raw)).expanduser()
    if not path.exists():
        raise AssertionError(f"local model path does not exist: {path}")
    return path.resolve()


def _run_cli_checked(args: argparse.Namespace, workdir: Path, command_args: list[str]) -> str:
    output = _run_cli(
        python=args.python,
        repo_root=args.repo_root,
        cwd=workdir,
        args=command_args,
        timeout=args.timeout,
    )
    return _strip_ansi(output)


def _latest_adapter_version(workdir: Path, workspace: str) -> str:
    adapter_root = workdir / ".pfe" / "adapters" / workspace
    if not adapter_root.is_dir():
        raise AssertionError(f"adapter root was not created: {adapter_root}")
    versions = sorted(path.name for path in adapter_root.iterdir() if path.is_dir() and ADAPTER_VERSION_RE.match(path.name))
    if not versions:
        raise AssertionError(f"no adapter version directories were created under {adapter_root}")
    return versions[-1]


def _read_manifest(workdir: Path, workspace: str, version: str) -> dict[str, object]:
    manifest_path = workdir / ".pfe" / "adapters" / workspace / version / "adapter_manifest.json"
    if not manifest_path.is_file():
        raise AssertionError(f"adapter manifest was not created: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _verify_real_execution_manifest(manifest: dict[str, object], *, base_model: Path) -> dict[str, object]:
    metadata = manifest.get("metadata")
    if not isinstance(metadata, dict):
        raise AssertionError(f"adapter manifest did not include metadata: {manifest}")
    real_summary = metadata.get("real_execution_summary")
    if not isinstance(real_summary, dict):
        raise AssertionError(f"adapter manifest did not include real execution summary: {manifest}")
    kind = str(real_summary.get("kind") or "")
    if kind not in {"real_peft", "real_local_peft"}:
        raise AssertionError(f"unexpected real execution kind: {real_summary}")
    path = str(real_summary.get("path") or "")
    if path not in {"real_import", "real_local"}:
        raise AssertionError(f"unexpected real execution path: {real_summary}")

    text = json.dumps(manifest, ensure_ascii=False, sort_keys=True)
    if str(base_model) not in text:
        raise AssertionError(f"adapter manifest did not retain the local base model path {base_model}:\n{text}")

    artifacts = metadata.get("real_execution_artifacts")
    if not isinstance(artifacts, dict):
        raise AssertionError(f"adapter manifest did not include real execution artifacts: {manifest}")
    if artifacts.get("success") is not True:
        raise AssertionError(f"real execution artifacts did not report success: {artifacts}")
    return real_summary


def _run_smoke(args: argparse.Namespace, workdir: Path, base_model: Path) -> dict[str, str]:
    workspace = args.workspace
    base_model_value = str(base_model)

    init_output = _run_cli_checked(
        args,
        workdir,
        [
            "init",
            "--workspace",
            workspace,
            "--base-model",
            base_model_value,
            "--home",
            ".pfe",
        ],
    )
    _require(init_output, "PFE workspace initialized", label="init output")
    _require(init_output, f"base model:  {base_model_value}", label="init output")

    doctor_output = _run_cli_checked(args, workdir, ["doctor", "--workspace", workspace])
    _require(doctor_output, "local model: available=yes", label="doctor output")
    _require(doctor_output, f"requested_base_model={base_model_value}", label="doctor output")

    generate_output = _run_cli_checked(
        args,
        workdir,
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
        ],
    )
    _require(generate_output, f"Saved {args.num_samples} distilled sample(s)", label="generate output")

    train_output = _run_cli_checked(
        args,
        workdir,
        [
            "train",
            "--workspace",
            workspace,
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

    version = _latest_adapter_version(workdir, workspace)
    manifest = _read_manifest(workdir, workspace, version)
    real_summary = _verify_real_execution_manifest(manifest, base_model=base_model)

    eval_output = _run_cli_checked(
        args,
        workdir,
        ["eval", "--base-model", "base", "--adapter", version, "--num-samples", "3", "--workspace", workspace],
    )
    _require(eval_output, "[ EVALUATION RESULT ]", label="eval output")

    promote_output = _run_cli_checked(args, workdir, ["adapter", "promote", version, "--workspace", workspace])
    _require(promote_output, version, label="promote output")

    serve_preview_output = _run_cli_checked(
        args,
        workdir,
        ["serve", "--workspace", workspace, "--host", "127.0.0.1", "--port", str(args.port), "--real-local"],
    )
    _require(serve_preview_output, "[ SERVE PREVIEW ]", label="serve preview output")
    _require(serve_preview_output, "REAL", label="serve preview output")

    return {
        "base_model": base_model_value,
        "kind": str(real_summary.get("kind") or ""),
        "path": str(real_summary.get("path") or ""),
        "version": version,
        "workspace": workspace,
    }


def main() -> int:
    repo_root = _repo_root()
    parser = argparse.ArgumentParser(
        description=(
            "Run a real local model happy path. Set PFE_REAL_LOCAL_MODEL or pass --base-model "
            "to a local Hugging Face-style model/config directory. Missing model paths skip by default; "
            "provided model paths require torch, transformers, peft, and accelerate."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--python", default=_default_python(repo_root))
    parser.add_argument("--workspace", default="real_local_happy")
    parser.add_argument("--base-model", default=None, help="Local model/config directory. Defaults to PFE_REAL_LOCAL_MODEL.")
    parser.add_argument("--port", type=int, default=8921)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--strict", action="store_true", help="Fail instead of skipping when no local model path is configured.")
    parser.add_argument("--workdir", type=Path, default=None)
    parser.add_argument("--keep-workdir", action="store_true")
    args = parser.parse_args()
    args.repo_root = args.repo_root.resolve()

    if args.num_samples < 4:
        raise SystemExit("--num-samples must be at least 4 so train/val/test splits are populated")

    try:
        base_model = _resolve_base_model(args)
    except AssertionError as exc:
        print("REAL-LOCAL HAPPY PATH SMOKE FAILED")
        print(f"reason: {exc}")
        return 2

    if base_model is None:
        print("REAL-LOCAL HAPPY PATH SMOKE SKIPPED")
        print("reason: set PFE_REAL_LOCAL_MODEL=/abs/path/to/local-model or pass --base-model")
        return 2 if args.strict else 0

    missing = _missing_modules()
    if missing:
        print("REAL-LOCAL HAPPY PATH SMOKE FAILED")
        print("reason: configured local model path requires training runtime modules")
        print(f"missing: {', '.join(missing)}")
        print("hint: install the training extras, then rerun with the same PFE_REAL_LOCAL_MODEL")
        return 2

    tempdir = None
    if args.workdir is None:
        tempdir = tempfile.TemporaryDirectory(prefix="pfe-real-local-happy-")
        workdir = Path(tempdir.name)
    else:
        workdir = args.workdir.resolve()
        if workdir.exists():
            shutil.rmtree(workdir)
        workdir.mkdir(parents=True, exist_ok=True)

    print(f"workdir: {workdir}")
    print(f"python:  {args.python}")
    print(f"model:   {base_model}")
    print()
    try:
        summary = _run_smoke(args, workdir, base_model)
        print("REAL-LOCAL HAPPY PATH SMOKE PASSED")
        print(f"workspace:  {summary['workspace']}")
        print(f"version:    {summary['version']}")
        print(f"base_model: {summary['base_model']}")
        print(f"execution:  kind={summary['kind']} | path={summary['path']}")
        return 0
    finally:
        if tempdir is not None and not args.keep_workdir:
            tempdir.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
