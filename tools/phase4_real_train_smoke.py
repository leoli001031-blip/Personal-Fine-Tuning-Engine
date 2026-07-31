#!/usr/bin/env python3
"""Smoke-test Phase 4 training handoff and opt-in real local training."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from first_run_smoke import (
    ADAPTER_VERSION_RE,
    _default_python,
    _repo_root,
    _require,
    _run_cli,
    _strip_ansi,
)
from prepare_tiny_hf_model import prepare_tiny_model
from pfe_server.app import build_serve_plan, smoke_test_request
from real_local_happy_path_smoke import _missing_modules, _verify_real_execution_manifest


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _default_tiny_model_dir() -> Path:
    return Path.home() / ".cache" / "pfe" / "release-models" / "tiny-gpt2-local"


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
    versions = sorted(
        path.name for path in adapter_root.iterdir() if path.is_dir() and ADAPTER_VERSION_RE.match(path.name)
    )
    if not versions:
        raise AssertionError(f"no adapter version directories were created under {adapter_root}")
    return versions[-1]


def _read_manifest(workdir: Path, workspace: str, version: str) -> dict[str, object]:
    manifest_path = workdir / ".pfe" / "adapters" / workspace / version / "adapter_manifest.json"
    if not manifest_path.is_file():
        raise AssertionError(f"adapter manifest was not created: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _resolve_train_model(args: argparse.Namespace) -> tuple[Path | None, dict[str, str], str | None]:
    raw_model = args.base_model or os.environ.get("PFE_PHASE4_REAL_TRAIN_MODEL", "").strip()
    if raw_model:
        model_path = Path(raw_model).expanduser()
        if model_path.exists():
            return model_path.resolve(), {}, None
        return None, {}, f"PFE_PHASE4_REAL_TRAIN_MODEL does not exist: {raw_model}"

    if args.prepare_tiny_model or _truthy(os.environ.get("PFE_PHASE4_PREPARE_TINY_MODEL")):
        try:
            prepared = prepare_tiny_model(args.tiny_model_dir.expanduser().resolve())
        except Exception as exc:
            return None, {}, f"tiny model preparation failed: {exc.__class__.__name__}: {exc}"
        return Path(prepared["output_dir"]).resolve(), prepared, None

    return None, {}, "PFE_PHASE4_REAL_TRAIN_MODEL is not set"


def _allocate_workdir(args: argparse.Namespace) -> tuple[Path, bool]:
    if args.workdir is not None:
        workdir = args.workdir.expanduser().resolve()
        if workdir.exists():
            shutil.rmtree(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        return workdir, False
    workdir = Path(tempfile.mkdtemp(prefix="pfe-phase4-train-"))
    return workdir, not args.keep_workdir


async def _request(
    app: Any,
    path: str,
    *,
    method: str = "GET",
    body: dict[str, Any] | None = None,
    allow_status: set[int] | None = None,
) -> dict[str, Any]:
    result = await smoke_test_request(app, path=path, method=method, body=body)
    if result["status_code"] != 200 and (not allow_status or result["status_code"] not in allow_status):
        raise AssertionError(f"{method} {path} failed: {result}")
    body_payload = dict(result.get("body") or {})
    body_payload["_status_code"] = result["status_code"]
    return body_payload


async def _seed_phase4_training_samples(args: argparse.Namespace, workdir: Path) -> tuple[Any, dict[str, Any]]:
    source_path = workdir / "phase4-training.txt"
    source_path.write_text(
        (
            "Phase4 training should export eligible real-corpus candidates to the existing "
            "SFT sample store. Real LoRA training is attempted only when a local trainable "
            "model path is explicitly configured for this smoke. The assistant should stay "
            "inside supplied material, preserve citations, and ask for human confirmation "
            "for legal, medical, financial, or otherwise high-risk conclusions."
        ),
        encoding="utf-8",
    )
    plan = build_serve_plan(workspace=args.workspace, dry_run=True)
    app = plan.app
    await _request(app, "/pfe/phase4/sources", method="POST", body={"path": str(source_path)})
    await _request(
        app,
        "/pfe/phase4/training-candidates",
        method="POST",
        body={"limit": args.candidate_limit, "export": True},
    )
    sample_export = await _request(
        app,
        "/pfe/phase4/training-candidates/export",
        method="POST",
        body={"target": "samples_db"},
    )
    return app, sample_export


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    workdir, cleanup_workdir = _allocate_workdir(args)
    previous_home = os.environ.get("PFE_HOME")
    os.environ["PFE_HOME"] = str(workdir / ".pfe")
    try:
        app, sample_export = await _seed_phase4_training_samples(args, workdir)
        train_model, prepared_tiny_model, model_skip_reason = _resolve_train_model(args)

        if train_model is None:
            adapter = await _request(app, "/pfe/phase4/candidate-adapter", method="POST")
            return {
                "ok": True,
                "workspace": args.workspace,
                "workdir": str(workdir),
                "workdir_retained": not cleanup_workdir,
                "real_training": "skipped",
                "skip_reason": model_skip_reason,
                "saved_training_samples": sample_export["saved_samples"],
                "split_counts": sample_export["split_counts"],
                "mock_fallback": True,
                "candidate_adapter_version": adapter["adapter_version"],
                "candidate_adapter_state": adapter["state"],
                "training_path": "phase4_candidate_adapter_fallback",
            }

        missing = _missing_modules()
        if missing:
            adapter = await _request(
                app,
                "/pfe/phase4/candidate-adapter",
                method="POST",
                body={"base_model": str(train_model)},
            )
            return {
                "ok": True,
                "workspace": args.workspace,
                "workdir": str(workdir),
                "workdir_retained": not cleanup_workdir,
                "real_training": "skipped",
                "skip_reason": "configured local model path requires training runtime modules",
                "missing_modules": missing,
                "base_model": str(train_model),
                "saved_training_samples": sample_export["saved_samples"],
                "split_counts": sample_export["split_counts"],
                "mock_fallback": True,
                "candidate_adapter_version": adapter["adapter_version"],
                "candidate_adapter_state": adapter["state"],
                "training_path": "phase4_candidate_adapter_fallback",
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
        metadata = dict(manifest.get("metadata") or {})
        artifacts = dict(metadata.get("real_execution_artifacts") or {})
        artifact_evidence = {
            "success": artifacts.get("success"),
            "artifact_kind": artifacts.get("artifact_kind") or real_summary.get("kind"),
            "artifact_dir": artifacts.get("artifact_dir") or real_summary.get("artifact_dir"),
            "train_loss": artifacts.get("train_loss") or real_summary.get("train_loss"),
            "num_examples": artifacts.get("num_examples") or real_summary.get("num_examples"),
            "source_path": artifacts.get("source_path") or real_summary.get("source_path"),
        }
        return {
            "ok": True,
            "workspace": args.workspace,
            "workdir": str(workdir),
            "workdir_retained": not cleanup_workdir,
            "real_training": "completed",
            "saved_training_samples": sample_export["saved_samples"],
            "split_counts": sample_export["split_counts"],
            "base_model": base_model_value,
            "prepared_tiny_model": prepared_tiny_model or None,
            "adapter_version": version,
            "adapter_state": manifest.get("state"),
            "manifest_path": str(
                workdir / ".pfe" / "adapters" / args.workspace / version / "adapter_manifest.json"
            ),
            "real_execution_summary": real_summary,
            "real_execution_artifacts": artifact_evidence,
            "training_command": "pfe train --backend peft --real-local --epochs 1",
            "training_output_contains": "TRAINING COMPLETE",
        }
    finally:
        if previous_home is None:
            os.environ.pop("PFE_HOME", None)
        else:
            os.environ["PFE_HOME"] = previous_home
        if cleanup_workdir:
            shutil.rmtree(workdir, ignore_errors=True)


def _build_parser() -> argparse.ArgumentParser:
    repo_root = _repo_root()
    parser = argparse.ArgumentParser(
        description=(
            "Smoke-test Phase4 corpus-to-training handoff. By default it verifies sample export "
            "and records a clear skip with mock fallback. Pass --prepare-tiny-model or "
            "PFE_PHASE4_REAL_TRAIN_MODEL=/abs/path to run a real local PEFT adapter proof."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--python", default=_default_python(repo_root))
    parser.add_argument("--workspace", default="phase4_train_smoke")
    parser.add_argument("--candidate-limit", type=int, default=6)
    parser.add_argument(
        "--base-model",
        default=None,
        help="Local model/config directory. Defaults to PFE_PHASE4_REAL_TRAIN_MODEL.",
    )
    parser.add_argument(
        "--prepare-tiny-model",
        action="store_true",
        help="Create/reuse the repo tiny HF model and use it for real training.",
    )
    parser.add_argument("--tiny-model-dir", type=Path, default=_default_tiny_model_dir())
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--strict-real", action="store_true", help="Return non-zero when real training is skipped.")
    parser.add_argument("--workdir", type=Path, default=None)
    parser.add_argument("--keep-workdir", action="store_true")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    args.repo_root = args.repo_root.resolve()

    result = asyncio.run(_run(args))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if args.strict_real and result.get("real_training") != "completed":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
