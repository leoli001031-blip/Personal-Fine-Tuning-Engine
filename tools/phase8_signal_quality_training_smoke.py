#!/usr/bin/env python3
"""Run the Phase8 high-quality signal training lift smoke."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping

from first_run_smoke import _default_python, _pythonpath, _repo_root, _strip_ansi
from phase4_real_train_smoke import _read_manifest
from pfe_core.phase8_signal_quality_training import (
    PHASE8_RECOMMENDED_MODEL,
    Phase8SignalQualityTrainingStore,
    finalize_phase8_signal_quality_trial,
    prepare_phase8_signal_quality_trial,
)


def _allocate_workdir(args: argparse.Namespace) -> tuple[Path, bool]:
    if args.workdir is not None:
        workdir = args.workdir.expanduser().resolve()
        if workdir.exists():
            shutil.rmtree(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        return workdir, False
    return Path(tempfile.mkdtemp(prefix="pfe-phase8-signal-quality-")), not args.keep_workdir


def _latest_adapter_version(workdir: Path, workspace: str) -> str:
    adapter_root = workdir / ".pfe" / "adapters" / workspace
    versions = sorted(path.name for path in adapter_root.iterdir() if path.is_dir()) if adapter_root.is_dir() else []
    versions = [version for version in versions if len(version) == 12 and version[8] == "-"]
    if not versions:
        raise AssertionError(f"no adapter version directories were created under {adapter_root}")
    return versions[-1]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _resolve_training_adapter_path(workdir: Path, workspace: str, version: str) -> str:
    version_dir = workdir / ".pfe" / "adapters" / workspace / version
    result = _read_json(version_dir / "training_job_result.json")
    adapter_path = str(dict(dict(result.get("result") or {}).get("result") or {}).get("adapter_path") or "").strip()
    candidates: list[Path] = []
    if adapter_path:
        path = Path(adapter_path)
        candidates.append(path if path.is_absolute() else version_dir / path)
    candidates.extend(
        [
            version_dir / "adapters",
            version_dir / "mlx_output" / "adapters",
        ]
    )
    for path in candidates:
        if path.exists():
            return str(path)
    return str(candidates[0] if candidates else version_dir / "adapters")


def _run_cli_capture(args: argparse.Namespace, workdir: Path, command_args: list[str], *, log_path: Path) -> dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONPATH"] = _pythonpath(args.repo_root)
    env["PFE_HOME"] = str(workdir / ".pfe")
    completed: subprocess.CompletedProcess[str] | None = None
    timed_out = False
    try:
        completed = subprocess.run(
            [args.python, "-m", "pfe_cli.main", *command_args],
            cwd=str(workdir),
            env=env,
            text=True,
            capture_output=True,
            timeout=args.timeout,
            check=False,
        )
        stdout = _strip_ansi(completed.stdout)
        stderr = _strip_ansi(completed.stderr)
        returncode = completed.returncode
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        stdout = _strip_ansi(exc.stdout or "")
        stderr = _strip_ansi(exc.stderr or "")
        returncode = None

    log = {
        "command": "pfe " + " ".join(command_args),
        "returncode": returncode,
        "timed_out": timed_out,
        "stdout": stdout,
        "stderr": stderr,
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(log, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {**log, "log_path": str(log_path)}


def _maybe_run_qwen_training(args: argparse.Namespace, workdir: Path, preflight: Mapping[str, Any]) -> dict[str, Any]:
    logs_dir = workdir / "phase8-command-logs"
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
            "skip_reason": "phase8_qwen_mlx_preflight_blocked",
            "blocked_by": preflight.get("blocked_by") or [preflight.get("status")],
        }

    base_model_value = str(args.model_path or args.model_id)
    init_log = _run_cli_capture(
        args,
        workdir,
        ["init", "--workspace", args.workspace, "--base-model", base_model_value, "--home", ".pfe"],
        log_path=logs_dir / "pfe-init.json",
    )
    if init_log["returncode"] != 0:
        return {
            "real_training": "blocked",
            "mock_fallback": False,
            "skip_reason": "pfe_init_failed",
            "base_model": base_model_value,
            "init_log": init_log,
        }

    train_log = _run_cli_capture(
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
        log_path=logs_dir / "pfe-train-mlx.json",
    )
    if train_log["returncode"] != 0:
        return {
            "real_training": "blocked",
            "mock_fallback": False,
            "skip_reason": "pfe_train_failed",
            "base_model": base_model_value,
            "init_log": init_log,
            "train_log": train_log,
        }

    version = _latest_adapter_version(workdir, args.workspace)
    manifest = _read_manifest(workdir, args.workspace, version)
    adapter_path = _resolve_training_adapter_path(workdir, args.workspace, version)
    return {
        "real_training": "completed",
        "mock_fallback": False,
        "base_model": base_model_value,
        "adapter_version": version,
        "adapter_path": adapter_path,
        "adapter_state": manifest.get("state"),
        "manifest_path": str(workdir / ".pfe" / "adapters" / args.workspace / version / "adapter_manifest.json"),
        "real_execution_summary": manifest.get("real_execution") or {},
        "training_command": "pfe train --backend mlx --real-local --train-type sft",
        "init_log": init_log,
        "train_log": train_log,
    }


def _generate_one(model: Any, tokenizer: Any, prompt: str, *, max_tokens: int) -> str:
    from mlx_lm import generate

    return str(generate(model, tokenizer, prompt=prompt, verbose=False, max_tokens=max_tokens))


def _maybe_run_real_eval(args: argparse.Namespace, workdir: Path, training: Mapping[str, Any]) -> dict[str, Any]:
    if not args.run_real_eval:
        return {"real_model_calls": False, "skip_reason": "pass --run-real-eval after real training completes"}
    if training.get("real_training") != "completed":
        return {"real_model_calls": False, "skip_reason": "real_training_not_completed"}

    try:
        import mlx.core as mx
        from mlx_lm import load
    except Exception as exc:
        return {
            "real_model_calls": False,
            "skip_reason": "mlx_eval_dependencies_missing",
            "error": str(exc),
        }

    store = Phase8SignalQualityTrainingStore(home=workdir / ".pfe", workspace=args.workspace)
    holdouts = store.read_holdouts()[: max(1, args.eval_samples)]
    adapter_version = str(training.get("adapter_version") or "")
    adapter_path = Path(
        str(
            training.get("adapter_path")
            or workdir / ".pfe" / "adapters" / args.workspace / adapter_version / "adapters"
        )
    )
    if not adapter_path.exists():
        return {
            "real_model_calls": False,
            "skip_reason": "adapter_path_missing",
            "adapter_path": str(adapter_path),
        }
    base_model_value = str(training.get("base_model") or args.model_path or args.model_id)
    details: list[dict[str, Any]] = []

    try:
        base_model, base_tokenizer = load(base_model_value)
        try:
            for item in holdouts:
                prompt = str(item.get("prompt") or "")
                base_output = _generate_one(base_model, base_tokenizer, prompt, max_tokens=args.eval_max_tokens)
                details.append({"prompt_id": item.get("prompt_id"), "base_output": base_output})
        finally:
            del base_model
            mx.clear_cache()

        adapter_model, adapter_tokenizer = load(base_model_value, adapter_path=str(adapter_path))
        try:
            by_id = {str(item["prompt_id"]): item for item in details}
            for item in holdouts:
                prompt = str(item.get("prompt") or "")
                adapter_output = _generate_one(adapter_model, adapter_tokenizer, prompt, max_tokens=args.eval_max_tokens)
                by_id[str(item.get("prompt_id"))]["adapter_output"] = adapter_output
        finally:
            del adapter_model
            mx.clear_cache()
    except Exception as exc:
        mx.clear_cache()
        return {
            "real_model_calls": False,
            "skip_reason": "real_eval_failed",
            "error": str(exc),
            "base_model": base_model_value,
            "adapter_path": str(adapter_path),
            "details": details,
        }

    return {
        "real_model_calls": True,
        "base_model": base_model_value,
        "adapter_path": str(adapter_path),
        "eval_samples": len(details),
        "details": details,
    }


def _copy_evidence(args: argparse.Namespace, payload: Mapping[str, Any]) -> dict[str, str]:
    if args.evidence_dir is None:
        return {}
    evidence_dir = args.evidence_dir.expanduser().resolve()
    if evidence_dir.exists() and args.clean_evidence:
        shutil.rmtree(evidence_dir)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    copied: dict[str, str] = {}
    for key, raw in dict(payload.get("paths") or {}).items():
        path = Path(str(raw))
        if not path.exists() or not path.is_file():
            continue
        suffix = path.suffix or ".json"
        target = evidence_dir / f"{key}{suffix}"
        shutil.copy2(path, target)
        copied[key] = str(target)
    for log_key in ("init_log", "train_log"):
        raw_log_path = str(dict(payload.get("training_result", {}).get("training") or {}).get(log_key, {}).get("log_path") or "").strip()
        if not raw_log_path:
            continue
        log_path = Path(raw_log_path)
        if log_path.exists():
            target = evidence_dir / f"{log_key}.json"
            shutil.copy2(log_path, target)
            copied[log_key] = str(target)
    output_path = evidence_dir / "phase8-smoke-output.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    copied["smoke_output"] = str(output_path)
    return copied


def _build_parser() -> argparse.ArgumentParser:
    repo_root = _repo_root()
    parser = argparse.ArgumentParser(
        description=(
            "Run Phase8: public contract sources -> high-quality signals -> "
            "quality-gated samples -> Qwen3-0.6B MLX training attempt -> holdout eval decision."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--python", default=_default_python(repo_root))
    parser.add_argument("--workspace", default="phase8_signal_quality_training")
    parser.add_argument("--source-limit", type=int, default=11)
    parser.add_argument("--signal-count", type=int, default=60)
    parser.add_argument("--candidate-limit", type=int, default=60)
    parser.add_argument("--holdout-count", type=int, default=10)
    parser.add_argument("--model-id", default=PHASE8_RECOMMENDED_MODEL)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--require-local-model", action="store_true")
    parser.add_argument("--allow-remote-download", action="store_true")
    parser.add_argument("--run-real-training", action="store_true")
    parser.add_argument("--run-real-eval", action="store_true")
    parser.add_argument("--strict-real", action="store_true", help="Return non-zero unless real MLX training completes.")
    parser.add_argument("--strict-real-eval", action="store_true", help="Return non-zero unless real base/adapter eval completes.")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--timeout", type=int, default=2400)
    parser.add_argument("--eval-samples", type=int, default=10)
    parser.add_argument("--eval-max-tokens", type=int, default=160)
    parser.add_argument("--workdir", type=Path, default=None)
    parser.add_argument("--keep-workdir", action="store_true")
    parser.add_argument("--evidence-dir", type=Path, default=None)
    parser.add_argument("--clean-evidence", action="store_true")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    args.repo_root = args.repo_root.resolve()
    workdir, cleanup_workdir = _allocate_workdir(args)
    previous_home = os.environ.get("PFE_HOME")
    os.environ["PFE_HOME"] = str(workdir / ".pfe")
    try:
        prepared = prepare_phase8_signal_quality_trial(
            home=workdir / ".pfe",
            workspace=args.workspace,
            source_limit=args.source_limit,
            signal_count=args.signal_count,
            candidate_limit=args.candidate_limit,
            holdout_count=args.holdout_count,
            model_id=args.model_id,
            model_path=args.model_path,
            require_local_model=args.require_local_model,
            allow_remote_download=args.allow_remote_download,
        )
        training = _maybe_run_qwen_training(args, workdir, prepared["preflight"])
        generations = _maybe_run_real_eval(args, workdir, training)
        finalized = finalize_phase8_signal_quality_trial(
            home=workdir / ".pfe",
            workspace=args.workspace,
            training=training,
            generations=generations,
            real_model_calls=bool(generations.get("real_model_calls")),
        )
        payload = {
            "ok": True,
            "workspace": args.workspace,
            "workdir": str(workdir),
            "workdir_retained": not cleanup_workdir,
            **prepared,
            **finalized,
        }
        evidence_copies = _copy_evidence(args, payload)
        payload["evidence_copies"] = evidence_copies
        if evidence_copies:
            Path(evidence_copies["smoke_output"]).write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        if args.strict_real and training.get("real_training") != "completed":
            return 2
        if args.strict_real_eval and not generations.get("real_model_calls"):
            return 3
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
