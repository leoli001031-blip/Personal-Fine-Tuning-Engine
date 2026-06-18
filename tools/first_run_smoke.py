#!/usr/bin/env python3
"""Run the local first-run PFE smoke path in an isolated temp directory."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
ADAPTER_VERSION_RE = re.compile(r"^\d{8}-\d{3}$")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_python(repo_root: Path) -> str:
    for candidate in (
        repo_root / ".venv" / "bin" / "python",
        repo_root / ".venv" / "bin" / "python3",
    ):
        if candidate.exists():
            return str(candidate)
    if sys.version_info >= (3, 10):
        return sys.executable
    for name in ("python3.12", "python3.11", "python3.10"):
        resolved = shutil.which(name)
        if resolved:
            return resolved
    return sys.executable


def _pythonpath(repo_root: Path) -> str:
    package_paths = [
        repo_root / "pfe-core",
        repo_root / "pfe-cli",
        repo_root / "pfe-server",
    ]
    existing = os.environ.get("PYTHONPATH")
    parts = [str(path) for path in package_paths]
    if existing:
        parts.append(existing)
    return os.pathsep.join(parts)


def _strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def _run_cli(
    *,
    python: str,
    repo_root: Path,
    cwd: Path,
    args: list[str],
    timeout: int,
) -> str:
    env = os.environ.copy()
    env.pop("PFE_HOME", None)
    env.pop("PFE_ENABLE_REAL_LOCAL_INFERENCE", None)
    env["PYTHONPATH"] = _pythonpath(repo_root)

    completed = subprocess.run(
        [python, "-m", "pfe_cli.main", *args],
        cwd=str(cwd),
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        command = "pfe " + " ".join(args)
        raise AssertionError(
            f"{command} failed with exit code {completed.returncode}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return completed.stdout


def _require(text: str, expected: str, *, label: str) -> None:
    if expected not in text:
        raise AssertionError(f"{label} did not contain {expected!r}\n{text}")


def _latest_queue_adapter_version(*, workdir: Path, workspace: str) -> str:
    queue_path = workdir / ".pfe" / "data" / f"train_queue_{workspace}.json"
    if not queue_path.is_file():
        raise AssertionError(f"train queue file was not created: {queue_path}")
    queue_payload = json.loads(queue_path.read_text(encoding="utf-8"))
    last_item = queue_payload.get("last_item") or {}
    version = str(last_item.get("adapter_version") or "")
    if ADAPTER_VERSION_RE.match(version) is None:
        raise AssertionError(f"train queue did not record an adapter version\n{queue_path.read_text(encoding='utf-8')}")
    return version


def _verify_queue_completion_artifacts(*, workdir: Path, workspace: str, version: str, base_model: str) -> dict[str, str]:
    config_path = workdir / ".pfe" / "config.toml"
    manifest_path = workdir / ".pfe" / "adapters" / workspace / version / "adapter_manifest.json"
    queue_path = workdir / ".pfe" / "data" / f"train_queue_{workspace}.json"

    if not config_path.is_file():
        raise AssertionError(f"config file was not created: {config_path}")
    if not manifest_path.is_file():
        raise AssertionError(f"adapter manifest was not created: {manifest_path}")
    if not queue_path.is_file():
        raise AssertionError(f"train queue file was not created: {queue_path}")

    queue_payload = json.loads(queue_path.read_text(encoding="utf-8"))
    last_item = queue_payload.get("last_item") or {}
    if last_item.get("state") != "completed":
        raise AssertionError(f"train queue did not complete the latest item\n{queue_path.read_text(encoding='utf-8')}")
    if last_item.get("adapter_version") != version:
        raise AssertionError(
            "train queue latest adapter version did not match processed version: "
            f"expected {version!r}, got {last_item.get('adapter_version')!r}"
        )
    history_events = [str(item.get("event") or "") for item in last_item.get("history") or []]
    for event in ("enqueued", "running", "completed"):
        if event not in history_events:
            raise AssertionError(f"train queue history did not include {event!r}: {history_events}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("base_model") != base_model:
        raise AssertionError(
            "queue training did not use the initialized base_model: "
            f"expected {base_model!r}, got {manifest.get('base_model')!r}"
        )
    manifest_text = manifest_path.read_text(encoding="utf-8")
    if "mock_local" not in manifest_text:
        raise AssertionError(f"queue training did not record the mock_local backend\n{manifest_text}")

    return {
        "config_path": str(config_path),
        "manifest_path": str(manifest_path),
        "queue_path": str(queue_path),
    }


def _run_smoke(args: argparse.Namespace, workdir: Path) -> dict[str, str]:
    workspace = args.workspace
    base_model = "./models/local-base"
    (workdir / "models" / "local-base").mkdir(parents=True, exist_ok=True)
    (workdir / "models" / "local-base" / "config.json").write_text(
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

    def run(command_args: list[str]) -> str:
        print("$ pfe " + " ".join(command_args))
        output = _run_cli(
            python=args.python,
            repo_root=args.repo_root,
            cwd=workdir,
            args=command_args,
            timeout=args.timeout,
        )
        clean = _strip_ansi(output)
        print(clean.rstrip())
        print()
        return clean

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
    _require(doctor_output, "PFE doctor", label="doctor output")
    _require(doctor_output, "local model: available=yes", label="doctor output")
    _require(doctor_output, f"requested_base_model={base_model}", label="doctor output")
    _require(doctor_output, f"pfe next --workspace {workspace}", label="doctor output")

    first_next_output = run(["next", "--workspace", workspace])
    _require(first_next_output, "PFE next", label="first next output")
    _require(first_next_output, "state: collect_feedback", label="first next output")
    _require(first_next_output, "pfe generate --scenario life-coach", label="first next output")

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

    trigger_config_output = run(
        [
            "trigger",
            "configure",
            "--workspace",
            workspace,
            "--enable",
            "--min-new-samples",
            "1",
            "--queue-mode",
            "deferred",
            "--max-interval-days",
            "0",
            "--no-require-confirmation",
            "--epochs",
            "1",
            "--backend",
            "mock_local",
        ]
    )
    _require(trigger_config_output, "[ AUTO TRAIN ACTION ]", label="trigger configure output")
    _require(trigger_config_output, "configure", label="trigger configure output")
    _require(trigger_config_output, "queue mode", label="trigger configure output")
    _require(trigger_config_output, "deferred", label="trigger configure output")
    _require(trigger_config_output, "mock_local", label="trigger configure output")

    signal_output = run(
        [
            "collect",
            "ingest",
            "--workspace",
            workspace,
            "--event-id",
            "evt-first-run-feedback-1",
            "--request-id",
            "req-first-run-feedback-1",
            "--session-id",
            "sess-first-run-feedback-1",
            "--source-event-id",
            "evt-first-run-chat-1",
            "--user-input",
            "Help me choose one next step for today.",
            "--model-output",
            "Pick one task you can finish in the next 20 minutes.",
            "--action",
            "accept",
            "--scenario",
            "life-coach",
        ]
    )
    _require(signal_output, "Signal ingested", label="collect ingest output")
    _require(signal_output, "Signal ID: evt-first-run-feedback-1", label="collect ingest output")
    _require(signal_output, "Recorded: True", label="collect ingest output")
    _require(signal_output, "Event Chain Complete: True", label="collect ingest output")
    _require(signal_output, "Curated Samples: 1", label="collect ingest output")
    _require(signal_output, "Auto Train: queued (enqueued)", label="collect ingest output")

    collect_status_output = run(["collect", "status", "--workspace", workspace])
    _require(collect_status_output, "Total Signals: 1", label="collect status output")
    _require(collect_status_output, "Curated Samples: 1", label="collect status output")
    _require(collect_status_output, "Dataset Splits: train=1", label="collect status output")
    _require(collect_status_output, "Latest Signal: evt-first-run-feedback-1", label="collect status output")

    collect_review_output = run(["collect", "review", "--workspace", workspace, "--type", "accept", "--limit", "5"])
    _require(collect_review_output, "Signal ID: evt-first-run-feedback-1", label="collect review output")
    _require(collect_review_output, "Type: accept", label="collect review output")
    _require(collect_review_output, "Confidence: 0.90", label="collect review output")
    _require(collect_review_output, "Session: sess-first-run-feedback-1", label="collect review output")

    queue_next_output = run(["next", "--workspace", workspace])
    _require(queue_next_output, "state: queue_ready", label="queue next output")
    _require(queue_next_output, f"pfe trigger process-next --workspace {workspace}", label="queue next output")

    trigger_status_output = run(["trigger", "status", "--workspace", workspace])
    _require(trigger_status_output, "[ AUTO TRAIN TRIGGER ]", label="trigger status output")
    _require(trigger_status_output, "last result", label="trigger status output")
    _require(trigger_status_output, "queued", label="trigger status output")

    process_output = run(["trigger", "process-next", "--workspace", workspace])
    _require(process_output, "[ AUTO TRAIN ACTION ]", label="trigger process-next output")
    _require(process_output, "process_next", label="trigger process-next output")
    _require(process_output, "triggered", label="trigger process-next output")
    _require(process_output, "[ TRAIN QUEUE ]", label="trigger process-next output")
    _require(process_output, "completed", label="trigger process-next output")
    version = _latest_queue_adapter_version(workdir=workdir, workspace=workspace)
    print(f"queue adapter version: {version}")
    print()
    artifacts = _verify_queue_completion_artifacts(
        workdir=workdir,
        workspace=workspace,
        version=version,
        base_model=base_model,
    )

    candidate_next_output = run(["next", "--workspace", workspace])
    _require(candidate_next_output, "state: evaluate_candidate", label="candidate next output")
    _require(candidate_next_output, f"pfe eval --base-model base --adapter {version}", label="candidate next output")

    if args.stop_after == "queue":
        return {
            "base_model": base_model,
            "config_path": artifacts["config_path"],
            "manifest_path": artifacts["manifest_path"],
            "queue_path": artifacts["queue_path"],
            "scope": "auto_train_queue",
            "version": version,
            "workspace": workspace,
        }

    eval_output = run(
        [
            "eval",
            "--base-model",
            "base",
            "--adapter",
            version,
            "--num-samples",
            "3",
            "--workspace",
            workspace,
        ]
    )
    _require(eval_output, "[ EVALUATION RESULT ]", label="eval output")
    _require(eval_output, version, label="eval output")
    _require(eval_output, "recommendation:", label="eval output")

    promote_output = run(["adapter", "promote", version, "--workspace", workspace])
    _require(promote_output, "latest:", label="promote output")
    _require(promote_output, version, label="promote output")

    serve_output = run(
        [
            "serve",
            "--host",
            "127.0.0.1",
            "--port",
            str(args.port),
            "--workspace",
            workspace,
        ]
    )
    _require(serve_output, "[ SERVE PREVIEW ]", label="serve output")
    _require(serve_output, f"workspace:               {workspace}", label="serve output")
    _require(serve_output, "[ LATEST PROMOTED ]", label="serve output")
    _require(serve_output, version, label="serve output")
    _require(serve_output, "preview only", label="serve output")
    _require(serve_output, "dry_run=True", label="serve output")

    return {
        "base_model": base_model,
        "config_path": artifacts["config_path"],
        "manifest_path": artifacts["manifest_path"],
        "queue_path": artifacts["queue_path"],
        "scope": "full",
        "version": version,
        "workspace": workspace,
    }


def main() -> int:
    repo_root = _repo_root()
    parser = argparse.ArgumentParser(
        description=(
            "Smoke-test a first-run local PFE path without network, real model downloads, "
            "or long-lived servers."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--python", default=_default_python(repo_root))
    parser.add_argument("--workspace", default="first_run")
    parser.add_argument("--port", type=int, default=8921)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--workdir", type=Path, default=None)
    parser.add_argument("--keep-workdir", action="store_true")
    parser.add_argument(
        "--stop-after",
        choices=("queue", "full"),
        default="full",
        help="Stop after the auto-train queue smoke or continue through eval/promote/serve.",
    )
    args = parser.parse_args()
    args.repo_root = args.repo_root.resolve()

    if args.num_samples < 4:
        raise SystemExit("--num-samples must be at least 4 so train/val/test splits are populated")

    tempdir = None
    if args.workdir is None:
        tempdir = tempfile.TemporaryDirectory(prefix="pfe-first-run-smoke-")
        workdir = Path(tempdir.name)
    else:
        workdir = args.workdir.resolve()
        workdir.mkdir(parents=True, exist_ok=True)

    print(f"workdir: {workdir}")
    print(f"python:  {args.python}")
    print()
    try:
        summary = _run_smoke(args, workdir)
        if summary["scope"] == "auto_train_queue":
            print("AUTO-TRAIN QUEUE SMOKE PASSED")
        else:
            print("FIRST-RUN SMOKE PASSED")
        print(f"workspace: {summary['workspace']}")
        print(f"version:   {summary['version']}")
        print(f"base_model: {summary['base_model']}")
        print(f"config:   {summary['config_path']}")
        print(f"manifest: {summary['manifest_path']}")
        print(f"queue:    {summary['queue_path']}")
        return 0
    finally:
        if tempdir is not None and not args.keep_workdir:
            tempdir.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
