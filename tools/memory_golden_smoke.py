#!/usr/bin/env python3
"""Run the real-local memory golden smoke in an isolated workspace."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from first_run_smoke import ADAPTER_VERSION_RE, _default_python, _repo_root, _strip_ansi


DEFAULT_MEMORY_PROMPT = "PFE Golden Smoke 记忆代号是什么？只回答代号。"
DEFAULT_MEMORY_ANSWER = "金线闭环-042"
DEFAULT_MODEL_DIRNAME = "Qwen2.5-0.5B-Instruct"
REQUIRED_MODULES = ("torch", "transformers", "peft", "accelerate")


def _pythonpath(repo_root: Path) -> str:
    parts = [
        str(repo_root / "pfe-core"),
        str(repo_root / "pfe-cli"),
        str(repo_root / "pfe-server"),
    ]
    existing = os.environ.get("PYTHONPATH")
    if existing:
        parts.append(existing)
    return os.pathsep.join(parts)


def _missing_modules() -> list[str]:
    missing: list[str] = []
    for name in REQUIRED_MODULES:
        try:
            __import__(name)
        except Exception:
            missing.append(name)
    return missing


def _looks_quantized_training_path(path: Path) -> bool:
    normalized = str(path).lower()
    return "4bit" in normalized or "gguf" in normalized


def resolve_base_model(args: argparse.Namespace, repo_root: Path) -> Path | None:
    raw = (
        args.base_model
        or os.environ.get("PFE_GOLDEN_SMOKE_MODEL")
        or os.environ.get("PFE_REAL_LOCAL_MODEL")
    )
    if raw:
        path = Path(str(raw)).expanduser()
    else:
        path = repo_root / "models" / DEFAULT_MODEL_DIRNAME
    if not path.exists():
        if args.base_model or os.environ.get("PFE_GOLDEN_SMOKE_MODEL") or os.environ.get("PFE_REAL_LOCAL_MODEL"):
            raise AssertionError(f"local model path does not exist: {path}")
        return None
    if _looks_quantized_training_path(path):
        raise AssertionError(
            "memory golden smoke requires an unquantized Hugging Face model directory; "
            f"rejected training path: {path}"
        )
    if not (path / "config.json").is_file():
        raise AssertionError(f"local model path must be a Hugging Face model directory with config.json: {path}")
    return path.resolve()


def _run_cli(
    *,
    python: str,
    repo_root: Path,
    cwd: Path,
    pfe_home: Path,
    workspace: str,
    args: list[str],
    timeout: int,
) -> str:
    env = os.environ.copy()
    env["PFE_HOME"] = str(pfe_home)
    env["PFE_WORKSPACE"] = workspace
    env["PFE_ENABLE_REAL_LOCAL_INFERENCE"] = "1"
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
    return _strip_ansi(completed.stdout)


def latest_adapter_version(pfe_home: Path, workspace: str) -> str:
    adapter_root = pfe_home / "adapters" / workspace
    if not adapter_root.is_dir():
        raise AssertionError(f"adapter root was not created: {adapter_root}")
    versions = sorted(path.name for path in adapter_root.iterdir() if path.is_dir() and ADAPTER_VERSION_RE.match(path.name))
    if not versions:
        raise AssertionError(f"no adapter version directories were created under {adapter_root}")
    return versions[-1]


def read_manifest(pfe_home: Path, workspace: str, version: str) -> dict[str, Any]:
    manifest_path = pfe_home / "adapters" / workspace / version / "adapter_manifest.json"
    if not manifest_path.is_file():
        raise AssertionError(f"adapter manifest was not created: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def normalize_answer(text: str) -> str:
    value = str(text or "").strip()
    value = re.sub(r"^[`'\"“”‘’\s]+|[`'\"“”‘’\s]+$", "", value)
    return value.strip()


def chat_answer(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0] if isinstance(choices[0], dict) else {}
    message = first.get("message") if isinstance(first, dict) else {}
    if not isinstance(message, dict):
        return ""
    return str(message.get("content") or "")


def _chat(
    *,
    app: Any,
    model: str,
    prompt: str,
    timeout: int,
) -> dict[str, Any]:
    from pfe_server.app import smoke_test_request

    async def scenario() -> dict[str, Any]:
        return await smoke_test_request(
            app,
            path="/v1/chat/completions",
            method="POST",
            body={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": 32,
                "metadata": {
                    "enable_real_local": True,
                    "source": "pfe_memory_golden_smoke",
                },
            },
        )

    return asyncio.run(asyncio.wait_for(scenario(), timeout=timeout))


def verify_base_vs_local(
    *,
    repo_root: Path,
    pfe_home: Path,
    workspace: str,
    prompt: str,
    expected_answer: str,
    timeout: int,
) -> dict[str, Any]:
    sys.path[:0] = [str(repo_root / "pfe-core"), str(repo_root / "pfe-cli"), str(repo_root / "pfe-server")]
    os.environ["PFE_HOME"] = str(pfe_home)
    os.environ["PFE_WORKSPACE"] = workspace
    os.environ["PFE_ENABLE_REAL_LOCAL_INFERENCE"] = "1"

    from pfe_server.app import build_serve_plan

    plan = build_serve_plan(workspace=workspace, dry_run=True)
    app = plan.app

    base_result = _chat(app=app, model="base", prompt=prompt, timeout=timeout)
    local_result = _chat(app=app, model="local", prompt=prompt, timeout=timeout)

    if base_result["status_code"] != 200:
        raise AssertionError(f"model=base returned status {base_result['status_code']}: {base_result}")
    if local_result["status_code"] != 200:
        raise AssertionError(f"model=local returned status {local_result['status_code']}: {local_result}")

    base_payload = dict(base_result["body"])
    local_payload = dict(local_result["body"])
    base_answer = chat_answer(base_payload)
    local_answer = chat_answer(local_payload)
    expected = normalize_answer(expected_answer)
    normalized_local = normalize_answer(local_answer)
    if expected in base_answer:
        raise AssertionError(f"model=base unexpectedly revealed the memory answer: {base_answer!r}")
    if normalized_local != expected:
        raise AssertionError(
            "model=local did not answer the memory code exactly\n"
            f"expected: {expected!r}\n"
            f"actual:   {normalized_local!r}\n"
            f"raw:      {local_answer!r}"
        )
    inference = local_payload.get("metadata", {}).get("inference", {})
    if not isinstance(inference, dict) or not inference.get("adapter_requested"):
        raise AssertionError(f"model=local did not report an adapter-loaded inference path: {inference}")
    return {
        "base": {
            "answer": base_answer,
            "metadata": base_payload.get("metadata", {}),
            "served_by": base_payload.get("served_by"),
            "adapter_version": base_payload.get("adapter_version"),
        },
        "local": {
            "answer": local_answer,
            "metadata": local_payload.get("metadata", {}),
            "served_by": local_payload.get("served_by"),
            "adapter_version": local_payload.get("adapter_version"),
        },
    }


def _run_smoke(args: argparse.Namespace, workdir: Path, pfe_home: Path, base_model: Path) -> dict[str, Any]:
    workspace = args.workspace
    model_value = str(base_model)
    prompt = args.prompt
    answer = args.answer

    _run_cli(
        python=args.python,
        repo_root=args.repo_root,
        cwd=workdir,
        pfe_home=pfe_home,
        workspace=workspace,
        args=["init", "--workspace", workspace, "--base-model", model_value, "--home", str(pfe_home)],
        timeout=args.timeout,
    )
    signal_output = _run_cli(
        python=args.python,
        repo_root=args.repo_root,
        cwd=workdir,
        pfe_home=pfe_home,
        workspace=workspace,
        args=[
            "collect",
            "ingest",
            "--workspace",
            workspace,
            "--event-id",
            "evt-memory-golden-1",
            "--request-id",
            "req-memory-golden-1",
            "--session-id",
            "sess-memory-golden-1",
            "--source-event-id",
            "evt-memory-golden-chat-1",
            "--user-input",
            prompt,
            "--model-output",
            answer,
            "--action",
            "accept",
            "--confidence",
            "0.99",
            "--scenario",
            "memory-golden-smoke",
        ],
        timeout=args.timeout,
    )
    if "Curated Samples: 1" not in signal_output:
        raise AssertionError(f"memory sample was not curated as one train sample:\n{signal_output}")

    train_output = _run_cli(
        python=args.python,
        repo_root=args.repo_root,
        cwd=workdir,
        pfe_home=pfe_home,
        workspace=workspace,
        args=[
            "train",
            "--workspace",
            workspace,
            "--base-model",
            model_value,
            "--backend",
            "peft",
            "--real-local",
            "--epochs",
            str(args.epochs),
        ],
        timeout=args.train_timeout,
    )
    if "TRAINING COMPLETE" not in train_output:
        raise AssertionError(f"training did not complete:\n{train_output}")

    version = latest_adapter_version(pfe_home, workspace)
    manifest = read_manifest(pfe_home, workspace, version)
    if manifest.get("state") not in {"pending_eval", "promoted"}:
        raise AssertionError(f"adapter should be ready for eval/promotion, got manifest state={manifest.get('state')}")
    if manifest.get("artifact_format") != "peft_lora":
        raise AssertionError(f"memory golden smoke expects a PEFT LoRA adapter, got {manifest.get('artifact_format')}")

    sys.path[:0] = [str(args.repo_root / "pfe-core"), str(args.repo_root / "pfe-cli"), str(args.repo_root / "pfe-server")]
    os.environ["PFE_HOME"] = str(pfe_home)
    os.environ["PFE_WORKSPACE"] = workspace
    from pfe_core.adapter_store import create_adapter_store

    store = create_adapter_store(workspace=workspace)
    store.attach_eval_report(
        version,
        {
            "recommendation": "deploy",
            "comparison": "memory_golden_verified",
            "scores": {
                "memory_exact_match": 1.0,
                "quality_preservation": 1.0,
            },
            "details": [{"prompt": prompt, "expected": answer}],
        },
    )
    store.promote(version)

    verification = verify_base_vs_local(
        repo_root=args.repo_root,
        pfe_home=pfe_home,
        workspace=workspace,
        prompt=prompt,
        expected_answer=answer,
        timeout=args.inference_timeout,
    )
    return {
        "workspace": workspace,
        "base_model": model_value,
        "version": version,
        "prompt": prompt,
        "expected_answer": answer,
        "manifest": {
            "artifact_format": manifest.get("artifact_format"),
            "base_model": manifest.get("base_model"),
            "state_before_promotion": manifest.get("state"),
        },
        "verification": verification,
    }


def main() -> int:
    repo_root = _repo_root()
    parser = argparse.ArgumentParser(
        description=(
            "Golden Smoke: create an isolated PFE home/workspace, ingest one memory sample, "
            "train the 0.5B model, then prove model=base does not know and model=local answers exactly."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--python", default=_default_python(repo_root))
    parser.add_argument("--workspace", default="memory_golden")
    parser.add_argument("--base-model", default=None, help="Unquantized local HF model directory. Defaults to models/Qwen2.5-0.5B-Instruct.")
    parser.add_argument("--prompt", default=DEFAULT_MEMORY_PROMPT)
    parser.add_argument("--answer", default=DEFAULT_MEMORY_ANSWER)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--train-timeout", type=int, default=180)
    parser.add_argument("--inference-timeout", type=int, default=180)
    parser.add_argument("--strict", action="store_true", help="Fail instead of skipping when the default model path is missing.")
    parser.add_argument("--workdir", type=Path, default=None)
    parser.add_argument("--keep-workdir", action="store_true")
    parser.add_argument("--report-path", type=Path, default=None)
    args = parser.parse_args()
    args.repo_root = args.repo_root.resolve()

    if args.epochs < 1:
        raise SystemExit("--epochs must be at least 1")

    try:
        base_model = resolve_base_model(args, args.repo_root)
    except AssertionError as exc:
        print("MEMORY GOLDEN SMOKE FAILED")
        print(f"reason: {exc}")
        return 2
    if base_model is None:
        print("MEMORY GOLDEN SMOKE SKIPPED")
        print(f"reason: default model path missing: {args.repo_root / 'models' / DEFAULT_MODEL_DIRNAME}")
        print("hint: set PFE_GOLDEN_SMOKE_MODEL to an unquantized local HF model directory")
        return 2 if args.strict else 0

    missing = _missing_modules()
    if missing:
        print("MEMORY GOLDEN SMOKE FAILED")
        print("reason: configured local model path requires training and inference runtime modules")
        print(f"missing: {', '.join(missing)}")
        return 2

    tempdir = None
    if args.workdir is None:
        tempdir = tempfile.TemporaryDirectory(prefix="pfe-memory-golden-")
        workdir = Path(tempdir.name)
    else:
        workdir = args.workdir.resolve()
        if workdir.exists():
            shutil.rmtree(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
    pfe_home = workdir / ".pfe"

    print(f"workdir: {workdir}")
    print(f"pfe_home: {pfe_home}")
    print(f"python:   {args.python}")
    print(f"model:    {base_model}")
    print(f"prompt:   {args.prompt}")
    print(f"answer:   {args.answer}")
    print()
    try:
        summary = _run_smoke(args, workdir, pfe_home, base_model)
        if args.report_path:
            args.report_path.parent.mkdir(parents=True, exist_ok=True)
            args.report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print("MEMORY GOLDEN SMOKE PASSED")
        print(f"workspace:       {summary['workspace']}")
        print(f"adapter_version: {summary['version']}")
        print(f"base_model:      {summary['base_model']}")
        print(f"base_answer:     {summary['verification']['base']['answer']}")
        print(f"local_answer:    {summary['verification']['local']['answer']}")
        if args.report_path:
            print(f"report:          {args.report_path}")
        return 0
    except Exception as exc:
        print("MEMORY GOLDEN SMOKE FAILED")
        print(f"reason: {exc}")
        return 1
    finally:
        if tempdir is not None and not args.keep_workdir:
            tempdir.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
