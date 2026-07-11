#!/usr/bin/env python3
"""Run an isolated live-server smoke against real HTTP endpoints."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from first_run_smoke import (
    _default_python,
    _latest_queue_adapter_version,
    _pythonpath,
    _repo_root,
    _require,
    _run_cli,
    _strip_ansi,
    _verify_queue_completion_artifacts,
)


def _free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


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


def _prepare_workspace(args: argparse.Namespace, workdir: Path) -> dict[str, str]:
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

    doctor_output = run(["doctor", "--workspace", workspace])
    _require(doctor_output, "local model: available=yes", label="doctor output")

    generate_output = run(["generate", "--scenario", "life-coach", "--style", "warm", "--num", "8", "--workspace", workspace])
    _require(generate_output, "Saved 8 distilled sample(s)", label="generate output")

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

    signal_output = run(
        [
            "collect",
            "ingest",
            "--workspace",
            workspace,
            "--event-id",
            "evt-server-live-feedback-1",
            "--request-id",
            "req-server-live-feedback-1",
            "--session-id",
            "sess-server-live-feedback-1",
            "--source-event-id",
            "evt-server-live-chat-1",
            "--user-input",
            "Help me pick a focused next step.",
            "--model-output",
            "Choose one task that can be completed in 20 minutes.",
            "--action",
            "accept",
            "--scenario",
            "life-coach",
        ]
    )
    _require(signal_output, "Auto Train: queued (enqueued)", label="collect ingest output")

    process_output = run(["trigger", "process-next", "--workspace", workspace])
    _require(process_output, "completed", label="trigger process-next output")
    version = _latest_queue_adapter_version(workdir=workdir, workspace=workspace)
    artifacts = _verify_queue_completion_artifacts(
        workdir=workdir,
        workspace=workspace,
        version=version,
        base_model=base_model,
    )

    eval_output = run(["eval", "--base-model", "base", "--adapter", version, "--num-samples", "3", "--workspace", workspace])
    _require(eval_output, "[ EVALUATION RESULT ]", label="eval output")

    promote_output = run(["adapter", "promote", version, "--workspace", workspace])
    _require(promote_output, version, label="promote output")

    return {
        "base_model": base_model,
        "config_path": artifacts["config_path"],
        "manifest_path": artifacts["manifest_path"],
        "queue_path": artifacts["queue_path"],
        "version": version,
        "workspace": workspace,
    }


def _request_json(url: str, *, method: str = "GET", body: dict[str, object] | None = None, timeout: float = 5.0) -> dict[str, object]:
    data = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
    headers = {"accept": "application/json"}
    if data is not None:
        headers["content-type"] = "application/json"
    request = Request(url, data=data, headers=headers, method=method)
    with urlopen(request, timeout=timeout) as response:
        payload = response.read().decode("utf-8")
        return {
            "status": int(response.status),
            "body": json.loads(payload) if payload else None,
        }


def _request_text(url: str, *, timeout: float = 5.0) -> dict[str, object]:
    request = Request(url, headers={"accept": "text/html"}, method="GET")
    with urlopen(request, timeout=timeout) as response:
        payload = response.read().decode("utf-8", errors="replace")
        return {"status": int(response.status), "body": payload}


def _request_sse(url: str, *, body: dict[str, object], timeout: float = 5.0) -> dict[str, object]:
    request = Request(
        url,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"accept": "text/event-stream", "content-type": "application/json"},
        method="POST",
    )
    events: list[dict[str, object]] = []
    done = False
    with urlopen(request, timeout=timeout) as response:
        content_type = str(response.headers.get("content-type") or "")
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                done = True
                break
            payload = json.loads(data)
            if isinstance(payload, dict):
                events.append(payload)
        return {
            "status": int(response.status),
            "content_type": content_type,
            "events": events,
            "done": done,
        }


def _request_openai_sdk_stream(base_url: str, *, timeout: float = 5.0) -> dict[str, object]:
    from openai import OpenAI

    client = OpenAI(base_url=f"{base_url}/v1", api_key="pfe-local-smoke", timeout=timeout)
    content_parts: list[str] = []
    finish_reason = None
    chunk_count = 0
    stream = client.chat.completions.create(
        model="local",
        messages=[{"role": "user", "content": "Give me one SDK-streamed next step."}],
        stream=True,
    )
    for chunk in stream:
        chunk_count += 1
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta.content:
            content_parts.append(delta.content)
        if chunk.choices[0].finish_reason:
            finish_reason = chunk.choices[0].finish_reason
    return {
        "chunk_count": chunk_count,
        "content": "".join(content_parts),
        "finish_reason": finish_reason,
    }


def _start_server(args: argparse.Namespace, workdir: Path, port: int) -> subprocess.Popen[str]:
    env = os.environ.copy()
    env.pop("PFE_HOME", None)
    env["PYTHONPATH"] = _pythonpath(args.repo_root)
    command = [
        args.python,
        "-m",
        "pfe_cli.main",
        "serve",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--workspace",
        args.workspace,
        "--live",
    ]
    return subprocess.Popen(
        command,
        cwd=str(workdir),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _stop_server(process: subprocess.Popen[str]) -> tuple[str, str]:
    if process.poll() is None:
        process.terminate()
    try:
        stdout, stderr = process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate(timeout=5)
    return stdout or "", stderr or ""


def _wait_for_healthz(base_url: str, process: subprocess.Popen[str], *, timeout_seconds: float) -> dict[str, object]:
    deadline = time.monotonic() + timeout_seconds
    last_error = "server did not respond"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stdout, stderr = process.communicate(timeout=1)
            raise AssertionError(
                f"server exited before healthz was ready (code={process.returncode})\n"
                f"stdout:\n{stdout}\n"
                f"stderr:\n{stderr}"
            )
        try:
            result = _request_json(f"{base_url}/healthz", timeout=1.0)
            if result["status"] == 200:
                return result
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            last_error = f"{exc.__class__.__name__}: {exc}"
        time.sleep(0.25)
    raise AssertionError(f"healthz was not ready after {timeout_seconds:.1f}s: {last_error}")


def _run_smoke(args: argparse.Namespace, workdir: Path) -> dict[str, str]:
    setup = _prepare_workspace(args, workdir)
    port = args.port if args.port else _free_loopback_port()
    base_url = f"http://127.0.0.1:{port}"
    process = _start_server(args, workdir, port)
    try:
        healthz = _wait_for_healthz(base_url, process, timeout_seconds=args.server_timeout)
        if healthz["body"].get("status") != "ok":  # type: ignore[union-attr]
            raise AssertionError(f"unexpected healthz payload: {healthz}")

        status = _request_json(f"{base_url}/pfe/status?detail=full", timeout=args.request_timeout)
        if status["status"] != 200:
            raise AssertionError(f"unexpected /pfe/status response: {status}")
        status_body = status["body"]
        if not isinstance(status_body, dict):
            raise AssertionError(f"/pfe/status did not return an object: {status_body!r}")
        latest = status_body.get("latest_adapter") or {}
        if not isinstance(latest, dict) or latest.get("version") != setup["version"]:
            raise AssertionError(f"/pfe/status did not expose promoted adapter {setup['version']}: {status_body}")

        dashboard = _request_text(f"{base_url}/dashboard", timeout=args.request_timeout)
        dashboard_body = str(dashboard["body"])
        if dashboard["status"] != 200 or "PFE" not in dashboard_body:
            raise AssertionError(f"dashboard HTML did not look ready: status={dashboard['status']}")

        dashboard_metrics = _request_json(f"{base_url}/pfe/dashboard/metrics", timeout=args.request_timeout)
        if dashboard_metrics["status"] != 200 or not isinstance(dashboard_metrics["body"], dict):
            raise AssertionError(f"dashboard metrics were not ready: {dashboard_metrics}")

        chat = _request_json(
            f"{base_url}/v1/chat/completions",
            method="POST",
            body={
                "model": "local",
                "messages": [{"role": "user", "content": "Give me one focused next step."}],
                "stream": False,
            },
            timeout=args.request_timeout,
        )
        chat_body = chat["body"]
        if chat["status"] != 200 or not isinstance(chat_body, dict) or not chat_body.get("choices"):
            raise AssertionError(f"chat completion response was not ready: {chat}")

        stream = _request_sse(
            f"{base_url}/v1/chat/completions",
            body={
                "model": "local",
                "messages": [{"role": "user", "content": "Give me one streamed next step."}],
                "stream": True,
            },
            timeout=args.request_timeout,
        )
        events = stream.get("events") or []
        if stream["status"] != 200 or "text/event-stream" not in str(stream["content_type"]):
            raise AssertionError(f"streaming response was not SSE: {stream}")
        if not stream["done"] or not isinstance(events, list) or len(events) < 2:
            raise AssertionError(f"streaming response did not terminate with [DONE]: {stream}")
        choices = [choice for event in events for choice in event.get("choices", []) if isinstance(choice, dict)]
        if not any((choice.get("delta") or {}).get("role") == "assistant" for choice in choices):
            raise AssertionError(f"streaming response did not emit an assistant role chunk: {stream}")
        if not any(choice.get("finish_reason") in {"stop", "length"} for choice in choices):
            raise AssertionError(f"streaming response did not emit a finish reason: {stream}")

        sdk_stream = _request_openai_sdk_stream(base_url, timeout=args.request_timeout)
        if not sdk_stream["content"] or sdk_stream["finish_reason"] not in {"stop", "length"}:
            raise AssertionError(f"OpenAI SDK could not consume the PFE stream: {sdk_stream}")

    finally:
        stdout, stderr = _stop_server(process)

    if process.returncode not in {0, -15, -9, None}:
        raise AssertionError(
            f"server exited unexpectedly after smoke (code={process.returncode})\n"
            f"stdout:\n{stdout}\n"
            f"stderr:\n{stderr}"
        )

    return {
        **setup,
        "base_url": base_url,
        "port": str(port),
        "sdk_stream_chunk_count": str(sdk_stream["chunk_count"]),
        "sdk_stream_finish_reason": str(sdk_stream["finish_reason"]),
        "sdk_stream_content": str(sdk_stream["content"]),
    }


def main() -> int:
    repo_root = _repo_root()
    parser = argparse.ArgumentParser(
        description=(
            "Build an isolated mock-local adapter, launch pfe serve --live on loopback, "
            "and verify healthz, status, dashboard, dashboard metrics, JSON chat, and OpenAI SSE over HTTP."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--python", default=_default_python(repo_root))
    parser.add_argument("--workspace", default="server_live")
    parser.add_argument("--port", type=int, default=0, help="Port to bind. Defaults to a free loopback port.")
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--server-timeout", type=float, default=15.0)
    parser.add_argument("--request-timeout", type=float, default=5.0)
    parser.add_argument("--workdir", type=Path, default=None)
    parser.add_argument("--keep-workdir", action="store_true")
    args = parser.parse_args()
    args.repo_root = args.repo_root.resolve()

    tempdir = None
    if args.workdir is None:
        tempdir = tempfile.TemporaryDirectory(prefix="pfe-server-live-smoke-")
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
        print("SERVER LIVE SMOKE PASSED")
        print(f"workspace: {summary['workspace']}")
        print(f"version:   {summary['version']}")
        print(f"base_url:  {summary['base_url']}")
        print(f"manifest:  {summary['manifest_path']}")
        print(f"sdk_chunks: {summary['sdk_stream_chunk_count']}")
        print(f"sdk_finish: {summary['sdk_stream_finish_reason']}")
        print(f"sdk_content: {summary['sdk_stream_content']}")
        return 0
    finally:
        if tempdir is not None and not args.keep_workdir:
            tempdir.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
