#!/usr/bin/env python3
"""Run an isolated live smoke for dashboard, Studio, and chat feedback APIs."""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

from first_run_smoke import _default_python, _repo_root, _require
from server_live_smoke import (
    _free_loopback_port,
    _prepare_workspace,
    _request_json,
    _request_text,
    _start_server,
    _stop_server,
    _wait_for_healthz,
)


def _require_json_response(result: dict[str, object], *, label: str) -> object:
    if result.get("status") != 200 or result.get("body") is None:
        raise AssertionError(f"{label} did not return a 200 JSON payload: {result}")
    return result["body"]  # type: ignore[return-value]


def _require_dict_response(result: dict[str, object], *, label: str) -> dict[str, object]:
    body = _require_json_response(result, label=label)
    if not isinstance(body, dict):
        raise AssertionError(f"{label} did not return a JSON object: {result}")
    return body


def _check_dashboard_surface(base_url: str, *, request_timeout: float) -> None:
    dashboard = _request_text(f"{base_url}/dashboard", timeout=request_timeout)
    dashboard_body = str(dashboard["body"])
    if dashboard["status"] != 200:
        raise AssertionError(f"dashboard HTML returned unexpected status: {dashboard['status']}")
    for expected in (
        "PFE Observability Dashboard",
        "const API_BASE = '/pfe/dashboard'",
        "fetch(`${API_BASE}/metrics`)",
        "window.Chart = OfflineChart",
        "id=\"refreshBtn\"",
    ):
        _require(dashboard_body, expected, label="dashboard HTML")
    if "https://" in dashboard_body or "http://" in dashboard_body:
        raise AssertionError("dashboard HTML includes external resource URLs")

    for path in (
        "/pfe/dashboard/metrics",
        "/pfe/dashboard/training",
        "/pfe/dashboard/signals",
        "/pfe/dashboard/adapters",
        "/pfe/dashboard/health",
    ):
        _require_json_response(
            _request_json(f"{base_url}{path}", timeout=request_timeout),
            label=path,
        )


def _check_studio_surface(base_url: str, *, request_timeout: float) -> None:
    studio = _request_text(f"{base_url}/", timeout=request_timeout)
    studio_body = str(studio["body"])
    if studio["status"] != 200:
        raise AssertionError(f"Studio HTML returned unexpected status: {studio['status']}")
    for expected in (
        "PFE / 本地模型工作台",
        "选择模型，拿到本机 API。",
        "复制 API 地址",
        "/pfe/runtime",
        "/pfe/models",
        "/pfe/training/jobs",
    ):
        _require(studio_body, expected, label="Studio HTML")

    chat_body = _require_dict_response(
        _request_json(
            f"{base_url}/v1/chat/completions",
            method="POST",
            body={
                "model": "local",
                "messages": [{"role": "user", "content": "Give me one beta validation step."}],
                "stream": False,
            },
            timeout=request_timeout,
        ),
        label="/v1/chat/completions",
    )
    if not chat_body.get("choices"):
        raise AssertionError(f"chat completion did not include choices: {chat_body}")
    session_id = str(chat_body.get("session_id") or "")
    request_id = str(chat_body.get("request_id") or "")
    if not session_id or not request_id:
        raise AssertionError(f"chat completion did not expose session/request ids: {chat_body}")

    assistant_message = ""
    choices = chat_body.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict):
                assistant_message = str(message.get("content") or "")

    feedback_body = _require_dict_response(
        _request_json(
            f"{base_url}/pfe/feedback",
            method="POST",
            body={
                "session_id": session_id,
                "request_id": request_id,
                "action": "accept",
                "response_time_seconds": 0.25,
                "user_message": "Give me one beta validation step.",
                "assistant_message": assistant_message,
            },
            timeout=request_timeout,
        ),
        label="/pfe/feedback",
    )
    if feedback_body.get("success") is not True or feedback_body.get("signal_type") != "accept":
        raise AssertionError(f"feedback round trip did not record an accept signal: {feedback_body}")


def _run_smoke(args: argparse.Namespace, workdir: Path) -> dict[str, str]:
    setup = _prepare_workspace(args, workdir)
    port = args.port if args.port else _free_loopback_port()
    base_url = f"http://127.0.0.1:{port}"
    process = _start_server(args, workdir, port)
    try:
        healthz = _wait_for_healthz(base_url, process, timeout_seconds=args.server_timeout)
        if healthz["body"].get("status") != "ok":  # type: ignore[union-attr]
            raise AssertionError(f"unexpected healthz payload: {healthz}")
        _check_dashboard_surface(base_url, request_timeout=args.request_timeout)
        _check_studio_surface(base_url, request_timeout=args.request_timeout)
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
    }


def main() -> int:
    repo_root = _repo_root()
    parser = argparse.ArgumentParser(
        description=(
            "Build an isolated mock-local adapter, launch pfe serve --live, and verify "
            "dashboard HTML/API plus Studio root HTML and chat/feedback round trip over real HTTP."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--python", default=_default_python(repo_root))
    parser.add_argument("--workspace", default="dashboard_console_live")
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
        tempdir = tempfile.TemporaryDirectory(prefix="pfe-dashboard-console-live-smoke-")
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
        print("DASHBOARD CONSOLE LIVE SMOKE PASSED")
        print(f"workspace:     {summary['workspace']}")
        print(f"version:       {summary['version']}")
        print(f"base_url:      {summary['base_url']}")
        print("dashboard_api: ok")
        print("chat_feedback: ok")
        return 0
    finally:
        if tempdir is not None and not args.keep_workdir:
            tempdir.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
