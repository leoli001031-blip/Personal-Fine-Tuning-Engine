#!/usr/bin/env python3
"""Smoke-test Studio from a clean editable install and first launch."""

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

from first_run_smoke import _default_python, _repo_root, _require


def _free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _venv_python(venv_dir: Path) -> Path:
    return venv_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _venv_script(venv_dir: Path, name: str) -> Path:
    suffix = ".exe" if os.name == "nt" else ""
    return venv_dir / ("Scripts" if os.name == "nt" else "bin") / f"{name}{suffix}"


def _run_checked(command: list[str], *, cwd: Path, timeout: int, env: dict[str, str] | None = None) -> str:
    completed = subprocess.run(
        command,
        cwd=str(cwd),
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"command failed with exit code {completed.returncode}: {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return completed.stdout


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


def _wait_for_text(url: str, *, timeout: float) -> dict[str, object]:
    deadline = time.time() + timeout
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            return _request_text(url)
        except (HTTPError, URLError, TimeoutError, ConnectionError) as exc:
            last_error = exc
            time.sleep(0.25)
    raise AssertionError(f"server did not become ready at {url}: {last_error}")


def _stop_process(process: subprocess.Popen[str]) -> tuple[str, str]:
    if process.poll() is None:
        process.terminate()
    try:
        stdout, stderr = process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate(timeout=5)
    return stdout or "", stderr or ""


def _install_clean(args: argparse.Namespace, workdir: Path) -> dict[str, Path]:
    venv_dir = workdir / "clean-venv"
    _run_checked([args.python, "-m", "venv", str(venv_dir)], cwd=args.repo_root, timeout=args.install_timeout)
    python_bin = _venv_python(venv_dir)
    _run_checked([str(python_bin), "-m", "pip", "install", "--upgrade", "pip"], cwd=args.repo_root, timeout=args.install_timeout)
    install_spec = str(args.repo_root)
    _run_checked([str(python_bin), "-m", "pip", "install", "-e", install_spec], cwd=args.repo_root, timeout=args.install_timeout)
    pfe_studio = _venv_script(venv_dir, "pfe-studio")
    if not pfe_studio.exists():
        raise AssertionError(f"pfe-studio console script was not installed: {pfe_studio}")
    return {"venv": venv_dir, "python": python_bin, "pfe_studio": pfe_studio}


def _run_smoke(args: argparse.Namespace, workdir: Path) -> dict[str, str]:
    install = _install_clean(args, workdir)
    port = args.port or _free_loopback_port()
    pfe_home = workdir / ".pfe"
    env = os.environ.copy()
    env["PFE_HOME"] = str(pfe_home)
    env.pop("PFE_ENABLE_REAL_LOCAL_INFERENCE", None)
    env.pop("PYTHONPATH", None)
    command = [
        str(install["pfe_studio"]),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--workspace",
        args.workspace,
        "--no-open",
    ]
    process = subprocess.Popen(
        command,
        cwd=str(workdir),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        base_url = f"http://127.0.0.1:{port}"
        studio = _wait_for_text(f"{base_url}/studio", timeout=args.server_timeout)
        if studio["status"] != 200:
            raise AssertionError(f"unexpected Studio status: {studio}")
        studio_html = str(studio["body"])
        _require(studio_html, "PFE / 本地模型工作台", label="studio first launch html")
        _require(studio_html, "选择模型，拿到本机 API。", label="studio first launch html")
        _require(studio_html, "测试接入", label="studio first launch html")
        _require(studio_html, "/pfe/static/studio.css", label="studio first launch html")
        _require(studio_html, "/pfe/static/studio.js", label="studio first launch html")
        studio_js = _wait_for_text(f"{base_url}/pfe/static/studio.js", timeout=args.server_timeout)
        if studio_js["status"] != 200:
            raise AssertionError(f"unexpected Studio JS status: {studio_js}")
        _require(str(studio_js["body"]), "/pfe/handoff/test", label="studio first launch js")

        runtime = _request_json(f"{base_url}/pfe/runtime")
        if runtime["status"] != 200:
            raise AssertionError(f"unexpected runtime status: {runtime}")
        runtime_body = runtime["body"] if isinstance(runtime["body"], dict) else {}
        if runtime_body.get("workspace") != args.workspace:
            raise AssertionError(f"runtime did not use requested workspace: {runtime_body}")

        handoff = _request_json(f"{base_url}/pfe/handoff")
        handoff_body = handoff["body"] if isinstance(handoff["body"], dict) else {}
        urls = handoff_body.get("urls") if isinstance(handoff_body.get("urls"), dict) else {}
        if urls.get("api") != f"{base_url}/v1/chat/completions":
            raise AssertionError(f"handoff did not expose first-launch chat API: {handoff_body}")
        if urls.get("feedback") != f"{base_url}/pfe/feedback":
            raise AssertionError(f"handoff did not expose first-launch feedback API: {handoff_body}")

        chat = _request_json(
            f"{base_url}/v1/chat/completions",
            method="POST",
            body={
                "model": "local",
                "messages": [{"role": "user", "content": "hello from first launch"}],
                "metadata": {"source": "studio_first_launch_smoke"},
            },
        )
        chat_body = chat["body"] if isinstance(chat["body"], dict) else {}
        if chat["status"] != 200 or not chat_body.get("session_id") or not chat_body.get("request_id"):
            raise AssertionError(f"first-launch chat did not return response ids: {chat_body}")

        return {
            "workspace": args.workspace,
            "pfe_home": str(pfe_home),
            "venv": str(install["venv"]),
            "studio_url": f"{base_url}/studio",
            "api_url": str(urls.get("api")),
            "request_id": str(chat_body.get("request_id")),
        }
    finally:
        stdout, stderr = _stop_process(process)
        if process.returncode not in (0, -15, -9, 143):
            raise AssertionError(
                f"pfe-studio exited unexpectedly with {process.returncode}\n"
                f"stdout:\n{stdout}\nstderr:\n{stderr}"
            )


def main() -> int:
    repo_root = _repo_root()
    parser = argparse.ArgumentParser(
        description="Smoke-test clean install and first Studio launch without relying on the repo venv."
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--python", default=_default_python(repo_root))
    parser.add_argument("--workspace", default="first_launch")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--install-timeout", type=int, default=180)
    parser.add_argument("--server-timeout", type=int, default=30)
    parser.add_argument("--workdir", type=Path, default=None)
    parser.add_argument("--keep-workdir", action="store_true")
    args = parser.parse_args()
    args.repo_root = args.repo_root.resolve()

    tempdir = None
    if args.workdir is None:
        tempdir = tempfile.TemporaryDirectory(prefix="pfe-studio-first-launch-")
        workdir = Path(tempdir.name)
    else:
        workdir = args.workdir.resolve()
        workdir.mkdir(parents=True, exist_ok=True)

    print(f"workdir: {workdir}")
    print(f"python:  {args.python}")
    try:
        summary = _run_smoke(args, workdir)
        print()
        print("STUDIO FIRST-LAUNCH SMOKE PASSED")
        print(f"workspace:  {summary['workspace']}")
        print(f"pfe_home:   {summary['pfe_home']}")
        print(f"venv:       {summary['venv']}")
        print(f"studio_url: {summary['studio_url']}")
        print(f"api_url:    {summary['api_url']}")
        print(f"request_id: {summary['request_id']}")
        return 0
    finally:
        if tempdir is not None and not args.keep_workdir:
            tempdir.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
