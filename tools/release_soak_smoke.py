#!/usr/bin/env python3
"""Run a bounded release-readiness soak against live PFE surfaces."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from first_run_smoke import _default_python, _pythonpath, _repo_root
from server_live_smoke import (
    _free_loopback_port,
    _prepare_workspace,
    _request_json,
    _request_text,
    _wait_for_healthz,
)


class SoakStats:
    def __init__(self) -> None:
        self.probe_count = 0
        self.chat_turns = 0
        self.iterations = 0
        self.latencies_ms: list[float] = []
        self.last_probe: dict[str, Any] = {}
        self.failed_probe: dict[str, Any] | None = None

    def begin_probe(self, method: str, path: str) -> None:
        self.last_probe = {
            "method": method,
            "path": path,
            "iteration": self.iterations,
            "probe_count": self.probe_count,
        }

    def record(self, elapsed_seconds: float) -> None:
        self.probe_count += 1
        self.latencies_ms.append(round(elapsed_seconds * 1000, 2))

    def record_failure(self, method: str, path: str, elapsed_seconds: float, exc: BaseException) -> None:
        self.record(elapsed_seconds)
        self.failed_probe = {
            "method": method,
            "path": path,
            "iteration": self.iterations,
            "probe_count": self.probe_count,
            "elapsed_ms": round(elapsed_seconds * 1000, 2),
            "error_type": exc.__class__.__name__,
            "error": str(exc),
        }

    def summary(self) -> dict[str, Any]:
        if not self.latencies_ms:
            return {"probe_count": self.probe_count, "latency_ms": {}}
        ordered = sorted(self.latencies_ms)
        p95_index = min(len(ordered) - 1, int(len(ordered) * 0.95))
        return {
            "probe_count": self.probe_count,
            "iterations": self.iterations,
            "chat_turns": self.chat_turns,
            "latency_ms": {
                "avg": round(sum(ordered) / len(ordered), 2),
                "max": ordered[-1],
                "p95": ordered[p95_index],
            },
        }


def _require_json_payload(result: dict[str, object], *, label: str) -> object:
    if result.get("status") != 200 or result.get("body") is None:
        raise AssertionError(f"{label} did not return a 200 JSON payload: {result}")
    return result["body"]


def _json_probe(
    base_url: str,
    path: str,
    *,
    args: argparse.Namespace,
    stats: SoakStats,
    method: str = "GET",
    body: dict[str, object] | None = None,
) -> dict[str, Any]:
    payload = _json_payload_probe(base_url, path, args=args, stats=stats, method=method, body=body)
    if not isinstance(payload, dict):
        raise AssertionError(f"{path} did not return a JSON object: {payload!r}")
    return payload


def _json_payload_probe(
    base_url: str,
    path: str,
    *,
    args: argparse.Namespace,
    stats: SoakStats,
    method: str = "GET",
    body: dict[str, object] | None = None,
) -> object:
    start = time.perf_counter()
    stats.begin_probe(method, path)
    try:
        result = _request_json(f"{base_url}{path}", method=method, body=body, timeout=args.request_timeout)
    except Exception as exc:
        elapsed = time.perf_counter() - start
        stats.record_failure(method, path, elapsed, exc)
        raise AssertionError(
            f"{method} {path} failed after {elapsed:.2f}s "
            f"at iteration={stats.iterations} probe={stats.probe_count}: {exc.__class__.__name__}: {exc}"
        ) from exc
    stats.record(time.perf_counter() - start)
    return _require_json_payload(result, label=path)


def _text_probe(base_url: str, path: str, *, args: argparse.Namespace, stats: SoakStats, expected: str) -> None:
    start = time.perf_counter()
    stats.begin_probe("GET", path)
    try:
        result = _request_text(f"{base_url}{path}", timeout=args.request_timeout)
    except Exception as exc:
        elapsed = time.perf_counter() - start
        stats.record_failure("GET", path, elapsed, exc)
        raise AssertionError(
            f"GET {path} failed after {elapsed:.2f}s "
            f"at iteration={stats.iterations} probe={stats.probe_count}: {exc.__class__.__name__}: {exc}"
        ) from exc
    stats.record(time.perf_counter() - start)
    body = str(result.get("body") or "")
    if result.get("status") != 200 or expected not in body:
        raise AssertionError(f"{path} did not return expected HTML marker {expected!r}: status={result.get('status')}")


def _validate_status(payload: dict[str, Any], *, expected_version: str) -> None:
    latest = payload.get("latest_adapter")
    if not isinstance(latest, dict) or latest.get("version") != expected_version:
        raise AssertionError(f"/pfe/status lost promoted adapter {expected_version}: {latest}")
    if not isinstance(payload.get("operations_console"), dict):
        raise AssertionError("/pfe/status did not expose operations_console")
    if not isinstance(payload.get("operations_dashboard"), dict):
        raise AssertionError("/pfe/status did not expose operations_dashboard")
    train_queue = payload.get("train_queue")
    if not isinstance(train_queue, dict) or not isinstance(train_queue.get("daemon"), dict):
        raise AssertionError("/pfe/status did not expose train_queue.daemon")


def _validate_daemon(payload: dict[str, Any]) -> None:
    health_state = str(payload.get("health_state") or "").lower()
    heartbeat_state = str(payload.get("heartbeat_state") or "").lower()
    lease_state = str(payload.get("lease_state") or "").lower()
    lock_state = str(payload.get("lock_state") or "").lower()
    if health_state in {"stale", "expired", "blocked"}:
        raise AssertionError(f"daemon health became unhealthy: {payload}")
    if heartbeat_state in {"stale", "missing"}:
        raise AssertionError(f"daemon heartbeat became unhealthy: {payload}")
    if lease_state == "expired":
        raise AssertionError(f"daemon lease expired: {payload}")
    if lock_state not in {"active", "idle"}:
        raise AssertionError(f"daemon lock state is unexpected: {payload}")


def _wait_for_daemon_ready(base_url: str, *, args: argparse.Namespace, stats: SoakStats) -> dict[str, Any]:
    _json_probe(
        base_url,
        "/pfe/auto-train/start-worker-daemon?note=release_soak_start",
        args=args,
        stats=stats,
        method="POST",
    )
    deadline = time.monotonic() + args.daemon_start_timeout
    last_payload: dict[str, Any] = {}
    while time.monotonic() < deadline:
        last_payload = _json_probe(base_url, "/pfe/auto-train/worker-daemon", args=args, stats=stats)
        health_state = str(last_payload.get("health_state") or "").lower()
        heartbeat_state = str(last_payload.get("heartbeat_state") or "").lower()
        lock_state = str(last_payload.get("lock_state") or "").lower()
        if health_state == "healthy" and heartbeat_state in {"fresh", "delayed"} and lock_state == "active":
            return last_payload
        time.sleep(0.5)
    raise AssertionError(f"daemon did not become healthy within {args.daemon_start_timeout}s: {last_payload}")


def _stop_daemon(base_url: str, *, args: argparse.Namespace, stats: SoakStats) -> None:
    stop_payload = _json_probe(
        base_url,
        "/pfe/auto-train/stop-worker-daemon?note=release_soak_stop",
        args=args,
        stats=stats,
        method="POST",
    )
    deadline = time.monotonic() + args.daemon_stop_timeout
    last_payload = stop_payload
    while time.monotonic() < deadline:
        last_payload = _json_probe(base_url, "/pfe/auto-train/worker-daemon", args=args, stats=stats)
        active = bool(last_payload.get("active", False))
        observed_state = str(last_payload.get("observed_state") or "").lower()
        lock_state = str(last_payload.get("lock_state") or "").lower()
        if not active and observed_state in {"stopped", "idle", ""} and lock_state in {"idle", "stopped"}:
            return
        time.sleep(0.25)

    pid = last_payload.get("pid")
    if pid not in (None, os.getpid()):
        try:
            os.kill(int(pid), signal.SIGTERM)
        except Exception:
            pass
    raise AssertionError(f"daemon did not stop within {args.daemon_stop_timeout}s: {last_payload}")


def _chat_and_feedback(base_url: str, *, args: argparse.Namespace, stats: SoakStats, iteration: int) -> None:
    message = f"Give me one release soak validation step. iteration={iteration}"
    chat = _json_probe(
        base_url,
        "/v1/chat/completions",
        args=args,
        stats=stats,
        method="POST",
        body={
            "model": "local",
            "messages": [{"role": "user", "content": message}],
            "stream": False,
        },
    )
    choices = chat.get("choices")
    if not isinstance(choices, list) or not choices:
        raise AssertionError(f"chat completion did not return choices: {chat}")
    request_id = str(chat.get("request_id") or "")
    session_id = str(chat.get("session_id") or "")
    if not request_id or not session_id:
        raise AssertionError(f"chat completion did not expose request/session ids: {chat}")
    first = choices[0] if isinstance(choices[0], dict) else {}
    assistant = ""
    if isinstance(first, dict) and isinstance(first.get("message"), dict):
        assistant = str(first["message"].get("content") or "")

    feedback = _json_probe(
        base_url,
        "/pfe/feedback",
        args=args,
        stats=stats,
        method="POST",
        body={
            "session_id": session_id,
            "request_id": request_id,
            "action": "accept",
            "response_time_seconds": 0.25,
            "user_message": message,
            "assistant_message": assistant,
        },
    )
    if feedback.get("success") is not True or feedback.get("signal_type") != "accept":
        raise AssertionError(f"feedback did not record an accept signal: {feedback}")
    stats.chat_turns += 1


def _run_iteration(base_url: str, *, args: argparse.Namespace, stats: SoakStats, expected_version: str) -> dict[str, Any]:
    _json_probe(base_url, "/healthz", args=args, stats=stats)
    status = _json_probe(base_url, "/pfe/status?detail=full", args=args, stats=stats)
    _validate_status(status, expected_version=expected_version)
    _text_probe(base_url, "/dashboard", args=args, stats=stats, expected="PFE Observability Dashboard")
    _text_probe(base_url, "/chat", args=args, stats=stats, expected="PFE Local Chat")
    for path in (
        "/pfe/dashboard/metrics",
        "/pfe/dashboard/training",
        "/pfe/dashboard/signals",
        "/pfe/dashboard/health",
        "/pfe/auto-train/worker-runner",
        "/pfe/auto-train/worker-runner/history?limit=5",
        "/pfe/auto-train/worker-daemon/history?limit=5",
        "/pfe/auto-train/queue-history?limit=5",
    ):
        _json_probe(base_url, path, args=args, stats=stats)
    _json_payload_probe(base_url, "/pfe/dashboard/adapters", args=args, stats=stats)
    daemon = _json_probe(base_url, "/pfe/auto-train/worker-daemon", args=args, stats=stats)
    _validate_daemon(daemon)
    if args.chat_every > 0 and stats.iterations % args.chat_every == 0:
        _chat_and_feedback(base_url, args=args, stats=stats, iteration=stats.iterations)
    return daemon


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _tail_lines(text: str, *, lines: int = 80) -> str:
    return "\n".join((text or "").splitlines()[-lines:])


def _start_soak_server(
    args: argparse.Namespace,
    workdir: Path,
    port: int,
) -> tuple[subprocess.Popen[str], Any, Any, Path, Path]:
    stdout_path = workdir / "server.stdout.log"
    stderr_path = workdir / "server.stderr.log"
    stdout_handle = stdout_path.open("w", encoding="utf-8", errors="replace")
    stderr_handle = stderr_path.open("w", encoding="utf-8", errors="replace")
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
    process: subprocess.Popen[str] = subprocess.Popen(
        command,
        cwd=str(workdir),
        env=env,
        text=True,
        stdout=stdout_handle,
        stderr=stderr_handle,
    )
    return process, stdout_handle, stderr_handle, stdout_path, stderr_path


def _stop_soak_server(
    process: subprocess.Popen[str],
    stdout_handle: Any,
    stderr_handle: Any,
    stdout_path: Path,
    stderr_path: Path,
) -> tuple[str, str]:
    if process.poll() is None:
        process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)
    stdout_handle.close()
    stderr_handle.close()
    stdout = stdout_path.read_text(encoding="utf-8", errors="replace") if stdout_path.exists() else ""
    stderr = stderr_path.read_text(encoding="utf-8", errors="replace") if stderr_path.exists() else ""
    return stdout, stderr


def _run_soak(args: argparse.Namespace, workdir: Path) -> dict[str, Any]:
    setup = _prepare_workspace(args, workdir)
    port = args.port if args.port else _free_loopback_port()
    base_url = f"http://127.0.0.1:{port}"
    process, stdout_handle, stderr_handle, stdout_path, stderr_path = _start_soak_server(args, workdir, port)
    stats = SoakStats()
    daemon: dict[str, Any] = {}
    cleanup_errors: list[str] = []
    failure: dict[str, Any] | None = None
    stdout = ""
    stderr = ""
    started_at = time.monotonic()
    try:
        healthz = _wait_for_healthz(base_url, process, timeout_seconds=args.server_timeout)
        if healthz["body"].get("status") != "ok":  # type: ignore[union-attr]
            raise AssertionError(f"unexpected healthz payload: {healthz}")
        daemon = _wait_for_daemon_ready(base_url, args=args, stats=stats)

        deadline = time.monotonic() + args.duration_seconds
        while True:
            stats.iterations += 1
            daemon = _run_iteration(base_url, args=args, stats=stats, expected_version=setup["version"])
            print(
                "soak iteration "
                f"{stats.iterations}: probes={stats.probe_count} chat_turns={stats.chat_turns} "
                f"daemon={daemon.get('health_state')}/{daemon.get('heartbeat_state')}/{daemon.get('lease_state')}",
                flush=True,
            )
            if time.monotonic() >= deadline and stats.iterations >= args.min_iterations:
                break
            time.sleep(args.interval_seconds)
    except Exception as exc:
        failure = {
            "error_type": exc.__class__.__name__,
            "error": str(exc),
            "last_probe": stats.last_probe,
            "failed_probe": stats.failed_probe,
        }
    finally:
        try:
            _stop_daemon(base_url, args=args, stats=stats)
        except Exception as exc:
            cleanup_errors.append(f"daemon stop failed: {exc}")
        stdout, stderr = _stop_soak_server(process, stdout_handle, stderr_handle, stdout_path, stderr_path)

    if process.returncode not in {0, -15, -9, None} and failure is None:
        failure = {
            "error_type": "AssertionError",
            "error": f"server exited unexpectedly after soak (code={process.returncode})",
            "last_probe": stats.last_probe,
            "failed_probe": stats.failed_probe,
        }
    if cleanup_errors:
        if failure is None:
            failure = {
                "error_type": "AssertionError",
                "error": "; ".join(cleanup_errors),
                "last_probe": stats.last_probe,
                "failed_probe": stats.failed_probe,
            }
        else:
            failure["cleanup_errors"] = cleanup_errors

    elapsed = round(time.monotonic() - started_at, 2)
    return {
        **setup,
        "status": "failed" if failure else "passed",
        "base_url": base_url,
        "port": str(port),
        "duration_seconds": elapsed,
        "daemon": {
            "health_state": daemon.get("health_state"),
            "heartbeat_state": daemon.get("heartbeat_state"),
            "lease_state": daemon.get("lease_state"),
            "lock_state": daemon.get("lock_state"),
            "pid": daemon.get("pid"),
        },
        "failure": failure,
        "server_returncode": process.returncode,
        "server_stdout_tail": _tail_lines(stdout),
        "server_stderr_tail": _tail_lines(stderr),
        **stats.summary(),
    }


def main() -> int:
    repo_root = _repo_root()
    parser = argparse.ArgumentParser(
        description=(
            "Run a bounded release soak over a temporary live server, dashboard APIs, chat/feedback, "
            "queue surfaces, and worker daemon status."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--python", default=_default_python(repo_root))
    parser.add_argument("--workspace", default="release_soak")
    parser.add_argument("--port", type=int, default=0, help="Port to bind. Defaults to a free loopback port.")
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--server-timeout", type=float, default=15.0)
    parser.add_argument("--request-timeout", type=float, default=5.0)
    parser.add_argument("--daemon-start-timeout", type=float, default=20.0)
    parser.add_argument("--daemon-stop-timeout", type=float, default=10.0)
    parser.add_argument("--duration-seconds", type=float, default=60.0)
    parser.add_argument("--interval-seconds", type=float, default=2.0)
    parser.add_argument("--min-iterations", type=int, default=2)
    parser.add_argument("--chat-every", type=int, default=3)
    parser.add_argument("--report-path", type=Path, default=Path(tempfile.gettempdir()) / "pfe-release-soak-report.json")
    parser.add_argument("--workdir", type=Path, default=None)
    parser.add_argument("--keep-workdir", action="store_true")
    args = parser.parse_args()
    args.repo_root = args.repo_root.resolve()
    args.duration_seconds = max(0.0, float(args.duration_seconds))
    args.interval_seconds = max(0.0, float(args.interval_seconds))
    args.min_iterations = max(1, int(args.min_iterations))

    tempdir = None
    if args.workdir is None:
        tempdir = tempfile.TemporaryDirectory(prefix="pfe-release-soak-")
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
        summary = _run_soak(args, workdir)
        _write_report(args.report_path.expanduser().resolve(), summary)
        if summary.get("status") == "failed":
            failure = summary.get("failure") if isinstance(summary.get("failure"), dict) else {}
            print("RELEASE SOAK SMOKE FAILED")
            print(f"workspace:  {summary['workspace']}")
            print(f"version:    {summary['version']}")
            print(f"base_url:   {summary['base_url']}")
            print(f"duration:   {summary['duration_seconds']}s")
            print(f"iterations: {summary['iterations']}")
            print(f"probes:     {summary['probe_count']}")
            print(f"chat_turns: {summary['chat_turns']}")
            print(f"failure:    {failure.get('error_type')}: {failure.get('error')}")
            print(f"failed_probe: {failure.get('failed_probe')}")
            print(f"report:     {args.report_path.expanduser().resolve()}")
            return 1
        print("RELEASE SOAK SMOKE PASSED")
        print(f"workspace:  {summary['workspace']}")
        print(f"version:    {summary['version']}")
        print(f"base_url:   {summary['base_url']}")
        print(f"duration:   {summary['duration_seconds']}s")
        print(f"iterations: {summary['iterations']}")
        print(f"probes:     {summary['probe_count']}")
        print(f"chat_turns: {summary['chat_turns']}")
        print(f"latency_ms: {summary['latency_ms']}")
        print(f"daemon:     {summary['daemon']}")
        print(f"report:     {args.report_path.expanduser().resolve()}")
        return 0
    finally:
        if tempdir is not None and not args.keep_workdir:
            tempdir.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
