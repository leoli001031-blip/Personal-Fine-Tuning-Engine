#!/usr/bin/env python3
"""Collect release-readiness timing and memory baselines for key PFE gates."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from first_run_smoke import _default_python, _repo_root


try:
    import psutil  # type: ignore
except Exception:
    psutil = None  # type: ignore


DEFAULT_TINY_MODEL = Path.home() / ".cache" / "pfe" / "release-models" / "tiny-gpt2-local"
DEFAULT_THRESHOLDS: dict[str, dict[str, float]] = {
    "first_run_full": {"elapsed_seconds": 30.0, "peak_rss_mb": 800.0},
    "browser_ui_strict": {"elapsed_seconds": 30.0, "peak_rss_mb": 1600.0},
    "real_local_happy": {"elapsed_seconds": 45.0, "peak_rss_mb": 1800.0},
    "release_soak_short": {"elapsed_seconds": 45.0, "peak_rss_mb": 1400.0},
}


def _tail_text(path: Path, *, lines: int = 40) -> str:
    if not path.exists():
        return ""
    content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(content[-max(1, int(lines or 40)) :])


def _process_tree_rss_mb(pid: int) -> float | None:
    if psutil is None:
        return None
    try:
        root = psutil.Process(pid)
        processes = [root, *root.children(recursive=True)]
    except Exception:
        return None
    total = 0
    for proc in processes:
        try:
            total += int(proc.memory_info().rss)
        except Exception:
            continue
    return round(total / (1024 * 1024), 2)


def _terminate_process_tree(process: subprocess.Popen[object]) -> None:
    if psutil is not None:
        try:
            root = psutil.Process(process.pid)
            children = root.children(recursive=True)
            for child in children:
                child.terminate()
            root.terminate()
            _, alive = psutil.wait_procs([root, *children], timeout=3)
            for proc in alive:
                proc.kill()
            return
        except Exception:
            pass
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=3)
    except Exception:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass


def _run_command(
    *,
    label: str,
    command: list[str],
    cwd: Path,
    env: dict[str, str],
    timeout_seconds: float,
    sample_interval_seconds: float,
    log_dir: Path,
) -> dict[str, Any]:
    log_path = log_dir / f"{label}.log"
    start = time.perf_counter()
    peak_rss_mb: float | None = None
    timed_out = False
    with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
        process: subprocess.Popen[object] = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        deadline = start + timeout_seconds
        while process.poll() is None:
            sample = _process_tree_rss_mb(process.pid)
            if sample is not None:
                peak_rss_mb = sample if peak_rss_mb is None else max(peak_rss_mb, sample)
            if time.perf_counter() > deadline:
                timed_out = True
                _terminate_process_tree(process)
                break
            time.sleep(max(0.05, sample_interval_seconds))
        try:
            returncode = process.wait(timeout=3)
        except Exception:
            _terminate_process_tree(process)
            returncode = process.poll()
        sample = _process_tree_rss_mb(process.pid)
        if sample is not None:
            peak_rss_mb = sample if peak_rss_mb is None else max(peak_rss_mb, sample)

    elapsed_seconds = round(time.perf_counter() - start, 3)
    return {
        "label": label,
        "command": command,
        "elapsed_seconds": elapsed_seconds,
        "exit_code": int(returncode if returncode is not None else -1),
        "timed_out": timed_out,
        "peak_rss_mb": peak_rss_mb,
        "memory_sampler": "psutil" if psutil is not None else "unavailable",
        "log_path": str(log_path),
        "tail": _tail_text(log_path),
    }


def _task_commands(args: argparse.Namespace) -> dict[str, tuple[list[str], float]]:
    python = args.python
    return {
        "first_run_full": (
            [python, "tools/first_run_smoke.py", "--timeout", "45"],
            180.0,
        ),
        "browser_ui_strict": (
            [python, "tools/browser_ui_live_smoke.py", "--strict", "--browser-timeout-ms", "45000"],
            180.0,
        ),
        "real_local_happy": (
            [python, "tools/real_local_happy_path_smoke.py", "--strict", "--timeout", "120"],
            240.0,
        ),
        "release_soak_short": (
            [
                python,
                "tools/release_soak_smoke.py",
                "--duration-seconds",
                str(args.soak_duration_seconds),
                "--interval-seconds",
                "1",
                "--min-iterations",
                "2",
                "--chat-every",
                "2",
            ],
            max(180.0, float(args.soak_duration_seconds) + 90.0),
        ),
    }


def _selected_tasks(args: argparse.Namespace) -> list[str]:
    if not args.task or "all" in args.task:
        return ["first_run_full", "browser_ui_strict", "real_local_happy", "release_soak_short"]
    return list(dict.fromkeys(args.task))


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _threshold_violations(results: list[dict[str, Any]], thresholds: dict[str, dict[str, float]]) -> list[str]:
    violations: list[str] = []
    for result in results:
        label = str(result["label"])
        budget = thresholds.get(label)
        if not budget:
            continue
        elapsed_limit = budget.get("elapsed_seconds")
        if elapsed_limit is not None and float(result["elapsed_seconds"]) > elapsed_limit:
            violations.append(f"{label} elapsed {result['elapsed_seconds']}s > {elapsed_limit}s")
        rss_limit = budget.get("peak_rss_mb")
        peak_rss_mb = result.get("peak_rss_mb")
        if rss_limit is not None and peak_rss_mb is not None and float(peak_rss_mb) > rss_limit:
            violations.append(f"{label} peak_rss_mb {peak_rss_mb} > {rss_limit}")
    return violations


def main() -> int:
    repo_root = _repo_root()
    parser = argparse.ArgumentParser(
        description="Run release benchmark tasks and record elapsed time plus peak process-tree RSS."
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--python", default=_default_python(repo_root))
    parser.add_argument(
        "--task",
        action="append",
        choices=("all", "first_run_full", "browser_ui_strict", "real_local_happy", "release_soak_short"),
        help="Task to run. Repeatable. Defaults to all.",
    )
    parser.add_argument("--sample-interval-seconds", type=float, default=0.1)
    parser.add_argument("--soak-duration-seconds", type=float, default=10.0)
    parser.add_argument("--report-path", type=Path, default=Path(tempfile.gettempdir()) / "pfe-release-perf-report.json")
    parser.add_argument("--log-dir", type=Path, default=Path(tempfile.gettempdir()) / "pfe-release-perf-logs")
    parser.add_argument(
        "--no-thresholds",
        action="store_true",
        help="Record raw baselines without enforcing the default release performance budgets.",
    )
    args = parser.parse_args()
    args.repo_root = args.repo_root.resolve()
    args.log_dir = args.log_dir.expanduser().resolve()
    args.log_dir.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    if not env.get("PFE_REAL_LOCAL_MODEL") and DEFAULT_TINY_MODEL.exists():
        env["PFE_REAL_LOCAL_MODEL"] = str(DEFAULT_TINY_MODEL)

    commands = _task_commands(args)
    selected = _selected_tasks(args)
    results: list[dict[str, Any]] = []
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    for label in selected:
        command, timeout_seconds = commands[label]
        print(f"> benchmark {label}: {' '.join(command)}", flush=True)
        result = _run_command(
            label=label,
            command=command,
            cwd=args.repo_root,
            env=env,
            timeout_seconds=timeout_seconds,
            sample_interval_seconds=args.sample_interval_seconds,
            log_dir=args.log_dir,
        )
        results.append(result)
        print(
            f"  exit={result['exit_code']} elapsed={result['elapsed_seconds']}s "
            f"peak_rss_mb={result['peak_rss_mb']}",
            flush=True,
        )
        if result["exit_code"] != 0 or result["timed_out"]:
            _write_report(
                args.report_path.expanduser().resolve(),
                {
                    "status": "failed",
                    "started_at": started_at,
                    "repo_root": str(args.repo_root),
                    "python": args.python,
                    "thresholds": None if args.no_thresholds else DEFAULT_THRESHOLDS,
                    "threshold_violations": [],
                    "results": results,
                },
            )
            print(f"RELEASE PERF BENCHMARK FAILED: {label}")
            print(result["tail"])
            return int(result["exit_code"] or 1)

    elapsed_total = round(sum(float(item["elapsed_seconds"]) for item in results), 3)
    thresholds = None if args.no_thresholds else DEFAULT_THRESHOLDS
    threshold_violations = [] if thresholds is None else _threshold_violations(results, thresholds)
    report = {
        "status": "failed" if threshold_violations else "passed",
        "started_at": started_at,
        "repo_root": str(args.repo_root),
        "python": args.python,
        "model": env.get("PFE_REAL_LOCAL_MODEL"),
        "memory_sampler": "psutil" if psutil is not None else "unavailable",
        "thresholds": thresholds,
        "threshold_violations": threshold_violations,
        "elapsed_total_seconds": elapsed_total,
        "results": results,
    }
    report_path = args.report_path.expanduser().resolve()
    _write_report(report_path, report)
    if threshold_violations:
        print("RELEASE PERF BENCHMARK FAILED: threshold budget exceeded")
        for violation in threshold_violations:
            print(f"- {violation}")
        print(f"report: {report_path}")
        return 1
    print("RELEASE PERF BENCHMARK PASSED")
    print(f"tasks:  {', '.join(selected)}")
    print(f"total:  {elapsed_total}s")
    print(f"thresholds: {'not enforced' if args.no_thresholds else 'enforced'}")
    print(f"report: {report_path}")
    for item in results:
        print(f"- {item['label']}: elapsed={item['elapsed_seconds']}s peak_rss_mb={item['peak_rss_mb']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
