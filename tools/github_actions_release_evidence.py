#!/usr/bin/env python3
"""Record GitHub Actions release-gate evidence as JSON."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_REPO = "leoli001031-blip/Personal-Fine-Tuning-Engine"
DEFAULT_WORKFLOW_NAME = "PFE release gates"


def _gh_runs(repo: str, *, per_page: int) -> dict[str, Any]:
    path = f"repos/{repo}/actions/runs?per_page={per_page}"
    try:
        result = subprocess.run(
            ["gh", "api", path],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("gh CLI not found") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("gh API request timed out") from exc
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or f"exit_code={result.returncode}"
        raise RuntimeError(f"gh API request failed: {detail}")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("gh API returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("gh API returned unexpected payload")
    return payload


def select_release_run(
    payload: dict[str, Any],
    *,
    workflow_name: str = DEFAULT_WORKFLOW_NAME,
    branch: str | None = None,
    event: str | None = None,
) -> dict[str, Any] | None:
    runs = payload.get("workflow_runs", [])
    if not isinstance(runs, list):
        return None
    for run in runs:
        if not isinstance(run, dict):
            continue
        if run.get("name") != workflow_name:
            continue
        if branch and run.get("head_branch") != branch:
            continue
        if event and run.get("event") != event:
            continue
        return run
    return None


def build_evidence(
    *,
    repo: str,
    workflow_name: str,
    run: dict[str, Any] | None,
    require_success: bool,
) -> dict[str, Any]:
    if run is None:
        return {
            "status": "missing",
            "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "repo": repo,
            "workflow_name": workflow_name,
            "require_success": require_success,
            "run": None,
            "release_ready": False,
            "blockers": ["matching GitHub Actions run not found"],
        }

    run_status = run.get("status")
    conclusion = run.get("conclusion")
    run_ready = run_status == "completed" and conclusion == "success"
    blockers: list[str] = []
    if require_success and not run_ready:
        blockers.append(f"latest matching run is not successful: status={run_status} conclusion={conclusion}")
    return {
        "status": "passed" if not blockers else "blocked",
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "repo": repo,
        "workflow_name": workflow_name,
        "require_success": require_success,
        "release_ready": run_ready,
        "blockers": blockers,
        "run": {
            "id": run.get("id"),
            "name": run.get("name"),
            "event": run.get("event"),
            "status": run_status,
            "conclusion": conclusion,
            "html_url": run.get("html_url"),
            "head_branch": run.get("head_branch"),
            "head_sha": run.get("head_sha"),
            "created_at": run.get("created_at"),
            "updated_at": run.get("updated_at"),
            "run_started_at": run.get("run_started_at"),
        },
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--workflow-name", default=DEFAULT_WORKFLOW_NAME)
    parser.add_argument("--branch")
    parser.add_argument("--event")
    parser.add_argument("--per-page", type=int, default=20)
    parser.add_argument("--output-path", type=Path, default=Path("/tmp/pfe-github-actions-release-evidence.json"))
    parser.add_argument("--require-success", action="store_true")
    args = parser.parse_args(argv)

    try:
        payload = _gh_runs(args.repo, per_page=args.per_page)
        run = select_release_run(
            payload,
            workflow_name=args.workflow_name,
            branch=args.branch,
            event=args.event,
        )
        evidence = build_evidence(
            repo=args.repo,
            workflow_name=args.workflow_name,
            run=run,
            require_success=args.require_success,
        )
    except RuntimeError as exc:
        evidence = {
            "status": "blocked",
            "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "repo": args.repo,
            "workflow_name": args.workflow_name,
            "require_success": args.require_success,
            "run": None,
            "release_ready": False,
            "blockers": [str(exc)],
        }

    _write_json(args.output_path, evidence)
    status = str(evidence["status"]).upper()
    print(f"GITHUB ACTIONS RELEASE EVIDENCE {status}")
    print(f"report: {args.output_path}")
    run = evidence.get("run")
    if isinstance(run, dict):
        print(f"run:    {run.get('html_url')}")
        print(f"state:  status={run.get('status')} conclusion={run.get('conclusion')}")
    for blocker in evidence.get("blockers", []):
        print(f"blocker: {blocker}")
    return 0 if evidence["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
