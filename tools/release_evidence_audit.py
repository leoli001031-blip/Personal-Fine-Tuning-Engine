#!/usr/bin/env python3
"""Audit local release evidence before claiming release readiness."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class AuditItem:
    code: str
    status: str
    message: str


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _make_targets(makefile: str) -> set[str]:
    return {
        match.group(1)
        for match in re.finditer(r"^([A-Za-z0-9_.-]+):(?:\s|$)", makefile, flags=re.MULTILINE)
    }


def _workflow_count(repo: str) -> int | None:
    try:
        result = subprocess.run(
            [
                "gh",
                "api",
                f"repos/{repo}/actions/workflows",
                "--jq",
                ".total_count",
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=20,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    try:
        return int(result.stdout.strip())
    except ValueError:
        return None


def _process_matches() -> list[str]:
    try:
        result = subprocess.run(
            ["ps", "-axo", "pid,ppid,command"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    patterns = (
        "pfe_core.worker_daemon",
        "pfe_cli.main serve",
        "release_soak",
        "release_perf",
        "browser_ui_live_smoke",
        "real_local_happy_path_smoke",
    )
    return [line for line in result.stdout.splitlines() if any(pattern in line for pattern in patterns)]


def audit_release_evidence(
    *,
    root: Path = ROOT,
    require_remote: bool = False,
    check_remote: bool = False,
    check_processes: bool = True,
    repo: str = "leoli001031-blip/Personal-Fine-Tuning-Engine",
    remote_evidence_report: Path = Path("/tmp/pfe-github-actions-release-evidence.json"),
) -> list[AuditItem]:
    items: list[AuditItem] = []

    def add(code: str, ok: bool, message: str, *, warn: bool = False) -> None:
        if ok:
            status = "ok"
        elif warn:
            status = "warn"
        else:
            status = "blocker"
        items.append(AuditItem(code=code, status=status, message=message))

    required_files = [
        ".github/workflows/pfe-release-gates.yml",
        "docs/reference/release-readiness-evidence.md",
        "docs/reference/release-candidate-checklist.md",
        "docs/reference/release-notes-phase2-rc.md",
        "docs/guides/install-upgrade.md",
        "docs/reference/known-limitations.md",
        "docs/reference/user-acceptance-checklist.md",
        "tools/release_perf_benchmark.py",
        "tools/release_soak_smoke.py",
        "tools/release_evidence_audit.py",
        "tools/release_evidence_bundle.py",
        "tools/github_actions_release_evidence.py",
        "tools/render_remote_release_evidence.py",
        "tools/browser_ui_live_smoke.py",
        "tools/real_local_happy_path_smoke.py",
        "tests/test_release_evidence_audit.py",
        "tests/test_release_evidence_bundle.py",
        "tests/test_github_actions_release_evidence.py",
        "tests/test_render_remote_release_evidence.py",
        "tests/test_release_workflow_contract.py",
    ]
    missing = [path for path in required_files if not (root / path).exists()]
    add("required_files", not missing, "required release files present" if not missing else f"missing: {missing}")

    makefile = _read(root / "Makefile")
    targets = _make_targets(makefile)
    required_targets = {
        "test-e2e-mock",
        "smoke-beta",
        "smoke-release-strict",
        "soak-release",
        "benchmark-release",
        "release-local-evidence",
        "audit-release-evidence",
        "audit-release-evidence-report",
        "bundle-release-evidence",
        "record-remote-release-evidence",
        "render-remote-release-evidence",
    }
    missing_targets = sorted(required_targets - targets)
    add(
        "make_targets",
        not missing_targets,
        "release Makefile targets present" if not missing_targets else f"missing targets: {missing_targets}",
    )

    workflow = _read(root / ".github" / "workflows" / "pfe-release-gates.yml")
    workflow_targets = {
        match.group(1) for match in re.finditer(r"\bmake\s+([A-Za-z0-9_.-]+)\b", workflow)
    }
    add(
        "workflow_targets",
        {
            "test-unit",
            "test-surface",
            "test-e2e-mock",
            "smoke-beta",
            "smoke-release-strict",
            "benchmark-release",
            "audit-release-evidence-report",
            "bundle-release-evidence",
        }.issubset(workflow_targets)
        and workflow_targets <= targets,
        f"workflow make targets valid: {sorted(workflow_targets)}",
    )
    strict_needles = [
        "release-gate:",
        "if: github.event_name != 'pull_request'",
        '.venv/bin/python -m pip install -e ".[e2e]"',
        ".venv/bin/python -m playwright install --with-deps chromium",
        ".venv/bin/python tools/prepare_tiny_hf_model.py",
        'export PFE_REAL_LOCAL_MODEL="$HOME/.cache/pfe/release-models/tiny-gpt2-local"',
        "make test-e2e-mock",
        "make smoke-release-strict",
        "make benchmark-release",
        "make audit-release-evidence-report",
        "make bundle-release-evidence",
        "actions/upload-artifact@v4",
        "pfe-release-evidence",
        "/tmp/pfe-release-perf-report.json",
        "/tmp/pfe-release-evidence-audit.json",
        "/tmp/pfe-release-evidence-bundle.json",
    ]
    missing_needles = [needle for needle in strict_needles if needle not in workflow]
    add(
        "workflow_strict_gate",
        not missing_needles,
        "strict workflow gate retained" if not missing_needles else f"missing workflow text: {missing_needles}",
    )

    dashboard = _read(root / "pfe-server" / "pfe_server" / "static" / "dashboard.html")
    add(
        "dashboard_offline",
        "http://" not in dashboard and "https://" not in dashboard and "OfflineChart" in dashboard,
        "dashboard is offline-first with OfflineChart",
    )

    evidence = _read(root / "docs" / "reference" / "release-readiness-evidence.md")
    required_evidence = [
        "smoke-beta",
        "make test-e2e-mock",
        "make smoke-release-strict",
        "release_soak_smoke.py --duration-seconds 1800",
        "make benchmark-release",
        "make release-local-evidence",
        "Dashboard offline-first",
        "CI workflow contract",
        "make audit-release-evidence",
        "record-remote-release-evidence",
        "render-remote-release-evidence",
    ]
    missing_evidence = [needle for needle in required_evidence if needle not in evidence]
    add(
        "release_evidence_doc",
        not missing_evidence,
        "release evidence records local gates and remote gap"
        if not missing_evidence
        else f"missing evidence text: {missing_evidence}",
    )
    add(
        "remote_state_recorded",
        "workflow_count=0" in evidence or "/actions/runs/" in evidence,
        "release evidence records remote CI state",
    )

    add("root_pfe_absent", not (root / ".pfe").exists(), "root .pfe absent")
    add("uv_lock_absent", not (root / "uv.lock").exists(), "root uv.lock absent")

    if check_processes:
        processes = _process_matches()
        add(
            "process_residue",
            not processes,
            "no release smoke/server/daemon process residue" if not processes else "residue: " + " | ".join(processes),
        )

    has_remote_run = "https://github.com/" in evidence and "/actions/runs/" in evidence
    add(
        "remote_ci_run_evidence",
        has_remote_run,
        "GitHub Actions run evidence recorded"
        if has_remote_run
        else "GitHub Actions run URL missing from release evidence",
        warn=not require_remote,
    )

    remote_payload = _read_json(remote_evidence_report)
    remote_ready = (
        isinstance(remote_payload, dict)
        and remote_payload.get("status") == "passed"
        and bool(remote_payload.get("release_ready"))
        and isinstance(remote_payload.get("run"), dict)
        and bool(remote_payload["run"].get("html_url"))
    )
    if require_remote or remote_payload is not None:
        if remote_payload is None:
            message = f"remote evidence report missing or invalid: {remote_evidence_report}"
        else:
            run = remote_payload.get("run")
            run_url = run.get("html_url") if isinstance(run, dict) else None
            message = (
                "remote evidence report ready"
                if remote_ready
                else (
                    f"remote evidence report not ready: status={remote_payload.get('status')} "
                    f"release_ready={remote_payload.get('release_ready')} run={run_url}"
                )
            )
        add("remote_evidence_report", remote_ready, message, warn=not require_remote)

    if check_remote:
        count = _workflow_count(repo)
        add(
            "remote_workflow_registered",
            count is not None and count > 0,
            f"remote workflow_count={count}" if count is not None else "could not read remote workflow count",
            warn=not require_remote,
        )

    return items


def _write_report(
    *,
    report_path: Path,
    items: list[AuditItem],
    require_remote: bool,
    check_remote: bool,
    check_processes: bool,
    repo: str,
) -> None:
    blockers = [item for item in items if item.status == "blocker"]
    warnings = [item for item in items if item.status == "warn"]
    payload = {
        "status": "blocked" if blockers else "passed",
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "require_remote": require_remote,
        "check_remote": check_remote,
        "check_processes": check_processes,
        "repo": repo,
        "summary": {
            "total": len(items),
            "ok": sum(1 for item in items if item.status == "ok"),
            "warn": len(warnings),
            "blocker": len(blockers),
        },
        "items": [asdict(item) for item in items],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-remote", action="store_true", help="fail until GitHub Actions run evidence exists")
    parser.add_argument("--check-remote", action="store_true", help="query GitHub workflow registration with gh")
    parser.add_argument(
        "--skip-process-check",
        action="store_true",
        help="skip local process residue checks",
    )
    parser.add_argument(
        "--repo",
        default="leoli001031-blip/Personal-Fine-Tuning-Engine",
        help="GitHub repo for --check-remote",
    )
    parser.add_argument(
        "--remote-evidence-report",
        type=Path,
        default=Path("/tmp/pfe-github-actions-release-evidence.json"),
        help="GitHub Actions evidence JSON produced by record-remote-release-evidence",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        help="write a machine-readable JSON audit report",
    )
    args = parser.parse_args(argv)

    items = audit_release_evidence(
        require_remote=args.require_remote,
        check_remote=args.check_remote,
        check_processes=not args.skip_process_check,
        repo=args.repo,
        remote_evidence_report=args.remote_evidence_report,
    )
    blockers = [item for item in items if item.status == "blocker"]

    print("RELEASE EVIDENCE AUDIT " + ("BLOCKED" if blockers else "PASSED"))
    for item in items:
        print(f"{item.status.upper():7} {item.code}: {item.message}")
    if args.report_path:
        _write_report(
            report_path=args.report_path,
            items=items,
            require_remote=args.require_remote,
            check_remote=args.check_remote,
            check_processes=not args.skip_process_check,
            repo=args.repo,
        )
        print(f"report:  {args.report_path}")

    return 2 if blockers else 0


if __name__ == "__main__":
    raise SystemExit(main())
