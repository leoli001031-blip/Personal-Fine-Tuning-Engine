#!/usr/bin/env python3
"""Render GitHub Actions release evidence JSON as a Markdown snippet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return payload


def _yes_no(value: object) -> str:
    return "yes" if bool(value) else "no"


def render_remote_release_evidence(
    *,
    remote_evidence: dict[str, Any],
    bundle: dict[str, Any] | None = None,
) -> str:
    run = remote_evidence.get("run")
    blockers = remote_evidence.get("blockers") or []
    lines = ["## Remote CI evidence", ""]
    lines.append(f"- status: `{remote_evidence.get('status')}`")
    lines.append(f"- release_ready: `{_yes_no(remote_evidence.get('release_ready'))}`")
    lines.append(f"- workflow: `{remote_evidence.get('workflow_name')}`")
    lines.append(f"- repo: `{remote_evidence.get('repo')}`")
    if isinstance(run, dict):
        lines.extend(
            [
                f"- run: {run.get('html_url')}",
                f"- run status: `{run.get('status')}`",
                f"- conclusion: `{run.get('conclusion')}`",
                f"- event: `{run.get('event')}`",
                f"- branch: `{run.get('head_branch')}`",
                f"- commit: `{run.get('head_sha')}`",
                f"- started: `{run.get('run_started_at')}`",
                f"- updated: `{run.get('updated_at')}`",
            ]
        )
    if blockers:
        lines.append("- blockers: " + "; ".join(str(item) for item in blockers))

    if bundle:
        summary = bundle.get("summary") or {}
        lines.extend(
            [
                "",
                "Artifact bundle:",
                f"- status: `{bundle.get('status')}`",
                (
                    "- summary: "
                    f"blockers={summary.get('blockers')} | "
                    f"present={summary.get('present')} | "
                    f"total={summary.get('total')} | "
                    f"warnings={summary.get('warnings')}"
                ),
            ]
        )
        for report in bundle.get("reports") or []:
            if not isinstance(report, dict):
                continue
            label = report.get("label")
            status = report.get("status", report.get("error"))
            digest = report.get("sha256")
            size = report.get("size_bytes")
            suffix = f" | sha256={digest} | bytes={size}" if digest else ""
            lines.append(f"- {label}: `{status}`{suffix}")

    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--remote-evidence-report",
        type=Path,
        default=Path("/tmp/pfe-github-actions-release-evidence.json"),
    )
    parser.add_argument("--bundle-report", type=Path)
    parser.add_argument("--output-path", type=Path, default=Path("/tmp/pfe-remote-release-evidence.md"))
    parser.add_argument("--require-success", action="store_true")
    args = parser.parse_args(argv)

    remote = _read_json(args.remote_evidence_report)
    bundle = _read_json(args.bundle_report) if args.bundle_report else None
    snippet = render_remote_release_evidence(remote_evidence=remote, bundle=bundle)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(snippet, encoding="utf-8")
    print(f"REMOTE RELEASE EVIDENCE SNIPPET: {args.output_path}")
    if args.require_success and not (remote.get("status") == "passed" and remote.get("release_ready")):
        print("blocker: remote release evidence is not successful")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
