#!/usr/bin/env python3
"""Validate and summarize release evidence JSON artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _load_report(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.exists():
        return None, "missing"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, f"invalid_json: {exc}"
    if not isinstance(payload, dict):
        return None, "invalid_json: top-level value is not an object"
    return payload, None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_bundle(
    *,
    perf_report: Path,
    audit_report: Path,
    remote_evidence_report: Path,
    require_remote: bool,
) -> dict[str, Any]:
    reports: list[dict[str, Any]] = []
    blockers: list[str] = []
    warnings: list[str] = []

    expected = [
        ("performance", perf_report, True),
        ("evidence_audit", audit_report, True),
        ("remote_actions", remote_evidence_report, require_remote),
    ]
    for label, path, required in expected:
        payload, error = _load_report(path)
        if error:
            message = f"{label}: {error} ({path})"
            if required:
                blockers.append(message)
            else:
                warnings.append(message)
            reports.append({"label": label, "path": str(path), "present": False, "error": error})
            continue

        status = str(payload.get("status"))
        if label in {"performance", "evidence_audit"} and status != "passed":
            blockers.append(f"{label}: status={status}")
        if label == "remote_actions":
            release_ready = bool(payload.get("release_ready"))
            if require_remote and (status != "passed" or not release_ready):
                blockers.append(f"{label}: status={status} release_ready={release_ready}")
            elif status != "passed" or not release_ready:
                warnings.append(f"{label}: status={status} release_ready={release_ready}")

        reports.append(
            {
                "label": label,
                "path": str(path),
                "present": True,
                "status": status,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )

    return {
        "status": "blocked" if blockers else "passed",
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "require_remote": require_remote,
        "reports": reports,
        "summary": {
            "total": len(reports),
            "present": sum(1 for item in reports if item.get("present")),
            "warnings": len(warnings),
            "blockers": len(blockers),
        },
        "warnings": warnings,
        "blockers": blockers,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--perf-report", type=Path, default=Path("/tmp/pfe-release-perf-report.json"))
    parser.add_argument("--audit-report", type=Path, default=Path("/tmp/pfe-release-evidence-audit.json"))
    parser.add_argument(
        "--remote-evidence-report",
        type=Path,
        default=Path("/tmp/pfe-github-actions-release-evidence.json"),
    )
    parser.add_argument("--output-path", type=Path, default=Path("/tmp/pfe-release-evidence-bundle.json"))
    parser.add_argument("--require-remote", action="store_true")
    args = parser.parse_args(argv)

    bundle = build_bundle(
        perf_report=args.perf_report,
        audit_report=args.audit_report,
        remote_evidence_report=args.remote_evidence_report,
        require_remote=args.require_remote,
    )
    _write_json(args.output_path, bundle)
    print("RELEASE EVIDENCE BUNDLE " + str(bundle["status"]).upper())
    print(f"report: {args.output_path}")
    for warning in bundle["warnings"]:
        print(f"warning: {warning}")
    for blocker in bundle["blockers"]:
        print(f"blocker: {blocker}")
    return 0 if bundle["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
