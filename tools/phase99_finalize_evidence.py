#!/usr/bin/env python3
"""Finalize and validate archived Phase99 evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_ROOT = REPO_ROOT / "docs/demo/phase99-qwen3-native-generation-boundary"
DYNAMIC = {"evidence_manifest.json", "validation_summary.json"}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest() -> dict[str, Any]:
    files = [
        {"path": str(path.relative_to(EVIDENCE_ROOT)), "sha256": _sha256(path), "size_bytes": path.stat().st_size}
        for path in sorted(EVIDENCE_ROOT.rglob("*"))
        if path.is_file() and path.name not in DYNAMIC
    ]
    return {"kind": "phase99_evidence_manifest", "file_count": len(files), "files": files}


def _validate() -> dict[str, Any]:
    manifest = _read_json(EVIDENCE_ROOT / "evidence_manifest.json")
    decision = _read_json(EVIDENCE_ROOT / "phase99-final-decision.json")
    audit = _read_json(EVIDENCE_ROOT / "generation_stop_audit.json")
    failures = []
    for row in manifest.get("files") or []:
        path = EVIDENCE_ROOT / str(row["path"])
        if not path.is_file() or _sha256(path) != row.get("sha256"):
            failures.append(str(row["path"]))
    checks = {
        "manifest_files_unchanged": not failures,
        "archive_status": str(decision.get("status") or "").startswith("archive_"),
        "sft_not_run": decision.get("sft_status") == "not_run_phase99_gate_failed",
        "dpo_not_run": decision.get("dpo_status") == "not_run_phase99_gate_failed",
        "exactly_24_local_calls": decision.get("model_call_count") == 24,
        "all_turns_native_stop": audit.get("native_generation_stop_count") == audit.get("total_turn_count") == 24,
        "no_token_cap_termination": audit.get("max_new_tokens_termination_count") == 0,
        "no_post_hoc_truncation": audit.get("post_hoc_truncation_count") == 0,
        "no_forbidden_generation": audit.get("forbidden_generation_hit_count") == 0,
        "no_auto_promotion": decision.get("automatic_promotion_allowed") is False,
        "no_automatic_deployment": decision.get("automatic_deployment_allowed") is False,
        "simulated_only": decision.get("simulated_usage") is True and decision.get("actual_user_feedback_count") == 0,
    }
    return {"kind": "phase99_validation_summary", "passed": all(checks.values()), "checks": checks, "manifest_failures": failures, "decision_status": decision.get("status")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("finalize", "validate"))
    args = parser.parse_args()
    if args.command == "finalize":
        _write_json(EVIDENCE_ROOT / "evidence_manifest.json", _manifest())
    payload = _validate()
    _write_json(EVIDENCE_ROOT / "validation_summary.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
