#!/usr/bin/env python3
"""Generate Phase42 baseline, adapter-gate, and candidate-quality evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = REPO_ROOT / "pfe-core"
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

from pfe_core.adapter_store.quality import validate_adapter_artifact
from pfe_core.adapter_store.store import AdapterStore
from pfe_core.inference.engine import InferenceConfig, InferenceEngine
from pfe_core.phase41_simulated_review_preferences import (
    build_phase41_candidate_manifest,
    build_phase41_review_summary,
)
from pfe_core.phase42_reliability_hardening import (
    PHASE42_GENERIC_HOLDOUTS,
    build_phase41_v2_simulated_candidates,
    evaluate_adapter_generic_holdout,
)
from pfe_core.security.identifiers import safe_user_storage_id


EVIDENCE_ROOT = REPO_ROOT / "docs" / "demo" / "phase42-trustworthy-training-runtime-hardening"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _command(*args: str) -> dict[str, Any]:
    completed = subprocess.run(args, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    return {
        "command": list(args),
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _adapter_snapshot(store: AdapterStore, version: str) -> dict[str, Any]:
    version_dir = store.root / version
    manifest_path = version_dir / "adapter_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = []
    for path in sorted(version_dir.rglob("*")):
        if path.is_file():
            files.append(
                {
                    "path": str(path),
                    "relative_path": str(path.relative_to(version_dir)),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    manifest_summary = {
        key: manifest.get(key)
        for key in (
            "version",
            "workspace",
            "base_model",
            "created_at",
            "state",
            "num_samples",
            "artifact_format",
            "artifact_name",
            "training_backend",
            "inference_backend",
            "promoted_at",
            "eval_summary",
        )
        if key in manifest
    }
    return {
        "version": version,
        "version_dir": str(version_dir),
        "manifest": manifest_summary,
        "manifest_sha256": _sha256(manifest_path),
        "artifact_validation": validate_adapter_artifact(version_dir, manifest),
        "files": files,
    }


def _run_holdout(*, base_model: str, adapter_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    os.environ["PFE_ENABLE_REAL_LOCAL_INFERENCE"] = "1"
    base_engine = InferenceEngine(InferenceConfig(base_model=base_model))
    adapter_engine = InferenceEngine(InferenceConfig(base_model=base_model, adapter_path=str(adapter_dir)))
    base_outputs: list[dict[str, Any]] = []
    adapter_outputs: list[dict[str, Any]] = []
    for item in PHASE42_GENERIC_HOLDOUTS:
        messages = [{"role": "user", "content": item["prompt"]}]
        base_text = base_engine.generate(messages, max_tokens=64, temperature=0, metadata={"enable_real_local": True})
        base_outputs.append(
            {
                "holdout_id": item["id"],
                "prompt": item["prompt"],
                "expected_keywords": item["keywords"],
                "response": base_text,
                "generation": dict(base_engine.status().get("generation") or {}),
            }
        )
        adapter_text = adapter_engine.generate(messages, max_tokens=64, temperature=0, metadata={"enable_real_local": True})
        adapter_outputs.append(
            {
                "holdout_id": item["id"],
                "prompt": item["prompt"],
                "expected_keywords": item["keywords"],
                "response": adapter_text,
                "generation": dict(adapter_engine.status().get("generation") or {}),
            }
        )
    return base_outputs, adapter_outputs


def _candidate_evidence() -> dict[str, Any]:
    phase41 = REPO_ROOT / "docs" / "demo" / "phase41-simulated-review-preference-candidates"
    review_items = _jsonl(phase41 / "evidence-review" / "phase40_review_items_snapshot.jsonl")
    review_decisions = _jsonl(phase41 / "evidence-review" / "simulated_review_decisions.jsonl")
    review_summary = build_phase41_review_summary(
        review_items=review_items,
        review_decisions=review_decisions,
    )
    current = build_phase41_candidate_manifest(
        review_items=review_items,
        review_summary=review_summary,
    )
    v2 = build_phase41_v2_simulated_candidates(
        review_items=review_items,
        review_decisions=review_decisions,
    )
    return {"current_phase41": current, "phase41_v2": v2}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default="user_default")
    parser.add_argument("--home", type=Path, default=Path.home() / ".pfe")
    parser.add_argument("--clean-evidence", action="store_true")
    parser.add_argument("--apply-adapter-decision", action="store_true")
    parser.add_argument("--skip-real-holdout", action="store_true")
    parser.add_argument("--reuse-existing-holdout", action="store_true")
    args = parser.parse_args()

    if args.clean_evidence and EVIDENCE_ROOT.exists():
        shutil.rmtree(EVIDENCE_ROOT)
    for name in (
        "evidence-baseline",
        "evidence-adapter-gate",
        "evidence-real-training",
        "evidence-hermes-streaming",
        "evidence-security",
        "evidence-candidate-quality",
    ):
        (EVIDENCE_ROOT / name).mkdir(parents=True, exist_ok=True)

    store = AdapterStore(home=args.home, workspace=args.workspace)
    version = store.current_latest_version()
    if version is None:
        raise SystemExit("Phase42 baseline requires the currently promoted adapter before archival")
    snapshot = _adapter_snapshot(store, version)
    baseline = {
        "kind": "phase42_pre_change_baseline",
        "starting_commit": _command("git", "rev-parse", "HEAD"),
        "git_status": _command("git", "status", "--short", "--branch"),
        "pfe_doctor": _command(str(REPO_ROOT / ".venv" / "bin" / "pfe"), "doctor"),
        "pfe_next": _command(str(REPO_ROOT / ".venv" / "bin" / "pfe"), "next", "--workspace", args.workspace),
        "adapter": snapshot,
        "hermes_profile": {
            "path": str(Path.home() / ".hermes" / "profiles" / "pfeqwen36"),
            "config_exists": (Path.home() / ".hermes" / "profiles" / "pfeqwen36" / "config.yaml").exists(),
            "env_exists": (Path.home() / ".hermes" / "profiles" / "pfeqwen36" / ".env").exists(),
            "secrets_read": False,
        },
    }
    _write_json(EVIDENCE_ROOT / "evidence-baseline" / "pre_change_baseline.json", baseline)

    existing_adapter_report = EVIDENCE_ROOT / "evidence-adapter-gate" / "adapter_holdout_report.json"
    if args.reuse_existing_holdout:
        if not existing_adapter_report.exists():
            raise SystemExit("--reuse-existing-holdout requires an existing adapter_holdout_report.json")
        adapter_report = json.loads(existing_adapter_report.read_text(encoding="utf-8"))
    elif args.skip_real_holdout:
        adapter_report = {
            "kind": "phase42_adapter_generic_holdout_report",
            "passed": False,
            "holdout": {"count": 0, "passed": False},
            "training_leakage_detected": False,
            "reasons": ["real_holdout_skipped"],
        }
    else:
        manifest = snapshot["manifest"]
        base_outputs, adapter_outputs = _run_holdout(
            base_model=str(manifest.get("base_model")),
            adapter_dir=store.root / version,
        )
        _write_jsonl(EVIDENCE_ROOT / "evidence-adapter-gate" / "base_outputs.jsonl", base_outputs)
        _write_jsonl(EVIDENCE_ROOT / "evidence-adapter-gate" / "adapter_outputs.jsonl", adapter_outputs)
        base_report = evaluate_adapter_generic_holdout(base_outputs)
        adapter_report = evaluate_adapter_generic_holdout(adapter_outputs)
        adapter_report["version"] = version
        adapter_report["artifact_validation"] = snapshot["artifact_validation"]
        _write_json(EVIDENCE_ROOT / "evidence-adapter-gate" / "base_holdout_report.json", base_report)
    _write_json(EVIDENCE_ROOT / "evidence-adapter-gate" / "adapter_holdout_report.json", adapter_report)

    lifecycle = {"version": version, "decision_applied": False, "report_passed": adapter_report.get("passed")}
    if args.apply_adapter_decision:
        if adapter_report.get("passed") is True:
            gate = store.attach_serving_quality_report(version, adapter_report)
            lifecycle.update({"decision_applied": True, "action": "retained", "serving_gate": gate})
        else:
            message = store.archive_failed_serving_quality(version, adapter_report)
            lifecycle.update({"decision_applied": True, "action": "archived", "message": message})
        lifecycle["latest_after"] = store.current_latest_version()
        lifecycle["artifact_retained"] = (store.root / version).exists()
        lifecycle["pfe_next_after"] = _command(
            str(REPO_ROOT / ".venv" / "bin" / "pfe"), "next", "--workspace", args.workspace
        )
    _write_json(EVIDENCE_ROOT / "evidence-adapter-gate" / "lifecycle_decision.json", lifecycle)

    candidates = _candidate_evidence()
    current = candidates["current_phase41"]
    v2 = candidates["phase41_v2"]
    _write_json(EVIDENCE_ROOT / "evidence-candidate-quality" / "phase41_current_quality_gate.json", current)
    _write_json(EVIDENCE_ROOT / "evidence-candidate-quality" / "phase41_v2_manifest.json", v2)
    _write_jsonl(
        EVIDENCE_ROOT / "evidence-candidate-quality" / "phase41_v2_selected_preference_pairs.jsonl",
        v2.get("selected_preference_pairs") or [],
    )
    _write_json(
        EVIDENCE_ROOT / "evidence-security" / "path_identifier_probe.json",
        {
            "input": "../../escaped-profile",
            "safe_storage_id": safe_user_storage_id("../../escaped-profile"),
            "path_traversal_blocked": ".." not in safe_user_storage_id("../../escaped-profile"),
        },
    )

    print("PHASE42 EVIDENCE GENERATED")
    print(f"adapter_version: {version}")
    print(f"adapter_holdout_passed: {adapter_report.get('passed')}")
    print(f"decision_applied: {lifecycle.get('decision_applied')}")
    print(f"phase41_current: {current.get('training_candidate_status')}")
    print(f"phase41_v2: {v2.get('status')}")
    print(f"evidence_root: {EVIDENCE_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
