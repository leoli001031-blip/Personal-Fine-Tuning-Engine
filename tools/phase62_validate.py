#!/usr/bin/env python3
"""Run and persist the Phase62 validation matrix."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_ROOT = REPO_ROOT / "docs/demo/phase62-risk-asymmetric-candidate-consensus"


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _write_text(path: Path, value: str) -> None:
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def _run(name: str, command: list[str]) -> dict[str, Any]:
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    combined = completed.stdout + completed.stderr
    lines = combined.splitlines()
    return {
        "name": name,
        "command": command,
        "returncode": completed.returncode,
        "passed": completed.returncode == 0,
        "duration_seconds": round(time.perf_counter() - started, 4),
        "output_line_count": len(lines),
        "output_sha256": hashlib.sha256(combined.encode("utf-8")).hexdigest(),
        "output_tail": lines[-24:],
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _manifest_verified() -> bool:
    manifest = _read_json(EVIDENCE_ROOT / "evidence_manifest.json")
    rows = list(manifest.get("files") or [])
    return bool(rows) and all(
        (REPO_ROOT / row["path"]).is_file()
        and hashlib.sha256((REPO_ROOT / row["path"]).read_bytes()).hexdigest() == row["sha256"]
        for row in rows
    )


def _evidence_check() -> dict[str, Any]:
    started = time.perf_counter()
    integrity = _read_json(EVIDENCE_ROOT / "evidence_integrity.json")
    decision = _read_json(EVIDENCE_ROOT / "phase62-final-decision.json")
    comparison = _read_json(EVIDENCE_ROOT / "comparison_summary.json")
    phase62 = dict(comparison.get("phase62") or {})
    preflight_passed = phase62.get("protocol_preflight_status") == "passed"
    calibration_qualified = phase62.get("calibration_status") == "qualified"
    holdout_qualified = phase62.get("holdout_status") == "qualified"
    conflict_free = int(phase62.get("holdout_candidate_value_conflict_count") or 0) == 0
    expected = (
        "recommend_phase62_risk_asymmetric_consensus_for_manual_review_only"
        if preflight_passed and calibration_qualified and holdout_qualified and conflict_free
        else "hold_phase62_risk_asymmetric_candidate_consensus"
    )
    preflight_outcomes = int(phase62.get("protocol_preflight_successful_model_output_count") or 0) + int(
        phase62.get("protocol_preflight_failed_judge_item_count") or 0
    )
    calibration_outcomes = int(phase62.get("calibration_successful_model_output_count") or 0) + int(
        phase62.get("calibration_failed_judge_item_count") or 0
    )
    holdout_outcomes = int(phase62.get("holdout_successful_model_output_count") or 0) + int(
        phase62.get("holdout_failed_judge_item_count") or 0
    )
    raw_rows = []
    for directory_name in ("evidence-protocol-preflight", "evidence-evaluator-calibration", "evidence-evaluator-holdout"):
        directory = EVIDENCE_ROOT / directory_name
        for alias in ("semantic_judge_alpha", "semantic_judge_beta"):
            raw_rows.extend(_read_jsonl(directory / f"wire_failure_attempts_{alias}.jsonl"))
    raw_hashes_valid = all(
        row.get("raw_response_sha256")
        == hashlib.sha256(str(row.get("raw_response") or "").encode("utf-8")).hexdigest()
        for row in raw_rows
    )
    passed = (
        integrity.get("passed") is True
        and _manifest_verified()
        and decision.get("recommendation") == expected
        and preflight_outcomes == 12
        and ((preflight_passed and calibration_outcomes == 60) or (not preflight_passed and calibration_outcomes == 0))
        and (
            (preflight_passed and calibration_qualified and holdout_outcomes == 120)
            or (not (preflight_passed and calibration_qualified) and holdout_outcomes == 0)
        )
        and raw_hashes_valid
        and decision.get("runtime_replay_allowed_in_phase62") is False
        and decision.get("new_training_allowed") is False
        and decision.get("product_default_change_allowed") is False
        and int(comparison.get("actual_user_feedback_count") or 0) == 0
    )
    return {
        "name": "phase62_evidence_consistency",
        "command": ["internal", "phase62_evidence_consistency"],
        "returncode": 0 if passed else 1,
        "passed": passed,
        "duration_seconds": round(time.perf_counter() - started, 4),
        "output_line_count": 1,
        "output_sha256": hashlib.sha256(str(passed).encode("utf-8")).hexdigest(),
        "output_tail": [
            f"integrity={integrity.get('passed')} manifest={_manifest_verified()} "
            f"decision={decision.get('recommendation')} preflight={preflight_outcomes} "
            f"calibration={calibration_outcomes} holdout={holdout_outcomes} raw_failures={len(raw_rows)}"
        ],
    }


def main() -> int:
    python = str(REPO_ROOT / ".venv/bin/python")
    regression_tests = [
        "tests/test_phase62_risk_asymmetric_candidate_consensus.py",
        "tests/test_phase61_compact_candidate_wire_protocol.py",
        "tests/test_phase60_flat_schema_compatibility.py",
        "tests/test_phase59_proposition_addressed_grounding.py",
        "tests/test_phase58_clause_addressed_grounding.py",
        "tests/test_phase57_span_evaluator_historical_replay.py",
        "tests/test_phase56_evidence_span_grounded_atomic.py",
        "tests/test_phase55_atomic_boundary_composition.py",
        "tests/test_phase54_typed_proposition_evaluator.py",
        "tests/test_phase53_evaluator_scope_recovery.py",
        "tests/test_phase52_adversarial_evaluator_generalization.py",
        "tests/test_phase51_dual_evaluator_hardening.py",
        "tests/test_phase50_conditional_provenance_guard.py",
        "tests/test_phase49_provenance_boundary_recovery.py",
        "tests/test_phase48_compact_intent_runtime.py",
        "tests/test_phase47_simulated_user_review.py",
        "tests/test_phase46_runtime_first_latest_intent.py",
        "tests/test_phase45_privacy_multiturn_preference.py",
    ]
    checks = [
        (
            "py_compile",
            [
                python,
                "-m",
                "py_compile",
                "pfe-core/pfe_core/phase62_risk_asymmetric_candidate_consensus.py",
                "tools/phase62_prepare.py",
                "tools/phase62_execute.py",
                "tools/phase62_finalize_evidence.py",
                "tools/phase62_validate.py",
                "tests/test_phase62_risk_asymmetric_candidate_consensus.py",
            ],
        ),
        ("phase62_focused_and_phase61_to_45_regression", [python, "-m", "pytest", "-q", *regression_tests]),
        ("test_unit", ["make", "test-unit"]),
        ("test_surface", ["make", "test-surface"]),
        ("test_e2e_mock", ["make", "test-e2e-mock"]),
        ("smoke_beta", ["make", "smoke-beta"]),
        ("git_diff_check", ["git", "diff", "--check"]),
    ]
    results = [_evidence_check()]
    results.extend(_run(name, command) for name, command in checks)
    passed = all(row["passed"] for row in results)
    summary = {
        "kind": "phase62_validation_summary",
        "created_at": _utcnow(),
        "status": "passed" if passed else "failed",
        "check_count": len(results),
        "passed_count": sum(row["passed"] for row in results),
        "failed_count": sum(not row["passed"] for row in results),
        "checks": results,
    }
    _write_json(EVIDENCE_ROOT / "validation_summary.json", summary)
    lines = [f"Phase62 validation: {summary['status']}"]
    lines.extend(
        f"{row['name']}: {'PASS' if row['passed'] else 'FAIL'} ({row['duration_seconds']}s, rc={row['returncode']})"
        for row in results
    )
    _write_text(EVIDENCE_ROOT / "validation_gate.txt", "\n".join(lines))
    print("\n".join(lines))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
