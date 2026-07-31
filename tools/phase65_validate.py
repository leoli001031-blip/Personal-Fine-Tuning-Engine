#!/usr/bin/env python3
"""Run and persist the Phase65 validation matrix."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_ROOT = REPO_ROOT / "docs/demo/phase65-aggregate-safe-boundary-coverage"


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_text(path: Path, value: str) -> None:
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def _run(name: str, command: list[str]) -> dict[str, Any]:
    started = time.perf_counter()
    completed = subprocess.run(
        command, cwd=REPO_ROOT, text=True, capture_output=True, check=False
    )
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


def _evidence_check() -> dict[str, Any]:
    started = time.perf_counter()
    integrity = json.loads(
        (EVIDENCE_ROOT / "evidence_integrity.json").read_text(encoding="utf-8")
    )
    decision = json.loads(
        (EVIDENCE_ROOT / "phase65-final-decision.json").read_text(encoding="utf-8")
    )
    comparison = json.loads(
        (EVIDENCE_ROOT / "comparison_summary.json").read_text(encoding="utf-8")
    )
    phase65 = dict(comparison.get("phase65") or {})
    expected = (
        "recommend_phase65_scope_aware_candidates_for_manual_review_only"
        if phase65.get("preflight_status") == "passed"
        and phase65.get("calibration_status") == "qualified"
        and phase65.get("holdout_status") == "qualified"
        and int(phase65.get("holdout_false_accept_count") or 0) == 0
        and int(phase65.get("holdout_schema_failure_count") or 0) == 0
        and int(phase65.get("holdout_candidate_value_conflict_count") or 0) == 0
        else "hold_phase65_aggregate_safe_boundary_coverage"
    )
    passed = (
        integrity.get("passed") is True
        and decision.get("recommendation") == expected
        and comparison.get("phase64", {}).get("recommendation")
        == "hold_phase64_field_typed_historical_replay"
        and int(comparison.get("actual_user_feedback_count") or 0) == 0
        and comparison.get("training_executed") is False
        and comparison.get("runtime_replay_executed") is False
        and decision.get("runtime_replay_allowed_in_phase65") is False
        and decision.get("new_training_allowed") is False
        and decision.get("product_default_change_allowed") is False
    )
    return {
        "name": "phase65_evidence_consistency",
        "command": ["internal", "phase65_evidence_consistency"],
        "returncode": 0 if passed else 1,
        "passed": passed,
        "duration_seconds": round(time.perf_counter() - started, 4),
        "output_line_count": 1,
        "output_sha256": hashlib.sha256(str(passed).encode("utf-8")).hexdigest(),
        "output_tail": [
            f"integrity={integrity.get('passed')} decision={decision.get('recommendation')} "
            f"calibration={phase65.get('calibration_accuracy')} "
            f"holdout={phase65.get('holdout_accuracy')}"
        ],
    }


def main() -> int:
    python = str(REPO_ROOT / ".venv/bin/python")
    checks = [
        (
            "py_compile",
            [
                python,
                "-m",
                "py_compile",
                "pfe-core/pfe_core/phase59_proposition_addressed_grounding.py",
                "pfe-core/pfe_core/phase65_aggregate_safe_boundary_coverage.py",
                "tools/phase65_prepare.py",
                "tools/phase65_execute.py",
                "tools/phase65_finalize_evidence.py",
                "tools/phase65_validate.py",
                "tests/test_phase65_aggregate_safe_boundary_coverage.py",
            ],
        ),
        (
            "phase65_focused_and_phase64_to_45_regression",
            [
                python,
                "-m",
                "pytest",
                "-q",
                "tests/test_phase65_aggregate_safe_boundary_coverage.py",
                "tests/test_phase64_field_typed_historical_replay.py",
                "tests/test_phase63_field_typed_candidate_wire.py",
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
            ],
        ),
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
        "kind": "phase65_validation_summary",
        "created_at": _utcnow(),
        "status": "passed" if passed else "failed",
        "check_count": len(results),
        "passed_count": sum(row["passed"] for row in results),
        "failed_count": sum(not row["passed"] for row in results),
        "checks": results,
    }
    _write_json(EVIDENCE_ROOT / "validation_summary.json", summary)
    lines = [f"Phase65 validation: {summary['status']}"]
    lines.extend(
        f"{row['name']}: {'PASS' if row['passed'] else 'FAIL'} "
        f"({row['duration_seconds']}s, rc={row['returncode']})"
        for row in results
    )
    _write_text(EVIDENCE_ROOT / "validation_gate.txt", "\n".join(lines))
    print("\n".join(lines))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
