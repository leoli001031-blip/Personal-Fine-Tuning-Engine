#!/usr/bin/env python3
"""Run and persist the Phase68 validation matrix."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_ROOT = REPO_ROOT / "docs/demo/phase68-aligned-candidate-scope-recovery"


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
        (EVIDENCE_ROOT / "phase68-final-decision.json").read_text(encoding="utf-8")
    )
    comparison = json.loads(
        (EVIDENCE_ROOT / "comparison_summary.json").read_text(encoding="utf-8")
    )
    fresh_cal = dict(comparison.get("phase68_fresh_calibration") or {})
    fresh_hold = dict(comparison.get("phase68_fresh_holdout") or {})
    aligned = dict(comparison.get("phase68_aligned_phase55_label_regression") or {})
    passed = (
        integrity.get("passed") is True
        and decision.get("recommendation")
        == "recommend_phase68_evaluator_qualification_for_manual_review_only"
        and fresh_cal.get("status") == "qualified"
        and fresh_cal.get("accuracy") == 1.0
        and fresh_cal.get("typed_exact_match_rate") == 1.0
        and fresh_hold.get("status") == "qualified"
        and fresh_hold.get("accuracy") == 1.0
        and fresh_hold.get("typed_exact_match_rate") == 1.0
        and aligned.get("accuracy") == 1.0
        and aligned.get("accuracy_delta_from_phase66") == 0.2667
        and int(aligned.get("false_accept_count") or 0) == 0
        and int(aligned.get("schema_failure_count") or 0) == 0
        and int(aligned.get("candidate_value_conflict_count") or 0) == 0
        and int(comparison.get("actual_model_output_count") or 0) == 672
        and decision.get("phase69_minimal_runtime_ab_design_eligible") is True
        and decision.get("runtime_ab_allowed_in_phase68") is False
        and decision.get("training_allowed") is False
        and int(comparison.get("actual_user_feedback_count") or 0) == 0
    )
    return {
        "name": "phase68_evidence_consistency",
        "command": ["internal", "phase68_evidence_consistency"],
        "returncode": 0 if passed else 1,
        "passed": passed,
        "duration_seconds": round(time.perf_counter() - started, 4),
        "output_line_count": 1,
        "output_sha256": hashlib.sha256(str(passed).encode("utf-8")).hexdigest(),
        "output_tail": [
            f"integrity={integrity.get('passed')} recommendation={decision.get('recommendation')} "
            f"fresh={fresh_hold.get('accuracy')} aligned={aligned.get('accuracy')}"
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
                "pfe-core/pfe_core/phase56_evidence_span_grounded_atomic.py",
                "pfe-core/pfe_core/phase62_risk_asymmetric_candidate_consensus.py",
                "pfe-core/pfe_core/phase68_aligned_candidate_scope_recovery.py",
                "tools/phase68_prepare.py",
                "tools/phase68_execute.py",
                "tools/phase68_finalize_evidence.py",
                "tools/phase68_validate.py",
                "tests/test_phase68_aligned_candidate_scope_recovery.py",
            ],
        ),
        (
            "phase68_focused_and_phase67_to_45_regression",
            [
                python,
                "-m",
                "pytest",
                "-q",
                *[
                    f"tests/test_phase{phase}_{name}.py"
                    for phase, name in (
                        (68, "aligned_candidate_scope_recovery"),
                        (67, "historical_contract_compatibility_audit"),
                        (66, "external_distribution_regression"),
                        (65, "aggregate_safe_boundary_coverage"),
                        (64, "field_typed_historical_replay"),
                        (63, "field_typed_candidate_wire"),
                        (62, "risk_asymmetric_candidate_consensus"),
                        (61, "compact_candidate_wire_protocol"),
                        (60, "flat_schema_compatibility"),
                        (59, "proposition_addressed_grounding"),
                        (58, "clause_addressed_grounding"),
                        (57, "span_evaluator_historical_replay"),
                        (56, "evidence_span_grounded_atomic"),
                        (55, "atomic_boundary_composition"),
                        (54, "typed_proposition_evaluator"),
                        (53, "evaluator_scope_recovery"),
                        (52, "adversarial_evaluator_generalization"),
                        (51, "dual_evaluator_hardening"),
                        (50, "conditional_provenance_guard"),
                        (49, "provenance_boundary_recovery"),
                        (48, "compact_intent_runtime"),
                        (47, "simulated_user_review"),
                        (46, "runtime_first_latest_intent"),
                        (45, "privacy_multiturn_preference"),
                    )
                ],
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
        "kind": "phase68_validation_summary",
        "created_at": _utcnow(),
        "status": "passed" if passed else "failed",
        "check_count": len(results),
        "passed_count": sum(row["passed"] for row in results),
        "failed_count": sum(not row["passed"] for row in results),
        "checks": results,
    }
    _write_json(EVIDENCE_ROOT / "validation_summary.json", summary)
    lines = [f"Phase68 validation: {summary['status']}"]
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
