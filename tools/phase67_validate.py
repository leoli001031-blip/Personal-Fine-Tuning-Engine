#!/usr/bin/env python3
"""Run and persist the Phase67 validation matrix."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_ROOT = REPO_ROOT / "docs/demo/phase67-historical-contract-compatibility-audit"


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
        (EVIDENCE_ROOT / "phase67-final-decision.json").read_text(encoding="utf-8")
    )
    comparison = json.loads(
        (EVIDENCE_ROOT / "comparison_summary.json").read_text(encoding="utf-8")
    )
    partition = json.loads(
        (EVIDENCE_ROOT / "historical_partition.json").read_text(encoding="utf-8")
    )
    passed = (
        integrity.get("passed") is True
        and decision.get("recommendation")
        == "recommend_phase67_contract_aware_partition_for_manual_review_only"
        and comparison.get("phase66_fresh_current_contract", {}).get("accuracy")
        == 1.0
        and comparison.get("aligned_legacy_phase55_regression", {}).get("accuracy")
        == 0.7333
        and partition.get("aligned_legacy_regression_phases") == ["phase55"]
        and int(partition.get("aligned_legacy_regression_count") or 0) == 150
        and int(partition.get("legacy_diagnostic_only_count") or 0) == 408
        and int(partition.get("automatic_relabel_count") or 0) == 0
        and comparison.get("current_evaluator_qualified_for_runtime_ab") is False
        and int(comparison.get("model_call_count") or 0) == 0
        and decision.get("runtime_ab_allowed") is False
        and decision.get("training_allowed") is False
        and int(comparison.get("actual_user_feedback_count") or 0) == 0
    )
    return {
        "name": "phase67_evidence_consistency",
        "command": ["internal", "phase67_evidence_consistency"],
        "returncode": 0 if passed else 1,
        "passed": passed,
        "duration_seconds": round(time.perf_counter() - started, 4),
        "output_line_count": 1,
        "output_sha256": hashlib.sha256(str(passed).encode("utf-8")).hexdigest(),
        "output_tail": [
            f"integrity={integrity.get('passed')} recommendation={decision.get('recommendation')} "
            f"aligned={partition.get('aligned_legacy_regression_count')} "
            f"diagnostic={partition.get('legacy_diagnostic_only_count')}"
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
                "pfe-core/pfe_core/phase67_historical_contract_compatibility_audit.py",
                "tools/phase67_prepare.py",
                "tools/phase67_finalize_evidence.py",
                "tools/phase67_validate.py",
                "tests/test_phase67_historical_contract_compatibility_audit.py",
            ],
        ),
        (
            "phase67_focused_and_phase66_to_45_regression",
            [
                python,
                "-m",
                "pytest",
                "-q",
                *[
                    f"tests/test_phase{phase}_{name}.py"
                    for phase, name in (
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
        "kind": "phase67_validation_summary",
        "created_at": _utcnow(),
        "status": "passed" if passed else "failed",
        "check_count": len(results),
        "passed_count": sum(row["passed"] for row in results),
        "failed_count": sum(not row["passed"] for row in results),
        "checks": results,
    }
    _write_json(EVIDENCE_ROOT / "validation_summary.json", summary)
    lines = [f"Phase67 validation: {summary['status']}"]
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
