#!/usr/bin/env python3
"""Validate the complete blocked Phase72 evidence package."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_ROOT = REPO_ROOT / "docs/demo/phase72-deterministic-boundary-serializer"


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
        "output_sha256": hashlib.sha256(combined.encode()).hexdigest(),
        "output_tail": lines[-24:],
    }


def _evidence_check() -> dict[str, Any]:
    integrity = json.loads((EVIDENCE_ROOT / "evidence_integrity.json").read_text())
    decision = json.loads((EVIDENCE_ROOT / "phase72-final-decision.json").read_text())
    comparison = json.loads((EVIDENCE_ROOT / "comparison_summary.json").read_text())
    audit = json.loads((EVIDENCE_ROOT / "post_call_packaging_audit.json").read_text())
    passed = (
        integrity.get("passed") is True
        and integrity.get("blocked_evidence_complete") is True
        and integrity.get("failed_stage") == "wire_preflight"
        and decision.get("recommendation")
        == "hold_phase72_deterministic_boundary_serializer"
        and decision.get("experiment_status") == "blocked_at_wire_preflight"
        and comparison.get("actual_judge_output_counts")
        == {"sparse_preflight": 34, "phase68_regression": 0, "product": 0}
        and comparison.get("actual_generation_call_count") == 0
        and comparison.get("actual_model_output_count_total") == 34
        and comparison.get("actual_user_feedback_count") == 0
        and comparison.get("training_executed") is False
        and comparison.get("adapter_created") is False
        and comparison.get("product_default_changed") is False
        and decision.get("auto_promote_allowed") is False
        and audit.get("passed") is True
        and audit.get("frozen_source_changes") == []
    )
    return {
        "name": "phase72_blocked_evidence_consistency",
        "command": ["internal", "phase72_blocked_evidence_consistency"],
        "returncode": 0 if passed else 1,
        "passed": passed,
        "duration_seconds": 0.0,
        "output_line_count": 1,
        "output_sha256": hashlib.sha256(str(passed).encode()).hexdigest(),
        "output_tail": [
            f"integrity={integrity.get('passed')} recommendation={decision.get('recommendation')} outputs={comparison.get('actual_model_output_count_total')}"
        ],
    }


def main() -> int:
    python = str(REPO_ROOT / ".venv/bin/python")
    phase_tests = [
        f"tests/test_phase{phase}_{name}.py"
        for phase, name in (
            (72, "deterministic_boundary_serializer"),
            (71, "qualified_structured_contract_ab"),
            (70, "structured_boundary_contract"),
            (69, "minimal_runtime_ab"),
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
    ]
    checks = [
        (
            "py_compile",
            [
                python,
                "-m",
                "py_compile",
                "pfe-core/pfe_core/phase72_deterministic_boundary_serializer.py",
                "tools/phase72_deterministic_boundary_serializer.py",
                "tools/phase72_finalize_blocked.py",
                "tools/phase72_validate_blocked.py",
                "tests/test_phase72_deterministic_boundary_serializer.py",
            ],
        ),
        (
            "phase72_focused_and_phase71_to45_regression",
            [python, "-m", "pytest", "-q", *phase_tests],
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
        "kind": "phase72_validation_summary",
        "created_at": _utcnow(),
        "status": "passed" if passed else "failed",
        "check_count": len(results),
        "passed_count": sum(row["passed"] for row in results),
        "failed_count": sum(not row["passed"] for row in results),
        "checks": results,
    }
    _write_json(EVIDENCE_ROOT / "validation_summary.json", summary)
    lines = [f"Phase72 validation: {summary['status']}"]
    lines.extend(
        f"{row['name']}: {'PASS' if row['passed'] else 'FAIL'} ({row['duration_seconds']}s, rc={row['returncode']})"
        for row in results
    )
    _write_text(EVIDENCE_ROOT / "validation_gate.txt", "\n".join(lines))
    print("\n".join(lines))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
