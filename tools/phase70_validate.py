#!/usr/bin/env python3
"""Run and persist the Phase70 validation matrix."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_ROOT = REPO_ROOT / "docs/demo/phase70-structured-boundary-contract"


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
        "output_sha256": hashlib.sha256(combined.encode()).hexdigest(),
        "output_tail": lines[-24:],
    }


def _evidence_check() -> dict[str, Any]:
    integrity = json.loads((EVIDENCE_ROOT / "evidence_integrity.json").read_text())
    decision = json.loads((EVIDENCE_ROOT / "phase70-final-decision.json").read_text())
    comparison = json.loads((EVIDENCE_ROOT / "comparison_summary.json").read_text())
    allowed = {
        "recommend_phase70_structured_contract_for_manual_review_only",
        "hold_phase70_structured_boundary_contract",
    }
    full_experiment = (
        integrity.get("passed") is True
        and decision.get("recommendation") in allowed
        and comparison.get("actual_generation_call_count") == 288
        and comparison.get("actual_judge_output_counts")
        == {"sparse_preflight": 24, "phase68_regression": 60, "product": 144}
        and comparison.get("actual_model_output_count_total") == 516
        and comparison.get("actual_user_feedback_count") == 0
        and comparison.get("training_executed") is False
        and comparison.get("adapter_created") is False
        and comparison.get("product_default_changed") is False
        and decision.get("auto_promote_allowed") is False
    )
    blocked_preflight = (
        integrity.get("passed") is True
        and integrity.get("experiment_succeeded") is False
        and integrity.get("blocked_evidence_complete") is True
        and decision.get("recommendation") == "hold_phase70_structured_boundary_contract"
        and decision.get("experiment_status") == "blocked_before_product_ab"
        and decision.get("transport_envelope_qualified") is True
        and decision.get("full_sparse_composer_qualification") is False
        and comparison.get("actual_generation_call_count") == 0
        and comparison.get("actual_judge_output_counts")
        == {"sparse_preflight": 24, "phase68_regression": 0, "product": 0}
        and comparison.get("actual_model_output_count_total") == 24
        and comparison.get("actual_user_feedback_count") == 0
        and comparison.get("training_executed") is False
        and comparison.get("adapter_created") is False
        and comparison.get("product_default_changed") is False
        and decision.get("auto_promote_allowed") is False
    )
    passed = full_experiment or blocked_preflight
    return {
        "name": "phase70_evidence_consistency",
        "command": ["internal", "phase70_evidence_consistency"],
        "returncode": 0 if passed else 1,
        "passed": passed,
        "duration_seconds": 0.0,
        "output_line_count": 1,
        "output_sha256": hashlib.sha256(str(passed).encode()).hexdigest(),
        "output_tail": [f"integrity={integrity.get('passed')} recommendation={decision.get('recommendation')} status={decision.get('experiment_status', 'completed')} outputs={comparison.get('actual_model_output_count_total')}"],
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
                "pfe-core/pfe_core/phase70_structured_boundary_contract.py",
                "tools/phase70_prepare.py",
                "tools/phase70_generate.py",
                "tools/phase70_prepare_product_eval.py",
                "tools/phase70_execute_eval.py",
                "tools/phase70_finalize_evidence.py",
                "tools/phase70_validate.py",
                "tests/test_phase70_structured_boundary_contract.py",
            ],
        ),
        (
            "phase70_focused_and_phase69_to45_regression",
            [
                python,
                "-m",
                "pytest",
                "-q",
                *[
                    f"tests/test_phase{phase}_{name}.py"
                    for phase, name in (
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
        "kind": "phase70_validation_summary",
        "created_at": _utcnow(),
        "status": "passed" if passed else "failed",
        "check_count": len(results),
        "passed_count": sum(row["passed"] for row in results),
        "failed_count": sum(not row["passed"] for row in results),
        "checks": results,
    }
    _write_json(EVIDENCE_ROOT / "validation_summary.json", summary)
    lines = [f"Phase70 validation: {summary['status']}"]
    lines.extend(
        f"{row['name']}: {'PASS' if row['passed'] else 'FAIL'} ({row['duration_seconds']}s, rc={row['returncode']})"
        for row in results
    )
    _write_text(EVIDENCE_ROOT / "validation_gate.txt", "\n".join(lines))
    print("\n".join(lines))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
