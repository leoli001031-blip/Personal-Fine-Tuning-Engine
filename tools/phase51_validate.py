#!/usr/bin/env python3
"""Run and persist the Phase51 release validation matrix."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_ROOT = REPO_ROOT / "docs" / "demo" / "phase51-dual-evaluator-hardening"


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _evidence_check() -> dict[str, Any]:
    started = time.perf_counter()
    integrity = json.loads((EVIDENCE_ROOT / "evidence_integrity.json").read_text(encoding="utf-8"))
    decision = json.loads((EVIDENCE_ROOT / "phase51-final-decision.json").read_text(encoding="utf-8"))
    comparison = json.loads((EVIDENCE_ROOT / "comparison_summary.json").read_text(encoding="utf-8"))
    audit = json.loads(
        (EVIDENCE_ROOT / "evidence-posthoc-veto/evaluator_runtime_generalization_audit.json").read_text(
            encoding="utf-8"
        )
    )
    semantic = dict(comparison.get("runtime", {}).get("semantic_by_variant") or {})
    passed = (
        integrity.get("passed") is True
        and decision.get("recommendation") == "hold_evaluator_runtime_generalization_gap"
        and decision.get("manual_shadow_trial_allowed") is False
        and decision.get("product_default_change_allowed") is False
        and decision.get("new_training_allowed") is False
        and comparison.get("formal_evaluator", {}).get("calibration_status") == "qualified"
        and comparison.get("formal_evaluator", {}).get("holdout_status") == "qualified"
        and float(comparison.get("formal_evaluator", {}).get("holdout_accuracy") or 0.0) == 0.9833
        and int(comparison.get("formal_evaluator", {}).get("holdout_false_accept_count") or 0) == 0
        and int(comparison.get("runtime", {}).get("qwen3_4b_real_model_call_count") or 0) == 432
        and int(comparison.get("runtime", {}).get("semantic_judge_real_model_call_count") or 0) == 144
        and float(semantic.get("base_compact_v1", {}).get("accept_rate") or 0.0) == 0.625
        and float(semantic.get("base_conditional_guard", {}).get("accept_rate") or 0.0) == 0.7917
        and float(semantic.get("base_global_v2", {}).get("accept_rate") or 0.0) == 0.8333
        and int(audit.get("likely_hard_reject_false_positive_count") or 0) == 1
        and audit.get("posthoc_audit_can_promote") is False
        and int(comparison.get("actual_user_feedback_count") or 0) == 0
    )
    return {
        "name": "phase51_evidence_consistency",
        "command": ["internal", "phase51_evidence_consistency"],
        "returncode": 0 if passed else 1,
        "passed": passed,
        "duration_seconds": round(time.perf_counter() - started, 4),
        "output_line_count": 1,
        "output_sha256": hashlib.sha256(str(passed).encode("utf-8")).hexdigest(),
        "output_tail": [
            f"integrity={integrity.get('passed')} decision={decision.get('recommendation')} "
            f"eval={comparison.get('formal_evaluator', {}).get('holdout_accuracy')} "
            f"runtime_calls={comparison.get('runtime', {}).get('qwen3_4b_real_model_call_count')}+"
            f"{comparison.get('runtime', {}).get('semantic_judge_real_model_call_count')} "
            f"hard_false_positive={audit.get('likely_hard_reject_false_positive_count')}"
        ],
    }


def main() -> int:
    python = str(REPO_ROOT / ".venv" / "bin" / "python")
    checks = [
        (
            "py_compile",
            [
                python,
                "-m",
                "py_compile",
                "pfe-core/pfe_core/phase51_dual_evaluator_hardening.py",
                "tools/phase51_prepare.py",
                "tools/phase51_dual_evaluator.py",
                "tools/phase51_qwen3_4b_generate.py",
                "tools/phase51_finalize_evidence.py",
                "tools/phase51_validate.py",
                "tests/test_phase51_dual_evaluator_hardening.py",
            ],
        ),
        (
            "phase51_focused_and_phase50_49_48_47_46_45_regression",
            [
                python,
                "-m",
                "pytest",
                "-q",
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
        "kind": "phase51_validation_summary",
        "created_at": _utcnow(),
        "status": "passed" if passed else "failed",
        "check_count": len(results),
        "passed_count": sum(row["passed"] for row in results),
        "failed_count": sum(not row["passed"] for row in results),
        "checks": results,
    }
    _write_json(EVIDENCE_ROOT / "validation_summary.json", summary)
    lines = [f"Phase51 validation: {summary['status']}"]
    lines.extend(
        f"{row['name']}: {'PASS' if row['passed'] else 'FAIL'} ({row['duration_seconds']}s, rc={row['returncode']})"
        for row in results
    )
    _write_text(EVIDENCE_ROOT / "validation_gate.txt", "\n".join(lines))
    print("\n".join(lines))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
