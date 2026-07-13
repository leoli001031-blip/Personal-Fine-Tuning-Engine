#!/usr/bin/env python3
"""Run and persist the Phase48 release validation matrix."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_ROOT = REPO_ROOT / "docs" / "demo" / "phase48-compact-intent-runtime-ablation"


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
        "output_tail": lines[-20:],
    }


def _evidence_check() -> dict[str, Any]:
    started = time.perf_counter()
    integrity = json.loads((EVIDENCE_ROOT / "evidence_integrity.json").read_text(encoding="utf-8"))
    decision = json.loads((EVIDENCE_ROOT / "phase48-final-decision.json").read_text(encoding="utf-8"))
    comparison = json.loads((EVIDENCE_ROOT / "comparison_summary.json").read_text(encoding="utf-8"))
    passed = (
        integrity.get("passed") is True
        and decision.get("recommendation") == "hold_compact_runtime"
        and decision.get("new_training_allowed") is False
        and decision.get("product_default_change_allowed") is False
        and int(comparison.get("formal_qwen_real_model_calls") or 0) == 576
        and int(comparison.get("independent_gemma_real_model_calls") or 0) == 128
        and int(comparison.get("actual_user_feedback_count") or 0) == 0
    )
    return {
        "name": "phase48_evidence_consistency",
        "command": ["internal", "phase48_evidence_consistency"],
        "returncode": 0 if passed else 1,
        "passed": passed,
        "duration_seconds": round(time.perf_counter() - started, 4),
        "output_line_count": 1,
        "output_sha256": hashlib.sha256(str(passed).encode("utf-8")).hexdigest(),
        "output_tail": [
            f"integrity={integrity.get('passed')} decision={decision.get('recommendation')} "
            f"qwen_calls={comparison.get('formal_qwen_real_model_calls')} "
            f"gemma_calls={comparison.get('independent_gemma_real_model_calls')}"
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
                "pfe-core/pfe_core/phase45_privacy_multiturn_preference.py",
                "pfe-core/pfe_core/phase48_compact_intent_runtime.py",
                "tools/phase48_prepare.py",
                "tools/phase48_qwen3_4b_generate.py",
                "tools/phase48_blind_eval.py",
                "tools/phase48_finalize_evidence.py",
                "tools/phase48_validate.py",
                "tests/test_phase48_compact_intent_runtime.py",
            ],
        ),
        (
            "phase48_focused_and_phase47_46_45_regression",
            [
                python,
                "-m",
                "pytest",
                "-q",
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
        "kind": "phase48_validation_summary",
        "created_at": _utcnow(),
        "status": "passed" if passed else "failed",
        "check_count": len(results),
        "passed_count": sum(row["passed"] for row in results),
        "failed_count": sum(not row["passed"] for row in results),
        "checks": results,
    }
    _write_json(EVIDENCE_ROOT / "validation_summary.json", summary)
    lines = [f"Phase48 validation: {summary['status']}"]
    lines.extend(
        f"{row['name']}: {'PASS' if row['passed'] else 'FAIL'} ({row['duration_seconds']}s, rc={row['returncode']})"
        for row in results
    )
    _write_text(EVIDENCE_ROOT / "validation_gate.txt", "\n".join(lines))
    print("\n".join(lines))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
