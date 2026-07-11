#!/usr/bin/env python3
"""Build the Phase42 decision only from persisted machine evidence."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = REPO_ROOT / "pfe-core"
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

from pfe_core.phase42_reliability_hardening import build_phase42_final_decision


ROOT = REPO_ROOT / "docs" / "demo" / "phase42-trustworthy-training-runtime-hardening"


def _load(relative: str) -> dict[str, Any]:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def _write(relative: str, payload: Any) -> None:
    path = ROOT / relative
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    baseline_path = ROOT / "evidence-baseline/pre_change_baseline.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    adapter_snapshot = baseline.get("adapter") if isinstance(baseline.get("adapter"), dict) else {}
    manifest = adapter_snapshot.get("manifest") if isinstance(adapter_snapshot.get("manifest"), dict) else {}
    if manifest:
        adapter_snapshot["manifest"] = {
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
        baseline["adapter"] = adapter_snapshot
        baseline_path.write_text(
            json.dumps(baseline, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
    adapter = _load("evidence-adapter-gate/adapter_holdout_report.json")
    base = _load("evidence-adapter-gate/base_holdout_report.json")
    lifecycle = _load("evidence-adapter-gate/lifecycle_decision.json")
    training = _load("evidence-real-training/training_attempt.json")
    context = _load("evidence-hermes-streaming/context_budget_smoke.json")
    phase41 = _load("evidence-candidate-quality/phase41_current_quality_gate.json")
    phase41_v2 = _load("evidence-candidate-quality/phase41_v2_manifest.json")
    streaming_text = (ROOT / "evidence-hermes-streaming/hermes_openai_sdk_live_smoke.txt").read_text(encoding="utf-8")
    security_text = (ROOT / "evidence-security/phase42_focused_tests.txt").read_text(encoding="utf-8")
    validation_text = (ROOT / "validation_gate.txt").read_text(encoding="utf-8")
    hermes_passed = (
        "SERVER LIVE SMOKE PASSED" in streaming_text
        and re.search(r"sdk_chunks:\s*[1-9]\d*", streaming_text) is not None
        and re.search(r"sdk_finish:\s*(stop|length)", streaming_text) is not None
    )
    security_passed = re.search(r"\d+ passed", security_text) is not None and " failed" not in security_text
    full_validation_passed = (
        re.search(r"\d+ passed, 30 deselected", validation_text) is not None
        and "162 passed" in validation_text
        and "13 passed, 22 deselected" in validation_text
        and all(
            marker in validation_text
            for marker in (
                "FIRST-RUN SMOKE PASSED",
                "AUTO-TRAIN QUEUE SMOKE PASSED",
                "SERVER LIVE SMOKE PASSED",
                "DASHBOARD CONSOLE LIVE SMOKE PASSED",
            )
        )
    )
    decision = build_phase42_final_decision(
        adapter_report=adapter,
        lifecycle_decision=lifecycle,
        training_attempt=training,
        context_smoke=context,
        hermes_streaming_passed=hermes_passed,
        security_tests_passed=security_passed,
        full_validation_passed=full_validation_passed,
        phase41_current=phase41,
        phase41_v2=phase41_v2,
    )
    comparison = {
        "kind": "phase42_comparison_summary",
        "adapter_version": adapter.get("version"),
        "base_holdout": {"passed": base.get("passed"), "scores": base.get("scores")},
        "adapter_holdout": {
            "passed": adapter.get("passed"),
            "scores": adapter.get("scores"),
            "training_leakage_detected": adapter.get("training_leakage_detected"),
        },
        "adapter_action": lifecycle.get("action"),
        "adapter_artifact_retained": lifecycle.get("artifact_retained"),
        "real_training": {
            "completed": training.get("real_training"),
            "steps": (training.get("execution") or {}).get("steps"),
            "initial_loss": (training.get("execution") or {}).get("initial_loss"),
            "final_loss": (training.get("execution") or {}).get("final_loss"),
            "parameters_updated": (training.get("execution") or {}).get("parameters_updated"),
            "adapter_validation": training.get("adapter_validation"),
        },
        "hermes_openai_sdk_streaming_passed": hermes_passed,
        "context_budget": context.get("token_budget"),
        "security_tests_passed": security_passed,
        "full_validation_gate_passed": full_validation_passed,
        "phase41_current_quality": phase41.get("candidate_quality"),
        "phase41_v2_quality": phase41_v2.get("candidate_quality"),
        "actual_product_benefit_claim_allowed": False,
        "auto_promotion_allowed": False,
    }
    _write("phase42-final-decision.json", decision)
    _write("comparison_summary.json", comparison)
    print(json.dumps(decision, ensure_ascii=False, sort_keys=True))
    return 0 if decision["reliability_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
