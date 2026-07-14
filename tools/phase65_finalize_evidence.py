#!/usr/bin/env python3
"""Finalize Phase65 scope-aware candidate evidence."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = REPO_ROOT / "pfe-core"
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

from pfe_core.phase65_aggregate_safe_boundary_coverage import build_phase65_decision


EVIDENCE_ROOT = REPO_ROOT / "docs/demo/phase65-aggregate-safe-boundary-coverage"
ALIASES = ("semantic_judge_alpha", "semantic_judge_beta")
DYNAMIC_FILES = {
    "evidence_manifest.json",
    "finalization_state.json",
    "validation_gate.txt",
    "validation_summary.json",
}


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return dict(value) if isinstance(value, Mapping) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stage_counts(directory: Path) -> dict[str, int]:
    successes = sum(
        len(_read_jsonl(directory / f"judge_typed_wire_results_{alias}.jsonl"))
        for alias in ALIASES
    )
    failure_attempts = sum(
        len(_read_jsonl(directory / f"typed_wire_failure_attempts_{alias}.jsonl"))
        for alias in ALIASES
    )
    return {
        "successful_model_output_count": successes,
        "raw_failure_attempt_count": failure_attempts,
    }


def _failure_rows(directory: Path) -> list[dict[str, Any]]:
    rows = []
    for alias in ALIASES:
        rows.extend(_read_jsonl(directory / f"typed_wire_failure_attempts_{alias}.jsonl"))
    return rows


def _raw_failures_preserved(rows: list[dict[str, Any]]) -> bool:
    return all(
        "raw_response" in row
        and row.get("raw_response_sha256")
        == hashlib.sha256(str(row.get("raw_response") or "").encode("utf-8")).hexdigest()
        for row in rows
    )


def _failure_analysis(report: Mapping[str, Any]) -> dict[str, Any]:
    failures = [dict(row) for row in report.get("details") or [] if row.get("passed") is False]
    transitions: dict[str, int] = {}
    for row in failures:
        transition = f"{row.get('expected_label')}->{row.get('actual_label')}"
        transitions[transition] = transitions.get(transition, 0) + 1
    return {
        "kind": "phase65_scope_aware_failure_analysis",
        "label_failure_count": len(failures),
        "label_transition_counts": dict(sorted(transitions.items())),
        "safe_abstention_recovery_count": report.get("safe_abstention_recovery_count"),
        "dangerous_any_consensus_count": report.get("dangerous_any_consensus_count"),
        "candidate_value_conflict_count": report.get("candidate_value_conflict_count"),
        "post_model_tuning_performed": False,
        "label_failures": failures,
    }


def _manifest() -> dict[str, Any]:
    files = []
    for path in sorted(EVIDENCE_ROOT.rglob("*")):
        if not path.is_file() or path.name in DYNAMIC_FILES:
            continue
        files.append(
            {
                "path": str(path.relative_to(REPO_ROOT)),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    digest = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "kind": "phase65_evidence_manifest",
        "file_count": len(files),
        "files": files,
        "manifest_sha256": digest,
    }


def main() -> int:
    preflight_dir = EVIDENCE_ROOT / "evidence-typed-wire-preflight"
    calibration_dir = EVIDENCE_ROOT / "evidence-evaluator-calibration"
    holdout_dir = EVIDENCE_ROOT / "evidence-evaluator-holdout"
    baseline = _read_json(EVIDENCE_ROOT / "evidence-baseline/phase64_canonical_snapshot.json")
    aggregate = _read_json(EVIDENCE_ROOT / "aggregate_failure_taxonomy.json")
    preflight = _read_json(preflight_dir / "preflight_report.json")
    calibration = _read_json(calibration_dir / "candidate_evaluator_report.json")
    holdout = _read_json(holdout_dir / "candidate_evaluator_report.json")
    split = _read_json(holdout_dir / "split_integrity.json")
    calibration_audit = _read_json(calibration_dir / "fixture_semantic_audit.json")
    holdout_audit = _read_json(holdout_dir / "fixture_semantic_audit.json")
    scope_calibration = _read_json(calibration_dir / "scope_rule_audit.json")
    scope_holdout = _read_json(holdout_dir / "scope_rule_audit.json")
    hard_calibration = _read_json(calibration_dir / "hard_rule_compatibility.json")
    hard_holdout = _read_json(holdout_dir / "hard_rule_compatibility.json")
    runtime = _read_json(EVIDENCE_ROOT / "evidence-no-runtime/runtime_status.json")
    training = _read_json(EVIDENCE_ROOT / "evidence-no-training/training_attempt.json")
    protocol = _read_json(EVIDENCE_ROOT / "evaluator_protocol.json")
    decision = build_phase65_decision(
        phase64_snapshot=baseline,
        aggregate_failure_taxonomy=aggregate,
        preflight_report=preflight,
        calibration_report=calibration,
        holdout_report=holdout,
        calibration_audit=calibration_audit,
        holdout_audit=holdout_audit,
        scope_calibration=scope_calibration,
        scope_holdout=scope_holdout,
        hard_calibration=hard_calibration,
        hard_holdout=hard_holdout,
        split_integrity=split,
    )
    preflight_counts = _stage_counts(preflight_dir)
    calibration_counts = _stage_counts(calibration_dir)
    holdout_counts = _stage_counts(holdout_dir)
    preflight_failed = int(preflight.get("failed_judge_item_count") or 0)
    calibration_failed = int(calibration.get("failure_count") or 0)
    holdout_failed = int(holdout.get("failure_count") or 0)
    preflight_outcomes = preflight_counts["successful_model_output_count"] + preflight_failed
    calibration_outcomes = calibration_counts["successful_model_output_count"] + calibration_failed
    holdout_outcomes = holdout_counts["successful_model_output_count"] + holdout_failed
    preflight_passed = preflight.get("status") == "passed"
    calibration_qualified = calibration.get("status") == "qualified"
    all_raw_failures = (
        _failure_rows(preflight_dir)
        + _failure_rows(calibration_dir)
        + _failure_rows(holdout_dir)
    )
    raw_failures_preserved = _raw_failures_preserved(all_raw_failures)
    call_evidence_complete = (
        preflight_outcomes == 12
        and (
            (preflight_passed and calibration_outcomes == 120)
            or (not preflight_passed and calibration_outcomes == 0)
        )
        and (
            (preflight_passed and calibration_qualified and holdout_outcomes == 240)
            or (not (preflight_passed and calibration_qualified) and holdout_outcomes == 0)
        )
    )
    comparison = {
        "kind": "phase65_scope_aware_candidate_comparison",
        "phase64": {
            "recommendation": baseline.get("phase64_recommendation"),
            "historical_accuracy": baseline.get("phase64_accuracy"),
            "false_accept_count": baseline.get("phase64_false_accept_count"),
            "schema_failure_count": baseline.get("phase64_schema_failure_count"),
            "candidate_value_conflict_count": baseline.get(
                "phase64_candidate_value_conflict_count"
            ),
        },
        "phase65": {
            "preflight_status": preflight.get("status") or "not_run",
            "calibration_status": calibration.get("status")
            or "not_run_after_preflight_failure",
            "calibration_accuracy": calibration.get("accuracy"),
            "calibration_typed_exact_match_rate": calibration.get("typed_exact_match_rate"),
            "calibration_candidate_selection_exact_match_rate": calibration.get(
                "candidate_selection_exact_match_rate"
            ),
            "calibration_raw_judge_typed_exact_match_rate": calibration.get(
                "raw_judge_typed_exact_match_rate"
            ),
            "holdout_status": holdout.get("status") or "not_run_after_prior_gate_failure",
            "holdout_accuracy": holdout.get("accuracy"),
            "holdout_typed_exact_match_rate": holdout.get("typed_exact_match_rate"),
            "holdout_candidate_selection_exact_match_rate": holdout.get(
                "candidate_selection_exact_match_rate"
            ),
            "holdout_raw_judge_typed_exact_match_rate": holdout.get(
                "raw_judge_typed_exact_match_rate"
            ),
            "holdout_safe_abstention_recovery_count": holdout.get(
                "safe_abstention_recovery_count"
            ),
            "holdout_dangerous_any_consensus_count": holdout.get(
                "dangerous_any_consensus_count"
            ),
            "holdout_candidate_value_conflict_count": holdout.get(
                "candidate_value_conflict_count"
            ),
            "holdout_false_accept_count": holdout.get(
                "false_accept_count_on_reject_cases"
            ),
            "holdout_schema_failure_count": holdout.get("schema_failure_count"),
            "raw_typed_wire_failures_preserved": raw_failures_preserved,
        },
        "recommendation": decision["recommendation"],
        "actual_user_feedback_count": 0,
        "training_executed": False,
        "adapter_created": False,
        "runtime_replay_executed": False,
        "hermes_attached": False,
        "product_default_changed": False,
    }
    integrity_checks = {
        "phase64_canonical_snapshot_passed": baseline.get("passed") is True,
        "aggregate_failure_taxonomy_passed": aggregate.get("passed") is True,
        "split_integrity_passed": split.get("passed") is True,
        "fixture_and_scope_audits_passed": (
            calibration_audit.get("status") == "passed"
            and holdout_audit.get("status") == "passed"
            and scope_calibration.get("status") == "passed"
            and scope_holdout.get("status") == "passed"
        ),
        "preflight_freeze_check_passed": _read_json(
            preflight_dir / "freeze_check.json"
        ).get("passed")
        is True,
        "call_evidence_complete_for_gated_path": call_evidence_complete,
        "raw_failure_attempts_preserved_and_hashed": raw_failures_preserved,
        "calibration_freeze_check_consistent": (
            _read_json(calibration_dir / "freeze_check.json").get("passed") is True
            if preflight_passed
            else calibration_outcomes == 0
        ),
        "holdout_freeze_check_consistent": (
            _read_json(holdout_dir / "freeze_check.json").get("passed") is True
            if preflight_passed and calibration_qualified
            else holdout_outcomes == 0
        ),
        "no_post_model_call_tuning": protocol.get("post_model_call_tuning_allowed") is False,
        "runtime_not_run": int(runtime.get("runtime_replay_model_call_count") or 0) == 0,
        "training_not_run": training.get("training_executed") is False
        and training.get("adapter_created") is False,
        "no_training_adapter_hermes_or_default_change": (
            decision["new_training_allowed"] is False
            and decision["new_adapter_created"] is False
            and decision["hermes_attachment_allowed"] is False
            and decision["product_default_change_allowed"] is False
        ),
    }
    integrity = {
        "kind": "phase65_evidence_integrity",
        "passed": all(integrity_checks.values()),
        "checks": integrity_checks,
        "created_at": _utcnow(),
    }
    if calibration:
        _write_json(calibration_dir / "failure_analysis.json", _failure_analysis(calibration))
    if holdout:
        _write_json(holdout_dir / "failure_analysis.json", _failure_analysis(holdout))
    if all_raw_failures:
        _write_json(
            EVIDENCE_ROOT / "typed_wire_failure_evidence.json",
            {
                "kind": "phase65_typed_wire_failure_evidence",
                "status": "preserved",
                "raw_failure_attempt_count": len(all_raw_failures),
                "raw_failures_preserved_and_hashed": raw_failures_preserved,
                "failure_classes": sorted(
                    {str(row.get("failure_class") or "") for row in all_raw_failures}
                ),
                "post_failure_protocol_change_performed": False,
            },
        )
    _write_json(EVIDENCE_ROOT / "comparison_summary.json", comparison)
    _write_json(EVIDENCE_ROOT / "phase65-final-decision.json", decision)
    _write_json(EVIDENCE_ROOT / "evidence_integrity.json", integrity)
    _write_text(
        EVIDENCE_ROOT / "phase65-final-decision.md",
        f"""# Phase65 Final Decision

## 结论

最终 recommendation 为 **{decision['recommendation']}**。Preflight 为 `{preflight.get('status')}`，calibration accuracy 为 `{calibration.get('accuracy')}`，holdout accuracy 为 `{holdout.get('accuracy')}`。

## 冻结边界

- 唯一结构改动是关系分句中保留明确的安全 outcome 候选，同时继续排除嵌套的危险 asserted outcome。
- 危险候选识别、Phase62 risk-asymmetric consensus、Phase63 PFE2 wire、composer 和评分门槛均未放宽。
- 新 calibration/holdout 只依据 Phase64 聚合失败类型设计，不复用单条历史失败原文。
- 所有输入均为 simulated evaluator fixtures，不是 actual user feedback，也不用于训练。
- Phase65 不运行 runtime A/B、不训练、不创建 adapter、不接 Hermes、不改产品默认路径。
""",
    )
    _write_text(
        EVIDENCE_ROOT / "phase65-runbook.md",
        """# Phase65 Runbook

```bash
OLLAMA_HOST=127.0.0.1:11435 OLLAMA_NUM_PARALLEL=4 OLLAMA_MAX_LOADED_MODELS=1 ollama serve
```

```bash
.venv/bin/python tools/phase65_prepare.py --clean-evidence
.venv/bin/python tools/phase65_execute.py --stage preflight --ollama-endpoint http://127.0.0.1:11435 --timeout 900
.venv/bin/python tools/phase65_execute.py --stage calibration --ollama-endpoint http://127.0.0.1:11435 --timeout 900
.venv/bin/python tools/phase65_execute.py --stage holdout --ollama-endpoint http://127.0.0.1:11435 --timeout 900
.venv/bin/python tools/phase65_finalize_evidence.py
.venv/bin/python tools/phase65_validate.py
```

Do not change fixtures, scope-aware candidate rule, wire, consensus, retry count, or gates after prepare. Stop after the first failed gate and finalize that path.
""",
    )
    next_goal = (
        "Build Phase66 as an external regression of the qualified Phase65 evaluator. Freeze a fresh paraphrase holdout plus a sealed historical distributional replay before calls; require zero false accepts, zero schema failures, zero candidate conflicts, and material accuracy improvement over Phase64 before any runtime A/B. Do not train, attach Hermes, change defaults, auto-promote, or claim actual user benefit."
        if decision["phase66_external_regression_design_eligible"]
        else
        "Hold Phase65. Analyze only aggregate frozen failures, choose one structural correction, and freeze new calibration and holdout before more model calls. Do not run runtime A/B, train, attach Hermes, change defaults, or relax gates."
    )
    _write_text(EVIDENCE_ROOT / "next-pursuit-goal.md", f"# Next Pursuit Goal\n\n{next_goal}\n")
    manifest = _manifest()
    _write_json(EVIDENCE_ROOT / "evidence_manifest.json", manifest)
    finalization = {
        "kind": "phase65_finalization_state",
        "status": "completed" if integrity["passed"] else "blocked",
        "recommendation": decision["recommendation"],
        "evidence_integrity_passed": integrity["passed"],
        "manifest_file_count": manifest["file_count"],
        "created_at": _utcnow(),
    }
    _write_json(EVIDENCE_ROOT / "finalization_state.json", finalization)
    print(json.dumps(finalization, ensure_ascii=False, indent=2))
    return 0 if integrity["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
