#!/usr/bin/env python3
"""Finalize Phase59 proposition-candidate evidence."""

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

from pfe_core.phase59_proposition_addressed_grounding import build_phase59_decision


EVIDENCE_ROOT = REPO_ROOT / "docs/demo/phase59-proposition-addressed-grounding"
ALIASES = ("semantic_judge_alpha", "semantic_judge_beta")
DYNAMIC_FILES = {"evidence_manifest.json", "finalization_state.json", "validation_gate.txt", "validation_summary.json"}


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
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _call_count(directory: Path) -> int:
    return sum(len(_read_jsonl(directory / f"judge_candidate_results_{alias}.jsonl")) for alias in ALIASES)


def _failure_analysis(report: Mapping[str, Any]) -> dict[str, Any]:
    failures = [dict(row) for row in report.get("details") or [] if row.get("passed") is False]
    transitions: dict[str, int] = {}
    wrong_selections = []
    for row in report.get("details") or []:
        if row.get("passed") is False:
            key = f"{row.get('expected_label')}->{row.get('actual_label')}"
            transitions[key] = transitions.get(key, 0) + 1
        expected = dict(row.get("expected_candidate_ids") or {})
        for selection in row.get("grounded_selections") or []:
            for field in ("source_registration", "user_outcome_status", "test_to_user_outcome_relation"):
                actual = selection.get(f"{field}_candidate_id")
                if actual == expected.get(field):
                    continue
                wrong_selections.append(
                    {
                        "item_id": row.get("item_id"),
                        "case_id": row.get("case_id"),
                        "category": row.get("category"),
                        "judge_alias": selection.get("judge_alias"),
                        "field": field,
                        "expected_candidate_id": expected.get(field),
                        "actual_candidate_id": actual,
                    }
                )
    return {
        "kind": "phase59_candidate_selection_failure_analysis",
        "label_failure_count": len(failures),
        "wrong_candidate_selection_count": len(wrong_selections),
        "label_transition_counts": dict(sorted(transitions.items())),
        "post_model_tuning_performed": False,
        "label_failures": failures,
        "wrong_candidate_selections": wrong_selections,
    }


def _manifest() -> dict[str, Any]:
    files = []
    for path in sorted(EVIDENCE_ROOT.rglob("*")):
        if not path.is_file() or path.name in DYNAMIC_FILES:
            continue
        files.append(
            {"path": str(path.relative_to(REPO_ROOT)), "sha256": _sha256(path), "size_bytes": path.stat().st_size}
        )
    digest = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return {"kind": "phase59_evidence_manifest", "file_count": len(files), "files": files, "manifest_sha256": digest}


def main() -> int:
    calibration_dir = EVIDENCE_ROOT / "evidence-evaluator-calibration"
    holdout_dir = EVIDENCE_ROOT / "evidence-evaluator-holdout"
    calibration = _read_json(calibration_dir / "candidate_evaluator_report.json")
    holdout = _read_json(holdout_dir / "candidate_evaluator_report.json")
    baseline = _read_json(EVIDENCE_ROOT / "evidence-baseline/phase58_canonical_snapshot.json")
    split = _read_json(holdout_dir / "split_integrity.json")
    calibration_audit = _read_json(calibration_dir / "fixture_semantic_audit.json")
    holdout_audit = _read_json(holdout_dir / "fixture_semantic_audit.json")
    hard_calibration = _read_json(calibration_dir / "hard_rule_compatibility.json")
    hard_holdout = _read_json(holdout_dir / "hard_rule_compatibility.json")
    runtime = _read_json(EVIDENCE_ROOT / "evidence-no-runtime/runtime_status.json")
    decision = build_phase59_decision(
        phase58_snapshot=baseline,
        calibration_report=calibration,
        holdout_report=holdout,
        calibration_audit=calibration_audit,
        holdout_audit=holdout_audit,
        hard_calibration=hard_calibration,
        hard_holdout=hard_holdout,
        split_integrity=split,
    )
    calibration_calls = _call_count(calibration_dir)
    holdout_calls = _call_count(holdout_dir)
    calibration_failures = int(calibration.get("failure_count") or 0)
    holdout_failures = int(holdout.get("failure_count") or 0)
    calibration_outcomes = calibration_calls + calibration_failures
    holdout_outcomes = holdout_calls + holdout_failures
    calibration_qualified = calibration.get("status") == "qualified"
    call_evidence_complete = calibration_outcomes == 60 and (
        (calibration_qualified and holdout_outcomes == 120)
        or (not calibration_qualified and holdout_outcomes == 0)
    )
    comparison = {
        "kind": "phase59_proposition_candidate_comparison",
        "phase58": {
            "recommendation": baseline.get("phase58_recommendation"),
            "calibration_accuracy": baseline.get("phase58_accuracy"),
            "typed_exact_match_rate": baseline.get("phase58_typed_exact_match_rate"),
            "grounding_validity_rate": baseline.get("phase58_grounding_validity_rate"),
            "invalid_atom_count": baseline.get("phase58_invalid_atom_count"),
            "invalid_dangerous_atom_count": baseline.get("phase58_invalid_dangerous_atom_count"),
        },
        "phase59": {
            "calibration_status": calibration.get("status"),
            "calibration_accuracy": calibration.get("accuracy"),
            "calibration_typed_exact_match_rate": calibration.get("typed_exact_match_rate"),
            "calibration_candidate_selection_exact_match_rate": calibration.get("candidate_selection_exact_match_rate"),
            "holdout_status": holdout.get("status") or "not_run_after_calibration_failure",
            "holdout_accuracy": holdout.get("accuracy"),
            "holdout_typed_exact_match_rate": holdout.get("typed_exact_match_rate"),
            "holdout_candidate_selection_exact_match_rate": holdout.get("candidate_selection_exact_match_rate"),
            "holdout_invalid_dangerous_atom_count": holdout.get("invalid_dangerous_atom_count"),
            "holdout_false_accept_count": holdout.get("false_accept_count_on_reject_cases"),
            "calibration_successful_model_output_count": calibration_calls,
            "calibration_failed_judge_item_count": calibration_failures,
            "calibration_judge_item_outcome_count": calibration_outcomes,
            "holdout_successful_model_output_count": holdout_calls,
            "holdout_failed_judge_item_count": holdout_failures,
            "holdout_judge_item_outcome_count": holdout_outcomes,
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
        "phase58_canonical_snapshot_passed": baseline.get("passed") is True,
        "split_integrity_passed": split.get("passed") is True,
        "fixture_semantic_audits_passed": (
            calibration_audit.get("status") == "passed" and holdout_audit.get("status") == "passed"
        ),
        "calibration_freeze_check_passed": _read_json(calibration_dir / "freeze_check.json").get("passed") is True,
        "call_evidence_complete_for_frozen_gate": call_evidence_complete,
        "holdout_freeze_check_consistent": (
            _read_json(holdout_dir / "freeze_check.json").get("passed") is True
            if calibration_qualified else holdout_outcomes == 0
        ),
        "no_post_model_tuning": _read_json(EVIDENCE_ROOT / "evaluator_protocol.json").get("post_calibration_tuning_allowed") is False,
        "runtime_not_run": int(runtime.get("runtime_replay_model_call_count") or 0) == 0,
        "no_training_adapter_hermes_or_default_change": (
            decision["new_training_allowed"] is False
            and decision["new_adapter_created"] is False
            and decision["hermes_attachment_allowed"] is False
            and decision["product_default_change_allowed"] is False
        ),
    }
    integrity = {
        "kind": "phase59_evidence_integrity",
        "passed": all(integrity_checks.values()),
        "checks": integrity_checks,
        "created_at": _utcnow(),
    }
    _write_json(calibration_dir / "failure_analysis.json", _failure_analysis(calibration))
    if calibration_failures:
        errors = [str(row.get("error") or "") for row in calibration.get("failures") or []]
        _write_json(
            calibration_dir / "protocol_compatibility_failure.json",
            {
                "kind": "phase59_protocol_compatibility_failure",
                "status": "sealed_not_qualified",
                "successful_model_output_count": calibration_calls,
                "failed_judge_item_count": calibration_failures,
                "judge_item_outcome_count": calibration_outcomes,
                "schema_failure_count": calibration.get("schema_failure_count"),
                "failed_judge_aliases": sorted(
                    {str(row.get("judge_alias") or "") for row in calibration.get("failures") or []}
                ),
                "failure_errors": sorted(set(errors)),
                "frozen_retry_limit_per_failed_item": 2,
                "raw_invalid_responses_preserved": False,
                "evidence_gap": "runner preserved validation errors but not raw responses that failed schema validation",
                "post_failure_protocol_change_performed": False,
                "holdout_executed": False,
            },
        )
    if holdout:
        _write_json(holdout_dir / "failure_analysis.json", _failure_analysis(holdout))
    _write_json(EVIDENCE_ROOT / "comparison_summary.json", comparison)
    _write_json(EVIDENCE_ROOT / "phase59-final-decision.json", decision)
    _write_json(EVIDENCE_ROOT / "evidence_integrity.json", integrity)
    _write_text(
        EVIDENCE_ROOT / "phase59-final-decision.md",
        f"""# Phase59 Final Decision

## 结论

最终 recommendation 为 **{decision['recommendation']}**。Calibration accuracy 为 `{calibration.get('accuracy')}`，holdout accuracy 为 `{holdout.get('accuracy')}`，holdout candidate selection exact match 为 `{holdout.get('candidate_selection_exact_match_rate')}`。

## 冻结边界

- Phase59 只引入预落地 proposition candidates 与冻结前 fixture semantic audit；Phase53 hard detector、Phase56 composer、Phase58 clause segmenter 未修改。
- 30 条 calibration 与 60 条 holdout 在模型调用前冻结；holdout 仅在 calibration qualified 后运行。
- 两位 judge 不直接输出 label，也看不到 gold label、typed fields 或 candidate IDs。
- 所有输入都是 simulated evaluator fixtures，不是 actual user feedback，不用于训练。
- Phase59 不运行 runtime A/B、不训练、不创建 adapter、不接 Hermes、不改产品默认路径。
""",
    )
    _write_text(
        EVIDENCE_ROOT / "phase59-runbook.md",
        """# Phase59 Runbook

```bash
OLLAMA_HOST=127.0.0.1:11435 OLLAMA_NUM_PARALLEL=4 OLLAMA_MAX_LOADED_MODELS=1 ollama serve
```

In a second terminal:

```bash
.venv/bin/python tools/phase59_prepare.py --clean-evidence
.venv/bin/python tools/phase59_candidate_evaluator.py --split calibration --ollama-endpoint http://127.0.0.1:11435 --timeout 900
.venv/bin/python tools/phase59_candidate_evaluator.py --split holdout --ollama-endpoint http://127.0.0.1:11435 --timeout 900
.venv/bin/python tools/phase59_finalize_evidence.py
.venv/bin/python tools/phase59_validate.py
```

Do not change candidate generation, fixture audits, prompts, schemas, or gates after any calibration call. A failed calibration seals Phase59 without holdout.
""",
    )
    next_goal = (
        "Build Phase60 as a frozen external historical replay for the manually reviewable Phase59 proposition evaluator. Replay "
        "representative sealed Phase51-58 fixtures without modifying candidate generation or gates. Require per-phase accuracy, zero "
        "false accepts, zero unsupported candidates, and no hard-rule conflict before any runtime A/B is considered. Do not train, "
        "attach Hermes, change defaults, auto-promote, or claim actual user benefit."
        if decision["phase60_external_replay_design_eligible"]
        else
        "Hold Phase59. Analyze aggregate frozen candidate-selection failures, choose one structural correction, and freeze new fixtures "
        "before more calls. Do not run runtime A/B, train, attach Hermes, change defaults, or relax gates."
    )
    _write_text(EVIDENCE_ROOT / "next-pursuit-goal.md", f"# Next Pursuit Goal\n\n{next_goal}\n")
    manifest = _manifest()
    _write_json(EVIDENCE_ROOT / "evidence_manifest.json", manifest)
    finalization = {
        "kind": "phase59_finalization_state",
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
