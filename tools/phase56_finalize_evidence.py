#!/usr/bin/env python3
"""Finalize Phase56 span-grounded atomic-evaluator evidence without runtime changes."""

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

from pfe_core.phase56_evidence_span_grounded_atomic import build_phase56_decision


EVIDENCE_ROOT = REPO_ROOT / "docs/demo/phase56-evidence-span-grounded-atomic"
PHASE55_ROOT = REPO_ROOT / "docs/demo/phase55-atomic-boundary-composition"
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
    return sum(len(_read_jsonl(directory / f"judge_span_results_{alias}.jsonl")) for alias in ALIASES)


def _attempt_summaries() -> list[dict[str, Any]]:
    root = EVIDENCE_ROOT / "evidence-evaluator-debug"
    return [_read_json(path) for path in sorted(root.glob("*/attempt_status.json"))] if root.exists() else []


def _failure_analysis(directory: Path, report: Mapping[str, Any]) -> dict[str, Any]:
    dangerous_values = {
        "source_registration": "allow_actual",
        "user_outcome_status": "asserted_current",
        "test_to_user_outcome_relation": "establishes",
    }
    public = {str(row.get("item_id") or ""): row for row in _read_jsonl(directory / "blind_items_public.jsonl")}
    failed = [row for row in report.get("details") or [] if row.get("passed") is False]
    field_mismatches = []
    invalid_groundings = []
    transition_counts: dict[str, int] = {}
    for row in report.get("details") or []:
        expected_typed = dict(row.get("expected_typed") or {})
        expected_atoms = dict(row.get("expected_atoms") or {})
        for extraction in row.get("grounded_extractions") or []:
            for field, expected in expected_typed.items():
                actual = extraction.get(field)
                if extraction.get(f"{field}_grounded") is not True:
                    invalid_groundings.append(
                        {
                            "item_id": row.get("item_id"),
                            "case_id": row.get("case_id"),
                            "category": row.get("category"),
                            "expected_label": row.get("expected_label"),
                            "judge_alias": extraction.get("judge_alias"),
                            "field": field,
                            "expected": expected,
                            "raw_value": extraction.get(f"raw_{field}"),
                            "effective_value": actual,
                            "raw_span": extraction.get(f"{field}_span"),
                            "grounding_reason": extraction.get(f"{field}_grounding_reason"),
                            "dangerous_raw_atom": extraction.get(f"raw_{field}") == dangerous_values[field],
                        }
                    )
                if actual == expected:
                    continue
                transition = f"{field}:{expected}->{actual}"
                transition_counts[transition] = transition_counts.get(transition, 0) + 1
                field_mismatches.append(
                    {
                        "item_id": row.get("item_id"),
                        "case_id": row.get("case_id"),
                        "category": row.get("category"),
                        "judge_alias": extraction.get("judge_alias"),
                        "field": field,
                        "expected": expected,
                        "actual": actual,
                        "raw_value": extraction.get(f"raw_{field}"),
                        "raw_span": extraction.get(f"{field}_span"),
                        "expected_span": dict(expected_atoms.get(field) or {}).get("evidence_span"),
                        "grounded": extraction.get(f"{field}_grounded"),
                        "grounding_reason": extraction.get(f"{field}_grounding_reason"),
                    }
                )
    return {
        "kind": "phase56_span_evaluator_failure_analysis",
        "split": report.get("split"),
        "status": "no_post_result_tuning",
        "label_failure_count": len(failed),
        "field_mismatch_count": len(field_mismatches),
        "invalid_grounding_count": len(invalid_groundings),
        "invalid_dangerous_grounding_count": sum(
            row["dangerous_raw_atom"] for row in invalid_groundings
        ),
        "field_transition_counts": dict(sorted(transition_counts.items())),
        "rubric_schema_or_fixture_modified_after_result": False,
        "label_failures": [
            {
                "item_id": row.get("item_id"),
                "case_id": row.get("case_id"),
                "category": row.get("category"),
                "assistant_response": public.get(str(row.get("item_id") or ""), {}).get("assistant_response"),
                "expected_label": row.get("expected_label"),
                "actual_label": row.get("actual_label"),
                "expected_typed": row.get("expected_typed"),
                "expected_atoms": row.get("expected_atoms"),
                "grounded_extractions": row.get("grounded_extractions"),
                "per_judge_composed_labels": row.get("per_judge_composed_labels"),
                "hard_reject": row.get("hard_reject"),
            }
            for row in failed
        ],
        "field_mismatches": field_mismatches,
        "invalid_groundings": invalid_groundings,
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
    return {"kind": "phase56_evidence_manifest", "file_count": len(files), "files": files, "manifest_sha256": digest}


def main() -> int:
    calibration_dir = EVIDENCE_ROOT / "evidence-evaluator-calibration"
    holdout_dir = EVIDENCE_ROOT / "evidence-evaluator-holdout"
    calibration = _read_json(calibration_dir / "span_evaluator_report.json")
    holdout = _read_json(holdout_dir / "span_evaluator_report.json")
    hard_calibration = _read_json(calibration_dir / "hard_reject_report.json")
    hard_holdout = _read_json(holdout_dir / "hard_reject_report.json")
    split = _read_json(holdout_dir / "split_integrity.json")
    runtime_status = _read_json(EVIDENCE_ROOT / "evidence-no-runtime/runtime_status.json")
    decision = build_phase56_decision(
        calibration_report=calibration,
        holdout_report=holdout,
        hard_calibration=hard_calibration,
        hard_holdout=hard_holdout,
        split_integrity=split,
        runtime_replay_model_call_count=int(runtime_status.get("runtime_replay_model_call_count") or 0),
        boundary_clause_design_created=runtime_status.get("boundary_clause_design_created") is True,
    )
    calibration_calls = _call_count(calibration_dir)
    holdout_calls = _call_count(holdout_dir)
    holdout_called = bool(holdout)
    attempts = _attempt_summaries()
    comparison = {
        "kind": "phase56_span_evaluator_comparison",
        "phase55": {
            "recommendation": _read_json(PHASE55_ROOT / "phase55-final-decision.json").get("recommendation"),
            "holdout_accuracy": _read_json(
                PHASE55_ROOT / "evidence-evaluator-holdout/typed_evaluator_report.json"
            ).get("accuracy"),
            "typed_exact_match_rate": _read_json(
                PHASE55_ROOT / "evidence-evaluator-holdout/typed_evaluator_report.json"
            ).get("typed_exact_match_rate"),
            "source_registration_accuracy": _read_json(
                PHASE55_ROOT / "evidence-evaluator-holdout/typed_evaluator_report.json"
            ).get("per_field", {}).get("source_registration", {}).get("accuracy"),
            "evidence_spans_required": False,
        },
        "phase56": {
            "invalidated_calibration_attempts": attempts,
            "calibration_status": calibration.get("status"),
            "calibration_accuracy": calibration.get("accuracy"),
            "calibration_typed_exact_match_rate": calibration.get("typed_exact_match_rate"),
            "calibration_grounding_validity_rate": calibration.get("raw_grounding_validity_rate"),
            "calibration_per_field": calibration.get("per_field"),
            "calibration_per_category": calibration.get("per_category"),
            "calibration_real_model_call_count": calibration_calls,
            "holdout_status": holdout.get("status") if holdout_called else "not_called",
            "holdout_accuracy": holdout.get("accuracy"),
            "holdout_typed_exact_match_rate": holdout.get("typed_exact_match_rate"),
            "holdout_grounding_validity_rate": holdout.get("raw_grounding_validity_rate"),
            "holdout_expected_span_exact_match_rate_diagnostic": holdout.get(
                "expected_span_exact_match_rate_diagnostic"
            ),
            "holdout_invalid_atom_count": holdout.get("invalid_atom_count"),
            "holdout_invalid_dangerous_atom_count": holdout.get("invalid_dangerous_atom_count"),
            "holdout_composer_received_ungrounded_atom_count": holdout.get(
                "composer_received_ungrounded_atom_count"
            ),
            "holdout_per_field": holdout.get("per_field"),
            "holdout_per_category": holdout.get("per_category"),
            "holdout_false_accept_count": holdout.get("false_accept_count_on_reject_cases"),
            "holdout_hard_vs_two_safe_accept_conflict_count": holdout.get(
                "hard_reject_vs_two_safe_accept_conflict_count"
            ),
            "holdout_judge_direct_label_count": holdout.get("judge_direct_label_count"),
            "holdout_real_model_call_count": holdout_calls,
            "final_label_generated_by_deterministic_composer": True,
        },
        "runtime_replay": {
            "status": runtime_status.get("runtime_replay_status"),
            "real_model_call_count": runtime_status.get("runtime_replay_model_call_count"),
        },
        "boundary_clause_design": {
            "status": runtime_status.get("boundary_clause_design_status"),
            "created": runtime_status.get("boundary_clause_design_created"),
        },
        "recommendation": decision["recommendation"],
        "actual_user_feedback_count": 0,
        "actual_human_review_count": 0,
        "training_executed": False,
        "adapter_created": False,
        "hermes_attached": False,
        "runtime_prompt_changed": False,
        "router_changed": False,
        "product_default_changed": False,
    }
    calibration_complete = (
        calibration_calls == 240
        and int(calibration.get("completed_item_count") or 0) == 120
        and int(calibration.get("failure_count") or 0) == 0
    )
    if calibration.get("status") == "qualified":
        holdout_completion_ok = (
            holdout_called
            and holdout_calls == 300
            and int(holdout.get("completed_item_count") or 0) == 150
            and int(holdout.get("failure_count") or 0) == 0
        )
    else:
        holdout_completion_ok = not holdout_called and holdout_calls == 0
    expected_recommendation = (
        "recommend_phase56_span_evaluator_for_manual_review_only"
        if all(decision.get("checks", {}).values())
        else "hold_phase56_evidence_span_grounded_atomic"
    )
    integrity_checks = {
        "phase55_canonical_snapshot_passed": _read_json(
            EVIDENCE_ROOT / "evidence-baseline/phase55_canonical_snapshot.json"
        ).get("passed") is True,
        "formal_calibration_complete": calibration_complete,
        "holdout_protocol_respected": holdout_completion_ok,
        "decision_matches_frozen_gates": decision["recommendation"] == expected_recommendation,
        "runtime_replay_not_run": int(runtime_status.get("runtime_replay_model_call_count") or 0) == 0,
        "boundary_clause_not_designed": runtime_status.get("boundary_clause_design_created") is False,
        "historical_holdouts_not_reused": (
            split.get("checks", {}).get("prior_calibration_exact_overlap_zero") is True
            and split.get("checks", {}).get("prior_holdout_exact_overlap_zero") is True
            and split.get("checks", {}).get("historical_failure_response_reuse_zero") is True
        ),
        "invalidated_calibration_attempts_are_nonformal": (
            all(row.get("formal_result_eligible") is False for row in attempts)
            and all(int(row.get("holdout_model_call_count") or 0) == 0 for row in attempts)
        ),
        "no_direct_model_labels_in_formal_results": (
            int(calibration.get("judge_direct_label_count") or 0) == 0
            and int(holdout.get("judge_direct_label_count") or 0) == 0
        ),
        "grounding_gates_applied": (
            float(calibration.get("raw_grounding_validity_rate") or 0.0)
            >= float(calibration.get("raw_grounding_validity_gate") or 1.0)
            and (
                not holdout_called
                or float(holdout.get("raw_grounding_validity_rate") or 0.0)
                >= float(holdout.get("raw_grounding_validity_gate") or 1.0)
            )
        ),
        "composer_received_no_ungrounded_atoms": (
            int(calibration.get("composer_received_ungrounded_atom_count") or 0) == 0
            and int(holdout.get("composer_received_ungrounded_atom_count") or 0) == 0
        ),
        "no_training_adapter_hermes_or_default_change": (
            decision["new_training_allowed"] is False
            and decision["new_adapter_created"] is False
            and decision["hermes_attachment_allowed"] is False
            and decision["product_default_change_allowed"] is False
        ),
    }
    integrity = {
        "kind": "phase56_evidence_integrity",
        "passed": all(integrity_checks.values()),
        "checks": integrity_checks,
        "created_at": _utcnow(),
    }
    _write_json(calibration_dir / "failure_analysis.json", _failure_analysis(calibration_dir, calibration))
    if holdout_called:
        _write_json(holdout_dir / "failure_analysis.json", _failure_analysis(holdout_dir, holdout))
    _write_json(EVIDENCE_ROOT / "comparison_summary.json", comparison)
    _write_json(EVIDENCE_ROOT / "phase56-final-decision.json", decision)
    _write_json(EVIDENCE_ROOT / "evidence_integrity.json", integrity)
    holdout_summary = (
        f"独立 holdout 准确率 `{holdout.get('accuracy')}`，typed exact match `{holdout.get('typed_exact_match_rate')}`，"
        f"grounding validity `{holdout.get('raw_grounding_validity_rate')}`，无效原子 "
        f"`{holdout.get('invalid_atom_count')}`，危险无依据原子 `{holdout.get('invalid_dangerous_atom_count')}`，"
        f"字段准确率 `{holdout.get('per_field')}`。"
        if holdout_called
        else "Calibration 未合格，因此独立 holdout 按冻结协议没有调用。"
    )
    _write_text(
        EVIDENCE_ROOT / "phase56-final-decision.md",
        f"""# Phase56 Final Decision

## 结论

最终 recommendation 为 **{decision['recommendation']}**。正式 calibration 准确率 `{calibration.get('accuracy')}`，typed exact match `{calibration.get('typed_exact_match_rate')}`。{holdout_summary}

本阶段在 Phase55 三个原子字段上增加逐字 evidence span。任何未合格 calibration 尝试只作为 debug evidence，均不得调用 holdout 或成为正式结果。正式独立 holdout 最多调用一次，结果后不得修改 rubric、schema、grounding、composer、fixture 或门槛。

## 架构与边界

- 两个模型只输出三个原子字段及各自的精确原文 span；不输出 accept/edit/reject。
- 无效或缺失 span 不进入 composer；无依据的危险原子保守拒绝。
- 最终标签完全由 deterministic composer 根据 grounded atoms 生成。
- Phase51-55 holdout 只作为历史诊断，没有进入 Phase56 calibration、holdout 或训练。
- runtime replay 调用数：`0`；boundary clause design：未创建。
- Phase56 不训练、不创建 adapter、不接 Hermes、不改 router、runtime prompt 或产品默认路径，不自动 promote。
""",
    )
    _write_text(
        EVIDENCE_ROOT / "phase56-runbook.md",
        """# Phase56 Runbook

```bash
.venv/bin/python tools/phase56_prepare.py --clean-evidence
.venv/bin/python tools/phase56_span_evaluator.py --split calibration
# Only after calibration status=qualified:
.venv/bin/python tools/phase56_span_evaluator.py --split holdout
.venv/bin/python tools/phase56_finalize_evidence.py
.venv/bin/python tools/phase56_validate.py
```

The independent holdout remains sealed until calibration qualifies. Models emit atomic fields plus exact evidence spans; deterministic grounding validates each atom before the composer owns every final label.
""",
    )
    next_goal = (
        "Build Phase57 as a frozen historical external-replay qualification for the Phase56 span-grounded atomic evaluator. "
        "Re-blind Phase51-55 holdout responses as historical diagnostics, run the unchanged span schema, grounding rules, and "
        "deterministic composer, and require each "
        "historical phase at or above 0.95 with zero false accepts and zero hard-rule overrides against two safe accepts. Do not "
        "tune on replay results. Only after this external replay qualifies may Phase58 design one minimal runtime contract A/B. "
        "Do not train, attach Hermes, or change the product default."
        if decision["evaluator_manual_review_use_allowed"]
        else
        "Build Phase57 from the saved Phase56 raw-versus-grounded failure transitions. Treat the Phase56 holdout as sealed historical "
        "diagnostics only, make one structural extraction improvement without tuning on holdout rows, and freeze completely new "
        "calibration and holdout splits with the same or stricter grounding, field, category, and zero-false-accept gates. Do not replay "
        "runtime outputs, train, attach Hermes, or change the product default."
    )
    _write_text(EVIDENCE_ROOT / "next-pursuit-goal.md", f"# Next Pursuit Goal\n\n{next_goal}\n")
    manifest = _manifest()
    _write_json(EVIDENCE_ROOT / "evidence_manifest.json", manifest)
    finalization = {
        "kind": "phase56_finalization_state",
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
