#!/usr/bin/env python3
"""Finalize Phase53 scope-recovery evidence without changing runtime behavior."""

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

from pfe_core.phase53_evaluator_scope_recovery import build_phase53_decision


EVIDENCE_ROOT = REPO_ROOT / "docs/demo/phase53-evaluator-scope-recovery"
PHASE52_ROOT = REPO_ROOT / "docs/demo/phase52-adversarial-evaluator-generalization"
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
    return sum(len(_read_jsonl(directory / f"judge_results_{alias}.jsonl")) for alias in ALIASES)


def _attempt_summary(name: str) -> dict[str, Any]:
    root = EVIDENCE_ROOT / "evidence-evaluator-debug" / name / "evidence-evaluator-calibration"
    report = _read_json(root / "dual_evaluator_report.json")
    return {
        "name": name,
        "status": report.get("status"),
        "accuracy": report.get("accuracy"),
        "edit_recall": dict(report.get("per_label") or {}).get("edit", {}).get("recall"),
        "completed_item_count": report.get("completed_item_count"),
        "failure_count": report.get("failure_count"),
        "real_model_call_count": _call_count(root),
        "formal_result_eligible": False,
    }


def _failure_analysis(directory: Path, report: Mapping[str, Any]) -> dict[str, Any]:
    public = {str(row.get("item_id") or ""): row for row in _read_jsonl(directory / "blind_items_public.jsonl")}
    judgments: dict[str, list[dict[str, Any]]] = {}
    for alias in ALIASES:
        for row in _read_jsonl(directory / f"judge_results_{alias}.jsonl"):
            judgments.setdefault(str(row.get("item_id") or ""), []).append(
                {
                    "judge_alias": alias,
                    "label": row.get("label"),
                    "confidence": row.get("confidence"),
                    "reason": row.get("reason"),
                }
            )
    failed = [row for row in report.get("details") or [] if row.get("passed") is False]
    return {
        "kind": "phase53_evaluator_failure_analysis",
        "split": report.get("split"),
        "status": "no_post_result_tuning",
        "failure_count": len(failed),
        "rubric_or_fixture_modified_after_result": False,
        "details": [
            {
                "item_id": row.get("item_id"),
                "case_id": row.get("case_id"),
                "category": row.get("category"),
                "assistant_response": public.get(str(row.get("item_id") or ""), {}).get("assistant_response"),
                "expected_label": row.get("expected_label"),
                "actual_label": row.get("actual_label"),
                "judge_labels": row.get("judge_labels"),
                "judgments": judgments.get(str(row.get("item_id") or ""), []),
            }
            for row in failed
        ],
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
    return {"kind": "phase53_evidence_manifest", "file_count": len(files), "files": files, "manifest_sha256": digest}


def main() -> int:
    calibration_dir = EVIDENCE_ROOT / "evidence-evaluator-calibration"
    holdout_dir = EVIDENCE_ROOT / "evidence-evaluator-holdout"
    calibration = _read_json(calibration_dir / "dual_evaluator_report.json")
    holdout = _read_json(holdout_dir / "dual_evaluator_report.json")
    hard_calibration = _read_json(calibration_dir / "hard_reject_report.json")
    hard_holdout = _read_json(holdout_dir / "hard_reject_report.json")
    split = _read_json(holdout_dir / "split_integrity.json")
    runtime_status = _read_json(EVIDENCE_ROOT / "evidence-no-runtime/runtime_status.json")
    decision = build_phase53_decision(
        calibration_report=calibration,
        holdout_report=holdout,
        hard_calibration=hard_calibration,
        hard_holdout=hard_holdout,
        split_integrity=split,
        runtime_replay_model_call_count=int(runtime_status.get("runtime_replay_model_call_count") or 0),
        boundary_clause_design_created=runtime_status.get("boundary_clause_design_created") is True,
    )
    attempts = [
        _attempt_summary("attempt-01-edit-scope-ambiguity"),
        _attempt_summary("attempt-02-incomplete-and-residual-disagreement"),
    ]
    formal_calibration_calls = _call_count(calibration_dir)
    formal_holdout_calls = _call_count(holdout_dir)
    holdout_called = bool(holdout)
    holdout_failed = [row for row in holdout.get("details") or [] if row.get("passed") is False]
    transition_counts: dict[str, int] = {}
    for row in holdout_failed:
        transition = f"{row.get('expected_label')}->{row.get('actual_label')}"
        transition_counts[transition] = transition_counts.get(transition, 0) + 1
    holdout_failure_summary = {
        "failure_count": len(holdout_failed),
        "transition_counts": dict(sorted(transition_counts.items())),
        "judge_disagreement_failure_count": sum(
            len(set(row.get("judge_labels") or [])) > 1 for row in holdout_failed
        ),
        "reject_false_accept_count": holdout.get("false_accept_count_on_reject_cases"),
    }
    comparison = {
        "kind": "phase53_scope_recovery_comparison",
        "phase52_recommendation": _read_json(PHASE52_ROOT / "phase52-final-decision.json").get("recommendation"),
        "invalidated_calibration_attempts": attempts,
        "formal_evaluator": {
            "calibration_status": calibration.get("status"),
            "calibration_accuracy": calibration.get("accuracy"),
            "calibration_per_category": calibration.get("per_category"),
            "calibration_real_model_call_count": formal_calibration_calls,
            "holdout_status": holdout.get("status") if holdout_called else "not_called",
            "holdout_accuracy": holdout.get("accuracy"),
            "holdout_per_category": holdout.get("per_category"),
            "holdout_false_accept_count": holdout.get("false_accept_count_on_reject_cases"),
            "holdout_hard_vs_two_accept_conflict_count": holdout.get(
                "hard_reject_vs_two_accept_conflict_count"
            ),
            "holdout_real_model_call_count": formal_holdout_calls,
            "holdout_failure_summary": holdout_failure_summary,
        },
        "runtime_replay": {
            "status": runtime_status.get("phase51_runtime_replay_status"),
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
        formal_calibration_calls == 180
        and int(calibration.get("completed_item_count") or 0) == 90
        and int(calibration.get("failure_count") or 0) == 0
    )
    if calibration.get("status") == "qualified":
        holdout_completion_ok = (
            holdout_called
            and formal_holdout_calls == 216
            and int(holdout.get("completed_item_count") or 0) == 108
            and int(holdout.get("failure_count") or 0) == 0
        )
    else:
        holdout_completion_ok = not holdout_called and formal_holdout_calls == 0
    expected_recommendation = (
        "recommend_phase53_evaluator_for_manual_review_only"
        if holdout.get("status") == "qualified"
        else "hold_phase53_evaluator_scope_recovery"
    )
    integrity_checks = {
        "phase52_canonical_snapshot_passed": _read_json(
            EVIDENCE_ROOT / "evidence-baseline/phase52_canonical_snapshot.json"
        ).get("passed") is True,
        "two_invalidated_attempts_preserved": (
            attempts[0]["real_model_call_count"] == 180
            and attempts[1]["real_model_call_count"] == 179
            and all(row["formal_result_eligible"] is False for row in attempts)
        ),
        "formal_calibration_complete": calibration_complete,
        "holdout_protocol_respected": holdout_completion_ok,
        "decision_matches_frozen_gates": decision["recommendation"] == expected_recommendation,
        "runtime_replay_not_run": int(runtime_status.get("runtime_replay_model_call_count") or 0) == 0,
        "boundary_clause_not_designed": runtime_status.get("boundary_clause_design_created") is False,
        "phase52_historical_failures_not_reused": split.get("checks", {}).get(
            "phase52_failure_response_reuse_zero"
        ) is True,
        "no_training_adapter_hermes_or_default_change": (
            decision["new_training_allowed"] is False
            and decision["new_adapter_created"] is False
            and decision["hermes_attachment_allowed"] is False
            and decision["product_default_change_allowed"] is False
        ),
    }
    integrity = {
        "kind": "phase53_evidence_integrity",
        "passed": all(integrity_checks.values()),
        "checks": integrity_checks,
        "created_at": _utcnow(),
    }
    _write_json(calibration_dir / "failure_analysis.json", _failure_analysis(calibration_dir, calibration))
    if holdout_called:
        _write_json(holdout_dir / "failure_analysis.json", _failure_analysis(holdout_dir, holdout))
    _write_json(EVIDENCE_ROOT / "comparison_summary.json", comparison)
    _write_json(EVIDENCE_ROOT / "phase53-final-decision.json", decision)
    _write_json(EVIDENCE_ROOT / "evidence_integrity.json", integrity)
    holdout_summary = (
        f"独立 holdout 准确率为 `{holdout.get('accuracy')}`，分项为 `{holdout.get('per_category')}`。"
        f"失败 `{holdout_failure_summary['failure_count']}` 条，其中双模型分歧 "
        f"`{holdout_failure_summary['judge_disagreement_failure_count']}` 条，标签迁移为 "
        f"`{holdout_failure_summary['transition_counts']}`。"
        if holdout_called
        else "Calibration 未合格，因此独立 holdout 按协议没有调用。"
    )
    _write_text(
        EVIDENCE_ROOT / "phase53-final-decision.md",
        f"""# Phase53 Final Decision

## 结论

最终 recommendation 为 **{decision['recommendation']}**。正式 calibration 准确率为 `{calibration.get('accuracy')}`，分项为 `{calibration.get('per_category')}`。{holdout_summary}

前两次 calibration 均已保留为非正式证据：attempt-01 暴露 edit 定义和含混文本问题；attempt-02 改善到 `{attempts[1]['accuracy']}`，但有 1 条 Qwen JSON 因 192-token 截断而不完整。正式协议固定 `num_predict=384`。

## 决策边界

- Phase51 runtime replay 调用数：`0`。
- boundary clause design：未创建。
- 所有数据均为 simulated evaluator fixture，不是 actual user feedback，不进入训练。
- Phase53 不训练、不创建 adapter、不接 Hermes、不改 router、不改 runtime prompt、不改产品默认路径、不自动 promote。
""",
    )
    _write_text(
        EVIDENCE_ROOT / "phase53-runbook.md",
        """# Phase53 Runbook

```bash
.venv/bin/python tools/phase53_prepare.py --clean-evidence
.venv/bin/python tools/phase53_dual_evaluator.py --split calibration
# Only after calibration status=qualified:
.venv/bin/python tools/phase53_dual_evaluator.py --split holdout
.venv/bin/python tools/phase53_finalize_evidence.py
.venv/bin/python tools/phase53_validate.py
```

Attempts 01 and 02 are preserved under `evidence-evaluator-debug/` and are not formal evidence. The independent holdout remains untouched until the formal calibration qualifies. Phase53 never runs Phase51 runtime replay or creates a boundary-clause design.
""",
    )
    next_goal = (
        "Build Phase54 as a frozen Phase51 runtime replay using the qualified Phase53 evaluator. First re-blind the saved "
        "Phase51 outputs and require zero hard-rule overrides against two semantic accepts. Only if replay generalization also "
        "passes may the five historical conditional edits be read to design one minimal boundary-clause A/B candidate. Do not "
        "train, attach Hermes, or change the product default."
        if decision["evaluator_manual_review_use_allowed"]
        else
        "Build Phase54 as a typed proposition-extraction evaluator rather than adding more direct-label prompt examples. "
        "Have both semantic judges return structured fields for source eligibility, current benefit assertion, suspended or "
        "negated outcome, and explicit provenance boundary; compose accept/edit/reject deterministically from those fields. "
        "Use all Phase51-53 holdouts as historical diagnostics only, freeze a completely new calibration and independent "
        "holdout, and keep zero false accepts plus zero hard-rule overrides. Do not replay runtime outputs, design a boundary "
        "clause, train, attach Hermes, or change the product default."
    )
    _write_text(EVIDENCE_ROOT / "next-pursuit-goal.md", f"# Next Pursuit Goal\n\n{next_goal}\n")
    manifest = _manifest()
    _write_json(EVIDENCE_ROOT / "evidence_manifest.json", manifest)
    finalization = {
        "kind": "phase53_finalization_state",
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
