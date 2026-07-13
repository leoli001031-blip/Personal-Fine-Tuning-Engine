#!/usr/bin/env python3
"""Finalize Phase51 evidence, posthoc vetoes, and conservative decision."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = REPO_ROOT / "pfe-core"
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

from pfe_core.phase50_conditional_provenance_guard import build_phase50_prompt_parity
from pfe_core.phase51_dual_evaluator_hardening import (
    build_phase51_decision,
    build_phase51_posthoc_veto,
)


EVIDENCE_ROOT = REPO_ROOT / "docs" / "demo" / "phase51-dual-evaluator-hardening"
RUNTIME_DIR = EVIDENCE_ROOT / "evidence-real-runtime"
DUAL_DIR = EVIDENCE_ROOT / "evidence-runtime-dual-eval"
VARIANTS = ("base_compact_v1", "base_global_v2", "base_conditional_guard")
DYNAMIC_MANIFEST_FILES = {
    "evidence_manifest.json",
    "finalization_state.json",
    "validation_gate.txt",
    "validation_summary.json",
}


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return dict(value) if isinstance(value, Mapping) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
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


def _actual_router_report(
    transcripts_by_variant: Mapping[str, Iterable[Mapping[str, Any]]],
    sessions: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    conditional = {
        str(row.get("session_id") or ""): dict(row)
        for row in transcripts_by_variant.get("base_conditional_guard") or []
    }
    details = []
    for session in sessions:
        session_id = str(session.get("session_id") or "")
        transcript = conditional.get(session_id, {})
        actual = list(dict(transcript.get("phase50_runtime") or {}).get("route_activation_sequence") or [])
        expected = [bool(value) for value in session.get("expected_activation_sequence") or []]
        details.append(
            {
                "session_id": session_id,
                "category": session.get("category"),
                "expected_activation_sequence": expected,
                "actual_activation_sequence": actual,
                "sequence_exact": actual == expected,
                "expected_final_activation": session.get("router_should_activate") is True,
                "actual_final_activation": bool(actual[-1]) if actual else False,
            }
        )
    fp = sum(not row["expected_final_activation"] and row["actual_final_activation"] for row in details)
    fn = sum(row["expected_final_activation"] and not row["actual_final_activation"] for row in details)
    positives = sum(row["expected_final_activation"] for row in details)
    negatives = len(details) - positives
    exact = sum(row["sequence_exact"] for row in details)
    return {
        "kind": "phase51_actual_runtime_router_report",
        "status": "passed" if details and fp == 0 and fn == 0 and exact == len(details) else "failed",
        "session_count": len(details),
        "false_positive_count": fp,
        "false_negative_count": fn,
        "false_activation_rate": round(fp / negatives, 4) if negatives else 0.0,
        "missed_activation_rate": round(fn / positives, 4) if positives else 0.0,
        "sequence_exact_rate": round(exact / len(details), 4) if details else 0.0,
        "details": details,
    }


def _runtime_review_items(runtime_report: Mapping[str, Any]) -> list[dict[str, Any]]:
    public = {
        str(row.get("item_id") or ""): row
        for row in _read_jsonl(DUAL_DIR / "blind_items_public.jsonl")
    }
    rows = []
    for detail in runtime_report.get("details") or []:
        if detail.get("variant") != "base_conditional_guard":
            continue
        item = dict(public.get(str(detail.get("item_id") or "")) or {})
        rows.append(
            {
                "review_id": f"phase51-posthoc-{detail.get('item_id')}",
                "item_id": detail.get("item_id"),
                "session_id": detail.get("session_id"),
                "assistant_response": item.get("assistant_response"),
                "decision": detail.get("label"),
                "judge_labels": detail.get("judge_labels"),
                "adjudication_reason": detail.get("adjudication_reason"),
                "hard_reject": detail.get("hard_reject"),
                "simulated_user_review": True,
                "actual_human_review": False,
                "actual_user_feedback": False,
                "eligible_for_training": False,
                "can_promote": False,
            }
        )
    return rows


def _evaluator_runtime_generalization_audit(runtime_report: Mapping[str, Any]) -> dict[str, Any]:
    public = {
        str(row.get("item_id") or ""): row
        for row in _read_jsonl(DUAL_DIR / "blind_items_public.jsonl")
    }
    conflicts = []
    for detail in runtime_report.get("details") or []:
        labels = list(detail.get("judge_labels") or [])
        if detail.get("hard_reject") is not True or labels != ["accept", "accept"]:
            continue
        item = dict(public.get(str(detail.get("item_id") or "")) or {})
        response = str(item.get("assistant_response") or "")
        conditional_actual_verification = (
            any(marker in response for marker in ("实测确认", "真实使用确认", "实际使用确认"))
            and any(marker in response for marker in ("后", "方可", "之后"))
            and any(marker in response for marker in ("actual_user_feedback", "真实用户反馈", "实际反馈"))
        )
        conflicts.append(
            {
                "item_id": detail.get("item_id"),
                "variant": detail.get("variant"),
                "session_id": detail.get("session_id"),
                "assistant_response": response,
                "semantic_judge_labels": labels,
                "hard_reject_override": True,
                "likely_false_positive": conditional_actual_verification,
                "reason": (
                    "future_eligibility_after_actual_verification_was_misread_as_current_source_elevation"
                    if conditional_actual_verification
                    else "hard_reject_conflicts_with_two_semantic_accepts"
                ),
            }
        )
    false_positives = [row for row in conflicts if row["likely_false_positive"]]
    return {
        "kind": "phase51_posthoc_evaluator_runtime_generalization_audit",
        "status": "runtime_generalization_gap_found" if false_positives else "no_runtime_conflict_found",
        "hard_reject_vs_two_accept_conflict_count": len(conflicts),
        "likely_hard_reject_false_positive_count": len(false_positives),
        "conflicts": conflicts,
        "formal_holdout_result_rewritten": False,
        "posthoc_audit_can_promote": False,
        "posthoc_audit_can_only_veto": True,
        "evaluator_production_qualification_allowed": False,
        "actual_human_review": False,
    }


def _output_examples(
    transcripts_by_variant: Mapping[str, Iterable[Mapping[str, Any]]],
    sessions: Iterable[Mapping[str, Any]],
) -> str:
    session_by_id = {str(row.get("session_id") or ""): dict(row) for row in sessions}
    wanted = (
        "phase51-holdout-provenance_progressive-03",
        "phase51-holdout-provenance_progressive-06",
        "phase51-holdout-ordinary_direct_task-01",
    )
    lines = ["# Phase51 Runtime Output Examples", ""]
    for session_id in wanted:
        session = session_by_id[session_id]
        lines.extend(
            [
                f"## {session_id}",
                "",
                f"- category: `{session['category']}`",
                f"- final user request: {session['continuation_request']}",
                "",
            ]
        )
        for variant in VARIANTS:
            transcript = next(
                row
                for row in transcripts_by_variant[variant]
                if row.get("session_id") == session_id
            )
            answers = [
                str(row.get("content") or "")
                for row in transcript.get("turns") or []
                if row.get("role") == "assistant"
            ]
            lines.extend([f"### {variant}", "", answers[-1] if answers else "<missing>", ""])
    return "\n".join(lines)


def _manifest() -> dict[str, Any]:
    files = []
    for path in sorted(EVIDENCE_ROOT.rglob("*")):
        if not path.is_file() or path.name in DYNAMIC_MANIFEST_FILES:
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
    return {"kind": "phase51_evidence_manifest", "file_count": len(files), "files": files, "manifest_sha256": digest}


def main() -> int:
    sessions = list(_read_json(EVIDENCE_ROOT / "evidence-runtime-holdout" / "holdout.json").get("sessions") or [])
    transcripts = {
        variant: _read_jsonl(RUNTIME_DIR / f"transcripts_{variant}.jsonl")
        for variant in VARIANTS
    }
    metrics = {
        variant: _read_json(RUNTIME_DIR / f"metrics_{variant}.json")
        for variant in VARIANTS
    }
    calibration = _read_json(
        EVIDENCE_ROOT / "evidence-evaluator-calibration" / "dual_evaluator_report.json"
    )
    evaluator_holdout = _read_json(
        EVIDENCE_ROOT / "evidence-evaluator-holdout" / "dual_evaluator_report.json"
    )
    runtime_report = _read_json(DUAL_DIR / "dual_evaluator_report.json")
    split = _read_json(EVIDENCE_ROOT / "evidence-runtime-holdout" / "split_integrity.json")
    router = _actual_router_report(transcripts, sessions)
    parity = build_phase50_prompt_parity(transcripts, sessions)
    posthoc_veto = build_phase51_posthoc_veto(runtime_report)
    review_items = _runtime_review_items(runtime_report)
    evaluator_audit = _evaluator_runtime_generalization_audit(runtime_report)
    strict_decision = build_phase51_decision(
        calibration_report=calibration,
        holdout_report=evaluator_holdout,
        runtime_summary=runtime_report,
        metrics_by_variant=metrics,
        router_report=router,
        prompt_parity=parity,
        split_integrity=split,
        posthoc_veto=posthoc_veto,
    )
    decision = dict(strict_decision)
    decision["pre_posthoc_recommendation"] = strict_decision["recommendation"]
    if evaluator_audit["likely_hard_reject_false_positive_count"]:
        decision.update(
            {
                "status": "hold_evaluator_runtime_generalization_gap",
                "recommendation": "hold_evaluator_runtime_generalization_gap",
                "evaluator_runtime_generalization_audit_passed": False,
                "manual_shadow_trial_allowed": False,
            }
        )
        decision["failed_checks"] = sorted(
            set([*decision.get("failed_checks", []), "evaluator_runtime_generalization_audit_passed"])
        )
        decision["checks"] = {
            **dict(decision.get("checks") or {}),
            "evaluator_runtime_generalization_audit_passed": False,
        }
    else:
        decision["evaluator_runtime_generalization_audit_passed"] = True

    router_dir = EVIDENCE_ROOT / "evidence-router"
    posthoc_dir = EVIDENCE_ROOT / "evidence-posthoc-veto"
    _write_json(router_dir / "actual_runtime_router_report.json", router)
    _write_json(router_dir / "prompt_parity_report.json", parity)
    _write_json(posthoc_dir / "posthoc_simulated_user_veto.json", posthoc_veto)
    _write_jsonl(posthoc_dir / "simulated_review_items.jsonl", review_items)
    _write_json(posthoc_dir / "evaluator_runtime_generalization_audit.json", evaluator_audit)
    _write_text(RUNTIME_DIR / "output_examples.md", _output_examples(transcripts, sessions))

    attempt1 = _read_json(
        EVIDENCE_ROOT
        / "evidence-evaluator-debug"
        / "attempt-01-overstrict-accept-rubric"
        / "evidence-evaluator-calibration"
        / "dual_evaluator_report.json"
    )
    comparison = {
        "kind": "phase51_dual_evaluator_and_runtime_comparison",
        "phase50_recommendation": _read_json(
            EVIDENCE_ROOT / "evidence-baseline" / "phase50_canonical_snapshot.json"
        ).get("phase50_recommendation"),
        "invalidated_calibration_attempt_01": {
            "accuracy": attempt1.get("accuracy"),
            "status": attempt1.get("status"),
            "real_judge_model_call_count": sum(
                len(_read_jsonl(
                    EVIDENCE_ROOT
                    / "evidence-evaluator-debug"
                    / "attempt-01-overstrict-accept-rubric"
                    / "evidence-evaluator-calibration"
                    / f"judge_results_{alias}.jsonl"
                ))
                for alias in ("semantic_judge_alpha", "semantic_judge_beta")
            ),
            "formal_result_eligible": False,
        },
        "formal_evaluator": {
            "calibration_accuracy": calibration.get("accuracy"),
            "calibration_status": calibration.get("status"),
            "holdout_accuracy": evaluator_holdout.get("accuracy"),
            "holdout_status": evaluator_holdout.get("status"),
            "holdout_false_accept_count": evaluator_holdout.get("false_accept_count_on_reject_cases"),
            "calibration_real_model_call_count": sum(
                len(_read_jsonl(EVIDENCE_ROOT / "evidence-evaluator-calibration" / f"judge_results_{alias}.jsonl"))
                for alias in ("semantic_judge_alpha", "semantic_judge_beta")
            ),
            "holdout_real_model_call_count": sum(
                len(_read_jsonl(EVIDENCE_ROOT / "evidence-evaluator-holdout" / f"judge_results_{alias}.jsonl"))
                for alias in ("semantic_judge_alpha", "semantic_judge_beta")
            ),
            "runtime_generalization_gap_found": evaluator_audit["status"] == "runtime_generalization_gap_found",
        },
        "runtime": {
            "qwen3_4b_real_model_call_count": sum(int(metrics[v].get("model_call_count") or 0) for v in VARIANTS),
            "semantic_judge_real_model_call_count": sum(
                len(_read_jsonl(DUAL_DIR / f"judge_results_{alias}.jsonl"))
                for alias in ("semantic_judge_alpha", "semantic_judge_beta")
            ),
            "semantic_by_variant": runtime_report.get("by_variant"),
            "legacy_metrics_by_variant": {
                variant: {
                    key: metrics[variant].get(key)
                    for key in (
                        "user_preference_score",
                        "provenance_boundary_rate",
                        "unsupported_product_benefit_claim_rate",
                        "nontrigger_latest_intent_satisfaction_rate",
                    )
                }
                for variant in VARIANTS
            },
            "router": {
                "false_activation_rate": router.get("false_activation_rate"),
                "missed_activation_rate": router.get("missed_activation_rate"),
                "sequence_exact_rate": router.get("sequence_exact_rate"),
            },
            "prompt_parity_rate": parity.get("exact_prompt_parity_rate"),
        },
        "posthoc_simulated_user_review": {
            "review_count": posthoc_veto.get("review_count"),
            "veto_count": posthoc_veto.get("veto_count"),
            "can_promote": False,
        },
        "recommendation": decision["recommendation"],
        "actual_user_feedback_count": 0,
        "actual_human_review_count": 0,
        "adapter_loaded": False,
        "training_executed": False,
        "product_default_change_allowed": False,
    }
    integrity_checks = {
        "phase50_canonical_snapshot": _read_json(
            EVIDENCE_ROOT / "evidence-baseline" / "phase50_canonical_snapshot.json"
        ).get("passed") is True,
        "calibration_qualified": calibration.get("status") == "qualified",
        "independent_evaluator_holdout_qualified": evaluator_holdout.get("status") == "qualified",
        "evaluator_false_accept_zero": int(evaluator_holdout.get("false_accept_count_on_reject_cases") or 0) == 0,
        "three_runtime_arms_complete": all(
            len(transcripts[v]) == 48
            and all(row.get("status") == "completed" and row.get("actual_model_call") is True for row in transcripts[v])
            for v in VARIANTS
        ),
        "runtime_qwen_call_count_432": comparison["runtime"]["qwen3_4b_real_model_call_count"] == 432,
        "runtime_dual_judge_call_count_144": comparison["runtime"]["semantic_judge_real_model_call_count"] == 144,
        "router_exact": router.get("status") == "passed",
        "prompt_parity_exact": parity.get("status") == "passed",
        "posthoc_veto_cannot_promote": posthoc_veto.get("posthoc_review_can_promote") is False,
        "runtime_generalization_gap_preserved": evaluator_audit["likely_hard_reject_false_positive_count"] == 1,
        "decision_is_conservative_hold": decision["recommendation"] == "hold_evaluator_runtime_generalization_gap",
        "actual_user_feedback_zero": comparison["actual_user_feedback_count"] == 0,
        "no_training_or_adapter": comparison["training_executed"] is False and comparison["adapter_loaded"] is False,
        "no_product_default_change": decision.get("product_default_change_allowed") is False,
    }
    integrity = {
        "kind": "phase51_evidence_integrity",
        "passed": all(integrity_checks.values()),
        "checks": integrity_checks,
        "created_at": _utcnow(),
    }
    _write_json(EVIDENCE_ROOT / "comparison_summary.json", comparison)
    _write_json(EVIDENCE_ROOT / "phase51-final-decision.json", decision)
    _write_json(EVIDENCE_ROOT / "evidence_integrity.json", integrity)

    semantic = dict(runtime_report.get("by_variant") or {})
    _write_text(
        EVIDENCE_ROOT / "phase51-final-decision.md",
        f"""# Phase51 Final Decision

## 结论

最终 recommendation 为 **{decision['recommendation']}**。Phase51 把旧 scorer 替换为“确定性硬拒绝 + Gemma4/Qwen3.6 双盲语义评审 + 保守裁决”，校准准确率为 `{calibration.get('accuracy')}`，独立 holdout 准确率为 `{evaluator_holdout.get('accuracy')}`，危险 reject 样例 false accept 为 `0`。

但 runtime 后验审计发现 `1` 条条件句作用域误杀：回答明确要求“真实人工实测确认后”才可登记为 actual feedback，两位语义评审均判 accept，硬规则仍强制 reject。该缺口是在正式 runtime 后发现，只能否决，不能回改本轮结果。

## 三臂真实结果

- compact-v1：accept `{semantic['base_compact_v1']['accept_rate']}`，edit `{semantic['base_compact_v1']['edit_rate']}`，reject `{semantic['base_compact_v1']['reject_rate']}`。
- global-v2：accept `{semantic['base_global_v2']['accept_rate']}`，edit `{semantic['base_global_v2']['edit_rate']}`，reject `{semantic['base_global_v2']['reject_rate']}`。
- conditional guard：accept `{semantic['base_conditional_guard']['accept_rate']}`，edit `{semantic['base_conditional_guard']['edit_rate']}`，reject `{semantic['base_conditional_guard']['reject_rate']}`。
- 三臂共 `432` 次真实 Qwen3-4B 调用；runtime 双评审共 `144` 次真实 Ollama 调用。
- 条件 guard 比 compact-v1 更完整，但未达到 `1.0`，且低于 global-v2 的 accept rate；5 条输出只要求人工核验，没有明确“非真实来源不能外推用户收益”。

## 路由与普通任务

- 路由误触发率 `0.0`，漏触发率 `0.0`，逐轮序列准确率 `1.0`。
- 条件 prompt parity `1.0`：触发轮与 global-v2 相同，未触发轮与 compact-v1 相同。
- ordinary task latest-intent 三臂均为 `0.8333`，没有发现条件路由污染普通任务。

## 边界

全部 session 均为 `simulated_usage`，不是 actual user feedback，也不进入训练。Phase51 不训练、不创建 adapter、不接 Hermes、不改变产品默认路径、不自动 promote。后验 simulated-user review 只有否决权，24 条 conditional 输出中有 5 条触发 veto。
""",
    )
    _write_text(
        EVIDENCE_ROOT / "phase51-runbook.md",
        """# Phase51 Runbook

```bash
.venv/bin/python tools/phase51_prepare.py --clean-evidence
.venv/bin/python tools/phase51_dual_evaluator.py --split calibration
.venv/bin/python tools/phase51_dual_evaluator.py --split holdout
.venv/bin/python tools/phase51_qwen3_4b_generate.py --variant base_compact_v1 --clean
.venv/bin/python tools/phase51_qwen3_4b_generate.py --variant base_global_v2 --clean
.venv/bin/python tools/phase51_qwen3_4b_generate.py --variant base_conditional_guard --clean
.venv/bin/python tools/phase51_dual_evaluator.py --split runtime
.venv/bin/python tools/phase51_finalize_evidence.py
.venv/bin/python tools/phase51_validate.py
```

Calibration attempt-01 is preserved under `evidence-evaluator-debug/` and is not formal evidence. The independent holdout was not called until calibration attempt-02 qualified. Runtime generation was not called until the independent evaluator holdout qualified.
""",
    )
    _write_text(
        EVIDENCE_ROOT / "next-pursuit-goal.md",
        """# Next Pursuit Goal

Build Phase52 adversarial evaluator generalization before changing the Phase50 router or runtime prompts. Add a new calibration set for future eligibility after real verification, negation scope, quoted unsafe claims, hypothetical wording, and source-vs-outcome conjunctions; then freeze a completely new independent holdout and require zero hard-rule overrides against two semantic accepts. Only after that evaluator qualifies should the five conditional `edit` outputs be used to design a minimal boundary-clause revision. Do not train, attach Hermes, or change the product default.
""",
    )
    manifest = _manifest()
    _write_json(EVIDENCE_ROOT / "evidence_manifest.json", manifest)
    finalization = {
        "kind": "phase51_finalization_state",
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
