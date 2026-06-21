#!/usr/bin/env python3
"""Generate Phase28 real-feedback loop-engineering evidence."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import shutil
from typing import Any, Iterable, Mapping

from pfe_core.phase28_real_feedback_loop_engineering import (
    append_phase28_import_batch,
    apply_phase28_review_decision,
    build_phase28_comparison_summary,
    build_phase28_feedback_templates,
    build_phase28_import_batch,
    build_phase28_loop_state,
    build_phase28_readiness,
    build_phase28_review_state,
    build_phase28_simulation_review,
    build_phase28_task_pack,
    build_phase28_training_attempt,
    load_phase28_state,
    phase28_payloads_from_csv,
    phase28_payloads_from_jsonl,
)


PHASE27_SIM_DIR = Path("docs/demo/phase27-actual-feedback-review-training-loop/simulation")
PHASE28_DIR = Path("docs/demo/phase28-real-feedback-loop-engineering")
PHASE28_WORKSPACE = "phase28-evidence"


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = dict(payload)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return data


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "task_id",
        "collection_id",
        "scenario_id",
        "prompt",
        "messages",
        "runtime_output",
        "response_under_review",
        "metadata",
        "feedback_source",
        "feedback_action",
        "edited_text",
        "user_feedback",
        "signal_id",
        "attestation",
        "reviewer_decision",
        "reviewer_reason",
    ]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({column: row.get(column, "") for column in columns})
    path.write_text(buffer.getvalue(), encoding="utf-8")


def _clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _read_payloads(*, jsonl_path: str | None, csv_path: str | None) -> list[dict[str, Any]]:
    if jsonl_path:
        return phase28_payloads_from_jsonl(Path(jsonl_path).read_text(encoding="utf-8"))
    if csv_path:
        return phase28_payloads_from_csv(Path(csv_path).read_text(encoding="utf-8"))
    return []


def _read_review_decisions(path: str | None) -> list[dict[str, Any]]:
    if not path:
        return []
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return []
    if isinstance(payload, list):
        return [dict(item) for item in payload if isinstance(item, Mapping)]
    if isinstance(payload, Mapping):
        return [dict(item) for item in payload.get("items") or [] if isinstance(item, Mapping)]
    return []


def _model_inventory() -> list[dict[str, Any]]:
    roots = [
        Path.home() / ".cache" / "huggingface" / "hub",
        Path.home() / ".cache" / "modelscope" / "hub",
        Path.home() / ".lmstudio" / "models",
        Path("models"),
        Path("local_models"),
    ]
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for root in roots:
        if not root.exists():
            continue
        for path in list(root.glob("*"))[:300]:
            lower = str(path).lower()
            if "qwen" not in lower:
                continue
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            non_generative = any(token in lower for token in ("embedding", "reranker", "vl", "tts", "audio", "omni"))
            looks_oversized = "27b" in lower or "32b" in lower or "72b" in lower
            trainable_size_hint = any(token in lower for token in ("0.6b", "1.5b", "1.7b", "3b", "4b", "7b", "8b", "14b"))
            candidates.append(
                {
                    "name": path.name,
                    "path": str(path),
                    "exists": path.exists(),
                    "qwen": True,
                    "non_generative": non_generative,
                    "looks_oversized_for_local_training": looks_oversized,
                    "trainable": bool(trainable_size_hint and not non_generative and not looks_oversized and not lower.endswith(".gguf")),
                    "selection_note": "local inventory hint only; Phase28 does not auto-train",
                }
            )
    return candidates


def _runbook(summary: Mapping[str, Any]) -> str:
    loop_state = _dict(summary.get("loop_state"))
    readiness = _dict(summary.get("training_readiness"))
    return f"""# Phase28 Runbook

## Goal

Run the real-feedback loop-engineering path: collect attested feedback, import,
validate, review, build candidates, and open training readiness only when the
approved actual-feedback threshold is met.

## Current Evidence

- Task count: {summary.get("task_count")}
- Actual feedback count: {summary.get("actual_feedback_count")}
- Approved actual candidates: {summary.get("approved_actual_candidate_count")}
- Loop state: {loop_state.get("current_state")}
- Readiness: {readiness.get("status")}
- Blockers: {", ".join(readiness.get("blockers") or [])}

## Default Command

```bash
.venv/bin/python tools/phase28_real_feedback_loop_engineering.py --clean-evidence
```

## Optional Real Feedback Import

```bash
.venv/bin/python tools/phase28_real_feedback_loop_engineering.py --feedback-jsonl path/to/actual_feedback.jsonl --review-decisions-json path/to/review_decisions.json
```

Templates and Phase27 simulation rows are not valid actual feedback.
"""


def _final_decision(summary: Mapping[str, Any]) -> str:
    loop_state = _dict(summary.get("loop_state"))
    attempt = _dict(summary.get("training_attempt"))
    readiness = _dict(summary.get("training_readiness"))
    return f"""# Phase28 Final Decision

## Decision

Phase28 has a loop-engineering state and real-feedback task pack. Current
evidence does not include attested real feedback, so training remains blocked.

## Evidence

- Actual feedback count: {summary.get("actual_feedback_count")}
- Approved actual candidates: {summary.get("approved_actual_candidate_count")}
- Loop state: {loop_state.get("current_state")}
- Training attempt: {attempt.get("status")}
- Readiness: {readiness.get("status")}
- Blockers: {", ".join(readiness.get("blockers") or [])}

## Recommendation

{summary.get("final_recommendation")}
"""


def _next_pursuit_goal(summary: Mapping[str, Any]) -> str:
    return f"""目标：推进 PFE Phase29：真实反馈达标后的 Qwen 小步训练收益验证。

当前 Phase28 状态：
- actual_feedback_count = {summary.get("actual_feedback_count")}
- approved_actual_candidate_count = {summary.get("approved_actual_candidate_count")}
- loop_state = {_dict(summary.get("loop_state")).get("current_state")}
- final_recommendation = {summary.get("final_recommendation")}

下一步只有在真实 approved feedback >= 12 时才启动训练；否则继续收集和审核真实反馈，不使用 template/simulation 数据补齐。
"""


def generate_phase28_evidence(
    *,
    clean_evidence: bool = False,
    feedback_jsonl: str | None = None,
    feedback_csv: str | None = None,
    review_decisions_json: str | None = None,
) -> dict[str, Any]:
    if clean_evidence:
        _clean_dir(PHASE28_DIR)
    for subdir in ("evidence", "evidence-collection", "evidence-review", "evidence-training", "evidence-eval"):
        (PHASE28_DIR / subdir).mkdir(parents=True, exist_ok=True)

    task_pack = build_phase28_task_pack(count=36)
    templates = build_phase28_feedback_templates(task_pack)
    payloads = _read_payloads(jsonl_path=feedback_jsonl, csv_path=feedback_csv)
    import_batch = build_phase28_import_batch(payloads)
    store = PHASE28_DIR / "evidence-review" / f"phase28_real_feedback_{PHASE28_WORKSPACE}.json"
    if store.exists():
        store.unlink()
    append_phase28_import_batch(store, import_batch)
    for decision in _read_review_decisions(review_decisions_json):
        apply_phase28_review_decision(store, decision)
    state = load_phase28_state(store)
    signals = [dict(item) for item in state.get("signals") or [] if isinstance(item, Mapping)]
    review_decisions = [dict(item) for item in state.get("review_decisions") or [] if isinstance(item, Mapping)]
    readiness = build_phase28_readiness(signals=signals, review_decisions=review_decisions, local_models=_model_inventory())
    review_state = build_phase28_review_state(signals=signals, review_decisions=review_decisions)
    training_attempt = build_phase28_training_attempt(readiness)
    loop_state = build_phase28_loop_state(
        readiness_payload=readiness,
        training_attempt=training_attempt,
        evidence_path=str(PHASE28_DIR / "loop_state.json"),
        import_batch=import_batch,
    )
    simulation_review = build_phase28_simulation_review(PHASE27_SIM_DIR)
    summary = build_phase28_comparison_summary(
        task_pack=task_pack,
        import_batch=import_batch,
        readiness_payload=readiness,
        training_attempt=training_attempt,
        loop_state=loop_state,
        simulation_review=simulation_review,
    )
    candidates = _dict(readiness.get("candidate_artifacts"))

    collection_dir = PHASE28_DIR / "evidence-collection"
    review_dir = PHASE28_DIR / "evidence-review"
    training_dir = PHASE28_DIR / "evidence-training"
    eval_dir = PHASE28_DIR / "evidence-eval"
    evidence_dir = PHASE28_DIR / "evidence"

    _write_json(PHASE28_DIR / "task_pack.json", task_pack)
    _write_jsonl(PHASE28_DIR / "actual_feedback_template.jsonl", templates["jsonl_rows"])
    _write_csv(PHASE28_DIR / "actual_feedback_template.csv", templates["csv_rows"])
    _write_json(PHASE28_DIR / "loop_state.json", loop_state)
    _write_json(evidence_dir / "comparison_summary.json", summary)
    _write_json(evidence_dir / "phase27_simulation_review.json", simulation_review)

    _write_json(collection_dir / "task_pack.json", task_pack)
    _write_json(collection_dir / "feedback_templates.json", templates)
    _write_jsonl(collection_dir / "actual_feedback_batch.jsonl", payloads)
    _write_json(collection_dir / "actual_feedback_import_batch.json", import_batch)
    _write_jsonl(collection_dir / "accepted_actual_feedback_signals.jsonl", import_batch["accepted_signals"])
    _write_json(collection_dir / "api_task_pack_payload.json", {
        "kind": "phase28_task_pack_surface",
        "status": "ready",
        "task_pack": task_pack,
        "auto_promotion_allowed": False,
    })

    _write_json(review_dir / "review_queue.json", review_state.get("queue") or {})
    _write_json(review_dir / "review_summary.json", review_state.get("reviewed") or {})
    _write_json(review_dir / "review_decisions.json", {"kind": "phase28_review_decisions", "items": review_decisions})
    _write_json(review_dir / "reviewer_audit_log.json", {
        "kind": "phase28_reviewer_audit_log",
        "items": [dict(item) for item in state.get("reviewer_audit_log") or [] if isinstance(item, Mapping)],
    })
    _write_json(review_dir / "api_review_queue_payload.json", {
        "kind": "phase28_review_queue_surface",
        "status": "ready",
        "review_state": review_state,
        "training_readiness": readiness.get("training_readiness"),
        "auto_promotion_allowed": False,
    })

    _write_json(training_dir / "training_readiness_report.json", readiness["training_readiness"])
    _write_json(training_dir / "training_job_specs.json", readiness["training_job_specs"])
    _write_json(training_dir / "training_attempt.json", training_attempt)
    _write_json(training_dir / "model_selection.json", readiness["model_selection"])
    _write_json(training_dir / "candidate_manifest.json", candidates.get("candidate_manifest") or {})
    _write_json(training_dir / "candidate_quality_report.json", candidates.get("quality_report") or {})
    _write_jsonl(training_dir / "actual_feedback_sft_candidates.jsonl", candidates.get("sft_samples") or [])
    _write_jsonl(training_dir / "actual_feedback_dpo_pairs.jsonl", candidates.get("dpo_pairs") or [])

    _write_json(eval_dir / "holdout_integrity_check.json", readiness["holdout_integrity_check"])
    _write_json(eval_dir / "runtime_contract_eval_report.json", readiness["runtime_eval"])
    _write_json(eval_dir / "runtime_contract_decision.json", readiness["runtime_decision"])
    _write_json(eval_dir / "adapter_eval_decision.json", {
        "kind": "phase28_adapter_eval_decision",
        "recommendation": summary["final_recommendation"],
        "reason": "adapter_eval_not_available_without_real_training",
        "auto_promotion_allowed": False,
        "created_at": _utcnow_iso(),
    })

    (PHASE28_DIR / "phase28-runbook.md").write_text(_runbook(summary), encoding="utf-8")
    (PHASE28_DIR / "phase28-final-decision.md").write_text(_final_decision(summary), encoding="utf-8")
    (PHASE28_DIR / "next-pursuit-goal.md").write_text(_next_pursuit_goal(summary), encoding="utf-8")

    return {
        "kind": "phase28_generation_result",
        "status": "completed",
        "phase28_dir": str(PHASE28_DIR),
        "loop_state": loop_state["current_state"],
        "final_recommendation": summary["final_recommendation"],
        "created_at": _utcnow_iso(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-evidence", action="store_true", help="remove and recreate the Phase28 evidence directory")
    parser.add_argument("--feedback-jsonl", help="optional attested actual-feedback JSONL input")
    parser.add_argument("--feedback-csv", help="optional attested actual-feedback CSV input")
    parser.add_argument("--review-decisions-json", help="optional review decisions JSON input")
    args = parser.parse_args()
    print(
        json.dumps(
            generate_phase28_evidence(
                clean_evidence=args.clean_evidence,
                feedback_jsonl=args.feedback_jsonl,
                feedback_csv=args.feedback_csv,
                review_decisions_json=args.review_decisions_json,
            ),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
