#!/usr/bin/env python3
"""Generate Phase27 actual-feedback review and training-loop evidence."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import shutil
from typing import Any, Iterable, Mapping

from pfe_core.phase27_actual_feedback_review_training_loop import (
    build_phase27_collection_pack,
    build_phase27_comparison_summary,
    build_phase27_feedback_templates,
    build_phase27_import_batch,
    build_phase27_readiness,
    build_phase27_review_state,
    build_phase27_training_attempt,
)


PHASE26_DIR = Path("docs/demo/phase26-actual-feedback-collection-training-probe")
PHASE27_DIR = Path("docs/demo/phase27-actual-feedback-review-training-loop")


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
        "collection_id",
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


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return dict(data) if isinstance(data, dict) else {}


def _clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


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
            looks_quantized = "4bit" in lower or "4-bit" in lower or "awq" in lower or "gguf" in lower
            trainable_size_hint = any(token in lower for token in ("0.6b", "1.5b", "1.7b", "3b", "4b", "7b", "8b", "14b"))
            candidates.append(
                {
                    "name": path.name,
                    "path": str(path),
                    "exists": path.exists(),
                    "qwen": True,
                    "non_generative": non_generative,
                    "looks_quantized": looks_quantized,
                    "looks_oversized_for_local_training": looks_oversized,
                    "trainable": bool(trainable_size_hint and not non_generative and not looks_oversized and not lower.endswith(".gguf")),
                    "selection_note": "local inventory hint only; Phase27 training remains blocked until approved actual feedback exists",
                }
            )
    return candidates


def _reviewer_checklist() -> str:
    return """# Phase27 Reviewer Checklist

- Confirm the feedback came from a real user interaction.
- Confirm attestation fields are complete and consent is true.
- Confirm the corrected output has exactly four sections: 摘要 / 风险提示 / 引用依据 / 人工确认.
- Confirm citations use the supplied source_id:chunk_id and no external law is added.
- Exclude or quarantine PII, missing citation, external legal references, direct sign advice, or legal conclusions.
- Approve only corrections that are useful training candidates and preserve the boundary contract.
"""


def _import_instructions() -> str:
    return """# Phase27 Import Instructions

The template files are collection aids, not training data. Do not import them as
actual feedback until a real user has filled the correction/preference fields
and the attestation is complete.

Preferred import format is JSONL. CSV is supported for review handoff.

Required policy:

- `feedback_source` must be `actual_user_feedback`.
- `attestation.confirmed_actual_user_feedback` must be true.
- `attestation.not_scripted_or_curated` must be true.
- `attestation.consent_for_training_candidate_review` must be true.
- Keep source citation metadata intact.
- Do not output legal conclusions; only summarize contract material, risk notes, citations, and manual confirmation needs.
"""


def _runbook(summary: Mapping[str, Any]) -> str:
    readiness = _dict(summary.get("training_readiness"))
    return f"""# Phase27 Runbook

## Goal

Run the actual-feedback review workflow after Phase26: collect real feedback, import it, review it, then open Qwen3-4B training only when 12 approved actual candidates exist.

## Current State

- Collection tasks prepared: {summary.get("collection_count")}
- Actual feedback count: {summary.get("actual_feedback_count")}
- Accepted pending review: {summary.get("accepted_pending_review_count")}
- Approved actual candidates: {summary.get("approved_actual_candidate_count")}
- Readiness: {readiness.get("status")}
- Blockers: {", ".join(readiness.get("blockers") or [])}

## Commands

```bash
.venv/bin/python tools/phase27_actual_feedback_review_training_loop.py --clean-evidence
```

## API

- `GET /pfe/phase27/collection-pack`
- `POST /pfe/phase27/actual-feedback-batch`
- `GET /pfe/phase27/review-queue`
- `POST /pfe/phase27/review-decisions`
- `GET /pfe/phase27/training-readiness`
"""


def _final_decision(summary: Mapping[str, Any]) -> str:
    readiness = _dict(summary.get("training_readiness"))
    attempt = _dict(summary.get("training_attempt"))
    return f"""# Phase27 Final Decision

## Decision

Phase27 has built the actual-feedback review workflow. Current evidence still requires more real user feedback before training can start.

## Evidence

- Collection tasks: {summary.get("collection_count")}
- Actual feedback count: {summary.get("actual_feedback_count")}
- Approved actual candidates: {summary.get("approved_actual_candidate_count")}
- Training attempt: {attempt.get("status")}
- Readiness: {readiness.get("status")}
- Blockers: {", ".join(readiness.get("blockers") or [])}

## Recommendation

{summary.get("final_recommendation")}
"""


def generate_phase27_evidence(*, clean_evidence: bool = False) -> dict[str, Any]:
    if clean_evidence:
        _clean_dir(PHASE27_DIR)
    for subdir in ("evidence", "evidence-collection", "evidence-review", "evidence-training", "evidence-eval"):
        (PHASE27_DIR / subdir).mkdir(parents=True, exist_ok=True)

    phase26_summary = _read_json(PHASE26_DIR / "evidence" / "comparison_summary.json")
    collection_pack = build_phase27_collection_pack()
    templates = build_phase27_feedback_templates(collection_pack)
    import_batch = build_phase27_import_batch([])
    review_state = build_phase27_review_state(signals=[])
    readiness = build_phase27_readiness(signals=[], review_decisions=[], local_models=_model_inventory())
    training_attempt = build_phase27_training_attempt(readiness)
    summary = build_phase27_comparison_summary(
        phase26_summary=phase26_summary,
        collection_pack=collection_pack,
        import_batch=import_batch,
        readiness_payload=readiness,
        training_attempt=training_attempt,
    )

    evidence_dir = PHASE27_DIR / "evidence"
    collection_dir = PHASE27_DIR / "evidence-collection"
    review_dir = PHASE27_DIR / "evidence-review"
    training_dir = PHASE27_DIR / "evidence-training"
    eval_dir = PHASE27_DIR / "evidence-eval"
    candidates = _dict(readiness.get("candidate_artifacts"))

    _write_json(PHASE27_DIR / "collection_pack.json", collection_pack)
    _write_jsonl(PHASE27_DIR / "actual_feedback_template.jsonl", templates["jsonl_rows"])
    _write_csv(PHASE27_DIR / "actual_feedback_template.csv", templates["csv_rows"])
    (PHASE27_DIR / "reviewer_checklist.md").write_text(_reviewer_checklist(), encoding="utf-8")
    (PHASE27_DIR / "import_instructions.md").write_text(_import_instructions(), encoding="utf-8")

    _write_json(evidence_dir / "phase26_review.json", {
        "kind": "phase27_phase26_review",
        "phase26_collection_count": phase26_summary.get("collection_count"),
        "phase26_actual_feedback_count": phase26_summary.get("actual_feedback_count"),
        "phase26_approved_actual_candidate_count": phase26_summary.get("approved_actual_candidate_count"),
        "phase26_final_recommendation": phase26_summary.get("final_recommendation"),
        "conclusion": "Phase26 is a collection/readiness gate, not a training result.",
        "created_at": _utcnow_iso(),
    })
    _write_json(evidence_dir / "comparison_summary.json", summary)
    _write_json(evidence_dir / "api_training_readiness_payload.json", {
        "kind": "phase27_training_readiness_surface",
        "status": readiness["training_readiness"]["status"],
        "comparison_summary": summary,
        "training_readiness": readiness["training_readiness"],
        "auto_promotion_allowed": False,
    })

    _write_json(collection_dir / "collection_pack.json", collection_pack)
    _write_json(collection_dir / "feedback_templates.json", templates)
    _write_json(collection_dir / "actual_feedback_import_batch.json", import_batch)
    _write_jsonl(collection_dir / "accepted_actual_feedback_signals.jsonl", import_batch["accepted_signals"])
    _write_jsonl(collection_dir / "actual_feedback_batch.jsonl", [])
    _write_json(collection_dir / "api_collection_pack_payload.json", {
        "kind": "phase27_collection_pack_surface",
        "status": "ready",
        "collection_pack": collection_pack,
        "auto_promotion_allowed": False,
    })
    _write_json(review_dir / "review_queue.json", review_state["queue"])
    _write_json(review_dir / "review_summary.json", review_state["reviewed"])
    _write_json(review_dir / "review_decisions.json", {
        "kind": "phase27_review_decisions",
        "decision_count": 0,
        "items": [],
        "created_at": _utcnow_iso(),
    })
    _write_json(review_dir / "api_review_queue_payload.json", {
        "kind": "phase27_review_queue_surface",
        "status": "ready",
        "review_state": review_state,
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
        "kind": "phase27_adapter_eval_decision",
        "recommendation": "collect_more_actual_feedback",
        "reason": "training_not_started_without_approved_actual_feedback",
        "auto_promotion_allowed": False,
        "created_at": _utcnow_iso(),
    })
    (PHASE27_DIR / "phase27-runbook.md").write_text(_runbook(summary), encoding="utf-8")
    (PHASE27_DIR / "phase27-final-decision.md").write_text(_final_decision(summary), encoding="utf-8")

    return {
        "kind": "phase27_generation_result",
        "status": "completed",
        "phase27_dir": str(PHASE27_DIR),
        "comparison_summary": str(evidence_dir / "comparison_summary.json"),
        "final_recommendation": summary["final_recommendation"],
        "created_at": _utcnow_iso(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-evidence", action="store_true", help="remove and recreate the Phase27 evidence directory")
    args = parser.parse_args()
    print(json.dumps(generate_phase27_evidence(clean_evidence=args.clean_evidence), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
