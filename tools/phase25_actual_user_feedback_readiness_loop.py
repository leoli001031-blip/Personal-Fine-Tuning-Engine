#!/usr/bin/env python3
"""Generate Phase25 actual-user feedback readiness evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any, Iterable, Mapping

from pfe_core.phase25_actual_user_feedback_loop import (
    build_phase25_comparison_summary,
    build_phase25_empty_readiness,
)


PHASE24_DIR = Path("docs/demo/phase24-real-signal-review-candidate-value-probe")
PHASE25_DIR = Path("docs/demo/phase25-actual-user-feedback-readiness-loop")


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
            looks_27b = "27b" in lower or "32b" in lower or "72b" in lower
            looks_quant = "4bit" in lower or "4-bit" in lower or "awq" in lower or "gguf" in lower
            trainable_size_hint = any(token in lower for token in ("0.6b", "1.5b", "1.7b", "3b", "4b", "7b", "8b", "14b"))
            candidates.append(
                {
                    "name": path.name,
                    "path": str(path),
                    "exists": path.exists(),
                    "qwen": True,
                    "non_generative": non_generative,
                    "looks_quantized": looks_quant,
                    "looks_oversized_for_local_training": looks_27b,
                    "trainable": bool(trainable_size_hint and not non_generative and not looks_27b and not lower.endswith(".gguf")),
                    "selection_note": "local inventory hint only; Phase25 still requires actual user feedback",
                }
            )
    return candidates


def _runbook(summary: Mapping[str, Any]) -> str:
    readiness = _dict(summary.get("training_readiness"))
    runtime = _dict(_dict(summary.get("runtime_contract_eval")).get("scores"))
    return f"""# Phase25 Runbook

## Goal

Collect attested actual user feedback before any product-value SFT/DPO training probe.

## Current Evidence

- Actual feedback count: {summary.get("actual_feedback_count")}
- Approved actual candidates: {summary.get("approved_actual_candidate_count")}
- Readiness status: {readiness.get("status")}
- Blockers: {", ".join(readiness.get("blockers") or [])}
- Runtime holdout scores: {json.dumps(runtime, ensure_ascii=False, sort_keys=True)}

## Generate Evidence

```bash
.venv/bin/python tools/phase25_actual_user_feedback_readiness_loop.py --clean-evidence
```

## Collect One Actual Feedback Signal

Use `/pfe/phase25/actual-feedback` with `feedback_source=actual_user_feedback` and an attestation that confirms the feedback is not scripted or curated. The signal remains pending review until explicitly approved.
"""


def _final_decision(summary: Mapping[str, Any]) -> str:
    readiness = _dict(summary.get("training_readiness"))
    return f"""# Phase25 Final Decision

## Decision

Phase25 is ready to collect actual user feedback, but real training remains blocked.

## Reason

Training readiness requires at least {readiness.get("minimum_approved_actual_candidates")} approved actual-user candidates. Current approved actual candidates: {readiness.get("approved_actual_candidate_count")}.

## Recommendation

Collect real user corrections through the Phase25 intake endpoint, review them manually, then rerun the readiness gate before starting Qwen3-4B SFT/DPO training.
"""


def generate_phase25_evidence(*, clean_evidence: bool = False) -> dict[str, Any]:
    if clean_evidence:
        _clean_dir(PHASE25_DIR)
    for subdir in ("evidence", "evidence-feedback", "evidence-review", "evidence-training", "evidence-eval"):
        (PHASE25_DIR / subdir).mkdir(parents=True, exist_ok=True)

    phase24_summary = _read_json(PHASE24_DIR / "evidence" / "comparison_summary.json")
    readiness_payload = build_phase25_empty_readiness(local_models=_model_inventory())
    summary = build_phase25_comparison_summary(readiness_payload)

    evidence_dir = PHASE25_DIR / "evidence"
    feedback_dir = PHASE25_DIR / "evidence-feedback"
    review_dir = PHASE25_DIR / "evidence-review"
    training_dir = PHASE25_DIR / "evidence-training"
    eval_dir = PHASE25_DIR / "evidence-eval"

    _write_json(evidence_dir / "phase24_review.json", {
        "kind": "phase25_phase24_review",
        "phase24_final_recommendation": phase24_summary.get("final_recommendation"),
        "phase24_training_blockers": _dict(phase24_summary.get("training_feasibility")).get("blockers"),
        "conclusion": "Phase25 must collect attested actual user feedback before product-value training.",
        "created_at": _utcnow_iso(),
    })
    _write_json(evidence_dir / "comparison_summary.json", summary)
    _write_json(evidence_dir / "api_actual_feedback_readiness_payload.json", {
        "kind": "phase25_actual_feedback_readiness",
        "status": "ready",
        "comparison_summary": summary,
        "attestation_template": readiness_payload["attestation_template"],
        "auto_promotion_allowed": False,
    })
    _write_json(feedback_dir / "attestation_template.json", readiness_payload["attestation_template"])
    _write_json(feedback_dir / "actual_feedback_manifest.json", {
        "kind": "phase25_actual_feedback_manifest",
        "actual_feedback_count": 0,
        "accepted_pending_review_count": 0,
        "approved_for_candidate_count": 0,
        "source_policy": "no scripted or curated feedback may be counted as actual user feedback",
        "created_at": _utcnow_iso(),
    })
    _write_jsonl(feedback_dir / "actual_feedback_inbox.jsonl", [])
    _write_jsonl(feedback_dir / "accepted_actual_feedback_signals.jsonl", [])
    _write_json(review_dir / "review_queue.json", readiness_payload["queue"])
    _write_json(review_dir / "review_summary.json", readiness_payload["reviewed"])
    _write_json(review_dir / "routing_report.json", readiness_payload["routing_report"])
    _write_json(training_dir / "training_readiness_report.json", readiness_payload["training_readiness"])
    _write_json(training_dir / "training_job_specs.json", readiness_payload["training_job_specs"])
    _write_json(training_dir / "blocked_reason.json", {
        "kind": "phase25_training_blocked_reason",
        "blockers": readiness_payload["training_readiness"]["blockers"],
        "created_at": _utcnow_iso(),
    })
    _write_json(training_dir / "model_selection.json", readiness_payload["model_selection"])
    _write_json(eval_dir / "runtime_contract_eval_report.json", readiness_payload["runtime_eval"])
    _write_json(eval_dir / "runtime_contract_decision.json", readiness_payload["runtime_decision"])
    _write_json(eval_dir / "holdout_integrity_check.json", readiness_payload["holdout_integrity_check"])
    (PHASE25_DIR / "phase25-runbook.md").write_text(_runbook(summary), encoding="utf-8")
    (PHASE25_DIR / "phase25-final-decision.md").write_text(_final_decision(summary), encoding="utf-8")
    return {
        "kind": "phase25_generation_result",
        "status": "completed",
        "phase25_dir": str(PHASE25_DIR),
        "comparison_summary": str(evidence_dir / "comparison_summary.json"),
        "final_recommendation": summary["final_recommendation"],
        "created_at": _utcnow_iso(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-evidence", action="store_true", help="remove and recreate the Phase25 evidence directory")
    args = parser.parse_args()
    print(json.dumps(generate_phase25_evidence(clean_evidence=args.clean_evidence), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
