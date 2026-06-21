#!/usr/bin/env python3
"""Generate Phase26 actual-feedback collection and training-probe evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any, Iterable, Mapping

from pfe_core.phase26_actual_feedback_collection_probe import (
    build_phase26_comparison_summary,
    build_phase26_empty_state,
)


PHASE25_DIR = Path("docs/demo/phase25-actual-user-feedback-readiness-loop")
PHASE26_DIR = Path("docs/demo/phase26-actual-feedback-collection-training-probe")


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
                    "selection_note": "local inventory hint only; Phase26 still requires approved actual user feedback",
                }
            )
    return candidates


def _runbook(summary: Mapping[str, Any]) -> str:
    readiness = _dict(summary.get("training_readiness"))
    return f"""# Phase26 Runbook

## Goal

Execute the next product step after Phase25: prepare a real feedback collection pack, accept attested actual-feedback batches, manually approve candidates, then run a Qwen3-4B SFT/DPO probe only when the gate is ready.

## Current State

- Collection tasks prepared: {summary.get("collection_count")}
- Actual feedback count: {summary.get("actual_feedback_count")}
- Approved actual candidates: {summary.get("approved_actual_candidate_count")}
- Readiness: {readiness.get("status")}
- Blockers: {", ".join(readiness.get("blockers") or [])}

## Commands

```bash
.venv/bin/python tools/phase26_actual_feedback_collection_training_probe.py --clean-evidence
```

## API

- `GET /pfe/phase26/feedback-collection-pack`
- `POST /pfe/phase26/actual-feedback-batch`
- `GET /pfe/phase26/training-probe-readiness`
"""


def _final_decision(summary: Mapping[str, Any]) -> str:
    readiness = _dict(summary.get("training_readiness"))
    return f"""# Phase26 Final Decision

## Decision

Phase26 has prepared the actual-feedback collection and Qwen3-4B training-probe path. Training remains blocked until real, attested feedback is collected and manually approved.

## Evidence

- Collection tasks: {summary.get("collection_count")}
- Approved actual candidates: {summary.get("approved_actual_candidate_count")}
- Readiness: {readiness.get("status")}
- Blockers: {", ".join(readiness.get("blockers") or [])}

## Next Action

Use the collection pack to gather 12 real user corrections, submit them through the batch endpoint, approve them after review, then rerun readiness before launching the training probe.
"""


def generate_phase26_evidence(*, clean_evidence: bool = False) -> dict[str, Any]:
    if clean_evidence:
        _clean_dir(PHASE26_DIR)
    for subdir in ("evidence", "evidence-feedback", "evidence-review", "evidence-training", "evidence-eval"):
        (PHASE26_DIR / subdir).mkdir(parents=True, exist_ok=True)

    phase25_summary = _read_json(PHASE25_DIR / "evidence" / "comparison_summary.json")
    state = build_phase26_empty_state(local_models=_model_inventory())
    summary = build_phase26_comparison_summary(state)
    collection_pack = _dict(state.get("collection_pack"))
    probe = _dict(state.get("probe_readiness"))

    evidence_dir = PHASE26_DIR / "evidence"
    feedback_dir = PHASE26_DIR / "evidence-feedback"
    review_dir = PHASE26_DIR / "evidence-review"
    training_dir = PHASE26_DIR / "evidence-training"
    eval_dir = PHASE26_DIR / "evidence-eval"

    _write_json(evidence_dir / "phase25_review.json", {
        "kind": "phase26_phase25_review",
        "phase25_final_recommendation": phase25_summary.get("final_recommendation"),
        "phase25_readiness": _dict(phase25_summary.get("training_readiness")).get("status"),
        "conclusion": "Phase26 turns ready-to-collect into a concrete feedback collection pack and batch intake path.",
        "created_at": _utcnow_iso(),
    })
    _write_json(evidence_dir / "comparison_summary.json", summary)
    _write_json(evidence_dir / "api_training_probe_readiness_payload.json", {
        "kind": "phase26_training_probe_readiness_surface",
        "status": "ready",
        "comparison_summary": summary,
        "training_readiness": probe.get("training_readiness"),
        "auto_promotion_allowed": False,
    })
    _write_json(feedback_dir / "collection_pack.json", collection_pack)
    _write_json(feedback_dir / "api_collection_pack_payload.json", {
        "kind": "phase26_feedback_collection_pack_surface",
        "status": "ready",
        "collection_pack": collection_pack,
        "auto_promotion_allowed": False,
    })
    _write_json(feedback_dir / "feedback_batch_manifest.json", {
        "kind": "phase26_feedback_batch_manifest",
        "payload_count": 0,
        "accepted_pending_review_count": 0,
        "blocked_count": 0,
        "source_policy": "only attested actual user feedback is accepted",
        "created_at": _utcnow_iso(),
    })
    _write_jsonl(feedback_dir / "actual_feedback_batch.jsonl", [])
    _write_jsonl(feedback_dir / "accepted_actual_feedback_signals.jsonl", [])
    _write_json(review_dir / "review_queue.json", probe.get("queue") or {})
    _write_json(review_dir / "review_summary.json", probe.get("reviewed") or {})
    _write_json(review_dir / "routing_report.json", probe.get("routing_report") or {})
    _write_json(training_dir / "training_readiness_report.json", probe.get("training_readiness") or {})
    _write_json(training_dir / "training_job_specs.json", probe.get("training_job_specs") or {})
    _write_json(training_dir / "training_attempt.json", {
        "kind": "phase26_training_attempt",
        "status": "blocked",
        "reason": ";".join(_dict(probe.get("training_readiness")).get("blockers") or []),
        "adapter_artifact_created": False,
        "auto_promotion_allowed": False,
        "created_at": _utcnow_iso(),
    })
    _write_json(training_dir / "model_selection.json", probe.get("model_selection") or {})
    _write_json(eval_dir / "runtime_contract_eval_report.json", probe.get("runtime_eval") or {})
    _write_json(eval_dir / "runtime_contract_decision.json", probe.get("runtime_decision") or {})
    _write_json(eval_dir / "holdout_integrity_check.json", probe.get("holdout_integrity_check") or {})
    (PHASE26_DIR / "phase26-runbook.md").write_text(_runbook(summary), encoding="utf-8")
    (PHASE26_DIR / "phase26-final-decision.md").write_text(_final_decision(summary), encoding="utf-8")
    return {
        "kind": "phase26_generation_result",
        "status": "completed",
        "phase26_dir": str(PHASE26_DIR),
        "comparison_summary": str(evidence_dir / "comparison_summary.json"),
        "final_recommendation": summary["final_recommendation"],
        "created_at": _utcnow_iso(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-evidence", action="store_true", help="remove and recreate the Phase26 evidence directory")
    args = parser.parse_args()
    print(json.dumps(generate_phase26_evidence(clean_evidence=args.clean_evidence), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
