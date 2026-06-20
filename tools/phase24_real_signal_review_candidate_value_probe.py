#!/usr/bin/env python3
"""Generate Phase24 real signal review and candidate-value probe evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any, Iterable, Mapping

from pfe_core.phase24_real_signal_review_candidate_value import (
    build_phase24_candidate_artifacts,
    build_phase24_comparison_summary,
    build_phase24_feedback_signals,
    build_phase24_holdout,
    build_phase24_interactions,
    build_phase24_model_selection,
    build_phase24_review_queue,
    build_phase24_routing_report,
    build_phase24_source_manifest,
    build_phase24_training_feasibility,
    build_phase24_training_job_specs,
    apply_phase24_review_decisions,
    evaluate_phase24_runtime_contract_holdout,
    phase24_holdout_integrity_check,
    phase24_runtime_product_decision,
    phase24_training_decision,
)


PHASE13_DIR = Path("docs/demo/phase13-boundary-contract-runtime-and-trainable-probe")
PHASE17_DIR = Path("docs/demo/phase17-qwen-dpo-product-probe")
PHASE18_DIR = Path("docs/demo/phase18-dpo-degeneration-guardrails")
PHASE23_DIR = Path("docs/demo/phase23-runtime-contract-product-loop")
PHASE24_DIR = Path("docs/demo/phase24-real-signal-review-candidate-value-probe")


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
            name = path.name
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
                    "name": name,
                    "path": str(path),
                    "exists": path.exists(),
                    "qwen": True,
                    "non_generative": non_generative,
                    "looks_quantized": looks_quant,
                    "looks_oversized_for_local_training": looks_27b,
                    "trainable": bool(trainable_size_hint and not non_generative and not looks_27b and not lower.endswith(".gguf")),
                    "selection_note": "local inventory hint only; real product probe still requires actual user feedback",
                }
            )
    return candidates


def _historical_reference() -> dict[str, Any]:
    phase18_summary = _read_json(PHASE18_DIR / "comparison_summary.json")
    phase17_summary = _read_json(PHASE17_DIR / "comparison_summary.json")
    phase13_decision = _read_json(PHASE13_DIR / "phase13-final-decision.json")
    phase23_summary = _read_json(PHASE23_DIR / "evidence" / "comparison_summary.json")
    return {
        "kind": "phase24_historical_archived_adapter_reference",
        "phase13_reference": {
            "recommendation": phase13_decision.get("recommendation") or phase13_decision.get("final_recommendation"),
        },
        "phase17_reference": {
            "final_recommendation": phase17_summary.get("final_recommendation"),
            "training_candidate_decision": _dict(phase17_summary.get("training_candidate_decision")).get("recommendation"),
        },
        "phase18_reference": {
            "final_recommendation": phase18_summary.get("final_recommendation"),
            "training_candidate_decision": _dict(phase18_summary.get("training_candidate_decision")).get("recommendation"),
        },
        "phase23_reference": {
            "runtime_decision": _dict(phase23_summary.get("runtime_contract_decision")).get("recommendation"),
            "training_candidate_decision": _dict(phase23_summary.get("training_candidate_decision")).get("recommendation"),
        },
        "note": "Historical adapters are archived references, not promoted product paths.",
        "created_at": _utcnow_iso(),
    }


def _review_previous_phase23() -> dict[str, Any]:
    summary = _read_json(PHASE23_DIR / "evidence" / "comparison_summary.json")
    return {
        "kind": "phase24_phase23_review",
        "phase23_runtime_decision": _dict(summary.get("runtime_contract_decision")).get("recommendation"),
        "phase23_runtime_scores": _dict(_dict(summary.get("runtime_contract_eval")).get("scores")),
        "phase23_training_candidate_decision": _dict(summary.get("training_candidate_decision")).get("recommendation"),
        "conclusions": [
            "Phase23 runtime contract is stable and remains primary product path.",
            "Phase23 training candidates are guarded and archived unless they beat runtime contract.",
            "Phase24 should test signal review quality before claiming adapter product value.",
        ],
        "created_at": _utcnow_iso(),
    }


def _sample_routing_examples(routing_report: Mapping[str, Any]) -> str:
    lines = [
        "# Phase24 Sample Routing Examples",
        "",
        "| signal_id | lanes | eligible | excluded_reason | rule_hits |",
        "| --- | --- | --- | --- | --- |",
    ]
    for item in list(routing_report.get("routed_signals") or [])[:16]:
        if not isinstance(item, Mapping):
            continue
        route = _dict(item.get("phase24_route"))
        lines.append(
            "| {signal_id} | {lanes} | {eligible} | {excluded} | {hits} |".format(
                signal_id=item.get("signal_id"),
                lanes=", ".join(route.get("lanes") or []),
                eligible=route.get("eligible_for_training"),
                excluded=route.get("excluded_reason") or "",
                hits=", ".join(route.get("rule_hits") or []),
            )
        )
    return "\n".join(lines) + "\n"


def _runbook(summary: Mapping[str, Any]) -> str:
    runtime = _dict(_dict(summary.get("runtime_contract_eval")).get("scores"))
    return f"""# Phase24 Runbook

## Goal

Validate a longer PFE product loop: runtime contract interactions -> explicit feedback provenance -> review queue -> strict routing -> candidate sample specs -> training value decision.

## Commands

```bash
.venv/bin/python tools/phase24_real_signal_review_candidate_value_probe.py --clean-evidence
.venv/bin/python -m pytest tests/test_phase24_real_signal_review_candidate_value.py tests/test_phase24_real_signal_review_surface.py tests/test_phase23_runtime_contract_product_loop.py tests/test_phase23_runtime_contract_loop_surface.py -q
make test-unit test-surface test-e2e-mock smoke-beta
```

## Current Result

- Runtime interactions: {summary.get("interaction_count")}
- Feedback signals: {summary.get("feedback_signal_count")}
- Runtime holdout: {_dict(summary.get("runtime_contract_eval")).get("holdout_count")}
- Runtime scores: {json.dumps(runtime, ensure_ascii=False, sort_keys=True)}
- Final recommendation: {summary.get("final_recommendation")}

## Important Boundary

Phase24 generated real PFE runtime-contract outputs, but feedback is labelled as curated/scripted lab review. It is not represented as actual user feedback, so real product-value adapter training is blocked and archived.
"""


def _final_decision(summary: Mapping[str, Any]) -> str:
    training = _dict(summary.get("training_candidate_decision"))
    feasibility = _dict(summary.get("training_feasibility"))
    runtime = _dict(_dict(summary.get("runtime_contract_eval")).get("scores"))
    return f"""# Phase24 Final Decision

## Decision

Runtime contract remains the primary product path. Training candidates are archived for product-value claims in this phase.

## Evidence

- Runtime holdout scores: {json.dumps(runtime, ensure_ascii=False, sort_keys=True)}
- Candidate recommendation: {training.get("recommendation")}
- Training blockers: {", ".join(feasibility.get("blockers") or [])}
- Auto promotion allowed: false

## Interpretation

Phase24 proves the product loop can collect runtime interactions, label feedback provenance, review signals, route exclusions, generate SFT/DPO candidate specs, and block unsafe or under-evidenced training. It does not prove adapter product lift because there is no actual user feedback approved for product-value training.
"""


def generate_phase24_evidence(*, clean_evidence: bool = False) -> dict[str, Any]:
    if clean_evidence:
        _clean_dir(PHASE24_DIR)
    for subdir in (
        "evidence",
        "evidence-interactions",
        "evidence-review",
        "evidence-routing",
        "evidence-candidates",
        "evidence-training",
        "evidence-eval",
    ):
        (PHASE24_DIR / subdir).mkdir(parents=True, exist_ok=True)

    review_previous = _review_previous_phase23()
    capture = build_phase24_interactions(count=80)
    feedback = build_phase24_feedback_signals(capture)
    queue = build_phase24_review_queue(feedback["signals"])
    reviewed = apply_phase24_review_decisions(queue, feedback["signals"])
    routing = build_phase24_routing_report(reviewed, feedback["signals"])
    holdout = build_phase24_holdout(regression_count=50, hard_count=50)
    hard_holdout = {
        "kind": "phase24_hard_holdout_export",
        "holdout_count": 50,
        "prompts": [item for item in holdout["prompts"] if item.get("phase24_holdout_group") == "phase24_hard"],
        "created_at": _utcnow_iso(),
    }
    holdout_chunk_ids = {
        str(item.get("chunk_id"))
        for item in holdout.get("prompts") or []
        if isinstance(item, Mapping) and item.get("chunk_id")
    }
    candidate_artifacts = build_phase24_candidate_artifacts(
        signals=feedback["signals"],
        reviewed=reviewed,
        routing_report=routing,
        holdout_chunk_ids=holdout_chunk_ids,
    )
    integrity = phase24_holdout_integrity_check(
        holdout=holdout,
        sft_samples=candidate_artifacts["sft_samples"],
        dpo_pairs=candidate_artifacts["dpo_pairs"],
    )
    source_manifest = build_phase24_source_manifest(capture["interactions"], holdout)
    model_selection = build_phase24_model_selection(local_models=_model_inventory())
    feasibility = build_phase24_training_feasibility(
        candidate_manifest=candidate_artifacts["candidate_manifest"],
        model_selection=model_selection,
        actual_user_feedback_count=int(feedback["actual_user_feedback_count"]),
    )
    job_specs = build_phase24_training_job_specs(
        candidate_manifest=candidate_artifacts["candidate_manifest"],
        model_selection=model_selection,
        feasibility=feasibility,
    )
    runtime_eval = evaluate_phase24_runtime_contract_holdout(holdout)
    runtime_decision_payload = phase24_runtime_product_decision(runtime_eval)
    training_decision_payload = phase24_training_decision(
        runtime_scores=runtime_eval["scores"],
        sft_scores=None,
        dpo_scores=None,
        feasibility=feasibility,
        candidate_manifest=candidate_artifacts["candidate_manifest"],
    )
    historical = _historical_reference()
    summary = build_phase24_comparison_summary(
        interaction_capture=capture,
        feedback_capture=feedback,
        review_summary=reviewed,
        routing_report=routing,
        candidate_manifest=candidate_artifacts["candidate_manifest"],
        candidate_quality_report=candidate_artifacts["quality_report"],
        holdout_integrity=integrity,
        runtime_eval=runtime_eval,
        runtime_decision_payload=runtime_decision_payload,
        model_selection=model_selection,
        training_feasibility=feasibility,
        training_decision_payload=training_decision_payload,
        historical_reference=historical,
    )

    evidence_dir = PHASE24_DIR / "evidence"
    interactions_dir = PHASE24_DIR / "evidence-interactions"
    review_dir = PHASE24_DIR / "evidence-review"
    routing_dir = PHASE24_DIR / "evidence-routing"
    candidates_dir = PHASE24_DIR / "evidence-candidates"
    training_dir = PHASE24_DIR / "evidence-training"
    eval_dir = PHASE24_DIR / "evidence-eval"

    _write_json(evidence_dir / "phase23_review.json", review_previous)
    _write_json(evidence_dir / "comparison_summary.json", summary)
    _write_json(evidence_dir / "api_training_candidate_value_payload.json", {
        "kind": "phase24_training_candidate_value",
        "status": "ready",
        "comparison_summary": summary,
        "workspace_phase": "phase24",
        "auto_promotion_allowed": False,
    })
    _write_json(interactions_dir / "interaction_manifest.json", {
        "kind": "phase24_interaction_manifest",
        "interaction_count": capture["interaction_count"],
        "runtime_output_count": capture["runtime_output_count"],
        "real_runtime_contract_calls": True,
        "feedback_is_actual_user_feedback": False,
        "created_at": _utcnow_iso(),
    })
    _write_json(interactions_dir / "source_manifest.json", source_manifest)
    _write_jsonl(interactions_dir / "interactions.jsonl", capture["interactions"])
    _write_jsonl(interactions_dir / "runtime_outputs.jsonl", capture["runtime_outputs"])
    _write_json(review_dir / "review_manifest.json", {
        "kind": "phase24_review_manifest",
        "signal_count": feedback["signal_count"],
        "feedback_type_counts": feedback["feedback_type_counts"],
        "feedback_source_counts": feedback["feedback_source_counts"],
        "actual_user_feedback_count": feedback["actual_user_feedback_count"],
        "created_at": _utcnow_iso(),
    })
    _write_json(review_dir / "review_queue.json", queue)
    _write_json(review_dir / "review_summary.json", reviewed)
    _write_json(review_dir / "api_review_queue_payload.json", {
        "kind": "phase24_review_queue_surface",
        "status": "ready",
        "queue": queue,
        "review_summary": reviewed,
        "auto_promotion_allowed": False,
    })
    _write_jsonl(review_dir / "feedback_signals.jsonl", feedback["signals"])
    _write_jsonl(review_dir / "review_log.jsonl", feedback["review_log"])
    _write_jsonl(review_dir / "reviewed_signals.jsonl", reviewed["items"])
    _write_json(routing_dir / "routing_report.json", routing)
    _write_json(routing_dir / "excluded_reasons.json", {
        "kind": "phase24_excluded_reasons",
        "excluded_reason_counts": routing["excluded_reason_counts"],
        "created_at": _utcnow_iso(),
    })
    approved_ids = {
        str(item.get("signal_id"))
        for item in routing.get("routed_signals") or []
        if _dict(item.get("phase24_route")).get("eligible_for_training")
    }
    _write_jsonl(routing_dir / "approved_candidate_signals.jsonl", [
        signal for signal in feedback["signals"] if str(signal.get("signal_id")) in approved_ids
    ])
    (routing_dir / "sample_routing_examples.md").write_text(_sample_routing_examples(routing), encoding="utf-8")
    _write_jsonl(candidates_dir / "candidate_sft_samples.jsonl", candidate_artifacts["sft_samples"])
    _write_jsonl(candidates_dir / "candidate_dpo_pairs.jsonl", candidate_artifacts["dpo_pairs"])
    _write_json(candidates_dir / "candidate_quality_report.json", candidate_artifacts["quality_report"])
    _write_json(candidates_dir / "candidate_manifest.json", candidate_artifacts["candidate_manifest"])
    _write_json(eval_dir / "holdout.json", holdout)
    _write_json(eval_dir / "hard_holdout.json", hard_holdout)
    _write_json(eval_dir / "holdout_integrity_check.json", integrity)
    _write_json(eval_dir / "source_manifest.json", source_manifest)
    _write_json(eval_dir / "runtime_contract_eval_report.json", runtime_eval)
    _write_json(eval_dir / "runtime_contract_decision.json", runtime_decision_payload)
    _write_json(training_dir / "model_selection.json", model_selection)
    _write_json(training_dir / "training_feasibility.json", feasibility)
    _write_json(training_dir / "training_job_specs.json", job_specs)
    _write_json(training_dir / "training_attempt.json", {
        "kind": "phase24_training_attempt",
        "status": "blocked",
        "method": "sft_or_dpo",
        "reason": ";".join(feasibility.get("blockers") or []),
        "adapter_artifact_created": False,
        "auto_promotion_allowed": False,
        "created_at": _utcnow_iso(),
    })
    _write_json(training_dir / "train_log.json", {
        "kind": "phase24_train_log",
        "status": "not_started",
        "events": [{"level": "info", "message": "training blocked before execution", "created_at": _utcnow_iso()}],
    })
    _write_json(training_dir / "adapter_validation.json", {
        "kind": "phase24_adapter_validation",
        "status": "not_available",
        "adapter_artifact_created": False,
    })
    _write_json(training_dir / "blocked_reason.json", {
        "kind": "phase24_training_blocked_reason",
        "blockers": feasibility.get("blockers") or [],
        "reason": ";".join(feasibility.get("blockers") or []),
        "created_at": _utcnow_iso(),
    })
    _write_json(training_dir / "training_candidate_decision.json", training_decision_payload)
    (PHASE24_DIR / "phase24-runbook.md").write_text(_runbook(summary), encoding="utf-8")
    (PHASE24_DIR / "phase24-final-decision.md").write_text(_final_decision(summary), encoding="utf-8")
    return {
        "kind": "phase24_generation_result",
        "status": "completed",
        "phase24_dir": str(PHASE24_DIR),
        "comparison_summary": str(evidence_dir / "comparison_summary.json"),
        "runtime_decision": runtime_decision_payload,
        "training_decision": training_decision_payload,
        "created_at": _utcnow_iso(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-evidence", action="store_true", help="remove and recreate the Phase24 evidence directory")
    args = parser.parse_args()
    result = generate_phase24_evidence(clean_evidence=args.clean_evidence)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
