#!/usr/bin/env python3
"""Generate a Phase27 actual-feedback workflow simulation package.

This dry run exercises the happy path and guardrail branches without creating
real training evidence. The simulated positive feedback is intentionally marked
as simulation-only in every saved artifact.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any, Iterable, Mapping

from pfe_core.phase27_actual_feedback_review_training_loop import (
    append_phase27_import_batch,
    apply_phase27_review_decision,
    build_phase27_collection_pack,
    build_phase27_feedback_templates,
    build_phase27_import_batch,
    build_phase27_readiness,
    build_phase27_review_state,
    build_phase27_training_attempt,
    load_phase27_state,
    phase27_store_path,
)


SIM_DIR = Path("docs/demo/phase27-actual-feedback-review-training-loop/simulation")
SIM_WORKSPACE = "phase27-simulation"


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def _clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _simulated_feedback_payload(item: Mapping[str, Any], index: int) -> dict[str, Any]:
    metadata = _dict(item.get("metadata"))
    return {
        "collection_id": item.get("collection_id"),
        "prompt": item.get("prompt"),
        "messages": item.get("messages") or [],
        "runtime_output": item.get("runtime_output"),
        "response_under_review": item.get("runtime_output"),
        "metadata": {
            **metadata,
            "simulation_only": True,
            "simulation_policy": "counterfactual happy-path replay; not real user feedback",
        },
        "feedback_source": "actual_user_feedback",
        "feedback": {
            "action": "correction",
            "edited_text": item.get("suggested_target_template") or "",
            "user_feedback": "SIMULATION ONLY: reviewer would prefer the corrected four-section boundary output.",
            "signal_id": f"phase27-sim-signal-{index:03d}",
        },
        "attestation": {
            "operator_id": "simulation-reviewer",
            "capture_method": "phase27_workflow_simulation",
            "captured_at": "2026-06-21T10:00:00+08:00",
            "confirmed_actual_user_feedback": True,
            "not_scripted_or_curated": True,
            "consent_for_training_candidate_review": True,
            "simulation_only": True,
        },
        "request_id": f"phase27-sim-request-{index:03d}",
        "session_id": "phase27-simulation-session",
        "simulation_only": True,
        "not_valid_for_production_training": True,
    }


def _guardrail_payloads(collection_pack: Mapping[str, Any]) -> list[dict[str, Any]]:
    item = dict((collection_pack.get("items") or [{}])[0])
    templates = build_phase27_feedback_templates(collection_pack)
    template_payload = dict((templates.get("jsonl_rows") or [{}])[0])

    missing_consent = _simulated_feedback_payload(item, 901)
    missing_consent["feedback"]["signal_id"] = "phase27-sim-guard-missing-consent"
    missing_consent["attestation"]["consent_for_training_candidate_review"] = False

    pii_missing_citation = _simulated_feedback_payload(item, 902)
    pii_missing_citation["feedback"]["signal_id"] = "phase27-sim-guard-pii"
    pii_missing_citation["metadata"]["expected_citation"] = ""
    pii_missing_citation["feedback"]["edited_text"] += "\nContact reviewer@example.com"

    return [template_payload, missing_consent, pii_missing_citation]


def _simulated_model_inventory() -> list[dict[str, Any]]:
    return [
        {
            "name": "Qwen3-4B workflow simulation",
            "path": "/simulation/qwen3-4b",
            "exists": False,
            "trainable": True,
            "simulation_only": True,
            "selection_note": "simulated model inventory to show the readiness gate after 12 approved signals",
        }
    ]


def _notes(summary: Mapping[str, Any]) -> str:
    return f"""# Phase27 Workflow Simulation

This package is a dry run. It is not actual user-feedback evidence and must not
be used for product-value training claims.

## What Was Simulated

1. Exported the Phase27 collection pack.
2. Created 12 simulation-only feedback payloads.
3. Imported them through the Phase27 intake validator.
4. Persisted review decisions into a sandbox store.
5. Approved the 12 simulation signals for candidate generation.
6. Generated SFT/DPO candidate artifacts.
7. Opened the readiness gate with a simulated Qwen3-4B inventory.
8. Stopped at `ready_to_launch`; no real training or adapter eval was run.

## Guardrail Replay

The simulation also checks three negative branches:

- template feedback -> non_training
- missing consent -> blocked
- PII plus missing citation -> quarantined

## Summary

- accepted_pending_review_count: {summary.get("accepted_pending_review_count")}
- approved_for_candidate_count: {summary.get("approved_for_candidate_count")}
- sft_sample_count: {summary.get("sft_sample_count")}
- dpo_pair_count: {summary.get("dpo_pair_count")}
- readiness_status: {summary.get("readiness_status")}
- training_attempt_status: {summary.get("training_attempt_status")}

## Important Boundary

This proves the workflow shape, not product value. The real next step is still
collecting attested human feedback and rerunning the same path without
simulation markers.
"""


def generate_simulation(*, clean: bool = False) -> dict[str, Any]:
    if clean:
        _clean_dir(SIM_DIR)
    else:
        SIM_DIR.mkdir(parents=True, exist_ok=True)

    collection_pack = build_phase27_collection_pack()
    feedback_payloads = [
        _simulated_feedback_payload(item, index)
        for index, item in enumerate(collection_pack.get("items") or [], start=1)
        if isinstance(item, Mapping)
    ]
    import_batch = build_phase27_import_batch(feedback_payloads)

    sandbox_root = SIM_DIR / "sandbox_store"
    store = phase27_store_path(sandbox_root, SIM_WORKSPACE)
    if store.exists():
        store.unlink()
    append_phase27_import_batch(store, import_batch)

    review_decisions = [
        {
            "signal_id": signal.get("signal_id"),
            "state": "approved_for_candidate",
            "reason": "simulation replay: corrected output preserves four sections, citations, and manual confirmation boundary",
            "reviewer_id": "simulation-reviewer",
        }
        for signal in import_batch.get("accepted_signals") or []
        if isinstance(signal, Mapping)
    ]
    decision_results = [apply_phase27_review_decision(store, decision) for decision in review_decisions]
    state = load_phase27_state(store)
    review_state = build_phase27_review_state(
        signals=[dict(item) for item in state.get("signals") or [] if isinstance(item, Mapping)],
        review_decisions=[dict(item) for item in state.get("review_decisions") or [] if isinstance(item, Mapping)],
    )
    readiness = build_phase27_readiness(
        signals=[dict(item) for item in state.get("signals") or [] if isinstance(item, Mapping)],
        review_decisions=[dict(item) for item in state.get("review_decisions") or [] if isinstance(item, Mapping)],
        local_models=_simulated_model_inventory(),
    )
    training_attempt = build_phase27_training_attempt(readiness)

    guardrail_batch = build_phase27_import_batch(_guardrail_payloads(collection_pack))
    candidates = _dict(readiness.get("candidate_artifacts"))
    manifest = _dict(candidates.get("candidate_manifest"))
    training_readiness = _dict(readiness.get("training_readiness"))
    summary = {
        "kind": "phase27_workflow_simulation_summary",
        "status": "completed",
        "simulation_only": True,
        "not_valid_for_production_training": True,
        "collection_count": collection_pack.get("collection_count", 0),
        "payload_count": len(feedback_payloads),
        "accepted_pending_review_count": import_batch.get("accepted_pending_review_count", 0),
        "approved_for_candidate_count": review_state.get("approved_for_candidate_count", 0),
        "sft_sample_count": manifest.get("sft_sample_count", 0),
        "dpo_pair_count": manifest.get("dpo_pair_count", 0),
        "holdout_integrity_passed": _dict(readiness.get("holdout_integrity_check")).get("passed"),
        "readiness_status": training_readiness.get("status"),
        "readiness_blockers": training_readiness.get("blockers") or [],
        "training_attempt_status": training_attempt.get("status"),
        "training_attempt_reason": training_attempt.get("reason"),
        "adapter_artifact_created": training_attempt.get("adapter_artifact_created", False),
        "auto_promotion_allowed": False,
        "guardrail_counts": {
            "blocked_count": guardrail_batch.get("blocked_count", 0),
            "non_training_count": guardrail_batch.get("non_training_count", 0),
            "quarantined_count": guardrail_batch.get("quarantined_count", 0),
        },
        "created_at": _utcnow_iso(),
    }

    _write_json(SIM_DIR / "collection_pack_snapshot.json", collection_pack)
    _write_jsonl(SIM_DIR / "simulated_feedback_batch.jsonl", feedback_payloads)
    _write_json(SIM_DIR / "simulated_import_batch.json", import_batch)
    compact_decision_results = [
        {
            "kind": result.get("kind"),
            "status": result.get("status"),
            "applied": result.get("applied") or [],
            "auto_promotion_allowed": result.get("auto_promotion_allowed", False),
            "created_at": result.get("created_at"),
        }
        for result in decision_results
        if isinstance(result, Mapping)
    ]
    _write_json(SIM_DIR / "simulated_review_decisions.json", {"items": review_decisions, "results": compact_decision_results})
    _write_json(SIM_DIR / "simulated_state.json", state)
    _write_json(SIM_DIR / "simulated_review_state.json", review_state)
    _write_json(SIM_DIR / "simulated_training_readiness_payload.json", readiness)
    _write_json(SIM_DIR / "simulated_training_attempt.json", training_attempt)
    _write_json(SIM_DIR / "simulated_candidate_manifest.json", manifest)
    _write_json(SIM_DIR / "simulated_candidate_quality_report.json", candidates.get("quality_report") or {})
    _write_jsonl(SIM_DIR / "simulated_sft_candidates.jsonl", candidates.get("sft_samples") or [])
    _write_jsonl(SIM_DIR / "simulated_dpo_pairs.jsonl", candidates.get("dpo_pairs") or [])
    _write_json(SIM_DIR / "guardrail_replay_batch.json", guardrail_batch)
    _write_json(SIM_DIR / "simulation_summary.json", summary)
    (SIM_DIR / "simulation_notes.md").write_text(_notes(summary), encoding="utf-8")
    return {
        "kind": "phase27_workflow_simulation_result",
        "status": "completed",
        "simulation_dir": str(SIM_DIR),
        "summary": summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", action="store_true", help="remove and recreate the simulation evidence directory")
    args = parser.parse_args()
    print(json.dumps(generate_simulation(clean=args.clean), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
