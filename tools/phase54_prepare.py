#!/usr/bin/env python3
"""Prepare and freeze Phase54 typed-evaluator evidence before model calls."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = REPO_ROOT / "pfe-core"
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

from pfe_core.phase46_runtime_first_latest_intent import stable_hash
from pfe_core.phase54_typed_proposition_evaluator import (
    PHASE54_CALIBRATION_ACCURACY_GATE,
    PHASE54_CATEGORIES,
    PHASE54_EXTRACTION_RUBRIC,
    PHASE54_HOLDOUT_ACCURACY_GATE,
    PHASE54_PER_CATEGORY_ACCURACY_GATE,
    PHASE54_PER_FIELD_ACCURACY_GATE,
    PHASE54_TYPED_EXACT_MATCH_GATE,
    PHASE54_TYPED_FIELDS,
    build_phase54_blind_items,
    build_phase54_calibration_cases,
    build_phase54_holdout_cases,
    build_phase54_split_integrity,
    evaluate_phase54_hard_reject_cases,
    phase54_ollama_json_schema,
)


EVIDENCE_ROOT = REPO_ROOT / "docs/demo/phase54-typed-proposition-evaluator"
PHASE53_ROOT = REPO_ROOT / "docs/demo/phase53-evaluator-scope-recovery"
PHASE53_SOURCE = CORE_ROOT / "pfe_core/phase53_evaluator_scope_recovery.py"
PHASE54_SOURCE = CORE_ROOT / "pfe_core/phase54_typed_proposition_evaluator.py"
JUDGE_ALIASES = ("semantic_judge_alpha", "semantic_judge_beta")
JUDGE_MODELS = {
    "semantic_judge_alpha": "gemma4:31b",
    "semantic_judge_beta": "qwen3.6:latest",
}


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return dict(value) if isinstance(value, Mapping) else {}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _phase53_snapshot() -> dict[str, Any]:
    manifest = _read_json(PHASE53_ROOT / "evidence_manifest.json")
    mismatches = []
    for item in manifest.get("files") or []:
        path = REPO_ROOT / str(item.get("path") or "")
        current = _sha256(path) if path.exists() else None
        if current != item.get("sha256"):
            mismatches.append({"path": item.get("path"), "expected": item.get("sha256"), "current": current})
    decision = _read_json(PHASE53_ROOT / "phase53-final-decision.json")
    integrity = _read_json(PHASE53_ROOT / "evidence_integrity.json")
    passed = (
        not mismatches
        and integrity.get("passed") is True
        and decision.get("recommendation") == "hold_phase53_evaluator_scope_recovery"
    )
    return {
        "kind": "phase54_phase53_canonical_snapshot",
        "passed": passed,
        "phase53_recommendation": decision.get("recommendation"),
        "manifest_sha256": manifest.get("manifest_sha256"),
        "manifest_file_count": manifest.get("file_count"),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "created_at": _utcnow(),
    }


def _prior_cases_and_failures() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cases = []
    failures = []
    phases = (
        "phase51-dual-evaluator-hardening",
        "phase52-adversarial-evaluator-generalization",
        "phase53-evaluator-scope-recovery",
    )
    for phase in phases:
        root = REPO_ROOT / "docs/demo" / phase
        for directory, filename in (
            ("evidence-evaluator-calibration", "calibration_labeled.json"),
            ("evidence-evaluator-holdout", "holdout_labeled.json"),
        ):
            cases.extend(dict(row) for row in _read_json(root / directory / filename).get("cases") or [])
        failure_path = root / "evidence-evaluator-holdout/failure_analysis.json"
        for row in _read_json(failure_path).get("details") or []:
            failures.append({"phase": phase.split("-", 1)[0], **dict(row)})
    return cases, failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-evidence", action="store_true")
    args = parser.parse_args()
    if args.clean_evidence and EVIDENCE_ROOT.exists():
        preserved = REPO_ROOT / ".phase54-evaluator-debug-preserve"
        if preserved.exists():
            shutil.rmtree(preserved)
        debug = EVIDENCE_ROOT / "evidence-evaluator-debug"
        if debug.exists():
            shutil.copytree(debug, preserved)
        shutil.rmtree(EVIDENCE_ROOT)
        if preserved.exists():
            shutil.copytree(preserved, debug)
            shutil.rmtree(preserved)

    baseline = _phase53_snapshot()
    prior_cases, historical_failures = _prior_cases_and_failures()
    calibration = build_phase54_calibration_cases()
    holdout = build_phase54_holdout_cases()
    split = build_phase54_split_integrity(
        calibration["cases"],
        holdout["cases"],
        prior_cases=prior_cases,
        historical_failure_responses=[str(row.get("assistant_response") or "") for row in historical_failures],
    )
    calibration_blind = build_phase54_blind_items(
        calibration["cases"], seed=5401, prefix="phase54-calibration-blind"
    )
    holdout_blind = build_phase54_blind_items(
        holdout["cases"], seed=5402, prefix="phase54-holdout-blind"
    )
    hard_calibration = evaluate_phase54_hard_reject_cases(calibration["cases"])
    hard_holdout = evaluate_phase54_hard_reject_cases(holdout["cases"])
    schema = phase54_ollama_json_schema()
    protocol = {
        "kind": "phase54_frozen_typed_proposition_protocol",
        "categories": list(PHASE54_CATEGORIES),
        "typed_fields": list(PHASE54_TYPED_FIELDS),
        "semantic_judge_aliases": list(JUDGE_ALIASES),
        "semantic_judge_models_private": dict(JUDGE_MODELS),
        "judges_receive_other_judge_identity": False,
        "judges_receive_gold_label": False,
        "judges_receive_gold_typed_fields": False,
        "judges_return_direct_label": False,
        "ollama_json_schema": schema,
        "composition": "hard_reject_then_any_source_elevation_or_benefit_assertion_reject_else_all_complete_safe_accept_else_edit",
        "calibration_accuracy_gate": PHASE54_CALIBRATION_ACCURACY_GATE,
        "holdout_accuracy_gate": PHASE54_HOLDOUT_ACCURACY_GATE,
        "per_category_accuracy_gate": PHASE54_PER_CATEGORY_ACCURACY_GATE,
        "per_field_accuracy_gate": PHASE54_PER_FIELD_ACCURACY_GATE,
        "typed_exact_match_gate": PHASE54_TYPED_EXACT_MATCH_GATE,
        "false_accept_gate_on_reject_cases": 0,
        "hard_reject_vs_two_safe_accept_conflict_gate": 0,
        "judge_direct_label_gate": 0,
        "hard_reject_recall_gate_on_audited_cases": 1.0,
        "hard_reject_false_positive_gate": 0,
        "temperature": 0,
        "think": False,
        "num_predict": 256,
        "holdout_allowed_only_after_calibration_qualification": True,
        "holdout_may_be_called_once": True,
        "phase51_to_53_holdouts_are_historical_diagnostics_only": True,
        "runtime_replay_allowed": False,
        "boundary_clause_design_allowed": False,
        "runtime_prompt_or_router_change_allowed": False,
        "training_allowed": False,
        "hermes_attachment_allowed": False,
        "product_default_change_allowed": False,
    }
    protocol["protocol_sha256"] = stable_hash(protocol)
    freeze = {
        "kind": "phase54_pre_model_call_freeze",
        "phase53_canonical_snapshot_passed": baseline["passed"],
        "phase53_evaluator_source_sha256": _sha256(PHASE53_SOURCE),
        "phase54_evaluator_source_sha256": _sha256(PHASE54_SOURCE),
        "extraction_rubric_sha256": stable_hash(PHASE54_EXTRACTION_RUBRIC),
        "ollama_json_schema_sha256": stable_hash(schema),
        "calibration_public_sha256": stable_hash(calibration_blind["public_items"]),
        "calibration_hidden_sha256": stable_hash(calibration_blind["hidden_key"]),
        "holdout_public_sha256": stable_hash(holdout_blind["public_items"]),
        "holdout_hidden_sha256": stable_hash(holdout_blind["hidden_key"]),
        "historical_failures_sha256": stable_hash(historical_failures),
        "protocol_sha256": protocol["protocol_sha256"],
        "frozen_before_calibration_calls": True,
        "frozen_before_holdout_calls": True,
        "created_at": _utcnow(),
    }
    historical_snapshot = {
        "kind": "phase54_phase51_to_53_historical_diagnostics",
        "status": "historical_evidence_only",
        "failure_count": len(historical_failures),
        "used_as_phase54_calibration": False,
        "used_as_phase54_holdout": False,
        "used_for_training": False,
        "details": historical_failures,
    }
    source_manifest = {
        "kind": "phase54_source_boundary_manifest",
        "calibration_count": calibration["case_count"],
        "holdout_count": holdout["case_count"],
        "prior_fixture_count": len(prior_cases),
        "historical_failure_count": len(historical_failures),
        "simulated_evaluator_fixture": True,
        "actual_user_feedback_count": 0,
        "actual_human_review_count": 0,
        "not_for_training": True,
        "private_user_material_used": False,
        "holdout_reused": False,
        "runtime_outputs_used": False,
    }
    preparation = {
        "kind": "phase54_preparation_decision",
        "status": "ready_for_calibration" if all(
            (
                baseline["passed"],
                split["passed"],
                hard_calibration["status"] == "passed",
                hard_holdout["status"] == "passed",
            )
        ) else "blocked",
        "phase53_canonical_passed": baseline["passed"],
        "historical_failure_count": len(historical_failures),
        "split_integrity_passed": split["passed"],
        "hard_calibration_passed": hard_calibration["status"] == "passed",
        "hard_holdout_passed": hard_holdout["status"] == "passed",
        "holdout_allowed_only_after_calibration_qualification": True,
    }
    no_runtime = {
        "kind": "phase54_runtime_replay_and_boundary_design_status",
        "runtime_replay_status": "not_requested_in_phase54",
        "runtime_replay_model_call_count": 0,
        "boundary_clause_design_status": "not_requested_in_phase54",
        "boundary_clause_design_created": False,
        "runtime_prompt_changed": False,
        "router_changed": False,
    }
    training = {
        "kind": "phase54_training_attempt",
        "status": "not_requested",
        "training_executed": False,
        "adapter_created": False,
        "auto_training_allowed": False,
    }

    calibration_dir = EVIDENCE_ROOT / "evidence-evaluator-calibration"
    holdout_dir = EVIDENCE_ROOT / "evidence-evaluator-holdout"
    _write_json(EVIDENCE_ROOT / "evidence-baseline/phase53_canonical_snapshot.json", baseline)
    _write_json(EVIDENCE_ROOT / "evidence-baseline/phase51_to_53_historical_diagnostics.json", historical_snapshot)
    _write_json(calibration_dir / "calibration_labeled.json", calibration)
    _write_jsonl(calibration_dir / "blind_items_public.jsonl", calibration_blind["public_items"])
    _write_json(calibration_dir / "blind_hidden_key.json", {"items": calibration_blind["hidden_key"]})
    _write_json(calibration_dir / "hard_reject_report.json", hard_calibration)
    _write_json(holdout_dir / "holdout_labeled.json", holdout)
    _write_jsonl(holdout_dir / "blind_items_public.jsonl", holdout_blind["public_items"])
    _write_json(holdout_dir / "blind_hidden_key.json", {"items": holdout_blind["hidden_key"]})
    _write_json(holdout_dir / "hard_reject_report.json", hard_holdout)
    _write_json(holdout_dir / "split_integrity.json", split)
    _write_json(EVIDENCE_ROOT / "evidence-no-runtime/runtime_status.json", no_runtime)
    _write_json(EVIDENCE_ROOT / "evidence-no-training/training_attempt.json", training)
    _write_json(EVIDENCE_ROOT / "evaluator_protocol.json", protocol)
    _write_json(EVIDENCE_ROOT / "pre_model_call_freeze.json", freeze)
    _write_json(EVIDENCE_ROOT / "source_manifest.json", source_manifest)
    _write_json(EVIDENCE_ROOT / "preparation_decision.json", preparation)
    print(json.dumps(preparation, ensure_ascii=False, indent=2))
    return 0 if preparation["status"] == "ready_for_calibration" else 1


if __name__ == "__main__":
    raise SystemExit(main())
