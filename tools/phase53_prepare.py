#!/usr/bin/env python3
"""Prepare and freeze Phase53 scope-recovery evidence before model calls."""

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
from pfe_core.phase53_evaluator_scope_recovery import (
    PHASE53_CALIBRATION_ACCURACY_GATE,
    PHASE53_CATEGORIES,
    PHASE53_EVALUATOR_RUBRIC,
    PHASE53_HOLDOUT_ACCURACY_GATE,
    PHASE53_PER_CATEGORY_ACCURACY_GATE,
    build_phase53_blind_items,
    build_phase53_calibration_cases,
    build_phase53_holdout_cases,
    build_phase53_split_integrity,
    evaluate_phase53_hard_reject_cases,
)


EVIDENCE_ROOT = REPO_ROOT / "docs/demo/phase53-evaluator-scope-recovery"
PHASE52_ROOT = REPO_ROOT / "docs/demo/phase52-adversarial-evaluator-generalization"
PHASE51_ROOT = REPO_ROOT / "docs/demo/phase51-dual-evaluator-hardening"
PHASE52_SOURCE = CORE_ROOT / "pfe_core/phase52_adversarial_evaluator_generalization.py"
PHASE53_SOURCE = CORE_ROOT / "pfe_core/phase53_evaluator_scope_recovery.py"
JUDGE_ALIASES = ("semantic_judge_alpha", "semantic_judge_beta")
JUDGE_MODELS = {
    "semantic_judge_alpha": "gemma4:31b",
    "semantic_judge_beta": "qwen3.6:latest",
}


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
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


def _phase52_snapshot() -> dict[str, Any]:
    manifest = _read_json(PHASE52_ROOT / "evidence_manifest.json")
    mismatches = []
    for item in manifest.get("files") or []:
        path = REPO_ROOT / str(item.get("path") or "")
        current = _sha256(path) if path.exists() else None
        if current != item.get("sha256"):
            mismatches.append({"path": item.get("path"), "expected": item.get("sha256"), "current": current})
    decision = _read_json(PHASE52_ROOT / "phase52-final-decision.json")
    integrity = _read_json(PHASE52_ROOT / "evidence_integrity.json")
    passed = (
        not mismatches
        and integrity.get("passed") is True
        and decision.get("recommendation") == "hold_phase52_evaluator_generalization"
    )
    return {
        "kind": "phase53_phase52_canonical_snapshot",
        "passed": passed,
        "phase52_recommendation": decision.get("recommendation"),
        "manifest_sha256": manifest.get("manifest_sha256"),
        "manifest_file_count": manifest.get("file_count"),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "created_at": _utcnow(),
    }


def _prior_cases() -> list[dict[str, Any]]:
    rows = []
    for root in (PHASE51_ROOT, PHASE52_ROOT):
        for directory, filename in (
            ("evidence-evaluator-calibration", "calibration_labeled.json"),
            ("evidence-evaluator-holdout", "holdout_labeled.json"),
        ):
            path = root / directory / filename
            if path.exists():
                rows.extend(dict(row) for row in _read_json(path).get("cases") or [])
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-evidence", action="store_true")
    args = parser.parse_args()
    if args.clean_evidence and EVIDENCE_ROOT.exists():
        preserved = REPO_ROOT / ".phase53-evaluator-debug-preserve"
        if preserved.exists():
            shutil.rmtree(preserved)
        if (EVIDENCE_ROOT / "evidence-evaluator-debug").exists():
            shutil.copytree(EVIDENCE_ROOT / "evidence-evaluator-debug", preserved)
        shutil.rmtree(EVIDENCE_ROOT)
        if preserved.exists():
            shutil.copytree(preserved, EVIDENCE_ROOT / "evidence-evaluator-debug")
            shutil.rmtree(preserved)

    baseline = _phase52_snapshot()
    historical = _read_json(PHASE52_ROOT / "evidence-evaluator-holdout/failure_analysis.json")
    historical_details = [dict(row) for row in historical.get("details") or []]
    calibration = build_phase53_calibration_cases()
    holdout = build_phase53_holdout_cases()
    prior = _prior_cases()
    split = build_phase53_split_integrity(
        calibration["cases"],
        holdout["cases"],
        prior_cases=prior,
        historical_failure_responses=[str(row.get("assistant_response") or "") for row in historical_details],
    )
    calibration_blind = build_phase53_blind_items(
        calibration["cases"], seed=5301, prefix="phase53-calibration-blind"
    )
    holdout_blind = build_phase53_blind_items(
        holdout["cases"], seed=5302, prefix="phase53-holdout-blind"
    )
    hard_calibration = evaluate_phase53_hard_reject_cases(calibration["cases"])
    hard_holdout = evaluate_phase53_hard_reject_cases(holdout["cases"])
    protocol = {
        "kind": "phase53_frozen_scope_recovery_protocol",
        "labels": list(("accept", "edit", "reject")),
        "categories": list(PHASE53_CATEGORIES),
        "semantic_judge_aliases": list(JUDGE_ALIASES),
        "semantic_judge_models_private": dict(JUDGE_MODELS),
        "judges_receive_other_judge_identity": False,
        "judges_receive_gold_label": False,
        "adjudication": "hard_reject_then_agreement_else_reject_over_edit_over_accept",
        "calibration_accuracy_gate": PHASE53_CALIBRATION_ACCURACY_GATE,
        "holdout_accuracy_gate": PHASE53_HOLDOUT_ACCURACY_GATE,
        "per_category_accuracy_gate": PHASE53_PER_CATEGORY_ACCURACY_GATE,
        "false_accept_gate_on_reject_cases": 0,
        "hard_reject_vs_two_accept_conflict_gate": 0,
        "hard_reject_recall_gate": 1.0,
        "hard_reject_false_positive_gate": 0,
        "temperature": 0,
        "think": False,
        "num_predict": 384,
        "holdout_allowed_only_after_calibration_qualification": True,
        "holdout_may_be_called_once": True,
        "phase51_runtime_replay_allowed": False,
        "boundary_clause_design_allowed": False,
        "runtime_prompt_or_router_change_allowed": False,
        "training_allowed": False,
        "hermes_attachment_allowed": False,
        "product_default_change_allowed": False,
    }
    protocol["protocol_sha256"] = stable_hash(protocol)
    freeze = {
        "kind": "phase53_pre_model_call_freeze",
        "phase52_canonical_snapshot_passed": baseline["passed"],
        "phase52_evaluator_source_sha256": _sha256(PHASE52_SOURCE),
        "phase53_evaluator_source_sha256": _sha256(PHASE53_SOURCE),
        "evaluator_rubric_sha256": stable_hash(PHASE53_EVALUATOR_RUBRIC),
        "calibration_public_sha256": stable_hash(calibration_blind["public_items"]),
        "calibration_hidden_sha256": stable_hash(calibration_blind["hidden_key"]),
        "holdout_public_sha256": stable_hash(holdout_blind["public_items"]),
        "holdout_hidden_sha256": stable_hash(holdout_blind["hidden_key"]),
        "historical_phase52_failures_sha256": stable_hash(historical_details),
        "protocol_sha256": protocol["protocol_sha256"],
        "frozen_before_calibration_calls": True,
        "frozen_before_holdout_calls": True,
        "created_at": _utcnow(),
    }
    historical_snapshot = {
        "kind": "phase53_phase52_historical_failure_snapshot",
        "status": "historical_evidence_only",
        "failure_count": len(historical_details),
        "used_as_phase53_calibration": False,
        "used_as_phase53_holdout": False,
        "used_for_training": False,
        "details": historical_details,
    }
    source_manifest = {
        "kind": "phase53_source_boundary_manifest",
        "calibration_count": calibration["case_count"],
        "holdout_count": holdout["case_count"],
        "historical_failure_count": len(historical_details),
        "simulated_evaluator_fixture": True,
        "actual_user_feedback_count": 0,
        "actual_human_review_count": 0,
        "not_for_training": True,
        "private_user_material_used": False,
        "holdout_reused": False,
        "runtime_outputs_used": False,
    }
    preparation = {
        "kind": "phase53_preparation_decision",
        "status": "ready_for_calibration" if all(
            (
                baseline["passed"],
                len(historical_details) == 2,
                split["passed"],
                hard_calibration["status"] == "passed",
                hard_holdout["status"] == "passed",
            )
        ) else "blocked",
        "phase52_canonical_passed": baseline["passed"],
        "historical_failure_count": len(historical_details),
        "split_integrity_passed": split["passed"],
        "hard_calibration_passed": hard_calibration["status"] == "passed",
        "hard_holdout_passed": hard_holdout["status"] == "passed",
        "holdout_allowed_only_after_calibration_qualification": True,
    }
    no_runtime = {
        "kind": "phase53_runtime_replay_and_boundary_design_status",
        "phase51_runtime_replay_status": "not_requested_in_phase53",
        "runtime_replay_model_call_count": 0,
        "boundary_clause_design_status": "not_requested_in_phase53",
        "boundary_clause_design_created": False,
        "runtime_prompt_changed": False,
        "router_changed": False,
    }
    training = {
        "kind": "phase53_training_attempt",
        "status": "not_requested",
        "training_executed": False,
        "adapter_created": False,
        "auto_training_allowed": False,
    }

    calibration_dir = EVIDENCE_ROOT / "evidence-evaluator-calibration"
    holdout_dir = EVIDENCE_ROOT / "evidence-evaluator-holdout"
    _write_json(EVIDENCE_ROOT / "evidence-baseline/phase52_canonical_snapshot.json", baseline)
    _write_json(EVIDENCE_ROOT / "evidence-baseline/phase52_historical_failures.json", historical_snapshot)
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
