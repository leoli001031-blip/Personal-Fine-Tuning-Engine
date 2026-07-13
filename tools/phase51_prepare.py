#!/usr/bin/env python3
"""Prepare and freeze Phase51 evaluator and runtime holdouts before model calls."""

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

from pfe_core.phase46_runtime_first_latest_intent import PHASE46_LENGTH_CONTRACT, stable_hash
from pfe_core.phase48_compact_intent_runtime import PHASE48_COMPACT_INTENT_CONTRACT
from pfe_core.phase49_provenance_boundary_recovery import PHASE49_EVIDENCE_BOUNDARY_CLAUSE
from pfe_core.phase50_conditional_provenance_guard import (
    PHASE50_ROUTER_VERSION,
    evaluate_phase50_router_holdout,
)
from pfe_core.phase51_dual_evaluator_hardening import (
    PHASE51_CALIBRATION_ACCURACY_GATE,
    PHASE51_EVALUATOR_RUBRIC,
    PHASE51_HOLDOUT_ACCURACY_GATE,
    build_phase51_blind_items,
    build_phase51_evaluator_calibration_cases,
    build_phase51_evaluator_holdout_cases,
    build_phase51_evaluator_split_integrity,
    build_phase51_runtime_holdout_sessions,
    build_phase51_runtime_split_integrity,
    evaluate_phase51_hard_reject_cases,
)


EVIDENCE_ROOT = REPO_ROOT / "docs" / "demo" / "phase51-dual-evaluator-hardening"
PHASE50_ROOT = REPO_ROOT / "docs" / "demo" / "phase50-conditional-provenance-guard"
PHASE49_ROOT = REPO_ROOT / "docs" / "demo" / "phase49-provenance-boundary-runtime-recovery"
PHASE48_ROOT = REPO_ROOT / "docs" / "demo" / "phase48-compact-intent-runtime-ablation"
PHASE51_SOURCE = CORE_ROOT / "pfe_core" / "phase51_dual_evaluator_hardening.py"
PHASE50_SOURCE = CORE_ROOT / "pfe_core" / "phase50_conditional_provenance_guard.py"
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


def _phase50_snapshot() -> dict[str, Any]:
    manifest = _read_json(PHASE50_ROOT / "evidence_manifest.json")
    mismatches = []
    for item in manifest.get("files") or []:
        path = REPO_ROOT / str(item.get("path") or "")
        current = _sha256(path) if path.exists() else None
        if current != item.get("sha256"):
            mismatches.append({"path": item.get("path"), "expected": item.get("sha256"), "current": current})
    decision = _read_json(PHASE50_ROOT / "phase50-final-decision.json")
    integrity = _read_json(PHASE50_ROOT / "evidence_integrity.json")
    passed = (
        not mismatches
        and integrity.get("passed") is True
        and decision.get("recommendation") == "hold_conditional_provenance_guard_evaluator_unstable"
    )
    return {
        "kind": "phase51_phase50_canonical_snapshot",
        "passed": passed,
        "phase50_recommendation": decision.get("recommendation"),
        "manifest_sha256": manifest.get("manifest_sha256"),
        "manifest_file_count": manifest.get("file_count"),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "created_at": _utcnow(),
    }


def _prior_runtime_sessions() -> list[dict[str, Any]]:
    paths = (
        PHASE48_ROOT / "evidence-holdout" / "holdout.json",
        PHASE49_ROOT / "evidence-holdout" / "holdout.json",
        PHASE50_ROOT / "evidence-holdout" / "holdout.json",
    )
    rows = []
    for path in paths:
        if path.exists():
            rows.extend(dict(row) for row in _read_json(path).get("sessions") or [])
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-evidence", action="store_true")
    args = parser.parse_args()
    if args.clean_evidence and EVIDENCE_ROOT.exists():
        preserved_debug = REPO_ROOT / ".phase51-evaluator-debug-preserve"
        if preserved_debug.exists():
            shutil.rmtree(preserved_debug)
        if (EVIDENCE_ROOT / "evidence-evaluator-debug").exists():
            shutil.copytree(EVIDENCE_ROOT / "evidence-evaluator-debug", preserved_debug)
        shutil.rmtree(EVIDENCE_ROOT)
        if preserved_debug.exists():
            shutil.copytree(preserved_debug, EVIDENCE_ROOT / "evidence-evaluator-debug")
            shutil.rmtree(preserved_debug)

    baseline = _phase50_snapshot()
    calibration = build_phase51_evaluator_calibration_cases()
    evaluator_holdout = build_phase51_evaluator_holdout_cases()
    evaluator_split = build_phase51_evaluator_split_integrity(
        calibration["cases"], evaluator_holdout["cases"]
    )
    calibration_blind = build_phase51_blind_items(
        calibration["cases"], seed=5101, prefix="phase51-calibration-blind"
    )
    holdout_blind = build_phase51_blind_items(
        evaluator_holdout["cases"], seed=5102, prefix="phase51-holdout-blind"
    )
    calibration_hard = evaluate_phase51_hard_reject_cases(calibration["cases"])
    holdout_hard = evaluate_phase51_hard_reject_cases(evaluator_holdout["cases"])

    runtime_holdout = build_phase51_runtime_holdout_sessions()
    runtime_sessions = list(runtime_holdout["sessions"])
    runtime_split = build_phase51_runtime_split_integrity(
        runtime_sessions, prior_sessions=_prior_runtime_sessions()
    )
    router_report = evaluate_phase50_router_holdout(runtime_sessions)
    protocol = {
        "kind": "phase51_frozen_evaluator_and_runtime_protocol",
        "evaluator": {
            "labels": ["accept", "edit", "reject"],
            "deterministic_hard_reject_scope": "explicit_source_elevation_only",
            "semantic_judge_aliases": list(JUDGE_ALIASES),
            "semantic_judge_models_private": dict(JUDGE_MODELS),
            "judges_receive_other_judge_identity": False,
            "judges_receive_variant_identity": False,
            "judges_receive_gold_label": False,
            "adjudication": "hard_reject_then_agreement_else_reject_over_edit_over_accept",
            "calibration_accuracy_gate": PHASE51_CALIBRATION_ACCURACY_GATE,
            "holdout_accuracy_gate": PHASE51_HOLDOUT_ACCURACY_GATE,
            "false_accept_gate_on_reject_cases": 0,
            "hard_reject_recall_gate": 1.0,
            "hard_reject_false_positive_gate": 0,
            "temperature": 0,
            "think": False,
        },
        "runtime": {
            "variants": ["base_compact_v1", "base_global_v2", "base_conditional_guard"],
            "same_qwen3_4b_base_all_arms": True,
            "adapter_loaded_any_arm": False,
            "phase50_router_and_prompts_unchanged": True,
            "privacy_transform_all_arms": True,
            "length_contract": PHASE46_LENGTH_CONTRACT,
            "compact_v1_contract": PHASE48_COMPACT_INTENT_CONTRACT,
            "global_v2_clause": PHASE49_EVIDENCE_BOUNDARY_CLAUSE,
            "conditional_router_version": PHASE50_ROUTER_VERSION,
            "do_sample": False,
            "think": False,
            "input_max_length": 4096,
            "max_new_tokens": 384,
            "repetition_penalty": 1.05,
        },
        "decision_limits": {
            "auto_promote": False,
            "product_default_change": False,
            "training": False,
            "hermes_attachment": False,
            "maximum_recommendation": "manual_shadow_only",
        },
    }
    protocol["protocol_sha256"] = stable_hash(protocol)
    freeze = {
        "kind": "phase51_pre_model_call_freeze",
        "phase50_canonical_snapshot_passed": baseline["passed"],
        "phase50_runtime_source_sha256": _sha256(PHASE50_SOURCE),
        "phase51_evaluator_source_sha256": _sha256(PHASE51_SOURCE),
        "evaluator_rubric_sha256": stable_hash(PHASE51_EVALUATOR_RUBRIC),
        "calibration_public_sha256": stable_hash(calibration_blind["public_items"]),
        "calibration_hidden_sha256": stable_hash(calibration_blind["hidden_key"]),
        "holdout_public_sha256": stable_hash(holdout_blind["public_items"]),
        "holdout_hidden_sha256": stable_hash(holdout_blind["hidden_key"]),
        "runtime_holdout_sha256": stable_hash(runtime_sessions),
        "protocol_sha256": protocol["protocol_sha256"],
        "frozen_before_semantic_judge_calls": True,
        "frozen_before_runtime_generation_calls": True,
        "created_at": _utcnow(),
    }
    source_manifest = {
        "kind": "phase51_source_boundary_manifest",
        "calibration_count": calibration["case_count"],
        "evaluator_holdout_count": evaluator_holdout["case_count"],
        "runtime_holdout_count": runtime_holdout["holdout_count"],
        "runtime_trigger_count": runtime_holdout["expected_trigger_count"],
        "runtime_passthrough_count": runtime_holdout["expected_passthrough_count"],
        "simulated_usage": True,
        "simulated_evaluator_fixture": True,
        "actual_user_feedback_count": 0,
        "actual_human_review_count": 0,
        "not_for_training": True,
        "private_user_material_used": False,
        "prior_holdout_reused": False,
    }
    training_attempt = {
        "kind": "phase51_training_attempt",
        "status": "not_requested",
        "training_executed": False,
        "adapter_created": False,
        "reason": "Phase51 evaluates evaluator reliability and unchanged runtime arms only.",
        "auto_training_allowed": False,
    }
    preparation = {
        "kind": "phase51_preparation_decision",
        "status": "ready_for_calibration" if all(
            (
                baseline["passed"],
                evaluator_split["passed"],
                calibration_hard["status"] == "passed",
                holdout_hard["status"] == "passed",
                runtime_split["passed"],
                router_report["status"] == "passed",
            )
        ) else "blocked",
        "phase50_canonical_passed": baseline["passed"],
        "evaluator_split_passed": evaluator_split["passed"],
        "hard_reject_calibration_passed": calibration_hard["status"] == "passed",
        "hard_reject_holdout_passed": holdout_hard["status"] == "passed",
        "runtime_split_passed": runtime_split["passed"],
        "router_holdout_passed": router_report["status"] == "passed",
        "runtime_generation_allowed_only_after_evaluator_qualification": True,
    }

    calibration_dir = EVIDENCE_ROOT / "evidence-evaluator-calibration"
    holdout_dir = EVIDENCE_ROOT / "evidence-evaluator-holdout"
    runtime_dir = EVIDENCE_ROOT / "evidence-runtime-holdout"
    baseline_dir = EVIDENCE_ROOT / "evidence-baseline"
    _write_json(baseline_dir / "phase50_canonical_snapshot.json", baseline)
    _write_json(calibration_dir / "calibration_labeled.json", calibration)
    _write_jsonl(calibration_dir / "blind_items_public.jsonl", calibration_blind["public_items"])
    _write_json(calibration_dir / "blind_hidden_key.json", {"items": calibration_blind["hidden_key"]})
    _write_json(calibration_dir / "hard_reject_report.json", calibration_hard)
    _write_json(holdout_dir / "holdout_labeled.json", evaluator_holdout)
    _write_jsonl(holdout_dir / "blind_items_public.jsonl", holdout_blind["public_items"])
    _write_json(holdout_dir / "blind_hidden_key.json", {"items": holdout_blind["hidden_key"]})
    _write_json(holdout_dir / "hard_reject_report.json", holdout_hard)
    _write_json(holdout_dir / "split_integrity.json", evaluator_split)
    _write_json(runtime_dir / "holdout.json", runtime_holdout)
    _write_json(runtime_dir / "split_integrity.json", runtime_split)
    _write_json(runtime_dir / "premodel_router_report.json", router_report)
    _write_json(EVIDENCE_ROOT / "evidence-no-training" / "training_attempt.json", training_attempt)
    _write_json(EVIDENCE_ROOT / "evaluator_runtime_protocol.json", protocol)
    _write_json(EVIDENCE_ROOT / "pre_model_call_freeze.json", freeze)
    _write_json(EVIDENCE_ROOT / "source_manifest.json", source_manifest)
    _write_json(EVIDENCE_ROOT / "preparation_decision.json", preparation)
    print(json.dumps(preparation, ensure_ascii=False, indent=2))
    return 0 if preparation["status"] == "ready_for_calibration" else 1


if __name__ == "__main__":
    raise SystemExit(main())
