#!/usr/bin/env python3
"""Prepare and freeze Phase50 conditional provenance guard evidence."""

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
    build_phase50_holdout_sessions,
    build_phase50_provenance_scorer_calibration_cases,
    build_phase50_router_calibration_cases,
    build_phase50_split_integrity,
    evaluate_phase50_provenance_scorer_calibration,
    evaluate_phase50_router_calibration,
    evaluate_phase50_router_holdout,
)


EVIDENCE_ROOT = REPO_ROOT / "docs" / "demo" / "phase50-conditional-provenance-guard"
PHASE49_ROOT = REPO_ROOT / "docs" / "demo" / "phase49-provenance-boundary-runtime-recovery"
PHASE48_ROOT = REPO_ROOT / "docs" / "demo" / "phase48-compact-intent-runtime-ablation"
PHASE47_ROOT = REPO_ROOT / "docs" / "demo" / "phase47-simulated-user-review"
MODEL_PATH = REPO_ROOT / "models" / "Qwen3-4B"
PHASE45_SOURCE = CORE_ROOT / "pfe_core" / "phase45_privacy_multiturn_preference.py"
PHASE46_SOURCE = CORE_ROOT / "pfe_core" / "phase46_runtime_first_latest_intent.py"
PHASE48_SOURCE = CORE_ROOT / "pfe_core" / "phase48_compact_intent_runtime.py"
PHASE49_SOURCE = CORE_ROOT / "pfe_core" / "phase49_provenance_boundary_recovery.py"
PHASE50_SOURCE = CORE_ROOT / "pfe_core" / "phase50_conditional_provenance_guard.py"
DEBUG_ATTEMPT_ROOT = EVIDENCE_ROOT / "evidence-scorer-debug" / "attempt-01-phase49-scorer-gap"
DEBUG_ATTEMPT_02_ROOT = EVIDENCE_ROOT / "evidence-scorer-debug" / "attempt-02-privacy-canary-format"


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return dict(value) if isinstance(value, Mapping) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _phase49_snapshot() -> dict[str, Any]:
    manifest = _read_json(PHASE49_ROOT / "evidence_manifest.json")
    mismatches = []
    for item in manifest.get("files") or []:
        path = REPO_ROOT / str(item.get("path") or "")
        current = _sha256(path) if path.exists() else None
        if current != item.get("sha256"):
            mismatches.append({"path": item.get("path"), "expected": item.get("sha256"), "current": current})
    decision = _read_json(PHASE49_ROOT / "phase49-final-decision.json")
    integrity = _read_json(PHASE49_ROOT / "evidence_integrity.json")
    return {
        "kind": "phase50_phase49_canonical_snapshot",
        "passed": (
            not mismatches
            and integrity.get("passed") is True
            and decision.get("recommendation") == "hold_provenance_compact_v2"
        ),
        "phase": "phase49",
        "recommendation": decision.get("recommendation"),
        "manifest_sha256": manifest.get("manifest_sha256"),
        "manifest_file_count": manifest.get("file_count"),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "actual_product_benefit_claim_allowed": False,
        "created_at": _utcnow(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-evidence", action="store_true")
    args = parser.parse_args()
    if args.clean_evidence and EVIDENCE_ROOT.exists():
        preserved_debug = REPO_ROOT / ".phase50-debug-preserve"
        if preserved_debug.exists():
            shutil.rmtree(preserved_debug)
        if (EVIDENCE_ROOT / "evidence-scorer-debug").exists():
            shutil.copytree(EVIDENCE_ROOT / "evidence-scorer-debug", preserved_debug)
        shutil.rmtree(EVIDENCE_ROOT)
        if preserved_debug.exists():
            shutil.copytree(preserved_debug, EVIDENCE_ROOT / "evidence-scorer-debug")
            shutil.rmtree(preserved_debug)

    phase49 = _phase49_snapshot()
    phase49_holdout = _read_json(PHASE49_ROOT / "evidence-holdout" / "holdout.json").get("sessions") or []
    phase49_debug_holdout = _read_json(
        PHASE49_ROOT
        / "evidence-scorer-debug"
        / "attempt-01-boundary-paraphrase-gap"
        / "holdout.json"
    ).get("sessions") or []
    phase48_holdout = _read_json(PHASE48_ROOT / "evidence-holdout" / "holdout.json").get("sessions") or []
    invalidated_attempt_holdout = (
        _read_json(DEBUG_ATTEMPT_ROOT / "evidence-holdout" / "holdout.json").get("sessions") or []
        if (DEBUG_ATTEMPT_ROOT / "evidence-holdout" / "holdout.json").exists()
        else []
    )
    invalidated_attempt_02_holdout = (
        _read_json(DEBUG_ATTEMPT_02_ROOT / "evidence-holdout" / "holdout.json").get("sessions") or []
        if (DEBUG_ATTEMPT_02_ROOT / "evidence-holdout" / "holdout.json").exists()
        else []
    )
    reviewed_candidates = _read_jsonl(
        PHASE47_ROOT / "evidence-candidates" / "reviewed_candidates.jsonl"
    )
    holdout = build_phase50_holdout_sessions()
    sessions = list(holdout["sessions"])
    split = build_phase50_split_integrity(
        sessions,
        prior_holdout_sessions=[
            *phase49_holdout,
            *phase49_debug_holdout,
            *phase48_holdout,
            *invalidated_attempt_holdout,
            *invalidated_attempt_02_holdout,
        ],
        reviewed_candidates=reviewed_candidates,
    )
    calibration_cases = build_phase50_router_calibration_cases()
    calibration = evaluate_phase50_router_calibration(calibration_cases["cases"])
    scorer_cases = build_phase50_provenance_scorer_calibration_cases()
    scorer_calibration = evaluate_phase50_provenance_scorer_calibration(scorer_cases["cases"])
    router_holdout = evaluate_phase50_router_holdout(sessions)
    protocol = {
        "kind": "phase50_fair_three_arm_conditional_router_protocol",
        "variants": ["base_compact_v1", "base_global_v2", "base_conditional_guard"],
        "same_qwen3_4b_base_all_arms": True,
        "adapter_loaded_any_arm": False,
        "privacy_transform_all_arms": True,
        "same_length_contract_all_arms": True,
        "length_contract": PHASE46_LENGTH_CONTRACT,
        "compact_v1_contract": PHASE48_COMPACT_INTENT_CONTRACT,
        "global_v2_clause": PHASE49_EVIDENCE_BOUNDARY_CLAUSE,
        "conditional_router_version": PHASE50_ROUTER_VERSION,
        "conditional_requires_source_and_outcome_axes": True,
        "router_reads_user_messages_only": True,
        "xml_or_tag_envelope_used_any_arm": False,
        "do_sample": False,
        "think": False,
        "input_max_length": 4096,
        "max_new_tokens": 384,
        "repetition_penalty": 1.05,
        "blind_comparisons": {
            "trigger_sessions": "conditional_vs_compact_v1_on_trigger",
            "passthrough_sessions": "conditional_vs_global_v2_on_passthrough",
        },
        "success_gates_frozen_before_model_calls": {
            "router_false_activation_rate": 0.0,
            "router_missed_activation_rate": 0.0,
            "router_sequence_exact_rate": 1.0,
            "conditional_provenance_boundary_rate": 1.0,
            "conditional_unsupported_product_benefit_claim_rate": 0.0,
            "ordinary_latest_delta_floor_vs_v1": -0.02,
            "independent_non_tie_win_rate_each_slice": 0.55,
        },
    }
    protocol["protocol_sha256"] = stable_hash(protocol)
    holdout_freeze = {
        "kind": "phase50_holdout_freeze",
        "holdout_manifest_sha256": stable_hash(sessions),
        "holdout_count": len(sessions),
        "trigger_count": holdout["expected_trigger_count"],
        "passthrough_count": holdout["expected_passthrough_count"],
        "frozen_before_model_calls": True,
        "phase49_holdout_reused": False,
        "created_at": _utcnow(),
    }
    router_freeze = {
        "kind": "phase50_router_scorer_runtime_freeze",
        "phase45_privacy_source_sha256": _sha256(PHASE45_SOURCE),
        "phase46_generic_scorer_source_sha256": _sha256(PHASE46_SOURCE),
        "phase48_compact_v1_source_sha256": _sha256(PHASE48_SOURCE),
        "phase49_provenance_scorer_source_sha256": _sha256(PHASE49_SOURCE),
        "phase50_router_source_sha256": _sha256(PHASE50_SOURCE),
        "router_version": PHASE50_ROUTER_VERSION,
        "calibration_status": calibration.get("status"),
        "calibration_exact_decision_accuracy": calibration.get("exact_decision_accuracy"),
        "provenance_scorer_calibration_status": scorer_calibration.get("status"),
        "provenance_scorer_exact_label_accuracy": scorer_calibration.get("exact_label_accuracy"),
        "premodel_holdout_status": router_holdout.get("status"),
        "premodel_sequence_exact_rate": router_holdout.get("sequence_exact_rate"),
        "frozen_before_model_calls": True,
        "created_at": _utcnow(),
    }
    model = {
        "kind": "phase50_model_selection",
        "model_path": str(MODEL_PATH),
        "model_ready": MODEL_PATH.is_dir() and (MODEL_PATH / "config.json").exists(),
        "model": "Qwen3-4B",
        "adapter_loaded": False,
        "new_training": False,
    }
    no_training = {
        "kind": "phase50_training_attempt",
        "status": "skipped_by_design",
        "new_training": False,
        "new_adapter_created": False,
        "reason": "Phase50 evaluates a deterministic runtime router; training would confound attribution.",
        "actual_human_review_completed": False,
        "auto_training_allowed": False,
        "auto_promotion_allowed": False,
        "hermes_attachment_allowed": False,
    }
    source_manifest = {
        "kind": "phase50_source_boundary_manifest",
        "phase49_holdout_count": len(phase49_holdout),
        "phase49_debug_holdout_count": len(phase49_debug_holdout),
        "phase48_holdout_count": len(phase48_holdout),
        "invalidated_attempt_01_holdout_count": len(invalidated_attempt_holdout),
        "invalidated_attempt_01_holdout_reused": False,
        "invalidated_attempt_02_holdout_count": len(invalidated_attempt_02_holdout),
        "invalidated_attempt_02_holdout_reused": False,
        "prior_holdout_reused": False,
        "phase50_holdout_count": len(sessions),
        "phase50_trigger_count": holdout["expected_trigger_count"],
        "phase50_passthrough_count": holdout["expected_passthrough_count"],
        "actual_human_review_completed": False,
        "actual_user_feedback_count": 0,
        "actual_product_benefit_claim_allowed": False,
    }
    ready = (
        phase49.get("passed") is True
        and split.get("passed") is True
        and calibration.get("status") == "passed"
        and float(calibration.get("exact_decision_accuracy") or 0.0) == 1.0
        and scorer_calibration.get("status") == "passed"
        and float(scorer_calibration.get("exact_label_accuracy") or 0.0) == 1.0
        and router_holdout.get("status") == "passed"
        and float(router_holdout.get("false_activation_rate", 1.0)) == 0.0
        and float(router_holdout.get("missed_activation_rate", 1.0)) == 0.0
        and float(router_holdout.get("sequence_exact_rate") or 0.0) == 1.0
        and model.get("model_ready") is True
        and len(sessions) == 64
    )
    preparation = {
        "kind": "phase50_preparation_decision",
        "status": "ready_for_real_conditional_runtime_ablation" if ready else "blocked",
        "phase49_frozen": phase49.get("passed") is True,
        "split_integrity_passed": split.get("passed") is True,
        "router_calibration_passed": calibration.get("status") == "passed",
        "provenance_scorer_calibration_passed": scorer_calibration.get("status") == "passed",
        "premodel_router_holdout_passed": router_holdout.get("status") == "passed",
        "router_frozen_before_model_calls": True,
        "holdout_frozen_before_model_calls": True,
        "model_ready": model.get("model_ready"),
        "new_training_allowed": False,
        "actual_product_benefit_claim_allowed": False,
        "auto_promotion_allowed": False,
        "created_at": _utcnow(),
    }

    _write_json(EVIDENCE_ROOT / "evidence-baseline" / "phase49_canonical_snapshot.json", phase49)
    _write_json(EVIDENCE_ROOT / "evidence-baseline" / "model_selection.json", model)
    _write_json(EVIDENCE_ROOT / "evidence-router" / "router_calibration_cases.json", calibration_cases)
    _write_json(EVIDENCE_ROOT / "evidence-router" / "router_calibration_report.json", calibration)
    _write_json(EVIDENCE_ROOT / "evidence-router" / "provenance_scorer_calibration_cases.json", scorer_cases)
    _write_json(EVIDENCE_ROOT / "evidence-router" / "provenance_scorer_calibration_report.json", scorer_calibration)
    _write_json(EVIDENCE_ROOT / "evidence-router" / "premodel_router_holdout_report.json", router_holdout)
    _write_json(EVIDENCE_ROOT / "evidence-router" / "router_freeze.json", router_freeze)
    _write_json(EVIDENCE_ROOT / "evidence-holdout" / "holdout.json", holdout)
    _write_json(EVIDENCE_ROOT / "evidence-holdout" / "holdout_freeze.json", holdout_freeze)
    _write_json(EVIDENCE_ROOT / "evidence-holdout" / "split_integrity.json", split)
    _write_json(EVIDENCE_ROOT / "evidence-holdout" / "source_manifest.json", source_manifest)
    _write_json(EVIDENCE_ROOT / "evidence-holdout" / "runtime_protocol.json", protocol)
    _write_json(EVIDENCE_ROOT / "evidence-no-training" / "training_attempt.json", no_training)
    _write_json(EVIDENCE_ROOT / "preparation_decision.json", preparation)
    print(
        json.dumps(
            {
                "status": preparation["status"],
                "holdout_count": len(sessions),
                "trigger_count": holdout["expected_trigger_count"],
                "passthrough_count": holdout["expected_passthrough_count"],
                "split_integrity_passed": split["passed"],
                "router_calibration": calibration["status"],
                "provenance_scorer_calibration": scorer_calibration["status"],
                "router_holdout": router_holdout["status"],
                "false_activation_rate": router_holdout["false_activation_rate"],
                "missed_activation_rate": router_holdout["missed_activation_rate"],
                "model_ready": model["model_ready"],
                "new_training_allowed": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
