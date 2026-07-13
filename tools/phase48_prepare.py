#!/usr/bin/env python3
"""Prepare and freeze Phase48 compact-intent runtime ablation evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = REPO_ROOT / "pfe-core"
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

from pfe_core.phase46_runtime_first_latest_intent import (
    build_phase46_scorer_calibration_cases,
    evaluate_phase46_scorer_calibration,
)
from pfe_core.phase48_compact_intent_runtime import (
    PHASE46_LATEST_INTENT_CONTRACT,
    PHASE46_LENGTH_CONTRACT,
    PHASE48_COMPACT_INTENT_CONTRACT,
    build_phase48_holdout_sessions,
    build_phase48_split_integrity,
    stable_hash,
)


EVIDENCE_ROOT = REPO_ROOT / "docs" / "demo" / "phase48-compact-intent-runtime-ablation"
PHASE47_ROOT = REPO_ROOT / "docs" / "demo" / "phase47-simulated-user-review"
PHASE46_ROOT = REPO_ROOT / "docs" / "demo" / "phase46-runtime-first-latest-intent-ablation"
MODEL_PATH = REPO_ROOT / "models" / "Qwen3-4B"
PHASE45_SOURCE = CORE_ROOT / "pfe_core" / "phase45_privacy_multiturn_preference.py"
PHASE46_SOURCE = CORE_ROOT / "pfe_core" / "phase46_runtime_first_latest_intent.py"
PHASE48_SOURCE = CORE_ROOT / "pfe_core" / "phase48_compact_intent_runtime.py"


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


def _canonical_snapshot(root: Path, *, phase: str, expected_recommendation: str) -> dict[str, Any]:
    manifest = _read_json(root / "evidence_manifest.json")
    mismatches = []
    for item in manifest.get("files") or []:
        path = REPO_ROOT / str(item.get("path") or "")
        current = _sha256(path) if path.exists() else None
        if current != item.get("sha256"):
            mismatches.append({"path": item.get("path"), "expected": item.get("sha256"), "current": current})
    decision = _read_json(root / f"{phase}-final-decision.json")
    integrity = _read_json(root / "evidence_integrity.json")
    return {
        "kind": f"phase48_{phase}_canonical_snapshot",
        "passed": not mismatches and integrity.get("passed") is True and decision.get("recommendation") == expected_recommendation,
        "phase": phase,
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
        shutil.rmtree(EVIDENCE_ROOT)

    phase47 = _canonical_snapshot(
        PHASE47_ROOT,
        phase="phase47",
        expected_recommendation="use_simulated_reviewed_pack_for_fresh_runtime_ablation",
    )
    phase46 = _canonical_snapshot(
        PHASE46_ROOT,
        phase="phase46",
        expected_recommendation="hold_runtime_and_revise_eval_or_data",
    )
    reviewed = _read_jsonl(PHASE47_ROOT / "evidence-candidates" / "reviewed_candidates.jsonl")
    prior_holdout = _read_json(PHASE46_ROOT / "evidence-holdout" / "holdout.json").get("sessions") or []
    holdout = build_phase48_holdout_sessions()
    sessions = list(holdout["sessions"])
    split = build_phase48_split_integrity(reviewed, sessions, prior_holdout_sessions=prior_holdout)
    calibration_cases = build_phase46_scorer_calibration_cases()
    calibration = evaluate_phase46_scorer_calibration(calibration_cases["cases"])
    protocol = {
        "kind": "phase48_fair_three_arm_runtime_protocol",
        "variants": ["base_privacy", "base_compact_intent", "base_full_intent"],
        "same_qwen3_4b_base_all_arms": True,
        "adapter_loaded_any_arm": False,
        "privacy_transform_all_arms": True,
        "same_length_contract_all_arms": True,
        "length_contract": PHASE46_LENGTH_CONTRACT,
        "compact_intent_contract": PHASE48_COMPACT_INTENT_CONTRACT,
        "full_intent_contract": PHASE46_LATEST_INTENT_CONTRACT,
        "compact_uses_xml_or_tag_envelope": False,
        "full_uses_phase46_latest_user_envelope": True,
        "do_sample": False,
        "think": False,
        "input_max_length": 4096,
        "max_new_tokens": 384,
        "repetition_penalty": 1.05,
        "formal_eval_requires_all_arm_truncation_at_most_0_05": True,
    }
    protocol["protocol_sha256"] = stable_hash(protocol)
    holdout_freeze = {
        "kind": "phase48_holdout_freeze",
        "holdout_manifest_sha256": stable_hash(sessions),
        "holdout_count": len(sessions),
        "frozen_before_model_calls": True,
        "phase46_holdout_reused": False,
        "phase47_candidate_reused": False,
        "created_at": _utcnow(),
    }
    scorer_freeze = {
        "kind": "phase48_scorer_and_runtime_freeze",
        "phase45_privacy_source_sha256": _sha256(PHASE45_SOURCE),
        "phase46_scorer_source_sha256": _sha256(PHASE46_SOURCE),
        "phase48_runtime_source_sha256": _sha256(PHASE48_SOURCE),
        "calibration_status": calibration.get("status"),
        "calibration_precision": calibration.get("precision"),
        "calibration_recall": calibration.get("recall"),
        "frozen_before_model_calls": True,
        "created_at": _utcnow(),
    }
    model = {
        "kind": "phase48_model_selection",
        "model_path": str(MODEL_PATH),
        "model_ready": MODEL_PATH.is_dir() and (MODEL_PATH / "config.json").exists(),
        "model": "Qwen3-4B",
        "adapter_loaded": False,
        "new_training": False,
        "qwen27b_training_allowed": False,
    }
    no_training = {
        "kind": "phase48_training_attempt",
        "status": "skipped_by_design",
        "new_training": False,
        "new_adapter_created": False,
        "reason": "Phase48 isolates compact runtime behavior before any additional training.",
        "actual_human_review_completed": False,
        "auto_training_allowed": False,
        "auto_promotion_allowed": False,
    }
    source_manifest = {
        "kind": "phase48_source_boundary_manifest",
        "phase47_reviewed_candidate_count": len(reviewed),
        "phase47_reviewed_pack_used_for_taxonomy_guidance_only": True,
        "phase47_reviewed_text_copied_into_holdout": False,
        "phase47_reviewed_pack_used_for_training": False,
        "phase46_holdout_count": len(prior_holdout),
        "phase46_holdout_reused": False,
        "phase48_holdout_count": len(sessions),
        "actual_user_feedback_count": 0,
        "actual_product_benefit_claim_allowed": False,
    }
    ready = (
        phase47.get("passed") is True
        and phase46.get("passed") is True
        and split.get("passed") is True
        and calibration.get("status") == "passed"
        and model.get("model_ready") is True
        and len(sessions) == 64
    )
    preparation = {
        "kind": "phase48_preparation_decision",
        "status": "ready_for_real_runtime_ablation" if ready else "blocked",
        "phase47_frozen": phase47.get("passed") is True,
        "phase46_frozen": phase46.get("passed") is True,
        "split_integrity_passed": split.get("passed") is True,
        "scorer_calibration_passed": calibration.get("status") == "passed",
        "scorer_frozen_before_model_calls": True,
        "holdout_frozen_before_model_calls": True,
        "model_ready": model.get("model_ready"),
        "new_training_allowed": False,
        "actual_product_benefit_claim_allowed": False,
        "auto_promotion_allowed": False,
        "created_at": _utcnow(),
    }

    _write_json(EVIDENCE_ROOT / "evidence-baseline" / "phase47_canonical_snapshot.json", phase47)
    _write_json(EVIDENCE_ROOT / "evidence-baseline" / "phase46_canonical_snapshot.json", phase46)
    _write_json(EVIDENCE_ROOT / "evidence-baseline" / "model_selection.json", model)
    _write_json(EVIDENCE_ROOT / "evidence-holdout" / "holdout.json", holdout)
    _write_json(EVIDENCE_ROOT / "evidence-holdout" / "holdout_freeze.json", holdout_freeze)
    _write_json(EVIDENCE_ROOT / "evidence-holdout" / "split_integrity.json", split)
    _write_json(EVIDENCE_ROOT / "evidence-holdout" / "source_manifest.json", source_manifest)
    _write_json(EVIDENCE_ROOT / "evidence-holdout" / "runtime_protocol.json", protocol)
    _write_json(EVIDENCE_ROOT / "evidence-scorer-calibration" / "calibration_cases.json", calibration_cases)
    _write_json(EVIDENCE_ROOT / "evidence-scorer-calibration" / "calibration_report.json", calibration)
    _write_json(EVIDENCE_ROOT / "evidence-scorer-calibration" / "scorer_freeze.json", scorer_freeze)
    _write_json(EVIDENCE_ROOT / "evidence-no-training" / "training_attempt.json", no_training)
    _write_json(EVIDENCE_ROOT / "preparation_decision.json", preparation)
    print(json.dumps({
        "status": preparation["status"],
        "holdout_count": len(sessions),
        "category_counts": holdout["category_counts"],
        "split_integrity_passed": split["passed"],
        "calibration_status": calibration["status"],
        "model_ready": model["model_ready"],
        "new_training_allowed": False,
    }, ensure_ascii=False, indent=2))
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
