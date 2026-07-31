#!/usr/bin/env python3
"""Prepare and freeze Phase49 provenance-boundary runtime recovery evidence."""

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
from pfe_core.phase49_provenance_boundary_recovery import (
    PHASE49_EVIDENCE_BOUNDARY_CLAUSE,
    build_phase49_holdout_sessions,
    build_phase49_scorer_calibration_cases,
    build_phase49_simulated_review,
    build_phase49_split_integrity,
    evaluate_phase49_scorer_calibration,
)


EVIDENCE_ROOT = REPO_ROOT / "docs" / "demo" / "phase49-provenance-boundary-runtime-recovery"
PHASE48_ROOT = REPO_ROOT / "docs" / "demo" / "phase48-compact-intent-runtime-ablation"
PHASE47_ROOT = REPO_ROOT / "docs" / "demo" / "phase47-simulated-user-review"
MODEL_PATH = REPO_ROOT / "models" / "Qwen3-4B"
PHASE45_SOURCE = CORE_ROOT / "pfe_core" / "phase45_privacy_multiturn_preference.py"
PHASE46_SOURCE = CORE_ROOT / "pfe_core" / "phase46_runtime_first_latest_intent.py"
PHASE48_SOURCE = CORE_ROOT / "pfe_core" / "phase48_compact_intent_runtime.py"
PHASE49_SOURCE = CORE_ROOT / "pfe_core" / "phase49_provenance_boundary_recovery.py"
PHASE48_VARIANTS = ("base_privacy", "base_compact_intent", "base_full_intent")
INVALIDATED_HOLDOUT_PATH = (
    EVIDENCE_ROOT
    / "evidence-scorer-debug"
    / "attempt-01-boundary-paraphrase-gap"
    / "holdout.json"
)


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


def _phase48_snapshot() -> dict[str, Any]:
    manifest = _read_json(PHASE48_ROOT / "evidence_manifest.json")
    mismatches = []
    for item in manifest.get("files") or []:
        path = REPO_ROOT / str(item.get("path") or "")
        current = _sha256(path) if path.exists() else None
        if current != item.get("sha256"):
            mismatches.append({"path": item.get("path"), "expected": item.get("sha256"), "current": current})
    decision = _read_json(PHASE48_ROOT / "phase48-final-decision.json")
    integrity = _read_json(PHASE48_ROOT / "evidence_integrity.json")
    return {
        "kind": "phase49_phase48_canonical_snapshot",
        "passed": (
            not mismatches
            and integrity.get("passed") is True
            and decision.get("recommendation") == "hold_compact_runtime"
        ),
        "phase": "phase48",
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

    phase48 = _phase48_snapshot()
    phase48_holdout = _read_json(PHASE48_ROOT / "evidence-holdout" / "holdout.json").get("sessions") or []
    phase48_transcripts = {
        variant: _read_jsonl(PHASE48_ROOT / "evidence-real-runtime-ablation" / f"transcripts_{variant}.jsonl")
        for variant in PHASE48_VARIANTS
    }
    review = build_phase49_simulated_review(phase48_transcripts, phase48_holdout)
    reviewed_candidates = _read_jsonl(PHASE47_ROOT / "evidence-candidates" / "reviewed_candidates.jsonl")
    invalidated_holdout = _read_json(INVALIDATED_HOLDOUT_PATH).get("sessions") or []
    holdout = build_phase49_holdout_sessions()
    sessions = list(holdout["sessions"])
    split = build_phase49_split_integrity(
        sessions,
        prior_holdout_sessions=[*phase48_holdout, *invalidated_holdout],
        reviewed_candidates=reviewed_candidates,
    )
    calibration_cases = build_phase49_scorer_calibration_cases()
    calibration = evaluate_phase49_scorer_calibration(calibration_cases["cases"])
    protocol = {
        "kind": "phase49_fair_three_arm_runtime_protocol",
        "variants": ["base_privacy", "base_compact_v1", "base_compact_v2"],
        "same_qwen3_4b_base_all_arms": True,
        "adapter_loaded_any_arm": False,
        "privacy_transform_all_arms": True,
        "same_length_contract_all_arms": True,
        "length_contract": PHASE46_LENGTH_CONTRACT,
        "compact_v1_contract": PHASE48_COMPACT_INTENT_CONTRACT,
        "compact_v2_additional_clause": PHASE49_EVIDENCE_BOUNDARY_CLAUSE,
        "xml_or_tag_envelope_used_any_arm": False,
        "do_sample": False,
        "think": False,
        "input_max_length": 4096,
        "max_new_tokens": 384,
        "repetition_penalty": 1.05,
        "formal_eval_requires_all_arm_truncation_at_most_0_05": True,
    }
    protocol["protocol_sha256"] = stable_hash(protocol)
    holdout_freeze = {
        "kind": "phase49_holdout_freeze",
        "holdout_manifest_sha256": stable_hash(sessions),
        "holdout_count": len(sessions),
        "frozen_before_model_calls": True,
        "phase48_holdout_reused": False,
        "phase47_candidate_reused": False,
        "created_at": _utcnow(),
    }
    scorer_freeze = {
        "kind": "phase49_scorer_and_runtime_freeze",
        "phase45_privacy_source_sha256": _sha256(PHASE45_SOURCE),
        "phase46_generic_scorer_source_sha256": _sha256(PHASE46_SOURCE),
        "phase48_compact_v1_source_sha256": _sha256(PHASE48_SOURCE),
        "phase49_source_sha256": _sha256(PHASE49_SOURCE),
        "calibration_status": calibration.get("status"),
        "calibration_exact_label_accuracy": calibration.get("exact_label_accuracy"),
        "frozen_before_model_calls": True,
        "created_at": _utcnow(),
    }
    model = {
        "kind": "phase49_model_selection",
        "model_path": str(MODEL_PATH),
        "model_ready": MODEL_PATH.is_dir() and (MODEL_PATH / "config.json").exists(),
        "model": "Qwen3-4B",
        "adapter_loaded": False,
        "new_training": False,
        "qwen27b_training_allowed": False,
    }
    no_training = {
        "kind": "phase49_training_attempt",
        "status": "skipped_by_design",
        "new_training": False,
        "new_adapter_created": False,
        "reason": "Phase49 repairs provenance evaluation and isolates one minimal runtime clause.",
        "actual_human_review_completed": False,
        "auto_training_allowed": False,
        "auto_promotion_allowed": False,
        "hermes_attachment_allowed": False,
    }
    source_manifest = {
        "kind": "phase49_source_boundary_manifest",
        "phase48_provenance_outputs_reviewed": review.get("review_count"),
        "phase48_outputs_used_for_simulated_review_only": True,
        "phase48_outputs_used_for_training": False,
        "phase48_holdout_count": len(phase48_holdout),
        "phase48_holdout_reused": False,
        "invalidated_attempt_01_holdout_count": len(invalidated_holdout),
        "invalidated_attempt_01_holdout_reused": False,
        "phase49_holdout_count": len(sessions),
        "actual_human_review_completed": False,
        "actual_user_feedback_count": 0,
        "actual_product_benefit_claim_allowed": False,
    }
    ready = (
        phase48.get("passed") is True
        and review.get("status") == "completed"
        and int(review.get("review_count") or 0) == 24
        and split.get("passed") is True
        and calibration.get("status") == "passed"
        and float(calibration.get("exact_label_accuracy") or 0.0) == 1.0
        and model.get("model_ready") is True
        and len(sessions) == 64
    )
    preparation = {
        "kind": "phase49_preparation_decision",
        "status": "ready_for_real_runtime_ablation" if ready else "blocked",
        "phase48_frozen": phase48.get("passed") is True,
        "simulated_review_completed": review.get("status") == "completed",
        "simulated_review_not_actual_feedback": review.get("actual_human_review_completed") is False,
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

    _write_json(EVIDENCE_ROOT / "evidence-baseline" / "phase48_canonical_snapshot.json", phase48)
    _write_json(EVIDENCE_ROOT / "evidence-baseline" / "model_selection.json", model)
    _write_json(EVIDENCE_ROOT / "evidence-simulated-review" / "review_summary.json", review)
    _write_jsonl(EVIDENCE_ROOT / "evidence-simulated-review" / "review_items.jsonl", review["items"])
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
    print(
        json.dumps(
            {
                "status": preparation["status"],
                "review_count": review["review_count"],
                "review_label_counts": review["label_counts"],
                "holdout_count": len(sessions),
                "category_counts": holdout["category_counts"],
                "split_integrity_passed": split["passed"],
                "calibration_status": calibration["status"],
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
