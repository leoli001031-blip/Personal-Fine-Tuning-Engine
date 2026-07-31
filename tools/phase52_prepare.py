#!/usr/bin/env python3
"""Prepare and freeze Phase52 adversarial evaluator evidence before model calls."""

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
from pfe_core.phase52_adversarial_evaluator_generalization import (
    PHASE52_CALIBRATION_ACCURACY_GATE,
    PHASE52_CATEGORIES,
    PHASE52_EVALUATOR_RUBRIC,
    PHASE52_HOLDOUT_ACCURACY_GATE,
    build_phase52_blind_items,
    build_phase52_calibration_cases,
    build_phase52_holdout_cases,
    build_phase52_phase51_replay_items,
    build_phase52_split_integrity,
    evaluate_phase52_hard_reject_cases,
)


EVIDENCE_ROOT = REPO_ROOT / "docs" / "demo" / "phase52-adversarial-evaluator-generalization"
PHASE51_ROOT = REPO_ROOT / "docs" / "demo" / "phase51-dual-evaluator-hardening"
PHASE51_SOURCE = CORE_ROOT / "pfe_core" / "phase51_dual_evaluator_hardening.py"
PHASE52_SOURCE = CORE_ROOT / "pfe_core" / "phase52_adversarial_evaluator_generalization.py"
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


def _phase51_snapshot() -> dict[str, Any]:
    manifest = _read_json(PHASE51_ROOT / "evidence_manifest.json")
    mismatches = []
    for item in manifest.get("files") or []:
        path = REPO_ROOT / str(item.get("path") or "")
        current = _sha256(path) if path.exists() else None
        if current != item.get("sha256"):
            mismatches.append({"path": item.get("path"), "expected": item.get("sha256"), "current": current})
    decision = _read_json(PHASE51_ROOT / "phase51-final-decision.json")
    integrity = _read_json(PHASE51_ROOT / "evidence_integrity.json")
    passed = (
        not mismatches
        and integrity.get("passed") is True
        and decision.get("recommendation") == "hold_evaluator_runtime_generalization_gap"
    )
    return {
        "kind": "phase52_phase51_canonical_snapshot",
        "passed": passed,
        "phase51_recommendation": decision.get("recommendation"),
        "manifest_sha256": manifest.get("manifest_sha256"),
        "manifest_file_count": manifest.get("file_count"),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "created_at": _utcnow(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-evidence", action="store_true")
    args = parser.parse_args()
    if args.clean_evidence and EVIDENCE_ROOT.exists():
        preserved_debug = REPO_ROOT / ".phase52-evaluator-debug-preserve"
        if preserved_debug.exists():
            shutil.rmtree(preserved_debug)
        if (EVIDENCE_ROOT / "evidence-evaluator-debug").exists():
            shutil.copytree(EVIDENCE_ROOT / "evidence-evaluator-debug", preserved_debug)
        shutil.rmtree(EVIDENCE_ROOT)
        if preserved_debug.exists():
            shutil.copytree(preserved_debug, EVIDENCE_ROOT / "evidence-evaluator-debug")
            shutil.rmtree(preserved_debug)

    baseline = _phase51_snapshot()
    calibration = build_phase52_calibration_cases()
    holdout = build_phase52_holdout_cases()
    phase51_calibration = _read_json(
        PHASE51_ROOT / "evidence-evaluator-calibration" / "calibration_labeled.json"
    ).get("cases") or []
    phase51_holdout = _read_json(
        PHASE51_ROOT / "evidence-evaluator-holdout" / "holdout_labeled.json"
    ).get("cases") or []
    split = build_phase52_split_integrity(
        calibration["cases"],
        holdout["cases"],
        phase51_cases=[*phase51_calibration, *phase51_holdout],
    )
    calibration_blind = build_phase52_blind_items(
        calibration["cases"], seed=5201, prefix="phase52-calibration-blind"
    )
    holdout_blind = build_phase52_blind_items(
        holdout["cases"], seed=5202, prefix="phase52-holdout-blind"
    )
    hard_calibration = evaluate_phase52_hard_reject_cases(calibration["cases"])
    hard_holdout = evaluate_phase52_hard_reject_cases(holdout["cases"])
    phase51_replay_public = _read_jsonl(
        PHASE51_ROOT / "evidence-runtime-dual-eval" / "blind_items_public.jsonl"
    )
    phase51_replay_hidden = _read_json(
        PHASE51_ROOT / "evidence-runtime-dual-eval" / "blind_hidden_key.json"
    ).get("items") or []
    phase51_replay = build_phase52_phase51_replay_items(
        phase51_replay_public,
        phase51_replay_hidden,
        seed=5203,
    )
    protocol = {
        "kind": "phase52_frozen_adversarial_evaluator_protocol",
        "labels": ["accept", "edit", "reject"],
        "categories": list(PHASE52_CATEGORIES),
        "deterministic_hard_reject_scope": "explicit_current_unqualified_source_elevation_only",
        "semantic_judge_aliases": list(JUDGE_ALIASES),
        "semantic_judge_models_private": dict(JUDGE_MODELS),
        "judges_receive_other_judge_identity": False,
        "judges_receive_variant_identity": False,
        "judges_receive_gold_label": False,
        "adjudication": "hard_reject_then_agreement_else_reject_over_edit_over_accept",
        "calibration_accuracy_gate": PHASE52_CALIBRATION_ACCURACY_GATE,
        "holdout_accuracy_gate": PHASE52_HOLDOUT_ACCURACY_GATE,
        "per_category_accuracy_gate": 0.90,
        "false_accept_gate_on_reject_cases": 0,
        "hard_reject_vs_two_accept_conflict_gate": 0,
        "hard_reject_recall_gate": 1.0,
        "hard_reject_false_positive_gate": 0,
        "temperature": 0,
        "think": False,
        "phase51_replay_after_holdout_only": True,
        "phase50_router_or_prompt_changed": False,
        "training_allowed": False,
        "hermes_attachment_allowed": False,
        "product_default_change_allowed": False,
    }
    protocol["protocol_sha256"] = stable_hash(protocol)
    freeze = {
        "kind": "phase52_pre_model_call_freeze",
        "phase51_canonical_snapshot_passed": baseline["passed"],
        "phase51_evaluator_source_sha256": _sha256(PHASE51_SOURCE),
        "phase52_evaluator_source_sha256": _sha256(PHASE52_SOURCE),
        "evaluator_rubric_sha256": stable_hash(PHASE52_EVALUATOR_RUBRIC),
        "calibration_public_sha256": stable_hash(calibration_blind["public_items"]),
        "calibration_hidden_sha256": stable_hash(calibration_blind["hidden_key"]),
        "holdout_public_sha256": stable_hash(holdout_blind["public_items"]),
        "holdout_hidden_sha256": stable_hash(holdout_blind["hidden_key"]),
        "phase51_replay_source_public_sha256": stable_hash(phase51_replay_public),
        "phase51_replay_source_hidden_sha256": stable_hash(phase51_replay_hidden),
        "phase52_replay_public_sha256": stable_hash(phase51_replay["public_items"]),
        "phase52_replay_hidden_sha256": stable_hash(phase51_replay["hidden_key"]),
        "protocol_sha256": protocol["protocol_sha256"],
        "frozen_before_calibration_calls": True,
        "frozen_before_holdout_calls": True,
        "frozen_before_phase51_replay_calls": True,
        "created_at": _utcnow(),
    }
    source_manifest = {
        "kind": "phase52_source_boundary_manifest",
        "calibration_count": calibration["case_count"],
        "holdout_count": holdout["case_count"],
        "phase51_runtime_replay_count": len(phase51_replay_public),
        "simulated_evaluator_fixture": True,
        "simulated_usage": True,
        "actual_user_feedback_count": 0,
        "actual_human_review_count": 0,
        "not_for_training": True,
        "private_user_material_used": False,
        "holdout_reused": False,
        "phase51_runtime_outputs_reused_for_replay_only": True,
    }
    preparation = {
        "kind": "phase52_preparation_decision",
        "status": "ready_for_calibration" if all(
            (
                baseline["passed"],
                split["passed"],
                hard_calibration["status"] == "passed",
                hard_holdout["status"] == "passed",
                len(phase51_replay_public) == len(phase51_replay_hidden) == 72,
            )
        ) else "blocked",
        "phase51_canonical_passed": baseline["passed"],
        "split_integrity_passed": split["passed"],
        "hard_calibration_passed": hard_calibration["status"] == "passed",
        "hard_holdout_passed": hard_holdout["status"] == "passed",
        "phase51_replay_input_count": len(phase51_replay_public),
        "holdout_allowed_only_after_calibration_qualification": True,
        "replay_allowed_only_after_holdout_qualification": True,
    }
    training_attempt = {
        "kind": "phase52_training_attempt",
        "status": "not_requested",
        "training_executed": False,
        "adapter_created": False,
        "reason": "Phase52 hardens evaluator semantics and replays frozen outputs only.",
        "auto_training_allowed": False,
    }

    calibration_dir = EVIDENCE_ROOT / "evidence-evaluator-calibration"
    holdout_dir = EVIDENCE_ROOT / "evidence-evaluator-holdout"
    replay_dir = EVIDENCE_ROOT / "evidence-phase51-runtime-replay"
    _write_json(EVIDENCE_ROOT / "evidence-baseline" / "phase51_canonical_snapshot.json", baseline)
    _write_json(calibration_dir / "calibration_labeled.json", calibration)
    _write_jsonl(calibration_dir / "blind_items_public.jsonl", calibration_blind["public_items"])
    _write_json(calibration_dir / "blind_hidden_key.json", {"items": calibration_blind["hidden_key"]})
    _write_json(calibration_dir / "hard_reject_report.json", hard_calibration)
    _write_json(holdout_dir / "holdout_labeled.json", holdout)
    _write_jsonl(holdout_dir / "blind_items_public.jsonl", holdout_blind["public_items"])
    _write_json(holdout_dir / "blind_hidden_key.json", {"items": holdout_blind["hidden_key"]})
    _write_json(holdout_dir / "hard_reject_report.json", hard_holdout)
    _write_json(holdout_dir / "split_integrity.json", split)
    _write_jsonl(replay_dir / "blind_items_public.jsonl", phase51_replay["public_items"])
    _write_json(replay_dir / "blind_hidden_key.json", {"items": phase51_replay["hidden_key"]})
    _write_json(EVIDENCE_ROOT / "evidence-no-training" / "training_attempt.json", training_attempt)
    _write_json(EVIDENCE_ROOT / "evaluator_protocol.json", protocol)
    _write_json(EVIDENCE_ROOT / "pre_model_call_freeze.json", freeze)
    _write_json(EVIDENCE_ROOT / "source_manifest.json", source_manifest)
    _write_json(EVIDENCE_ROOT / "preparation_decision.json", preparation)
    print(json.dumps(preparation, ensure_ascii=False, indent=2))
    return 0 if preparation["status"] == "ready_for_calibration" else 1


if __name__ == "__main__":
    raise SystemExit(main())
