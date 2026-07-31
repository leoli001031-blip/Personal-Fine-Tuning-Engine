#!/usr/bin/env python3
"""Freeze Phase45 and prepare the Phase46 runtime-first ablation."""

from __future__ import annotations

import argparse
from collections import defaultdict
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

from pfe_core.phase46_runtime_first_latest_intent import (
    PHASE46_LATEST_INTENT_CONTRACT,
    PHASE46_LENGTH_CONTRACT,
    build_phase46_curated_candidates,
    build_phase46_holdout_sessions,
    build_phase46_scorer_calibration_cases,
    build_phase46_split_integrity,
    evaluate_phase46_scorer_calibration,
    stable_hash,
)


PHASE45_ROOT = REPO_ROOT / "docs" / "demo" / "phase45-privacy-structural-multiturn-preference"
EVIDENCE_ROOT = REPO_ROOT / "docs" / "demo" / "phase46-runtime-first-latest-intent-ablation"
MODEL_PATH = REPO_ROOT / "models" / "Qwen3-4B"
SCORER_SOURCE = CORE_ROOT / "pfe_core" / "phase46_runtime_first_latest_intent.py"


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return dict(value) if isinstance(value, Mapping) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _phase45_manifest() -> dict[str, Any]:
    files = []
    for path in sorted(PHASE45_ROOT.rglob("*")):
        if path.is_file():
            files.append({"path": str(path.relative_to(REPO_ROOT)), "size_bytes": path.stat().st_size, "sha256": _sha256(path)})
    decision = _read_json(PHASE45_ROOT / "phase45-final-decision.json")
    return {
        "kind": "phase46_frozen_phase45_canonical_manifest",
        "created_at": _utcnow(),
        "phase45_root": str(PHASE45_ROOT),
        "file_count": len(files),
        "files": files,
        "manifest_sha256": stable_hash(files),
        "phase45_commit": "6294d83",
        "phase45_pr_number": 56,
        "phase45_adapter_sha256": decision.get("selected_adapter_sha256"),
        "phase45_recommendation": decision.get("recommendation"),
        "phase45_archive_preserved": decision.get("recommendation") == "archive",
        "phase45_canonical_evidence_modified": False,
    }


def _phase45_failure_analysis() -> dict[str, Any]:
    real = PHASE45_ROOT / "evidence-holdout" / "real-80-session"
    metrics = {name: _read_json(real / f"metrics_{name}.json") for name in ("base_privacy", "adapter_privacy")}
    category_metrics: dict[str, dict[str, Any]] = {}
    for variant, payload in metrics.items():
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in payload.get("details") or []:
            grouped[str(row.get("category") or "")].append(dict(row.get("scores") or {}))
        category_metrics[variant] = {}
        for category, rows in sorted(grouped.items()):
            category_metrics[variant][category] = {
                "count": len(rows),
                "score": round(sum(float(row.get("composite_preference_score") or 0.0) for row in rows) / len(rows), 4),
                "latest_intent": round(sum(float(row.get("follows_latest_user_intent") or 0.0) for row in rows) / len(rows), 4),
                "correction": round(sum(float(row.get("correction_responsiveness") or 0.0) for row in rows) / len(rows), 4),
                "repetition": round(sum(float(row.get("repetition_rate") or 0.0) for row in rows) / len(rows), 4),
            }
    return {
        "kind": "phase46_phase45_failure_analysis",
        "evidence_basis": "Phase45 frozen 80-session real Qwen A/B/C/D outputs",
        "category_metrics": category_metrics,
        "finding": (
            "The explicit latest_goal category was already perfect. Regression concentrated in ordinary tasks, failure handling, "
            "Git/process truthfulness, evidence status, and provenance where the latest request must override a category-specific old goal."
        ),
        "phase46_hypothesis": (
            "A privacy-safe runtime envelope that marks the last user turn as the only current intent may improve cross-category correction "
            "without another adapter; if it does, training remains blocked until actual manual review of heterogeneous candidates."
        ),
        "new_training_justified": False,
    }


def _model_selection() -> dict[str, Any]:
    selection = _read_json(PHASE45_ROOT / "evidence-diagnostic" / "candidate_selection.json")
    adapter_path = Path(str(selection.get("selected_adapter_path") or ""))
    model_ready = (MODEL_PATH / "config.json").exists() and len(list(MODEL_PATH.glob("*.safetensors"))) == 3
    adapter_ready = adapter_path.is_dir() and (adapter_path / "adapter_model.safetensors").exists()
    return {
        "kind": "phase46_model_and_archived_adapter_selection",
        "status": "ready" if model_ready and adapter_ready else "blocked",
        "base_model": str(MODEL_PATH),
        "base_model_ready": model_ready,
        "archived_adapter_path": str(adapter_path),
        "archived_adapter_sha256": selection.get("selected_adapter_sha256"),
        "archived_adapter_ready_for_eval_only": adapter_ready,
        "archived_adapter_training_or_promotion_allowed": False,
        "new_training_requested": False,
        "qwen27b_training_allowed": False,
        "dpo_allowed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-evidence", action="store_true")
    args = parser.parse_args()
    if args.clean_evidence and EVIDENCE_ROOT.exists():
        shutil.rmtree(EVIDENCE_ROOT)
    for name in (
        "evidence-baseline",
        "evidence-failure-analysis",
        "evidence-curated-candidates",
        "evidence-scorer-calibration",
        "evidence-holdout",
        "evidence-real-runtime-ablation",
        "evidence-blind-eval",
        "evidence-no-training",
    ):
        (EVIDENCE_ROOT / name).mkdir(parents=True, exist_ok=True)

    phase45_manifest = _phase45_manifest()
    failure = _phase45_failure_analysis()
    candidates = build_phase46_curated_candidates()
    holdout = build_phase46_holdout_sessions()
    phase45_holdout = _read_json(PHASE45_ROOT / "evidence-holdout" / "holdout.json")
    split = build_phase46_split_integrity(
        candidates["candidates"],
        holdout["sessions"],
        phase45_holdout_sessions=phase45_holdout.get("sessions") or [],
    )
    calibration_cases = build_phase46_scorer_calibration_cases()
    calibration = evaluate_phase46_scorer_calibration(calibration_cases["cases"])
    scorer_freeze = {
        "kind": "phase46_scorer_freeze",
        "frozen_at": _utcnow(),
        "frozen_before_phase46_model_calls": True,
        "source_path": str(SCORER_SOURCE),
        "source_sha256": _sha256(SCORER_SOURCE),
        "calibration_manifest_sha256": calibration_cases["manifest_sha256"],
        "calibration_status": calibration["status"],
        "changes_after_model_calls_allowed": False,
    }
    protocol = {
        "kind": "phase46_fair_runtime_ablation_protocol",
        "max_new_tokens": 384,
        "input_max_length": 4096,
        "do_sample": False,
        "repetition_penalty": 1.05,
        "think": False,
        "length_contract": PHASE46_LENGTH_CONTRACT,
        "latest_intent_contract": PHASE46_LATEST_INTENT_CONTRACT,
        "same_length_contract_all_arms": True,
        "privacy_transform_all_arms": True,
        "formal_eval_requires_all_arm_truncation_at_most_0_05": True,
    }
    protocol["protocol_sha256"] = stable_hash(protocol)
    model = _model_selection()

    _write_json(EVIDENCE_ROOT / "evidence-baseline" / "phase45_canonical_manifest.json", phase45_manifest)
    _write_json(EVIDENCE_ROOT / "evidence-baseline" / "model_selection.json", model)
    _write_json(EVIDENCE_ROOT / "evidence-failure-analysis" / "phase45_category_failure_analysis.json", failure)
    _write_json(EVIDENCE_ROOT / "evidence-curated-candidates" / "candidate_manifest.json", {key: value for key, value in candidates.items() if key != "candidates"})
    _write_json(EVIDENCE_ROOT / "evidence-curated-candidates" / "candidate_audit.json", candidates["audit"])
    _write_jsonl(EVIDENCE_ROOT / "evidence-curated-candidates" / "simulated_review_candidates.jsonl", candidates["candidates"])
    _write_json(EVIDENCE_ROOT / "evidence-scorer-calibration" / "calibration_cases.json", calibration_cases)
    _write_json(EVIDENCE_ROOT / "evidence-scorer-calibration" / "calibration_report.json", calibration)
    _write_json(EVIDENCE_ROOT / "evidence-scorer-calibration" / "scorer_freeze.json", scorer_freeze)
    _write_json(EVIDENCE_ROOT / "evidence-holdout" / "holdout.json", holdout)
    _write_json(EVIDENCE_ROOT / "evidence-holdout" / "holdout_freeze.json", {
        "kind": "phase46_holdout_freeze",
        "frozen_at": _utcnow(),
        "frozen_before_model_calls": True,
        "holdout_manifest_sha256": holdout["manifest_sha256"],
        "session_count": holdout["holdout_count"],
        "not_for_training": True,
    })
    _write_json(EVIDENCE_ROOT / "evidence-holdout" / "split_integrity.json", split)
    _write_json(EVIDENCE_ROOT / "evidence-holdout" / "source_manifest.json", {
        "kind": "phase46_holdout_source_manifest",
        "source": "deterministic_fresh_simulated_usage",
        "actual_user_feedback": False,
        "not_for_training": True,
        "synthetic_privacy_canaries_only": True,
        "session_count": holdout["holdout_count"],
        "category_counts": holdout["category_counts"],
    })
    _write_json(EVIDENCE_ROOT / "evidence-holdout" / "runtime_protocol.json", protocol)
    _write_json(EVIDENCE_ROOT / "evidence-no-training" / "training_attempt.json", {
        "kind": "phase46_training_attempt",
        "status": "skipped_by_design",
        "new_training": False,
        "new_adapter_created": False,
        "reason": "Phase46 tests runtime-first latest-intent handling before any new training.",
        "candidate_count": candidates["candidate_count"],
        "candidate_training_eligible_count": 0,
        "blocker": "pending_actual_manual_user_review",
        "auto_training_allowed": False,
        "auto_promotion_allowed": False,
    })

    ready = (
        phase45_manifest["phase45_archive_preserved"] is True
        and candidates["audit"]["passed"] is True
        and candidates["eligible_for_training"] is False
        and split["passed"] is True
        and calibration["status"] == "passed"
        and model["status"] == "ready"
    )
    preparation = {
        "kind": "phase46_preparation_decision",
        "status": "ready_for_real_runtime_ablation" if ready else "blocked",
        "phase45_frozen": True,
        "phase45_archive_preserved": phase45_manifest["phase45_archive_preserved"],
        "curated_candidate_count": candidates["candidate_count"],
        "actual_human_review_completed": False,
        "candidate_training_allowed": False,
        "holdout_count": holdout["holdout_count"],
        "split_integrity_passed": split["passed"],
        "calibration_status": calibration["status"],
        "scorer_frozen_before_model_calls": True,
        "next_action": "run_base_privacy_base_privacy_intent_archived_adapter_intent" if ready else "repair_preparation_gate",
        "actual_product_benefit_claim_allowed": False,
        "auto_promotion_allowed": False,
    }
    _write_json(EVIDENCE_ROOT / "preparation_decision.json", preparation)
    print(json.dumps(preparation, ensure_ascii=False, indent=2))
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
