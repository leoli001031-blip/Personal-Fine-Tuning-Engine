#!/usr/bin/env python3
"""Freeze Phase68 fresh and aligned evaluator evidence before model calls."""

from __future__ import annotations

import argparse
from collections import Counter
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
from pfe_core.phase55_atomic_boundary_composition import (
    PHASE55_CATEGORIES,
    build_phase55_holdout_cases,
)
from pfe_core.phase59_proposition_addressed_grounding import (
    PHASE59_CALIBRATION_ACCURACY_GATE,
    PHASE59_CANDIDATE_SELECTION_EXACT_GATE,
    PHASE59_HOLDOUT_ACCURACY_GATE,
    PHASE59_PER_CATEGORY_ACCURACY_GATE,
    PHASE59_PER_FIELD_ACCURACY_GATE,
    PHASE59_TYPED_EXACT_MATCH_GATE,
)
from pfe_core.phase63_field_typed_candidate_wire import (
    PHASE63_FIELD_PREFIXES,
    PHASE63_WIRE_PATTERN,
    PHASE63_WIRE_VERSION,
)
from pfe_core.phase68_aligned_candidate_scope_recovery import (
    PHASE68_CATEGORIES,
    build_phase68_blind_items,
    build_phase68_calibration_cases,
    build_phase68_candidate_audit,
    build_phase68_fixture_semantic_audit,
    build_phase68_holdout_cases,
    build_phase68_preflight_items,
    build_phase68_split_integrity,
    evaluate_phase68_hard_rule_compatibility,
)


EVIDENCE_ROOT = REPO_ROOT / "docs/demo/phase68-aligned-candidate-scope-recovery"
PHASE67_ROOT = REPO_ROOT / "docs/demo/phase67-historical-contract-compatibility-audit"
PHASE66_REPORT = REPO_ROOT / (
    "docs/demo/phase66-external-distribution-regression/"
    "evidence-historical-replay/historical_replay_report.json"
)
SOURCE_FILES = {
    "phase53_hard_detector": CORE_ROOT / "pfe_core/phase53_evaluator_scope_recovery.py",
    "phase56_grounder_composer": CORE_ROOT / "pfe_core/phase56_evidence_span_grounded_atomic.py",
    "phase58_clause_grounder": CORE_ROOT / "pfe_core/phase58_clause_addressed_grounding.py",
    "phase59_candidates": CORE_ROOT / "pfe_core/phase59_proposition_addressed_grounding.py",
    "phase62_consensus": CORE_ROOT / "pfe_core/phase62_risk_asymmetric_candidate_consensus.py",
    "phase63_wire": CORE_ROOT / "pfe_core/phase63_field_typed_candidate_wire.py",
    "phase68_core": CORE_ROOT / "pfe_core/phase68_aligned_candidate_scope_recovery.py",
    "phase68_executor": REPO_ROOT / "tools/phase68_execute.py",
}
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
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_manifest(manifest: Mapping[str, Any]) -> bool:
    files = list(manifest.get("files") or [])
    return bool(files) and all(
        (REPO_ROOT / str(row.get("path") or "")).is_file()
        and _sha256(REPO_ROOT / str(row.get("path") or "")) == row.get("sha256")
        for row in files
    )


def _phase67_snapshot() -> dict[str, Any]:
    decision = _read_json(PHASE67_ROOT / "phase67-final-decision.json")
    integrity = _read_json(PHASE67_ROOT / "evidence_integrity.json")
    partition = _read_json(PHASE67_ROOT / "historical_partition.json")
    manifest = _read_json(PHASE67_ROOT / "evidence_manifest.json")
    checks = {
        "phase67_recommendation_preserved": decision.get("recommendation")
        == "recommend_phase67_contract_aware_partition_for_manual_review_only",
        "phase67_integrity_passed": integrity.get("passed") is True,
        "only_phase55_aligned": partition.get("aligned_legacy_regression_phases")
        == ["phase55"],
        "phase67_manifest_verified": _verify_manifest(manifest),
        "runtime_remained_blocked": decision.get("runtime_ab_allowed") is False,
    }
    return {
        "kind": "phase68_phase67_canonical_snapshot",
        "passed": all(checks.values()),
        "checks": checks,
        "recommendation": decision.get("recommendation"),
        "manifest_sha256": manifest.get("manifest_sha256"),
    }


def _aggregate_phase55_failure_audit() -> dict[str, Any]:
    report = _read_json(PHASE66_REPORT)
    rows = [row for row in report.get("details") or [] if row.get("phase") == "phase55"]
    transitions = Counter(
        f"{row.get('expected_label')}->{row.get('actual_label') or 'incomplete'}"
        for row in rows
    )
    category_failures = Counter(
        f"{row.get('category')}:{row.get('expected_label')}->{row.get('actual_label') or 'incomplete'}"
        for row in rows
        if row.get("passed") is False
    )
    failed_accepts = [
        row for row in rows if row.get("expected_label") == "accept" and row.get("passed") is False
    ]
    outcome_values = Counter(
        str(dict(row.get("grounded_consensus") or {}).get("user_outcome_status") or "incomplete")
        for row in failed_accepts
    )
    relation_values = Counter(
        str(dict(row.get("grounded_consensus") or {}).get("test_to_user_outcome_relation") or "incomplete")
        for row in failed_accepts
    )
    checks = {
        "phase55_case_count_exact": len(rows) == 150,
        "phase55_accept_failure_count_exact": len(failed_accepts) == 40,
        "accept_to_edit_exact": transitions["accept->edit"] == 37,
        "accept_to_reject_exact": transitions["accept->reject"] == 3,
        "dominant_missing_outcome_observed": outcome_values["unstated"] == 31,
        "negative_relation_misread_observed": relation_values["establishes"] == 3,
        "no_individual_text_exported": True,
    }
    return {
        "kind": "phase68_phase55_aggregate_failure_audit",
        "passed": all(checks.values()),
        "checks": checks,
        "case_count": len(rows),
        "label_transition_counts": dict(sorted(transitions.items())),
        "category_failure_counts": dict(sorted(category_failures.items())),
        "failed_accept_outcome_value_counts": dict(sorted(outcome_values.items())),
        "failed_accept_relation_value_counts": dict(sorted(relation_values.items())),
        "single_correction_hypothesis": (
            "negation-first recognition for explicit outcome suspension variants and "
            "unmodalized negative test-to-outcome relations"
        ),
        "individual_failure_rows_included": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-evidence", action="store_true")
    args = parser.parse_args()
    if args.clean_evidence and EVIDENCE_ROOT.exists():
        shutil.rmtree(EVIDENCE_ROOT)

    phase67 = _phase67_snapshot()
    failure_audit = _aggregate_phase55_failure_audit()
    calibration = build_phase68_calibration_cases()
    holdout = build_phase68_holdout_cases()
    preflight = build_phase68_preflight_items()
    phase55 = build_phase55_holdout_cases()
    calibration_blind = build_phase68_blind_items(
        calibration["cases"], seed=6801, prefix="phase68-cal"
    )
    holdout_blind = build_phase68_blind_items(
        holdout["cases"], seed=6802, prefix="phase68-hold"
    )
    phase55_blind = build_phase68_blind_items(
        phase55["cases"], seed=6803, prefix="phase68-p55"
    )
    fresh_candidate_audit = build_phase68_candidate_audit(
        calibration["cases"] + holdout["cases"]
    )
    aligned_candidate_audit = build_phase68_candidate_audit(
        phase55["cases"], include_details=False, require_typed_exact=False
    )
    calibration_semantic = build_phase68_fixture_semantic_audit(calibration["cases"])
    holdout_semantic = build_phase68_fixture_semantic_audit(holdout["cases"])
    calibration_hard = evaluate_phase68_hard_rule_compatibility(calibration["cases"])
    holdout_hard = evaluate_phase68_hard_rule_compatibility(holdout["cases"])
    aligned_hard = evaluate_phase68_hard_rule_compatibility(phase55["cases"])
    split_integrity = build_phase68_split_integrity(
        calibration["cases"],
        holdout["cases"],
        preflight_items=preflight["items"],
        historical_cases=phase55["cases"],
    )

    protocol = {
        "kind": "phase68_frozen_negation_scope_protocol",
        "stages": ["preflight", "calibration", "holdout", "phase55_regression"],
        "semantic_judge_aliases": list(JUDGE_ALIASES),
        "semantic_judge_models_private": dict(JUDGE_MODELS),
        "fresh_categories": list(PHASE68_CATEGORIES),
        "aligned_phase55_categories": list(PHASE55_CATEGORIES),
        "fresh_calibration_count": calibration["case_count"],
        "fresh_holdout_count": holdout["case_count"],
        "aligned_phase55_count": phase55["case_count"],
        "fresh_calibration_accuracy_gate": PHASE59_CALIBRATION_ACCURACY_GATE,
        "fresh_holdout_accuracy_gate": PHASE59_HOLDOUT_ACCURACY_GATE,
        "fresh_per_category_accuracy_gate": PHASE59_PER_CATEGORY_ACCURACY_GATE,
        "fresh_per_field_accuracy_gate": PHASE59_PER_FIELD_ACCURACY_GATE,
        "fresh_typed_exact_gate": PHASE59_TYPED_EXACT_MATCH_GATE,
        "fresh_candidate_exact_gate": PHASE59_CANDIDATE_SELECTION_EXACT_GATE,
        "aligned_phase55_label_accuracy_gate": 0.95,
        "aligned_phase55_typed_metrics_diagnostic_only": True,
        "false_accept_gate": 0,
        "schema_failure_gate": 0,
        "candidate_conflict_gate": 0,
        "typed_wire": {
            "version": PHASE63_WIRE_VERSION,
            "pattern": PHASE63_WIRE_PATTERN.pattern,
            "field_prefixes": dict(PHASE63_FIELD_PREFIXES),
        },
        "single_structural_correction": (
            "negation_first_outcome_and_relation_candidate_scope"
        ),
        "temperature": 0,
        "think": False,
        "num_ctx": 4096,
        "num_predict": 64,
        "parallel_worker_count": 4,
        "frozen_retry_limit_per_failed_item": 2,
        "one_independent_call_per_item_per_judge": True,
        "each_scored_split_may_be_called_once": True,
        "post_model_call_tuning_allowed": False,
        "runtime_replay_allowed": False,
        "training_allowed": False,
        "hermes_attachment_allowed": False,
        "product_default_change_allowed": False,
        "auto_promote_allowed": False,
    }
    protocol["protocol_sha256"] = stable_hash(protocol)
    freeze = {
        "kind": "phase68_pre_model_call_freeze",
        "source_sha256": {name: _sha256(path) for name, path in SOURCE_FILES.items()},
        "phase67_snapshot_sha256": stable_hash(phase67),
        "aggregate_failure_audit_sha256": stable_hash(failure_audit),
        "preflight_public_sha256": stable_hash(preflight["items"]),
        "calibration_public_sha256": stable_hash(calibration_blind["public_items"]),
        "calibration_hidden_sha256": stable_hash(calibration_blind["hidden_key"]),
        "holdout_public_sha256": stable_hash(holdout_blind["public_items"]),
        "holdout_hidden_sha256": stable_hash(holdout_blind["hidden_key"]),
        "phase55_regression_public_sha256": stable_hash(phase55_blind["public_items"]),
        "phase55_regression_hidden_sha256": stable_hash(phase55_blind["hidden_key"]),
        "protocol_sha256": protocol["protocol_sha256"],
        "frozen_before_model_calls": True,
        "created_at": _utcnow(),
    }
    source_manifest = {
        "kind": "phase68_source_manifest",
        "fresh_calibration": {
            "case_count": calibration["case_count"],
            "category_counts": calibration["category_counts"],
            "label_counts": calibration["label_counts"],
        },
        "fresh_holdout": {
            "case_count": holdout["case_count"],
            "category_counts": holdout["category_counts"],
            "label_counts": holdout["label_counts"],
        },
        "aligned_phase55": {
            "case_count": phase55["case_count"],
            "category_counts": phase55["category_counts"],
            "label_counts": phase55["label_counts"],
            "label_contract_only": True,
            "individual_rows_remain_sealed_from_design": True,
        },
        "simulated_evaluator_fixture": True,
        "actual_user_feedback_count": 0,
        "not_for_training": True,
        "private_user_material_used": False,
    }
    checks = {
        "phase67_snapshot_passed": phase67["passed"],
        "aggregate_failure_audit_passed": failure_audit["passed"],
        "fresh_candidate_audit_passed": fresh_candidate_audit["status"] == "passed",
        "aligned_label_candidate_audit_passed": aligned_candidate_audit["status"]
        == "passed",
        "calibration_semantic_audit_passed": calibration_semantic["status"] == "passed",
        "holdout_semantic_audit_passed": holdout_semantic["status"] == "passed",
        "calibration_hard_rule_compatible": calibration_hard["status"] == "passed",
        "holdout_hard_rule_compatible": holdout_hard["status"] == "passed",
        "aligned_hard_rule_compatible": aligned_hard["status"] == "passed",
        "split_integrity_passed": split_integrity["passed"] is True,
    }
    preparation = {
        "kind": "phase68_preparation_decision",
        "status": "ready_for_typed_wire_preflight" if all(checks.values()) else "blocked",
        "checks": checks,
        "failed_checks": [key for key, value in checks.items() if not value],
        "post_model_call_tuning_allowed": False,
    }

    _write_json(EVIDENCE_ROOT / "evidence-baseline/phase67_canonical_snapshot.json", phase67)
    _write_json(EVIDENCE_ROOT / "aggregate_phase55_failure_audit.json", failure_audit)
    _write_json(EVIDENCE_ROOT / "candidate_correction_audit.json", fresh_candidate_audit)
    _write_json(EVIDENCE_ROOT / "aligned_phase55_candidate_audit.json", aligned_candidate_audit)
    _write_json(EVIDENCE_ROOT / "split_integrity.json", split_integrity)
    _write_json(EVIDENCE_ROOT / "evaluator_protocol.json", protocol)
    _write_json(EVIDENCE_ROOT / "pre_model_call_freeze.json", freeze)
    _write_json(EVIDENCE_ROOT / "source_manifest.json", source_manifest)
    _write_json(EVIDENCE_ROOT / "preparation_decision.json", preparation)

    preflight_dir = EVIDENCE_ROOT / "evidence-typed-wire-preflight"
    calibration_dir = EVIDENCE_ROOT / "evidence-evaluator-calibration"
    holdout_dir = EVIDENCE_ROOT / "evidence-evaluator-holdout"
    aligned_dir = EVIDENCE_ROOT / "evidence-aligned-phase55-regression"
    _write_jsonl(preflight_dir / "preflight_items_public.jsonl", preflight["items"])
    for directory, labeled, blind, semantic, hard in (
        (calibration_dir, calibration, calibration_blind, calibration_semantic, calibration_hard),
        (holdout_dir, holdout, holdout_blind, holdout_semantic, holdout_hard),
    ):
        _write_json(directory / f"{labeled['split']}_labeled.json", labeled)
        _write_jsonl(directory / "blind_items_public.jsonl", blind["public_items"])
        _write_json(directory / "blind_hidden_key.json", {"items": blind["hidden_key"]})
        _write_json(directory / "fixture_semantic_audit.json", semantic)
        _write_json(directory / "hard_rule_compatibility.json", hard)
    _write_jsonl(aligned_dir / "blind_items_public.jsonl", phase55_blind["public_items"])
    _write_json(aligned_dir / "blind_hidden_key.json", {"items": phase55_blind["hidden_key"]})
    _write_json(aligned_dir / "hard_rule_compatibility.json", aligned_hard)
    _write_json(
        EVIDENCE_ROOT / "evidence-no-runtime/runtime_status.json",
        {
            "kind": "phase68_runtime_status",
            "status": "not_allowed_in_phase68",
            "runtime_model_call_count": 0,
            "product_default_changed": False,
        },
    )
    _write_json(
        EVIDENCE_ROOT / "evidence-no-training/training_attempt.json",
        {
            "kind": "phase68_training_attempt",
            "status": "not_requested",
            "training_executed": False,
            "adapter_created": False,
            "auto_training_allowed": False,
        },
    )
    print(json.dumps(preparation, ensure_ascii=False, indent=2))
    return 0 if preparation["status"] == "ready_for_typed_wire_preflight" else 1


if __name__ == "__main__":
    raise SystemExit(main())
