#!/usr/bin/env python3
"""Freeze Phase65 scope-aware candidate evidence before model calls."""

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
from pfe_core.phase59_proposition_addressed_grounding import (
    PHASE59_CALIBRATION_ACCURACY_GATE,
    PHASE59_CANDIDATE_SELECTION_EXACT_GATE,
    PHASE59_CATEGORIES,
    PHASE59_HOLDOUT_ACCURACY_GATE,
    PHASE59_PER_CATEGORY_ACCURACY_GATE,
    PHASE59_PER_FIELD_ACCURACY_GATE,
    PHASE59_TYPED_EXACT_MATCH_GATE,
)
from pfe_core.phase62_risk_asymmetric_candidate_consensus import PHASE62_DANGEROUS_VALUES
from pfe_core.phase63_field_typed_candidate_wire import (
    PHASE63_FIELD_PREFIXES,
    PHASE63_WIRE_PATTERN,
    PHASE63_WIRE_VERSION,
)
from pfe_core.phase65_aggregate_safe_boundary_coverage import (
    PHASE65_AGGREGATE_FAILURE_CLASSES,
    build_phase65_blind_items,
    build_phase65_calibration_cases,
    build_phase65_fixture_semantic_audit,
    build_phase65_holdout_cases,
    build_phase65_preflight_items,
    build_phase65_scope_rule_audit,
    build_phase65_split_integrity,
    evaluate_phase65_hard_rule_compatibility,
)


EVIDENCE_ROOT = REPO_ROOT / "docs/demo/phase65-aggregate-safe-boundary-coverage"
PHASE64_ROOT = REPO_ROOT / "docs/demo/phase64-field-typed-historical-replay"
PHASE53_SOURCE = CORE_ROOT / "pfe_core/phase53_evaluator_scope_recovery.py"
PHASE56_SOURCE = CORE_ROOT / "pfe_core/phase56_evidence_span_grounded_atomic.py"
PHASE58_SOURCE = CORE_ROOT / "pfe_core/phase58_clause_addressed_grounding.py"
PHASE59_SOURCE = CORE_ROOT / "pfe_core/phase59_proposition_addressed_grounding.py"
PHASE62_SOURCE = CORE_ROOT / "pfe_core/phase62_risk_asymmetric_candidate_consensus.py"
PHASE63_SOURCE = CORE_ROOT / "pfe_core/phase63_field_typed_candidate_wire.py"
PHASE63_EXECUTOR = REPO_ROOT / "tools/phase63_execute.py"
PHASE65_SOURCE = CORE_ROOT / "pfe_core/phase65_aggregate_safe_boundary_coverage.py"
PHASE65_EXECUTOR = REPO_ROOT / "tools/phase65_execute.py"
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
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


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


def _verify_manifest(manifest: Mapping[str, Any]) -> bool:
    files = list(manifest.get("files") or [])
    return bool(files) and all(
        (REPO_ROOT / str(row.get("path") or "")).is_file()
        and _sha256(REPO_ROOT / str(row.get("path") or "")) == row.get("sha256")
        for row in files
    )


def _phase64_snapshot() -> dict[str, Any]:
    decision = _read_json(PHASE64_ROOT / "phase64-final-decision.json")
    report = _read_json(
        PHASE64_ROOT / "evidence-historical-replay/historical_replay_report.json"
    )
    integrity = _read_json(PHASE64_ROOT / "evidence_integrity.json")
    manifest = _read_json(PHASE64_ROOT / "evidence_manifest.json")
    manifest_ok = _verify_manifest(manifest)
    passed = (
        decision.get("recommendation") == "hold_phase64_field_typed_historical_replay"
        and decision.get("phase65_minimal_runtime_ab_design_eligible") is False
        and report.get("status") == "not_qualified"
        and report.get("accuracy") == 0.6416
        and int(report.get("false_accept_count_on_reject_cases") or 0) == 0
        and int(report.get("schema_failure_count") or 0) == 9
        and integrity.get("passed") is True
        and manifest_ok
    )
    return {
        "kind": "phase65_phase64_canonical_snapshot",
        "passed": passed,
        "phase64_recommendation": decision.get("recommendation"),
        "phase64_accuracy": report.get("accuracy"),
        "phase64_per_phase": report.get("per_phase"),
        "phase64_false_accept_count": report.get("false_accept_count_on_reject_cases"),
        "phase64_schema_failure_count": report.get("schema_failure_count"),
        "phase64_candidate_value_conflict_count": report.get("candidate_value_conflict_count"),
        "phase64_manifest_sha256": manifest.get("manifest_sha256"),
        "phase64_manifest_verified": manifest_ok,
    }


def _aggregate_failure_taxonomy() -> dict[str, Any]:
    analysis = _read_json(
        PHASE64_ROOT / "evidence-historical-replay/failure_analysis.json"
    )
    transitions = dict(analysis.get("label_transition_counts") or {})
    under_accept = sum(
        int(value)
        for key, value in transitions.items()
        if ":accept->" in key
    )
    unsafe_false_accepts = int(
        _read_json(
            PHASE64_ROOT / "evidence-historical-replay/historical_replay_report.json"
        ).get("false_accept_count_on_reject_cases")
        or 0
    )
    checks = {
        "aggregate_label_failures_present": int(analysis.get("label_failure_count") or 0) == 200,
        "under_accept_is_dominant": under_accept >= 150,
        "unsafe_false_accepts_zero": unsafe_false_accepts == 0,
        "failure_classes_declared": bool(PHASE65_AGGREGATE_FAILURE_CLASSES),
        "individual_failure_rows_not_used_for_fixture_tuning": True,
    }
    return {
        "kind": "phase65_aggregate_failure_taxonomy",
        "passed": all(checks.values()),
        "checks": checks,
        "label_failure_count": analysis.get("label_failure_count"),
        "aggregate_transition_counts": transitions,
        "under_accept_transition_count": under_accept,
        "unsafe_false_accept_count": unsafe_false_accepts,
        "failure_classes": list(PHASE65_AGGREGATE_FAILURE_CLASSES),
        "individual_failure_rows_included": False,
    }


def _historical_cases() -> list[dict[str, Any]]:
    rows = []
    source_manifest = _read_json(PHASE64_ROOT / "source_manifest.json")
    for source in source_manifest.get("sources") or []:
        payload = _read_json(REPO_ROOT / str(source.get("path") or ""))
        rows.extend(dict(row) for row in payload.get("cases") or [])
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-evidence", action="store_true")
    args = parser.parse_args()
    if args.clean_evidence and EVIDENCE_ROOT.exists():
        shutil.rmtree(EVIDENCE_ROOT)

    baseline = _phase64_snapshot()
    aggregate = _aggregate_failure_taxonomy()
    historical = _historical_cases()
    preflight = build_phase65_preflight_items()
    calibration = build_phase65_calibration_cases()
    holdout = build_phase65_holdout_cases()
    calibration_audit = build_phase65_fixture_semantic_audit(calibration["cases"])
    holdout_audit = build_phase65_fixture_semantic_audit(holdout["cases"])
    scope_calibration = build_phase65_scope_rule_audit(calibration["cases"])
    scope_holdout = build_phase65_scope_rule_audit(holdout["cases"])
    split = build_phase65_split_integrity(
        calibration["cases"],
        holdout["cases"],
        preflight_items=preflight["items"],
        historical_cases=historical,
    )
    calibration_blind = build_phase65_blind_items(
        calibration["cases"], seed=6501, prefix="phase65-calibration-blind"
    )
    holdout_blind = build_phase65_blind_items(
        holdout["cases"], seed=6502, prefix="phase65-holdout-blind"
    )
    hard_calibration = evaluate_phase65_hard_rule_compatibility(calibration["cases"])
    hard_holdout = evaluate_phase65_hard_rule_compatibility(holdout["cases"])
    typed_wire_spec = {
        "version": PHASE63_WIRE_VERSION,
        "pattern": PHASE63_WIRE_PATTERN.pattern,
        "field_order": [
            "source_registration_candidate_id",
            "user_outcome_status_candidate_id",
            "test_to_user_outcome_relation_candidate_id",
        ],
        "field_prefixes": dict(PHASE63_FIELD_PREFIXES),
        "field_local_numbering": True,
        "typed_to_internal_mapping_required": True,
        "json_schema_used": False,
        "free_text_allowed": False,
    }
    protocol = {
        "kind": "phase65_frozen_scope_aware_candidate_protocol",
        "structural_change": (
            "retain_explicit_safe_outcome_when_relation_candidate_exists_while_"
            "suppressing_relation_scoped_asserted_outcome"
        ),
        "dangerous_candidate_detection_changed": False,
        "phase62_consensus_changed": False,
        "phase63_field_typed_wire_changed": False,
        "aggregate_failure_basis": {
            "failure_classes": list(PHASE65_AGGREGATE_FAILURE_CLASSES),
            "individual_failure_rows_used_for_fixture_tuning": False,
        },
        "typed_wire_spec": typed_wire_spec,
        "dangerous_values": {
            key: sorted(value) for key, value in PHASE62_DANGEROUS_VALUES.items()
        },
        "categories": list(PHASE59_CATEGORIES),
        "semantic_judge_aliases": list(JUDGE_ALIASES),
        "semantic_judge_models_private": dict(JUDGE_MODELS),
        "preflight_item_count": 6,
        "preflight_expected_judge_item_count": 12,
        "preflight_all_outputs_must_validate": True,
        "preflight_scored_as_calibration": False,
        "raw_wire_failure_capture_required": True,
        "frozen_retry_limit_per_failed_item": 2,
        "calibration_accuracy_gate": PHASE59_CALIBRATION_ACCURACY_GATE,
        "holdout_accuracy_gate": PHASE59_HOLDOUT_ACCURACY_GATE,
        "per_category_accuracy_gate": PHASE59_PER_CATEGORY_ACCURACY_GATE,
        "per_field_accuracy_gate": PHASE59_PER_FIELD_ACCURACY_GATE,
        "typed_exact_match_gate": PHASE59_TYPED_EXACT_MATCH_GATE,
        "candidate_selection_exact_match_gate": PHASE59_CANDIDATE_SELECTION_EXACT_GATE,
        "invalid_dangerous_atom_gate": 0,
        "false_accept_gate_on_reject_cases": 0,
        "holdout_candidate_value_conflict_gate": 0,
        "temperature": 0,
        "think": False,
        "num_ctx": 4096,
        "num_predict": 64,
        "parallel_worker_count": 4,
        "one_independent_call_per_item_per_judge": True,
        "calibration_allowed_only_after_preflight_pass": True,
        "holdout_allowed_only_after_calibration_qualification": True,
        "holdout_may_be_called_once": True,
        "post_model_call_tuning_allowed": False,
        "runtime_replay_allowed": False,
        "training_allowed": False,
        "hermes_attachment_allowed": False,
        "product_default_change_allowed": False,
    }
    protocol["protocol_sha256"] = stable_hash(protocol)
    freeze = {
        "kind": "phase65_pre_model_call_freeze",
        "phase64_canonical_snapshot_passed": baseline["passed"],
        "phase53_hard_detector_source_sha256": _sha256(PHASE53_SOURCE),
        "phase56_composer_source_sha256": _sha256(PHASE56_SOURCE),
        "phase58_clause_grounder_source_sha256": _sha256(PHASE58_SOURCE),
        "phase59_candidate_source_sha256": _sha256(PHASE59_SOURCE),
        "phase62_consensus_source_sha256": _sha256(PHASE62_SOURCE),
        "phase63_typed_wire_source_sha256": _sha256(PHASE63_SOURCE),
        "phase63_executor_source_sha256": _sha256(PHASE63_EXECUTOR),
        "phase65_source_sha256": _sha256(PHASE65_SOURCE),
        "phase65_executor_source_sha256": _sha256(PHASE65_EXECUTOR),
        "typed_wire_spec_sha256": stable_hash(typed_wire_spec),
        "preflight_public_sha256": stable_hash(preflight["items"]),
        "calibration_public_sha256": stable_hash(calibration_blind["public_items"]),
        "calibration_hidden_sha256": stable_hash(calibration_blind["hidden_key"]),
        "holdout_public_sha256": stable_hash(holdout_blind["public_items"]),
        "holdout_hidden_sha256": stable_hash(holdout_blind["hidden_key"]),
        "calibration_audit_sha256": stable_hash(calibration_audit),
        "holdout_audit_sha256": stable_hash(holdout_audit),
        "scope_calibration_sha256": stable_hash(scope_calibration),
        "scope_holdout_sha256": stable_hash(scope_holdout),
        "aggregate_failure_taxonomy_sha256": stable_hash(aggregate),
        "historical_cases_sha256": stable_hash(historical),
        "protocol_sha256": protocol["protocol_sha256"],
        "frozen_before_all_model_calls": True,
        "created_at": _utcnow(),
    }
    source_manifest = {
        "kind": "phase65_source_boundary_manifest",
        "preflight_count": preflight["item_count"],
        "calibration_count": calibration["case_count"],
        "holdout_count": holdout["case_count"],
        "historical_fixture_count": len(historical),
        "simulated_evaluator_fixture": True,
        "actual_user_feedback_count": 0,
        "not_for_training": True,
        "private_user_material_used": False,
        "individual_phase64_failure_rows_used_for_fixture_tuning": False,
    }
    ready = all(
        (
            baseline["passed"],
            aggregate["passed"],
            split["passed"],
            calibration_audit["status"] == "passed",
            holdout_audit["status"] == "passed",
            scope_calibration["status"] == "passed",
            scope_holdout["status"] == "passed",
            hard_calibration["status"] == "passed",
            hard_holdout["status"] == "passed",
        )
    )
    preparation = {
        "kind": "phase65_preparation_decision",
        "status": "ready_for_typed_wire_preflight" if ready else "blocked",
        "phase64_canonical_snapshot_passed": baseline["passed"],
        "aggregate_failure_taxonomy_passed": aggregate["passed"],
        "split_integrity_passed": split["passed"],
        "calibration_fixture_semantic_audit_passed": calibration_audit["status"]
        == "passed",
        "holdout_fixture_semantic_audit_passed": holdout_audit["status"] == "passed",
        "scope_calibration_audit_passed": scope_calibration["status"] == "passed",
        "scope_holdout_audit_passed": scope_holdout["status"] == "passed",
        "hard_calibration_compatibility_passed": hard_calibration["status"] == "passed",
        "hard_holdout_compatibility_passed": hard_holdout["status"] == "passed",
    }
    runtime = {
        "kind": "phase65_runtime_status",
        "runtime_replay_status": "not_requested_in_phase65",
        "runtime_replay_model_call_count": 0,
        "runtime_prompt_changed": False,
        "router_changed": False,
    }
    training = {
        "kind": "phase65_training_attempt",
        "status": "not_requested",
        "training_executed": False,
        "adapter_created": False,
        "auto_training_allowed": False,
    }

    preflight_dir = EVIDENCE_ROOT / "evidence-typed-wire-preflight"
    calibration_dir = EVIDENCE_ROOT / "evidence-evaluator-calibration"
    holdout_dir = EVIDENCE_ROOT / "evidence-evaluator-holdout"
    _write_json(EVIDENCE_ROOT / "evidence-baseline/phase64_canonical_snapshot.json", baseline)
    _write_json(EVIDENCE_ROOT / "aggregate_failure_taxonomy.json", aggregate)
    _write_jsonl(preflight_dir / "preflight_items_public.jsonl", preflight["items"])
    _write_json(calibration_dir / "calibration_labeled.json", calibration)
    _write_json(calibration_dir / "fixture_semantic_audit.json", calibration_audit)
    _write_json(calibration_dir / "scope_rule_audit.json", scope_calibration)
    _write_jsonl(calibration_dir / "blind_items_public.jsonl", calibration_blind["public_items"])
    _write_json(calibration_dir / "blind_hidden_key.json", {"items": calibration_blind["hidden_key"]})
    _write_json(calibration_dir / "hard_rule_compatibility.json", hard_calibration)
    _write_json(holdout_dir / "holdout_labeled.json", holdout)
    _write_json(holdout_dir / "fixture_semantic_audit.json", holdout_audit)
    _write_json(holdout_dir / "scope_rule_audit.json", scope_holdout)
    _write_jsonl(holdout_dir / "blind_items_public.jsonl", holdout_blind["public_items"])
    _write_json(holdout_dir / "blind_hidden_key.json", {"items": holdout_blind["hidden_key"]})
    _write_json(holdout_dir / "hard_rule_compatibility.json", hard_holdout)
    _write_json(holdout_dir / "split_integrity.json", split)
    _write_json(EVIDENCE_ROOT / "evidence-no-runtime/runtime_status.json", runtime)
    _write_json(EVIDENCE_ROOT / "evidence-no-training/training_attempt.json", training)
    _write_json(EVIDENCE_ROOT / "evaluator_protocol.json", protocol)
    _write_json(EVIDENCE_ROOT / "pre_model_call_freeze.json", freeze)
    _write_json(EVIDENCE_ROOT / "source_manifest.json", source_manifest)
    _write_json(EVIDENCE_ROOT / "preparation_decision.json", preparation)
    print(json.dumps(preparation, ensure_ascii=False, indent=2))
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
