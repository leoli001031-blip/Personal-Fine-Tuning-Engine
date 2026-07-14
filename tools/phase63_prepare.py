#!/usr/bin/env python3
"""Freeze Phase63 field-typed wire evidence before model calls."""

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
    build_phase63_blind_items,
    build_phase63_calibration_cases,
    build_phase63_fixture_semantic_audit,
    build_phase63_holdout_cases,
    build_phase63_preflight_items,
    build_phase63_split_integrity,
    evaluate_phase63_hard_rule_compatibility,
)


EVIDENCE_ROOT = REPO_ROOT / "docs/demo/phase63-field-typed-candidate-wire"
PHASE62_ROOT = REPO_ROOT / "docs/demo/phase62-risk-asymmetric-candidate-consensus"
PHASE53_SOURCE = CORE_ROOT / "pfe_core/phase53_evaluator_scope_recovery.py"
PHASE56_SOURCE = CORE_ROOT / "pfe_core/phase56_evidence_span_grounded_atomic.py"
PHASE58_SOURCE = CORE_ROOT / "pfe_core/phase58_clause_addressed_grounding.py"
PHASE59_SOURCE = CORE_ROOT / "pfe_core/phase59_proposition_addressed_grounding.py"
PHASE60_SOURCE = CORE_ROOT / "pfe_core/phase60_flat_schema_compatibility.py"
PHASE61_SOURCE = CORE_ROOT / "pfe_core/phase61_compact_candidate_wire_protocol.py"
PHASE62_SOURCE = CORE_ROOT / "pfe_core/phase62_risk_asymmetric_candidate_consensus.py"
PHASE63_SOURCE = CORE_ROOT / "pfe_core/phase63_field_typed_candidate_wire.py"
PHASE63_EXECUTOR = REPO_ROOT / "tools/phase63_execute.py"
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


def _verify_manifest(manifest: Mapping[str, Any]) -> bool:
    files = list(manifest.get("files") or [])
    return bool(files) and all(
        (REPO_ROOT / str(row.get("path") or "")).is_file()
        and _sha256(REPO_ROOT / str(row.get("path") or "")) == row.get("sha256")
        for row in files
    )


def _phase62_snapshot() -> dict[str, Any]:
    decision = _read_json(PHASE62_ROOT / "phase62-final-decision.json")
    preflight = _read_json(PHASE62_ROOT / "evidence-protocol-preflight/preflight_report.json")
    calibration = _read_json(PHASE62_ROOT / "evidence-evaluator-calibration/candidate_evaluator_report.json")
    failures = _read_json(PHASE62_ROOT / "evidence-evaluator-calibration/failure_analysis.json")
    integrity = _read_json(PHASE62_ROOT / "evidence_integrity.json")
    manifest = _read_json(PHASE62_ROOT / "evidence_manifest.json")
    raw_failure = _read_json(PHASE62_ROOT / "wire_protocol_failure_evidence.json")
    manifest_ok = _verify_manifest(manifest)
    passed = (
        decision.get("recommendation") == "hold_phase62_risk_asymmetric_candidate_consensus"
        and decision.get("phase63_external_replay_design_eligible") is False
        and preflight.get("status") == "passed"
        and calibration.get("status") == "not_qualified"
        and calibration.get("accuracy") == 0.9667
        and int(calibration.get("failure_count") or 0) == 1
        and int(calibration.get("schema_failure_count") or 0) == 1
        and int(calibration.get("safe_abstention_recovery_count") or 0) == 6
        and int(calibration.get("false_accept_count_on_reject_cases") or 0) == 0
        and int(failures.get("candidate_value_conflict_count") or 0) == 0
        and raw_failure.get("raw_failures_preserved_and_hashed") is True
        and integrity.get("passed") is True
        and manifest_ok
    )
    return {
        "kind": "phase63_phase62_canonical_snapshot",
        "passed": passed,
        "phase62_recommendation": decision.get("recommendation"),
        "phase62_preflight_status": preflight.get("status"),
        "phase62_calibration_status": calibration.get("status"),
        "phase62_calibration_accuracy": calibration.get("accuracy"),
        "phase62_typed_exact_match_rate": calibration.get("typed_exact_match_rate"),
        "phase62_candidate_selection_exact_match_rate": calibration.get("candidate_selection_exact_match_rate"),
        "phase62_safe_abstention_recovery_count": calibration.get("safe_abstention_recovery_count"),
        "phase62_schema_failure_count": calibration.get("schema_failure_count"),
        "phase62_false_accept_count": calibration.get("false_accept_count_on_reject_cases"),
        "phase62_manifest_sha256": manifest.get("manifest_sha256"),
        "phase62_manifest_verified": manifest_ok,
        "phase62_source_modified": False,
    }


def _historical_cases() -> list[dict[str, Any]]:
    rows = []
    for split, filename in (("calibration", "calibration_labeled.json"), ("holdout", "holdout_labeled.json")):
        rows.extend(
            dict(row)
            for row in _read_json(PHASE62_ROOT / f"evidence-evaluator-{split}/{filename}").get("cases") or []
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-evidence", action="store_true")
    args = parser.parse_args()
    if args.clean_evidence and EVIDENCE_ROOT.exists():
        shutil.rmtree(EVIDENCE_ROOT)

    baseline = _phase62_snapshot()
    historical = _historical_cases()
    preflight = build_phase63_preflight_items()
    calibration = build_phase63_calibration_cases()
    holdout = build_phase63_holdout_cases()
    calibration_audit = build_phase63_fixture_semantic_audit(calibration["cases"])
    holdout_audit = build_phase63_fixture_semantic_audit(holdout["cases"])
    split = build_phase63_split_integrity(
        calibration["cases"], holdout["cases"], preflight_items=preflight["items"], historical_cases=historical
    )
    calibration_blind = build_phase63_blind_items(
        calibration["cases"], seed=6301, prefix="phase63-calibration-blind"
    )
    holdout_blind = build_phase63_blind_items(
        holdout["cases"], seed=6302, prefix="phase63-holdout-blind"
    )
    hard_calibration = evaluate_phase63_hard_rule_compatibility(calibration["cases"])
    hard_holdout = evaluate_phase63_hard_rule_compatibility(holdout["cases"])
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
        "example": "PFE2|s001|u001|none",
        "json_schema_used": False,
        "system_message_used": False,
        "free_text_allowed": False,
    }
    consensus_spec = {
        "safe_candidate_plus_abstention": "retain_safe_candidate",
        "dangerous_candidate_from_any_judge": "retain_dangerous_candidate",
        "safe_dangerous_conflict": "dangerous_candidate_dominates",
        "all_judges_abstain": "none",
        "dangerous_values": {key: sorted(value) for key, value in PHASE62_DANGEROUS_VALUES.items()},
        "composer_input_count": 1,
        "changed_from_phase62": False,
    }
    protocol = {
        "kind": "phase63_frozen_field_typed_wire_protocol",
        "structural_change": "field_prefixed_local_candidate_ids_mapped_to_internal_ids",
        "phase62_consensus_changed": False,
        "phase62_candidate_semantics_changed": False,
        "phase62_failure_basis": {
            "schema_failure_count": baseline.get("phase62_schema_failure_count"),
            "failure_class": "global_candidate_id_shifted_across_missing_field",
            "safe_abstention_recovery_count": baseline.get("phase62_safe_abstention_recovery_count"),
            "false_accept_count": baseline.get("phase62_false_accept_count"),
            "individual_failure_rows_used_for_fixture_tuning": False,
        },
        "typed_wire_spec": typed_wire_spec,
        "consensus_spec": consensus_spec,
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
        "kind": "phase63_pre_model_call_freeze",
        "phase62_canonical_snapshot_passed": baseline["passed"],
        "phase53_hard_detector_source_sha256": _sha256(PHASE53_SOURCE),
        "phase56_composer_source_sha256": _sha256(PHASE56_SOURCE),
        "phase58_clause_grounder_source_sha256": _sha256(PHASE58_SOURCE),
        "phase59_candidate_source_sha256": _sha256(PHASE59_SOURCE),
        "phase60_flat_protocol_source_sha256": _sha256(PHASE60_SOURCE),
        "phase61_wire_source_sha256": _sha256(PHASE61_SOURCE),
        "phase62_consensus_source_sha256": _sha256(PHASE62_SOURCE),
        "phase63_typed_wire_source_sha256": _sha256(PHASE63_SOURCE),
        "phase63_executor_source_sha256": _sha256(PHASE63_EXECUTOR),
        "phase63_typed_wire_spec_sha256": stable_hash(typed_wire_spec),
        "phase63_consensus_spec_sha256": stable_hash(consensus_spec),
        "preflight_public_sha256": stable_hash(preflight["items"]),
        "calibration_public_sha256": stable_hash(calibration_blind["public_items"]),
        "calibration_hidden_sha256": stable_hash(calibration_blind["hidden_key"]),
        "holdout_public_sha256": stable_hash(holdout_blind["public_items"]),
        "holdout_hidden_sha256": stable_hash(holdout_blind["hidden_key"]),
        "calibration_audit_sha256": stable_hash(calibration_audit),
        "holdout_audit_sha256": stable_hash(holdout_audit),
        "historical_cases_sha256": stable_hash(historical),
        "protocol_sha256": protocol["protocol_sha256"],
        "frozen_before_preflight_calls": True,
        "frozen_before_calibration_calls": True,
        "frozen_before_holdout_calls": True,
        "created_at": _utcnow(),
    }
    source_manifest = {
        "kind": "phase63_source_boundary_manifest",
        "preflight_count": preflight["item_count"],
        "calibration_count": calibration["case_count"],
        "holdout_count": holdout["case_count"],
        "historical_fixture_count": len(historical),
        "simulated_evaluator_fixture": True,
        "actual_user_feedback_count": 0,
        "not_for_training": True,
        "private_user_material_used": False,
        "phase62_outputs_used_for_training": False,
        "phase62_individual_failure_rows_used_for_fixture_tuning": False,
    }
    ready = all(
        (
            baseline["passed"],
            split["passed"],
            calibration_audit["status"] == "passed",
            holdout_audit["status"] == "passed",
            hard_calibration["status"] == "passed",
            hard_holdout["status"] == "passed",
        )
    )
    preparation = {
        "kind": "phase63_preparation_decision",
        "status": "ready_for_typed_wire_preflight" if ready else "blocked",
        "phase62_canonical_snapshot_passed": baseline["passed"],
        "split_integrity_passed": split["passed"],
        "calibration_fixture_semantic_audit_passed": calibration_audit["status"] == "passed",
        "holdout_fixture_semantic_audit_passed": holdout_audit["status"] == "passed",
        "hard_calibration_compatibility_passed": hard_calibration["status"] == "passed",
        "hard_holdout_compatibility_passed": hard_holdout["status"] == "passed",
    }
    runtime = {
        "kind": "phase63_runtime_status",
        "runtime_replay_status": "not_requested_in_phase63",
        "runtime_replay_model_call_count": 0,
        "runtime_prompt_changed": False,
        "router_changed": False,
    }
    training = {
        "kind": "phase63_training_attempt",
        "status": "not_requested",
        "training_executed": False,
        "adapter_created": False,
        "auto_training_allowed": False,
    }

    preflight_dir = EVIDENCE_ROOT / "evidence-typed-wire-preflight"
    calibration_dir = EVIDENCE_ROOT / "evidence-evaluator-calibration"
    holdout_dir = EVIDENCE_ROOT / "evidence-evaluator-holdout"
    _write_json(EVIDENCE_ROOT / "evidence-baseline/phase62_canonical_snapshot.json", baseline)
    _write_jsonl(preflight_dir / "preflight_items_public.jsonl", preflight["items"])
    _write_json(calibration_dir / "calibration_labeled.json", calibration)
    _write_json(calibration_dir / "fixture_semantic_audit.json", calibration_audit)
    _write_jsonl(calibration_dir / "blind_items_public.jsonl", calibration_blind["public_items"])
    _write_json(calibration_dir / "blind_hidden_key.json", {"items": calibration_blind["hidden_key"]})
    _write_json(calibration_dir / "hard_rule_compatibility.json", hard_calibration)
    _write_json(holdout_dir / "holdout_labeled.json", holdout)
    _write_json(holdout_dir / "fixture_semantic_audit.json", holdout_audit)
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
