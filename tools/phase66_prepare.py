#!/usr/bin/env python3
"""Freeze Phase66 external-regression evidence before any model calls."""

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
from pfe_core.phase64_field_typed_historical_replay import (
    PHASE64_OVERALL_ACCURACY_GATE,
    PHASE64_PER_CATEGORY_ACCURACY_GATE,
    PHASE64_PER_PHASE_ACCURACY_GATE,
    PHASE64_PHASES,
)
from pfe_core.phase66_external_distribution_regression import (
    PHASE66_MATERIAL_ACCURACY_DELTA_GATE,
    build_phase66_external_blind_items,
    build_phase66_external_fixture_semantic_audit,
    build_phase66_external_holdout_cases,
    build_phase66_external_integrity,
    build_phase66_historical_blind_replay,
    build_phase66_historical_integrity,
    build_phase66_preflight_items,
    evaluate_phase66_external_hard_rule_compatibility,
)


EVIDENCE_ROOT = REPO_ROOT / "docs/demo/phase66-external-distribution-regression"
PHASE64_ROOT = REPO_ROOT / "docs/demo/phase64-field-typed-historical-replay"
PHASE65_ROOT = REPO_ROOT / "docs/demo/phase65-aggregate-safe-boundary-coverage"
HISTORICAL_ROOTS = {
    "phase51": REPO_ROOT / "docs/demo/phase51-dual-evaluator-hardening",
    "phase52": REPO_ROOT / "docs/demo/phase52-adversarial-evaluator-generalization",
    "phase53": REPO_ROOT / "docs/demo/phase53-evaluator-scope-recovery",
    "phase54": REPO_ROOT / "docs/demo/phase54-typed-proposition-evaluator",
    "phase55": REPO_ROOT / "docs/demo/phase55-atomic-boundary-composition",
}
PHASE53_SOURCE = CORE_ROOT / "pfe_core/phase53_evaluator_scope_recovery.py"
PHASE56_SOURCE = CORE_ROOT / "pfe_core/phase56_evidence_span_grounded_atomic.py"
PHASE58_SOURCE = CORE_ROOT / "pfe_core/phase58_clause_addressed_grounding.py"
PHASE59_SOURCE = CORE_ROOT / "pfe_core/phase59_proposition_addressed_grounding.py"
PHASE62_SOURCE = CORE_ROOT / "pfe_core/phase62_risk_asymmetric_candidate_consensus.py"
PHASE63_SOURCE = CORE_ROOT / "pfe_core/phase63_field_typed_candidate_wire.py"
PHASE63_EXECUTOR = REPO_ROOT / "tools/phase63_execute.py"
PHASE65_SOURCE = CORE_ROOT / "pfe_core/phase65_aggregate_safe_boundary_coverage.py"
PHASE66_SOURCE = CORE_ROOT / "pfe_core/phase66_external_distribution_regression.py"
PHASE66_EXECUTOR = REPO_ROOT / "tools/phase66_execute.py"
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


def _phase65_snapshot() -> dict[str, Any]:
    decision = _read_json(PHASE65_ROOT / "phase65-final-decision.json")
    holdout = _read_json(
        PHASE65_ROOT / "evidence-evaluator-holdout/candidate_evaluator_report.json"
    )
    integrity = _read_json(PHASE65_ROOT / "evidence_integrity.json")
    manifest = _read_json(PHASE65_ROOT / "evidence_manifest.json")
    manifest_ok = _verify_manifest(manifest)
    passed = (
        decision.get("recommendation")
        == "recommend_phase65_scope_aware_candidates_for_manual_review_only"
        and decision.get("phase66_external_regression_design_eligible") is True
        and holdout.get("status") == "qualified"
        and holdout.get("accuracy") == 1.0
        and int(holdout.get("false_accept_count_on_reject_cases") or 0) == 0
        and int(holdout.get("schema_failure_count") or 0) == 0
        and int(holdout.get("candidate_value_conflict_count") or 0) == 0
        and integrity.get("passed") is True
        and manifest_ok
    )
    return {
        "kind": "phase66_phase65_canonical_snapshot",
        "passed": passed,
        "phase65_recommendation": decision.get("recommendation"),
        "phase65_holdout_accuracy": holdout.get("accuracy"),
        "phase65_manifest_sha256": manifest.get("manifest_sha256"),
        "phase65_manifest_verified": manifest_ok,
        "phase66_design_eligible": decision.get(
            "phase66_external_regression_design_eligible"
        ),
    }


def _phase64_baseline() -> dict[str, Any]:
    report = _read_json(
        PHASE64_ROOT / "evidence-historical-replay/historical_replay_report.json"
    )
    decision = _read_json(PHASE64_ROOT / "phase64-final-decision.json")
    passed = (
        report.get("status") == "not_qualified"
        and report.get("accuracy") == 0.6416
        and decision.get("recommendation")
        == "hold_phase64_field_typed_historical_replay"
    )
    return {
        "kind": "phase66_phase64_historical_baseline",
        "passed": passed,
        "accuracy": report.get("accuracy"),
        "per_phase": report.get("per_phase"),
        "false_accept_count": report.get("false_accept_count_on_reject_cases"),
        "schema_failure_count": report.get("schema_failure_count"),
        "candidate_value_conflict_count": report.get(
            "candidate_value_conflict_count"
        ),
        "recommendation": decision.get("recommendation"),
    }


def _historical_cases() -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    by_phase = {}
    sources = []
    for phase in PHASE64_PHASES:
        path = (
            HISTORICAL_ROOTS[phase]
            / "evidence-evaluator-holdout/holdout_labeled.json"
        )
        payload = _read_json(path)
        by_phase[phase] = [dict(row) for row in payload.get("cases") or []]
        sources.append(
            {
                "phase": phase,
                "path": str(path.relative_to(REPO_ROOT)),
                "sha256": _sha256(path),
                "case_count": len(by_phase[phase]),
                "label_counts": payload.get("label_counts"),
            }
        )
    return by_phase, sources


def _phase65_cases() -> list[dict[str, Any]]:
    rows = []
    for split in ("calibration", "holdout"):
        payload = _read_json(
            PHASE65_ROOT / f"evidence-evaluator-{split}/{split}_labeled.json"
        )
        rows.extend(dict(row) for row in payload.get("cases") or [])
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-evidence", action="store_true")
    args = parser.parse_args()
    if args.clean_evidence and EVIDENCE_ROOT.exists():
        shutil.rmtree(EVIDENCE_ROOT)

    phase65 = _phase65_snapshot()
    phase64 = _phase64_baseline()
    historical, historical_sources = _historical_cases()
    historical_flat = [row for rows in historical.values() for row in rows]
    external = build_phase66_external_holdout_cases()
    preflight = build_phase66_preflight_items()
    external_blind = build_phase66_external_blind_items(external["cases"], seed=6601)
    historical_blind = build_phase66_historical_blind_replay(historical, seed=6602)
    external_audit = build_phase66_external_fixture_semantic_audit(external["cases"])
    external_hard = evaluate_phase66_external_hard_rule_compatibility(
        external["cases"]
    )
    external_integrity = build_phase66_external_integrity(
        external["cases"],
        historical_cases=historical_flat,
        phase65_cases=_phase65_cases(),
        preflight_items=preflight["items"],
    )
    historical_integrity = build_phase66_historical_integrity(
        historical_cases=historical,
        public_items=historical_blind["public_items"],
        hidden_key=historical_blind["hidden_key"],
    )

    protocol = {
        "kind": "phase66_frozen_external_distribution_protocol",
        "evaluator_under_test": "phase65_scope_aware_candidates",
        "stages": ["preflight", "external_holdout", "historical_replay"],
        "semantic_judge_aliases": list(JUDGE_ALIASES),
        "semantic_judge_models_private": dict(JUDGE_MODELS),
        "fresh_external_holdout_count": len(external["cases"]),
        "historical_replay_count": len(historical_blind["public_items"]),
        "historical_phases": list(PHASE64_PHASES),
        "judges_receive_gold_or_phase_identity": False,
        "phase65_candidate_rule_unchanged": True,
        "phase63_field_typed_wire_unchanged": True,
        "phase62_risk_asymmetric_consensus_unchanged": True,
        "phase56_deterministic_composer_unchanged": True,
        "typed_wire_spec": {
            "version": PHASE63_WIRE_VERSION,
            "pattern": PHASE63_WIRE_PATTERN.pattern,
            "field_prefixes": dict(PHASE63_FIELD_PREFIXES),
            "field_local_numbering": True,
            "free_text_allowed": False,
        },
        "external_holdout_accuracy_gate": PHASE59_HOLDOUT_ACCURACY_GATE,
        "external_per_category_accuracy_gate": PHASE59_PER_CATEGORY_ACCURACY_GATE,
        "external_per_field_accuracy_gate": PHASE59_PER_FIELD_ACCURACY_GATE,
        "external_typed_exact_gate": PHASE59_TYPED_EXACT_MATCH_GATE,
        "external_candidate_exact_gate": PHASE59_CANDIDATE_SELECTION_EXACT_GATE,
        "historical_overall_accuracy_gate": PHASE64_OVERALL_ACCURACY_GATE,
        "historical_per_phase_accuracy_gate": PHASE64_PER_PHASE_ACCURACY_GATE,
        "historical_per_category_accuracy_gate": PHASE64_PER_CATEGORY_ACCURACY_GATE,
        "material_accuracy_delta_from_phase64_gate": (
            PHASE66_MATERIAL_ACCURACY_DELTA_GATE
        ),
        "schema_failure_gate": 0,
        "false_accept_gate": 0,
        "candidate_value_conflict_gate": 0,
        "temperature": 0,
        "think": False,
        "num_ctx": 4096,
        "num_predict": 64,
        "parallel_worker_count": 4,
        "frozen_retry_limit_per_failed_item": 2,
        "one_independent_call_per_item_per_judge": True,
        "external_holdout_allowed_only_after_preflight_pass": True,
        "historical_replay_allowed_only_after_external_qualification": True,
        "each_scored_split_may_be_called_once": True,
        "resume_allowed_only_after_interruption": True,
        "post_model_call_tuning_allowed": False,
        "runtime_replay_allowed": False,
        "training_allowed": False,
        "hermes_attachment_allowed": False,
        "product_default_change_allowed": False,
        "auto_promote_allowed": False,
    }
    protocol["protocol_sha256"] = stable_hash(protocol)
    freeze = {
        "kind": "phase66_pre_model_call_freeze",
        "phase65_canonical_snapshot_passed": phase65["passed"],
        "phase64_baseline_passed": phase64["passed"],
        "phase53_hard_detector_source_sha256": _sha256(PHASE53_SOURCE),
        "phase56_composer_source_sha256": _sha256(PHASE56_SOURCE),
        "phase58_clause_grounder_source_sha256": _sha256(PHASE58_SOURCE),
        "phase59_candidate_source_sha256": _sha256(PHASE59_SOURCE),
        "phase62_consensus_source_sha256": _sha256(PHASE62_SOURCE),
        "phase63_typed_wire_source_sha256": _sha256(PHASE63_SOURCE),
        "phase63_executor_source_sha256": _sha256(PHASE63_EXECUTOR),
        "phase65_source_sha256": _sha256(PHASE65_SOURCE),
        "phase66_source_sha256": _sha256(PHASE66_SOURCE),
        "phase66_executor_source_sha256": _sha256(PHASE66_EXECUTOR),
        "preflight_public_sha256": stable_hash(preflight["items"]),
        "external_holdout_public_sha256": stable_hash(
            external_blind["public_items"]
        ),
        "external_holdout_hidden_sha256": stable_hash(
            external_blind["hidden_key"]
        ),
        "historical_replay_public_sha256": stable_hash(
            historical_blind["public_items"]
        ),
        "historical_replay_hidden_sha256": stable_hash(
            historical_blind["hidden_key"]
        ),
        "historical_sources_sha256": stable_hash(historical_sources),
        "protocol_sha256": protocol["protocol_sha256"],
        "frozen_before_model_calls": True,
        "created_at": _utcnow(),
    }
    source_manifest = {
        "kind": "phase66_external_distribution_source_manifest",
        "fresh_external_holdout": {
            "kind": external["kind"],
            "case_count": external["case_count"],
            "category_counts": external["category_counts"],
            "label_counts": external["label_counts"],
            "generated_before_calls": True,
        },
        "historical_sources": historical_sources,
        "historical_replay_count": len(historical_blind["public_items"]),
        "simulated_evaluator_fixture": True,
        "actual_user_feedback_count": 0,
        "not_for_training": True,
        "private_user_material_used": False,
    }
    preparation_checks = {
        "phase65_canonical_snapshot_passed": phase65["passed"],
        "phase64_baseline_passed": phase64["passed"],
        "external_integrity_passed": external_integrity["passed"],
        "historical_integrity_passed": historical_integrity["passed"],
        "external_fixture_semantic_audit_passed": external_audit.get("status")
        == "passed",
        "external_hard_rule_compatibility_passed": external_hard.get("status")
        == "passed",
    }
    preparation = {
        "kind": "phase66_preparation_decision",
        "status": "ready_for_typed_wire_preflight"
        if all(preparation_checks.values())
        else "blocked",
        "checks": preparation_checks,
        "failed_checks": [
            key for key, value in preparation_checks.items() if not value
        ],
        "post_model_call_tuning_allowed": False,
    }

    preflight_dir = EVIDENCE_ROOT / "evidence-typed-wire-preflight"
    external_dir = EVIDENCE_ROOT / "evidence-external-holdout"
    historical_dir = EVIDENCE_ROOT / "evidence-historical-replay"
    _write_json(EVIDENCE_ROOT / "evidence-baseline/phase65_canonical_snapshot.json", phase65)
    _write_json(EVIDENCE_ROOT / "evidence-baseline/phase64_historical_baseline.json", phase64)
    _write_jsonl(preflight_dir / "preflight_items_public.jsonl", preflight["items"])
    _write_json(external_dir / "holdout_labeled.json", external)
    _write_jsonl(external_dir / "blind_items_public.jsonl", external_blind["public_items"])
    _write_json(external_dir / "blind_hidden_key.json", {"items": external_blind["hidden_key"]})
    _write_json(external_dir / "fixture_semantic_audit.json", external_audit)
    _write_json(external_dir / "hard_rule_compatibility.json", external_hard)
    _write_json(external_dir / "external_integrity.json", external_integrity)
    _write_jsonl(historical_dir / "blind_items_public.jsonl", historical_blind["public_items"])
    _write_json(historical_dir / "blind_hidden_key.json", {"items": historical_blind["hidden_key"]})
    _write_json(historical_dir / "historical_integrity.json", historical_integrity)
    _write_json(historical_dir / "historical_source_manifest.json", source_manifest)
    _write_json(EVIDENCE_ROOT / "evaluator_protocol.json", protocol)
    _write_json(EVIDENCE_ROOT / "pre_model_call_freeze.json", freeze)
    _write_json(EVIDENCE_ROOT / "source_manifest.json", source_manifest)
    _write_json(EVIDENCE_ROOT / "preparation_decision.json", preparation)
    _write_json(
        EVIDENCE_ROOT / "evidence-no-runtime/runtime_status.json",
        {
            "kind": "phase66_runtime_status",
            "runtime_replay_status": "not_requested_in_phase66",
            "runtime_replay_model_call_count": 0,
            "runtime_prompt_changed": False,
            "router_changed": False,
        },
    )
    _write_json(
        EVIDENCE_ROOT / "evidence-no-training/training_attempt.json",
        {
            "kind": "phase66_training_attempt",
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
