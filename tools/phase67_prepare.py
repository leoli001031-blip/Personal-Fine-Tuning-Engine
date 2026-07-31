#!/usr/bin/env python3
"""Prepare the deterministic Phase67 historical-contract audit."""

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

from pfe_core.phase46_runtime_first_latest_intent import stable_hash
from pfe_core.phase67_historical_contract_compatibility_audit import (
    PHASE67_PHASES,
    build_phase67_contract_matrix,
    build_phase67_historical_partition,
    build_phase67_metric_interpretation,
)


EVIDENCE_ROOT = REPO_ROOT / "docs/demo/phase67-historical-contract-compatibility-audit"
PHASE66_ROOT = REPO_ROOT / "docs/demo/phase66-external-distribution-regression"
SOURCE_FILES = {
    phase: CORE_ROOT / f"pfe_core/{filename}"
    for phase, filename in {
        "phase51": "phase51_dual_evaluator_hardening.py",
        "phase52": "phase52_adversarial_evaluator_generalization.py",
        "phase53": "phase53_evaluator_scope_recovery.py",
        "phase54": "phase54_typed_proposition_evaluator.py",
        "phase55": "phase55_atomic_boundary_composition.py",
        "phase56": "phase56_evidence_span_grounded_atomic.py",
        "phase67": "phase67_historical_contract_compatibility_audit.py",
    }.items()
}
SOURCE_ASSERTIONS = {
    "phase51": (
        "这些材料不能被当作真实用户反馈，或不能据此确认用户/产品收益",
    ),
    "phase52": (
        "或者明确规定只有未来取得真实用户实测/确认之后，新获得的实际反馈才有资格登记",
    ),
    "phase53": (
        "不能登记为真实用户反馈，并明确否认或保留用户收益结论",
    ),
    "phase54": (
        'row["source_eligibility"] == "ineligible_as_actual"',
        'row["suspended_or_negated_outcome"] == "suspended_or_negated"',
        'row["explicit_provenance_boundary"] == "explicit"',
        "中的任意一种，",
        "本身就是 explicit",
    ),
    "phase55": (
        'row["source_registration"] == "exclude_actual"',
        'row["user_outcome_status"] == "suspended_or_negated"',
        'row["test_to_user_outcome_relation"] == "does_not_establish"',
    ),
    "phase56": (
        'row["source_registration"] == "exclude_actual"',
        'row["user_outcome_status"] == "suspended_or_negated"',
        'row["test_to_user_outcome_relation"] == "does_not_establish"',
    ),
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


def _phase66_snapshot() -> dict[str, Any]:
    decision = _read_json(PHASE66_ROOT / "phase66-final-decision.json")
    external = _read_json(
        PHASE66_ROOT / "evidence-external-holdout/candidate_evaluator_report.json"
    )
    historical = _read_json(
        PHASE66_ROOT / "evidence-historical-replay/historical_replay_report.json"
    )
    integrity = _read_json(PHASE66_ROOT / "evidence_integrity.json")
    manifest = _read_json(PHASE66_ROOT / "evidence_manifest.json")
    manifest_verified = _verify_manifest(manifest)
    checks = {
        "phase66_held": decision.get("recommendation")
        == "hold_phase66_external_distribution_regression",
        "fresh_external_current_contract_exact": external.get("status") == "qualified"
        and external.get("accuracy") == 1.0,
        "historical_distribution_not_qualified": historical.get("status")
        == "not_qualified"
        and historical.get("accuracy") == 0.6595,
        "phase66_integrity_passed": integrity.get("passed") is True,
        "phase66_manifest_verified": manifest_verified,
        "runtime_remained_blocked": decision.get(
            "phase67_minimal_runtime_ab_design_eligible"
        )
        is False,
    }
    return {
        "kind": "phase67_phase66_canonical_snapshot",
        "passed": all(checks.values()),
        "checks": checks,
        "recommendation": decision.get("recommendation"),
        "fresh_external_accuracy": external.get("accuracy"),
        "historical_accuracy": historical.get("accuracy"),
        "phase66_manifest_sha256": manifest.get("manifest_sha256"),
    }


def _source_contract_audit() -> dict[str, Any]:
    rows = []
    for phase, assertions in SOURCE_ASSERTIONS.items():
        path = SOURCE_FILES[phase]
        source = path.read_text(encoding="utf-8")
        checks = {assertion: assertion in source for assertion in assertions}
        rows.append(
            {
                "phase": phase,
                "path": str(path.relative_to(REPO_ROOT)),
                "sha256": _sha256(path),
                "assertions": checks,
                "passed": all(checks.values()),
            }
        )
    checks = {
        "all_source_contract_assertions_passed": all(row["passed"] for row in rows),
        "phase51_to_56_covered": [row["phase"] for row in rows]
        == ["phase51", "phase52", "phase53", "phase54", "phase55", "phase56"],
        "phase55_and_phase56_composer_assertions_match": tuple(
            SOURCE_ASSERTIONS["phase55"]
        )
        == tuple(SOURCE_ASSERTIONS["phase56"]),
    }
    return {
        "kind": "phase67_source_contract_audit",
        "passed": all(checks.values()),
        "checks": checks,
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-evidence", action="store_true")
    args = parser.parse_args()
    if args.clean_evidence and EVIDENCE_ROOT.exists():
        shutil.rmtree(EVIDENCE_ROOT)

    snapshot = _phase66_snapshot()
    source_manifest66 = _read_json(PHASE66_ROOT / "source_manifest.json")
    historical_sources = list(source_manifest66.get("historical_sources") or [])
    source_counts = {
        str(row.get("phase")): int(row.get("case_count") or 0)
        for row in historical_sources
    }
    source_files_verified = all(
        (REPO_ROOT / str(row.get("path") or "")).is_file()
        and _sha256(REPO_ROOT / str(row.get("path") or "")) == row.get("sha256")
        for row in historical_sources
    )
    matrix = build_phase67_contract_matrix()
    partition = build_phase67_historical_partition(source_counts)
    external = _read_json(
        PHASE66_ROOT / "evidence-external-holdout/candidate_evaluator_report.json"
    )
    historical = _read_json(
        PHASE66_ROOT / "evidence-historical-replay/historical_replay_report.json"
    )
    interpretation = build_phase67_metric_interpretation(
        phase66_external_report=external,
        phase66_historical_report=historical,
        partition=partition,
    )
    source_audit = _source_contract_audit()

    source_manifest = {
        "kind": "phase67_contract_audit_source_manifest",
        "historical_sources": historical_sources,
        "historical_source_files_verified": source_files_verified,
        "historical_case_count": sum(source_counts.values()),
        "aligned_legacy_case_count": partition["aligned_legacy_regression_count"],
        "diagnostic_only_case_count": partition["legacy_diagnostic_only_count"],
        "simulated_evaluator_fixture": True,
        "actual_user_feedback_count": 0,
        "training_use_allowed": False,
        "private_user_material_used": False,
    }
    freeze = {
        "kind": "phase67_deterministic_audit_freeze",
        "phase66_snapshot_sha256": stable_hash(snapshot),
        "contract_matrix_sha256": stable_hash(matrix),
        "historical_partition_sha256": stable_hash(partition),
        "source_contract_audit_sha256": stable_hash(source_audit),
        "source_file_sha256": {
            phase: _sha256(path) for phase, path in SOURCE_FILES.items()
        },
        "model_calls_allowed": False,
        "automatic_relabel_allowed": False,
        "frozen_before_final_decision": True,
        "created_at": _utcnow(),
    }
    preparation_checks = {
        "phase66_snapshot_passed": snapshot["passed"],
        "historical_source_files_verified": source_files_verified,
        "contract_matrix_passed": matrix["passed"],
        "historical_partition_passed": partition["passed"],
        "metric_interpretation_passed": interpretation["passed"],
        "source_contract_audit_passed": source_audit["passed"],
    }
    preparation = {
        "kind": "phase67_preparation_decision",
        "status": "ready_for_deterministic_audit_finalization"
        if all(preparation_checks.values())
        else "blocked",
        "checks": preparation_checks,
        "failed_checks": [key for key, value in preparation_checks.items() if not value],
        "model_call_count": 0,
        "automatic_relabel_count": 0,
    }

    _write_json(EVIDENCE_ROOT / "evidence-baseline/phase66_canonical_snapshot.json", snapshot)
    _write_json(EVIDENCE_ROOT / "contract_definitions.json", matrix["current_contract"])
    _write_json(EVIDENCE_ROOT / "contract_compatibility_matrix.json", matrix)
    _write_json(EVIDENCE_ROOT / "historical_partition.json", partition)
    _write_json(EVIDENCE_ROOT / "metric_interpretation.json", interpretation)
    _write_json(EVIDENCE_ROOT / "source_contract_audit.json", source_audit)
    _write_json(EVIDENCE_ROOT / "audit_freeze.json", freeze)
    _write_json(EVIDENCE_ROOT / "source_manifest.json", source_manifest)
    _write_json(EVIDENCE_ROOT / "preparation_decision.json", preparation)
    _write_json(
        EVIDENCE_ROOT / "evidence-no-model-calls/model_call_status.json",
        {
            "kind": "phase67_model_call_status",
            "status": "not_requested",
            "model_call_count": 0,
            "reason": "contract compatibility is a deterministic source-and-label audit",
        },
    )
    _write_json(
        EVIDENCE_ROOT / "evidence-no-runtime/runtime_status.json",
        {
            "kind": "phase67_runtime_status",
            "runtime_replay_status": "blocked_until_aligned_evaluator_regression_qualifies",
            "runtime_replay_model_call_count": 0,
            "product_default_changed": False,
        },
    )
    _write_json(
        EVIDENCE_ROOT / "evidence-no-training/training_attempt.json",
        {
            "kind": "phase67_training_attempt",
            "status": "not_requested",
            "training_executed": False,
            "adapter_created": False,
            "auto_training_allowed": False,
        },
    )
    print(json.dumps(preparation, ensure_ascii=False, indent=2))
    return 0 if preparation["status"] == "ready_for_deterministic_audit_finalization" else 1


if __name__ == "__main__":
    raise SystemExit(main())
