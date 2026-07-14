from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "pfe-core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from pfe_core.phase67_historical_contract_compatibility_audit import (
    PHASE67_CURRENT_ACCEPT_OBLIGATIONS,
    PHASE67_HISTORICAL_COUNTS,
    build_phase67_contract_matrix,
    build_phase67_current_contract,
    build_phase67_decision,
    build_phase67_historical_partition,
    build_phase67_metric_interpretation,
)


PHASE66_ROOT = ROOT / "docs/demo/phase66-external-distribution-regression"


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_phase67_current_accept_contract_requires_all_three_atoms() -> None:
    contract = build_phase67_current_contract()

    assert tuple(contract["accept_obligations_required_together"]) == (
        PHASE67_CURRENT_ACCEPT_OBLIGATIONS
    )
    assert contract["all_three_safe_atoms_required_for_accept"] is True
    assert contract["incomplete_safe_boundary_label"] == "edit"


def test_phase67_only_phase55_is_direct_label_compatible() -> None:
    matrix = build_phase67_contract_matrix()
    compatible = [
        row["phase"]
        for row in matrix["rows"]
        if row["direct_label_compatible_with_current_contract"]
    ]

    assert matrix["passed"] is True
    assert compatible == ["phase55"]
    assert all(row["automatic_relabel_allowed"] is False for row in matrix["rows"])


def test_phase67_partition_preserves_all_history_without_relabeling() -> None:
    partition = build_phase67_historical_partition(PHASE67_HISTORICAL_COUNTS)

    assert partition["passed"] is True
    assert partition["aligned_legacy_regression_count"] == 150
    assert partition["legacy_diagnostic_only_count"] == 408
    assert partition["automatic_relabel_count"] == 0
    assert partition["training_use_allowed"] is False


def test_phase67_interprets_only_phase55_as_aligned_regression() -> None:
    external = _read(
        PHASE66_ROOT / "evidence-external-holdout/candidate_evaluator_report.json"
    )
    historical = _read(
        PHASE66_ROOT / "evidence-historical-replay/historical_replay_report.json"
    )
    partition = build_phase67_historical_partition(PHASE67_HISTORICAL_COUNTS)
    interpretation = build_phase67_metric_interpretation(
        phase66_external_report=external,
        phase66_historical_report=historical,
        partition=partition,
    )

    assert interpretation["passed"] is True
    assert interpretation["current_contract_fresh_external"]["accuracy"] == 1.0
    assert interpretation["aligned_legacy_phase55_regression"]["accuracy"] == 0.7333
    assert set(interpretation["legacy_diagnostic_only_metrics"]) == {
        "phase51",
        "phase52",
        "phase53",
        "phase54",
    }
    assert interpretation["current_evaluator_qualified_for_runtime_ab"] is False


def test_phase67_decision_unlocks_only_aligned_candidate_recovery() -> None:
    decision = build_phase67_decision(
        phase66_snapshot={"passed": True},
        contract_matrix={"passed": True},
        historical_partition={
            "passed": True,
            "aligned_legacy_regression_phases": ["phase55"],
            "automatic_relabel_count": 0,
        },
        metric_interpretation={
            "passed": True,
            "current_evaluator_qualified_for_runtime_ab": False,
        },
        source_contract_audit={"passed": True},
    )

    assert decision["recommendation"] == (
        "recommend_phase67_contract_aware_partition_for_manual_review_only"
    )
    assert decision["phase68_aligned_candidate_recovery_design_eligible"] is True
    assert decision["runtime_ab_allowed"] is False
    assert decision["training_allowed"] is False


def test_phase66_hold_decision_remains_unchanged() -> None:
    decision = _read(PHASE66_ROOT / "phase66-final-decision.json")
    assert decision["recommendation"] == (
        "hold_phase66_external_distribution_regression"
    )
    assert decision["phase67_minimal_runtime_ab_design_eligible"] is False
