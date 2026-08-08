from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from pfe_core.phase111_112_evidence_failure_eval import (
    audit_holdout_isolation,
    build_expected_response,
    build_phase112_cases,
    load_claim_ledger,
    score_phase112_response,
    validate_claim_ledger,
    validate_phase112_cases,
)
from tools import phase111_112_evidence_failure_eval as driver


def _write_source_pack(root: Path) -> Path:
    root.mkdir(parents=True)
    evidence = root / "source-evidence.json"
    evidence.write_text('{"status":"archive"}\n', encoding="utf-8")
    claim_fields = (
        "claim_id",
        "claim",
        "evidence_path",
        "evidence_class",
        "observed_at",
        "authorization_state",
        "allowed_wording",
        "forbidden_wording",
    )
    with (root / "claim-evidence.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=claim_fields)
        writer.writeheader()
        for index in range(1, 29):
            writer.writerow(
                {
                    "claim_id": f"TEST-{index:03d}",
                    "claim": f"narrow claim {index}",
                    "evidence_path": str(evidence),
                    "evidence_class": "historical" if index <= 20 else "local",
                    "observed_at": "2026-08-09T00:00:00+08:00",
                    "authorization_state": "只读历史结论" if index <= 20 else "仅本地证据",
                    "allowed_wording": f"allowed {index}",
                    "forbidden_wording": f"forbidden {index}",
                }
            )
    with (root / "eval-briefs.jsonl").open("w", encoding="utf-8") as handle:
        for index in range(1, 31):
            handle.write(
                json.dumps(
                    {
                        "eval_id": f"TEST-E{index:03d}",
                        "title": f"eval {index}",
                        "system": "PFE",
                        "objective": "preserve the narrow evidence boundary",
                        "evidence_paths": [str(evidence)],
                        "evidence_class": "historical",
                        "authorization_state": "read only",
                        "expected_behavior": "report archive",
                        "failure_oracle": "qualified wording fails",
                    }
                )
                + "\n"
            )
    (root / "failure-modes.md").write_text(
        "".join(f"## FM-{index:02d}: mode\n" for index in range(1, 16)),
        encoding="utf-8",
    )
    (root / "authorization-matrix.md").write_text(
        "# Authorization\nNo production write.\n", encoding="utf-8"
    )
    return root


def test_claim_ledger_preserves_evidence_class_and_narrow_status(tmp_path: Path) -> None:
    source = _write_source_pack(tmp_path / "source")
    rows = load_claim_ledger(source / "claim-evidence.csv")

    report = validate_claim_ledger(rows, expected_count=28)

    assert report["passed"] is True
    assert report["status_counts"] == {
        "proven": 20,
        "partially_proven": 8,
        "not_proven": 0,
        "blocked_by_authorization": 0,
    }
    assert all(row["evidence_class"] == row["source_evidence_class"] for row in rows)


def test_claim_ledger_rejects_historical_to_authorized_real_escalation(
    tmp_path: Path,
) -> None:
    source = _write_source_pack(tmp_path / "source")
    rows = load_claim_ledger(source / "claim-evidence.csv")
    rows[0]["evidence_class"] = "authorized_real"

    with pytest.raises(ValueError, match="evidence class escalation"):
        validate_claim_ledger(rows, expected_count=28)


def test_phase112_builds_seventy_unique_simulated_cases_with_category_floor() -> None:
    cases = build_phase112_cases()

    report = validate_phase112_cases(cases)

    assert report["passed"] is True
    assert report["case_count"] == 70
    assert report["unique_holdout_fingerprint_count"] == 70
    assert set(report["category_counts"].values()) == {10}
    assert all(case["usage_label"] == "simulated_usage" for case in cases)
    assert all(case["actual_user_feedback"] is False for case in cases)


def test_phase112_scorer_reports_dimension_and_missing_fields() -> None:
    case = build_phase112_cases()[0]
    perfect = build_expected_response(case)
    failing = dict(perfect)
    failing["known_facts"] = []
    failing["correction_applied"] = False
    failing["product_gate_qualified"] = True

    perfect_score = score_phase112_response(case, perfect)
    failing_score = score_phase112_response(case, failing)

    assert perfect_score["passed"] is True
    assert perfect_score["overall_score"] == 1.0
    assert failing_score["passed"] is False
    assert {row["dimension"] for row in failing_score["failures"]} == {
        "fact_coverage",
        "correction_response",
        "boundary_safety",
    }
    assert all(row["missing_fields"] for row in failing_score["failures"])


@pytest.mark.parametrize(
    ("category", "expected_dimension"),
    [
        ("fact_omission", "fact_coverage"),
        ("latest_correction_ignored", "correction_response"),
        ("local_context_lost", "local_state_reference"),
        ("false_completion", "boundary_safety"),
        ("unnecessary_confirmation", "latest_intent_obedience"),
        ("format_instability", "format_correctness"),
        ("privacy_or_provenance_boundary", "boundary_safety"),
    ],
)
def test_each_failure_category_produces_explainable_primary_failure(
    category: str,
    expected_dimension: str,
) -> None:
    case = next(row for row in build_phase112_cases() if row["category"] == category)
    response = build_expected_response(case)
    if category == "fact_omission":
        response["known_facts"] = response["known_facts"][1:]
    elif category == "latest_correction_ignored":
        response["correction_applied"] = False
    elif category == "local_context_lost":
        response["local_state_refs"] = []
    elif category == "false_completion":
        response["product_gate_qualified"] = True
    elif category == "unnecessary_confirmation":
        response["next_action"] = "请确认是否继续"
    elif category == "format_instability":
        del response["uncertainties"]
    else:
        response["provenance_scope"] = "actual_user_feedback"

    score = score_phase112_response(case, response)

    assert score["passed"] is False
    matching = [
        failure
        for failure in score["failures"]
        if failure["dimension"] == expected_dimension
    ]
    assert matching
    assert all(failure["category"] == category for failure in matching)
    assert all(failure["missing_fields"] for failure in matching)


def test_phase112_holdout_isolation_detects_collision() -> None:
    cases = build_phase112_cases()

    clean = audit_holdout_isolation(cases, [])
    contaminated = audit_holdout_isolation(cases, [cases[0]["holdout_fingerprint"]])

    assert clean == {
        "holdout_fingerprint_count": 70,
        "training_fingerprint_count": 0,
        "collision_count": 0,
        "collisions": [],
        "passed": True,
    }
    assert contaminated["passed"] is False
    assert contaminated["collision_count"] == 1


def test_phase111_112_generator_round_trip_without_model_calls(tmp_path: Path) -> None:
    source = _write_source_pack(tmp_path / "source")
    evidence_root = tmp_path / "evidence"

    decision = driver.generate(source, evidence_root, clean=False)
    validation = driver.validate(evidence_root)

    assert decision["status"] == "phase111_112_evidence_eval_ready_no_training"
    assert decision["model_call_count"] == 0
    assert decision["training_run_count"] == 0
    assert decision["actual_user_feedback_count"] == 0
    assert decision["product_gate_qualified"] is False
    assert validation["claim_ledger"]["claim_count"] == 28
    assert validation["eval_manifest"]["eval_count"] == 30
    assert validation["phase112_cases"]["case_count"] == 70
    assert validation["holdout_integrity"]["collision_count"] == 0
