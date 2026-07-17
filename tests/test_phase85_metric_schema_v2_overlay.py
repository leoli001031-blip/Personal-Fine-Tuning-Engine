from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pytest

from tools import phase85_metric_schema_v2_overlay as overlay


def _artifact(tmp_path: Path, filename: str) -> Path:
    return tmp_path / overlay.OVERLAY_ROOT_RELATIVE / filename


def _assert_no_absolute_paths(value: Any) -> None:
    if isinstance(value, dict):
        for child in value.values():
            _assert_no_absolute_paths(child)
    elif isinstance(value, list):
        for child in value:
            _assert_no_absolute_paths(child)
    elif isinstance(value, str):
        assert not value.startswith("/")


def test_finalize_emits_deterministic_variant_aware_overlay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(overlay, "OUTPUT_REPO_ROOT", tmp_path)

    assert overlay.main(["finalize"]) == 0
    artifact = _artifact(tmp_path, overlay.OVERLAY_FILENAME)
    first_render = artifact.read_bytes()
    payload = json.loads(first_render)

    assert [path.name for path in artifact.parent.iterdir()] == [
        overlay.OVERLAY_FILENAME
    ]
    assert payload["schema_validation_passed"] is True
    assert payload["overlay_decision"] == {
        "canonical_archive_status": "archive_incomplete_phase85_evidence",
        "claim_boundaries": {
            "deployment_allowed": False,
            "hermes_attachment_allowed": False,
            "product_benefit_claim_allowed": False,
            "promotion_allowed": False,
            "training_benefit_claim_allowed": False,
        },
        "failed_benefit_check_count": 7,
        "failed_benefit_checks": list(overlay.EXPECTED_BENEFIT_FAILURES),
        "product_gate_qualified": False,
        "status": "archive_low_fallback_runtime_not_qualified",
        "validation_pass_does_not_imply_product_pass": True,
    }
    canonical = payload["source_bindings"]["canonical_decision"]
    canonical_path = overlay.SOURCE_REPO_ROOT / canonical["path"]
    assert canonical["sha256"] == hashlib.sha256(canonical_path.read_bytes()).hexdigest()
    assert canonical["status"] == "archive_incomplete_phase85_evidence"

    base = payload["variants"]["base_api_length_control_160"]
    assert base["format_metrics_basis"] == "counterfactual_common_v4_evaluator"
    assert base["factual_guard_fallback_turn_rate"] is None
    assert base["observed_runtime_fallback_turn_rate"] is None
    assert math.isfinite(base["counterfactual_guard_fallback_turn_rate"])
    assert (
        base["counterfactual_guard_fallback_turn_rate"]
        == base["fallback_turn_rate"]
    )

    for variant in (
        "persona_api_contract_v3_fresh",
        "persona_api_contract_v4",
    ):
        row = payload["variants"][variant]
        assert row["format_metrics_basis"] == "runtime_contract_output"
        assert row["counterfactual_guard_fallback_turn_rate"] is None
        assert math.isfinite(row["factual_guard_fallback_turn_rate"])
        assert math.isfinite(row["observed_runtime_fallback_turn_rate"])
        assert row["factual_guard_fallback_turn_rate"] == row["fallback_turn_rate"]
        assert row["observed_runtime_fallback_turn_rate"] == row["fallback_turn_rate"]

    for row in payload["variants"].values():
        assert row["session_count"] == 30
        assert row["completed_session_count"] == 30
        assert row["api_call_count"] == 90
        assert row["runtime_attempt_count"] == 90
        assert row["model_call_count"] == 90
        assert row["format_eligible_turn_count"] == 68
        assert row["format_accounting_exact"] is True
        assert row["stored_metrics"][
            "exactly_reaggregated_with_frozen_driver"
        ] is True

    unhashed = dict(payload)
    unhashed.pop("payload_sha256")
    assert payload["payload_sha256"] == overlay._stable_hash(unhashed)
    assert "created_at" not in json.dumps(payload, sort_keys=True)
    _assert_no_absolute_paths(payload)

    assert overlay.main(["finalize"]) == 0
    assert artifact.read_bytes() == first_render


def test_validate_rerenders_and_detects_overlay_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(overlay, "OUTPUT_REPO_ROOT", tmp_path)
    assert overlay.main(["finalize"]) == 0

    assert overlay.main(["validate"]) == 0
    summary_path = _artifact(tmp_path, overlay.VALIDATION_FILENAME)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["passed"] is True
    assert summary["checks"][
        "overlay_exactly_matches_deterministic_rerender"
    ] is True
    assert summary["product_gate_qualified"] is False
    assert summary["validation_pass_does_not_imply_product_pass"] is True
    assert "does not imply product pass" in summary["validation_scope"]

    artifact = _artifact(tmp_path, overlay.OVERLAY_FILENAME)
    tampered = json.loads(artifact.read_text(encoding="utf-8"))
    tampered["overlay_decision"]["product_gate_qualified"] = True
    artifact.write_text(json.dumps(tampered, sort_keys=True), encoding="utf-8")

    assert overlay.main(["validate"]) == 1
    failed = json.loads(summary_path.read_text(encoding="utf-8"))
    assert failed["passed"] is False
    assert failed["checks"][
        "overlay_exactly_matches_deterministic_rerender"
    ] is False
    assert failed["checks"]["overlay_payload_hash_valid"] is False
    assert failed["checks"]["product_gate_remains_not_qualified"] is False
    assert failed["product_gate_qualified"] is False


def test_finalize_fails_closed_on_frozen_phase85_hash_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(overlay, "OUTPUT_REPO_ROOT", tmp_path)
    mismatched_paths = dict(overlay.FROZEN_PHASE85_SOURCE_PATHS)
    mismatched_paths["phase85_driver"] = mismatched_paths["phase85_core"]
    monkeypatch.setattr(overlay, "FROZEN_PHASE85_SOURCE_PATHS", mismatched_paths)

    assert overlay.main(["finalize"]) == 1
    assert not _artifact(tmp_path, overlay.OVERLAY_FILENAME).exists()


def test_cli_exposes_finalize_and_validate_only() -> None:
    with pytest.raises(SystemExit):
        overlay.main(["generate"])
