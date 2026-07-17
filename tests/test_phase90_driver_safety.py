from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DRIVER_PATH = ROOT / "tools/phase90_native_format_curriculum_repair.py"


def _load_driver():
    spec = importlib.util.spec_from_file_location("phase90_driver", DRIVER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_phase90_driver_uses_local_model_and_denies_automatic_actions() -> None:
    driver = _load_driver()

    assert driver.MODEL_PATH == ROOT / "models/Qwen2.5-1.5B-Instruct"
    assert driver.PHASE90_TRAINING_MAX_LENGTH == 288
    assert driver.FROZEN_THRESHOLDS["auto_promotion_allowed"] is False
    source = DRIVER_PATH.read_text(encoding="utf-8")
    core_source = (
        ROOT / "pfe-core/pfe_core/phase90_native_format_curriculum.py"
    ).read_text(encoding="utf-8")
    assert "external provider" not in source.lower()
    assert '"automatic_deployment_allowed": False' in core_source
    assert '"hermes_attachment_allowed": False' in core_source


def test_phase90_sanity_ranking_prioritizes_native_format() -> None:
    driver = _load_driver()
    high_native = {
        "raw": {
            "native_format_rate": 0.8,
            "false_block_rate": 0.1,
            "truncated_session_rate": 0.0,
            "overall_score": 0.7,
        }
    }
    low_native = {
        "raw": {
            "native_format_rate": 0.6,
            "false_block_rate": 0.0,
            "truncated_session_rate": 0.0,
            "overall_score": 0.9,
        }
    }

    assert driver._sanity_sort_key(high_native, "format_first") > driver._sanity_sort_key(
        low_native, "balanced"
    )


def test_phase90_simulated_review_marks_format_and_false_block() -> None:
    driver = _load_driver()
    session = {
        "category": "verified_completion_positive",
        "format_expected": True,
        "completion_expected": True,
        "provenance_rejection_expected": False,
        "forbidden_claims": [],
        "declared_private_values": [],
    }

    findings = driver._simulated_findings("目前不能确认。", session)

    assert "false_block" in findings
    assert "format_failure" in findings


def test_phase90_parser_rejects_unregistered_generation_variants() -> None:
    driver = _load_driver()
    parser = driver._build_parser()

    try:
        parser.parse_args(
            ["generate", "--scope", "full", "--variant", "external_provider"]
        )
    except SystemExit as exc:
        assert exc.code != 0
    else:
        raise AssertionError("unregistered generation variant should fail")
