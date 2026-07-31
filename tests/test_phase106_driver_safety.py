from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DRIVER = REPO_ROOT / "tools/phase106_stratified_curriculum_repair.py"


def test_phase106_driver_is_a_single_variable_sampling_repair():
    text = DRIVER.read_text(encoding="utf-8")
    assert '"sampling_strategy"] = "seeded_stratified"' in text
    assert '"single_variable_repair": "sampling_strategy"' in text
    assert '"expected_30step_category_exposure": 6' in text
    assert "_build_seeded_stratified_training_order" in text


def test_phase106_driver_freezes_training_eval_and_no_promotion():
    text = DRIVER.read_text(encoding="utf-8")
    assert "if steps not in (1, 12, 30):" in text
    assert '"phase106_model_call_budget": 60' in text
    assert '"product_gate_qualified": False' in text
    assert '"automatic_promotion_allowed": False' in text
    assert "http://" not in text
    assert "https://" not in text


def test_phase106_driver_requires_actual_balanced_exposure_before_eval():
    text = DRIVER.read_text(encoding="utf-8")
    assert '"actual_30step_exposure_balanced"' in text
    assert '"stratified_exposure_balanced"' in text
    assert '"category_exposure_counts"' in text


def test_phase106_driver_keeps_transcripts_private():
    text = DRIVER.read_text(encoding="utf-8")
    assert 'PRIVATE_ROOT = Path("/private/tmp/pfe-phase106-simulated-review")' in text
    assert '"private_cache_outside_repo": True' in text
    assert '"private_transcripts_committed": False' in text
