from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DRIVER = REPO_ROOT / "tools/phase103_simulated_user_acceptance.py"


def test_phase103_driver_freezes_twenty_paired_sessions_and_budget():
    text = DRIVER.read_text(encoding="utf-8")
    assert '"session_count_20"' in text
    assert '"phase103_model_call_budget": 120' in text
    assert '"cumulative_model_call_budget": 240' in text
    assert '"long_run_total_call_budget": 270' in text
    assert '"variants": ["base", "dpo"]' in text


def test_phase103_driver_keeps_transcripts_private_and_simulated():
    text = DRIVER.read_text(encoding="utf-8")
    assert 'PRIVATE_ROOT = Path("/private/tmp/pfe-phase103-simulated-review")' in text
    assert '"private_cache_outside_repo": True' in text
    assert '"simulated_usage": True' in text
    assert '"actual_user_feedback_count": 0' in text
    assert "http://" not in text
    assert "https://" not in text


def test_phase103_driver_never_promotes_archived_candidate():
    text = DRIVER.read_text(encoding="utf-8")
    assert '"adapter_is_archived_candidate": variant == "dpo"' in text
    assert '"automatic_promotion_allowed": False' in text
    assert '"product_gate_qualified": False' in text
