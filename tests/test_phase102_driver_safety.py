from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DRIVER = REPO_ROOT / "tools/phase102_failure_targeted_dpo.py"


def test_phase102_driver_uses_stable_local_qwen3_runtime():
    text = DRIVER.read_text(encoding="utf-8")
    assert 'MODEL_PATH = REPO_ROOT / "models/Qwen3-4B"' in text
    assert '"runtime_device": "mps"' in text
    assert '"runtime_dtype": "float32"' in text
    assert '"learning_rate": 0.000005' in text
    assert "execute_dpo_training" in text
    assert "http://" not in text
    assert "https://" not in text


def test_phase102_driver_starts_from_base_and_freezes_steps():
    text = DRIVER.read_text(encoding="utf-8")
    assert '"starts_from_base_because_phase101_sft_archived": True' in text
    assert '"incremental_context" not in spec12["recipe"]["training"]' in text
    assert "if steps not in (12, 30):" in text
    assert 'checks["twelve_step_completed"]' in text
    assert 'checks["twelve_step_metrics_finite"]' in text


def test_phase102_driver_keeps_eval_private_and_no_promotion():
    text = DRIVER.read_text(encoding="utf-8")
    assert 'PRIVATE_ROOT = Path("/private/tmp/pfe-phase102-simulated-review")' in text
    assert '"private_cache_outside_repo": True' in text
    assert '"automatic_promotion_allowed": False' in text
    assert '"cumulative_model_call_count": 120 if candidate else 96' in text
