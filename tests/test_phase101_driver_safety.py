from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DRIVER = REPO_ROOT / "tools/phase101_failure_targeted_sft.py"


def test_phase101_driver_is_local_qwen3_and_completion_only():
    text = DRIVER.read_text(encoding="utf-8")
    assert 'MODEL_PATH = REPO_ROOT / "models/Qwen3-4B"' in text
    assert "_encode_sft_examples" in text
    assert "_run_real_local_peft_training" in text
    assert '"completion_only_loss_required": True' in text
    assert 'os.environ.setdefault("HF_HUB_OFFLINE", "1")' in text
    assert 'os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")' in text
    assert "local_files_only=True" in text
    assert "http://" not in text
    assert "https://" not in text


def test_phase101_driver_freezes_steps_and_eval_budget():
    text = DRIVER.read_text(encoding="utf-8")
    assert "if steps not in (1, 12, 30):" in text
    assert '"phase101_model_call_budget": 48' in text
    assert '"long_run_total_call_budget": 270' in text
    assert '"guided_target_allowed": False' in text
    assert '"premature_eos_suppression_allowed": False' in text
    assert '"post_hoc_truncation_allowed": False' in text


def test_phase101_thirty_step_requires_twelve_step_stability():
    text = DRIVER.read_text(encoding="utf-8")
    assert 'checks["twelve_step_completed"]' in text
    assert 'checks["twelve_step_parameters_updated"]' in text
    assert 'checks["twelve_step_losses_finite"]' in text
