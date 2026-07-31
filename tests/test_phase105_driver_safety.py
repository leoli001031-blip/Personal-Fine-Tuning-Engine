from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DRIVER = REPO_ROOT / "tools/phase105_qwen3_curriculum_alignment.py"


def test_phase105_driver_is_local_completion_only_qwen3():
    text = DRIVER.read_text(encoding="utf-8")
    assert 'MODEL_PATH = REPO_ROOT / "models/Qwen3-4B"' in text
    assert "_encode_sft_examples" in text
    assert "_run_real_local_peft_training" in text
    assert '"completion_only_loss_required": True' in text
    assert "local_files_only=True" in text
    assert "http://" not in text
    assert "https://" not in text


def test_phase105_driver_audits_no_think_system_and_multiturn_alignment():
    text = DRIVER.read_text(encoding="utf-8")
    assert '"system_contract_present"' in text
    assert '"empty_think_boundary_present"' in text
    assert '"all_system_contract_aligned"' in text
    assert '"all_empty_think_aligned"' in text
    assert '"system_contract_aligned": True' in text
    assert '"multiturn_correction_context": True' in text


def test_phase105_driver_freezes_training_and_eval_boundaries():
    text = DRIVER.read_text(encoding="utf-8")
    assert "if steps not in (1, 12, 30):" in text
    assert '"phase105_model_call_budget": 60' in text
    assert '"guided_target_allowed": False' in text
    assert '"post_hoc_truncation_allowed": False' in text
    assert '"product_gate_qualified": False' in text
    assert '"automatic_promotion_allowed": False' in text


def test_phase105_driver_keeps_transcripts_private():
    text = DRIVER.read_text(encoding="utf-8")
    assert 'PRIVATE_ROOT = Path("/private/tmp/pfe-phase105-simulated-review")' in text
    assert '"private_cache_outside_repo": True' in text
    assert '"private_transcripts_committed": False' in text
