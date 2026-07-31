from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "tools/phase110_task_grounded_sft_dpo_causal_proof.py"


def _source() -> str:
    return DRIVER.read_text(encoding="utf-8")


def test_phase110_driver_is_local_only_and_private_output_is_outside_repo() -> None:
    source = _source()
    assert "models/Qwen3-4B" in source
    assert "local_files_only=True" in source
    assert 'Path("/private/tmp/pfe-phase110-simulated-review")' in source
    assert "external_provider_allowed\": False" in source
    assert "paid_api_allowed\": False" in source
    assert "requests." not in source
    assert "httpx." not in source
    assert "subprocess" not in source


def test_phase110_driver_has_hard_budget_no_retry_and_conditional_dpo() -> None:
    source = _source()
    assert "MAX_MODEL_CALL_BUDGET = PHASE110_HOLDOUT_COUNT * len(PHASE110_FINAL_VARIANTS)" in source
    assert '"automatic_retry_count": 0' in source
    assert '"dpo_requires_sft_gate": True' in source
    assert '"sft_gate_passed": gate.get("passed") is True' in source
    assert '"phase110_sft_gate_failed"' in source
    assert "post_hoc_truncation_allowed\": False" in source


def test_phase110_driver_never_promotes_or_pushes() -> None:
    source = _source()
    assert '"automatic_promotion_allowed": False' in source
    assert '"product_gate_qualified": False' in source
    assert '"push_performed": False' in source
    assert '"deployment_performed": False' in source
    assert '"promotion_performed": False' in source
    assert "git push" not in source
    assert "gh pr" not in source


def test_phase110_driver_parses_and_exposes_expected_commands() -> None:
    tree = ast.parse(_source())
    functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert {"_prepare", "_diagnose_adapter", "_train_sft", "_evaluate", "_analyze_sft", "_train_dpo", "_decide", "_validate"} <= functions


def test_phase110_source_freeze_includes_tests_executor_and_phase109_decision() -> None:
    source = _source()
    assert '"core_test": REPO_ROOT / "tests/test_phase110_task_grounded_sft_dpo.py"' in source
    assert '"driver_test": REPO_ROOT / "tests/test_phase110_driver_safety.py"' in source
    assert '"executor": CORE_ROOT / "pfe_core/trainer/executors.py"' in source
    assert '"phase109_decision": PHASE109_ROOT / "phase109-final-decision.json"' in source
