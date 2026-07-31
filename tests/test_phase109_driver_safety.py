from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest

from pfe_core.phase109_personal_engineering_copilot import (
    PHASE109_MODEL_CALL_BUDGET,
    PHASE109_VARIANTS,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DRIVER_PATH = REPO_ROOT / "tools/phase109_personal_engineering_copilot.py"


def _source() -> str:
    return DRIVER_PATH.read_text(encoding="utf-8")


def _load_driver():
    spec = importlib.util.spec_from_file_location("phase109_driver_safety", DRIVER_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _called_functions(source: str) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            names.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            names.add(node.func.attr)
    return names


def test_phase109_driver_is_local_bounded_and_three_variant() -> None:
    driver = _load_driver()

    assert driver.MODEL_CALL_BUDGET == PHASE109_MODEL_CALL_BUDGET == 105
    assert tuple(driver.EVAL_VARIANTS) == PHASE109_VARIANTS
    assert str(driver.MODEL_PATH).endswith("/models/Qwen3-4B")
    parser = driver._parser()
    for variant in PHASE109_VARIANTS:
        assert parser.parse_args(["eval", "--variant", variant]).variant == variant
    with pytest.raises(SystemExit):
        parser.parse_args(["eval", "--variant", "external_provider"])


def test_phase109_call_ledger_is_append_only_and_hard_capped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    driver = _load_driver()
    ledger = tmp_path / "call_ledger.jsonl"
    monkeypatch.setattr(driver, "CALL_LEDGER", ledger)

    for index in range(PHASE109_MODEL_CALL_BUDGET):
        driver._reserve_call("base", f"session-{index:03d}")
    frozen = ledger.read_bytes()

    with pytest.raises(RuntimeError, match="budget"):
        driver._reserve_call("base", "over-budget")
    assert ledger.read_bytes() == frozen


def test_phase109_clean_refuses_after_any_model_attempt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    driver = _load_driver()
    evidence = tmp_path / "evidence"
    private = tmp_path / "private"
    ledger = evidence / "call_ledger.jsonl"
    ledger.parent.mkdir(parents=True)
    ledger.write_text('{"event":"attempted","call_id":"one"}\n', encoding="utf-8")
    monkeypatch.setattr(driver, "EVIDENCE_ROOT", evidence)
    monkeypatch.setattr(driver, "PRIVATE_ROOT", private)
    monkeypatch.setattr(driver, "CALL_LEDGER", ledger)

    with pytest.raises(RuntimeError, match="ledger"):
        driver._clean_prepare()
    assert ledger.is_file()


def test_phase109_driver_keeps_raw_outputs_outside_repo_and_off_network() -> None:
    driver = _load_driver()
    source = _source()
    called = _called_functions(source)

    assert str(driver.PRIVATE_ROOT).startswith("/private/tmp/")
    assert REPO_ROOT not in driver.PRIVATE_ROOT.parents
    assert 'os.environ.setdefault("HF_HUB_OFFLINE", "1")' in source
    assert 'os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")' in source
    assert "local_files_only=True" in source
    assert "http://" not in source
    assert "https://" not in source
    assert "requests" not in source
    assert "httpx" not in source
    assert "subprocess" not in source
    assert not ({"push", "deploy", "promote"} & called)


def test_phase109_training_is_frozen_to_local_1_12_30_step_dpo() -> None:
    source = _source()

    assert "if steps not in (1, 12, 30):" in source
    assert '"local_only": True' in source
    assert '"trainer_class": "trl.DPOTrainer"' in source
    assert '"runtime_device": "mps"' in source
    assert '"automatic_promotion_allowed": False' in source
    assert '"product_gate_qualified": False' in source


def test_phase109_adapter_loading_keeps_historical_and_new_lineages_separate() -> None:
    source = _source()
    tree = ast.parse(source)
    loader = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_load_model")
    segment = ast.get_source_segment(source, loader)
    assert segment is not None

    phase107_block = segment[segment.index('if variant == "phase107_dpo"'):segment.index('elif variant == "phase109_personal_dpo"')]
    phase109_block = segment[segment.index('elif variant == "phase109_personal_dpo"'):segment.index('elif variant != "base"')]
    assert "PHASE106_ADAPTER" in phase107_block
    assert "merge_and_unload" in phase107_block
    assert "PHASE107_ADAPTER" in phase107_block
    assert "_phase109_adapter" in phase109_block
    assert "PHASE106_ADAPTER" not in phase109_block


def test_phase109_generation_has_one_attempt_and_no_post_hoc_truncation() -> None:
    source = _source()
    tree = ast.parse(source)
    generator = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_generate_once")
    segment = ast.get_source_segment(source, generator)
    assert segment is not None
    parsed = ast.parse(segment)
    generate_calls = [
        node for node in ast.walk(parsed)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "generate"
    ]

    assert len(generate_calls) == 1
    assert not any(isinstance(node, (ast.For, ast.AsyncFor, ast.While)) for node in ast.walk(parsed))
    assert '"automatic_retry_count": 0' in source
    assert '"post_hoc_truncation_allowed": False' in source
    assert '"post_hoc_truncation_used": False' in source
