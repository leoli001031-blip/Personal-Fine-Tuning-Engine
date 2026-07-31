from __future__ import annotations

import ast
import importlib.util
import inspect
import json
from pathlib import Path

import pytest

from pfe_core.phase108_runtime_adapter_causal_value import (
    PHASE108_CALL_BUDGET,
    PHASE108_DIAGNOSTIC_VARIANT,
    PHASE108_MAIN_VARIANTS,
    build_phase108_decision,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DRIVER_PATH = REPO_ROOT / "tools/phase108_runtime_adapter_causal_value_proof.py"
EXPECTED_VARIANTS = {*PHASE108_MAIN_VARIANTS, PHASE108_DIAGNOSTIC_VARIANT}


def _driver_source() -> str:
    assert DRIVER_PATH.is_file(), f"Phase108 driver is missing: {DRIVER_PATH}"
    return DRIVER_PATH.read_text(encoding="utf-8")


def _load_driver():
    spec = importlib.util.spec_from_file_location("phase108_driver_safety", DRIVER_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ledger_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _called_function_names(source: str) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            names.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            names.add(node.func.attr)
    return names


def test_phase108_freezes_exact_hard_budget_and_four_eval_variants() -> None:
    driver = _load_driver()

    assert PHASE108_CALL_BUDGET == 300
    assert driver.MODEL_CALL_BUDGET == PHASE108_CALL_BUDGET
    assert set(driver.EVAL_VARIANTS) == EXPECTED_VARIANTS

    parser = driver._parser()
    for variant in EXPECTED_VARIANTS:
        parsed = parser.parse_args(["eval", "--variant", variant])
        assert parsed.variant == variant
    with pytest.raises(SystemExit):
        parser.parse_args(["eval", "--variant", "external_provider"])


def test_phase108_call_ledger_is_append_only_and_hard_capped(tmp_path: Path) -> None:
    driver = _load_driver()
    ledger = tmp_path / "call_ledger.jsonl"
    sentinel = {
        "attempt_id": "pre-existing-attempt",
        "variant": "base",
        "session_id": "pre-existing-session",
        "turn_index": 1,
    }
    ledger.write_text(json.dumps(sentinel) + "\n", encoding="utf-8")

    for index in range(1, PHASE108_CALL_BUDGET):
        driver._reserve_model_call(
            ledger,
            {
                "attempt_id": f"attempt-{index:03d}",
                "variant": "base",
                "session_id": f"session-{index:03d}",
                "turn_index": 1,
            },
        )

    rows = _ledger_rows(ledger)
    assert len(rows) == PHASE108_CALL_BUDGET
    assert rows[0] == sentinel
    frozen = ledger.read_bytes()

    with pytest.raises(RuntimeError, match="budget|300"):
        driver._reserve_model_call(
            ledger,
            {
                "attempt_id": "over-budget",
                "variant": "base",
                "session_id": "over-budget",
                "turn_index": 1,
            },
        )

    assert ledger.read_bytes() == frozen


def test_phase108_clean_refuses_to_erase_existing_call_ledger(tmp_path: Path) -> None:
    driver = _load_driver()
    evidence_root = tmp_path / "evidence"
    evidence_root.mkdir()
    disposable = evidence_root / "disposable.json"
    disposable.write_text("{}\n", encoding="utf-8")
    ledger = evidence_root / "call_ledger.jsonl"
    ledger.write_text('{"attempt_id":"already-counted"}\n', encoding="utf-8")
    frozen = ledger.read_bytes()

    with pytest.raises(RuntimeError, match="ledger|clean|attempt"):
        driver._clean_evidence(evidence_root=evidence_root, ledger_path=ledger)

    assert ledger.read_bytes() == frozen
    assert _ledger_rows(ledger) == [{"attempt_id": "already-counted"}]


def test_phase108_dpo_load_order_requires_phase106_merge_parent() -> None:
    source = _driver_source()
    model_loader = ast.get_source_segment(
        source,
        next(
            node
            for node in ast.walk(ast.parse(source))
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "_load_model_for_variant"
        ),
    )
    assert model_loader is not None

    raw_index = model_loader.index("AutoModelForCausalLM.from_pretrained")
    phase106_index = model_loader.index("PHASE106_ADAPTER_ROOT")
    merge_index = model_loader.index("merge_and_unload")
    phase107_index = model_loader.index("PHASE107_ADAPTER_ROOT")
    assert raw_index < phase106_index < merge_index < phase107_index
    assert "phase107_dpo" in model_loader
    assert "phase107_dpo_no_runtime" in model_loader


def test_phase108_raw_canary_is_checked_before_output_guard() -> None:
    driver = _load_driver()
    generation_source = inspect.getsource(driver._generate_session)

    reserve_index = generation_source.index("_reserve_model_call")
    generate_index = generation_source.index("_generate_once")
    raw_check_index = generation_source.index("raw_canary_echo")
    guard_index = generation_source.index("guard_phase77_output")
    assert reserve_index < generate_index < raw_check_index < guard_index
    assert "raw_canary_echo" in generation_source[guard_index:]


def test_phase108_generation_is_single_attempt_without_post_hoc_truncation() -> None:
    driver = _load_driver()
    source = _driver_source()
    generate_source = inspect.getsource(driver._generate_once)
    tree = ast.parse(generate_source)

    generate_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "generate"
    ]
    assert len(generate_calls) == 1
    assert not any(isinstance(node, (ast.For, ast.AsyncFor, ast.While)) for node in ast.walk(tree))
    assert '"automatic_retry_allowed": False' in source
    assert '"post_hoc_truncation_allowed": False' in source
    assert '"post_hoc_truncation_used": False' in source
    assert "retrying" not in source.lower()


def test_phase108_keeps_raw_outputs_outside_repo_and_has_no_external_side_effects() -> None:
    driver = _load_driver()
    source = _driver_source()
    called_functions = _called_function_names(source)

    assert str(driver.PRIVATE_REVIEW_ROOT).startswith("/private/tmp/")
    assert REPO_ROOT not in driver.PRIVATE_REVIEW_ROOT.parents
    assert '"private_cache_outside_repo": True' in source
    assert '"private_transcripts_committed": False' in source
    assert 'os.environ.setdefault("HF_HUB_OFFLINE", "1")' in source
    assert 'os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")' in source
    assert "local_files_only=True" in source
    assert "http://" not in source
    assert "https://" not in source
    assert "git push" not in source.lower()
    assert "requests." not in source
    assert "httpx." not in source
    assert not ({"push", "deploy", "promote"} & called_functions)


def test_phase108_decision_can_never_qualify_or_auto_promote() -> None:
    source = _driver_source()
    perfect = {
        "accepted_rate": 1.0,
        "task_complete_rate": 1.0,
        "correction_followed_rate": 1.0,
        "preference_adherence_rate": 1.0,
        "factual_guard_rate": 1.0,
        "privacy_boundary_rate": 1.0,
        "false_block_rate": 0.0,
    }
    comparison = {
        "candidate_wins": 40,
        "benchmark_wins": 0,
        "improved_domain_count": 4,
    }
    decision = build_phase108_decision(
        runtime_metrics={
            "provenance_envelope_valid_rate": 1.0,
            "provenance_injection_resisted_rate": 1.0,
            "source_id_integrity_rate": 1.0,
            "simulated_usage_truth_rate": 1.0,
            "training_eligibility_truth_rate": 1.0,
        },
        metrics={
            "base": {**perfect, "accepted_rate": 0.8},
            "phase106_sft": {**perfect, "accepted_rate": 0.9},
            "phase107_dpo": perfect,
        },
        comparisons={
            "phase107_dpo_vs_base": comparison,
            "phase107_dpo_vs_phase106_sft": comparison,
        },
        phase107_remains_archive=True,
        targeted_training_executed=False,
        confirmation_passed=None,
    )

    assert decision["passed"] is True
    assert decision["product_gate_qualified"] is False
    assert decision["automatic_promotion_allowed"] is False
    assert decision["recommendation"] == "promote_after_manual_review"
    assert '"product_gate_qualified": False' in source
    assert '"automatic_promotion_allowed": False' in source
