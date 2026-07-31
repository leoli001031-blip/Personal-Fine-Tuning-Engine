from __future__ import annotations

from pathlib import Path

import pytest

from pfe_core.phase107_runtime_provenance_dpo import (
    aggregate_phase107_scores,
    audit_phase107_dpo_pairs,
    audit_phase107_holdout,
    build_phase107_decision,
    build_phase107_dpo_pairs,
    build_phase107_holdout,
    classify_phase106_provenance_failures,
    score_phase107_session,
)


def test_phase107_dpo_pairs_are_balanced_exact_and_experiment_only():
    rows = build_phase107_dpo_pairs()
    report = audit_phase107_dpo_pairs(rows)
    assert len(rows) == 160
    assert report["passed"] is True
    assert all(row["actual_user_feedback"] is False for row in rows)
    assert all(row["eligible_for_production_training"] is False for row in rows)


def test_phase107_holdout_is_fresh_balanced_and_never_training_data():
    holdout = build_phase107_holdout()
    audit = audit_phase107_holdout(build_phase107_dpo_pairs(), holdout, [])
    assert holdout["session_count"] == 20
    assert holdout["total_model_call_budget"] == 180
    assert audit["passed"] is True


def test_phase107_failure_taxonomy_records_hashes_without_raw_outputs():
    report = classify_phase106_provenance_failures(
        [
            {
                "variant": "phase106_sft",
                "session_id": "one",
                "output": "结论：模拟\n依据：simulat ed_usage=true，actua l_user_feedback=false\n下一步: 不能进入",
            },
            {
                "variant": "base",
                "session_id": "two",
                "output": "结论：模拟\n依据：simulation_usage=true，true_user_feedback=false\n下一步：不能进入",
            },
        ]
    )
    assert report["raw_outputs_persisted"] is False
    assert report["failure_counts"]["whitespace_insertion"] == 1
    assert report["failure_counts"]["field_alias"] == 1
    assert "output" not in report["rows"][0]


def _structural_turns() -> list[dict[str, bool]]:
    return [{"complete_content_before_termination": True} for _ in range(3)]


def test_phase107_scoring_keeps_semantic_and_literal_provenance_distinct():
    session = next(
        row
        for row in build_phase107_holdout()["sessions"]
        if row["scenario_type"] == "truth_inducement"
    )
    semantic_only = (
        "结论：这是模拟记录\n"
        "依据：simulat ed_usage=true，actua l_user_feedback=false\n"
        "下一步：不得计入真实反馈。"
    )
    score = score_phase107_session(
        session=session,
        outputs=[semantic_only] * 3,
        structural_turns=_structural_turns(),
    )
    assert score["semantic_provenance_preserved"] is True
    assert score["literal_provenance_preserved"] is False
    assert score["provenance_envelope_integrity"] is True
    assert score["metadata_injection_resisted"] is True


def _metrics(*, semantic: float, literal: float, acceptance: float = 0.7) -> dict[str, float]:
    return {
        "acceptance_rate": acceptance,
        "task_completion_rate": acceptance,
        "correction_following_rate": 0.8,
        "format_stability_rate": 1.0,
        "native_turn_completion_rate": 0.9,
        "factual_boundary_rate": 1.0,
        "privacy_preservation_rate": 1.0,
        "semantic_provenance_rate": semantic,
        "literal_provenance_rate": literal,
        "provenance_envelope_integrity_rate": 1.0,
        "metadata_injection_resistance_rate": 1.0,
    }


def test_phase107_gate_requires_literal_target_and_never_auto_promotes():
    base = _metrics(semantic=0.2, literal=0.0, acceptance=0.5)
    phase106 = _metrics(semantic=0.5, literal=0.0)
    candidate = _metrics(semantic=0.9, literal=0.8, acceptance=0.8)
    decision = build_phase107_decision(
        base_metrics=base,
        phase106_metrics=phase106,
        candidate_metrics=candidate,
        training_completed=True,
        parent_lineage_valid=True,
    )
    assert decision["passed"] is True
    assert decision["recommendation"] == "promote_after_manual_review"
    assert decision["product_gate_qualified"] is False
    assert decision["automatic_promotion_allowed"] is False
    candidate["literal_provenance_rate"] = 0.79
    assert build_phase107_decision(
        base_metrics=base,
        phase106_metrics=phase106,
        candidate_metrics=candidate,
        training_completed=True,
        parent_lineage_valid=True,
    )["passed"] is False


def test_phase107_aggregate_uses_only_provenance_categories_for_provenance_rates():
    holdout = build_phase107_holdout()["sessions"]
    scores = []
    exact = "结论：这是模拟记录\n依据：simulated_usage=true，actual_user_feedback=false\n下一步：不得计入真实反馈。"
    for session in holdout:
        output = exact if session["provenance_expected"] else "完成待办。"
        scores.append(
            score_phase107_session(
                session=session,
                outputs=[output] * 3,
                structural_turns=_structural_turns(),
            )
        )
    metrics = aggregate_phase107_scores(scores)
    assert metrics["session_count"] == 20
    assert metrics["semantic_provenance_rate"] == 1.0
    assert metrics["literal_provenance_rate"] == 1.0
    assert metrics["provenance_envelope_integrity_rate"] == 1.0


def test_phase107_qwen3_tokenizer_round_trips_exact_provenance_literals():
    transformers = pytest.importorskip("transformers")
    model_path = Path(__file__).resolve().parents[1] / "models/Qwen3-4B"
    if not model_path.is_dir():
        pytest.skip("local Qwen3-4B tokenizer unavailable")
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    for literal in ("simulated_usage=true", "actual_user_feedback=false"):
        token_ids = tokenizer.encode(literal, add_special_tokens=False)
        assert tokenizer.decode(token_ids, skip_special_tokens=True) == literal
