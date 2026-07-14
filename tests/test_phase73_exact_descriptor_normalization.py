from __future__ import annotations

import pytest

from pfe_core.phase59_proposition_addressed_grounding import build_phase59_proposition_candidates
from pfe_core.phase73_exact_descriptor_normalization import (
    audit_phase73_historical_failure_shapes,
    build_phase73_fresh_transport_cases,
    build_phase73_typed_wire_prompt,
    normalize_phase73_typed_wire,
)


SOURCE_RELATION = (
    "反馈登记：当前模拟、脚本、测试或内部材料不得登记为 actual_user_feedback。\n"
    "证据关系：当前测试证据不能证明真实用户结果。"
)
SOURCE_OUTCOME = "非真实材料不可登记为实际用户反馈。真实用户结果仍未确认。"


def test_phase73_normalizes_only_exact_listed_descriptors() -> None:
    candidates = build_phase59_proposition_candidates(SOURCE_RELATION)
    result = normalize_phase73_typed_wire(
        "PFE2|s001=exclude_actual@c001|none only|r001=does_not_establish@c002",
        candidates=candidates,
    )

    assert result["normalized_wire"] == "PFE2|s001|none|r001"
    assert result["normalization_applied"] is True
    assert result["normalization_count"] == 3
    assert result["source_registration_candidate_id"] == "p001"
    assert result["user_outcome_status_candidate_id"] == "none"
    assert result["test_to_user_outcome_relation_candidate_id"] == "p002"


@pytest.mark.parametrize(
    "wire",
    (
        "PFE2|s001=allow_actual@c001|none only|r001=does_not_establish@c002",
        "PFE2|r001=does_not_establish@c002|none only|s001=exclude_actual@c001",
        "PFE2|s001=exclude_actual@c999|none only|r001=does_not_establish@c002",
        "PFE2|s001=exclude_actual@c001|u001=suspended_or_negated@c002|r001=does_not_establish@c002",
        "PFE2|s001=exclude_actual@c001|none only|r001=does_not_establish@c002|extra",
        " PFE2|s001|none|r001",
    ),
)
def test_phase73_rejects_mismatch_cross_field_unknown_and_extra_text(wire: str) -> None:
    candidates = build_phase59_proposition_candidates(SOURCE_RELATION)
    with pytest.raises(ValueError):
        normalize_phase73_typed_wire(wire, candidates=candidates)


def test_phase73_rejects_none_only_when_field_has_candidates() -> None:
    candidates = build_phase59_proposition_candidates(SOURCE_OUTCOME)
    with pytest.raises(ValueError, match="has candidates"):
        normalize_phase73_typed_wire(
            "PFE2|s001=exclude_actual@c001|none only|none only",
            candidates=candidates,
        )


def test_phase73_preserves_strict_token_wire() -> None:
    candidates = build_phase59_proposition_candidates(SOURCE_OUTCOME)
    result = normalize_phase73_typed_wire(
        "PFE2|s001|u001|none", candidates=candidates
    )

    assert result["normalized_wire"] == "PFE2|s001|u001|none"
    assert result["normalization_applied"] is False
    assert result["strict_token_wire"] is True


def test_phase73_fresh_cases_are_complete_detector_compatible_and_isolated() -> None:
    bundle = build_phase73_fresh_transport_cases()

    assert bundle["case_count"] == 24
    assert bundle["actual_user_feedback_count"] == 0
    assert bundle["not_for_training"] is True
    assert len({row["assistant_response"] for row in bundle["cases"]}) == 24
    assert all(row["case_id"].startswith("phase73-wire-") for row in bundle["cases"])
    assert all(row["expected_label"] in {"accept", "edit", "reject"} for row in bundle["cases"])


def test_phase73_prompt_keeps_token_only_request() -> None:
    case = build_phase73_fresh_transport_cases()["cases"][5]
    prompt = build_phase73_typed_wire_prompt(case)

    assert "PFE2|source_token|outcome_token|relation_token" in prompt
    assert "只复制等号前 token" in prompt
    assert "source_registration allowed tokens" in prompt


def test_phase73_historical_replay_is_not_counted_as_new_model_output() -> None:
    audit = audit_phase73_historical_failure_shapes(
        (
            "PFE2|s001=exclude_actual@c001|none only|r001=does_not_establish@c002",
            "PFE2|s001=exclude_actual@c001|u001=suspended_or_negated@c002|none only",
        ),
        assistant_responses=(SOURCE_RELATION, SOURCE_OUTCOME),
    )

    assert audit["passed"] is True
    assert audit["count"] == 2
    assert audit["counted_as_phase73_model_outputs"] is False
