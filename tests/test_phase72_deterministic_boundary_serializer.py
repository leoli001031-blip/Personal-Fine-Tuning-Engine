from __future__ import annotations

from pfe_core.phase63_field_typed_candidate_wire import PHASE63_WIRE_PATTERN
from pfe_core.phase72_deterministic_boundary_serializer import (
    PHASE72_EXACT_OUTPUT,
    apply_phase72_serializer,
    build_phase72_explicit_typed_wire_prompt,
    build_phase72_holdout,
    build_phase72_transport_preflight_cases,
    classify_phase72_boundary,
)


def _messages(*values: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": value} for value in values]


def test_phase72_router_covers_boundary_and_excludes_transform_tasks() -> None:
    boundary = classify_phase72_boundary(
        _messages("内部测试全绿。", "现在能否证明真实用户结果已经改善？")
    )
    translation = classify_phase72_boundary(
        _messages("把 actual_user_feedback 翻译成中文，不要判断登记规则。")
    )

    assert boundary["routed"] is True
    assert translation["routed"] is False
    assert "翻译" in translation["transform_exclusion_hits"]


def test_phase72_serializer_is_exact_and_passthrough_is_lossless() -> None:
    serialized, manifest = apply_phase72_serializer(
        _messages("脚本结果能否证明真实用户结果？"), "模型自由输出"
    )
    passthrough, ordinary_manifest = apply_phase72_serializer(
        _messages("把‘等待确认’翻译成英文。"), "Pending confirmation"
    )

    assert serialized == PHASE72_EXACT_OUTPUT
    assert manifest["serializer_applied"] is True
    assert passthrough == "Pending confirmation"
    assert ordinary_manifest["serializer_applied"] is False


def test_phase72_holdout_routes_all_boundary_and_no_ordinary_sessions() -> None:
    holdout = build_phase72_holdout()
    decisions = {}
    for row in holdout["sessions"]:
        messages = _messages(
            row["user_goal"],
            row["user_correction"],
            row["continuation_request"],
            row["acceptance_request"],
        )
        decisions[row["session_id"]] = classify_phase72_boundary(messages)["routed"]

    assert holdout["session_count"] == 54
    assert holdout["boundary_session_count"] == 36
    assert holdout["ordinary_session_count"] == 18
    assert all(
        decisions[row["session_id"]] is (row["task_type"] == "boundary")
        for row in holdout["sessions"]
    )
    assert all(row["actual_user_feedback"] is False for row in holdout["sessions"])
    assert all(row["not_for_training"] is True for row in holdout["sessions"])


def test_phase72_transport_preflight_covers_historical_placeholder_shape() -> None:
    cases = build_phase72_transport_preflight_cases()["cases"]
    target = next(row for row in cases if row["category"] == "source_outcome_no_relation")
    prompt = build_phase72_explicit_typed_wire_prompt(target)

    assert len(cases) == 18
    assert target["expected_candidate_ids"]["test_to_user_outcome_relation"] == "none"
    assert "test_to_user_outcome_relation allowed tokens: none" in prompt
    assert "rNNN" not in prompt
    assert "不得输出 NNN" in prompt
    assert PHASE63_WIRE_PATTERN.fullmatch("PFE2|s001|u001|none")
