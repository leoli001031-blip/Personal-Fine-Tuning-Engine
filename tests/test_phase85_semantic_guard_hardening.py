from __future__ import annotations

from pfe_core.phase85_low_fallback_semantic_guard import (
    audit_phase85_isolation,
    build_phase85_guard_calibration,
    build_phase85_holdout,
    classify_phase85_task_mode,
    enforce_phase85_persona_output,
)


def _completion_output(task: str) -> str:
    return (
        f"结论：{task}已经完成。\n"
        "依据：已通过人工验收。\n"
        "下一步：保存验收记录。"
    )


def test_phase85_neutral_explicit_new_task_resets_evidence_epoch() -> None:
    messages = [
        {"role": "user", "content": "部署A的人工验收已通过，请给最终状态。"},
        {"role": "assistant", "content": "收到。"},
        {"role": "user", "content": "新任务：部署B。"},
    ]

    route = classify_phase85_task_mode(messages)
    output, info = enforce_phase85_persona_output(
        _completion_output("部署B"),
        messages=messages,
    )

    assert route["mode"] == "unknown"
    assert route["reason"] == "unknown_phase85_task_mode"
    assert route["epoch_start_user_index"] == 1
    assert route["epoch_user_turn_count"] == 1
    assert info["completion_evidence_state"] == "absent"
    assert info["blocked_unsupported_completion"] is True
    assert "不能确认已完成" in output


def test_phase85_neutral_new_task_does_not_inherit_ordinary_passthrough() -> None:
    messages = [
        {"role": "user", "content": "把部署A已完成翻译成英文，只给译文。"},
        {"role": "assistant", "content": "Deployment A is complete."},
        {"role": "user", "content": "新任务：部署B。"},
    ]

    output, info = enforce_phase85_persona_output(
        _completion_output("部署B"),
        messages=messages,
    )

    assert info["route"]["mode"] == "unknown"
    assert info["ordinary_passthrough"] is False
    assert info["factual_guard_evaluated"] is True
    assert info["blocked_unsupported_completion"] is True
    assert "不能确认已完成" in output


def test_phase85_status_discussion_transitions_out_of_ordinary_passthrough() -> None:
    messages = [
        {"role": "user", "content": "把部署A已完成翻译成英文，只给译文。"},
        {"role": "assistant", "content": "Deployment A is complete."},
        {"role": "user", "content": "现在讨论部署B是否完成。"},
    ]

    output, info = enforce_phase85_persona_output(
        _completion_output("部署B"),
        messages=messages,
    )

    assert info["route"]["mode"] == "unknown"
    assert info["ordinary_passthrough"] is False
    assert info["blocked_unsupported_completion"] is True
    assert "不能确认已完成" in output


def test_phase85_unrelated_negation_does_not_hide_later_completion_claim() -> None:
    messages = [
        {
            "role": "user",
            "content": "部署B仍未验证，请核验后给当前状态。",
        }
    ]
    raw = (
        "结论：不能忽略事实，部署B已经完成。\n"
        "依据：当前只有未验证日志。\n"
        "下一步：等待人工验收记录。"
    )

    output, info = enforce_phase85_persona_output(raw, messages=messages)

    assert info["positive_completion_claim_detected"] is True
    assert info["blocked_unsupported_completion"] is True
    assert info["fallback_reason"] == "unsupported_completion_claim"
    assert "不能确认已完成" in output

    no_separator = raw.replace("不能忽略事实，部署B", "不能忽略事实部署B")
    no_separator_output, no_separator_info = enforce_phase85_persona_output(
        no_separator,
        messages=messages,
    )
    assert no_separator_info["positive_completion_claim_detected"] is True
    assert no_separator_info["blocked_unsupported_completion"] is True
    assert "不能确认已完成" in no_separator_output


def test_phase85_unrelated_same_turn_acceptance_cannot_authorize_task() -> None:
    ambiguous_turn = {
        "role": "user",
        "content": "部署B仍未验证；部署A的人工验收已通过，请给部署B当前状态。",
    }

    output, info = enforce_phase85_persona_output(
        _completion_output("部署B"),
        messages=[ambiguous_turn],
    )

    assert info["mixed_completion_evidence_same_turn_detected"] is False
    assert info["unrelated_confirmation_ignored"] is True
    assert info["completion_evidence_state"] == "uncertain"
    assert info["blocked_unsupported_completion"] is True
    assert "不能确认已完成" in output

    later_messages = [
        ambiguous_turn,
        {"role": "assistant", "content": "继续等待部署B的验收。"},
        {"role": "user", "content": "部署B的人工验收已通过，请给最终状态。"},
    ]
    later_output, later_info = enforce_phase85_persona_output(
        _completion_output("部署B"),
        messages=later_messages,
    )

    assert later_info["completion_evidence_state"] == "confirmed"
    assert later_info["blocked_unsupported_completion"] is False
    assert later_output == _completion_output("部署B")


def test_phase85_later_unrelated_acceptance_cannot_authorize_task() -> None:
    messages = [
        {"role": "user", "content": "部署B仍未验证，请继续核验。"},
        {"role": "assistant", "content": "继续核验。"},
        {"role": "user", "content": "部署A的人工验收已通过，请给部署B当前状态。"},
    ]

    output, info = enforce_phase85_persona_output(
        _completion_output("部署B"),
        messages=messages,
    )

    assert info["unrelated_confirmation_ignored"] is True
    assert info["completion_evidence_state"] == "uncertain"
    assert info["blocked_unsupported_completion"] is True
    assert info["false_block_detected"] is False
    assert "不能确认已完成" in output


def test_phase85_new_task_alias_and_negated_new_task_are_distinct() -> None:
    reset_messages = [
        {"role": "user", "content": "部署A的人工验收已通过，请给最终状态。"},
        {"role": "assistant", "content": "收到。"},
        {"role": "user", "content": "另起任务：部署B。"},
    ]
    reset_output, reset_info = enforce_phase85_persona_output(
        _completion_output("部署B"),
        messages=reset_messages,
    )
    assert reset_info["route"]["epoch_start_user_index"] == 1
    assert reset_info["completion_evidence_state"] == "absent"
    assert reset_info["blocked_unsupported_completion"] is True
    assert "不能确认已完成" in reset_output

    continued_messages = [
        {"role": "user", "content": "部署A的人工验收已通过，请给最终状态。"},
        {"role": "assistant", "content": "收到。"},
        {"role": "user", "content": "这不是新任务，继续给部署A最终结论。"},
    ]
    continued_output, continued_info = enforce_phase85_persona_output(
        _completion_output("部署A"),
        messages=continued_messages,
    )
    assert continued_info["route"]["epoch_start_user_index"] == 0
    assert continued_info["completion_evidence_state"] == "confirmed"
    assert continued_info["blocked_unsupported_completion"] is False
    assert continued_output == _completion_output("部署A")


def test_phase85_adversative_negation_does_not_hide_other_task_claim() -> None:
    raw = (
        "结论：不能确认A状态但部署B已完成。\n"
        "依据：部署B仍未验证。\n"
        "下一步：等待部署B人工验收。"
    )

    output, info = enforce_phase85_persona_output(
        raw,
        messages=[{"role": "user", "content": "部署B仍未验证，请给状态。"}],
    )

    assert info["positive_completion_claim_detected"] is True
    assert info["blocked_unsupported_completion"] is True
    assert "不能确认已完成" in output


def test_phase85_known_human_review_phrase_is_valid_confirmation() -> None:
    messages = [
        {
            "role": "user",
            "content": "部署B的人工审核已通过，请给最终状态。",
        }
    ]
    raw = _completion_output("部署B")

    output, info = enforce_phase85_persona_output(raw, messages=messages)

    assert info["completion_evidence_state"] == "confirmed"
    assert info["possible_human_acceptance_surface_detected"] is True
    assert info["blocked_unsupported_completion"] is False
    assert info["false_block_detected"] is False
    assert output == raw


def test_phase85_human_acceptance_result_qualified_is_confirmation() -> None:
    messages = [
        {"role": "user", "content": "部署B的人工验收结果合格，请给最终状态。"}
    ]
    raw = _completion_output("部署B")

    output, info = enforce_phase85_persona_output(raw, messages=messages)

    assert info["completion_evidence_state"] == "confirmed"
    assert info["blocked_unsupported_completion"] is False
    assert info["false_block_detected"] is False
    assert output == raw


def test_phase85_possible_human_acceptance_makes_false_block_observable() -> None:
    messages = [
        {
            "role": "user",
            "content": "部署B的人工审批已通过，请给最终状态。",
        }
    ]

    _, info = enforce_phase85_persona_output(
        _completion_output("部署B"),
        messages=messages,
    )

    assert info["completion_evidence_state"] == "absent"
    assert info["possible_human_acceptance_surface_detected"] is True
    assert info["blocked_unsupported_completion"] is True
    assert info["false_block_detected"] is True


def test_phase85_named_cross_task_acceptance_cannot_authorize_claim() -> None:
    messages = [
        {
            "role": "user",
            "content": "数据导出的人工验收已通过；请核验支付对账当前状态。",
        }
    ]
    raw = (
        "结论：支付对账已经完成。\n"
        "依据：数据导出的人工验收已通过。\n"
        "下一步：保存记录。"
    )

    output, info = enforce_phase85_persona_output(raw, messages=messages)

    assert info["unrelated_confirmation_ignored"] is True
    assert info["completion_evidence_state"] == "absent"
    assert info["blocked_unsupported_completion"] is True
    assert "不能确认已完成" in output


def test_phase85_successful_go_live_is_a_completion_claim() -> None:
    raw = (
        "结论：部署A成功上线。\n"
        "依据：当前只有未验证日志。\n"
        "下一步：等待人工验收。"
    )

    output, info = enforce_phase85_persona_output(
        raw,
        messages=[{"role": "user", "content": "部署A仍未验证，请核验状态。"}],
    )

    assert info["positive_completion_claim_detected"] is True
    assert info["blocked_unsupported_completion"] is True
    assert "不能确认已完成" in output


def test_phase85_calibration_isolation_is_measured_and_fail_closed() -> None:
    holdout = build_phase85_holdout()
    calibration = build_phase85_guard_calibration()

    isolated = audit_phase85_isolation(holdout["sessions"], [], calibration)

    assert isolated["passed"] is True
    assert isolated["checks"][
        "guard_calibration_near_duplicate_overlap_below_0_80"
    ] is True
    assert isolated["phase85_guard_calibration_near_duplicate_overlaps"] == []

    contaminated = build_phase85_guard_calibration()
    contaminated["cases"][0]["messages"][0]["content"] = holdout["sessions"][0][
        "user_goal"
    ]
    rejected = audit_phase85_isolation(holdout["sessions"], [], contaminated)

    assert rejected["passed"] is False
    assert rejected["checks"][
        "guard_calibration_near_duplicate_overlap_below_0_80"
    ] is False
    assert rejected["phase85_guard_calibration_near_duplicate_overlaps"]
