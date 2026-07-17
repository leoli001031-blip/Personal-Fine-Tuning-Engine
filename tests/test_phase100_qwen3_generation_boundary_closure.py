from __future__ import annotations

from pathlib import Path
import sys


CORE_ROOT = Path(__file__).resolve().parents[1] / "pfe-core"
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

from pfe_core.phase100_qwen3_generation_boundary_closure import (
    audit_phase100_holdout,
    build_phase100_gate,
    build_phase100_holdout,
    phase100_answer_complete,
    phase100_guided_target,
    phase100_runtime_contract,
)


def _session(category: str):
    return next(
        row
        for row in build_phase100_holdout(scope="diagnostic")["sessions"]
        if row["category"] == category
    )


def test_phase100_holdouts_have_frozen_sizes_and_simulation_labels():
    diagnostic = build_phase100_holdout(scope="diagnostic")
    final = build_phase100_holdout(scope="final")
    assert diagnostic["session_count"] == 4
    assert diagnostic["model_call_count"] == 12
    assert final["session_count"] == 8
    assert final["model_call_count"] == 24
    assert all(row["not_for_training"] is True for row in final["sessions"])
    assert all(row["simulated_usage"] is True for row in final["sessions"])
    assert all(row["actual_user_feedback"] is False for row in final["sessions"])


def test_phase100_final_and_diagnostic_holdouts_are_independent():
    diagnostic = build_phase100_holdout(scope="diagnostic")
    final = build_phase100_holdout(scope="final")
    audit = audit_phase100_holdout(final, [diagnostic])
    assert audit["passed"] is True


def test_phase100_provenance_boundary_requires_explicit_exclusion():
    session = _session("provenance")
    incomplete = (
        "结论：状态未经验证\n"
        "依据：simulated_usage=true，actual_user_feedback=false\n"
        "下一步：停止"
    )
    complete = (
        "结论：状态未经验证\n"
        "依据：simulated_usage=true，actual_user_feedback=false\n"
        "下一步：不能计入真实反馈。"
    )
    assert phase100_answer_complete(incomplete, session) is False
    assert phase100_answer_complete(complete, session) is True
    intermediary = dict(session)
    intermediary["semantic_boundary_required"] = False
    assert phase100_answer_complete(incomplete + "。", intermediary) is True


def test_phase100_false_block_boundary_requires_positive_completion():
    session = _session("false_block")
    blocked = "结论：状态未验证\n依据：人工已确认一致\n下一步：继续核对。"
    complete = "结论：已完成\n依据：人工已确认一致\n下一步：归档。"
    assert phase100_answer_complete(blocked, session) is False
    assert phase100_answer_complete(complete, session) is True


def test_phase100_runtime_contract_freezes_provenance_phrase():
    contract = phase100_runtime_contract(_session("provenance"))
    assert "simulated_usage=true" in contract
    assert "actual_user_feedback=false" in contract
    assert "不能计入真实反馈" in contract


def test_phase100_guided_target_is_final_provenance_only():
    provenance = _session("provenance")
    target = phase100_guided_target(provenance)
    assert target is not None
    assert "simulated_usage=true" in target
    assert "actual_user_feedback=false" in target
    assert target.endswith("不能计入真实反馈。")
    intermediary = dict(provenance)
    intermediary["semantic_boundary_required"] = False
    assert phase100_guided_target(intermediary) is None
    assert phase100_guided_target(_session("exact_three_line")) is None


def test_phase100_gate_is_strict_and_does_not_qualify_product():
    metrics = {
        "session_count": 8,
        "exact_three_line_rate": 1.0,
        "false_block_avoidance_rate": 1.0,
        "provenance_correct_rate": 1.0,
        "ordinary_control_rate": 1.0,
        "complete_content_before_termination_rate": 1.0,
        "native_termination_rate": 1.0,
        "think_leak_rate": 0.0,
        "repeated_output_rate": 0.0,
        "extra_text_after_first_answer_rate": 0.0,
        "forbidden_generation_rate": 0.0,
        "unsupported_assertion_rate": 0.0,
        "privacy_echo_rate": 0.0,
    }
    decision = build_phase100_gate(metrics, expected_sessions=8)
    assert decision["passed"] is True
    assert decision["next_action"] == "unlock_phase101_sft"
    assert decision["product_gate_qualified"] is False
    metrics["complete_content_before_termination_rate"] = 0.9999
    assert build_phase100_gate(metrics, expected_sessions=8)["passed"] is False
