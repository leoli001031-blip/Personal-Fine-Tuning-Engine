from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

import pfe_cli.main as cli_main
from pfe_core.phase35_local_interaction_capture import (
    PHASE35_SIMULATED_SOURCE,
    append_phase35_capture_batch,
    build_phase35_capture_batch,
    build_phase35_comparison_summary,
    build_phase35_interaction_record,
    build_phase35_phase34_review,
    build_phase35_readiness,
    build_phase35_review_queue,
    load_phase35_state,
    phase35_store_path,
    render_phase35_agent_response,
    validate_phase35_interaction_record,
)


def _response(goal: str = "帮我整理当前状态。") -> str:
    rendered = render_phase35_agent_response(user_goal=goal, model_variant="adapter")
    return str(rendered["assistant_response"])


def test_phase35_simulated_interaction_is_non_training() -> None:
    record = build_phase35_interaction_record(
        workspace="phase35-test",
        user_goal="帮我整理当前状态。",
        assistant_response=_response(),
        feedback_action="accept",
        user_feedback="模拟反馈，不是真实用户反馈。",
        simulated_local_interaction=True,
    )

    validation = validate_phase35_interaction_record(record)
    batch = build_phase35_capture_batch([record])

    assert record["source"] == PHASE35_SIMULATED_SOURCE
    assert record["feedback_source"] == PHASE35_SIMULATED_SOURCE
    assert record["confirmed_actual_user_feedback"] is False
    assert record["eligible_for_training"] is False
    assert validation["status"] == "non_training"
    assert "simulated_local_interaction_not_actual_feedback" in validation["non_training_reasons"]
    assert batch["accepted_pending_review_count"] == 0
    assert batch["non_training_count"] == 1
    assert batch["auto_training_allowed"] is False


def test_phase35_actual_local_feedback_requires_full_attestation() -> None:
    record = build_phase35_interaction_record(
        workspace="phase35-test",
        user_goal="这次回答是否真的有用？",
        assistant_response=_response("这次回答是否真的有用？"),
        feedback_action="correction",
        user_feedback="有效，但需要先做真实检查再下结论。",
        operator_id="local-user",
        confirmed_actual_user_feedback=True,
        consent_for_training_candidate_review=True,
        not_scripted_or_curated=True,
    )
    batch = build_phase35_capture_batch([record])

    assert record["feedback_source"] == "actual_user_feedback"
    assert record["eligible_for_phase36_review"] is True
    assert record["eligible_for_training"] is False
    assert batch["accepted_pending_review_count"] == 1
    assert batch["accepted_pending_review"][0]["review_state"] == "pending_review"
    assert batch["accepted_pending_review"][0]["validation"]["passed"] is True


def test_phase35_unattested_or_partial_actual_feedback_never_enters_review() -> None:
    unattested = build_phase35_interaction_record(
        workspace="phase35-test",
        user_goal="普通本地试用。",
        assistant_response=_response("普通本地试用。"),
        feedback_action="accept",
        user_feedback="没有真实反馈确认。",
    )
    partial = build_phase35_interaction_record(
        workspace="phase35-test",
        user_goal="半确认记录。",
        assistant_response=_response("半确认记录。"),
        feedback_action="accept",
        user_feedback="只确认了真实反馈，但没有 consent/operator。",
        confirmed_actual_user_feedback=True,
    )

    batch = build_phase35_capture_batch([unattested, partial])

    assert batch["accepted_pending_review_count"] == 0
    assert batch["non_training_count"] == 1
    assert batch["blocked_count"] == 1
    assert batch["non_training"][0]["validation"]["non_training_reasons"] == [
        "actual_feedback_attestation_required"
    ]
    assert "consent_for_training_candidate_review_required" in batch["blocked"][0]["validation"]["errors"]
    assert "operator_id_required" in batch["blocked"][0]["validation"]["errors"]


def test_phase35_private_text_is_quarantined() -> None:
    record = build_phase35_interaction_record(
        workspace="phase35-test",
        user_goal="不要提交私密正文。",
        assistant_response=_response("不要提交私密正文。"),
        feedback_action="reject",
        user_feedback="/Users/lichenhao/AgentMemory/Conversations/2026-06-22_14-01_Codex.md",
        operator_id="local-user",
        confirmed_actual_user_feedback=True,
        consent_for_training_candidate_review=True,
        not_scripted_or_curated=True,
    )
    batch = build_phase35_capture_batch([record])

    assert batch["accepted_pending_review_count"] == 0
    assert batch["quarantined_count"] == 1
    assert "raw_private_text_detected" in batch["quarantined"][0]["validation"]["quarantine_reasons"]


def test_phase35_persistence_review_queue_and_readiness(tmp_path: Path) -> None:
    store = phase35_store_path(tmp_path, "phase35-test")
    record = build_phase35_interaction_record(
        workspace="phase35-test",
        user_goal="下一步怎么推进？",
        assistant_response=_response("下一步怎么推进？"),
        feedback_action="final_acceptance",
        user_feedback="这个回答符合我想要的执行方式。",
        operator_id="local-user",
        confirmed_actual_user_feedback=True,
        consent_for_training_candidate_review=True,
        not_scripted_or_curated=True,
    )

    state = append_phase35_capture_batch(store, build_phase35_capture_batch([record]))
    reloaded = load_phase35_state(store)
    queue = build_phase35_review_queue(reloaded)
    readiness = build_phase35_readiness(state)

    assert len(reloaded["interactions"]) == 1
    assert queue["pending_review_count"] == 1
    assert readiness["ready_for_phase36_review"] is True
    assert readiness["training_status"] == "blocked"
    assert readiness["auto_training_allowed"] is False


def test_phase35_cli_captures_simulated_and_actual_local_interactions(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("PFE_HOME", str(tmp_path / ".pfe"))
    runner = CliRunner()

    simulated = runner.invoke(
        cli_main.app,
        [
            "phase35",
            "interact",
            "--workspace",
            "phase35-cli-test",
            "--user-goal",
            "帮我判断下一步。",
            "--feedback-action",
            "accept",
            "--user-feedback",
            "simulated only",
            "--simulated-local-interaction",
        ],
    )
    assert simulated.exit_code == 0, simulated.stdout
    assert "Phase35 local interaction captured" in simulated.stdout
    assert "Non Training: 1" in simulated.stdout
    assert "Pending Review: 0" in simulated.stdout

    actual = runner.invoke(
        cli_main.app,
        [
            "phase35",
            "interact",
            "--workspace",
            "phase35-cli-test",
            "--user-goal",
            "真实本地交互验收。",
            "--feedback-action",
            "accept",
            "--user-feedback",
            "这条可进入人工 review。",
            "--operator-id",
            "local-user",
            "--confirm-actual-user-feedback",
            "--consent-for-training-candidate-review",
            "--not-scripted-or-curated",
        ],
    )
    assert actual.exit_code == 0, actual.stdout
    assert "Feedback Source: actual_user_feedback" in actual.stdout
    assert "Accepted Pending Review: 1" in actual.stdout
    assert "Ready For Phase36 Review: True" in actual.stdout

    queue = runner.invoke(cli_main.app, ["phase35", "review-queue", "--workspace", "phase35-cli-test"])
    assert queue.exit_code == 0, queue.stdout
    assert "Pending Review: 1" in queue.stdout
    assert "actual_user_feedback" in queue.stdout


def test_phase35_phase34_regression_and_summary_do_not_auto_train_or_use_hermes() -> None:
    phase34_review = build_phase35_phase34_review(
        phase34_summary={
            "status": "completed",
            "actual_user_feedback_count": 0,
            "final_recommendation": "promote_after_manual_review",
            "acceptance_scores": {"adapter_win_rate": 0.8, "base_win_rate": 0.2},
        }
    )
    capture_batch = build_phase35_capture_batch([])
    state = {
        "kind": "phase35_persisted_state",
        "interactions": [],
        "capture_batches": [],
        "review_decisions": [],
    }
    readiness = build_phase35_readiness(state)
    summary = build_phase35_comparison_summary(
        phase34_review=phase34_review,
        capture_batch=capture_batch,
        state=state,
        readiness=readiness,
    )

    assert phase34_review["hermes_integration_required"] is False
    assert summary["hermes_integration_used"] is False
    assert summary["actual_training_run"] is False
    assert summary["auto_training_allowed"] is False
    assert summary["auto_promotion_allowed"] is False
    assert summary["final_recommendation"] == "capture_attested_actual_local_interactions"
    assert json.dumps(summary, ensure_ascii=False)
