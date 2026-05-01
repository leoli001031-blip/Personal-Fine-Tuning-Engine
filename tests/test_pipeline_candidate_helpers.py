from __future__ import annotations

from pfe_core.pipeline_candidate import (
    candidate_history_payload,
    candidate_history_summary,
    candidate_timeline_payload,
    candidate_timeline_stage,
    candidate_timeline_summary,
    normalize_candidate_history,
)


def test_candidate_history_helpers_normalize_and_summarize() -> None:
    history = normalize_candidate_history(
        [
            {"action": "promote_candidate", "status": "completed", "candidate_version": "v1"},
            "ignored",
            {"action": "archive_candidate", "status": "blocked", "reason": "needs_review"},
        ]
    )

    summary = candidate_history_summary(history)
    payload = candidate_history_payload(history=history, workspace="demo", limit=1)

    assert len(history) == 2
    assert summary["count"] == 2
    assert summary["last_action"] == "archive_candidate"
    assert summary["last_reason"] == "needs_review"
    assert summary["action_counts"] == {"promote_candidate": 1, "archive_candidate": 1}
    assert payload["workspace"] == "demo"
    assert payload["limit"] == 1
    assert payload["items"] == [history[-1]]


def test_candidate_timeline_helpers_derive_stages_and_labels() -> None:
    history = [
        {"timestamp": "t1", "action": "promote_candidate", "status": "completed"},
        {"timestamp": "t2", "action": "archive_candidate", "status": "completed"},
        {"timestamp": "t3", "action": "promote_candidate", "status": "blocked", "reason": "no_candidate"},
    ]

    summary = candidate_timeline_summary(history)
    payload = candidate_timeline_payload(history=history, limit=2)

    assert candidate_timeline_stage("promote_candidate", "completed") == "promoted"
    assert candidate_timeline_stage("archive_candidate", "completed") == "archived"
    assert summary["current_stage"] == "blocked"
    assert summary["transition_count"] == 3
    assert payload["items"][0]["stage"] == "archived"
    assert payload["items"][0]["label"] == "archive_candidate:completed"
    assert payload["items"][1]["stage"] == "blocked"
    assert payload["items"][1]["label"] == "promote_candidate:blocked"
