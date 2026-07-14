from __future__ import annotations

from pfe_core.phase70_structured_boundary_contract import stable_hash
from pfe_core.phase72_deterministic_boundary_serializer import classify_phase72_boundary
from pfe_core.phase74_shared_raw_deterministic_serializer_ab import (
    PHASE74_EXACT_OUTPUT,
    audit_phase74_parity,
    build_phase74_holdout,
    derive_phase74_transcripts,
)


def _messages(row: dict[str, object]) -> list[dict[str, str]]:
    return [
        {"role": "user", "content": str(row[key])}
        for key in (
            "user_goal",
            "user_correction",
            "continuation_request",
            "acceptance_request",
        )
    ]


def _raw(session: dict[str, object]) -> dict[str, object]:
    turns = []
    for text in _messages(session)[:3]:
        turns.extend((text, {"role": "assistant", "content": "共享模型原始输出"}))
    return {
        "session_id": session["session_id"],
        "task_type": session["task_type"],
        "category": session["category"],
        "model_id": "test-model",
        "task_sha256": stable_hash(session),
        "status": "completed",
        "turns": turns,
    }


def test_phase74_holdout_is_independent_and_routes_exactly() -> None:
    holdout = build_phase74_holdout()

    assert holdout["session_count"] == 54
    assert holdout["boundary_session_count"] == 36
    assert holdout["ordinary_session_count"] == 18
    assert all(row["session_id"].startswith("phase74-") for row in holdout["sessions"])
    assert all(row["actual_user_feedback"] is False for row in holdout["sessions"])
    assert all(row["not_for_training"] is True for row in holdout["sessions"])
    assert all(
        classify_phase72_boundary(_messages(row))["routed"]
        is (row["task_type"] == "boundary")
        for row in holdout["sessions"]
    )


def test_phase74_derives_both_arms_from_identical_raw_transcript() -> None:
    holdout = build_phase74_holdout()
    sessions = {row["session_id"]: row for row in holdout["sessions"][:2]}
    raw = [_raw(row) for row in sessions.values()]
    derived = derive_phase74_transcripts(raw, sessions)

    assert len(derived["structured_prompt_raw"]) == 2
    assert len(derived["deterministic_boundary_serializer"]) == 2
    for baseline, candidate in zip(
        derived["structured_prompt_raw"],
        derived["deterministic_boundary_serializer"],
        strict=True,
    ):
        assert baseline["shared_raw_transcript_sha256"] == candidate[
            "shared_raw_transcript_sha256"
        ]
        if candidate["task_type"] == "boundary":
            assert candidate["turns"][-1]["content"] == PHASE74_EXACT_OUTPUT


def test_phase74_parity_requires_shared_raw_and_ordinary_identity() -> None:
    holdout = build_phase74_holdout()
    selected = [
        next(row for row in holdout["sessions"] if row["task_type"] == task_type)
        for task_type in ("boundary", "ordinary")
    ]
    sessions = {row["session_id"]: row for row in selected}
    derived = derive_phase74_transcripts([_raw(row) for row in selected], sessions)

    audit = audit_phase74_parity(derived, selected)

    assert audit["passed"] is True
    assert audit["session_count"] == 2
    assert audit["failed_checks"] == []
