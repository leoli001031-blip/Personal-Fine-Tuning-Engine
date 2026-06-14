from __future__ import annotations

from pfe_core.pipeline_queue import (
    normalize_queue_items,
    queue_history_entry,
    queue_history_summary,
    queue_recent_history,
    queue_review_entries,
    queue_review_policy_summary,
    queue_review_summary,
    queue_sort_key,
    queue_state_counts,
    train_queue_history_payload,
)


def test_queue_history_entry_and_sort_key() -> None:
    item = {
        "job_id": "job-1",
        "state": "queued",
        "priority": 7,
        "created_at": "2026-03-25T10:00:00+00:00",
    }

    entry = queue_history_entry(
        event="enqueued",
        item=item,
        reason="ready",
        metadata={"note": "operator"},
    )

    assert entry["timestamp"] == "2026-03-25T10:00:00+00:00"
    assert entry["event"] == "enqueued"
    assert entry["reason"] == "ready"
    assert entry["note"] == "operator"
    assert queue_sort_key(item) == (-7, "2026-03-25T10:00:00+00:00")


def test_queue_snapshot_history_and_review_summaries() -> None:
    items = normalize_queue_items(
        [
            {
                "job_id": "job-1",
                "state": "awaiting_confirmation",
                "confirmation_reason": "manual_review_required_by_policy",
                "history_count": 2,
                "history": [
                    {"timestamp": "2026-03-25T10:00:00+00:00", "event": "enqueued", "reason": "queued"},
                    {"timestamp": "2026-03-25T10:01:00+00:00", "event": "approved", "reason": "confirmation_approved"},
                ],
            },
            {
                "job_id": "job-2",
                "state": "completed",
                "history_count": 1,
                "history": [
                    {"timestamp": "2026-03-25T10:02:00+00:00", "event": "completed", "reason": "done"},
                ],
            },
        ]
    )

    recent = queue_recent_history(items)
    reviews = queue_review_entries(items)
    history = queue_history_summary(items, recent)
    review = queue_review_summary(review_entries=reviews, awaiting_item=items[0])
    policy = queue_review_policy_summary(
        queue_mode="deferred",
        require_queue_confirmation=True,
        awaiting_confirmation_count=1,
        awaiting_item=items[0],
        queued_item=None,
    )

    assert queue_state_counts(items) == {"awaiting_confirmation": 1, "completed": 1}
    assert history["transition_count"] == 3
    assert history["last_reason"] == "done"
    assert review["approved_transition_count"] == 1
    assert review["next_job_id"] == "job-1"
    assert policy["review_mode"] == "manual_review"
    assert policy["queue_entry_mode"] == "awaiting_confirmation"
    assert "reason=manual_review_required_by_policy" in policy["summary_line"]


def test_train_queue_history_payload_selects_job_and_available_ids() -> None:
    payload = {
        "last_item": {"job_id": "job-2", "state": "completed", "history": []},
        "items": [
            {
                "job_id": "job-1",
                "state": "queued",
                "history_count": 2,
                "history": [
                    {"event": "enqueued", "timestamp": "t1"},
                    {"event": "approved", "timestamp": "t2"},
                ],
            },
            {
                "job_id": "job-2",
                "state": "completed",
                "history_count": 1,
                "history": [{"event": "completed", "timestamp": "t3"}],
            },
        ],
    }

    selected = train_queue_history_payload(
        payload=payload,
        workspace="demo",
        job_id="job-1",
        limit=1,
        history_summary={"last_reason": "done"},
    )

    assert selected["workspace"] == "demo"
    assert selected["job_id"] == "job-1"
    assert selected["count"] == 2
    assert selected["history"] == [{"event": "approved", "timestamp": "t2"}]
    assert selected["available_job_ids"] == ["job-1", "job-2"]
    assert selected["history_summary"] == {"last_reason": "done"}
