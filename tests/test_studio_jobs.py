from pfe_server.studio_jobs import (
    append_training_job_event,
    build_training_jobs_payload,
    latest_training_event_type,
    training_job_payload,
)


def _job(job_id: str, *, workspace: str, status: str, updated_at: str) -> dict:
    return {
        "job_id": job_id,
        "workspace": workspace,
        "status": status,
        "method": "sft",
        "adapter_version": None,
        "checkpoints": [],
        "events": [],
        "training_config": {"epochs": 1},
        "created_at": updated_at,
        "updated_at": updated_at,
    }


def test_training_job_payload_adds_urls_counts_latest_event_and_result_summary() -> None:
    job = _job("job-1", workspace="client-a", status="completed", updated_at="2026-06-15T00:00:00Z")
    job["checkpoints"] = [{"step": 1}, {"step": 2}]
    job["result"] = "x" * 300
    append_training_job_event(
        job,
        event_type="completed",
        status="completed",
        message="training job completed",
        now_seconds=1_800_000_000.0,
    )

    payload = training_job_payload(job)

    assert payload["status_url"] == "/pfe/training/jobs/job-1"
    assert payload["events_url"] == "/pfe/training/jobs/job-1/events"
    assert payload["cancel_url"] == "/pfe/training/jobs/job-1/cancel"
    assert payload["retry_url"] == "/pfe/training/jobs/job-1/retry"
    assert payload["checkpoint_count"] == 2
    assert payload["event_count"] == 1
    assert payload["latest_event"]["type"] == "completed"
    assert len(payload["result_summary"]) == 240


def test_append_training_job_event_recovers_non_list_events_and_updates_timestamp() -> None:
    job = _job("job-2", workspace="client-a", status="running", updated_at="old")
    job["events"] = "corrupt"

    event = append_training_job_event(
        job,
        event_type="started",
        status="running",
        message="training job started",
        metadata={"method": "sft"},
        now_seconds=1_800_000_000.0,
    )

    assert event["event_id"] == "job-2-started-1800000000000"
    assert event["created_at"] == "2027-01-15T08:00:00Z"
    assert event["metadata"] == {"method": "sft"}
    assert job["events"] == [event]
    assert job["updated_at"] == event["created_at"]
    assert latest_training_event_type(job) == "started"


def test_build_training_jobs_payload_merges_stored_and_memory_jobs_for_workspace() -> None:
    stored = {
        "stored-completed": _job(
            "stored-completed",
            workspace="client-a",
            status="completed",
            updated_at="2026-06-15T00:00:00Z",
        ),
        "memory-overrides": _job(
            "memory-overrides",
            workspace="client-a",
            status="queued",
            updated_at="2026-06-15T00:01:00Z",
        ),
    }
    memory = {
        "memory-overrides": _job(
            "memory-overrides",
            workspace="client-a",
            status="running",
            updated_at="2026-06-15T00:02:00Z",
        ),
        "other-workspace": _job(
            "other-workspace",
            workspace="client-b",
            status="running",
            updated_at="2026-06-15T00:03:00Z",
        ),
    }

    payload = build_training_jobs_payload(
        workspace="client-a",
        stored_jobs=stored,
        memory_jobs=memory,
        overall_state={"state": "running", "job_id": "memory-overrides"},
        limit=20,
    )

    assert payload["workspace"] == "client-a"
    assert payload["count"] == 2
    assert [item["job_id"] for item in payload["items"]] == ["memory-overrides", "stored-completed"]
    assert payload["latest"]["job_id"] == "memory-overrides"
    assert payload["active"]["job_id"] == "memory-overrides"
    assert payload["state"]["job_id"] == "memory-overrides"
    assert payload["create_api"] == "POST /pfe/training/jobs"


def test_build_training_jobs_payload_uses_idle_state_when_no_state_exists() -> None:
    payload = build_training_jobs_payload(
        workspace="client-a",
        stored_jobs={},
        memory_jobs={},
        overall_state=None,
    )

    assert payload["items"] == []
    assert payload["latest"] is None
    assert payload["active"] is None
    assert payload["state"] == {"state": "idle", "adapter_version": None}
