from pathlib import Path

from pfe_server.studio_job_store import StudioTrainingJobStore, load_json_state
from pfe_server.studio_jobs import append_training_job_event


def _job(job_id: str, *, workspace: str, status: str = "queued") -> dict:
    return {
        "job_id": job_id,
        "workspace": workspace,
        "status": status,
        "method": "sft",
        "adapter_version": None,
        "checkpoints": [],
        "events": [],
        "training_config": {"epochs": 1},
        "created_at": "2026-06-15T00:00:00Z",
        "updated_at": "2026-06-15T00:00:00Z",
    }


def _store(tmp_path: Path, *, workspace: str = "client-a") -> StudioTrainingJobStore:
    return StudioTrainingJobStore(
        workspace=workspace,
        workspace_dir=tmp_path / "workspaces" / workspace,
        memory_jobs={},
        overall_state={},
    )


def test_training_job_store_persists_jobs_and_builds_payload(tmp_path: Path) -> None:
    store = _store(tmp_path)
    job = _job("job-1", workspace="client-a", status="running")

    store.persist_job("job-1", job)
    store.persist_overall("client-a", {"state": "running", "job_id": "job-1"})

    payload = store.build_jobs_payload(limit=10)

    assert payload["latest"]["job_id"] == "job-1"
    assert payload["active"]["job_id"] == "job-1"
    assert payload["state"]["job_id"] == "job-1"
    assert load_json_state(store.jobs_path)["job-1"]["status"] == "running"


def test_training_job_store_cancel_running_keeps_memory_object_and_records_overall(tmp_path: Path) -> None:
    store = _store(tmp_path)
    job = _job("job-running", workspace="client-a", status="running")
    append_training_job_event(job, event_type="started", status="running", message="training job started")
    store.memory_jobs["job-running"] = job
    store.persist_job("job-running", job)

    result = store.cancel_job("job-running")

    assert result["outcome"] == "ok"
    assert result["action"] == "cancel_requested"
    assert job["cancellation_requested"] is True
    assert job["events"][-1]["type"] == "cancel_requested"
    assert store.current_overall_state()["cancellation_requested"] is True


def test_training_job_store_cancels_queued_job_and_blocks_terminal_job(tmp_path: Path) -> None:
    store = _store(tmp_path)
    queued = _job("job-queued", workspace="client-a", status="queued")
    completed = _job("job-done", workspace="client-a", status="completed")
    store.persist_job("job-queued", queued)
    store.persist_job("job-done", completed)

    cancelled = store.cancel_job("job-queued")
    blocked = store.cancel_job("job-done")

    assert cancelled["outcome"] == "ok"
    assert cancelled["action"] == "cancelled"
    assert cancelled["job"]["status"] == "cancelled"
    assert blocked["outcome"] == "not_cancellable"


def test_training_job_store_marks_retry_requested_for_retryable_job(tmp_path: Path) -> None:
    store = _store(tmp_path)
    failed = _job("job-failed", workspace="client-a", status="failed")
    running = _job("job-running", workspace="client-a", status="running")
    store.persist_job("job-failed", failed)
    store.persist_job("job-running", running)

    retryable = store.mark_retry_requested("job-failed")
    blocked = store.mark_retry_requested("job-running")

    assert retryable["outcome"] == "ok"
    assert failed["events"][-1]["type"] == "retry_requested"
    assert blocked["outcome"] == "not_retryable"
