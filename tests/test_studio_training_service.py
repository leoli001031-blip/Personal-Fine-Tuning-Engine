from pfe_server.studio_jobs import build_training_jobs_payload
from pfe_server.studio_training_service import (
    build_training_job_entry,
    extract_training_adapter_version,
    run_training_job,
    start_training_job,
    training_overall_state,
)


class FakePipeline:
    def __init__(self, *, result: str = "TRAINING COMPLETE 20260615-777", fail: bool = False):
        self.result = result
        self.fail = fail
        self.train_calls = 0
        self.train_dpo_calls = 0

    def train(self):
        self.train_calls += 1
        if self.fail:
            raise RuntimeError("trainer failed")
        return self.result

    def train_dpo(self):
        self.train_dpo_calls += 1
        if self.fail:
            raise RuntimeError("dpo failed")
        return self.result


def test_extract_training_adapter_version() -> None:
    assert extract_training_adapter_version("TRAINING COMPLETE 20260615-777") == "20260615-777"
    assert extract_training_adapter_version("no adapter here") is None


def test_run_training_job_records_started_completed_and_overall_state() -> None:
    job = build_training_job_entry(
        job_id="job-1",
        workspace="client-a",
        method="sft",
        training_config={"epochs": 1},
    )
    persisted_jobs = {}
    persisted_overall = {}
    pipeline = FakePipeline(result="TRAINING COMPLETE 20260615-777")

    run_training_job(
        job,
        pipeline=pipeline,
        method="sft",
        retry_of=None,
        persist_job=lambda job_id, entry: persisted_jobs.setdefault(job_id, []).append(dict(entry)),
        persist_overall=lambda workspace, state: persisted_overall.update({workspace: dict(state)}),
    )

    assert pipeline.train_calls == 1
    assert job["status"] == "completed"
    assert job["adapter_version"] == "20260615-777"
    assert [event["type"] for event in job["events"]] == ["started", "completed"]
    assert persisted_jobs["job-1"][-1]["status"] == "completed"
    assert persisted_overall["client-a"]["state"] == "completed"
    assert persisted_overall["client-a"]["adapter_version"] == "20260615-777"


def test_run_training_job_records_failure() -> None:
    job = build_training_job_entry(
        job_id="job-2",
        workspace="client-a",
        method="dpo",
        training_config={"epochs": 1},
    )
    persisted_overall = {}

    run_training_job(
        job,
        pipeline=FakePipeline(fail=True),
        method="dpo",
        retry_of="job-old",
        persist_job=lambda _job_id, _entry: None,
        persist_overall=lambda workspace, state: persisted_overall.update({workspace: dict(state)}),
    )

    assert job["status"] == "failed"
    assert job["error"] == "dpo failed"
    assert job["events"][-1]["type"] == "failed"
    assert job["events"][-1]["metadata"]["retry_of"] == "job-old"
    assert persisted_overall["client-a"]["state"] == "failed"


def test_start_training_job_persists_queued_runs_background_and_returns_payload() -> None:
    memory_jobs = {}
    overall = {}

    def persist_job(job_id, entry):
        memory_jobs[job_id] = dict(entry)

    def persist_overall(workspace, state):
        overall[workspace] = dict(state)

    def jobs_payload(limit):
        return build_training_jobs_payload(
            workspace="client-a",
            stored_jobs={},
            memory_jobs=memory_jobs,
            overall_state=overall.get("client-a"),
            limit=limit,
        )

    payload = start_training_job(
        workspace="client-a",
        pipeline=FakePipeline(result="TRAINING COMPLETE 20260615-888"),
        method="sft",
        training_config={"epochs": 2},
        preflight={"ready": True},
        retry_of="job-old",
        persist_job=persist_job,
        persist_overall=persist_overall,
        build_jobs_payload=jobs_payload,
        job_id_factory=lambda: "job-new",
        start_background=lambda target: target(),
    )

    assert payload["job_id"] == "job-new"
    assert payload["retry_of"] == "job-old"
    assert payload["action"] == "retry_started"
    assert payload["job"]["status"] == "completed"
    assert payload["job"]["adapter_version"] == "20260615-888"
    assert payload["jobs"]["latest"]["job_id"] == "job-new"
    assert payload["preflight"] == {"ready": True}


def test_training_overall_state_uses_job_status_and_updated_at() -> None:
    job = {
        "status": "cancelled",
        "adapter_version": None,
        "updated_at": "2026-06-15T00:00:00Z",
    }

    assert training_overall_state("job-3", job) == {
        "state": "cancelled",
        "adapter_version": None,
        "job_id": "job-3",
        "updated_at": "2026-06-15T00:00:00Z",
    }
