import json
from pathlib import Path

from pfe_core.adapter_store.store import AdapterStore
from pfe_server.studio_eval_service import (
    load_eval_report,
    run_eval_job,
    start_eval_job,
)


class FakePipeline:
    def __init__(self, *, result: str = "EVAL COMPLETE", fail: bool = False):
        self.result = result
        self.fail = fail
        self.evaluate_calls = []

    def evaluate(self, **kwargs):
        self.evaluate_calls.append(kwargs)
        if self.fail:
            raise RuntimeError("eval failed")
        return self.result


def test_load_eval_report_reads_json_mapping(tmp_path: Path) -> None:
    adapter_path = tmp_path / "adapter"
    adapter_path.mkdir()
    (adapter_path / "eval_report.json").write_text(
        json.dumps({"recommendation": "deploy", "scores": {"style": 0.9}}),
        encoding="utf-8",
    )

    assert load_eval_report(adapter_path) == {"recommendation": "deploy", "scores": {"style": 0.9}}
    assert load_eval_report(tmp_path / "missing") == {}


def test_run_eval_job_records_completed_state_with_report(tmp_path: Path) -> None:
    adapter_path = tmp_path / "adapter"
    adapter_path.mkdir()
    (adapter_path / "eval_report.json").write_text(
        json.dumps({"recommendation": "deploy", "comparison": "improved"}),
        encoding="utf-8",
    )
    states = {}
    pipeline = FakePipeline()

    run_eval_job(
        pipeline=pipeline,
        workspace="client-a",
        version="20260615-001",
        requested_version="latest",
        job_id="eval-job",
        request_body={"num_samples": 3, "base_model": "base-a"},
        default_base_model=lambda: "default-base",
        load_adapter_path=lambda _version: adapter_path,
        persist_state=lambda workspace, state: states.update({workspace: dict(state)}),
    )

    assert pipeline.evaluate_calls == [
        {
            "base_model": "base-a",
            "adapter": "20260615-001",
            "num_samples": 3,
            "workspace": "client-a",
        }
    ]
    assert states["client-a"]["state"] == "completed"
    assert states["client-a"]["raw_result"] == "EVAL COMPLETE"
    assert states["client-a"]["recommendation"] == "deploy"


def test_run_eval_job_records_failed_state() -> None:
    states = {}

    run_eval_job(
        pipeline=FakePipeline(fail=True),
        workspace="client-a",
        version="20260615-001",
        requested_version="20260615-001",
        job_id="eval-job",
        request_body={},
        default_base_model=lambda: "default-base",
        load_adapter_path=lambda _version: "/unused",
        persist_state=lambda workspace, state: states.update({workspace: dict(state)}),
    )

    assert states["client-a"]["state"] == "failed"
    assert states["client-a"]["error"] == "eval failed"


def test_run_eval_job_persists_failed_studio_suite_report(tmp_path: Path, monkeypatch) -> None:
    pfe_home = tmp_path / ".pfe"
    monkeypatch.setenv("PFE_HOME", str(pfe_home))
    store = AdapterStore(home=pfe_home, workspace="client-a")
    created = store.create_training_version(
        base_model="base-a",
        training_config={"backend": "mock_local", "train_type": "sft"},
    )
    version = str(created["version"])
    store.mark_pending_eval(version, num_samples=1, metrics={"loss": 0.1})
    states = {}

    def fake_suite(**_kwargs):
        return {
            "passed": False,
            "pass_rate": 2 / 3,
            "summary_line": "studio_eval_suite=failed:refusal",
            "results": [{"type": "refusal", "passed": False}],
            "failed_cases": ["refusal"],
        }

    monkeypatch.setattr("pfe_server.studio_eval_service.run_studio_eval_suite", fake_suite)

    run_eval_job(
        pipeline=FakePipeline(result="EVAL COMPLETE"),
        workspace="client-a",
        version=version,
        requested_version=version,
        job_id="eval-job",
        request_body={"suite": ["memory", "ordinary_chat", "refusal"], "base_model": "base-a"},
        default_base_model=lambda: "default-base",
        load_adapter_path=store.load,
        persist_state=lambda workspace, state: states.update({workspace: dict(state)}),
    )

    rows = store.list_version_records(limit=1)
    persisted_report = json.loads(rows[0]["eval_report"])
    assert rows[0]["state"] == "failed_eval"
    assert persisted_report["recommendation"] == "keep_previous"
    assert persisted_report["studio_eval_suite"]["failed_cases"] == ["refusal"]
    assert states["client-a"]["state"] == "completed"
    assert states["client-a"]["recommendation"] == "keep_previous"


def test_start_eval_job_persists_running_runs_background_and_returns_payload(tmp_path: Path) -> None:
    adapter_path = tmp_path / "adapter"
    adapter_path.mkdir()
    states = []

    payload = start_eval_job(
        pipeline=FakePipeline(result="EVAL COMPLETE"),
        workspace="client-a",
        version="20260615-001",
        requested_version="latest",
        request_body={"num_samples": 2},
        default_base_model=lambda: "default-base",
        load_adapter_path=lambda _version: adapter_path,
        persist_state=lambda _workspace, state: states.append(dict(state)),
        build_adapters_payload=lambda: {"count": 1},
        job_id_factory=lambda: "eval-job",
        start_background=lambda target: target(),
    )

    assert payload["state"] == "running"
    assert payload["job_id"] == "eval-job"
    assert payload["adapters"] == {"count": 1}
    assert [state["state"] for state in states] == ["running", "completed"]
