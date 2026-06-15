from pfe_server.studio_eval_jobs import (
    EVAL_STATUS_URL,
    auto_eval_state_from_last_result,
    build_eval_completed_state,
    build_eval_failed_state,
    build_eval_running_state,
    build_eval_status_payload,
    running_eval_summary,
    running_eval_version,
)


def test_eval_running_state_and_summary_contract() -> None:
    state = build_eval_running_state(
        version="20260615-001",
        requested_version="latest",
        job_id="eval-job",
        now_seconds=1_800_000_000.0,
    )

    assert state == {
        "state": "running",
        "version": "20260615-001",
        "requested_version": "latest",
        "job_id": "eval-job",
        "status_url": EVAL_STATUS_URL,
        "updated_at": "2027-01-15T08:00:00Z",
    }
    assert running_eval_version(state) == "20260615-001"
    assert running_eval_version({"state": "completed", "version": "20260615-001"}) == ""
    assert running_eval_summary()["summary_line"] == "评估结论：评估中"


def test_eval_completed_state_merges_eval_report_without_losing_job_contract() -> None:
    state = build_eval_completed_state(
        version="20260615-001",
        requested_version="latest",
        raw_result="EVAL COMPLETE",
        job_id="eval-job",
        eval_report={
            "recommendation": "deploy",
            "comparison": "improved",
            "scores": {"style_match": 0.93},
        },
        now_seconds=1_800_000_000.0,
    )

    assert state["state"] == "completed"
    assert state["version"] == "20260615-001"
    assert state["requested_version"] == "latest"
    assert state["raw_result"] == "EVAL COMPLETE"
    assert state["job_id"] == "eval-job"
    assert state["status_url"] == EVAL_STATUS_URL
    assert state["recommendation"] == "deploy"
    assert state["scores"] == {"style_match": 0.93}


def test_eval_failed_state_and_status_payload_contract() -> None:
    state = build_eval_failed_state(
        version="20260615-001",
        requested_version="latest",
        error=RuntimeError("judge failed"),
        job_id="eval-job",
        now_seconds=1_800_000_000.0,
    )
    payload = build_eval_status_payload(state, adapters={"count": 1})

    assert payload["state"] == "failed"
    assert payload["error"] == "judge failed"
    assert payload["status_url"] == EVAL_STATUS_URL
    assert payload["adapters"] == {"count": 1}
    assert build_eval_status_payload(None)["state"] == "idle"
    assert build_eval_status_payload(None)["status_url"] == EVAL_STATUS_URL


def test_auto_eval_state_from_last_result() -> None:
    completed = auto_eval_state_from_last_result(
        {
            "eval_triggered": True,
            "triggered_version": "20260615-001",
            "eval_recommendation": "deploy",
            "eval_comparison": "improved",
        }
    )
    assert completed == {
        "state": "completed",
        "version": "20260615-001",
        "recommendation": "deploy",
        "comparison": "improved",
        "auto_evaluate": True,
    }

    failed = auto_eval_state_from_last_result(
        {
            "eval_triggered": True,
            "promoted_version": "20260615-000",
            "eval_error": "judge failed",
        }
    )
    assert failed["state"] == "failed"
    assert failed["version"] == "20260615-000"
    assert failed["error"] == "judge failed"
    assert auto_eval_state_from_last_result({"eval_triggered": False}) is None
