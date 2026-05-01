from __future__ import annotations



from pfe_core.pipeline_operations import (  # noqa: E402
    classify_operations_event,
    generic_monitor_active,
    operations_event_severity_rank,
    ordered_unique_actions,
    prefer_inspection_summary_for_generic_monitor,
)

def test_classify_operations_event_promotes_daemon_stale_to_critical() -> None:
    payload = classify_operations_event(
        source="daemon",
        event="alert",
        reason="daemon_stale",
        status="expired",
    )

    assert payload == {"severity": "critical", "attention": True}
    assert operations_event_severity_rank(payload["severity"]) == 4

def test_classify_operations_event_tracks_queue_review_attention() -> None:
    payload = classify_operations_event(
        source="queue",
        event="queue_pending_review",
        reason="manual_review_required",
        state="awaiting_confirmation",
    )

    assert payload["severity"] == "info"
    assert payload["attention"] is True

def test_prefer_inspection_summary_for_generic_monitor_focuses() -> None:
    summary, inspection = prefer_inspection_summary_for_generic_monitor(
        focus="runner_active",
        summary_line="runner=active",
        inspection_summary_line="current_focus=runner_active | required_action=inspect_runtime_stability",
    )

    assert generic_monitor_active(
        focus="runner_active",
        inspection_summary_line=inspection,
    )
    assert summary == inspection

def test_ordered_unique_actions_filters_empty_none_and_duplicates() -> None:
    assert ordered_unique_actions(
        ["recover_worker_daemon", "inspect_daemon_heartbeat"],
        ["recover_worker_daemon", None, "none", ""],
        ["inspect_daemon_restart_policy"],
    ) == [
        "recover_worker_daemon",
        "inspect_daemon_heartbeat",
        "inspect_daemon_restart_policy",
    ]
