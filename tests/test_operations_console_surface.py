from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for package_dir in ("pfe-core", "pfe-cli", "pfe-server"):
    package_path = str(ROOT / package_dir)
    if package_path not in os.sys.path:
        os.sys.path.insert(0, package_path)

from pfe_core.adapter_store.store import AdapterStore
from pfe_core.pipeline import PipelineService
from pfe_core.server_services import InferenceServiceAdapter, PipelineServiceAdapter
from pfe_server.app import ServiceBundle, create_app, smoke_test_request
from pfe_server.auth import ServerSecurityConfig


class OperationsConsoleSurfaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.previous_home = os.environ.get("PFE_HOME")
        self.pfe_home = Path(self.tempdir.name) / ".pfe"
        os.environ["PFE_HOME"] = str(self.pfe_home)

    def tearDown(self) -> None:
        if self.previous_home is None:
            os.environ.pop("PFE_HOME", None)
        else:
            os.environ["PFE_HOME"] = self.previous_home
        self.tempdir.cleanup()

    def _service(self) -> PipelineService:
        return PipelineService()

    def _app(self, service: PipelineService):
        return create_app(
            ServiceBundle(
                inference=InferenceServiceAdapter(service),
                pipeline=PipelineServiceAdapter(service),
                security=ServerSecurityConfig(),
                provider="core",
                workspace=str(self.pfe_home),
            )
        )

    def test_status_exposes_operations_console_sections(self) -> None:
        service = self._service()
        service.generate(scenario="life-coach", style="warm", num_samples=8)
        first = service.train_result(method="qlora", epochs=1, train_type="sft")
        AdapterStore(home=self.pfe_home).promote(first.version)
        service.generate(scenario="work-coach", style="direct", num_samples=8)
        second = service.train_result(method="qlora", epochs=1, train_type="sft")
        service.promote_candidate(note="ready_for_rollout")
        service._append_train_queue_item(
            {
                "job_id": "job-console-1",
                "state": "awaiting_confirmation",
                "workspace": "user_default",
                "source": "signal_auto_train",
                "adapter_version": second.version,
                "confirmation_required": True,
                "confirmation_reason": "manual_review_required_by_policy",
            },
            workspace="user_default",
        )

        status = service.status()
        console = status["operations_console"]
        event_stream = console["event_stream"]

        self.assertTrue(console["attention_needed"])
        self.assertIn("review_queue_confirmation", console["next_actions"])
        self.assertIn("auto_train_policy", console)
        self.assertEqual(console["trigger_blocked_reason"], "queue_pending_review")
        self.assertEqual(console["trigger_blocked_action"], "review_queue_confirmation")
        self.assertEqual(console["trigger_blocked_category"], "queue")
        self.assertEqual(console["auto_train_policy"]["queue_entry_mode"], "disabled")
        self.assertEqual(console["auto_train_policy"]["stop_stage"], "trigger")
        self.assertIn("trigger_threshold_summary", console)
        self.assertIn("trigger_threshold_summary", status["operations_overview"])
        self.assertIn("samples=", status["operations_overview"]["trigger_threshold_summary_line"])
        self.assertIn("queue_review_policy", console)
        self.assertEqual(console["queue_review_policy"]["review_mode"], "manual_review")
        self.assertEqual(console["queue_review_policy"]["next_action"], "review_queue_confirmation")
        self.assertIn("trigger_policy_summary", status["operations_overview"])
        self.assertIn("trigger_blocked_reason", status["operations_overview"])
        self.assertIn("trigger_blocked_action", status["operations_overview"])
        self.assertEqual(status["operations_overview"]["queue_gate_reason"], "queue_pending_review")
        self.assertEqual(status["operations_overview"]["queue_gate_action"], "review_queue_confirmation")
        self.assertEqual(status["operations_overview"]["trigger_blocked_reason"], "queue_pending_review")
        self.assertEqual(status["operations_overview"]["trigger_blocked_action"], "review_queue_confirmation")
        self.assertEqual(status["operations_overview"]["auto_train_blocker"]["reason"], "queue_pending_review")
        self.assertEqual(status["operations_overview"]["auto_train_blocker"]["action"], "review_queue_confirmation")
        self.assertEqual(console["auto_train_blocker"]["reason"], "queue_pending_review")
        self.assertEqual(console["dashboard"]["auto_train_blocker"]["reason"], "queue_pending_review")
        self.assertEqual(console["event_stream"]["auto_train_blocker"]["reason"], "queue_pending_review")
        self.assertEqual(console["alert_policy"]["auto_train_blocker"]["reason"], "queue_pending_review")
        self.assertIn("queue_review_policy_summary", status["operations_overview"])
        self.assertEqual(console["candidate"]["current_stage"], "promoted")
        self.assertIn("candidate_promoted", console["candidate"]["last_reason"])
        self.assertEqual(console["queue"]["awaiting_confirmation_count"], 1)
        self.assertEqual(console["queue"]["review_policy_summary"]["review_mode"], "manual_review")
        self.assertEqual(console["runner"]["lock_state"], "idle")
        self.assertIn("runtime_stability_summary", console)
        self.assertIn("runner=idle", console["runtime_stability_summary"]["summary_line"])
        self.assertIn("alerts", console)
        self.assertIn("health", console)
        self.assertIn("recovery", console)
        self.assertIn("dashboard", console)
        self.assertEqual(console["dashboard"]["highest_priority_action"], "review_queue_confirmation")
        self.assertIn("action=review_queue_confirmation", console["dashboard"]["dashboard_digest"])
        self.assertIn("current_focus=candidate_idle", console["dashboard"]["dashboard_digest"])
        self.assertIn("required_action=review_queue_confirmation", console["dashboard"]["dashboard_digest"])
        self.assertIn(
            "next_actions=review_queue_confirmation,inspect_auto_train_gate,inspect_auto_train_trigger",
            console["dashboard"]["dashboard_digest"],
        )
        self.assertIn("required_action=review_queue_confirmation", console["summary_line"])
        self.assertIn("current_focus=candidate_idle", console["summary_line"])
        self.assertEqual(console["health"]["status"], "attention")
        self.assertTrue(any(alert["reason"] == "awaiting_confirmation" for alert in console["alerts"]))
        self.assertTrue(any(alert["reason"] == "queue_pending_review" for alert in status["operations_overview"]["alerts"]))
        self.assertIn("queue-gate=queue_pending_review", status["operations_overview"]["summary_line"])
        self.assertEqual(event_stream["severity"], "info")
        self.assertTrue(event_stream["attention_needed"])
        self.assertGreaterEqual(event_stream["attention_count"], 1)
        self.assertEqual(event_stream["status"], "attention")
        self.assertEqual(event_stream["highest_priority_action"], "review_queue_confirmation")
        self.assertEqual(event_stream["policy"]["secondary_action"], "inspect_auto_train_gate")
        self.assertEqual(
            event_stream["policy"]["secondary_actions"][:2],
            ["inspect_auto_train_gate", "inspect_auto_train_trigger"],
        )
        self.assertEqual(event_stream["active_recovery_hint"], "manual_review_required_by_policy")
        self.assertEqual(event_stream["queue_review_mode"], "manual_review")
        self.assertEqual(event_stream["queue_review_next_action"], "review_queue_confirmation")
        self.assertEqual(event_stream["dashboard"]["queue_review_mode"], "manual_review")
        self.assertEqual(event_stream["dashboard"]["queue_review_next_action"], "review_queue_confirmation")
        self.assertIn("awaiting_confirmation", event_stream["escalated_reasons"])
        self.assertIsNone(event_stream["latest_recovery"])
        self.assertEqual(event_stream["dashboard"]["highest_priority_action"], "review_queue_confirmation")
        self.assertEqual(event_stream["dashboard"]["active_recovery_hint"], "manual_review_required_by_policy")
        self.assertIn("action=review_queue_confirmation", event_stream["dashboard"]["dashboard_digest"])
        self.assertIn("current_focus=candidate_idle", event_stream["dashboard"]["dashboard_digest"])
        self.assertIn("required_action=review_queue_confirmation", event_stream["dashboard"]["dashboard_digest"])
        self.assertIn(
            "next_actions=review_queue_confirmation,inspect_auto_train_gate,inspect_auto_train_trigger",
            event_stream["dashboard"]["dashboard_digest"],
        )
        self.assertTrue(any(item.get("severity") == "info" and item.get("attention") for item in event_stream["items"]))

    def test_operations_console_derives_queue_review_policy_from_overview_when_queue_snapshot_missing(self) -> None:
        console = PipelineService._operations_console(
            operations_overview={
                "auto_train_policy": {
                    "queue_entry_mode": "awaiting_confirmation",
                    "evaluation_mode": "auto_evaluate",
                    "promotion_mode": "auto_promote",
                    "stop_stage": "confirmation",
                },
                "queue_gate_reason": "queue_pending_review",
                "queue_gate_action": "review_queue_confirmation",
                "queue_gate_review_mode": "manual_review",
                "trigger_blocked_reason": "queue_pending_review",
                "trigger_blocked_action": "review_queue_confirmation",
            },
            candidate_history=None,
            candidate_timeline=None,
            train_queue={},
        )

        self.assertEqual(console["queue_review_policy"]["review_mode"], "manual_review")
        self.assertEqual(console["queue_review_policy"]["queue_entry_mode"], "awaiting_confirmation")
        self.assertEqual(console["queue_review_policy"]["next_action"], "review_queue_confirmation")
        self.assertTrue(console["queue_review_policy"]["review_required_now"])
        self.assertIn("mode=manual_review", console["queue_review_policy_summary"])
        self.assertEqual(console["dashboard"]["queue_review_mode"], "manual_review")
        self.assertEqual(console["event_stream"]["queue_review_next_action"], "review_queue_confirmation")

    def test_status_surfaces_daemon_recovery_alerts_and_health(self) -> None:
        service = self._service()
        stale_time = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
        service._persist_train_queue_daemon_state(
            {
                "workspace": "user_default",
                "desired_state": "running",
                "observed_state": "running",
                "command_status": "running",
                "active": True,
                "pid": 999999,
                "started_at": stale_time,
                "last_heartbeat_at": stale_time,
                "auto_restart_enabled": False,
                "auto_recover_enabled": False,
                "restart_attempts": 0,
                "max_restart_attempts": 3,
                "restart_backoff_seconds": 15.0,
            },
            workspace="user_default",
        )

        status = service.status()
        overview = status["operations_overview"]
        console = status["operations_console"]
        event_stream = console["event_stream"]

        self.assertTrue(overview["attention_needed"])
        self.assertEqual(overview["attention_reason"], "daemon_stale")
        self.assertEqual(overview["health"]["daemon_lock_state"], "stale")
        self.assertEqual(overview["health"]["daemon_health_state"], "stale")
        self.assertEqual(overview["health"]["daemon_lease_state"], "expired")
        self.assertEqual(overview["health"]["daemon_heartbeat_state"], "stale")
        self.assertTrue(overview["recovery"]["daemon_recovery_needed"])
        self.assertEqual(overview["recovery"]["daemon_recovery_action"], "manual_recover")
        self.assertTrue(any(alert["reason"] == "daemon_stale" for alert in overview["alerts"]))
        self.assertTrue(any(alert["reason"] == "daemon_heartbeat_stale" for alert in overview["alerts"]))
        self.assertTrue(any(alert["reason"] == "daemon_lease_expired" for alert in overview["alerts"]))
        self.assertEqual(console["health"]["daemon_lock_state"], "stale")
        self.assertEqual(console["recovery"]["daemon_recovery_reason"], "daemon_stale")
        self.assertIn("recover_worker_daemon", console["next_actions"])
        self.assertIn("inspect_daemon_heartbeat", console["next_actions"])
        self.assertEqual(console["daemon"]["health_state"], "stale")
        self.assertEqual(console["daemon"]["heartbeat_state"], "stale")
        self.assertEqual(console["daemon"]["lease_state"], "expired")
        self.assertEqual(console["daemon"]["recovery_action"], "manual_recover")
        self.assertEqual(console["runtime_stability_summary"]["daemon_health_state"], "stale")
        self.assertEqual(console["runtime_stability_summary"]["daemon_heartbeat_state"], "stale")
        self.assertEqual(console["runtime_stability_summary"]["daemon_lease_state"], "expired")
        self.assertEqual(console["runtime_stability_summary"]["daemon_recovery_action"], "manual_recover")
        self.assertIn("recover=manual_recover", console["runtime_stability_summary"]["summary_line"])
        self.assertIn("health=stale", console["daemon"]["status_line"])
        self.assertIn("daemon-health=stale", console["summary_line"])
        self.assertEqual(console["dashboard"]["highest_priority_action"], "recover_worker_daemon")
        self.assertEqual(console["dashboard"]["active_recovery_hint"], "manual_recover")
        self.assertIn("runtime_stability_summary", console["dashboard"])
        self.assertIn("daemon=stale", console["dashboard"]["runtime_stability_summary_line"])
        self.assertIn("handling=auto", console["dashboard"]["dashboard_digest"])
        self.assertIn("daemon_stale", console["dashboard"]["escalated_reasons"])
        self.assertEqual(event_stream["severity"], "critical")
        self.assertTrue(event_stream["attention_needed"])
        self.assertEqual(event_stream["attention_reason"], "daemon_stale")
        self.assertEqual(event_stream["highest_priority_action"], "recover_worker_daemon")
        self.assertEqual(event_stream["active_recovery_hint"], "manual_recover")
        self.assertIn("runtime_stability_summary", event_stream)
        self.assertIn("daemon=stale", event_stream["runtime_stability_summary_line"])
        self.assertIn("daemon_stale", event_stream["escalated_reasons"])
        self.assertIsNone(event_stream["latest_recovery"])
        self.assertTrue(any(item.get("source") == "daemon" and item.get("severity") == "critical" for item in event_stream["items"]))
        daemon_timeline = console["daemon_timeline"]
        self.assertEqual(daemon_timeline["count"], 0)
        self.assertEqual(daemon_timeline["recovery_event_count"], 0)
        self.assertEqual(
            daemon_timeline["summary_line"],
            "count=0 | recovery_event_count=0",
        )

    def test_status_exposes_daemon_recovery_timeline_summary(self) -> None:
        service = self._service()
        service._append_train_queue_daemon_history(
            workspace="user_default",
            event="start_requested",
            reason="daemon_start_requested",
            metadata={"note": "boot"},
        )
        service._append_train_queue_daemon_history(
            workspace="user_default",
            event="recover_requested",
            reason="daemon_stale",
            metadata={"note": "auto_recovery"},
        )
        service._append_train_queue_daemon_history(
            workspace="user_default",
            event="completed",
            reason="stop_requested",
            metadata={},
        )

        status = service.status()
        daemon_timeline = status["operations_console"]["daemon_timeline"]

        self.assertEqual(daemon_timeline["count"], 3)
        self.assertEqual(daemon_timeline["last_event"], "completed")
        self.assertEqual(daemon_timeline["recovery_event_count"], 2)
        self.assertEqual(daemon_timeline["last_recovery_event"], "recover_requested")
        self.assertEqual(daemon_timeline["last_recovery_reason"], "daemon_stale")
        self.assertEqual(daemon_timeline["last_recovery_note"], "auto_recovery")
        self.assertGreaterEqual(len(daemon_timeline["recent_events"]), 3)
        self.assertGreaterEqual(len(daemon_timeline["recent_recovery_events"]), 2)
        self.assertIn("last_recovery_event=recover_requested", daemon_timeline["summary_line"])

    def test_status_uses_runner_active_as_dashboard_focus_when_monitoring_runner(self) -> None:
        service = self._service()
        service._persist_train_queue_worker_state(
            {
                "active": True,
                "stop_requested": False,
                "pid": 12345,
                "started_at": "2026-03-20T10:00:00+00:00",
                "last_heartbeat_at": datetime.now(timezone.utc).isoformat(),
                "last_completed_at": None,
                "loop_cycles": 1,
                "processed_count": 0,
                "failed_count": 0,
                "stopped_reason": None,
                "last_action": "run_worker_runner",
                "max_seconds": 30.0,
                "idle_sleep_seconds": 1.0,
            },
            workspace="user_default",
        )

        status = service.status()

        self.assertEqual(status["operations_overview"]["monitor_focus"], "runner_active")
        self.assertEqual(status["operations_overview"]["current_focus"], "runner_active")
        self.assertEqual(status["operations_overview"]["required_action"], "inspect_runtime_stability")
        self.assertEqual(
            status["operations_overview"]["summary_line"],
            "current_focus=runner_active | required_action=inspect_runtime_stability | next_actions=inspect_runtime_stability,inspect_worker_runner_history",
        )
        self.assertEqual(
            status["operations_overview"]["inspection_summary_line"],
            "current_focus=runner_active | required_action=inspect_runtime_stability | next_actions=inspect_runtime_stability,inspect_worker_runner_history",
        )
        self.assertEqual(status["operations_console"]["monitor_focus"], "runner_active")
        self.assertIn("current_focus=runner_active", status["operations_console"]["summary_line"])
        self.assertIn("required_action=inspect_runtime_stability", status["operations_console"]["summary_line"])
        self.assertIn(
            "next_actions=inspect_runtime_stability,inspect_worker_runner_history",
            status["operations_console"]["summary_line"],
        )
        self.assertEqual(status["operations_console"]["dashboard"]["current_focus"], "runner_active")
        self.assertEqual(status["operations_event_stream"]["dashboard"]["current_focus"], "runner_active")
        self.assertEqual(status["operations_dashboard"]["required_action"], "inspect_runtime_stability")
        self.assertEqual(status["operations_dashboard"]["secondary_action"], "inspect_worker_runner_history")
        self.assertEqual(
            status["operations_event_stream"]["dashboard"]["required_action"],
            "inspect_runtime_stability",
        )
        self.assertEqual(
            status["operations_event_stream"]["dashboard"]["secondary_action"],
            "inspect_worker_runner_history",
        )
        self.assertEqual(status["operations_alert_policy"]["required_action"], "inspect_runtime_stability")
        self.assertEqual(status["operations_alert_policy"]["secondary_action"], "inspect_worker_runner_history")
        self.assertEqual(
            status["operations_overview"]["runtime_action_summary"]["primary_action"],
            "inspect_runtime_stability",
        )
        self.assertEqual(
            status["operations_overview"]["runtime_action_summary"]["secondary_actions"],
            ["inspect_worker_runner_history"],
        )
        self.assertEqual(
            status["operations_dashboard"]["runtime_action_summary"]["primary_action"],
            "inspect_runtime_stability",
        )
        self.assertEqual(
            status["operations_console"]["next_actions"][:2],
            ["inspect_runtime_stability", "inspect_worker_runner_history"],
        )

    def test_status_uses_daemon_active_as_dashboard_focus_when_monitoring_daemon(self) -> None:
        service = self._service()
        service._persist_train_queue_daemon_state(
            {
                "workspace": "user_default",
                "desired_state": "running",
                "observed_state": "running",
                "command_status": "running",
                "active": True,
                "pid": os.getpid(),
                "started_at": datetime.now(timezone.utc).isoformat(),
                "last_heartbeat_at": datetime.now(timezone.utc).isoformat(),
                "lock_state": "active",
                "health_state": "healthy",
                "heartbeat_state": "fresh",
                "lease_state": "valid",
                "auto_restart_enabled": True,
                "auto_recover_enabled": True,
                "restart_attempts": 0,
                "max_restart_attempts": 3,
                "restart_backoff_seconds": 15.0,
                "heartbeat_interval_seconds": 30.0,
                "lease_timeout_seconds": 300.0,
            },
            workspace="user_default",
        )

        status = service.status()

        self.assertEqual(status["operations_overview"]["monitor_focus"], "daemon_active")
        self.assertEqual(status["operations_overview"]["current_focus"], "daemon_active")
        self.assertEqual(status["operations_overview"]["required_action"], "inspect_runtime_stability")
        self.assertEqual(
            status["operations_overview"]["summary_line"],
            "current_focus=daemon_active | required_action=inspect_runtime_stability | next_actions=inspect_runtime_stability,inspect_daemon_status",
        )
        self.assertEqual(
            status["operations_overview"]["inspection_summary_line"],
            "current_focus=daemon_active | required_action=inspect_runtime_stability | next_actions=inspect_runtime_stability,inspect_daemon_status",
        )
        self.assertEqual(status["operations_console"]["monitor_focus"], "daemon_active")
        self.assertIn("current_focus=daemon_active", status["operations_console"]["summary_line"])
        self.assertIn("required_action=inspect_runtime_stability", status["operations_console"]["summary_line"])
        self.assertIn("next_actions=inspect_runtime_stability,inspect_daemon_status", status["operations_console"]["summary_line"])
        self.assertEqual(status["operations_console"]["dashboard"]["current_focus"], "daemon_active")
        self.assertEqual(status["operations_event_stream"]["dashboard"]["current_focus"], "daemon_active")
        self.assertEqual(status["operations_dashboard"]["required_action"], "inspect_runtime_stability")
        self.assertIn("required_action=inspect_runtime_stability", status["operations_dashboard"]["summary_line"])
        self.assertEqual(status["operations_dashboard"]["secondary_action"], "inspect_daemon_status")
        self.assertEqual(
            status["operations_event_stream"]["dashboard"]["required_action"],
            "inspect_runtime_stability",
        )
        self.assertEqual(
            status["operations_event_stream"]["dashboard"]["secondary_action"],
            "inspect_daemon_status",
        )
        self.assertEqual(status["operations_alert_policy"]["required_action"], "inspect_runtime_stability")
        self.assertEqual(status["operations_alert_policy"]["secondary_action"], "inspect_daemon_status")
        self.assertEqual(
            status["operations_overview"]["runtime_action_summary"]["primary_action"],
            "inspect_runtime_stability",
        )
        self.assertEqual(
            status["operations_overview"]["runtime_action_summary"]["secondary_actions"],
            ["inspect_daemon_status"],
        )
        self.assertEqual(
            status["operations_dashboard"]["runtime_action_summary"]["primary_action"],
            "inspect_runtime_stability",
        )
        self.assertEqual(
            status["operations_console"]["next_actions"][:2],
            ["inspect_runtime_stability", "inspect_daemon_status"],
        )

    def test_status_uses_candidate_idle_actions_for_failed_eval_candidate(self) -> None:
        service = self._service()
        service.generate(scenario="life-coach", style="warm", num_samples=6)
        first = service.train_result(method="qlora", epochs=1, train_type="sft")
        store = AdapterStore(home=self.pfe_home)
        store.promote(first.version)
        service.generate(scenario="work-coach", style="direct", num_samples=6)
        second = service.train_result(method="qlora", epochs=1, train_type="sft")
        store.mark_failed_eval(second.version)

        status = service.status()

        self.assertEqual(status["operations_overview"]["monitor_focus"], "candidate_idle")
        self.assertEqual(status["operations_overview"]["current_focus"], "candidate_idle")
        self.assertEqual(status["operations_overview"]["required_action"], "archive_candidate")
        self.assertEqual(
            status["operations_overview"]["summary_line"],
            "current_focus=candidate_idle | required_action=archive_candidate | next_actions=archive_candidate,promote_candidate",
        )
        self.assertEqual(
            status["operations_overview"]["inspection_summary_line"],
            "current_focus=candidate_idle | required_action=archive_candidate | next_actions=archive_candidate,promote_candidate",
        )
        self.assertEqual(status["operations_console"]["monitor_focus"], "candidate_idle")
        self.assertIn("current_focus=candidate_idle", status["operations_console"]["summary_line"])
        self.assertIn("required_action=archive_candidate", status["operations_console"]["summary_line"])
        self.assertIn("next_actions=archive_candidate,promote_candidate", status["operations_console"]["summary_line"])
        self.assertEqual(status["operations_console"]["dashboard"]["current_focus"], "candidate_idle")
        self.assertEqual(status["operations_event_stream"]["dashboard"]["current_focus"], "candidate_idle")
        self.assertEqual(status["operations_dashboard"]["required_action"], "archive_candidate")
        self.assertEqual(status["operations_dashboard"]["secondary_action"], "promote_candidate")
        self.assertEqual(status["operations_event_stream"]["dashboard"]["required_action"], "archive_candidate")
        self.assertEqual(status["operations_event_stream"]["dashboard"]["secondary_action"], "promote_candidate")
        self.assertEqual(status["operations_alert_policy"]["required_action"], "archive_candidate")
        self.assertEqual(status["operations_alert_policy"]["secondary_action"], "promote_candidate")
        self.assertEqual(
            status["operations_console"]["next_actions"][:2],
            ["archive_candidate", "promote_candidate"],
        )

    def test_status_uses_candidate_idle_inspect_actions_for_archived_candidate(self) -> None:
        service = self._service()
        service.generate(scenario="life-coach", style="warm", num_samples=6)
        first = service.train_result(method="qlora", epochs=1, train_type="sft")
        store = AdapterStore(home=self.pfe_home)
        store.promote(first.version)
        service.generate(scenario="work-coach", style="direct", num_samples=6)
        second = service.train_result(method="qlora", epochs=1, train_type="sft")
        store.archive(second.version)

        status = service.status()

        self.assertEqual(status["operations_overview"]["monitor_focus"], "candidate_idle")
        self.assertEqual(status["operations_overview"]["current_focus"], "candidate_idle")
        self.assertEqual(status["operations_overview"]["required_action"], "inspect_candidate_status")
        self.assertEqual(
            status["operations_overview"]["summary_line"],
            "current_focus=candidate_idle | required_action=inspect_candidate_status | next_actions=inspect_candidate_status,inspect_candidate_timeline",
        )
        self.assertEqual(
            status["operations_overview"]["inspection_summary_line"],
            "current_focus=candidate_idle | required_action=inspect_candidate_status | next_actions=inspect_candidate_status,inspect_candidate_timeline",
        )
        self.assertEqual(status["operations_console"]["monitor_focus"], "candidate_idle")
        self.assertIn("current_focus=candidate_idle", status["operations_console"]["summary_line"])
        self.assertIn("required_action=inspect_candidate_status", status["operations_console"]["summary_line"])
        self.assertIn(
            "next_actions=inspect_candidate_status,inspect_candidate_timeline",
            status["operations_console"]["summary_line"],
        )
        self.assertEqual(status["operations_console"]["dashboard"]["current_focus"], "candidate_idle")
        self.assertEqual(status["operations_event_stream"]["dashboard"]["current_focus"], "candidate_idle")
        self.assertEqual(status["operations_dashboard"]["required_action"], "inspect_candidate_status")
        self.assertEqual(status["operations_dashboard"]["secondary_action"], "inspect_candidate_timeline")
        self.assertEqual(
            status["operations_event_stream"]["dashboard"]["required_action"],
            "inspect_candidate_status",
        )
        self.assertEqual(
            status["operations_event_stream"]["dashboard"]["secondary_action"],
            "inspect_candidate_timeline",
        )
        self.assertEqual(status["operations_alert_policy"]["required_action"], "inspect_candidate_status")
        self.assertEqual(status["operations_alert_policy"]["secondary_action"], "inspect_candidate_timeline")
        self.assertEqual(
            status["operations_overview"]["candidate_action_summary"]["primary_action"],
            "inspect_candidate_status",
        )
        self.assertEqual(
            status["operations_overview"]["candidate_action_summary"]["secondary_actions"],
            ["inspect_candidate_timeline"],
        )
        self.assertEqual(
            status["operations_dashboard"]["candidate_action_summary"]["primary_action"],
            "inspect_candidate_status",
        )
        self.assertEqual(
            status["operations_console"]["next_actions"][:2],
            ["inspect_candidate_status", "inspect_candidate_timeline"],
        )

    def test_status_uses_queue_waiting_execution_actions_when_queue_gate_is_active(self) -> None:
        service = self._service()
        service._append_train_queue_item(
            {
                "job_id": "job-queue-backlog-2",
                "state": "queued",
                "workspace": "user_default",
                "source": "manual_queue",
                "priority": 50,
            },
            workspace="user_default",
        )

        status = service.status()

        self.assertEqual(status["operations_overview"]["monitor_focus"], "queue_waiting_execution")
        self.assertEqual(status["operations_overview"]["current_focus"], "queue_waiting_execution")
        self.assertEqual(status["operations_overview"]["required_action"], "process_next_queue_item")
        self.assertEqual(
            status["operations_overview"]["summary_line"],
            "current_focus=queue_waiting_execution | required_action=process_next_queue_item | next_actions=process_next_queue_item,inspect_auto_train_trigger",
        )
        self.assertEqual(status["operations_console"]["monitor_focus"], "queue_waiting_execution")
        self.assertEqual(status["operations_console"]["dashboard"]["current_focus"], "queue_waiting_execution")
        self.assertEqual(status["operations_alert_policy"]["required_action"], "process_next_queue_item")
        self.assertEqual(status["operations_alert_policy"]["secondary_action"], "inspect_auto_train_trigger")
        self.assertEqual(
            status["operations_overview"]["queue_action_summary"]["primary_action"],
            "process_next_queue_item",
        )
        self.assertEqual(
            status["operations_overview"]["queue_action_summary"]["secondary_actions"],
            ["inspect_auto_train_trigger"],
        )
        self.assertEqual(
            status["operations_dashboard"]["queue_action_summary"]["primary_action"],
            "process_next_queue_item",
        )
        self.assertEqual(
            status["operations_console"]["next_actions"][:2],
            ["process_next_queue_item", "inspect_auto_train_trigger"],
        )

    def test_status_uses_queue_waiting_execution_actions_for_monitor_queued_work(self) -> None:
        service = self._service()
        service._append_train_queue_item(
            {
                "job_id": "job-queue-backlog-monitor",
                "state": "queued",
                "workspace": "user_default",
                "source": "manual_queue",
                "priority": 25,
            },
            workspace="user_default",
        )

        status = service.status()

        self.assertEqual(status["operations_overview"]["monitor_focus"], "queue_waiting_execution")
        self.assertEqual(status["operations_overview"]["current_focus"], "queue_waiting_execution")
        self.assertEqual(status["operations_overview"]["required_action"], "process_next_queue_item")
        self.assertEqual(
            status["operations_overview"]["summary_line"],
            "current_focus=queue_waiting_execution | required_action=process_next_queue_item | next_actions=process_next_queue_item,inspect_auto_train_trigger",
        )
        self.assertEqual(
            status["operations_overview"]["inspection_summary_line"],
            "current_focus=queue_waiting_execution | required_action=process_next_queue_item | next_actions=process_next_queue_item,inspect_auto_train_trigger",
        )
        self.assertEqual(status["operations_console"]["monitor_focus"], "queue_waiting_execution")
        self.assertEqual(status["operations_console"]["current_focus"], "queue_waiting_execution")
        self.assertEqual(status["operations_console"]["required_action"], "process_next_queue_item")
        self.assertEqual(
            status["operations_console"]["summary_line"],
            "current_focus=queue_waiting_execution | required_action=process_next_queue_item | next_actions=process_next_queue_item,inspect_auto_train_trigger",
        )
        self.assertEqual(
            status["operations_console"]["inspection_summary_line"],
            "current_focus=queue_waiting_execution | required_action=process_next_queue_item | next_actions=process_next_queue_item,inspect_auto_train_trigger",
        )
        self.assertEqual(status["operations_console"]["dashboard"]["current_focus"], "queue_waiting_execution")
        self.assertEqual(status["operations_event_stream"]["dashboard"]["current_focus"], "queue_waiting_execution")
        self.assertEqual(status["operations_dashboard"]["required_action"], "process_next_queue_item")
        self.assertEqual(
            status["operations_dashboard"]["summary_line"],
            "current_focus=queue_waiting_execution | required_action=process_next_queue_item | next_actions=process_next_queue_item,inspect_auto_train_trigger",
        )
        self.assertEqual(status["operations_dashboard"]["secondary_action"], "inspect_auto_train_trigger")
        self.assertEqual(
            status["operations_event_stream"]["dashboard"]["required_action"],
            "process_next_queue_item",
        )
        self.assertEqual(
            status["operations_event_stream"]["dashboard"]["secondary_action"],
            "inspect_auto_train_trigger",
        )
        self.assertEqual(status["operations_alert_policy"]["required_action"], "process_next_queue_item")
        self.assertEqual(status["operations_alert_policy"]["secondary_action"], "inspect_auto_train_trigger")
        self.assertEqual(
            status["operations_alert_policy"]["summary_line"],
            "current_focus=queue_waiting_execution | required_action=process_next_queue_item | next_actions=process_next_queue_item,inspect_auto_train_trigger",
        )
        self.assertEqual(
            status["operations_dashboard"]["inspection_summary_line"],
            "current_focus=queue_waiting_execution | required_action=process_next_queue_item | next_actions=process_next_queue_item,inspect_auto_train_trigger",
        )
        self.assertEqual(
            status["operations_event_stream"]["dashboard"]["inspection_summary_line"],
            "current_focus=queue_waiting_execution | required_action=process_next_queue_item | next_actions=process_next_queue_item,inspect_auto_train_trigger",
        )
        self.assertEqual(
            status["operations_event_stream"]["summary_line"],
            "current_focus=queue_waiting_execution | required_action=process_next_queue_item | next_actions=process_next_queue_item,inspect_auto_train_trigger",
        )
        self.assertEqual(
            status["operations_event_stream"]["dashboard"]["summary_line"],
            "current_focus=queue_waiting_execution | required_action=process_next_queue_item | next_actions=process_next_queue_item,inspect_auto_train_trigger",
        )
        self.assertEqual(
            status["operations_console"]["next_actions"][:2],
            ["process_next_queue_item", "inspect_auto_train_trigger"],
        )

    def test_http_status_embeds_operations_console(self) -> None:
        service = self._service()
        app = self._app(service)

        async def scenario() -> dict[str, object]:
            result = await smoke_test_request(app, path="/pfe/status", method="GET")
            return result["body"]

        body = asyncio.run(scenario())
        self.assertIn("operations_console", body)
        self.assertIn("operations_console", body["metadata"])
        self.assertIn("operations_dashboard", body)
        self.assertIn("operations_alert_policy", body)
        self.assertIn("operations_daemon_timeline", body)
        self.assertIn("operations_event_stream", body)
        self.assertIn("dashboard", body["operations_event_stream"])
        self.assertIn("next_actions", body["operations_event_stream"])
        self.assertIn("summary_line", body["operations_dashboard"])
        self.assertIn("next_actions", body["operations_dashboard"])
        self.assertEqual(body["operations_dashboard"]["current_focus"], "idle")
        self.assertEqual(body["operations_dashboard"]["required_action"], "observe_and_monitor")
        self.assertIsNone(body["operations_dashboard"]["last_recovery_event"])
        self.assertIn("inspection_summary_line", body["operations_dashboard"])
        self.assertEqual(body["operations_alert_policy"]["current_focus"], "idle")
        self.assertEqual(body["operations_alert_policy"]["required_action"], "observe_and_monitor")
        self.assertIsNone(body["operations_alert_policy"].get("last_recovery_event"))
        self.assertIn("inspection_summary_line", body["operations_alert_policy"])
        self.assertEqual(body["operations_event_stream"]["dashboard"]["current_focus"], "idle")
        self.assertEqual(body["operations_event_stream"]["dashboard"]["required_action"], "observe_and_monitor")
        self.assertIsNone(body["operations_event_stream"]["dashboard"].get("last_recovery_event"))
        self.assertIn("inspection_summary_line", body["operations_event_stream"]["dashboard"])
        self.assertIn("candidate", body["operations_console"])
        self.assertIn("queue", body["operations_console"])
        self.assertIn("runner", body["operations_console"])
        self.assertIn("runner_timeline", body["operations_console"])
        self.assertIn("daemon_timeline", body["operations_console"])
        self.assertIn("timelines", body["operations_console"])
        self.assertIn("operations_daemon_timeline", body["metadata"])
        self.assertIn("runner_timeline", body)
        self.assertIn("operations_runner_timeline", body)
        self.assertIn("runner_timeline", body["metadata"])
        self.assertIn("operations_runner_timeline", body["metadata"])
        self.assertIn("operations_dashboard", body["metadata"])
        self.assertIn("operations_alert_policy", body["metadata"])
        self.assertIn("operations_event_stream", body["metadata"])
        self.assertIn("dashboard", body["operations_console"]["event_stream"])
        self.assertIn("alerts", body["operations_console"])
        self.assertIn("health", body["operations_console"])
        self.assertIn("recovery", body["operations_console"])
        self.assertIn("event_stream", body["operations_console"])
        self.assertIn("auto_train_policy", body["operations_console"])
        self.assertIn("queue_review_policy", body["operations_console"])
        self.assertIn("queue_review_policy_summary", body["operations_event_stream"])
        self.assertIn("queue_review_policy_summary", body["operations_event_stream"]["dashboard"])
        self.assertIn("trigger_policy_summary", body["operations_overview"])
        self.assertIn("queue_review_policy_summary", body["operations_overview"])


if __name__ == "__main__":
    unittest.main()
