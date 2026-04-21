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

from pfe_cli.main import _format_status
from pfe_core.adapter_store.store import AdapterStore
from tests.matrix_test_compat import strip_ansi
from pfe_core.config import PFEConfig
from pfe_core.pipeline import PipelineService
from pfe_core.server_services import InferenceServiceAdapter, PipelineServiceAdapter
from pfe_server.app import ServiceBundle, create_app, smoke_test_request
from pfe_server.auth import ServerSecurityConfig


class OperationsEventStreamSurfaceTests(unittest.TestCase):
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

    def test_http_status_exposes_event_stream_severity_and_attention(self) -> None:
        service = self._service()
        stale_time = (datetime.now(timezone.utc) - timedelta(minutes=12)).isoformat()
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
        service.status()
        app = self._app(service)

        async def scenario() -> dict[str, object]:
            result = await smoke_test_request(app, path="/pfe/status", method="GET")
            return result["body"]

        body = asyncio.run(scenario())
        event_stream = body["operations_event_stream"]
        alert_policy = body["operations_alert_policy"]
        self.assertEqual(event_stream["severity"], "critical")
        self.assertTrue(event_stream["attention_needed"])
        self.assertEqual(event_stream["attention_reason"], "daemon_stale")
        self.assertEqual(event_stream["status"], "attention")
        self.assertIn("next_actions", event_stream)
        self.assertIn("dashboard", event_stream)
        self.assertIn("policy", event_stream)
        self.assertEqual(event_stream["current_focus"], "daemon_stale")
        self.assertEqual(event_stream["required_action"], "recover_worker_daemon")
        self.assertEqual(event_stream["highest_priority_action"], "recover_worker_daemon")
        self.assertEqual(event_stream["active_recovery_hint"], "manual_recover")
        self.assertIn("daemon_stale", event_stream["escalated_reasons"])
        self.assertIsNone(event_stream["latest_recovery"])
        self.assertIn("recover_worker_daemon", event_stream["next_actions"])
        self.assertIn("inspection_summary_line", event_stream)
        self.assertEqual(event_stream["dashboard"]["severity"], "critical")
        self.assertEqual(event_stream["dashboard"]["status"], "attention")
        self.assertTrue(event_stream["dashboard"]["attention_needed"])
        self.assertEqual(event_stream["dashboard"]["current_focus"], "daemon_stale")
        self.assertEqual(event_stream["dashboard"]["required_action"], "recover_worker_daemon")
        self.assertEqual(event_stream["dashboard"]["highest_priority_action"], "recover_worker_daemon")
        self.assertEqual(event_stream["dashboard"]["active_recovery_hint"], "manual_recover")
        self.assertEqual(event_stream["dashboard"]["remediation_mode"], "auto")
        self.assertTrue(event_stream["dashboard"]["auto_remediation_allowed"])
        self.assertIn("recover_worker_daemon", event_stream["dashboard"]["next_actions"])
        self.assertEqual(alert_policy["required_action"], "recover_worker_daemon")
        self.assertEqual(alert_policy["current_focus"], "daemon_stale")
        self.assertEqual(alert_policy["highest_priority_action"], "recover_worker_daemon")
        self.assertEqual(alert_policy["active_recovery_hint"], "manual_recover")
        self.assertEqual(alert_policy["remediation_mode"], "auto")
        self.assertIn("recover the worker daemon first", alert_policy["operator_guidance"])
        self.assertIn("daemon_stale", alert_policy["escalated_reasons"])
        self.assertIsNone(alert_policy["latest_recovery"])
        self.assertTrue(alert_policy["requires_immediate_action"])
        self.assertEqual(alert_policy["action_priority"], "p0")
        self.assertIn("severity=critical", event_stream["summary_line"])
        self.assertIn("highest_priority_action=recover_worker_daemon", event_stream["summary_line"])
        self.assertIn("active_recovery_hint=manual_recover", event_stream["summary_line"])
        self.assertIn("operations_event_stream", body["metadata"])
        self.assertIn("operations_alert_policy", body["metadata"])
        self.assertEqual(body["metadata"]["operations_event_stream"]["severity"], "critical")
        self.assertEqual(body["operations_console"]["event_stream"]["severity"], "critical")

    def test_status_exposes_queue_review_policy_in_event_stream_for_auto_queue(self) -> None:
        service = self._service()
        service.generate(scenario="life-coach", style="warm", num_samples=8)
        trained = service.train_result(method="qlora", epochs=1, train_type="sft")
        AdapterStore(home=self.pfe_home).promote(trained.version)
        service._append_train_queue_item(
            {
                "job_id": "job-queue-auto-1",
                "state": "queued",
                "workspace": "user_default",
                "source": "signal_auto_train",
                "priority": 100,
                "priority_source": "policy:source_default",
            },
            workspace="user_default",
        )

        status = service.status()
        event_stream = status["operations_console"]["event_stream"]
        dashboard = status["operations_console"]["dashboard"]

        self.assertEqual(event_stream["queue_review_mode"], "auto_queue")
        self.assertEqual(event_stream["queue_review_next_action"], "process_next_queue_item")
        self.assertEqual(event_stream["current_focus"], "queue_waiting_execution")
        self.assertEqual(event_stream["monitor_focus"], "queue_waiting_execution")
        self.assertEqual(event_stream["required_action"], "process_next_queue_item")
        self.assertEqual(event_stream["highest_priority_action"], "process_next_queue_item")
        self.assertEqual(event_stream["queue_action_summary"]["primary_action"], "process_next_queue_item")
        self.assertIn("inspection_summary_line", event_stream)
        self.assertEqual(event_stream["dashboard"]["queue_review_mode"], "auto_queue")
        self.assertEqual(event_stream["dashboard"]["queue_review_next_action"], "process_next_queue_item")
        self.assertEqual(event_stream["dashboard"]["queue_action_summary"]["primary_action"], "process_next_queue_item")
        self.assertEqual(event_stream["policy"]["queue_review_mode"], "auto_queue")
        self.assertEqual(event_stream["policy"]["required_action"], "process_next_queue_item")
        self.assertEqual(event_stream["policy"]["secondary_action"], "inspect_auto_train_trigger")
        self.assertEqual(event_stream["policy"]["queue_action_summary"]["primary_action"], "process_next_queue_item")
        self.assertIn("continue automatically", event_stream["policy"]["operator_guidance"])
        self.assertEqual(
            event_stream["summary_line"],
            "current_focus=queue_waiting_execution | required_action=process_next_queue_item | next_actions=process_next_queue_item,inspect_auto_train_trigger",
        )
        self.assertEqual(
            dashboard["summary_line"],
            "current_focus=queue_waiting_execution | required_action=process_next_queue_item | next_actions=process_next_queue_item,inspect_auto_train_trigger",
        )
        self.assertEqual(
            dashboard["dashboard_digest"],
            "current_focus=queue_waiting_execution | required_action=process_next_queue_item | next_actions=process_next_queue_item,inspect_auto_train_trigger",
        )
        self.assertEqual(dashboard["queue_review_mode"], "auto_queue")
        self.assertEqual(dashboard["queue_review_next_action"], "process_next_queue_item")

    def test_status_exposes_queue_processing_gate_in_event_stream(self) -> None:
        config = PFEConfig.load(home=self.pfe_home)
        config.trainer.trigger.enabled = True
        config.trainer.trigger.min_new_samples = 0
        config.trainer.trigger.max_interval_days = 0
        config.save(home=self.pfe_home)

        service = self._service()
        service.generate(scenario="life-coach", style="warm", num_samples=8)
        service._append_train_queue_item(
            {
                "job_id": "job-queue-running-1",
                "state": "running",
                "workspace": "user_default",
                "source": "signal_auto_train",
                "priority": 100,
            },
            workspace="user_default",
        )

        status = service.status()
        overview = status["operations_overview"]
        event_stream = status["operations_console"]["event_stream"]

        self.assertEqual(overview["queue_gate_reason"], "queue_processing_active")
        self.assertEqual(overview["queue_gate_action"], "wait_for_queue_completion")
        self.assertEqual(event_stream["highest_priority_action"], "wait_for_queue_completion")
        self.assertEqual(event_stream["severity"], "info")
        self.assertFalse(event_stream["attention_needed"])
        self.assertEqual(
            event_stream["summary_line"],
            "current_focus=queue_waiting_execution | required_action=wait_for_queue_completion | next_actions=wait_for_queue_completion,inspect_auto_train_trigger",
        )

    def test_status_exposes_auto_train_policy_gate_in_event_stream(self) -> None:
        config = PFEConfig.load(home=self.pfe_home)
        config.trainer.trigger.enabled = True
        config.trainer.trigger.auto_evaluate = False
        config.trainer.trigger.auto_promote = True
        config.save(home=self.pfe_home)

        service = self._service()
        status = service.status()
        event_stream = status["operations_console"]["event_stream"]
        dashboard = status["operations_console"]["dashboard"]

        self.assertEqual(event_stream["trigger_policy_reason"], "policy_requires_auto_evaluate")
        self.assertEqual(event_stream["trigger_policy_action"], "enable_auto_evaluate")
        self.assertEqual(event_stream["highest_priority_action"], "enable_auto_evaluate")
        self.assertIn("required_action=enable_auto_evaluate", event_stream["summary_line"])
        self.assertIn("trigger_policy_reason=policy_requires_auto_evaluate", event_stream["summary_line"])
        self.assertIn("trigger_policy_action=enable_auto_evaluate", event_stream["summary_line"])
        self.assertEqual(event_stream["policy"]["required_action"], "enable_auto_evaluate")
        self.assertEqual(event_stream["policy"]["trigger_policy_reason"], "policy_requires_auto_evaluate")
        self.assertEqual(event_stream["policy"]["trigger_policy_action"], "enable_auto_evaluate")
        self.assertIn("trigger_policy_gate_summary", event_stream)
        self.assertIn("eval=", event_stream["trigger_policy_gate_summary_line"])
        self.assertIn("enable auto-evaluate", event_stream["policy"]["operator_guidance"])
        self.assertIn("required_action=enable_auto_evaluate", dashboard["summary_line"])
        self.assertIn("pgate=policy_requires_auto_evaluate", dashboard["summary_line"])
        self.assertIn("pact=enable_auto_evaluate", dashboard["summary_line"])
        self.assertIn("pg=eval=", dashboard["summary_line"])
        self.assertIn("policy=enable_auto_evaluate", dashboard["dashboard_digest"])

    def test_status_exposes_trigger_threshold_summary_in_event_stream(self) -> None:
        config = PFEConfig.load(home=self.pfe_home)
        config.trainer.trigger.enabled = True
        config.trainer.trigger.min_new_samples = 50
        config.trainer.trigger.max_interval_days = 7
        config.trainer.trigger.min_trigger_interval_minutes = 30
        config.trainer.trigger.failure_backoff_minutes = 15
        config.save(home=self.pfe_home)

        service = self._service()
        service.generate(scenario="life-coach", style="warm", num_samples=8)
        status = service.status()
        event_stream = status["operations_console"]["event_stream"]
        dashboard = status["operations_console"]["dashboard"]

        self.assertIn("trigger_threshold_summary", event_stream)
        self.assertIn("trigger_gate=", event_stream["summary_line"])
        self.assertIn("samples=", event_stream["trigger_threshold_summary_line"])
        self.assertIn("trigger_threshold_summary", event_stream["dashboard"])
        self.assertIn("gate=samples=", dashboard["summary_line"])
        self.assertIn("gate=samples=", dashboard["dashboard_digest"])

    def test_status_exposes_candidate_action_summary_in_event_stream(self) -> None:
        service = self._service()
        service.generate(scenario="life-coach", style="warm", num_samples=6)
        first = service.train_result(method="qlora", epochs=1, train_type="sft")
        store = AdapterStore(home=self.pfe_home)
        store.promote(first.version)
        service.generate(scenario="work-coach", style="direct", num_samples=6)
        second = service.train_result(method="qlora", epochs=1, train_type="sft")
        store.archive(second.version)

        status = service.status()
        event_stream = status["operations_event_stream"]
        dashboard = status["operations_dashboard"]
        policy = status["operations_alert_policy"]

        self.assertEqual(event_stream["dashboard"]["current_focus"], "candidate_idle")
        self.assertEqual(event_stream["candidate_action_summary"]["primary_action"], "inspect_candidate_status")
        self.assertEqual(event_stream["highest_priority_action"], "inspect_candidate_status")
        self.assertEqual(
            event_stream["dashboard"]["candidate_action_summary"]["primary_action"],
            "inspect_candidate_status",
        )
        self.assertEqual(policy["candidate_action_summary"]["primary_action"], "inspect_candidate_status")
        self.assertEqual(dashboard["candidate_action_summary"]["primary_action"], "inspect_candidate_status")

    def test_cli_status_formats_event_stream_classification(self) -> None:
        payload = {
            "workspace": "user_default",
            "operations_event_stream": {
                "count": 3,
                "severity": "warning",
                "status": "attention",
                "attention_needed": True,
                "attention_reason": "awaiting_confirmation",
                "attention_source": "queue",
                "latest_source": "queue",
                "latest_event": "completed",
                "latest_reason": "awaiting_confirmation",
                "latest_timestamp": "2026-03-26T10:06:00+00:00",
                "alert_count": 2,
                "highest_priority_action": "review_queue_confirmation",
                "active_recovery_hint": "manual_review_required_by_policy",
                "queue_review_policy_summary": "mode=manual_review | entry=awaiting_confirmation | next=review_queue_confirmation | reason=manual_review_required_by_policy",
                "queue_review_mode": "manual_review",
                "queue_review_next_action": "review_queue_confirmation",
                "queue_review_reason": "manual_review_required_by_policy",
                "queue_review_required_now": True,
                "escalated_reasons": ["awaiting_confirmation"],
                "latest_recovery": {
                    "source": "daemon",
                    "event": "recover_requested",
                    "reason": "daemon_stale",
                },
                "current_focus": "awaiting_confirmation",
                "required_action": "review_queue_confirmation",
                "last_recovery_event": "recover_requested",
                "last_recovery_reason": "daemon_stale",
                "last_recovery_note": "auto_recovery",
                "next_actions": ["review_queue_confirmation"],
                "dashboard": {
                    "severity": "warning",
                    "status": "attention",
                    "attention_needed": True,
                    "attention_reason": "awaiting_confirmation",
                    "current_focus": "awaiting_confirmation",
                    "required_action": "review_queue_confirmation",
                    "last_recovery_event": "recover_requested",
                    "last_recovery_reason": "daemon_stale",
                    "last_recovery_note": "auto_recovery",
                    "highest_priority_action": "review_queue_confirmation",
                    "active_recovery_hint": "manual_review_required_by_policy",
                    "queue_review_policy_summary": "mode=manual_review | entry=awaiting_confirmation | next=review_queue_confirmation | reason=manual_review_required_by_policy",
                    "queue_review_mode": "manual_review",
                    "queue_review_next_action": "review_queue_confirmation",
                    "queue_review_reason": "manual_review_required_by_policy",
                    "queue_review_required_now": True,
                    "remediation_mode": "manual_review",
                    "operator_guidance": "review the warning, confirm intent, and decide whether to continue or intervene",
                    "latest_source": "queue",
                    "latest_event": "completed",
                    "latest_reason": "awaiting_confirmation",
                    "inspection_summary_line": "current_focus=awaiting_confirmation | required_action=review_queue_confirmation | last_recovery_event=recover_requested | last_recovery_reason=daemon_stale | last_recovery_note=auto_recovery | next_actions=review_queue_confirmation",
                },
                "policy": {
                    "severity": "warning",
                    "required_action": "review_queue_confirmation",
                    "primary_action": "review_queue_confirmation",
                    "action_priority": "p1",
                    "escalation_mode": "review_soon",
                    "requires_human_review": True,
                    "auto_remediation_allowed": False,
                    "remediation_mode": "manual_review",
                    "operator_guidance": "review the queued training request, confirm intent, and approve or reject the queue confirmation",
                    "highest_priority_action": "review_queue_confirmation",
                    "active_recovery_hint": "manual_review_required_by_policy",
                    "escalated_reasons": ["awaiting_confirmation"],
                    "queue_review_mode": "manual_review",
                    "queue_review_next_action": "review_queue_confirmation",
                    "queue_review_reason": "manual_review_required_by_policy",
                    "queue_review_required_now": True,
                    "current_focus": "awaiting_confirmation",
                    "last_recovery_event": "recover_requested",
                    "last_recovery_reason": "daemon_stale",
                    "last_recovery_note": "auto_recovery",
                    "next_actions": ["review_queue_confirmation"],
                    "inspection_summary_line": "current_focus=awaiting_confirmation | required_action=review_queue_confirmation | last_recovery_event=recover_requested | last_recovery_reason=daemon_stale | last_recovery_note=auto_recovery | next_actions=review_queue_confirmation",
                },
                "operations_dashboard": {
                    "severity": "warning",
                    "status": "attention",
                    "attention_needed": True,
                    "attention_reason": "awaiting_confirmation",
                    "current_focus": "awaiting_confirmation",
                    "required_action": "review_queue_confirmation",
                    "last_recovery_event": "recover_requested",
                    "last_recovery_reason": "daemon_stale",
                    "last_recovery_note": "auto_recovery",
                    "remediation_mode": "manual_review",
                    "operator_guidance": "review the warning, confirm intent, and decide whether to continue or intervene",
                    "next_actions": ["review_queue_confirmation"],
                    "inspection_summary_line": "current_focus=awaiting_confirmation | required_action=review_queue_confirmation | last_recovery_event=recover_requested | last_recovery_reason=daemon_stale | last_recovery_note=auto_recovery | next_actions=review_queue_confirmation",
                },
                "operations_alert_policy": {
                    "severity": "warning",
                    "required_action": "review_queue_confirmation",
                    "current_focus": "awaiting_confirmation",
                    "primary_action": "review_queue_confirmation",
                    "action_priority": "p1",
                    "escalation_mode": "review_soon",
                    "requires_human_review": True,
                    "auto_remediation_allowed": False,
                    "highest_priority_action": "review_queue_confirmation",
                    "active_recovery_hint": "manual_review_required_by_policy",
                    "escalated_reasons": ["awaiting_confirmation"],
                    "last_recovery_event": "recover_requested",
                    "last_recovery_reason": "daemon_stale",
                    "last_recovery_note": "auto_recovery",
                    "remediation_mode": "manual_review",
                    "operator_guidance": "review the warning, confirm intent, and decide whether to continue or intervene",
                    "next_actions": ["review_queue_confirmation"],
                    "inspection_summary_line": "current_focus=awaiting_confirmation | required_action=review_queue_confirmation | last_recovery_event=recover_requested | last_recovery_reason=daemon_stale | last_recovery_note=auto_recovery | next_actions=review_queue_confirmation",
                },
                "summary_line": "count=3 | severity=warning | attention_needed=yes | attention_reason=awaiting_confirmation | latest_source=queue | latest_event=completed | latest_reason=awaiting_confirmation",
                "items": [
                    {"timestamp": "2026-03-26T10:06:00+00:00", "source": "queue", "event": "completed", "reason": "awaiting_confirmation"},
                    {"timestamp": "2026-03-26T10:05:30+00:00", "source": "daemon", "event": "recovery", "reason": "daemon_stale"},
                ],
            },
            "operations_dashboard": {
                "severity": "warning",
                "status": "attention",
                "attention_needed": True,
                "attention_reason": "awaiting_confirmation",
                "current_focus": "awaiting_confirmation",
                "required_action": "review_queue_confirmation",
                "last_recovery_event": "recover_requested",
                "last_recovery_reason": "daemon_stale",
                "last_recovery_note": "auto_recovery",
                "remediation_mode": "manual_review",
                "operator_guidance": "review the warning, confirm intent, and decide whether to continue or intervene",
                "next_actions": ["review_queue_confirmation"],
                "inspection_summary_line": "current_focus=awaiting_confirmation | required_action=review_queue_confirmation | last_recovery_event=recover_requested | last_recovery_reason=daemon_stale | last_recovery_note=auto_recovery | next_actions=review_queue_confirmation",
            },
            "operations_alert_policy": {
                "severity": "warning",
                "required_action": "review_queue_confirmation",
                "current_focus": "awaiting_confirmation",
                "primary_action": "review_queue_confirmation",
                "action_priority": "p1",
                "escalation_mode": "review_soon",
                "requires_human_review": True,
                "auto_remediation_allowed": False,
                "remediation_mode": "manual_review",
                "operator_guidance": "review the warning, confirm intent, and decide whether to continue or intervene",
                "last_recovery_event": "recover_requested",
                "last_recovery_reason": "daemon_stale",
                "last_recovery_note": "auto_recovery",
                "next_actions": ["review_queue_confirmation"],
                "inspection_summary_line": "current_focus=awaiting_confirmation | required_action=review_queue_confirmation | last_recovery_event=recover_requested | last_recovery_reason=daemon_stale | last_recovery_note=auto_recovery | next_actions=review_queue_confirmation",
            },
        }

        text = _format_status(payload, workspace="user_default")
        clean = strip_ansi(text)
        self.assertIn("[ OPERATIONS EVENT STREAM ]", clean)
        self.assertIn("severity:", clean)
        self.assertIn("warning", clean)
        self.assertIn("attention needed:", clean)
        self.assertIn("awaiting_confirmation", clean)
        self.assertIn("status:", clean)
        self.assertIn("attention", clean)
        self.assertIn("count:", clean)
        self.assertIn("review_queue_confirmation", clean)
        self.assertIn("manual_review_required_by_policy", clean)
        self.assertIn("next actions:", clean)
        self.assertIn("current focus:", clean)
        self.assertIn("last recovery event:", clean)
        self.assertIn("recover_requested", clean)
        self.assertIn("manual_review", clean)
        self.assertIn("summary line:", clean)
        self.assertIn("current focus:", clean)
        self.assertIn("last recovery reason:", clean)
        self.assertIn("daemon_stale", clean)
        self.assertIn("last recovery note:", clean)
        self.assertIn("auto_recovery", clean)


if __name__ == "__main__":
    unittest.main()
