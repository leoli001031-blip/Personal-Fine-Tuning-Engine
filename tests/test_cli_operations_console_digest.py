from __future__ import annotations

import os
import unittest
from pathlib import Path

from typer.testing import CliRunner

ROOT = Path(__file__).resolve().parents[1]
for package_dir in ("pfe-core", "pfe-cli", "pfe-server"):
    package_path = str(ROOT / package_dir)
    if package_path not in os.sys.path:
        os.sys.path.insert(0, package_path)

from pfe_cli import main as cli_main
from pfe_cli.main import (
    _format_operations_alert_policy,
    _format_operations_alert_surface,
    _format_operations_dashboard,
    _format_operations_event_stream,
    _format_ops_attention,
    _format_status,
)
from tests.matrix_test_compat import strip_ansi


class CLIOperationsConsoleDigestTests(unittest.TestCase):
    def test_ops_attention_prefers_generic_monitor_summary_over_legacy_queue_bits(self) -> None:
        text = _format_ops_attention(
            operations_alerts=[],
            operations_overview=None,
            operations_dashboard={
                "current_focus": "none",
                "monitor_focus": "queue_waiting_execution",
                "required_action": "process_next_queue_item",
            },
            operations_alert_policy={
                "current_focus": "none",
                "required_action": "process_next_queue_item",
            },
            candidate_summary=None,
            train_queue={
                "counts": {"queued": 1},
                "worker_runner": {"active": False, "lock_state": "idle", "stop_requested": False},
            },
            latest_adapter_map=None,
            recent_adapter_map=None,
        )

        self.assertEqual(
            text,
            "ops attention: monitor current_focus=queue_waiting_execution | required_action=process_next_queue_item",
        )

    def test_ops_attention_prefers_generic_monitor_summary_for_candidate_and_runtime(self) -> None:
        candidate_text = _format_ops_attention(
            operations_alerts=[],
            operations_overview=None,
            operations_dashboard={
                "current_focus": "none",
                "monitor_focus": "candidate_idle",
                "required_action": "inspect_candidate_status",
            },
            operations_alert_policy={
                "current_focus": "none",
                "required_action": "inspect_candidate_status",
            },
            candidate_summary={
                "candidate_version": "20260404-001",
                "candidate_state": "archived",
                "candidate_needs_promotion": False,
            },
            train_queue=None,
            latest_adapter_map=None,
            recent_adapter_map=None,
        )
        self.assertEqual(
            candidate_text,
            "ops attention: monitor current_focus=candidate_idle | required_action=inspect_candidate_status",
        )

        runtime_text = _format_ops_attention(
            operations_alerts=[],
            operations_overview=None,
            operations_dashboard={
                "current_focus": "none",
                "monitor_focus": "runner_active",
                "required_action": "inspect_runtime_stability",
            },
            operations_alert_policy={
                "current_focus": "none",
                "required_action": "inspect_runtime_stability",
            },
            candidate_summary=None,
            train_queue={
                "worker_runner": {
                    "active": True,
                    "lock_state": "active",
                    "stop_requested": False,
                }
            },
            latest_adapter_map=None,
            recent_adapter_map=None,
        )
        self.assertEqual(
            runtime_text,
            "ops attention: monitor current_focus=runner_active | required_action=inspect_runtime_stability",
        )

    def test_ops_attention_prefers_overview_inspection_summary_for_generic_monitor(self) -> None:
        text = _format_ops_attention(
            operations_alerts=[],
            operations_overview={
                "current_focus": "queue_waiting_execution",
                "monitor_focus": "queue_waiting_execution",
                "required_action": "process_next_queue_item",
                "inspection_summary_line": "current_focus=queue_waiting_execution | required_action=process_next_queue_item | next_actions=process_next_queue_item,inspect_auto_train_trigger",
            },
            operations_dashboard={
                "current_focus": "none",
                "monitor_focus": "queue_waiting_execution",
                "required_action": "process_next_queue_item",
            },
            operations_alert_policy={
                "current_focus": "none",
                "required_action": "process_next_queue_item",
            },
            candidate_summary=None,
            train_queue={
                "counts": {"queued": 1},
                "worker_runner": {"active": False, "lock_state": "idle", "stop_requested": False},
            },
            latest_adapter_map=None,
            recent_adapter_map=None,
        )

        self.assertEqual(
            text,
            "ops attention: monitor current_focus=queue_waiting_execution | required_action=process_next_queue_item | next_actions=process_next_queue_item,inspect_auto_train_trigger",
        )

    def test_alert_surface_formatter_prefers_monitor_focus_when_current_focus_is_none(self) -> None:
        lines = _format_operations_alert_surface(
            {
                "operations_next_actions": ["process_next_queue_item"],
                "operations_overview": {
                    "attention_needed": False,
                    "summary_line": "queue=1 | runner=idle",
                    "monitor_focus": "queue_waiting_execution",
                    "required_action": "process_next_queue_item",
                    "inspection_summary_line": "current_focus=queue_waiting_execution | required_action=process_next_queue_item | next_actions=process_next_queue_item,inspect_auto_train_trigger",
                },
                "operations_dashboard": {
                    "current_focus": "none",
                    "monitor_focus": "queue_waiting_execution",
                    "required_action": "process_next_queue_item",
                },
                "operations_alert_policy": {
                    "current_focus": "none",
                    "required_action": "process_next_queue_item",
                },
                "operations_console": {
                    "summary_line": "queue=1 | runner=idle",
                },
            }
        )

        text = "\n".join(lines or [])
        self.assertIn("operations alerts:", text)
        self.assertIn("current_focus=queue_waiting_execution", text)
        self.assertIn("required_action=process_next_queue_item", text)
        self.assertIn(
            "summary=current_focus=queue_waiting_execution | required_action=process_next_queue_item | next_actions=process_next_queue_item,inspect_auto_train_trigger",
            text,
        )
        self.assertNotIn("inspection_summary=", text)
        self.assertNotIn("current_focus=none", text)

    def test_alert_surface_derived_alert_detail_prefers_inspection_summary_for_generic_monitor(self) -> None:
        lines = _format_operations_alert_surface(
            {
                "operations_overview": {
                    "attention_needed": True,
                    "attention_reason": "queue_waiting_execution",
                    "summary_line": "legacy queue=1 | runner=idle",
                    "monitor_focus": "queue_waiting_execution",
                    "required_action": "process_next_queue_item",
                    "inspection_summary_line": "current_focus=queue_waiting_execution | required_action=process_next_queue_item | next_actions=process_next_queue_item,inspect_auto_train_trigger",
                },
                "operations_dashboard": {
                    "current_focus": "none",
                    "monitor_focus": "queue_waiting_execution",
                    "required_action": "process_next_queue_item",
                },
                "operations_alert_policy": {
                    "current_focus": "none",
                    "required_action": "process_next_queue_item",
                },
                "operations_console": {
                    "attention_needed": True,
                    "summary_line": "legacy queue=1 | runner=idle",
                },
            }
        )

        text = "\n".join(lines or [])
        self.assertIn("alerts:", text)
        self.assertIn(
            "detail=current_focus=queue_waiting_execution | required_action=process_next_queue_item | next_actions=process_next_queue_item,inspect_auto_train_trigger",
            text,
        )
        self.assertNotIn("detail=legacy queue=1 | runner=idle", text)

    def test_alert_policy_formatter_prefers_monitor_focus_when_current_focus_is_none(self) -> None:
        lines = _format_operations_alert_policy(
            {
                "severity": "info",
                "current_focus": "none",
                "monitor_focus": "daemon_active",
                "required_action": "inspect_runtime_stability",
                "secondary_action": "inspect_daemon_status",
                "action_priority": "p2",
                "escalation_mode": "monitor",
                "summary_line": "severity=info | focus=daemon_active | next=inspect_runtime_stability",
                "inspection_summary_line": "current_focus=daemon_active | required_action=inspect_runtime_stability | next_actions=inspect_runtime_stability,inspect_daemon_status",
            }
        )

        text = "\n".join(lines or [])
        self.assertIn("operations alert policy:", text)
        self.assertIn("current_focus=daemon_active", text)
        self.assertIn(
            "summary=current_focus=daemon_active | required_action=inspect_runtime_stability | next_actions=inspect_runtime_stability,inspect_daemon_status",
            text,
        )
        self.assertNotIn("inspection_summary=", text)
        self.assertNotIn("current_focus=none", text)

    def test_event_stream_formatter_prefers_inspection_summary_for_generic_monitor(self) -> None:
        lines = _format_operations_event_stream(
            {
                "current_focus": "none",
                "monitor_focus": "queue_waiting_execution",
                "required_action": "process_next_queue_item",
                "summary_line": "severity=info | focus=queue_waiting_execution | next=process_next_queue_item",
                "inspection_summary_line": "current_focus=queue_waiting_execution | required_action=process_next_queue_item | next_actions=process_next_queue_item,inspect_auto_train_trigger",
                "dashboard": {
                    "current_focus": "none",
                    "monitor_focus": "queue_waiting_execution",
                    "required_action": "process_next_queue_item",
                    "summary_line": "severity=info | focus=queue_waiting_execution | next=process_next_queue_item",
                    "inspection_summary_line": "current_focus=queue_waiting_execution | required_action=process_next_queue_item | next_actions=process_next_queue_item,inspect_auto_train_trigger",
                },
            }
        )

        text = "\n".join(lines or [])
        self.assertIn(
            "summary=current_focus=queue_waiting_execution | required_action=process_next_queue_item | next_actions=process_next_queue_item,inspect_auto_train_trigger",
            text,
        )
        self.assertNotIn("inspection_summary=", text)
        self.assertIn(
            "dashboard: current_focus=queue_waiting_execution | required_action=process_next_queue_item | summary=current_focus=queue_waiting_execution | required_action=process_next_queue_item | next_actions=process_next_queue_item,inspect_auto_train_trigger",
            text,
        )

    def test_dashboard_formatter_prefers_inspection_summary_for_generic_monitor(self) -> None:
        lines = _format_operations_dashboard(
            {
                "current_focus": "none",
                "monitor_focus": "daemon_active",
                "required_action": "inspect_runtime_stability",
                "summary_line": "severity=info | focus=daemon_active | next=inspect_runtime_stability",
                "inspection_summary_line": "current_focus=daemon_active | required_action=inspect_runtime_stability | next_actions=inspect_runtime_stability,inspect_daemon_status",
            }
        )

        text = "\n".join(lines or [])
        self.assertIn(
            "summary=current_focus=daemon_active | required_action=inspect_runtime_stability | next_actions=inspect_runtime_stability,inspect_daemon_status",
            text,
        )
        self.assertNotIn("inspection_summary=", text)
        self.assertNotIn("dashboard_digest=", text)

    def test_status_formatter_prefers_monitor_focus_when_current_focus_is_none(self) -> None:
        payload = {
            "workspace": "user_default",
            "operations_overview": {
                "attention_needed": False,
                "summary_line": "queue=1 | runner=idle | daemon-health=stopped",
                "current_focus": "queue_waiting_execution",
                "monitor_focus": "queue_waiting_execution",
                "required_action": "process_next_queue_item",
                "inspection_summary_line": "current_focus=queue_waiting_execution | required_action=process_next_queue_item | next_actions=process_next_queue_item,inspect_auto_train_trigger",
            },
            "operations_dashboard": {
                "severity": "info",
                "status": "healthy",
                "attention_needed": False,
                "current_focus": "none",
                "monitor_focus": "queue_waiting_execution",
                "required_action": "process_next_queue_item",
                "summary_line": "severity=info | focus=queue_waiting_execution | next=process_next_queue_item",
                "next_actions": ["process_next_queue_item", "inspect_auto_train_trigger"],
            },
            "operations_alert_policy": {
                "severity": "info",
                "current_focus": "none",
                "required_action": "process_next_queue_item",
                "secondary_action": "inspect_auto_train_trigger",
                "primary_action": "process_next_queue_item",
                "action_priority": "p2",
                "escalation_mode": "monitor",
            },
            "train_queue": {
                "count": 1,
                "worker_runner": {"active": False, "lock_state": "idle", "stop_requested": False},
            },
        }

        text = _format_status(payload, workspace="user_default")
        clean = strip_ansi(text)

        self.assertIn("[ OPERATIONS ]", clean)
        self.assertIn("current focus:           queue_waiting_execution", clean)
        self.assertIn("required action:         process_next_queue_item", clean)
        self.assertIn(
            "summary:                 queue=1 | runner=idle | daemon-health=stopped",
            clean,
        )
        self.assertNotIn(
            "inspection_summary=current_focus=queue_waiting_execution | required_action=process_next_queue_item | next_actions=process_next_queue_item,inspect_auto_train_trigger",
            clean,
        )
        self.assertIn("current focus:           queue_waiting_execution", clean)
        self.assertIn("[ OPERATIONS DASHBOARD ]", clean)
        self.assertIn("[ OPERATIONS ALERT POLICY ]", clean)
        self.assertIn("required action:         process_next_queue_item", clean)

    def test_status_formatter_synthesizes_operations_console_digest(self) -> None:
        payload = {
            "workspace": "user_default",
            "operations_overview": {
                "attention_needed": True,
                "attention_reason": "awaiting_confirmation",
                "summary_line": "trigger=ready | candidate-stage=promoted | awaiting-confirm=1 | runner-lock=idle",
            },
            "operations_dashboard": {
                "severity": "warning",
                "status": "attention",
                "attention_needed": True,
                "attention_reason": "awaiting_confirmation",
                "current_focus": "awaiting_confirmation",
                "queue_state": "awaiting_confirmation",
                "summary_line": "severity=warning | status=attention | attention=yes | focus=awaiting_confirmation | next=review_queue_confirmation",
                "next_actions": ["review_queue_confirmation"],
            },
            "operations_alert_policy": {
                "severity": "warning",
                "required_action": "review_queue_confirmation",
                "primary_action": "review_queue_confirmation",
                "action_priority": "p1",
                "escalation_mode": "review_soon",
                "requires_human_review": True,
                "auto_remediation_allowed": False,
                "summary_line": "severity=warning | required_action=review_queue_confirmation | priority=p1 | mode=review_soon | focus=awaiting_confirmation",
            },
            "candidate_summary": {
                "candidate_version": "20260326-200",
                "candidate_state": "promoted",
                "candidate_needs_promotion": True,
            },
            "candidate_history": {
                "count": 3,
                "last_reason": "candidate_promoted",
                "last_candidate_version": "20260326-200",
                "latest_timestamp": "2026-03-26T10:00:00+00:00",
            },
            "candidate_timeline": {
                "current_stage": "promoted",
                "transition_count": 2,
            },
            "runner_timeline": {
                "count": 4,
                "last_event": "completed",
                "last_reason": "max_seconds",
                "takeover_event_count": 1,
                "last_takeover_event": "stale_lock_takeover",
                "last_takeover_reason": "runner_already_active",
                "recent_anomaly_reason": "runner_already_active",
                "latest_timestamp": "2026-03-26T10:06:00+00:00",
                "recent_takeover_events": [
                    {
                        "timestamp": "2026-03-26T10:05:45+00:00",
                        "event": "stale_lock_takeover",
                        "reason": "runner_already_active",
                        "note": "manual_takeover",
                    }
                ],
            },
            "operations_event_stream": {
                "count": 4,
                "severity": "warning",
                "status": "attention",
                "attention_needed": True,
                "attention_reason": "awaiting_confirmation",
                "attention_source": "queue",
                "latest_source": "runner",
                "latest_event": "completed",
                "latest_reason": "max_seconds",
                "latest_timestamp": "2026-03-26T10:06:00+00:00",
                "next_actions": ["review_queue_confirmation"],
                "dashboard": {
                    "severity": "warning",
                    "status": "attention",
                    "attention_needed": True,
                    "attention_reason": "awaiting_confirmation",
                    "latest_source": "runner",
                    "latest_event": "completed",
                    "latest_reason": "max_seconds",
                },
                "items": [
                    {"timestamp": "2026-03-26T10:06:00+00:00", "source": "runner", "event": "completed", "reason": "max_seconds", "severity": "info", "attention": False},
                    {"timestamp": "2026-03-26T10:05:30+00:00", "source": "queue", "event": "alert", "reason": "awaiting_confirmation", "severity": "attention", "attention": True},
                ],
            },
            "train_queue": {
                "count": 3,
                "confirmation_summary": {
                    "awaiting_confirmation_count": 1,
                    "next_confirmation_reason": "manual_review_required_by_policy",
                },
                "history_summary": {
                    "last_transition": {"job_id": "job-queue-2", "event": "completed", "state": "completed"},
                    "last_reason": "training_completed",
                },
                "review_summary": {
                    "reviewed_transition_count": 2,
                    "last_review_event": "approved",
                    "last_review_note": "looks_good",
                },
                "worker_runner": {
                    "active": True,
                    "lock_state": "idle",
                    "last_event": "completed",
                    "lease_expires_at": "2026-03-26T10:05:00+00:00",
                    "history_count": 4,
                },
            },
        }

        text = _format_status(payload, workspace="user_default")
        clean = strip_ansi(text)

        self.assertIn("[ OPERATIONS ]", clean)
        self.assertIn("[ OPERATIONS DASHBOARD ]", clean)
        self.assertIn("[ OPERATIONS ALERT POLICY ]", clean)
        self.assertIn("current focus:           awaiting_confirmation", clean)
        self.assertIn("required action:         review_queue_confirmation", clean)
        self.assertIn("[ OPERATIONS EVENT STREAM ]", clean)
        self.assertIn("severity:                warning", clean)
        self.assertIn("attention reason:        awaiting_confirmation", clean)
        self.assertIn("[ RUNNER TIMELINE ]", clean)
        self.assertIn("count:                   4", clean)
        self.assertIn("last event:              completed", clean)
        self.assertIn("last reason:             max_seconds", clean)
        self.assertIn("takeover event count:    1", clean)
        self.assertIn("recent anomaly reason:   runner_already_active", clean)
        self.assertIn("[ CANDIDATE SUMMARY ]", clean)
        self.assertIn("candidate version:       20260326-200", clean)
        self.assertIn("candidate state:         promoted", clean)

    def test_status_command_renders_operations_console_digest(self) -> None:
        payload = {
            "workspace": "user_default",
            "operations_console": {
                "attention_needed": True,
                "attention_reason": "candidate_ready_for_promotion",
                "summary_line": "candidate-stage=pending_eval | runner-lock=idle",
                "next_actions": ["promote_candidate", "archive_candidate"],
                "candidate": {
                    "current_stage": "pending_eval",
                    "last_candidate_version": "20260326-201",
                    "last_reason": "candidate_ready",
                    "latest_timestamp": "2026-03-26T09:55:00+00:00",
                    "transition_count": 5,
                    "history_count": 5,
                },
                "queue": {
                    "count": 0,
                    "awaiting_confirmation_count": 0,
                    "last_reason": "idle",
                },
                "runner": {
                    "active": False,
                    "lock_state": "idle",
                    "last_event": "stopped",
                    "history_count": 1,
                },
                "runner_timeline": {
                    "count": 2,
                    "last_event": "completed",
                    "last_reason": "idle_exit",
                    "takeover_event_count": 0,
                    "last_takeover_event": None,
                    "last_takeover_reason": None,
                    "recent_anomaly_reason": None,
                    "latest_timestamp": "2026-03-26T09:56:00+00:00",
                },
                "daemon_timeline": {
                    "count": 3,
                    "recovery_event_count": 2,
                    "last_event": "completed",
                    "last_reason": "idle_exit",
                    "last_recovery_event": "recover_requested",
                    "last_recovery_reason": "daemon_stale",
                    "last_recovery_note": "auto_recovery",
                    "recent_anomaly_reason": "daemon_stale",
                    "latest_timestamp": "2026-03-26T09:55:30+00:00",
                },
                "timelines": {
                    "summary_line": "candidate=pending_eval | queue=idle | runner=completed | daemon=completed | runner_anomaly=none | daemon_anomaly=daemon_stale",
                },
                "event_stream": {
                    "count": 3,
                    "severity": "attention",
                    "status": "attention",
                    "attention_needed": True,
                    "attention_reason": "candidate_ready_for_promotion",
                    "attention_source": "candidate",
                    "latest_source": "candidate",
                    "latest_event": "promote_candidate",
                    "latest_reason": "candidate_ready",
                    "next_actions": ["promote_candidate", "archive_candidate"],
                    "dashboard": {
                        "severity": "attention",
                        "status": "attention",
                        "attention_needed": True,
                        "attention_reason": "candidate_ready_for_promotion",
                        "latest_source": "candidate",
                        "latest_event": "promote_candidate",
                        "latest_reason": "candidate_ready",
                    },
                },
            },
            "operations_dashboard": {
                "severity": "attention",
                "status": "attention",
                "attention_needed": True,
                "attention_reason": "candidate_ready_for_promotion",
                "current_focus": "candidate_ready_for_promotion",
                "required_action": "promote_candidate",
                "candidate_stage": "pending_eval",
                "summary_line": "severity=attention | status=attention | attention=yes | focus=candidate_ready_for_promotion | next=promote_candidate",
                "next_actions": ["promote_candidate", "archive_candidate"],
                "inspection_summary_line": "current_focus=candidate_ready_for_promotion | required_action=promote_candidate | next_actions=promote_candidate,archive_candidate",
            },
            "operations_alert_policy": {
                "severity": "attention",
                "required_action": "promote_candidate",
                "current_focus": "candidate_ready_for_promotion",
                "primary_action": "promote_candidate",
                "action_priority": "p2",
                "escalation_mode": "review_soon",
                "requires_human_review": True,
                "auto_remediation_allowed": False,
                "summary_line": "severity=attention | required_action=promote_candidate | priority=p2 | mode=review_soon | focus=candidate_ready_for_promotion",
                "next_actions": ["promote_candidate", "archive_candidate"],
                "inspection_summary_line": "current_focus=candidate_ready_for_promotion | required_action=promote_candidate | next_actions=promote_candidate,archive_candidate",
            },
            "candidate_summary": {
                "candidate_version": "20260326-201",
                "candidate_state": "pending_eval",
                "candidate_needs_promotion": True,
            },
            "train_queue": {
                "count": 0,
                "counts": {},
                "worker_runner": {"active": False, "lock_state": "idle", "stop_requested": False},
            },
        }

        class FakeService:
            def status(self, *, workspace: str | None = None) -> dict[str, object]:
                del workspace
                return payload

        runner = CliRunner()
        original_load_service = cli_main._load_service
        original_resolve_handler = cli_main._resolve_handler
        try:
            cli_main._load_service = lambda *module_names: FakeService()
            cli_main._resolve_handler = lambda service, *names: getattr(service, "status")
            result = runner.invoke(cli_main.app, ["status", "--workspace", "user_default"])
        finally:
            cli_main._load_service = original_load_service
            cli_main._resolve_handler = original_resolve_handler

        clean = strip_ansi(result.stdout)
        self.assertEqual(result.exit_code, 0, msg=result.stdout)
        self.assertIn("[ OPERATIONS CONSOLE ]", clean)
        self.assertIn("[ OPERATIONS DASHBOARD ]", clean)
        self.assertIn("[ OPERATIONS ALERT POLICY ]", clean)
        self.assertIn("attention reason:        candidate_ready_for_promotion", clean)
        self.assertIn("current focus:           candidate_ready_for_promotion", clean)
        self.assertIn("required action:         promote_candidate", clean)
        self.assertIn("[ OPERATIONS EVENT STREAM ]", clean)
        self.assertIn("severity:                attention", clean)
        self.assertIn("attention reason:        candidate_ready_for_promotion", clean)
        self.assertIn("[ RUNNER TIMELINE ]", clean)
        self.assertIn("count:                   2", clean)
        self.assertIn("last event:              completed", clean)
        self.assertIn("[ DAEMON TIMELINE ]", clean)
        self.assertIn("last event:              completed", clean)
        self.assertIn("recovery event count:    2", clean)
        self.assertIn("last recovery event:     recover_requested", clean)
        self.assertIn("next actions:            promote_candidate, archive_candidate", clean)


if __name__ == "__main__":
    unittest.main()
