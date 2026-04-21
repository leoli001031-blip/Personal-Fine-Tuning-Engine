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
from pfe_core.db.sqlite import save_samples
from pfe_core.pipeline import PipelineService
from pfe_core.server_services import InferenceServiceAdapter, PipelineServiceAdapter
from pfe_server.app import ServiceBundle, create_app, smoke_test_request
from pfe_server.auth import ServerSecurityConfig


class OperationsAlertSurfaceTests(unittest.TestCase):
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

    def test_http_status_exposes_alert_health_recovery_surface(self) -> None:
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
        service._append_train_queue_item(
            {
                "job_id": "job-alert-1",
                "state": "awaiting_confirmation",
                "workspace": "user_default",
                "source": "signal_auto_train",
                "adapter_version": "20260326-400",
                "confirmation_required": True,
                "confirmation_reason": "manual_review_required_by_policy",
            },
            workspace="user_default",
        )
        app = self._app(service)

        async def scenario() -> dict[str, object]:
            result = await smoke_test_request(app, path="/pfe/status", method="GET")
            return result["body"]

        body = asyncio.run(scenario())
        self.assertIn("operations_alerts", body)
        self.assertIn("operations_health", body)
        self.assertIn("operations_recovery", body)
        self.assertIn("operations_next_actions", body)
        self.assertIn("operations_dashboard", body)
        self.assertIn("operations_alert_policy", body)
        self.assertTrue(body["operations_health"]["status"] in {"attention", "ok"})
        self.assertEqual(body["operations_health"]["daemon_health_state"], "stale")
        self.assertEqual(body["operations_health"]["daemon_lease_state"], "expired")
        self.assertEqual(body["operations_health"]["daemon_heartbeat_state"], "stale")
        self.assertEqual(body["operations_dashboard"]["highest_priority_action"], "recover_worker_daemon")
        self.assertEqual(body["operations_dashboard"]["active_recovery_hint"], "manual_recover")
        self.assertIn("runtime_stability_summary", body["operations_dashboard"])
        self.assertIn("daemon=stale", body["operations_dashboard"]["runtime_stability_summary_line"])
        self.assertIn("daemon_stale", body["operations_dashboard"]["escalated_reasons"])
        self.assertEqual(body["operations_alert_policy"]["highest_priority_action"], "recover_worker_daemon")
        self.assertEqual(body["operations_alert_policy"]["active_recovery_hint"], "manual_recover")
        self.assertIn("runtime_stability_summary", body["operations_alert_policy"])
        self.assertIn("daemon=stale", body["operations_alert_policy"]["runtime_stability_summary_line"])
        self.assertIn("recover_worker_daemon", body["operations_next_actions"])
        self.assertIn("inspect_daemon_heartbeat", body["operations_next_actions"])
        self.assertIn("review_queue_confirmation", body["operations_next_actions"])
        self.assertIn("heartbeat and lease health", body["operations_event_stream"]["policy"]["operator_guidance"])
        self.assertTrue(any(alert["reason"] == "queue_pending_review" for alert in body["operations_alerts"]))
        self.assertTrue(any(alert["reason"] == "daemon_stale" for alert in body["operations_alerts"]))
        self.assertTrue(any(alert["reason"] == "daemon_heartbeat_stale" for alert in body["operations_alerts"]))
        self.assertTrue(any(alert["reason"] == "daemon_lease_expired" for alert in body["operations_alerts"]))
        self.assertTrue(any(alert["reason"] == "awaiting_confirmation" for alert in body["operations_alerts"]))
        self.assertIn("operations_alerts", body["metadata"])
        self.assertIn("operations_health", body["metadata"])
        self.assertIn("operations_recovery", body["metadata"])
        self.assertIn("operations_next_actions", body["metadata"])

    def test_cli_status_formats_alert_surface_blocks(self) -> None:
        payload = {
            "workspace": "user_default",
            "operations_alerts": [
                {"reason": "daemon_stale", "detail": "daemon lease expired", "severity": "warning"},
                {"reason": "awaiting_confirmation", "detail": "1 queue item awaiting confirmation"},
            ],
            "operations_health": {
                "status": "attention",
                "daemon_lock_state": "stale",
                "daemon_health_state": "stale",
                "daemon_lease_state": "expired",
                "daemon_heartbeat_state": "stale",
                "runner_lock_state": "idle",
                "candidate_state": "pending_eval",
                "queue_state": "awaiting_confirmation",
            },
            "operations_recovery": {
                "daemon_recovery_needed": True,
                "daemon_recovery_reason": "daemon_stale",
                "daemon_recovery_state": "recoverable",
                "daemon_recovery_action": "manual_recover",
            },
            "operations_next_actions": ["recover_worker_daemon", "inspect_daemon_heartbeat", "review_queue_confirmation"],
            "operations_overview": {
                "attention_needed": True,
                "attention_reason": "daemon_stale",
                "summary_line": "candidate-stage=pending_eval | daemon=stale | daemon-health=stale | daemon-heartbeat=stale | daemon-lease=expired | daemon-action=manual_recover",
            },
            "operations_console": {
                "attention_needed": True,
                "attention_reason": "daemon_stale",
                "summary_line": "candidate-stage=pending_eval | daemon-lock=stale | daemon-health=stale",
                "next_actions": ["recover_worker_daemon", "inspect_daemon_heartbeat", "review_queue_confirmation"],
                "candidate": {"current_stage": "pending_eval", "last_candidate_version": "20260326-400"},
                "queue": {"count": 1, "awaiting_confirmation_count": 1},
                "runner": {"active": False, "lock_state": "idle"},
            },
            "candidate_summary": {
                "candidate_version": "20260326-400",
                "candidate_state": "pending_eval",
                "candidate_needs_promotion": True,
            },
            "train_queue": {
                "count": 1,
                "counts": {"awaiting_confirmation": 1},
                "worker_runner": {"active": False, "lock_state": "idle", "stop_requested": False},
            },
        }

        text = _format_status(payload, workspace="user_default")
        clean = strip_ansi(text)
        self.assertIn("[ OPERATIONS ALERTS ]", clean)
        self.assertIn("daemon_stale", clean)
        self.assertIn("[ OPERATIONS HEALTH ]", clean)
        self.assertIn("status:", clean)
        self.assertIn("attention", clean)
        self.assertIn("daemon lock state:", clean)
        self.assertIn("stale", clean)
        self.assertIn("[ OPERATIONS RECOVERY ]", clean)
        self.assertIn("daemon recovery needed:", clean)
        self.assertIn("daemon recovery reason:", clean)
        self.assertIn("daemon recovery action:", clean)
        self.assertIn("manual_recover", clean)
        self.assertIn("[ NEXT ACTIONS ]", clean)
        self.assertIn("recover_worker_daemon", clean)
        self.assertIn("inspect_daemon_heartbeat", clean)
        self.assertIn("review_queue_confirmation", clean)
        self.assertIn("[ OPERATIONS ]", clean)
        self.assertIn("ATTENTION REQUIRED:", clean)
        self.assertIn("daemon_stale", clean)

    def test_status_surfaces_auto_train_policy_gate_alert(self) -> None:
        config = PFEConfig()
        config.trainer.trigger.enabled = True
        config.trainer.trigger.auto_promote = True
        config.trainer.trigger.auto_evaluate = False
        config.save(home=self.pfe_home)

        service = self._service()
        status = service.status()

        self.assertTrue(any(alert["reason"] == "policy_requires_auto_evaluate" for alert in status["operations_overview"]["alerts"]))
        self.assertIn("enable_auto_evaluate", status["operations_console"]["next_actions"])
        self.assertEqual(status["operations_overview"]["attention_reason"], "policy_requires_auto_evaluate")
        self.assertEqual(status["operations_overview"]["auto_train_blocker"]["reason"], "policy_requires_auto_evaluate")
        self.assertEqual(status["operations_console"]["auto_train_blocker"]["reason"], "policy_requires_auto_evaluate")
        self.assertIn("policy-gate=policy_requires_auto_evaluate", status["operations_overview"]["summary_line"])
        self.assertIn("policy-action=enable_auto_evaluate", status["operations_overview"]["summary_line"])
        self.assertEqual(status["operations_alert_policy"]["required_action"], "enable_auto_evaluate")
        self.assertEqual(status["operations_alert_policy"]["secondary_action"], "inspect_auto_train_gate")
        self.assertEqual(status["operations_alert_policy"]["trigger_policy_reason"], "policy_requires_auto_evaluate")
        self.assertEqual(status["operations_alert_policy"]["trigger_policy_action"], "enable_auto_evaluate")
        self.assertEqual(status["operations_alert_policy"]["auto_train_blocker"]["reason"], "policy_requires_auto_evaluate")
        self.assertIn("trigger_policy_gate_summary", status["operations_alert_policy"])
        self.assertIn("eval=", status["operations_alert_policy"]["trigger_policy_gate_summary_line"])
        self.assertIn("enable auto-evaluate", status["operations_alert_policy"]["operator_guidance"])

    def test_status_surfaces_candidate_action_guidance(self) -> None:
        service = self._service()
        service.generate(scenario="life-coach", style="warm", num_samples=6)
        first = service.train_result(method="qlora", epochs=1, train_type="sft")
        AdapterStore(home=self.pfe_home).promote(first.version)
        service.generate(scenario="work-coach", style="direct", num_samples=6)
        service.train_result(method="qlora", epochs=1, train_type="sft")

        status = service.status()

        self.assertEqual(status["operations_overview"]["attention_reason"], "candidate_ready_for_promotion")
        self.assertEqual(status["operations_overview"]["candidate_primary_action"], "promote_candidate")
        self.assertIn("archive_candidate", status["operations_overview"]["candidate_secondary_actions"])
        self.assertIn("promote_candidate", status["operations_console"]["next_actions"])
        self.assertIn("archive_candidate", status["operations_console"]["next_actions"])
        self.assertEqual(status["operations_event_stream"]["highest_priority_action"], "promote_candidate")
        self.assertEqual(status["operations_alert_policy"]["required_action"], "promote_candidate")
        self.assertTrue(status["operations_alert_policy"]["requires_human_review"])
        self.assertEqual(status["operations_alert_policy"]["escalation_mode"], "review_soon")
        self.assertIn("promote the candidate", status["operations_alert_policy"]["operator_guidance"])
        self.assertIn("archive it", status["operations_alert_policy"]["operator_guidance"])

    def test_status_surfaces_failed_eval_candidate_monitor_actions(self) -> None:
        service = self._service()
        service.generate(scenario="life-coach", style="warm", num_samples=6)
        first = service.train_result(method="qlora", epochs=1, train_type="sft")
        store = AdapterStore(home=self.pfe_home)
        store.promote(first.version)
        service.generate(scenario="work-coach", style="direct", num_samples=6)
        second = service.train_result(method="qlora", epochs=1, train_type="sft")
        store.mark_failed_eval(second.version)

        status = service.status()

        self.assertEqual(status["operations_dashboard"]["current_focus"], "candidate_idle")
        self.assertEqual(status["operations_event_stream"]["highest_priority_action"], "archive_candidate")
        self.assertEqual(status["operations_alert_policy"]["required_action"], "archive_candidate")
        self.assertEqual(status["operations_alert_policy"]["secondary_action"], "promote_candidate")
        self.assertEqual(
            status["operations_alert_policy"]["candidate_action_summary"]["primary_action"],
            "archive_candidate",
        )
        self.assertIn("archive_candidate", status["operations_dashboard"]["next_actions"])
        self.assertIn("promote_candidate", status["operations_dashboard"]["next_actions"])

    def test_status_surfaces_archived_candidate_monitor_actions(self) -> None:
        service = self._service()
        service.generate(scenario="life-coach", style="warm", num_samples=6)
        first = service.train_result(method="qlora", epochs=1, train_type="sft")
        store = AdapterStore(home=self.pfe_home)
        store.promote(first.version)
        service.generate(scenario="work-coach", style="direct", num_samples=6)
        second = service.train_result(method="qlora", epochs=1, train_type="sft")
        store.archive(second.version)

        status = service.status()

        self.assertEqual(status["operations_dashboard"]["current_focus"], "candidate_idle")
        self.assertEqual(status["operations_event_stream"]["highest_priority_action"], "inspect_candidate_status")
        self.assertEqual(status["operations_alert_policy"]["required_action"], "inspect_candidate_status")
        self.assertEqual(status["operations_alert_policy"]["secondary_action"], "inspect_candidate_timeline")
        self.assertEqual(
            status["operations_alert_policy"]["candidate_action_summary"]["primary_action"],
            "inspect_candidate_status",
        )
        self.assertIn("inspect_candidate_status", status["operations_dashboard"]["next_actions"])
        self.assertIn("inspect_candidate_timeline", status["operations_dashboard"]["next_actions"])

    def test_operations_alert_policy_prefers_summary_for_monitor_sources_without_fixed_focus(self) -> None:
        candidate_policy = PipelineService._operations_alert_policy(
            severity="info",
            attention_needed=False,
            next_actions=[],
            current_focus="none",
            attention_source="candidate",
            candidate_action_summary={
                "primary_action": "inspect_candidate_status",
                "secondary_actions": ["inspect_candidate_timeline"],
            },
        )
        self.assertEqual(candidate_policy["required_action"], "inspect_candidate_status")
        self.assertEqual(candidate_policy["secondary_action"], "inspect_candidate_timeline")

        queue_policy = PipelineService._operations_alert_policy(
            severity="info",
            attention_needed=False,
            next_actions=[],
            current_focus="none",
            attention_source="queue",
            queue_action_summary={
                "primary_action": "process_next_queue_item",
                "secondary_actions": ["inspect_auto_train_trigger"],
            },
        )
        self.assertEqual(queue_policy["required_action"], "process_next_queue_item")
        self.assertEqual(queue_policy["secondary_action"], "inspect_auto_train_trigger")

        runtime_policy = PipelineService._operations_alert_policy(
            severity="info",
            attention_needed=False,
            next_actions=[],
            current_focus="none",
            attention_source="daemon",
            runtime_action_summary={
                "primary_action": "inspect_runtime_stability",
                "secondary_actions": ["inspect_daemon_status"],
            },
        )
        self.assertEqual(runtime_policy["required_action"], "inspect_runtime_stability")
        self.assertEqual(runtime_policy["secondary_action"], "inspect_daemon_status")

    def test_status_surfaces_queue_waiting_execution_monitor_actions(self) -> None:
        service = self._service()
        service._append_train_queue_item(
            {
                "job_id": "job-monitor-queue-1",
                "state": "queued",
                "workspace": "user_default",
                "source": "manual_queue",
                "priority": 10,
            },
            workspace="user_default",
        )

        status = service.status()

        self.assertEqual(status["operations_dashboard"]["current_focus"], "queue_waiting_execution")
        self.assertEqual(status["operations_event_stream"]["highest_priority_action"], "process_next_queue_item")
        self.assertEqual(status["operations_alert_policy"]["required_action"], "process_next_queue_item")
        self.assertEqual(status["operations_alert_policy"]["secondary_action"], "inspect_auto_train_trigger")
        self.assertEqual(
            status["operations_alert_policy"]["queue_action_summary"]["primary_action"],
            "process_next_queue_item",
        )
        self.assertIn("process_next_queue_item", status["operations_dashboard"]["next_actions"])
        self.assertIn("inspect_auto_train_trigger", status["operations_dashboard"]["next_actions"])

    def test_status_surfaces_auto_train_sample_shortage_alert(self) -> None:
        config = PFEConfig()
        config.trainer.trigger.enabled = True
        config.trainer.trigger.min_new_samples = 50
        config.trainer.trigger.max_interval_days = 0
        config.save(home=self.pfe_home)

        service = self._service()
        service.generate(scenario="life-coach", style="warm", num_samples=8)
        status = service.status()

        self.assertEqual(status["operations_overview"]["attention_reason"], "insufficient_new_signal_samples")
        self.assertTrue(any(alert["reason"] == "insufficient_new_signal_samples" for alert in status["operations_overview"]["alerts"]))
        self.assertIn("collect_more_signal_samples", status["operations_console"]["next_actions"])
        self.assertEqual(
            status["operations_console"]["next_actions"][:2],
            ["collect_more_signal_samples", "inspect_auto_train_policy"],
        )
        self.assertEqual(
            status["operations_dashboard"]["next_actions"][:2],
            ["collect_more_signal_samples", "inspect_auto_train_policy"],
        )
        self.assertEqual(status["operations_event_stream"]["highest_priority_action"], "collect_more_signal_samples")
        self.assertEqual(status["operations_alert_policy"]["required_action"], "collect_more_signal_samples")
        self.assertEqual(status["operations_alert_policy"]["secondary_action"], "inspect_auto_train_policy")
        self.assertIn("not enough new signal samples", status["operations_alert_policy"]["operator_guidance"])
        self.assertIn("trigger_threshold_summary", status["operations_alert_policy"])
        self.assertIn("samples=", status["operations_alert_policy"]["trigger_threshold_summary_line"])
        self.assertIn("gate=samples=", status["operations_alert_policy"]["summary_line"])

    def test_status_surfaces_auto_train_dpo_pair_shortage_alert(self) -> None:
        config = PFEConfig()
        config.trainer.train_type = "dpo"
        config.trainer.trigger.enabled = True
        config.trainer.trigger.min_new_samples = 2
        config.trainer.trigger.max_interval_days = 0
        config.save(home=self.pfe_home)

        save_samples(
            [
                {
                    "sample_id": "sft-1",
                    "sample_type": "sft",
                    "instruction": "check in",
                    "chosen": "warm answer 1",
                    "source": "signal",
                    "metadata": {"dataset_split": "train"},
                },
                {
                    "sample_id": "sft-2",
                    "sample_type": "sft",
                    "instruction": "check in again",
                    "chosen": "warm answer 2",
                    "source": "signal",
                    "metadata": {"dataset_split": "train"},
                },
                {
                    "sample_id": "sft-val",
                    "sample_type": "sft",
                    "instruction": "holdout",
                    "chosen": "warm answer val",
                    "source": "signal",
                    "metadata": {"dataset_split": "val"},
                },
            ],
            home=self.pfe_home,
        )

        service = self._service()
        status = service.status()

        self.assertEqual(status["auto_train_trigger"]["train_type"], "dpo")
        self.assertEqual(status["auto_train_trigger"]["blocked_primary_reason"], "insufficient_dpo_preference_pairs")
        self.assertEqual(status["auto_train_trigger"]["blocked_primary_action"], "collect_more_dpo_preference_pairs")
        self.assertIn("dpo_pairs=0/2", status["auto_train_trigger"]["threshold_summary"]["summary_line"])
        self.assertIn("DPO preference pairs", status["auto_train_trigger"]["blocked_summary"])
        self.assertEqual(status["operations_overview"]["attention_reason"], "insufficient_dpo_preference_pairs")
        self.assertEqual(status["operations_overview"]["auto_train_blocker"]["reason"], "insufficient_dpo_preference_pairs")
        self.assertEqual(status["operations_alert_policy"]["required_action"], "collect_more_dpo_preference_pairs")
        self.assertEqual(status["operations_alert_policy"]["secondary_action"], "inspect_auto_train_policy")
        self.assertIn("collect more DPO preference pairs", status["operations_alert_policy"]["operator_guidance"])
        self.assertIn("dpo_pairs=0/2", status["operations_alert_policy"]["trigger_threshold_summary_line"])
        self.assertIn("gate=dpo_pairs=0/2", status["operations_alert_policy"]["summary_line"])

    def test_status_surfaces_auto_train_holdout_gate_alert(self) -> None:
        config = PFEConfig()
        config.trainer.trigger.enabled = True
        config.trainer.trigger.min_new_samples = 0
        config.trainer.trigger.max_interval_days = 0
        config.save(home=self.pfe_home)

        save_samples(
            [
                {
                    "sample_id": "signal-train-1",
                    "sample_type": "sft",
                    "instruction": "check in",
                    "chosen": "warm answer",
                    "source": "signal",
                    "metadata": {"dataset_split": "train"},
                }
            ],
            home=self.pfe_home,
        )

        service = self._service()
        status = service.status()

        self.assertEqual(status["operations_overview"]["attention_reason"], "holdout_not_ready")
        self.assertEqual(status["operations_alert_policy"]["required_action"], "collect_holdout_samples")
        self.assertEqual(status["operations_alert_policy"]["secondary_action"], "inspect_auto_train_policy")
        self.assertIn("collect holdout samples", status["operations_alert_policy"]["operator_guidance"])
        self.assertEqual(
            status["operations_console"]["next_actions"][:2],
            ["collect_holdout_samples", "inspect_auto_train_policy"],
        )

    def test_status_surfaces_auto_train_cooldown_gate_alert(self) -> None:
        config = PFEConfig()
        config.trainer.trigger.enabled = True
        config.trainer.trigger.min_new_samples = 0
        config.trainer.trigger.max_interval_days = 7
        config.save(home=self.pfe_home)

        service = self._service()
        service.generate(scenario="life-coach", style="warm", num_samples=8)
        trained = service.train_result(method="qlora", epochs=1, train_type="sft")
        AdapterStore(home=self.pfe_home).promote(trained.version)
        status = service.status()

        self.assertEqual(status["operations_overview"]["attention_reason"], "cooldown_active")
        self.assertEqual(status["operations_alert_policy"]["required_action"], "wait_for_retrain_interval")
        self.assertEqual(status["operations_alert_policy"]["secondary_action"], "inspect_auto_train_trigger")
        self.assertIn("retrain interval gate", status["operations_alert_policy"]["operator_guidance"])
        self.assertEqual(
            status["operations_console"]["next_actions"][:2],
            ["wait_for_retrain_interval", "inspect_auto_train_trigger"],
        )

    def test_status_surfaces_auto_train_failure_backoff_alert(self) -> None:
        config = PFEConfig()
        config.trainer.trigger.enabled = True
        config.trainer.trigger.min_new_samples = 0
        config.trainer.trigger.max_interval_days = 0
        config.trainer.trigger.failure_backoff_minutes = 30
        config.save(home=self.pfe_home)

        service = self._service()
        service.generate(scenario="life-coach", style="warm", num_samples=8)
        service._persist_auto_trigger_state(
            {
                "workspace": "user_default",
                "last_failure_at": "2026-03-29T12:00:00+00:00",
                "failure_backoff_until": "2999-01-01T00:00:00+00:00",
                "consecutive_failures": 1,
            },
            workspace="user_default",
        )

        status = service.status()

        self.assertEqual(status["operations_overview"]["attention_reason"], "failure_backoff_active")
        self.assertEqual(status["operations_alert_policy"]["required_action"], "wait_for_failure_backoff")
        self.assertEqual(status["operations_alert_policy"]["secondary_action"], "inspect_auto_train_trigger")
        self.assertIn("wait for failure backoff", status["operations_alert_policy"]["operator_guidance"])
        self.assertEqual(
            status["operations_console"]["next_actions"][:2],
            ["wait_for_failure_backoff", "inspect_auto_train_trigger"],
        )

    def test_status_surfaces_runner_stale_lock_guidance(self) -> None:
        service = self._service()
        stale_time = "2026-03-20T10:00:00+00:00"
        service._persist_train_queue_worker_state(
            {
                "active": True,
                "stop_requested": False,
                "pid": 99999,
                "started_at": stale_time,
                "last_heartbeat_at": stale_time,
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
        self.assertEqual(status["operations_alert_policy"]["required_action"], "inspect_worker_stale_lock")
        self.assertEqual(status["operations_alert_policy"]["secondary_action"], "wait_for_runner_shutdown")
        self.assertEqual(status["operations_alert_policy"]["action_priority"], "p1")
        self.assertTrue(status["operations_alert_policy"]["requires_human_review"])
        self.assertIn("stale runner lock", status["operations_alert_policy"]["operator_guidance"])
        self.assertIn("inspect_worker_stale_lock", status["operations_console"]["next_actions"])

    def test_status_surfaces_runner_active_monitor_actions(self) -> None:
        service = self._service()
        service._persist_train_queue_worker_state(
            {
                "active": True,
                "stop_requested": False,
                "pid": 45678,
                "started_at": "2026-03-20T10:00:00+00:00",
                "last_heartbeat_at": datetime.now(timezone.utc).isoformat(),
                "last_completed_at": None,
                "loop_cycles": 3,
                "processed_count": 1,
                "failed_count": 0,
                "stopped_reason": None,
                "last_action": "run_worker_runner",
                "max_seconds": 30.0,
                "idle_sleep_seconds": 1.0,
            },
            workspace="user_default",
        )

        status = service.status()
        self.assertEqual(status["operations_dashboard"]["current_focus"], "runner_active")
        self.assertEqual(status["operations_event_stream"]["highest_priority_action"], "inspect_runtime_stability")
        self.assertEqual(status["operations_alert_policy"]["required_action"], "inspect_runtime_stability")
        self.assertEqual(status["operations_alert_policy"]["secondary_action"], "inspect_worker_runner_history")
        self.assertEqual(
            status["operations_alert_policy"]["runtime_action_summary"]["primary_action"],
            "inspect_runtime_stability",
        )
        self.assertIn("inspect_runtime_stability", status["operations_console"]["next_actions"])
        self.assertIn("inspect_worker_runner_history", status["operations_console"]["next_actions"])

    def test_status_surfaces_daemon_active_monitor_actions(self) -> None:
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
            },
            workspace="user_default",
        )

        status = service.status()

        self.assertEqual(status["operations_dashboard"]["current_focus"], "daemon_active")
        self.assertEqual(status["operations_event_stream"]["highest_priority_action"], "inspect_runtime_stability")
        self.assertEqual(status["operations_alert_policy"]["required_action"], "inspect_runtime_stability")
        self.assertEqual(status["operations_alert_policy"]["secondary_action"], "inspect_daemon_status")
        self.assertEqual(
            status["operations_alert_policy"]["runtime_action_summary"]["primary_action"],
            "inspect_runtime_stability",
        )
        self.assertIn("inspect_runtime_stability", status["operations_dashboard"]["next_actions"])
        self.assertIn("inspect_daemon_status", status["operations_dashboard"]["next_actions"])

    def test_status_surfaces_daemon_restart_backoff_guidance(self) -> None:
        service = self._service()
        next_restart_after = (datetime.now(timezone.utc) + timedelta(seconds=30)).isoformat()
        service._persist_train_queue_daemon_state(
            {
                "workspace": "user_default",
                "desired_state": "running",
                "observed_state": "stopped",
                "command_status": "idle",
                "active": False,
                "pid": None,
                "started_at": None,
                "last_heartbeat_at": None,
                "auto_restart_enabled": True,
                "auto_recover_enabled": False,
                "restart_attempts": 1,
                "max_restart_attempts": 3,
                "restart_backoff_seconds": 30.0,
                "next_restart_after": next_restart_after,
            },
            workspace="user_default",
        )

        status = service.status()
        self.assertEqual(status["operations_alert_policy"]["required_action"], "inspect_daemon_restart_policy")
        self.assertEqual(status["operations_alert_policy"]["secondary_action"], "inspect_daemon_heartbeat")
        self.assertEqual(status["operations_alert_policy"]["action_priority"], "p1")
        self.assertTrue(status["operations_alert_policy"]["requires_human_review"])
        self.assertIn("restart attempts and backoff", status["operations_alert_policy"]["operator_guidance"])
        self.assertIn("inspect_daemon_restart_policy", status["operations_console"]["next_actions"])


if __name__ == "__main__":
    unittest.main()
