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
from pfe_cli.console_app import (
    _compact_text,
    _chat_help_panel,
    _event_stream_panel,
    _footer_digest,
    _operations_panel,
    _payload_command_guidance,
    _prompt_action_guidance,
    _prompt_context_focus,
    _prompt_panel,
    _prompt_trigger_category,
    build_console_renderable,
    _prompt_ctx_digest,
    _prompt_feedback_digest,
    _prompt_hint_digest,
    _prompt_mode_help,
    _prompt_placeholder,
    _prompt_target_hint,
)
from rich.console import Console


class CLIConsoleSurfaceTests(unittest.TestCase):
    def test_console_command_renders_core_panels(self) -> None:
        payload = {
            "workspace": "user_default",
            "strict_local": True,
            "inference_backend": "llama_cpp",
            "latest_adapter": {"version": "20260325-001", "state": "promoted"},
            "operations_overview": {
                "summary_line": "auto=disabled | candidate=none | queue=0 | runner=idle | daemon-health=stopped",
                "trigger_policy_summary": "mode=deferred | entry=awaiting_confirmation | review=manual_review | eval=auto | promote=auto | stop=confirmation",
                "trigger_threshold_summary": {
                    "min_new_samples": 50,
                    "eligible_signal_train_samples": 2,
                    "holdout_ready": True,
                    "interval_elapsed": False,
                    "cooldown_elapsed": False,
                    "failure_backoff_elapsed": True,
                    "summary_line": "samples=2/50 | holdout=ready | interval=1.0/7d | cooldown=10.0m | backoff=ok",
                },
                "trigger_blocked_reason": "queue_pending_review",
                "trigger_blocked_action": "review_queue_confirmation",
                "auto_train_policy": {
                    "queue_entry_mode": "awaiting_confirmation",
                    "evaluation_mode": "auto_evaluate",
                    "promotion_mode": "auto_promote",
                    "stop_stage": "confirmation",
                },
            },
            "operations_console": {
                "trigger_policy_gate_summary": {
                    "eval_num_samples": 8,
                    "auto_evaluate_enabled": True,
                    "auto_promote_requested": True,
                    "promotion_requirement": "deploy_eval_recommendation",
                    "summary_line": "eval=8:on | promote=on | promote_ready=yes | req=deploy_eval_recommendation",
                },
                "trigger_threshold_summary": {
                    "min_new_samples": 50,
                    "eligible_signal_train_samples": 2,
                    "holdout_ready": True,
                    "interval_elapsed": False,
                    "cooldown_elapsed": False,
                    "failure_backoff_elapsed": True,
                    "summary_line": "samples=2/50 | holdout=ready | interval=1.0/7d | cooldown=10.0m | backoff=ok",
                },
                "trigger_blocked_reason": "queue_pending_review",
                "trigger_blocked_action": "review_queue_confirmation",
                "trigger_blocked_category": "queue",
                "auto_train_policy": {
                    "queue_entry_mode": "awaiting_confirmation",
                    "evaluation_mode": "auto_evaluate",
                    "promotion_mode": "auto_promote",
                    "stop_stage": "confirmation",
                },
                "queue_review_policy": {
                    "review_mode": "manual_review",
                    "queue_entry_mode": "awaiting_confirmation",
                    "next_action": "review_queue_confirmation",
                },
                "runtime_stability_summary": {
                    "runner_lock_state": "idle",
                    "runner_active": False,
                    "runner_stop_requested": False,
                    "daemon_health_state": "stale",
                    "daemon_heartbeat_state": "stale",
                    "daemon_lease_state": "expired",
                    "daemon_restart_policy_state": "ready",
                    "daemon_recovery_action": "manual_recover",
                    "summary_line": "runner=idle | stop=no | daemon=stale | hb=stale | lease=expired | restart=ready | recover=manual_recover",
                },
            },
            "operations_dashboard": {
                "severity": "warning",
                "status": "attention",
                "current_focus": "daemon_stale",
            },
            "operations_alert_policy": {
                "required_action": "recover_worker_daemon",
                "action_priority": "p0",
                "escalation_mode": "immediate",
                "requires_immediate_action": True,
                "requires_human_review": False,
                "auto_remediation_allowed": True,
                "operator_guidance": "recover the worker daemon before continuing queue execution",
            },
            "operations_event_stream": {
                "severity": "warning",
                "status": "attention",
                "attention_needed": True,
                "latest_source": "daemon",
                "alert_count": 1,
                "dashboard": {"current_focus": "daemon_stale"},
                "items": [
                    {
                        "source": "daemon",
                        "event": "alert",
                        "severity": "warning",
                        "reason": "daemon_stale",
                    }
                ],
            },
            "train_queue": {"count": 0},
        }

        class FakeService:
            def status(self, workspace: str | None = None):
                del workspace
                return payload

        runner = CliRunner()
        original_load_service = cli_main._load_service
        original_resolve_handler = cli_main._resolve_handler
        try:
            cli_main._load_service = lambda *module_names: FakeService()
            cli_main._resolve_handler = lambda service, *names: getattr(service, "status")
            result = runner.invoke(cli_main.app, ["console", "--workspace", "user_default"])
        finally:
            cli_main._load_service = original_load_service
            cli_main._resolve_handler = original_resolve_handler

        self.assertEqual(result.exit_code, 0, msg=result.stdout)
        self.assertIn("PFE Console", result.stdout)
        self.assertIn("ws=user_default", result.stdout)
        self.assertIn("md=strict_local", result.stdout)
        self.assertIn("infer=llama_cpp", result.stdout)
        self.assertIn("Operations", result.stdout)
        self.assertIn("Operations Sidebar", result.stdout)
        self.assertIn("SNAPSHOT", result.stdout)
        self.assertIn("OPERATIONS  LIVE !", result.stdout)
        self.assertIn("> chat>", result.stdout)
        self.assertIn("m=local", result.stdout)
        self.assertIn("a=latest", result.stdout)
        self.assertIn("Operations Sidebar  LIVE", result.stdout)
        self.assertIn("LIVE !", result.stdout)
        self.assertIn("IDLE", result.stdout)
        self.assertIn("LOCKED", result.stdout)
        self.assertIn("daemon_stale", result.stdout)
        self.assertIn("Pol", result.stdout)
        self.assertIn("PGate", result.stdout)
        self.assertIn("e=8:on", result.stdout)
        self.assertIn("Gate", result.stdout)
        self.assertIn("s=2/50", result.stdout)
        self.assertIn("h=yes", result.stdout)
        self.assertIn("Trig", result.stdout)
        self.assertIn("queue_pending_r...", result.stdout)
        self.assertIn("queue_pending_r...", result.stdout)
        self.assertIn("review_queue_confir...", result.stdout)
        self.assertIn("awaiting_confirmation", result.stdout)
        self.assertIn("QRev", result.stdout)
        self.assertIn("manual_review", result.stdout)
        self.assertIn("f=daemon_stale", result.stdout)
        self.assertIn("a=recover_worker_daemon", result.stdout)
        self.assertIn("DAEMON STALE", result.stdout)
        self.assertIn("RECOVER WORKER DAEMON", result.stdout)

    def test_console_help_exposes_watch_options(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli_main.app, ["console", "--help"])

        self.assertEqual(result.exit_code, 0, msg=result.stdout)
        self.assertIn("Render a Rich-based PFE operations console with optional prompt mode.", result.stdout)
        self.assertIn("--interactive", result.stdout)
        self.assertIn("--model", result.stdout)
        self.assertIn("--adapter", result.stdout)
        self.assertIn("--real-local", result.stdout)
        self.assertIn("--watch", result.stdout)
        self.assertIn("--refresh-seconds", result.stdout)
        self.assertIn("--cycles", result.stdout)

    def test_console_interactive_supports_help_and_quit(self) -> None:
        payload = {
            "workspace": "user_default",
            "strict_local": True,
            "inference_backend": "llama_cpp",
            "latest_adapter": {"version": "20260325-001", "state": "promoted"},
            "operations_overview": {"summary_line": "auto=disabled | queue=0 | runner=idle | daemon-health=stopped"},
            "operations_dashboard": {"severity": "info", "status": "healthy", "current_focus": "none"},
            "operations_alert_policy": {
                "required_action": "observe_and_monitor",
                "action_priority": "p2",
                "escalation_mode": "monitor",
            },
            "operations_event_stream": {
                "severity": "info",
                "status": "healthy",
                "attention_needed": False,
                "latest_source": "queue",
                "alert_count": 0,
                "dashboard": {"current_focus": "none"},
                "items": [],
            },
            "train_queue": {"count": 0},
        }

        class FakeService:
            def status(self, workspace: str | None = None):
                del workspace
                return payload

            def chat_completion(self, **kwargs):
                del kwargs
                return {
                    "choices": [
                        {
                            "message": {
                                "content": "ok",
                            }
                        }
                    ]
                }

            def candidate_timeline(self, workspace: str | None = None, limit: int = 5):
                del workspace, limit
                return {"current_stage": "idle", "items": []}

            def train_queue_daemon_status(self, workspace: str | None = None):
                del workspace
                return {"workspace": "user_default", "command_status": "idle"}

        runner = CliRunner()
        original_load_service = cli_main._load_service
        original_resolve_handler = cli_main._resolve_handler
        try:
            cli_main._load_service = lambda *module_names: FakeService()
            cli_main._resolve_handler = lambda service, *names: next((getattr(service, name) for name in names if hasattr(service, name)), None)
            result = runner.invoke(cli_main.app, ["console", "--workspace", "user_default", "--interactive"], input="/help\n/quit\n")
        finally:
            cli_main._load_service = original_load_service
            cli_main._resolve_handler = original_resolve_handler

        self.assertEqual(result.exit_code, 0, msg=result.stdout)
        self.assertIn("slash commands:", result.stdout)
        self.assertIn("/help - show this help", result.stdout)
        self.assertIn("/status compact", result.stdout)
        help_text = cli_main._console_help_text()
        self.assertIn("/sum - alias for /status compact", help_text)
        self.assertIn("/os - alias for /ops", help_text)
        self.assertIn("/ops - show", result.stdout)
        self.assertIn("/ops dashboard", help_text)
        self.assertIn("/dash - alias for /ops dashboard", help_text)
        self.assertIn("/ops dash - alias for /ops dashboard", help_text)
        self.assertIn("/alerts - alias for /ops alerts", help_text)
        self.assertIn("/policy - alias for /ops policy", help_text)
        self.assertIn("/ops pol - alias for /ops policy", help_text)
        self.assertIn("/trigger - show the auto-train trigger summary", help_text)
        self.assertIn("/trig - alias for /trigger summary", help_text)
        self.assertIn("/gate - show the auto-train gate summary", help_text)
        self.assertIn("/runtime - show the runtime stability summary", help_text)
        self.assertIn("/rt - alias for /runtime summary", help_text)
        self.assertIn("/approve [note] - approve the next queued review item", help_text)
        self.assertIn("/reject [note] - reject the next queued review item", help_text)
        self.assertIn("/retry - retry the current auto-train trigger evaluation", help_text)
        self.assertIn("/reset - reset the auto-train trigger state", help_text)
        self.assertIn("/recover daemon - recover the worker daemon", help_text)
        self.assertIn("/restart daemon - restart the worker daemon", help_text)
        self.assertIn("/event - alias for /event stream", help_text)
        self.assertIn("/cand - alias for /candidate summary", help_text)
        self.assertIn("/cand sum - alias for /candidate summary", help_text)
        self.assertIn("/cand tl - alias for /candidate timeline", help_text)
        self.assertIn("/cand hist - alias for /candidate history", help_text)
        self.assertIn("/qs - alias for /queue summary", help_text)
        self.assertIn("/queue sum - alias for /queue summary", help_text)
        self.assertIn("/queue hist - alias for /queue history", help_text)
        self.assertIn("/rs - alias for /runner summary", help_text)
        self.assertIn("/runner sum - alias for /runner summary", help_text)
        self.assertIn("/runner tl - alias for /runner timeline", help_text)
        self.assertIn("/runner hist - alias for /runner history", help_text)
        self.assertIn("/ds - alias for /daemon summary", help_text)
        self.assertIn("/daemon sum - alias for /daemon summary", help_text)
        self.assertIn("/daemon tl - alias for /daemon timeline", help_text)
        self.assertIn("/daemon hist - alias for /daemon history", help_text)
        self.assertIn("/queue - show the train queue summary", help_text)
        self.assertIn("/queue summary - show the train queue summary", help_text)
        self.assertIn("/runner - show the worker runner summary", help_text)
        self.assertIn("/runner summary - show the worker runner summary", help_text)
        self.assertIn("/runner timeline - show the worker runner timeline", help_text)
        self.assertIn("/daemon timeline - show the worker daemon timeline", help_text)
        self.assertIn("Exiting PFE Console.", result.stdout)

    def test_console_interactive_supports_mode_switching(self) -> None:
        payload = {
            "workspace": "user_default",
            "strict_local": True,
            "inference_backend": "llama_cpp",
            "latest_adapter": {"version": "20260325-001", "state": "promoted"},
            "operations_overview": {"summary_line": "auto=disabled | queue=0 | runner=idle | daemon-health=stopped"},
            "operations_dashboard": {"severity": "info", "status": "healthy", "current_focus": "none"},
            "operations_alert_policy": {
                "required_action": "observe_and_monitor",
                "action_priority": "p2",
                "escalation_mode": "monitor",
            },
            "operations_event_stream": {
                "severity": "info",
                "status": "healthy",
                "attention_needed": False,
                "latest_source": "queue",
                "alert_count": 0,
                "dashboard": {"current_focus": "none"},
                "items": [],
            },
            "train_queue": {"count": 0},
        }

        class FakeService:
            def status(self, workspace: str | None = None):
                del workspace
                return payload

            def chat_completion(self, **kwargs):
                del kwargs
                return {"choices": [{"message": {"content": "ok"}}]}

            def candidate_timeline(self, workspace: str | None = None, limit: int = 5):
                del workspace, limit
                return {"current_stage": "idle", "items": []}

            def train_queue_daemon_status(self, workspace: str | None = None):
                del workspace
                return {"workspace": "user_default", "command_status": "idle"}

        runner = CliRunner()
        original_load_service = cli_main._load_service
        original_resolve_handler = cli_main._resolve_handler
        try:
            cli_main._load_service = lambda *module_names: FakeService()
            cli_main._resolve_handler = lambda service, *names: next((getattr(service, name) for name in names if hasattr(service, name)), None)
            result = runner.invoke(
                cli_main.app,
                ["console", "--workspace", "user_default", "--interactive"],
                input="/cmd\nstatus\n/chat\n/quit\n",
            )
        finally:
            cli_main._load_service = original_load_service
            cli_main._resolve_handler = original_resolve_handler

        self.assertEqual(result.exit_code, 0, msg=result.stdout)
        self.assertIn("cmd>:", result.stdout)
        self.assertIn("chat>:", result.stdout)
        self.assertIn("MATRIX CONSOLE MODE", result.stdout)
        self.assertIn("EXEC", result.stdout)

    def test_console_interactive_shows_generating_feedback(self) -> None:
        payload = {
            "workspace": "user_default",
            "strict_local": True,
            "inference_backend": "llama_cpp",
            "latest_adapter": {"version": "20260325-001", "state": "promoted"},
            "operations_overview": {"summary_line": "auto=disabled | queue=0 | runner=idle | daemon-health=stopped"},
            "operations_dashboard": {"severity": "info", "status": "healthy", "current_focus": "none"},
            "operations_alert_policy": {
                "required_action": "observe_and_monitor",
                "action_priority": "p2",
                "escalation_mode": "monitor",
            },
            "operations_event_stream": {
                "severity": "info",
                "status": "healthy",
                "attention_needed": False,
                "latest_source": "queue",
                "alert_count": 0,
                "dashboard": {"current_focus": "none"},
                "items": [],
            },
            "train_queue": {"count": 0},
        }

        class FakeService:
            def status(self, workspace: str | None = None):
                del workspace
                return payload

            def chat_completion(self, **kwargs):
                del kwargs
                return {"choices": [{"message": {"content": "hello back"}}]}

        runner = CliRunner()
        original_load_service = cli_main._load_service
        original_resolve_handler = cli_main._resolve_handler
        try:
            cli_main._load_service = lambda *module_names: FakeService()
            cli_main._resolve_handler = lambda service, *names: next((getattr(service, name) for name in names if hasattr(service, name)), None)
            result = runner.invoke(
                cli_main.app,
                ["console", "--workspace", "user_default", "--interactive"],
                input="hello\n/quit\n",
            )
        finally:
            cli_main._load_service = original_load_service
            cli_main._resolve_handler = original_resolve_handler

        self.assertEqual(result.exit_code, 0, msg=result.stdout)
        self.assertIn("user> hello", result.stdout)
        self.assertIn("hello back", result.stdout)
        self.assertIn("EXEC", result.stdout)

    def test_console_interactive_supports_runtime_setting_commands(self) -> None:
        payload = {
            "workspace": "user_default",
            "strict_local": True,
            "inference_backend": "llama_cpp",
            "latest_adapter": {"version": "20260325-001", "state": "promoted"},
            "operations_overview": {"summary_line": "auto=disabled | queue=0 | runner=idle | daemon-health=stopped"},
            "operations_dashboard": {"severity": "info", "status": "healthy", "current_focus": "none"},
            "operations_alert_policy": {
                "required_action": "observe_and_monitor",
                "action_priority": "p2",
                "escalation_mode": "monitor",
            },
            "operations_event_stream": {
                "severity": "info",
                "status": "healthy",
                "attention_needed": False,
                "latest_source": "queue",
                "alert_count": 0,
                "dashboard": {"current_focus": "none"},
                "items": [],
            },
            "train_queue": {"count": 0},
        }

        class FakeService:
            def status(self, workspace: str | None = None):
                return {**payload, "workspace": workspace or "user_default"}

            def candidate_timeline(self, workspace: str | None = None, limit: int = 5):
                del limit
                return {"current_stage": "idle", "items": [], "workspace": workspace or "user_default"}

            def train_queue_daemon_status(self, workspace: str | None = None):
                return {"workspace": workspace or "user_default", "command_status": "idle"}

        runner = CliRunner()
        original_load_service = cli_main._load_service
        original_resolve_handler = cli_main._resolve_handler
        try:
            cli_main._load_service = lambda *module_names: FakeService()
            cli_main._resolve_handler = lambda service, *names: next((getattr(service, name) for name in names if hasattr(service, name)), None)
            result = runner.invoke(
                cli_main.app,
                ["console", "--workspace", "user_default", "--interactive"],
                input="/cmd\n/workspace qa-workspace\n/model qwen-test\n/adapter 20260325-001\n/temperature 0.2\n/max-tokens 144\n/real-local on\n/refresh 1.5\n/settings\n/quit\n",
            )
        finally:
            cli_main._load_service = original_load_service
            cli_main._resolve_handler = original_resolve_handler

        self.assertEqual(result.exit_code, 0, msg=result.stdout)
        self.assertIn("workspace set", result.stdout)
        self.assertIn("qa-workspace", result.stdout)
        self.assertIn("model set", result.stdout)
        self.assertIn("qwen-test", result.stdout)
        self.assertIn("adapter set", result.stdout)
        self.assertIn("20260325-001", result.stdout)
        self.assertIn("temperature", result.stdout)
        self.assertIn("0.20", result.stdout)
        self.assertIn("max tokens set", result.stdout)
        self.assertIn("144", result.stdout)
        self.assertIn("real-local", result.stdout)
        self.assertIn("PFE Console", result.stdout)
        self.assertIn("ws=qa-workspace", result.stdout)
        self.assertIn("REAL-LOCAL", result.stdout)
        self.assertIn("COMMAND", result.stdout)

    def test_console_command_output_supports_compact_aliases(self) -> None:
        payload = {
            "workspace": "user_default",
            "strict_local": True,
            "operations_overview": {"summary_line": "auto=disabled | queue=0"},
            "operations_dashboard": {"current_focus": "candidate_ready_for_promotion", "severity": "warning"},
            "operations_alert_policy": {
                "required_action": "review_candidate",
                "action_priority": "p1",
                "escalation_mode": "review_soon",
            },
            "operations_health": {"status": "healthy"},
            "operations_event_stream": {
                "count": 1,
                "severity": "warning",
                "status": "attention",
                "latest_source": "candidate",
                "items": [{"source": "candidate", "event": "ready", "severity": "warning"}],
            },
            "candidate_summary": {"candidate_version": "20260325-001", "candidate_state": "pending_eval"},
            "operations_console": {"summary_line": "console=ok"},
        }

        class FakeService:
            def candidate_timeline(self, workspace: str | None = None, limit: int = 5):
                del workspace, limit
                return {"current_stage": "pending_eval", "transition_count": 2}

        for command, expected_action, expected_text in (
            ("sum", "status-compact", "PFE status compact"),
            ("os", "ops", "operations console digest:"),
            ("dash", "ops-dashboard", "operations dashboard:"),
            ("ops dash", "ops-dashboard", "operations dashboard:"),
            ("alerts", "ops-alerts", "operations alerts:"),
            ("policy", "ops-policy", "operations alert policy:"),
            ("ops pol", "ops-policy", "operations alert policy:"),
            ("event", "event-stream", "operations event stream:"),
            ("cand", "candidate-summary", "PFE candidate summary"),
            ("cand sum", "candidate-summary", "PFE candidate summary"),
        ):
            text, action, updates = cli_main._console_command_output(
                command,
                payload=payload,
                workspace="user_default",
                service=FakeService(),
                current_workspace="user_default",
                mode="command",
                model="local",
                adapter="latest",
                temperature=0.7,
                max_tokens=None,
                real_local=False,
                refresh_seconds=2.0,
            )
            self.assertEqual(action, expected_action)
            self.assertIsNone(updates)
            self.assertIn(expected_text, text or "")

    def test_console_command_output_supports_history_and_timeline_aliases(self) -> None:
        payload = {
            "workspace": "user_default",
            "operations_dashboard": {"current_focus": "candidate_ready_for_promotion", "severity": "warning"},
            "candidate_timeline": {"current_stage": "pending_eval", "transition_count": 2},
            "runner_timeline": {"count": 2, "last_event": "started", "takeover_event_count": 0, "current_lock_state": "idle"},
            "daemon_timeline": {"count": 1, "recovery_event_count": 0, "last_event": "started", "last_reason": "manual"},
            "train_queue": {"count": 1},
        }

        class FakeService:
            def candidate_timeline(self, workspace: str | None = None, limit: int = 5):
                del workspace, limit
                return {"current_stage": "pending_eval", "transition_count": 2, "items": []}

            def candidate_history(self, workspace: str | None = None, limit: int = 10):
                del workspace, limit
                return {"count": 1, "items": [{"action": "promote_candidate"}]}

            def train_queue_history(self, workspace: str | None = None, limit: int = 10):
                del workspace, limit
                return {"count": 1, "history": [{"event": "completed", "reason": "success"}]}

            def train_queue_worker_runner_history(self, workspace: str | None = None, limit: int = 10):
                del workspace, limit
                return {"count": 1, "items": [{"event": "started", "reason": "manual"}]}

            def train_queue_daemon_history(self, workspace: str | None = None, limit: int = 10):
                del workspace, limit
                return {"count": 1, "items": [{"event": "started", "reason": "manual"}]}

        for command, expected_action, expected_text in (
            ("cand tl", "candidate", "PFE candidate timeline"),
            ("cand hist", "candidate-history", "PFE candidate history"),
            ("qs", "queue-summary", "PFE train queue summary"),
            ("queue sum", "queue-summary", "PFE train queue summary"),
            ("queue hist", "queue-history", "PFE train queue history"),
            ("rs", "runner-summary", "PFE worker runner summary"),
            ("runner sum", "runner-summary", "PFE worker runner summary"),
            ("runner tl", "runner-timeline", "runner timeline"),
            ("runner hist", "runner-history", "worker runner history"),
            ("ds", "daemon-summary", "PFE worker daemon summary"),
            ("daemon sum", "daemon-summary", "PFE worker daemon summary"),
            ("daemon tl", "daemon-timeline", "daemon timeline"),
            ("daemon hist", "daemon-history", "worker daemon history"),
        ):
            text, action, updates = cli_main._console_command_output(
                command,
                payload=payload,
                workspace="user_default",
                service=FakeService(),
                current_workspace="user_default",
                mode="command",
                model="local",
                adapter="latest",
                temperature=0.7,
                max_tokens=None,
                real_local=False,
                refresh_seconds=2.0,
            )
            self.assertEqual(action, expected_action)
            self.assertIsNone(updates)
            self.assertIn(expected_text, text or "")

    def test_console_shortcut_hint_tracks_mode_and_focus(self) -> None:
        self.assertEqual(cli_main._console_shortcut_hint("chat", {}), "Enter,/help,^C")
        self.assertEqual(
            cli_main._console_shortcut_hint(
                "chat",
                {"operations_dashboard": {"attention_needed": True}},
            ),
            "Enter,/do,/see",
        )
        self.assertEqual(
            cli_main._console_shortcut_hint(
                "command",
                {"operations_dashboard": {"current_focus": "daemon_stale"}},
            ),
            "/do,/see,/daemon,/chat",
        )
        self.assertEqual(
            cli_main._console_shortcut_hint(
                "command",
                {"operations_dashboard": {"current_focus": "queue_backlog"}},
            ),
            "/do,/see,/process,/chat",
        )
        self.assertEqual(
            cli_main._console_shortcut_hint(
                "command",
                {"operations_dashboard": {"current_focus": "queue_waiting_execution"}},
            ),
            "/do,/see,/process,/chat",
        )
        self.assertEqual(
            cli_main._console_shortcut_hint(
                "command",
                {"operations_dashboard": {"current_focus": "queue_pending_review"}},
            ),
            "/do,/see,/approve,/chat",
        )
        self.assertEqual(
            cli_main._console_shortcut_hint(
                "command",
                {"operations_dashboard": {"current_focus": "runner_restart_backoff"}},
            ),
            "/do,/see,/runner,/chat",
        )
        self.assertEqual(
            cli_main._console_shortcut_hint(
                "command",
                {"operations_dashboard": {"current_focus": "policy_requires_auto_evaluate"}},
            ),
            "/do,/see,/policy,/chat",
        )
        self.assertEqual(
            cli_main._console_shortcut_hint(
                "command",
                {"operations_dashboard": {"current_focus": "insufficient_new_signal_samples"}},
            ),
            "/do,/see,/gate,/chat",
        )
        self.assertEqual(
            cli_main._console_shortcut_hint(
                "command",
                {"operations_dashboard": {"current_focus": "failure_backoff_active"}},
            ),
            "/do,/see,/retry,/chat",
        )
        self.assertEqual(
            cli_main._console_shortcut_hint(
                "command",
                {"operations_dashboard": {"current_focus": "daemon_heartbeat_stale"}},
            ),
            "/do,/see,/daemon,/chat",
        )
        self.assertEqual(
            cli_main._console_shortcut_hint(
                "command",
                {"operations_dashboard": {"current_focus": "stale_runner_lock"}},
            ),
            "/do,/see,/runner,/chat",
        )
        self.assertEqual(
            cli_main._console_shortcut_hint(
                "command",
                {"operations_dashboard": {"current_focus": "daemon_restart_backoff"}},
            ),
            "/do,/see,/alerts,/chat",
        )
        self.assertEqual(
            cli_main._console_shortcut_hint(
                "command",
                {"operations_dashboard": {"current_focus": "candidate_ready_for_promotion"}},
            ),
            "/do,/see,/archive,/chat",
        )
        self.assertEqual(
            cli_main._console_shortcut_hint(
                "command",
                {
                    "operations_dashboard": {"current_focus": "candidate_idle"},
                    "operations_alert_policy": {"required_action": "inspect_candidate_status"},
                },
            ),
            "/do,/see,/candidate,/chat",
        )
        self.assertEqual(
            cli_main._console_shortcut_hint(
                "command",
                {
                    "operations_dashboard": {"current_focus": "runner_active"},
                    "operations_alert_policy": {
                        "required_action": "inspect_runtime_stability",
                        "secondary_action": "inspect_worker_runner_history",
                    },
                },
            ),
            "/do,/see,/runtime,/chat",
        )
        self.assertEqual(
            cli_main._console_shortcut_hint(
                "command",
                {
                    "operations_dashboard": {"current_focus": "daemon_active"},
                    "operations_alert_policy": {
                        "required_action": "inspect_runtime_stability",
                        "secondary_action": "inspect_daemon_status",
                    },
                },
            ),
            "/do,/see,/runtime,/chat",
        )
        self.assertEqual(
            cli_main._console_shortcut_hint(
                "command",
                {
                    "operations_dashboard": {
                        "current_focus": "runner_monitoring",
                        "runtime_action_summary": {
                            "primary_action": "inspect_runtime_stability",
                            "secondary_actions": ["inspect_worker_runner_history"],
                        },
                    }
                },
            ),
            "/do,/see,/runtime,/chat",
        )
        self.assertEqual(
            cli_main._console_shortcut_hint(
                "command",
                {
                    "operations_dashboard": {
                        "current_focus": "none",
                        "monitor_focus": "daemon_active",
                        "runtime_action_summary": {
                            "primary_action": "inspect_runtime_stability",
                            "secondary_actions": ["inspect_daemon_status"],
                        },
                    }
                },
            ),
            "/do,/see,/runtime,/chat",
        )
        self.assertEqual(
            cli_main._console_shortcut_hint(
                "command",
                {
                    "operations_dashboard": {
                        "current_focus": "daemon_monitoring",
                        "runtime_action_summary": {
                            "primary_action": "inspect_runtime_stability",
                            "secondary_actions": ["inspect_daemon_status"],
                        },
                    }
                },
            ),
            "/do,/see,/runtime,/chat",
        )
        self.assertEqual(
            cli_main._console_shortcut_hint(
                "command",
                {
                    "operations_dashboard": {
                        "current_focus": "candidate_archived",
                        "candidate_action_summary": {
                            "primary_action": "inspect_candidate_status",
                            "secondary_actions": ["inspect_candidate_timeline"],
                        },
                    }
                },
            ),
            "/do,/see,/candidate,/chat",
        )
        self.assertEqual(
            cli_main._console_shortcut_hint(
                "command",
                {
                    "operations_dashboard": {
                        "current_focus": "queue_idle",
                        "queue_action_summary": {
                            "primary_action": "process_next_queue_item",
                            "secondary_actions": ["inspect_auto_train_trigger"],
                        },
                    }
                },
            ),
            "/do,/see,/process,/chat",
        )
        self.assertEqual(
            cli_main._console_focus_actions(
                {
                    "operations_dashboard": {"current_focus": "insufficient_new_signal_samples"},
                    "operations_alert_policy": {"required_action": "collect_more_signal_samples"},
                }
            ),
            {
                "primary_label": "/gate",
                "primary_exec": "gate",
                "secondary_label": "/trigger /policy",
                "secondary_exec": "trigger",
            },
        )
        self.assertEqual(
            cli_main._console_focus_actions(
                {"operations_dashboard": {"current_focus": "policy_requires_auto_evaluate"}}
            ),
            {
                "primary_label": "/policy",
                "primary_exec": "policy",
                "secondary_label": "/gate",
                "secondary_exec": "gate",
            },
        )
        self.assertEqual(
            cli_main._console_focus_actions(
                {
                    "operations_dashboard": {
                        "current_focus": "none",
                        "monitor_focus": "candidate_idle",
                        "candidate_action_summary": {
                            "primary_action": "inspect_candidate_status",
                            "secondary_actions": ["inspect_candidate_timeline"],
                        },
                    }
                }
            ),
            {
                "primary_label": "/candidate",
                "primary_exec": "candidate",
                "secondary_label": "/cand tl",
                "secondary_exec": "cand tl",
            },
        )
        self.assertEqual(
            cli_main._console_focus_actions(
                {
                    "operations_dashboard": {"current_focus": "failure_backoff_active"},
                    "operations_alert_policy": {"required_action": "wait_for_failure_backoff"},
                }
            ),
            {
                "primary_label": "/retry",
                "primary_exec": "retry",
                "secondary_label": "/trigger /gate",
                "secondary_exec": "trigger",
            },
        )
        self.assertEqual(
            cli_main._console_focus_actions(
                {
                    "operations_dashboard": {"current_focus": "queue_pending_review"},
                    "operations_alert_policy": {
                        "required_action": "review_queue_confirmation",
                        "secondary_actions": ["inspect_auto_train_gate", "inspect_auto_train_trigger"],
                    },
                }
            ),
            {
                "primary_label": "/approve or /reject",
                "primary_exec": None,
                "secondary_label": "/gate /trigger",
                "secondary_exec": "gate",
            },
        )
        self.assertEqual(
            cli_main._console_focus_actions(
                {"operations_dashboard": {"current_focus": "queue_pending_review"}}
            ),
            {
                "primary_label": "/approve or /reject",
                "primary_exec": None,
                "secondary_label": "/gate /trigger",
                "secondary_exec": "gate",
            },
        )
        self.assertEqual(
            cli_main._console_focus_actions(
                {"operations_dashboard": {"current_focus": "daemon_restart_backoff"}}
            ),
            {
                "primary_label": "/restart daemon",
                "primary_exec": "restart daemon",
                "secondary_label": "/runtime /alerts",
                "secondary_exec": "runtime",
            },
        )
        self.assertEqual(
            cli_main._console_focus_actions(
                {"operations_dashboard": {"current_focus": "candidate_ready_for_promotion"}}
            ),
            {
                "primary_label": "/promote",
                "primary_exec": "promote",
                "secondary_label": "/candidate /cand sum",
                "secondary_exec": "candidate",
            },
        )
        self.assertEqual(
            cli_main._console_focus_actions(
                {
                    "operations_dashboard": {"current_focus": "candidate_idle"},
                    "candidate_summary": {"candidate_can_promote": False, "candidate_can_archive": True},
                }
            ),
            {
                "primary_label": "/archive",
                "primary_exec": "archive",
                "secondary_label": "/candidate /cand sum",
                "secondary_exec": "candidate",
            },
        )
        self.assertEqual(
            cli_main._console_focus_actions(
                {
                    "operations_dashboard": {
                        "current_focus": "candidate_idle",
                        "candidate_action_summary": {
                            "primary_action": "inspect_candidate_status",
                            "secondary_actions": ["inspect_candidate_timeline"],
                        },
                    }
                }
            ),
            {
                "primary_label": "/candidate",
                "primary_exec": "candidate",
                "secondary_label": "/cand tl",
                "secondary_exec": "cand tl",
            },
        )
        self.assertEqual(
            cli_main._console_focus_actions(
                {
                    "operations_dashboard": {
                        "current_focus": "queue_waiting_execution",
                        "queue_action_summary": {
                            "primary_action": "process_next_queue_item",
                            "secondary_actions": ["inspect_auto_train_trigger"],
                        },
                    }
                }
            ),
            {
                "primary_label": "/process",
                "primary_exec": "process",
                "secondary_label": "/trigger",
                "secondary_exec": "trigger",
            },
        )
        self.assertEqual(
            cli_main._console_focus_actions(
                {
                    "operations_dashboard": {"current_focus": "runner_active"},
                    "operations_alert_policy": {
                        "required_action": "inspect_runtime_stability",
                        "secondary_action": "inspect_worker_runner_history",
                    },
                }
            ),
            {
                "primary_label": "/runtime",
                "primary_exec": "runtime",
                "secondary_label": "/runner hist",
                "secondary_exec": "runner hist",
            },
        )
        self.assertEqual(
            cli_main._console_focus_actions(
                {
                    "operations_dashboard": {"current_focus": "daemon_active"},
                    "operations_alert_policy": {
                        "required_action": "inspect_runtime_stability",
                        "secondary_action": "inspect_daemon_status",
                    },
                }
            ),
            {
                "primary_label": "/runtime",
                "primary_exec": "runtime",
                "secondary_label": "/daemon",
                "secondary_exec": "daemon",
            },
        )
        self.assertEqual(
            cli_main._console_focus_actions(
                {
                    "operations_dashboard": {
                        "current_focus": "daemon_active",
                        "runtime_action_summary": {
                            "primary_action": "inspect_runtime_stability",
                            "secondary_actions": ["inspect_daemon_status"],
                        },
                    }
                }
            ),
            {
                "primary_label": "/runtime",
                "primary_exec": "runtime",
                "secondary_label": "/daemon",
                "secondary_exec": "daemon",
            },
        )
        self.assertEqual(
            cli_main._console_focus_actions(
                {
                    "operations_dashboard": {
                        "current_focus": "daemon_monitoring",
                        "runtime_action_summary": {
                            "primary_action": "inspect_runtime_stability",
                            "secondary_actions": ["inspect_daemon_status"],
                        },
                    }
                }
            ),
            {
                "primary_label": "/runtime",
                "primary_exec": "runtime",
                "secondary_label": "/daemon",
                "secondary_exec": "daemon",
            },
        )
        self.assertEqual(
            cli_main._console_focus_actions(
                {
                    "operations_dashboard": {
                        "current_focus": "runner_monitoring",
                        "runtime_action_summary": {
                            "primary_action": "inspect_runtime_stability",
                            "secondary_actions": ["inspect_worker_runner_history"],
                        },
                    }
                }
            ),
            {
                "primary_label": "/runtime",
                "primary_exec": "runtime",
                "secondary_label": "/runner hist",
                "secondary_exec": "runner hist",
            },
        )

    def test_console_prompt_helpers_track_mode_and_focus(self) -> None:
        self.assertEqual(_prompt_placeholder("chat", focus="none"), "Type message, /cmd, or /help")
        self.assertEqual(_prompt_placeholder("command", focus="daemon_stale"), "Type /daemon or /daemon timeline")
        self.assertEqual(_prompt_placeholder("command", focus="daemon_active"), "Type /do or /see")
        self.assertEqual(_prompt_placeholder("command", focus="daemon_restart_backoff"), "Type /do or /see")
        self.assertEqual(_prompt_placeholder("command", focus="daemon_heartbeat_stale"), "Type /do or /see")
        self.assertEqual(_prompt_placeholder("command", focus="stale_runner_lock"), "Type /do or /see")
        self.assertEqual(_prompt_placeholder("command", focus="queue_backlog"), "Type /do or /see")
        self.assertEqual(_prompt_placeholder("command", focus="queue_waiting_execution"), "Type /do or /see")
        self.assertEqual(_prompt_placeholder("command", focus="queue_pending_review"), "Type /do or /see")
        self.assertEqual(_prompt_placeholder("command", focus="policy_requires_auto_evaluate"), "Type /do or /see")
        self.assertEqual(_prompt_placeholder("command", focus="insufficient_new_signal_samples"), "Type /do or /see")
        self.assertEqual(_prompt_placeholder("command", focus="candidate_ready_for_promotion"), "Type /do or /see")
        self.assertEqual(_prompt_placeholder("command", focus="runner_active"), "Type /do or /see")
        self.assertEqual(
            _prompt_placeholder(
                "command",
                focus="candidate_idle",
                payload={"operations_alert_policy": {"required_action": "inspect_candidate_status"}},
            ),
            "Type /do or /see",
        )
        self.assertEqual(
            _prompt_placeholder(
                "command",
                focus="daemon_active",
                payload={"operations_alert_policy": {"required_action": "inspect_runtime_stability"}},
            ),
            "Type /do or /see",
        )
        self.assertEqual(
            _prompt_placeholder(
                "command",
                focus="none",
                payload={
                    "operations_dashboard": {
                        "current_focus": "none",
                        "monitor_focus": "queue_waiting_execution",
                        "queue_action_summary": {
                            "primary_action": "process_next_queue_item",
                            "secondary_actions": ["inspect_auto_train_trigger"],
                        },
                    }
                },
            ),
            "Type /do or /see",
        )
        self.assertEqual(_prompt_mode_help("chat", focus="none"), "reply")
        self.assertEqual(_prompt_mode_help("command", focus="daemon_active"), "runtime")
        self.assertEqual(_prompt_mode_help("command", focus="daemon_restart_backoff"), "restart")
        self.assertEqual(_prompt_mode_help("command", focus="daemon_heartbeat_stale"), "recover")
        self.assertEqual(_prompt_mode_help("command", focus="stale_runner_lock"), "runtime")
        self.assertEqual(_prompt_mode_help("command", focus="runner_restart_backoff"), "runner")
        self.assertEqual(_prompt_mode_help("command", focus="queue_pending_review"), "review")
        self.assertEqual(_prompt_mode_help("command", focus="queue_waiting_execution"), "process")
        self.assertEqual(_prompt_mode_help("command", focus="policy_requires_auto_evaluate"), "trigger")
        self.assertEqual(_prompt_mode_help("command", focus="insufficient_new_signal_samples"), "trigger")
        self.assertEqual(_prompt_mode_help("command", focus="candidate_ready_for_promotion"), "promote")
        self.assertEqual(
            _prompt_mode_help(
                "command",
                focus="candidate_idle",
                payload={"operations_alert_policy": {"required_action": "inspect_candidate_status"}},
            ),
            "candidate",
        )
        self.assertEqual(_prompt_target_hint("chat", focus="none"), "send")
        self.assertEqual(_prompt_target_hint("command", focus="daemon_active"), "runtime")
        self.assertEqual(_prompt_target_hint("command", focus="daemon_restart_backoff"), "restart")
        self.assertEqual(_prompt_target_hint("command", focus="daemon_heartbeat_stale"), "recover")
        self.assertEqual(_prompt_target_hint("command", focus="stale_runner_lock"), "runtime")
        self.assertEqual(_prompt_target_hint("command", focus="queue_backlog"), "process")
        self.assertEqual(_prompt_target_hint("command", focus="queue_waiting_execution"), "process")
        self.assertEqual(_prompt_target_hint("command", focus="queue_pending_review"), "review")
        self.assertEqual(_prompt_target_hint("command", focus="policy_requires_auto_evaluate"), "trigger")
        self.assertEqual(_prompt_target_hint("command", focus="insufficient_new_signal_samples"), "trigger")
        self.assertEqual(_prompt_target_hint("command", focus="candidate_ready_for_promotion"), "promote")
        self.assertEqual(
            _prompt_target_hint(
                "command",
                focus="daemon_active",
                payload={"operations_alert_policy": {"required_action": "inspect_runtime_stability"}},
            ),
            "runtime",
        )
        self.assertEqual(
            _prompt_context_focus(
                {
                    "operations_dashboard": {"current_focus": "none"},
                    "operations_overview": {"monitor_focus": "candidate_idle"},
                }
            ),
            "candidate_idle",
        )
        self.assertEqual(
            _prompt_trigger_category(
                "insufficient_new_signal_samples",
                payload={"operations_console": {"trigger_blocked_category": "data"}},
            ),
            "data",
        )
        self.assertEqual(_prompt_feedback_digest("assistant generating..."), "generating")
        self.assertEqual(_prompt_feedback_digest("running /status"), "/status")
        self.assertEqual(_prompt_ctx_digest("queue_backlog"), "queue_waiting_e...")
        self.assertEqual(_prompt_hint_digest("Enter,/help,^C"), "Enter,/help")
        self.assertEqual(
            _prompt_action_guidance("command", focus="queue_pending_review"),
            ("/approve /reject", "/gate /trigger"),
        )
        self.assertEqual(
            _prompt_action_guidance("command", focus="queue_backlog"),
            ("/process", "/queue /qs"),
        )
        self.assertEqual(
            _prompt_action_guidance("command", focus="queue_waiting_execution"),
            ("/process", "/queue /qs"),
        )
        self.assertEqual(
            _prompt_action_guidance(
                "command",
                focus="candidate_idle",
                payload={
                    "operations_dashboard": {
                        "candidate_action_summary": {
                            "primary_action": "inspect_candidate_status",
                            "secondary_actions": ["inspect_candidate_timeline"],
                        }
                    }
                },
            ),
            ("/candidate", "/cand tl"),
        )
        self.assertEqual(
            _prompt_action_guidance(
                "command",
                focus="queue_waiting_execution",
                payload={
                    "operations_dashboard": {
                        "queue_action_summary": {
                            "primary_action": "process_next_queue_item",
                            "secondary_actions": ["inspect_auto_train_trigger"],
                        }
                    }
                },
            ),
            ("/process", "/trigger"),
        )
        self.assertEqual(
            _prompt_action_guidance(
                "command",
                focus="daemon_monitoring",
                payload={
                    "operations_dashboard": {
                        "runtime_action_summary": {
                            "primary_action": "inspect_runtime_stability",
                            "secondary_actions": ["inspect_daemon_status"],
                        }
                    }
                },
            ),
            ("/runtime", "/daemon"),
        )
        self.assertEqual(
            _prompt_action_guidance(
                "command",
                focus="runner_monitoring",
                payload={
                    "operations_dashboard": {
                        "runtime_action_summary": {
                            "primary_action": "inspect_runtime_stability",
                            "secondary_actions": ["inspect_worker_runner_history"],
                        }
                    }
                },
            ),
            ("/runtime", "/runner hist"),
        )
        self.assertEqual(
            _prompt_action_guidance("command", focus="daemon_restart_backoff"),
            ("/restart daemon", "/runtime /alerts"),
        )
        self.assertEqual(
            _prompt_action_guidance("chat", focus="none"),
            ("send", "/cmd"),
        )
        self.assertEqual(
            _payload_command_guidance(
                {
                    "candidate_summary": {
                        "candidate_can_promote": False,
                        "candidate_can_archive": True,
                    }
                },
                "candidate_idle",
            ),
            ("/archive", "/candidate /cand sum"),
        )
        self.assertEqual(
            _payload_command_guidance(
                {
                    "operations_dashboard": {"current_focus": "none"},
                    "operations_console": {
                        "monitor_focus": "queue_waiting_execution",
                        "queue_action_summary": {
                            "primary_action": "process_next_queue_item",
                            "secondary_actions": ["inspect_auto_train_trigger"],
                        },
                    },
                },
                None,
            ),
            ("/process", "/trigger"),
        )
        self.assertEqual(
            _payload_command_guidance(
                {
                    "operations_dashboard": {"current_focus": "none"},
                    "operations_overview": {"monitor_focus": "candidate_idle"},
                    "operations_console": {
                        "candidate_action_summary": {
                            "primary_action": "inspect_candidate_status",
                            "secondary_actions": ["inspect_candidate_timeline"],
                        },
                    },
                },
                None,
            ),
            ("/candidate", "/cand tl"),
        )
        self.assertEqual(
            _payload_command_guidance(
                {
                    "operations_alert_policy": {
                        "required_action": "review_queue_confirmation",
                        "secondary_actions": ["inspect_auto_train_gate", "inspect_auto_train_trigger"],
                    }
                },
                "queue_pending_review",
            ),
            ("/approve /reject", "/gate /trigger"),
        )
        self.assertEqual(
            _payload_command_guidance(
                {
                    "operations_alert_policy": {
                        "required_action": "process_next_queue_item",
                        "secondary_action": "review_queue_confirmation",
                    }
                },
                "queue_waiting_execution",
            ),
            ("/process", "/approve /reject"),
        )
        self.assertEqual(
            _payload_command_guidance(
                {
                    "operations_alert_policy": {
                        "required_action": "collect_more_signal_samples",
                        "secondary_action": "inspect_auto_train_policy",
                    }
                },
                "insufficient_new_signal_samples",
            ),
            ("/gate", "/policy"),
        )
        self.assertEqual(
            _payload_command_guidance(
                {
                    "operations_alert_policy": {
                        "required_action": "archive_candidate",
                        "secondary_action": "promote_candidate",
                    }
                },
                "candidate_idle",
            ),
            ("/archive", "/promote"),
        )
        self.assertEqual(
            _payload_command_guidance(
                {
                    "operations_alert_policy": {
                        "required_action": "inspect_candidate_status",
                        "secondary_action": "inspect_candidate_timeline",
                    }
                },
                "candidate_idle",
            ),
            ("/candidate", "/cand tl"),
        )
        self.assertEqual(
            _payload_command_guidance(
                {
                    "operations_dashboard": {
                        "candidate_action_summary": {
                            "primary_action": "inspect_candidate_status",
                            "secondary_actions": ["inspect_candidate_timeline"],
                        }
                    }
                },
                "candidate_idle",
            ),
            ("/candidate", "/cand tl"),
        )
        self.assertEqual(
            _payload_command_guidance(
                {
                    "operations_dashboard": {
                        "queue_action_summary": {
                            "primary_action": "process_next_queue_item",
                            "secondary_actions": ["inspect_auto_train_trigger"],
                        }
                    }
                },
                "queue_waiting_execution",
            ),
            ("/process", "/trigger"),
        )
        self.assertEqual(
            _payload_command_guidance(
                {
                    "operations_alert_policy": {
                        "required_action": "inspect_runtime_stability",
                        "secondary_action": "inspect_worker_runner_history",
                    }
                },
                "runner_active",
            ),
            ("/runtime", "/runner hist"),
        )
        self.assertEqual(
            _payload_command_guidance(
                {
                    "operations_alert_policy": {
                        "required_action": "inspect_runtime_stability",
                        "secondary_action": "inspect_daemon_status",
                    }
                },
                "daemon_active",
            ),
            ("/runtime", "/daemon"),
        )
        self.assertEqual(
            _payload_command_guidance(
                {
                    "operations_dashboard": {
                        "runtime_action_summary": {
                            "primary_action": "inspect_runtime_stability",
                            "secondary_actions": ["inspect_daemon_status"],
                        }
                    }
                },
                "daemon_active",
            ),
            ("/runtime", "/daemon"),
        )
        self.assertEqual(
            _payload_command_guidance(
                {
                    "operations_alert_policy": {
                        "required_action": "wait_for_failure_backoff",
                    }
                },
                "failure_backoff_active",
            ),
            ("/retry", "/trigger /gate"),
        )

    def test_console_command_output_supports_trigger_gate_and_runtime_views(self) -> None:
        payload = {
            "auto_train_trigger": {
                "enabled": True,
                "state": "blocked",
                "ready": False,
                "reason": "queue_pending_review",
                "blocked_primary_reason": "queue_pending_review",
                "blocked_primary_action": "review_queue_confirmation",
                "blocked_primary_category": "queue",
                "queue_gate_reason": "queue_pending_review",
                "queue_gate_action": "review_queue_confirmation",
                "queue_review_mode": "manual_review",
                "blocked_summary": "queue pending review -> review_queue_confirmation",
                "policy": {
                    "queue_entry_mode": "awaiting_confirmation",
                    "review_mode": "manual_review",
                    "evaluation_mode": "auto_evaluate",
                    "promotion_mode": "auto_promote",
                    "stop_stage": "confirmation",
                },
                "threshold_summary": {
                    "eligible_signal_train_samples": 2,
                    "min_new_samples": 50,
                    "holdout_ready": True,
                    "interval_elapsed": False,
                    "cooldown_elapsed": False,
                    "failure_backoff_elapsed": True,
                },
            },
            "train_queue": {
                "review_policy_summary": {
                    "review_mode": "manual_review",
                    "review_required_now": True,
                    "next_action": "review_queue_confirmation",
                    "review_reason": "awaiting_confirmation",
                }
            },
            "operations_console": {
                "runtime_stability_summary": {
                    "runner_active": False,
                    "runner_lock_state": "stale",
                    "runner_stop_requested": False,
                    "daemon_health_state": "stale",
                    "daemon_heartbeat_state": "stale",
                    "daemon_lease_state": "expired",
                    "daemon_restart_policy_state": "backoff",
                    "daemon_recovery_action": "recover_worker_daemon",
                }
            },
            "operations_alert_policy": {
                "required_action": "inspect_daemon_restart_policy",
                "action_priority": "p1",
                "remediation_mode": "manual_review",
                "operator_guidance": "inspect daemon restart policy before retrying",
            },
        }

        for command, expected_action, expected_text in (
            ("trigger", "trigger-summary", "PFE auto-train trigger summary"),
            ("trig", "trigger-summary", "blocked_primary_reason=queue_pending_review"),
            ("gate", "gate-summary", "PFE gate summary"),
            ("runtime", "runtime-summary", "PFE runtime stability summary"),
            ("rt", "runtime-summary", "daemon_restart_policy_state=backoff"),
        ):
            text, action, updates = cli_main._console_command_output(
                command,
                payload=payload,
                workspace="user_default",
                service=object(),
                current_workspace="user_default",
                mode="command",
                model="local",
                adapter="latest",
                temperature=0.7,
                max_tokens=None,
                real_local=False,
                refresh_seconds=2.0,
            )
            self.assertEqual(action, expected_action)
            self.assertIsNone(updates)
            self.assertIn(expected_text, text or "")

    def test_console_command_output_supports_runtime_and_review_actions(self) -> None:
        payload = {
            "workspace": "user_default",
            "operations_dashboard": {"current_focus": "daemon_restart_backoff", "severity": "warning"},
        }

        class FakeService:
            def promote_candidate(self, workspace: str | None = None, note: str | None = None):
                del note
                return {
                    "workspace": workspace or "user_default",
                    "candidate_action": {
                        "action": "promote_candidate",
                        "status": "completed",
                        "reason": "candidate_promoted",
                        "candidate_version": "20260325-001",
                        "promoted_version": "20260325-001",
                    },
                    "operations_dashboard": {"severity": "info", "status": "healthy", "current_focus": "none"},
                }

            def archive_candidate(self, workspace: str | None = None, note: str | None = None):
                del note
                return {
                    "workspace": workspace or "user_default",
                    "candidate_action": {
                        "action": "archive_candidate",
                        "status": "completed",
                        "reason": "candidate_archived",
                        "candidate_version": "20260325-001",
                    },
                    "operations_dashboard": {"severity": "info", "status": "healthy", "current_focus": "none"},
                }

            def approve_next_train_queue(self, workspace: str | None = None, note: str | None = None):
                del note
                return {
                    "workspace": workspace or "user_default",
                    "auto_train_trigger_action": {"action": "approve_next", "status": "approved"},
                    "operations_dashboard": {"severity": "info", "status": "healthy", "current_focus": "none"},
                }

            def reject_next_train_queue(self, workspace: str | None = None, note: str | None = None):
                del note
                return {
                    "workspace": workspace or "user_default",
                    "auto_train_trigger_action": {"action": "reject_next", "status": "rejected"},
                    "operations_dashboard": {"severity": "warning", "status": "attention", "current_focus": "queue_pending_review"},
                }

            def process_next_train_queue(self, workspace: str | None = None):
                return {
                    "workspace": workspace or "user_default",
                    "auto_train_trigger_action": {"action": "process_next", "status": "completed"},
                    "operations_dashboard": {"severity": "info", "status": "healthy", "current_focus": "none"},
                }

            def retry_auto_train_trigger(self, workspace: str | None = None):
                return {
                    "workspace": workspace or "user_default",
                    "auto_train_trigger": {"state": "blocked", "reason": "queue_pending_review"},
                    "operations_dashboard": {"severity": "warning", "status": "attention", "current_focus": "queue_pending_review"},
                }

            def reset_auto_train_trigger(self, workspace: str | None = None):
                return {
                    "workspace": workspace or "user_default",
                    "auto_train_trigger": {"state": "idle", "reason": "reset"},
                    "operations_dashboard": {"severity": "info", "status": "healthy", "current_focus": "none"},
                }

            def recover_train_queue_daemon(self, workspace: str | None = None, note: str | None = None):
                del note
                return {
                    "workspace": workspace or "user_default",
                    "desired_state": "running",
                    "observed_state": "running",
                    "command_status": "recovered",
                    "lock_state": "active",
                    "health_state": "recovering",
                    "heartbeat_state": "fresh",
                    "lease_state": "valid",
                    "recovery_action": "recover_worker_daemon",
                }

            def restart_train_queue_daemon(self, workspace: str | None = None, note: str | None = None):
                del note
                return {
                    "workspace": workspace or "user_default",
                    "desired_state": "running",
                    "observed_state": "running",
                    "command_status": "restarted",
                    "lock_state": "active",
                    "health_state": "recovering",
                    "heartbeat_state": "fresh",
                    "lease_state": "valid",
                    "recovery_action": "restart_worker_daemon",
                }

        for command, expected_action, expected_text in (
            ("approve", "approve-next", "PFE STATUS"),
            ("reject", "reject-next", "PFE STATUS"),
            ("process", "process-next", "PFE STATUS"),
            ("retry", "retry-trigger", "PFE STATUS"),
            ("reset", "reset-trigger", "PFE STATUS"),
            ("recover daemon", "daemon-recover", "PFE worker daemon"),
            ("restart daemon", "daemon-restart", "PFE worker daemon"),
            ("do", "daemon-restart", "PFE worker daemon"),
            ("see", "runtime-summary", "PFE runtime stability summary"),
            ("promote", "candidate-promote", "PFE STATUS"),
            ("archive", "candidate-archive", "PFE STATUS"),
        ):
            text, action, updates = cli_main._console_command_output(
                command,
                payload=payload,
                workspace="user_default",
                service=FakeService(),
                current_workspace="user_default",
                mode="command",
                model="local",
                adapter="latest",
                temperature=0.7,
                max_tokens=None,
                real_local=False,
                refresh_seconds=2.0,
            )
            self.assertEqual(action, expected_action)
            self.assertIsNone(updates)
            from tests.matrix_test_compat import strip_ansi
            clean = strip_ansi(text or "")
            self.assertIn(expected_text, clean)

        review_text, review_action, review_updates = cli_main._console_command_output(
            "do",
            payload={
                "workspace": "user_default",
                "operations_dashboard": {"current_focus": "queue_pending_review", "severity": "warning"},
            },
            workspace="user_default",
            service=FakeService(),
            current_workspace="user_default",
            mode="command",
            model="local",
            adapter="latest",
            temperature=0.7,
            max_tokens=None,
            real_local=False,
            refresh_seconds=2.0,
        )
        self.assertEqual(review_action, "do-ambiguous")
        self.assertIsNone(review_updates)
        self.assertIn("Use /approve or /reject", review_text or "")

        candidate_text, candidate_action, candidate_updates = cli_main._console_command_output(
            "do",
            payload={
                "workspace": "user_default",
                "operations_dashboard": {"current_focus": "candidate_ready_for_promotion", "severity": "warning"},
            },
            workspace="user_default",
            service=FakeService(),
            current_workspace="user_default",
            mode="command",
            model="local",
            adapter="latest",
            temperature=0.7,
            max_tokens=None,
            real_local=False,
            refresh_seconds=2.0,
        )
        self.assertEqual(candidate_action, "candidate-promote")
        self.assertIsNone(candidate_updates)
        from tests.matrix_test_compat import strip_ansi
        self.assertIn("PFE STATUS", strip_ansi(candidate_text or ""))

        trigger_text, trigger_action, trigger_updates = cli_main._console_command_output(
            "do",
            payload={
                "workspace": "user_default",
                "operations_dashboard": {"current_focus": "insufficient_new_signal_samples", "severity": "warning"},
                "operations_alert_policy": {
                    "required_action": "collect_more_signal_samples",
                },
                "auto_train_trigger": {
                    "enabled": True,
                    "state": "blocked",
                    "blocked_primary_reason": "insufficient_new_signal_samples",
                    "blocked_primary_action": "collect_more_signal_samples",
                },
            },
            workspace="user_default",
            service=FakeService(),
            current_workspace="user_default",
            mode="command",
            model="local",
            adapter="latest",
            temperature=0.7,
            max_tokens=None,
            real_local=False,
            refresh_seconds=2.0,
        )
        self.assertEqual(trigger_action, "gate-summary")
        self.assertIsNone(trigger_updates)
        self.assertIn("PFE gate summary", trigger_text or "")

        backoff_text, backoff_action, backoff_updates = cli_main._console_command_output(
            "do",
            payload={
                "workspace": "user_default",
                "operations_dashboard": {"current_focus": "failure_backoff_active", "severity": "warning"},
                "operations_alert_policy": {
                    "required_action": "wait_for_failure_backoff",
                },
            },
            workspace="user_default",
            service=FakeService(),
            current_workspace="user_default",
            mode="command",
            model="local",
            adapter="latest",
            temperature=0.7,
            max_tokens=None,
            real_local=False,
            refresh_seconds=2.0,
        )
        self.assertEqual(backoff_action, "retry-trigger")
        self.assertIsNone(backoff_updates)
        self.assertIn("PFE STATUS", strip_ansi(backoff_text or ""))

        process_text, process_action, process_updates = cli_main._console_command_output(
            "do",
            payload={
                "workspace": "user_default",
                "operations_dashboard": {"current_focus": "queue_waiting_execution", "severity": "info"},
                "operations_alert_policy": {
                    "required_action": "process_next_queue_item",
                    "secondary_action": "review_queue_confirmation",
                },
            },
            workspace="user_default",
            service=FakeService(),
            current_workspace="user_default",
            mode="command",
            model="local",
            adapter="latest",
            temperature=0.7,
            max_tokens=None,
            real_local=False,
            refresh_seconds=2.0,
        )
        self.assertEqual(process_action, "process-next")
        self.assertIsNone(process_updates)
        self.assertIn("PFE STATUS", strip_ansi(process_text or ""))

        archive_text, archive_action, archive_updates = cli_main._console_command_output(
            "do",
            payload={
                "workspace": "user_default",
                "operations_dashboard": {"current_focus": "candidate_idle", "severity": "warning"},
                "candidate_summary": {"candidate_can_promote": False, "candidate_can_archive": True},
            },
            workspace="user_default",
            service=FakeService(),
            current_workspace="user_default",
            mode="command",
            model="local",
            adapter="latest",
            temperature=0.7,
            max_tokens=None,
            real_local=False,
            refresh_seconds=2.0,
        )
        self.assertEqual(archive_action, "candidate-archive")
        self.assertIsNone(archive_updates)
        self.assertIn("PFE STATUS", strip_ansi(archive_text or ""))

    def test_console_command_output_supports_session_setting_updates(self) -> None:
        payload = {
            "operations_console": {},
            "operations_dashboard": {"current_focus": "none"},
        }

        text, action, updates = cli_main._console_command_output(
            "workspace qa-workspace",
            payload=payload,
            workspace="user_default",
            service=object(),
            current_workspace="user_default",
            mode="command",
            model="local",
            adapter="latest",
            temperature=0.7,
            max_tokens=None,
            real_local=False,
            refresh_seconds=2.0,
        )
        self.assertEqual(action, "set-workspace")
        self.assertEqual(updates, {"workspace": "qa-workspace"})
        self.assertIn("workspace set to", text or "")

        text, action, updates = cli_main._console_command_output(
            "temperature 0.2",
            payload=payload,
            workspace="user_default",
            service=object(),
            current_workspace="user_default",
            mode="command",
            model="local",
            adapter="latest",
            temperature=0.7,
            max_tokens=None,
            real_local=False,
            refresh_seconds=2.0,
        )
        self.assertEqual(action, "set-temperature")
        self.assertEqual(updates, {"temperature": 0.2})
        self.assertIn("temperature set to 0.20", text or "")

        text, action, updates = cli_main._console_command_output(
            "max-tokens auto",
            payload=payload,
            workspace="user_default",
            service=object(),
            current_workspace="user_default",
            mode="command",
            model="local",
            adapter="latest",
            temperature=0.7,
            max_tokens=96,
            real_local=False,
            refresh_seconds=2.0,
        )
        self.assertEqual(action, "set-max-tokens")
        self.assertEqual(updates, {"max_tokens": None})
        self.assertIn("max tokens set to auto", text or "")

        text, action, updates = cli_main._console_command_output(
            "refresh 1.5",
            payload=payload,
            workspace="user_default",
            service=object(),
            current_workspace="user_default",
            mode="command",
            model="local",
            adapter="latest",
            temperature=0.7,
            max_tokens=None,
            real_local=False,
            refresh_seconds=2.0,
        )
        self.assertEqual(action, "set-refresh")
        self.assertEqual(updates, {"refresh_seconds": 1.5})
        self.assertIn("refresh set to 1.5s", text or "")

    def test_console_command_output_supports_ops_doctor_and_serve(self) -> None:
        payload = {
            "workspace": "user_default",
            "strict_local": True,
            "operations_overview": {"summary_line": "auto=disabled | queue=0"},
            "operations_dashboard": {"current_focus": "none", "severity": "info"},
            "operations_alert_policy": {
                "required_action": "observe_and_monitor",
                "action_priority": "p2",
                "escalation_mode": "monitor",
            },
            "operations_health": {"status": "healthy", "daemon_health_state": "stopped"},
            "operations_event_stream": {
                "count": 1,
                "severity": "info",
                "status": "healthy",
                "latest_source": "queue",
                "items": [{"source": "queue", "event": "idle", "severity": "info"}],
            },
            "candidate_summary": {
                "candidate_version": "20260325-001",
                "candidate_state": "pending_eval",
                "candidate_can_promote": False,
                "promotion_compare_comparison": "right_better",
                "promotion_compare_recommendation": "review",
                "promotion_compare_overall_delta": 0.14,
                "promotion_compare_style_preference_hit_rate_delta": 0.4,
                "promotion_compare_personalization_summary": "personalization delta +0.35",
            },
            "operations_console": {"summary_line": "console=ok"},
            "train_queue": {
                "count": 1,
                "max_priority": 100,
                "awaiting_confirmation_count": 0,
                "reviewed_transition_count": 1,
            },
            "train_queue_worker_runner": {
                "active": False,
                "lock_state": "idle",
                "stop_requested": False,
                "processed_count": 2,
                "failed_count": 0,
                "loop_cycles": 3,
            },
            "runner_timeline": {
                "count": 2,
                "last_event": "started",
                "takeover_event_count": 0,
                "current_lock_state": "idle",
            },
            "daemon_timeline": {
                "count": 1,
                "recovery_event_count": 0,
                "last_event": "started",
                "last_reason": "manual",
            },
        }

        class FakeService:
            def candidate_timeline(self, workspace: str | None = None, limit: int = 5):
                del limit
                return {
                    "workspace": workspace or "user_default",
                    "current_stage": "pending_eval",
                    "transition_count": 2,
                    "last_candidate_version": "20260325-001",
                }

            def candidate_history(self, workspace: str | None = None, limit: int = 10):
                del limit
                return {
                    "workspace": workspace or "user_default",
                    "count": 1,
                    "last_action": "promote_candidate",
                    "items": [
                        {
                            "timestamp": "2026-03-27T00:00:00Z",
                            "action": "promote_candidate",
                            "status": "completed",
                            "candidate_version": "20260325-001",
                        }
                    ],
                }

            def train_queue_daemon_status(self, workspace: str | None = None):
                return {
                    "workspace": workspace or "user_default",
                    "desired_state": "running",
                    "observed_state": "running",
                    "command_status": "active",
                    "lock_state": "active",
                    "health_state": "healthy",
                    "lease_state": "held",
                    "heartbeat_state": "fresh",
                    "recovery_action": "none",
                }

            def train_queue_daemon_history(self, workspace: str | None = None, limit: int = 10):
                del limit
                return {
                    "workspace": workspace or "user_default",
                    "count": 1,
                    "items": [
                        {
                            "timestamp": "2026-03-27T00:00:00Z",
                            "event": "started",
                            "reason": "manual",
                        }
                    ],
                }

            def train_queue_worker_runner_history(self, workspace: str | None = None, limit: int = 10):
                del limit
                return {
                    "workspace": workspace or "user_default",
                    "count": 1,
                    "items": [
                        {
                            "timestamp": "2026-03-27T00:00:00Z",
                            "event": "started",
                            "reason": "manual",
                        }
                    ],
                }

            def train_queue_history(self, workspace: str | None = None, limit: int = 10):
                del limit
                return {
                    "workspace": workspace or "user_default",
                    "count": 1,
                    "state": "completed",
                    "history": [
                        {
                            "timestamp": "2026-03-27T00:00:00Z",
                            "event": "completed",
                            "state": "completed",
                            "reason": "success",
                        }
                    ],
                }

        text, action, updates = cli_main._console_command_output(
            "status compact",
            payload=payload,
            workspace="user_default",
            service=FakeService(),
            current_workspace="user_default",
            mode="command",
            model="local",
            adapter="latest",
            temperature=0.7,
            max_tokens=None,
            real_local=False,
            refresh_seconds=2.0,
        )
        self.assertEqual(action, "status-compact")
        self.assertIsNone(updates)
        self.assertIn("PFE status compact", text or "")
        self.assertIn("focus=none", text or "")

        text, action, updates = cli_main._console_command_output(
            "ops dashboard",
            payload=payload,
            workspace="user_default",
            service=FakeService(),
            current_workspace="user_default",
            mode="command",
            model="local",
            adapter="latest",
            temperature=0.7,
            max_tokens=None,
            real_local=False,
            refresh_seconds=2.0,
        )
        self.assertEqual(action, "ops-dashboard")
        self.assertIsNone(updates)
        self.assertIn("operations dashboard:", text or "")
        self.assertIn("current_focus=none", text or "")

        text, action, updates = cli_main._console_command_output(
            "ops policy",
            payload=payload,
            workspace="user_default",
            service=FakeService(),
            current_workspace="user_default",
            mode="command",
            model="local",
            adapter="latest",
            temperature=0.7,
            max_tokens=None,
            real_local=False,
            refresh_seconds=2.0,
        )
        self.assertEqual(action, "ops-policy")
        self.assertIsNone(updates)
        self.assertIn("operations alert policy:", text or "")
        self.assertIn("required_action=observe_and_monitor", text or "")

        text, action, updates = cli_main._console_command_output(
            "ops alerts",
            payload=payload,
            workspace="user_default",
            service=FakeService(),
            current_workspace="user_default",
            mode="command",
            model="local",
            adapter="latest",
            temperature=0.7,
            max_tokens=None,
            real_local=False,
            refresh_seconds=2.0,
        )
        self.assertEqual(action, "ops-alerts")
        self.assertIsNone(updates)
        self.assertIn("operations alerts:", text or "")
        self.assertIn("status=healthy", text or "")

        text, action, updates = cli_main._console_command_output(
            "event stream",
            payload=payload,
            workspace="user_default",
            service=FakeService(),
            current_workspace="user_default",
            mode="command",
            model="local",
            adapter="latest",
            temperature=0.7,
            max_tokens=None,
            real_local=False,
            refresh_seconds=2.0,
        )
        self.assertEqual(action, "event-stream")
        self.assertIsNone(updates)
        self.assertIn("operations event stream:", text or "")
        self.assertIn("latest_source=queue", text or "")

        text, action, updates = cli_main._console_command_output(
            "ops",
            payload=payload,
            workspace="user_default",
            service=FakeService(),
            current_workspace="user_default",
            mode="command",
            model="local",
            adapter="latest",
            temperature=0.7,
            max_tokens=None,
            real_local=False,
            refresh_seconds=2.0,
        )
        self.assertEqual(action, "ops")
        self.assertIsNone(updates)
        self.assertIn("operations console digest:", text or "")

        text, action, updates = cli_main._console_command_output(
            "doctor",
            payload=payload,
            workspace="user_default",
            service=FakeService(),
            current_workspace="user_default",
            mode="command",
            model="local",
            adapter="latest",
            temperature=0.7,
            max_tokens=None,
            real_local=False,
            refresh_seconds=2.0,
        )
        self.assertEqual(action, "doctor")
        self.assertIsNone(updates)
        self.assertIn("PFE doctor", text or "")

        text, action, updates = cli_main._console_command_output(
            "serve",
            payload=payload,
            workspace="user_default",
            service=FakeService(),
            current_workspace="user_default",
            mode="command",
            model="local",
            adapter="latest",
            temperature=0.7,
            max_tokens=None,
            real_local=False,
            refresh_seconds=2.0,
        )
        self.assertEqual(action, "serve")
        self.assertIsNone(updates)
        from tests.matrix_test_compat import strip_ansi
        clean = strip_ansi(text or "")
        self.assertIn("SERVE PREVIEW", clean)

        text, action, updates = cli_main._console_command_output(
            "candidate summary",
            payload=payload,
            workspace="user_default",
            service=FakeService(),
            current_workspace="user_default",
            mode="command",
            model="local",
            adapter="latest",
            temperature=0.7,
            max_tokens=None,
            real_local=False,
            refresh_seconds=2.0,
        )
        self.assertEqual(action, "candidate-summary")
        self.assertIsNone(updates)
        self.assertIn("PFE candidate summary", text or "")
        self.assertIn("current_stage=pending_eval", text or "")
        self.assertIn("promotion_compare_style_preference_hit_rate_delta=0.4", text or "")
        self.assertIn("promotion_compare_personalization_summary=personalization delta +0.35", text or "")

        text, action, updates = cli_main._console_command_output(
            "candidate timeline",
            payload=payload,
            workspace="user_default",
            service=FakeService(),
            current_workspace="user_default",
            mode="command",
            model="local",
            adapter="latest",
            temperature=0.7,
            max_tokens=None,
            real_local=False,
            refresh_seconds=2.0,
        )
        self.assertEqual(action, "candidate")
        self.assertIsNone(updates)
        self.assertIn("PFE candidate timeline", text or "")
        self.assertIn("current_stage=pending_eval", text or "")

        text, action, updates = cli_main._console_command_output(
            "candidate history",
            payload=payload,
            workspace="user_default",
            service=FakeService(),
            current_workspace="user_default",
            mode="command",
            model="local",
            adapter="latest",
            temperature=0.7,
            max_tokens=None,
            real_local=False,
            refresh_seconds=2.0,
        )
        self.assertEqual(action, "candidate-history")
        self.assertIsNone(updates)
        self.assertIn("PFE candidate history", text or "")
        self.assertIn("last_action=promote_candidate", text or "")

        text, action, updates = cli_main._console_command_output(
            "queue",
            payload=payload,
            workspace="user_default",
            service=FakeService(),
            current_workspace="user_default",
            mode="command",
            model="local",
            adapter="latest",
            temperature=0.7,
            max_tokens=None,
            real_local=False,
            refresh_seconds=2.0,
        )
        self.assertEqual(action, "queue-summary")
        self.assertIsNone(updates)
        self.assertIn("PFE train queue summary", text or "")
        self.assertIn("count=1", text or "")

        text, action, updates = cli_main._console_command_output(
            "queue summary",
            payload=payload,
            workspace="user_default",
            service=FakeService(),
            current_workspace="user_default",
            mode="command",
            model="local",
            adapter="latest",
            temperature=0.7,
            max_tokens=None,
            real_local=False,
            refresh_seconds=2.0,
        )
        self.assertEqual(action, "queue-summary")
        self.assertIsNone(updates)
        self.assertIn("PFE train queue summary", text or "")
        self.assertIn("count=1", text or "")
        self.assertIn("max_priority=100", text or "")

        text, action, updates = cli_main._console_command_output(
            "runner",
            payload=payload,
            workspace="user_default",
            service=FakeService(),
            current_workspace="user_default",
            mode="command",
            model="local",
            adapter="latest",
            temperature=0.7,
            max_tokens=None,
            real_local=False,
            refresh_seconds=2.0,
        )
        self.assertEqual(action, "runner-summary")
        self.assertIsNone(updates)
        self.assertIn("PFE worker runner summary", text or "")
        self.assertIn("processed_count=2", text or "")

        text, action, updates = cli_main._console_command_output(
            "runner history",
            payload=payload,
            workspace="user_default",
            service=FakeService(),
            current_workspace="user_default",
            mode="command",
            model="local",
            adapter="latest",
            temperature=0.7,
            max_tokens=None,
            real_local=False,
            refresh_seconds=2.0,
        )
        self.assertEqual(action, "runner-history")
        self.assertIsNone(updates)
        self.assertIn("PFE worker runner history", text or "")
        self.assertIn("event=started", text or "")

        text, action, updates = cli_main._console_command_output(
            "runner summary",
            payload=payload,
            workspace="user_default",
            service=FakeService(),
            current_workspace="user_default",
            mode="command",
            model="local",
            adapter="latest",
            temperature=0.7,
            max_tokens=None,
            real_local=False,
            refresh_seconds=2.0,
        )
        self.assertEqual(action, "runner-summary")
        self.assertIsNone(updates)
        self.assertIn("PFE worker runner summary", text or "")
        self.assertIn("processed_count=2", text or "")
        self.assertIn("loop_cycles=3", text or "")

        text, action, updates = cli_main._console_command_output(
            "runner timeline",
            payload=payload,
            workspace="user_default",
            service=FakeService(),
            current_workspace="user_default",
            mode="command",
            model="local",
            adapter="latest",
            temperature=0.7,
            max_tokens=None,
            real_local=False,
            refresh_seconds=2.0,
        )
        self.assertEqual(action, "runner-timeline")
        self.assertIsNone(updates)
        self.assertIn("runner timeline:", text or "")
        self.assertIn("count=2", text or "")

        text, action, updates = cli_main._console_command_output(
            "queue history",
            payload=payload,
            workspace="user_default",
            service=FakeService(),
            current_workspace="user_default",
            mode="command",
            model="local",
            adapter="latest",
            temperature=0.7,
            max_tokens=None,
            real_local=False,
            refresh_seconds=2.0,
        )
        self.assertEqual(action, "queue-history")
        self.assertIsNone(updates)
        self.assertIn("PFE train queue history", text or "")
        self.assertIn("event=completed", text or "")

        text, action, updates = cli_main._console_command_output(
            "daemon summary",
            payload=payload,
            workspace="user_default",
            service=FakeService(),
            current_workspace="user_default",
            mode="command",
            model="local",
            adapter="latest",
            temperature=0.7,
            max_tokens=None,
            real_local=False,
            refresh_seconds=2.0,
        )
        self.assertEqual(action, "daemon-summary")
        self.assertIsNone(updates)
        self.assertIn("PFE worker daemon summary", text or "")
        self.assertIn("observed_state=running", text or "")

        text, action, updates = cli_main._console_command_output(
            "daemon timeline",
            payload=payload,
            workspace="user_default",
            service=FakeService(),
            current_workspace="user_default",
            mode="command",
            model="local",
            adapter="latest",
            temperature=0.7,
            max_tokens=None,
            real_local=False,
            refresh_seconds=2.0,
        )
        self.assertEqual(action, "daemon-timeline")
        self.assertIsNone(updates)
        self.assertIn("daemon timeline:", text or "")
        self.assertIn("last_reason=manual", text or "")

        text, action, updates = cli_main._console_command_output(
            "daemon history",
            payload=payload,
            workspace="user_default",
            service=FakeService(),
            current_workspace="user_default",
            mode="command",
            model="local",
            adapter="latest",
            temperature=0.7,
            max_tokens=None,
            real_local=False,
            refresh_seconds=2.0,
        )
        self.assertEqual(action, "daemon-history")
        self.assertIsNone(updates)
        self.assertIn("PFE worker daemon history", text or "")
        self.assertIn("event=started", text or "")

    def test_console_status_compact_prefers_monitor_focus_when_current_focus_is_none(self) -> None:
        text = cli_main._console_status_compact_text(
            {
                "workspace": "user_default",
                "latest_adapter": {"version": "20260325-001"},
                "operations_overview": {
                    "summary_line": "legacy queue=1 | state=queued",
                    "inspection_summary_line": "current_focus=queue_waiting_execution | required_action=process_next_queue_item | next_actions=process_next_queue_item,inspect_auto_train_trigger",
                },
                "operations_dashboard": {
                    "severity": "info",
                    "current_focus": "none",
                    "monitor_focus": "queue_waiting_execution",
                },
                "operations_alert_policy": {
                    "required_action": "process_next_queue_item",
                },
                "train_queue": {"count": 1},
            },
            workspace="user_default",
        )

        self.assertIn("PFE status compact", text)
        self.assertIn("focus=queue_waiting_execution", text)
        self.assertIn("action=process_next_queue_item", text)
        self.assertIn(
            "summary=current_focus=queue_waiting_execution | required_action=process_next_queue_item | next_actions=process_next_queue_item,inspect_auto_train_trigger",
            text,
        )
        self.assertNotIn("summary=legacy queue=1 | state=queued", text)
        self.assertNotIn("focus=none", text)

    def test_console_apply_edit_supports_insert_move_and_delete(self) -> None:
        text, cursor = cli_main._console_apply_edit("helo", 3, "insert", "l")
        self.assertEqual((text, cursor), ("hello", 4))
        text, cursor = cli_main._console_apply_edit(text, cursor, "left")
        self.assertEqual((text, cursor), ("hello", 3))
        text, cursor = cli_main._console_apply_edit(text, cursor, "delete")
        self.assertEqual((text, cursor), ("helo", 3))
        text, cursor = cli_main._console_apply_edit("hello world", 11, "word_backspace")
        self.assertEqual((text, cursor), ("hello ", 6))
        text, cursor = cli_main._console_apply_edit("hello world", 5, "clear_to_end")
        self.assertEqual((text, cursor), ("hello", 5))
        text, cursor = cli_main._console_apply_edit("hello", 3, "clear")
        self.assertEqual((text, cursor), ("", 0))

    def test_console_apply_history_supports_up_down_and_draft_restore(self) -> None:
        history = ["first", "second", "third"]
        text, cursor, index, draft = cli_main._console_apply_history(
            "draft",
            5,
            history=history,
            history_index=None,
            history_draft="",
            event="up",
        )
        self.assertEqual((text, cursor, index, draft), ("third", 5, 2, "draft"))

        text, cursor, index, draft = cli_main._console_apply_history(
            text,
            cursor,
            history=history,
            history_index=index,
            history_draft=draft,
            event="up",
        )
        self.assertEqual((text, cursor, index, draft), ("second", 6, 1, "draft"))

        text, cursor, index, draft = cli_main._console_apply_history(
            text,
            cursor,
            history=history,
            history_index=index,
            history_draft=draft,
            event="down",
        )
        self.assertEqual((text, cursor, index, draft), ("third", 5, 2, "draft"))

        text, cursor, index, draft = cli_main._console_apply_history(
            text,
            cursor,
            history=history,
            history_index=index,
            history_draft=draft,
            event="down",
        )
        self.assertEqual((text, cursor, index, draft), ("draft", 5, None, "draft"))

    def test_render_console_snapshot_shows_cursor_preview(self) -> None:
        payload = {
            "workspace": "user_default",
            "strict_local": True,
            "inference_backend": "llama_cpp",
            "operations_dashboard": {"severity": "info", "current_focus": "none"},
            "operations_alert_policy": {},
            "operations_event_stream": {"severity": "info", "items": []},
            "train_queue": {"count": 0},
        }
        console = Console(record=True, width=180)
        console.print(
            build_console_renderable(
                payload,
                workspace="user_default",
                interactive=True,
                mode="chat",
                prompt_label="chat>",
                input_active=True,
                input_text="hello",
                input_cursor=2,
                feedback="interactive mode ready",
                ops_refresh_state="live",
                ops_age_seconds=0.2,
            )
        )
        rendered = console.export_text()
        self.assertIn("he", rendered)
        self.assertIn("llo", rendered)
        self.assertIn("COMPOSE", rendered)
        self.assertIn("EDIT", rendered)
        self.assertIn("EDITABLE", rendered)
        self.assertIn("SNAPSHOT", rendered)
        self.assertIn("c=2/5", rendered)
        self.assertIn("o=send", rendered)
        self.assertIn("0.2s", rendered)

    def test_render_console_snapshot_shows_focus_aware_command_placeholder(self) -> None:
        payload = {
            "workspace": "user_default",
            "strict_local": True,
            "inference_backend": "llama_cpp",
            "operations_dashboard": {"severity": "warning", "current_focus": "queue_backlog"},
            "operations_alert_policy": {},
            "operations_event_stream": {"severity": "warning", "items": []},
            "train_queue": {"count": 2},
        }
        console = Console(record=True, width=120)
        console.print(
            build_console_renderable(
                payload,
                workspace="user_default",
                interactive=True,
                mode="command",
                prompt_label="cmd>",
                input_active=True,
                input_text="",
                input_cursor=0,
                feedback="interactive mode ready",
                ops_refresh_state="cached",
                ops_age_seconds=1.2,
            )
        )
        rendered = console.export_text()
        self.assertIn("Type /do or /see", rendered)
        self.assertEqual(_prompt_mode_help("command", focus="queue_backlog"), "process")
        self.assertEqual(_prompt_target_hint("command", focus="queue_backlog"), "process")
        self.assertIn("QUEUE WAITING", rendered)
        self.assertIn("EXECUTION", rendered)

        queue_waiting_payload = {
            "workspace": "user_default",
            "strict_local": True,
            "inference_backend": "llama_cpp",
            "operations_dashboard": {"severity": "info", "current_focus": "queue_waiting_execution"},
            "operations_alert_policy": {
                "required_action": "process_next_queue_item",
                "secondary_action": "inspect_auto_train_trigger",
            },
            "operations_event_stream": {"severity": "info", "items": []},
            "train_queue": {"count": 2},
        }
        console = Console(record=True, width=120)
        console.print(
            build_console_renderable(
                queue_waiting_payload,
                workspace="user_default",
                interactive=True,
                mode="command",
                prompt_label="cmd>",
                input_active=True,
                input_text="",
                input_cursor=0,
                feedback="interactive mode ready",
                ops_refresh_state="live",
                ops_age_seconds=0.2,
            )
        )
        rendered = console.export_text()
        self.assertIn("Type /do or /see", rendered)
        self.assertEqual(_prompt_mode_help("command", focus="queue_waiting_execution"), "process")
        self.assertEqual(_prompt_target_hint("command", focus="queue_waiting_execution"), "process")

    def test_render_console_snapshot_shows_runtime_badges_in_event_stream(self) -> None:
        payload = {
            "workspace": "user_default",
            "strict_local": True,
            "inference_backend": "llama_cpp",
            "latest_adapter": {"version": "20260325-001", "state": "promoted"},
            "operations_overview": {"summary_line": "auto=disabled | queue=0 | runner=idle | daemon-health=stopped"},
            "operations_console": {
                "runtime_stability_summary": {
                    "runner_lock_state": "stale",
                    "runner_active": False,
                    "runner_stop_requested": False,
                    "daemon_health_state": "blocked",
                    "daemon_heartbeat_state": "stale",
                    "daemon_lease_state": "expired",
                    "daemon_restart_policy_state": "backoff",
                    "daemon_recovery_action": "manual_recover",
                }
            },
            "operations_dashboard": {"severity": "warning", "status": "attention", "current_focus": "daemon_restart_backoff"},
            "operations_alert_policy": {
                "required_action": "inspect_daemon_restart_policy",
                "action_priority": "p1",
                "escalation_mode": "review_soon",
                "requires_human_review": True,
                "auto_remediation_allowed": False,
                "requires_immediate_action": False,
                "operator_guidance": "inspect daemon restart attempts and backoff before requesting another recovery cycle",
            },
            "operations_event_stream": {
                "severity": "warning",
                "status": "attention",
                "attention_needed": True,
                "latest_source": "daemon",
                "alert_count": 2,
                "dashboard": {"current_focus": "daemon_restart_backoff"},
                "items": [
                    {
                        "source": "daemon",
                        "event": "steady_state_alert",
                        "severity": "warning",
                        "reason": "daemon_restart_backoff",
                    },
                    {
                        "source": "runner",
                        "event": "steady_state_alert",
                        "severity": "warning",
                        "reason": "stale_runner_lock",
                    },
                ],
            },
            "train_queue": {"count": 0},
        }
        console = Console(record=True, width=100)
        console.print(_event_stream_panel(payload))
        rendered = console.export_text()
        self.assertIn("DAEMON RESTART BACKOFF", rendered)
        self.assertIn("INSPECT DAEMON RESTART POLICY", rendered)
        self.assertIn("BACKOFF", rendered)
        self.assertIn("INSPECT DAEMON RESTART POLICY", rendered)
        self.assertIn("STALE RUNNER LOCK", rendered)
        self.assertIn("INSPECT WORKER STALE LOCK", rendered)

    def test_render_console_snapshot_shows_trigger_category_badges_in_event_stream(self) -> None:
        payload = {
            "workspace": "user_default",
            "strict_local": True,
            "inference_backend": "llama_cpp",
            "operations_overview": {
                "summary_line": "auto=blocked | gate=samples=2/50",
                "trigger_blocked_category": "data",
            },
            "operations_console": {
                "trigger_blocked_category": "data",
            },
            "operations_dashboard": {
                "severity": "warning",
                "status": "attention",
                "current_focus": "insufficient_new_signal_samples",
            },
            "operations_alert_policy": {
                "required_action": "collect_more_signal_samples",
                "action_priority": "p1",
                "escalation_mode": "review_soon",
            },
            "operations_event_stream": {
                "severity": "warning",
                "status": "attention",
                "attention_needed": True,
                "latest_source": "trigger",
                "alert_count": 1,
                "dashboard": {"current_focus": "insufficient_new_signal_samples"},
                "items": [
                    {
                        "source": "trigger",
                        "event": "blocked",
                        "severity": "warning",
                        "reason": "insufficient_new_signal_samples",
                    }
                ],
            },
            "train_queue": {"count": 0},
        }
        console = Console(record=True, width=100)
        console.print(_event_stream_panel(payload))
        rendered = console.export_text()
        self.assertIn("DATA", rendered)
        self.assertIn("insufficient_new_signal_samples", rendered)
        self.assertIn("COLLECT MORE SIGNAL SAMPLES", rendered)

    def test_render_event_stream_panel_prefers_inspection_summary_for_generic_monitor(self) -> None:
        inspection_summary = (
            "current_focus=queue_waiting_execution | "
            "required_action=process_next_queue_item | "
            "next_actions=process_next_queue_item,inspect_auto_train_trigger"
        )
        payload = {
            "operations_overview": {
                "inspection_summary_line": inspection_summary,
            },
            "operations_dashboard": {
                "severity": "info",
                "status": "healthy",
                "current_focus": "queue_waiting_execution",
            },
            "operations_alert_policy": {
                "required_action": "process_next_queue_item",
                "action_priority": "p2",
                "escalation_mode": "monitor",
            },
            "operations_event_stream": {
                "severity": "info",
                "status": "healthy",
                "attention_needed": False,
                "latest_source": "queue",
                "alert_count": 0,
                "dashboard": {
                    "current_focus": "queue_waiting_execution",
                    "inspection_summary_line": inspection_summary,
                },
                "inspection_summary_line": inspection_summary,
                "items": [],
            },
        }
        console = Console(record=True, width=110)
        console.print(_event_stream_panel(payload))
        rendered = console.export_text()
        self.assertIn("I", rendered)
        self.assertIn(_compact_text(inspection_summary, max_len=42), rendered)

    def test_render_console_snapshot_shows_trigger_category_badges_in_prompt_and_footer(self) -> None:
        payload = {
            "workspace": "user_default",
            "strict_local": True,
            "inference_backend": "llama_cpp",
            "operations_overview": {
                "trigger_blocked_category": "data",
            },
            "operations_console": {
                "trigger_blocked_category": "data",
            },
            "operations_dashboard": {
                "severity": "warning",
                "status": "attention",
                "current_focus": "insufficient_new_signal_samples",
            },
            "operations_alert_policy": {
                "required_action": "collect_more_signal_samples",
                "primary_action": "collect_more_signal_samples",
                "action_priority": "p1",
                "escalation_mode": "review_soon",
                "remediation_mode": "manual_review",
                "requires_human_review": True,
                "auto_remediation_allowed": False,
                "requires_immediate_action": False,
            },
            "operations_event_stream": {"severity": "warning", "items": []},
            "train_queue": {"count": 0},
        }
        console = Console(record=True, width=220)
        console.print(
            _prompt_panel(
                mode="command",
                prompt_label="cmd>",
                input_active=True,
                input_text="",
                input_cursor=0,
                ops_refresh_state="live",
                ops_age_seconds=0.2,
                focus="insufficient_new_signal_samples",
                payload=payload,
            )
        )
        console.print(
            _footer_digest(
                payload,
                interactive=True,
                mode="command",
                ops_refresh_state="live",
                ops_age_seconds=0.2,
            )
        )
        rendered = console.export_text()
        self.assertIn("tg=", rendered)
        self.assertIn("DATA", rendered)

    def test_render_console_snapshot_shows_runtime_badges_in_prompt_and_footer(self) -> None:
        payload = {
            "workspace": "user_default",
            "strict_local": True,
            "inference_backend": "llama_cpp",
            "operations_dashboard": {
                "severity": "warning",
                "status": "attention",
                "current_focus": "daemon_restart_backoff",
            },
            "operations_alert_policy": {
                "required_action": "inspect_daemon_restart_policy",
                "primary_action": "inspect_daemon_restart_policy",
                "action_priority": "p1",
                "escalation_mode": "review_soon",
                "remediation_mode": "manual_review",
                "requires_human_review": True,
                "auto_remediation_allowed": False,
                "requires_immediate_action": False,
            },
            "operations_event_stream": {"severity": "warning", "items": []},
            "train_queue": {"count": 0},
        }
        console = Console(record=True, width=120)
        console.print(
            build_console_renderable(
                payload,
                workspace="user_default",
                interactive=True,
                mode="command",
                prompt_label="cmd>",
                input_active=True,
                input_text="",
                input_cursor=0,
                feedback="interactive mode ready",
                ops_refresh_state="live",
                ops_age_seconds=0.2,
            )
        )
        rendered = console.export_text()
        self.assertIn("INSPECT DAEMON RESTART POLICY", rendered)
        self.assertIn("BACKOFF", rendered)

    def test_render_console_snapshot_shows_review_action_guidance(self) -> None:
        payload = {
            "workspace": "user_default",
            "strict_local": True,
            "inference_backend": "llama_cpp",
            "operations_dashboard": {
                "severity": "warning",
                "status": "attention",
                "current_focus": "queue_pending_review",
            },
            "operations_alert_policy": {
                "required_action": "review_queue_confirmation",
                "primary_action": "review_queue_confirmation",
                "action_priority": "p1",
                "escalation_mode": "review_soon",
                "remediation_mode": "manual_review",
                "requires_human_review": True,
                "auto_remediation_allowed": False,
                "requires_immediate_action": False,
                "operator_guidance": "review the next queued training request",
            },
            "operations_event_stream": {"severity": "warning", "items": []},
            "train_queue": {"count": 1},
        }
        console = Console(record=True, width=80)
        console.print(_chat_help_panel(payload, interactive=True))
        rendered = console.export_text()
        self.assertIn("do: /approve /reject", rendered)
        self.assertIn("see: /gate /trigger", rendered)
        self.assertIn("cmd=/do /see", rendered)

    def test_render_help_panel_shows_runtime_action_guidance(self) -> None:
        payload = {
            "latest_adapter": {"version": "20260325-001"},
            "train_queue": {"count": 1},
            "operations_dashboard": {"current_focus": "daemon_stale"},
        }
        console = Console(record=True, width=80)
        console.print(_chat_help_panel(payload, interactive=True))
        rendered = console.export_text()
        self.assertIn("do: /recover daemon", rendered)
        self.assertIn("see: /runtime /daemon", rendered)
        self.assertIn("cmd=/do /see", rendered)

    def test_render_help_panel_shows_candidate_action_guidance(self) -> None:
        payload = {
            "latest_adapter": {"version": "20260325-001"},
            "train_queue": {"count": 0},
            "operations_dashboard": {"current_focus": "candidate_ready_for_promotion"},
        }
        console = Console(record=True, width=80)
        console.print(_chat_help_panel(payload, interactive=True))
        rendered = console.export_text()
        self.assertIn("do: /promote", rendered)
        self.assertIn("see: /candidate /cand sum", rendered)
        self.assertIn("cmd=/do /see", rendered)

    def test_render_help_panel_shows_candidate_archive_guidance(self) -> None:
        payload = {
            "latest_adapter": {"version": "20260325-001"},
            "train_queue": {"count": 0},
            "operations_dashboard": {"current_focus": "candidate_idle"},
            "candidate_summary": {
                "candidate_can_promote": False,
                "candidate_can_archive": True,
            },
        }
        console = Console(record=True, width=80)
        console.print(_chat_help_panel(payload, interactive=True))
        rendered = console.export_text()
        self.assertIn("do: /archive", rendered)
        self.assertIn("see: /candidate /cand sum", rendered)

    def test_render_help_panel_shows_trigger_gate_guidance(self) -> None:
        payload = {
            "latest_adapter": {"version": "20260325-001"},
            "train_queue": {"count": 0},
            "operations_dashboard": {"current_focus": "insufficient_new_signal_samples"},
            "operations_console": {
                "trigger_threshold_summary": {
                    "summary_line": "samples=2/50 | holdout=ready | interval=1.0/7d | cooldown=ok | backoff=ok",
                },
                "trigger_blocked_summary": "reason=insufficient_new_signal_samples | action=collect_more_signal_samples | category=data",
            },
        }
        console = Console(record=True, width=90)
        console.print(_chat_help_panel(payload, interactive=True))
        rendered = console.export_text()
        self.assertIn("gate=samples=2/50", rendered)
        self.assertIn("do: /gate", rendered)
        self.assertIn("see: /trigger /policy", rendered)

    def test_render_operations_panel_shows_primary_and_secondary_actions(self) -> None:
        payload = {
            "operations_dashboard": {
                "severity": "warning",
                "current_focus": "queue_pending_review",
            },
            "operations_alert_policy": {
                "required_action": "review_queue_confirmation",
                "primary_action": "review_queue_confirmation",
                "action_priority": "p1",
                "escalation_mode": "review_soon",
                "operator_guidance": "review the next queued training request",
            },
            "operations_console": {
                "trigger_blocked_reason": "queue_pending_review",
                "trigger_blocked_action": "review_queue_confirmation",
                "trigger_blocked_category": "queue",
                "queue_review_policy": {
                    "review_mode": "manual_review",
                    "queue_entry_mode": "awaiting_confirmation",
                    "next_action": "review_queue_confirmation",
                },
            },
        }
        console = Console(record=True, width=84)
        console.print(_operations_panel(payload))
        rendered = console.export_text()
        self.assertIn("Do", rendered)
        self.assertIn("/approve /reject", rendered)
        self.assertIn("See", rendered)
        self.assertIn("/gate /trigger", rendered)

    def test_render_operations_panel_uses_trigger_gate_summary_for_blockers(self) -> None:
        payload = {
            "operations_dashboard": {
                "severity": "warning",
                "current_focus": "insufficient_new_signal_samples",
            },
            "operations_alert_policy": {
                "required_action": "collect_more_signal_samples",
                "primary_action": "collect_more_signal_samples",
                "action_priority": "p1",
                "escalation_mode": "review_soon",
                "operator_guidance": "collect more signal samples before auto-train can continue",
            },
            "operations_console": {
                "trigger_blocked_reason": "insufficient_new_signal_samples",
                "trigger_blocked_action": "collect_more_signal_samples",
                "trigger_blocked_category": "data",
                "trigger_threshold_summary": {
                    "eligible_signal_train_samples": 2,
                    "min_new_samples": 50,
                    "holdout_ready": True,
                    "interval_elapsed": False,
                    "cooldown_elapsed": True,
                    "failure_backoff_elapsed": True,
                    "summary_line": "samples=2/50 | holdout=ready | interval=1.0/7d | cooldown=ok | backoff=ok",
                },
                "auto_train_policy": {
                    "queue_entry_mode": "deferred",
                    "evaluation_mode": "auto_evaluate",
                    "promotion_mode": "manual",
                    "stop_stage": "trigger",
                },
                "queue_review_policy": {
                    "review_mode": "auto_queue",
                    "queue_entry_mode": "deferred",
                    "next_action": "await_signal_trigger",
                },
                "runtime_stability_summary": {
                    "runner_lock_state": "idle",
                    "runner_active": False,
                    "runner_stop_requested": False,
                    "daemon_health_state": "stopped",
                    "daemon_heartbeat_state": "idle",
                    "daemon_lease_state": "idle",
                    "daemon_restart_policy_state": "ready",
                    "daemon_recovery_action": "none",
                },
            },
            "operations_overview": {
                "summary_line": "auto=blocked | gate=samples=2/50",
            },
        }
        console = Console(record=True, width=92)
        console.print(_operations_panel(payload))
        rendered = console.export_text()
        self.assertIn("DATA", rendered)
        self.assertIn("collect_more_signal...", rendered)
        self.assertIn("samples=2/50", rendered)

    def test_render_operations_panel_prefers_inspection_summary_for_generic_monitor(self) -> None:
        inspection_summary = (
            "current_focus=queue_waiting_execution | "
            "required_action=process_next_queue_item | "
            "next_actions=process_next_queue_item,inspect_auto_train_trigger"
        )
        payload = {
            "operations_dashboard": {
                "severity": "info",
                "current_focus": "queue_waiting_execution",
            },
            "operations_alert_policy": {
                "required_action": "process_next_queue_item",
                "primary_action": "process_next_queue_item",
                "action_priority": "p2",
                "escalation_mode": "monitor",
                "operator_guidance": "process the next queued item",
            },
            "operations_console": {
                "queue_review_policy": {
                    "review_mode": "auto_queue",
                    "queue_entry_mode": "inline_execute",
                    "next_action": "process_next_queue_item",
                },
            },
            "operations_overview": {
                "summary_line": "auto=disabled | queue=1 | runner=idle | daemon-health=stopped",
                "inspection_summary_line": inspection_summary,
            },
        }
        console = Console(record=True, width=96)
        console.print(_operations_panel(payload))
        rendered = console.export_text()
        self.assertIn(_compact_text(inspection_summary, max_len=30), rendered)
        self.assertNotIn("auto=disabled | queue=1", rendered)

    def test_console_focus_actions_supports_rollback_candidate(self) -> None:
        self.assertEqual(
            cli_main._console_focus_actions(
                {
                    "operations_dashboard": {"current_focus": "candidate_idle"},
                    "operations_alert_policy": {
                        "required_action": "rollback_candidate",
                        "secondary_actions": ["inspect_candidate_status"],
                    },
                }
            ),
            {
                "primary_label": "/rollback",
                "primary_exec": "rollback",
                "secondary_label": "/candidate",
                "secondary_exec": "candidate",
            },
        )

    def test_console_focus_actions_supports_inspect_candidate_gate(self) -> None:
        # inspect_candidate_gate maps to /gate with secondary from alert_policy
        result = cli_main._console_focus_actions(
            {
                "operations_dashboard": {"current_focus": "candidate_idle"},
                "operations_alert_policy": {
                    "required_action": "inspect_candidate_gate",
                    "secondary_actions": ["inspect_candidate_timeline"],
                },
            }
        )
        self.assertEqual(result["primary_label"], "/gate")
        self.assertEqual(result["primary_exec"], "gate")
        # Secondary actions come from the alert_policy secondary_actions mapping
        self.assertEqual(result["secondary_label"], "/cand tl")
        self.assertEqual(result["secondary_exec"], "cand tl")

    def test_console_command_output_supports_trigger_train_alias(self) -> None:
        class FakeService:
            def retry_auto_train_trigger(self, workspace: str | None = None):
                return {
                    "workspace": workspace or "user_default",
                    "auto_train_trigger": {"state": "blocked", "reason": "queue_pending_review"},
                    "operations_dashboard": {"severity": "warning", "status": "attention", "current_focus": "queue_pending_review"},
                }

        payload = {"workspace": "user_default", "operations_dashboard": {"current_focus": "none"}}

        for command in ("retry", "trigger-train", "trigger train"):
            text, action, updates = cli_main._console_command_output(
                command,
                payload=payload,
                workspace="user_default",
                service=FakeService(),
                current_workspace="user_default",
                mode="command",
                model="local",
                adapter="latest",
                temperature=0.7,
                max_tokens=None,
                real_local=False,
                refresh_seconds=2.0,
            )
            self.assertEqual(action, "retry-trigger")
            self.assertIsNone(updates)
            self.assertIn("auto train trigger", (text or "").lower())

    def test_console_help_shows_new_commands(self) -> None:
        help_text = cli_main._console_help_text()
        self.assertIn("/trigger-train", help_text)
        self.assertIn("/rollback [version]", help_text)
        self.assertIn("/approve [note]", help_text)
        self.assertIn("/approve <id> [note]", help_text)
        self.assertIn("/reject [note]", help_text)
        self.assertIn("/reject <id> [note]", help_text)

    def test_console_help_shows_high_frequency_commands(self) -> None:
        help_text = cli_main._console_help_text()
        self.assertIn("/list", help_text)
        self.assertIn("/generate", help_text)
        self.assertIn("/dpo", help_text)
        self.assertIn("/again", help_text)
        self.assertIn("/fix", help_text)

    def test_console_command_fix_returns_edited_text_in_updates(self) -> None:
        last_interaction = {
            "session_id": "s1",
            "request_id": "r1",
            "user_message": "hello",
            "assistant_message": "hi",
            "response_time_seconds": 0.5,
            "adapter_version": "latest",
        }
        output, action, updates = cli_main._console_command_output(
            "fix corrected text",
            payload={},
            workspace="user_default",
            service=None,
            current_workspace="user_default",
            mode="chat",
            model="local",
            adapter="latest",
            temperature=0.7,
            max_tokens=None,
            real_local=False,
            refresh_seconds=2.0,
            last_interaction=last_interaction,
        )
        self.assertEqual(action, "fix")
        self.assertIsNotNone(updates)
        self.assertEqual(updates.get("edited_text"), "corrected text")
        self.assertIn("Submitted edit", output or "")

    def test_console_command_again_returns_regenerate_flag(self) -> None:
        last_interaction = {
            "session_id": "s1",
            "request_id": "r1",
            "user_message": "hello",
            "assistant_message": "hi",
            "response_time_seconds": 0.5,
            "adapter_version": "latest",
        }
        output, action, updates = cli_main._console_command_output(
            "again",
            payload={},
            workspace="user_default",
            service=None,
            current_workspace="user_default",
            mode="chat",
            model="local",
            adapter="latest",
            temperature=0.7,
            max_tokens=None,
            real_local=False,
            refresh_seconds=2.0,
            last_interaction=last_interaction,
        )
        self.assertEqual(action, "again")
        self.assertIsNotNone(updates)
        self.assertTrue(updates.get("regenerate"))
        self.assertIn("regeneration", output or "")

    def test_console_command_list_adapters_when_handler_available(self) -> None:
        class FakeService:
            def list_versions(self, workspace=None, limit=20):
                return {
                    "versions": [
                        {"version": "v1", "state": "promoted", "latest": True, "num_samples": 10, "artifact_format": "peft_lora"},
                    ]
                }

        output, action, updates = cli_main._console_command_output(
            "list",
            payload={},
            workspace="user_default",
            service=FakeService(),
            current_workspace="user_default",
            mode="command",
            model="local",
            adapter="latest",
            temperature=0.7,
            max_tokens=None,
            real_local=False,
            refresh_seconds=2.0,
        )
        self.assertEqual(action, "adapter-list")
        self.assertIn("v1", output or "")

    def test_console_command_generate_when_handler_available(self) -> None:
        class FakeService:
            def generate(self, scenario, style, num_samples, workspace=None):
                return {"status": "ok", "num_samples": num_samples, "scenario": scenario}

        output, action, updates = cli_main._console_command_output(
            "generate life-coach friendly",
            payload={},
            workspace="user_default",
            service=FakeService(),
            current_workspace="user_default",
            mode="command",
            model="local",
            adapter="latest",
            temperature=0.7,
            max_tokens=None,
            real_local=False,
            refresh_seconds=2.0,
        )
        self.assertEqual(action, "generate")
        self.assertIsNotNone(output)
        self.assertTrue(len(output or "") > 0)

    def test_console_command_dpo_when_handler_available(self) -> None:
        class FakeService:
            def train_dpo(self, workspace=None):
                return {"status": "completed", "version": "dpo-v1"}

        output, action, updates = cli_main._console_command_output(
            "dpo",
            payload={},
            workspace="user_default",
            service=FakeService(),
            current_workspace="user_default",
            mode="command",
            model="local",
            adapter="latest",
            temperature=0.7,
            max_tokens=None,
            real_local=False,
            refresh_seconds=2.0,
        )
        self.assertEqual(action, "dpo")
        self.assertIsNotNone(output)

    def test_chat_help_panel_shows_trigger_train_action_when_threshold_met(self) -> None:
        payload = {
            "latest_adapter": {"version": "20260325-001", "state": "promoted"},
            "train_queue": {"count": 0},
            "operations_dashboard": {"current_focus": "policy_requires_auto_evaluate"},
            "operations_console": {
                "trigger_threshold_summary": {
                    "min_new_samples": 50,
                    "eligible_signal_train_samples": 75,
                }
            },
        }
        panel = _chat_help_panel(payload, interactive=True)
        panel_text = str(panel.renderable)
        self.assertIn("/do trigger-train", panel_text)
        self.assertIn("ready=75/50", panel_text)


if __name__ == "__main__":
    unittest.main()
