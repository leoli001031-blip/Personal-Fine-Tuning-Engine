from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for package_dir in ("pfe-core", "pfe-cli", "pfe-server"):
    package_path = str(ROOT / package_dir)
    if package_path not in os.sys.path:
        os.sys.path.insert(0, package_path)

from pfe_server.app import build_serve_plan, smoke_test_request


class ServerHttpSmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.previous_home = os.environ.get("PFE_HOME")
        self.pfe_home = Path(self.tempdir.name) / ".pfe"
        os.environ["PFE_HOME"] = str(self.pfe_home)
        self.plan = build_serve_plan(workspace=str(self.pfe_home), dry_run=False)
        self.app = self.plan.app

    def tearDown(self) -> None:
        if self.previous_home is None:
            os.environ.pop("PFE_HOME", None)
        else:
            os.environ["PFE_HOME"] = self.previous_home
        self.tempdir.cleanup()

    def _smoke(self, path: str, **kwargs):
        return asyncio.run(smoke_test_request(self.app, path=path, **kwargs))

    def test_healthz_returns_ok(self) -> None:
        result = self._smoke("/healthz")
        self.assertEqual(result["status_code"], 200)
        self.assertEqual(result["body"]["status"], "ok")
        self.assertIn("content-type", result["headers"])

    def test_root_serves_chat_frontend(self) -> None:
        result = self._smoke("/")
        self.assertEqual(result["status_code"], 200)
        self.assertIn("text/html", result["headers"].get("content-type", ""))
        self.assertIn("PFE Local Chat", result["text"])
        self.assertIn("/v1/chat/completions", result["text"])
        self.assertIn("/pfe/status", result["text"])

    def test_status_returns_runtime_snapshot(self) -> None:
        result = self._smoke("/pfe/status", query_params={"detail": "full"})
        self.assertEqual(result["status_code"], 200)
        body = result["body"]
        self.assertTrue(body["strict_local"])
        self.assertIn("provider", body)
        self.assertIn("runtime", body)
        self.assertIn("sample_counts", body)
        self.assertEqual(result["request"]["query_params"], {"detail": "full"})
        self.assertIn("inference", body["metadata"])
        self.assertIn("export", body["metadata"])
        self.assertIn("trainer", body["metadata"])
        self.assertIn("lifecycle", body["metadata"])
        self.assertIn("server_runtime", body["metadata"])
        self.assertIn("artifact_format", body["metadata"]["export"])
        self.assertIn("recommended_backend", body["metadata"]["export"])
        self.assertIn("requires_export_step", body["metadata"]["export"])
        self.assertIn("export_artifact_path", body["metadata"]["export"])
        self.assertIn("export_artifact_valid", body["metadata"]["export"])
        self.assertIn("export_artifact_size_bytes", body["metadata"]["export"])
        self.assertIn("artifact_directory", body["metadata"]["trainer"])
        self.assertIn("output_dir", body["metadata"]["trainer"])
        self.assertIn("recommended_backend", body["metadata"]["trainer"])
        self.assertIn("requires_export_step", body["metadata"]["trainer"])
        self.assertIn("export_artifact_summary", body["metadata"]["trainer"])
        self.assertIn("placeholder_files", body["metadata"]["export"])
        self.assertIn("materialized", body["metadata"]["export"])
        self.assertIn("write_state", body["metadata"]["export"])
        self.assertIn(body["metadata"]["export"]["write_state"], {"materialized", "pending"})
        self.assertIn("probe_paths", body["metadata"]["server_runtime"])
        self.assertIn("/pfe/status", {item["path"] for item in body["metadata"]["server_runtime"]["probe_paths"]})
        self.assertIn("launch_mode", body["metadata"]["server_runtime"])
        self.assertIn(body["metadata"]["server_runtime"]["launch_mode"], {"dry_run", "uvicorn.run"})
        self.assertIn("probe_status", body["metadata"]["server_runtime"])
        self.assertIn(body["metadata"]["server_runtime"]["probe_status"]["state"], {"ok", "degraded", "skipped", "deferred"})
        self.assertIn("last_serve_check", body["metadata"]["server_runtime"])
        self.assertIn("serve_summary", body["metadata"]["server_runtime"])
        self.assertIn("launch_state", body["metadata"]["server_runtime"])
        self.assertIn("probe_summary", body["metadata"]["server_runtime"]["serve_summary"])
        self.assertIn("checked_paths", body["metadata"]["server_runtime"]["serve_summary"])
        self.assertIn("train", body["metadata"]["lifecycle"])
        self.assertIn("eval", body["metadata"]["lifecycle"])
        self.assertIn("promotion", body["metadata"]["lifecycle"])
        self.assertIn("serve", body["metadata"]["lifecycle"])
        self.assertIn(body["metadata"]["lifecycle"]["train"]["state"], {"idle", "ready"})
        self.assertIn(body["metadata"]["lifecycle"]["eval"]["state"], {"ready", "waiting_for_holdout"})
        self.assertIn("latest_adapter_version", body["metadata"]["lifecycle"]["promotion"])
        self.assertIn("last_check", body["metadata"]["lifecycle"]["serve"])
        self.assertIn("launch_mode", body["metadata"]["lifecycle"]["serve"])

    def test_signal_accepts_local_management_event(self) -> None:
        result = self._smoke(
            "/pfe/signal",
            method="POST",
            body={
                "event_id": "evt-http-smoke-1",
                "request_id": "req-http-smoke-1",
                "session_id": "sess-http-smoke-1",
                "source_event_id": "evt-source-1",
                "source_event_ids": ["evt-source-1", "evt-http-smoke-1"],
                "event_type": "accept",
                "user_input": "我今天有点焦虑",
                "model_output": "我们先把任务拆成三个最小步骤。",
                "user_action": {"type": "accept"},
                "metadata": {"scenario": "life-coach"},
            },
        )
        self.assertEqual(result["status_code"], 200)
        body = result["body"]
        self.assertTrue(body["stored"])
        self.assertEqual(body["request_id"], "req-http-smoke-1")
        self.assertEqual(result["request"]["method"], "POST")
        self.assertIn("content-type", result["headers"])

    def test_serve_plan_exposes_command_and_runner_info(self) -> None:
        plan = build_serve_plan(workspace=str(self.pfe_home))
        self.assertEqual(plan.runtime.app_target, "pfe_server.app:app")
        self.assertEqual(plan.runtime.host, "127.0.0.1")
        self.assertEqual(plan.runtime.port, 8921)
        self.assertEqual(plan.runner["target"], "pfe_server.app:app")
        self.assertIn("uvicorn", plan.command)
        self.assertTrue(plan.runtime.dry_run)
        self.assertIsInstance(plan.runtime.uvicorn_available, bool)
        self.assertIn("kind", plan.runner)
        self.assertIn("notes", plan.runtime.model_dump())
        self.assertIn("dry_run", plan.runtime.model_dump())
        self.assertIn("command", plan.runtime.model_dump())

    def test_serve_plan_false_includes_runtime_probe(self) -> None:
        plan = build_serve_plan(workspace=str(self.pfe_home), dry_run=False)
        self.assertFalse(plan.runtime.dry_run)
        self.assertEqual(plan.runner["target"], "pfe_server.app:app")
        self.assertIn(plan.runner["kind"], {"uvicorn.run", "dry_run"})
        self.assertIn("launch_mode", plan.runtime_probe)
        self.assertIn(plan.runtime_probe["launch_mode"], {"dry_run", "uvicorn.run"})
        self.assertIn("probe_status", plan.runtime_probe)
        self.assertIn(plan.runtime_probe["probe_status"]["state"], {"ok", "degraded", "deferred"})
        self.assertIn("last_serve_check", plan.runtime_probe)
        self.assertIn(plan.runtime_probe["last_serve_check"].get("path", "/pfe/status"), {"/pfe/status", None})
        self.assertIn("serve_summary", plan.runtime_probe)
        self.assertIn("launch_state", plan.runtime_probe)
        self.assertIn("probe_summary", plan.runtime_probe["serve_summary"])
        self.assertIn("checked_paths", plan.runtime_probe["serve_summary"])
        self.assertIn("before", plan.runtime_probe["launch_state"])
        self.assertIn("after", plan.runtime_probe["launch_state"])
        self.assertIn("/pfe/status", {item["path"] for item in plan.runtime_probe["probe_paths"]})
        self.assertIn("command", plan.runtime_probe)
        self.assertIn("runner", plan.runtime_probe)
        self.assertIn("probe_status", plan.runtime_probe["serve_summary"])
        self.assertIn("probe_state", plan.runtime_probe["serve_summary"])

    def test_feedback_accepts_explicit_user_feedback(self) -> None:
        """Test that feedback endpoint accepts explicit user feedback signals."""
        # Test accept feedback
        result = self._smoke(
            "/pfe/feedback",
            method="POST",
            body={
                "session_id": "test-session",
                "request_id": "test-request-1",
                "action": "accept",
                "user_message": "Hello",
                "assistant_message": "Hi there!",
                "response_time_seconds": 3.0,
            },
        )
        self.assertEqual(result["status_code"], 200)
        body = result["body"]
        self.assertTrue(body["success"])
        self.assertEqual(body["signal_type"], "accept")
        # Explicit accept feedback should follow the normalized ChatCollector semantics.
        self.assertEqual(body["confidence"], 0.9)
        self.assertEqual(body["session_id"], "test-session")
        self.assertEqual(body["request_id"], "test-request-1")
        self.assertIn("pipeline_ingest", body["metadata"])
        self.assertEqual(body["metadata"]["pipeline_ingest"]["request_id"], "test-request-1")
        self.assertEqual(body["metadata"]["pipeline_ingest"]["session_id"], "test-session")
        self.assertIn("curation_state", body["metadata"]["pipeline_ingest"]["metadata"])
        self.assertIn("auto_train", body["metadata"]["pipeline_ingest"]["metadata"])

    def test_feedback_reject_signal(self) -> None:
        """Test reject feedback signal extraction."""
        result = self._smoke(
            "/pfe/feedback",
            method="POST",
            body={
                "session_id": "test-session",
                "request_id": "test-request-2",
                "action": "delete",
                "user_message": "Hello",
                "assistant_message": "Hi there!",
            },
        )
        self.assertEqual(result["status_code"], 200)
        body = result["body"]
        self.assertEqual(body["signal_type"], "reject")
        self.assertEqual(body["confidence"], 0.95)

    def test_feedback_edit_signal(self) -> None:
        """Test edit feedback signal extraction with edit distance calculation."""
        result = self._smoke(
            "/pfe/feedback",
            method="POST",
            body={
                "session_id": "test-session",
                "request_id": "test-request-3",
                "action": "edit",
                "user_message": "Hello",
                "assistant_message": "Hi there!",
                "edited_text": "Hello there!",
            },
        )
        self.assertEqual(result["status_code"], 200)
        body = result["body"]
        self.assertEqual(body["signal_type"], "edit")
        self.assertIn("metadata", body)

    def test_feedback_regenerate_signal(self) -> None:
        """Test regenerate feedback signal extraction."""
        result = self._smoke(
            "/pfe/feedback",
            method="POST",
            body={
                "session_id": "test-session",
                "request_id": "test-request-4",
                "action": "regenerate",
                "user_message": "Hello",
                "assistant_message": "Hi there!",
            },
        )
        self.assertEqual(result["status_code"], 200)
        body = result["body"]
        self.assertEqual(body["signal_type"], "regenerate")
        self.assertEqual(body["confidence"], 0.85)

    def test_chat_then_feedback_round_trip_preserves_ids_for_closed_loop(self) -> None:
        chat_result = self._smoke(
            "/v1/chat/completions",
            method="POST",
            body={
                "model": "local",
                "adapter_version": "latest",
                "messages": [
                    {
                        "role": "user",
                        "content": "My name is Alex. Please answer with short bullet points.",
                    }
                ],
            },
        )
        self.assertEqual(chat_result["status_code"], 200)
        chat_body = chat_result["body"]
        self.assertIn("session_id", chat_body)
        self.assertIn("request_id", chat_body)
        self.assertTrue(chat_body["session_id"])
        self.assertTrue(chat_body["request_id"])
        self.assertEqual(
            chat_body["metadata"]["signal_collection"]["session_id"],
            chat_body["session_id"],
        )
        self.assertEqual(
            chat_body["metadata"]["signal_collection"]["request_id"],
            chat_body["request_id"],
        )
        self.assertTrue(chat_body["metadata"]["signal_collection"]["interaction_stored"])

        feedback_result = self._smoke(
            "/pfe/feedback",
            method="POST",
            body={
                "session_id": chat_body["session_id"],
                "request_id": chat_body["request_id"],
                "action": "accept",
                "response_time_seconds": 2.5,
            },
        )
        self.assertEqual(feedback_result["status_code"], 200)
        feedback_body = feedback_result["body"]
        self.assertTrue(feedback_body["success"])
        self.assertEqual(feedback_body["session_id"], chat_body["session_id"])
        self.assertEqual(feedback_body["request_id"], chat_body["request_id"])
        self.assertEqual(feedback_body["signal_type"], "accept")
        self.assertGreaterEqual(feedback_body["metadata"]["signals_extracted"], 0)

    def test_chat_completion_requires_api_key_for_remote_clients_when_enabled(self) -> None:
        previous_allow_remote_access = self.app.state.pfe_services.security.allow_remote_access
        previous_auth_mode = self.app.state.pfe_services.security.auth_mode
        previous_api_key = os.environ.get("PFE_API_KEY")
        try:
            self.app.state.pfe_services.security.allow_remote_access = True
            self.app.state.pfe_services.security.auth_mode = "local_optional"
            os.environ.pop("PFE_API_KEY", None)

            remote_without_key = self._smoke(
                "/v1/chat/completions",
                method="POST",
                body={
                    "model": "local",
                    "messages": [{"role": "user", "content": "ping"}],
                },
                client_host="10.0.0.8",
            )
            self.assertEqual(remote_without_key["status_code"], 503)
            self.assertEqual(remote_without_key["body"]["code"], "api_key_not_configured")

            os.environ["PFE_API_KEY"] = "secret-remote-key"

            remote_without_header = self._smoke(
                "/v1/chat/completions",
                method="POST",
                body={
                    "model": "local",
                    "messages": [{"role": "user", "content": "ping"}],
                },
                client_host="10.0.0.8",
            )
            self.assertEqual(remote_without_header["status_code"], 401)
            self.assertEqual(remote_without_header["body"]["code"], "unauthorized")

            remote_with_header = self._smoke(
                "/v1/chat/completions",
                method="POST",
                body={
                    "model": "local",
                    "messages": [{"role": "user", "content": "ping"}],
                },
                headers={"Authorization": "Bearer secret-remote-key"},
                client_host="10.0.0.8",
            )
            self.assertEqual(remote_with_header["status_code"], 200)
            self.assertIn("session_id", remote_with_header["body"])
            self.assertIn("request_id", remote_with_header["body"])
        finally:
            self.app.state.pfe_services.security.allow_remote_access = previous_allow_remote_access
            self.app.state.pfe_services.security.auth_mode = previous_auth_mode
            if previous_api_key is None:
                os.environ.pop("PFE_API_KEY", None)
            else:
                os.environ["PFE_API_KEY"] = previous_api_key

    def test_chat_completion_requires_api_key_when_auth_mode_is_strict(self) -> None:
        previous_auth_mode = self.app.state.pfe_services.security.auth_mode
        previous_api_key = os.environ.get("PFE_API_KEY")
        try:
            self.app.state.pfe_services.security.auth_mode = "api_key_required"
            os.environ["PFE_API_KEY"] = "secret-local-key"

            local_without_key = self._smoke(
                "/v1/chat/completions",
                method="POST",
                body={
                    "model": "local",
                    "messages": [{"role": "user", "content": "ping"}],
                },
            )
            self.assertEqual(local_without_key["status_code"], 401)
            self.assertEqual(local_without_key["body"]["code"], "unauthorized")

            local_with_key = self._smoke(
                "/v1/chat/completions",
                method="POST",
                body={
                    "model": "local",
                    "messages": [{"role": "user", "content": "ping"}],
                },
                headers={"x-api-key": "secret-local-key"},
            )
            self.assertEqual(local_with_key["status_code"], 200)
            self.assertIn("session_id", local_with_key["body"])
            self.assertIn("request_id", local_with_key["body"])
        finally:
            self.app.state.pfe_services.security.auth_mode = previous_auth_mode
            if previous_api_key is None:
                os.environ.pop("PFE_API_KEY", None)
            else:
                os.environ["PFE_API_KEY"] = previous_api_key

    def test_feedback_requires_api_key_for_remote_clients_when_enabled(self) -> None:
        previous_allow_remote_access = self.app.state.pfe_services.security.allow_remote_access
        previous_auth_mode = self.app.state.pfe_services.security.auth_mode
        previous_api_key = os.environ.get("PFE_API_KEY")
        try:
            self.app.state.pfe_services.security.allow_remote_access = True
            self.app.state.pfe_services.security.auth_mode = "local_optional"
            os.environ.pop("PFE_API_KEY", None)

            remote_without_key = self._smoke(
                "/pfe/feedback",
                method="POST",
                body={
                    "session_id": "sess-remote",
                    "request_id": "req-remote",
                    "action": "accept",
                },
                client_host="10.0.0.8",
            )
            self.assertEqual(remote_without_key["status_code"], 503)
            self.assertEqual(remote_without_key["body"]["code"], "api_key_not_configured")
            self.assertIn("hint", remote_without_key["body"])

            os.environ["PFE_API_KEY"] = "secret-remote-key"

            remote_without_header = self._smoke(
                "/pfe/feedback",
                method="POST",
                body={
                    "session_id": "sess-remote",
                    "request_id": "req-remote",
                    "action": "accept",
                },
                client_host="10.0.0.8",
            )
            self.assertEqual(remote_without_header["status_code"], 401)
            self.assertEqual(remote_without_header["body"]["code"], "unauthorized")
            self.assertIn("hint", remote_without_header["body"])

            remote_with_header = self._smoke(
                "/pfe/feedback",
                method="POST",
                body={
                    "session_id": "sess-remote",
                    "request_id": "req-remote",
                    "action": "accept",
                },
                headers={"Authorization": "Bearer secret-remote-key"},
                client_host="10.0.0.8",
            )
            self.assertEqual(remote_with_header["status_code"], 200)
            self.assertTrue(remote_with_header["body"]["success"])
        finally:
            self.app.state.pfe_services.security.allow_remote_access = previous_allow_remote_access
            self.app.state.pfe_services.security.auth_mode = previous_auth_mode
            if previous_api_key is None:
                os.environ.pop("PFE_API_KEY", None)
            else:
                os.environ["PFE_API_KEY"] = previous_api_key

    def test_feedback_requires_api_key_when_auth_mode_is_strict(self) -> None:
        previous_auth_mode = self.app.state.pfe_services.security.auth_mode
        previous_api_key = os.environ.get("PFE_API_KEY")
        try:
            self.app.state.pfe_services.security.auth_mode = "api_key_required"
            os.environ["PFE_API_KEY"] = "secret-local-key"

            local_without_key = self._smoke(
                "/pfe/feedback",
                method="POST",
                body={
                    "session_id": "sess-local",
                    "request_id": "req-local",
                    "action": "accept",
                },
            )
            self.assertEqual(local_without_key["status_code"], 401)
            self.assertEqual(local_without_key["body"]["code"], "unauthorized")
            self.assertIn("hint", local_without_key["body"])

            local_with_key = self._smoke(
                "/pfe/feedback",
                method="POST",
                body={
                    "session_id": "sess-local",
                    "request_id": "req-local",
                    "action": "accept",
                },
                headers={"x-api-key": "secret-local-key"},
            )
            self.assertEqual(local_with_key["status_code"], 200)
            self.assertTrue(local_with_key["body"]["success"])
        finally:
            self.app.state.pfe_services.security.auth_mode = previous_auth_mode
            if previous_api_key is None:
                os.environ.pop("PFE_API_KEY", None)
            else:
                os.environ["PFE_API_KEY"] = previous_api_key

    def test_status_requires_api_key_for_remote_clients_when_enabled(self) -> None:
        previous_allow_remote_access = self.app.state.pfe_services.security.allow_remote_access
        previous_auth_mode = self.app.state.pfe_services.security.auth_mode
        previous_api_key = os.environ.get("PFE_API_KEY")
        try:
            self.app.state.pfe_services.security.allow_remote_access = True
            self.app.state.pfe_services.security.auth_mode = "local_optional"
            os.environ.pop("PFE_API_KEY", None)

            remote_without_key = self._smoke(
                "/pfe/status",
                query_params={"detail": "full"},
                client_host="10.0.0.8",
            )
            self.assertEqual(remote_without_key["status_code"], 503)
            self.assertEqual(remote_without_key["body"]["code"], "api_key_not_configured")
            self.assertIn("hint", remote_without_key["body"])

            os.environ["PFE_API_KEY"] = "secret-remote-key"

            remote_without_header = self._smoke(
                "/pfe/status",
                query_params={"detail": "full"},
                client_host="10.0.0.8",
            )
            self.assertEqual(remote_without_header["status_code"], 401)
            self.assertEqual(remote_without_header["body"]["code"], "unauthorized")
            self.assertIn("hint", remote_without_header["body"])

            remote_with_header = self._smoke(
                "/pfe/status",
                query_params={"detail": "full"},
                headers={"Authorization": "Bearer secret-remote-key"},
                client_host="10.0.0.8",
            )
            self.assertEqual(remote_with_header["status_code"], 200)
            self.assertIn("strict_local", remote_with_header["body"])
        finally:
            self.app.state.pfe_services.security.allow_remote_access = previous_allow_remote_access
            self.app.state.pfe_services.security.auth_mode = previous_auth_mode
            if previous_api_key is None:
                os.environ.pop("PFE_API_KEY", None)
            else:
                os.environ["PFE_API_KEY"] = previous_api_key

    def test_signals_requires_api_key_for_remote_clients_when_enabled(self) -> None:
        previous_allow_remote_access = self.app.state.pfe_services.security.allow_remote_access
        previous_auth_mode = self.app.state.pfe_services.security.auth_mode
        previous_api_key = os.environ.get("PFE_API_KEY")
        try:
            self.app.state.pfe_services.security.allow_remote_access = True
            self.app.state.pfe_services.security.auth_mode = "local_optional"
            os.environ.pop("PFE_API_KEY", None)

            remote_without_key = self._smoke(
                "/pfe/signals",
                client_host="10.0.0.8",
            )
            self.assertEqual(remote_without_key["status_code"], 503)
            self.assertEqual(remote_without_key["body"]["code"], "api_key_not_configured")
            self.assertIn("hint", remote_without_key["body"])

            os.environ["PFE_API_KEY"] = "secret-remote-key"

            remote_without_header = self._smoke(
                "/pfe/signals",
                client_host="10.0.0.8",
            )
            self.assertEqual(remote_without_header["status_code"], 401)
            self.assertEqual(remote_without_header["body"]["code"], "unauthorized")
            self.assertIn("hint", remote_without_header["body"])

            remote_with_header = self._smoke(
                "/pfe/signals",
                headers={"Authorization": "Bearer secret-remote-key"},
                client_host="10.0.0.8",
            )
            self.assertEqual(remote_with_header["status_code"], 200)
            self.assertIn("signals", remote_with_header["body"])
        finally:
            self.app.state.pfe_services.security.allow_remote_access = previous_allow_remote_access
            self.app.state.pfe_services.security.auth_mode = previous_auth_mode
            if previous_api_key is None:
                os.environ.pop("PFE_API_KEY", None)
            else:
                os.environ["PFE_API_KEY"] = previous_api_key


if __name__ == "__main__":
    unittest.main()
