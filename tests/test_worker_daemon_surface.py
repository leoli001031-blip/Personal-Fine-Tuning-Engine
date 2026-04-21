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

from pfe_core.pipeline import PipelineService
from pfe_core.server_services import InferenceServiceAdapter, PipelineServiceAdapter
from pfe_server.app import ServiceBundle, create_app, smoke_test_request
from pfe_server.auth import ServerSecurityConfig


class WorkerDaemonSurfaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.previous_home = os.environ.get("PFE_HOME")
        self.pfe_home = Path(self.tempdir.name) / ".pfe"
        os.environ["PFE_HOME"] = str(self.pfe_home)
        self.html = Path(str(Path(__file__).resolve().parents[1] / "pfe-server" / "pfe_server" / "static" / "chat.html")).read_text(
            encoding="utf-8"
        )

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

    def test_chat_shell_exposes_worker_daemon_controls_and_status_card(self) -> None:
        expected_fragments = [
            "Worker Daemon",
            "Daemon Status",
            "Daemon Recovery",
            "Daemon History",
            "workerDaemonValue",
            "workerDaemonRecoveryValue",
            "workerDaemonHistoryValue",
            "refreshDaemonBtn",
            "runWorkerRunnerDaemonBtn",
            "stopWorkerRunnerDaemonBtn",
            "restartWorkerRunnerDaemonBtn",
            "recoverWorkerRunnerDaemonBtn",
            "formatWorkerDaemonStatus",
            "formatWorkerDaemonRecovery",
            "/pfe/auto-train/worker-daemon",
            "/pfe/auto-train/recover-worker-daemon",
            "/pfe/auto-train/restart-worker-daemon",
        ]

        for fragment in expected_fragments:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, self.html)

    def test_root_page_serves_the_worker_daemon_surface(self) -> None:
        service = self._service()
        app = self._app(service)

        async def scenario() -> str:
            result = await smoke_test_request(app, path="/", method="GET")
            return result["text"]

        root_text = asyncio.run(scenario())
        self.assertIn("Worker Daemon", root_text)
        self.assertIn("刷新 Daemon", root_text)
        self.assertIn("/pfe/auto-train/start-worker-daemon", root_text)
        self.assertIn("/pfe/auto-train/stop-worker-daemon", root_text)
        self.assertIn("/pfe/auto-train/restart-worker-daemon", root_text)
        self.assertIn("/pfe/auto-train/recover-worker-daemon", root_text)
        self.assertIn("重启 Daemon", root_text)
        self.assertIn("恢复 Daemon", root_text)


if __name__ == "__main__":
    unittest.main()
