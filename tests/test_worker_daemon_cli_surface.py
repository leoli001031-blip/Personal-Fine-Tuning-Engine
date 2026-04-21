from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from typer.testing import CliRunner

ROOT = Path(__file__).resolve().parents[1]
for package_dir in ("pfe-core", "pfe-cli", "pfe-server"):
    package_path = str(ROOT / package_dir)
    if package_path not in os.sys.path:
        os.sys.path.insert(0, package_path)

from pfe_cli import main as cli_main


class WorkerDaemonCliSurfaceTests(unittest.TestCase):
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

    def test_daemon_help_and_status_surface_are_available(self) -> None:
        runner = CliRunner()

        help_result = runner.invoke(cli_main.app, ["daemon", "--help"])
        self.assertEqual(help_result.exit_code, 0, msg=help_result.stdout)
        self.assertIn("status", help_result.stdout)
        self.assertIn("start", help_result.stdout)
        self.assertIn("stop", help_result.stdout)
        self.assertIn("history", help_result.stdout)

        status_result = runner.invoke(cli_main.app, ["daemon", "status", "--workspace", "user_default"])
        self.assertEqual(status_result.exit_code, 0, msg=status_result.stdout)
        self.assertIn("PFE worker daemon", status_result.stdout)
        self.assertIn("command_status=idle", status_result.stdout)

    def test_daemon_start_stop_history_updates_local_control_plane(self) -> None:
        runner = CliRunner()

        start_result = runner.invoke(cli_main.app, ["daemon", "start", "--workspace", "user_default"])
        self.assertEqual(start_result.exit_code, 0, msg=start_result.stdout)
        self.assertIn("PFE worker daemon", start_result.stdout)
        self.assertIn("desired_state=running", start_result.stdout)
        self.assertIn("requested_action=start", start_result.stdout)
        self.assertIn("history_count=1", start_result.stdout)

        history_result = runner.invoke(cli_main.app, ["daemon", "history", "--workspace", "user_default", "--limit", "5"])
        self.assertEqual(history_result.exit_code, 0, msg=history_result.stdout)
        self.assertIn("PFE worker daemon history", history_result.stdout)
        self.assertIn("event=start_requested", history_result.stdout)
        self.assertIn("reason=daemon_start_requested", history_result.stdout)

        stop_result = runner.invoke(cli_main.app, ["daemon", "stop", "--workspace", "user_default"])
        self.assertEqual(stop_result.exit_code, 0, msg=stop_result.stdout)
        self.assertIn("desired_state=stopped", stop_result.stdout)
        self.assertIn("requested_action=stop", stop_result.stdout)
        self.assertIn("history_count=2", stop_result.stdout)

        root_status_result = runner.invoke(cli_main.app, ["status", "--workspace", "user_default"])
        self.assertEqual(root_status_result.exit_code, 0, msg=root_status_result.stdout)
        self.assertIn("DAEMON TIMELINE", root_status_result.stdout)
        self.assertIn("stop_requested", root_status_result.stdout)
        self.assertIn("count:", root_status_result.stdout)


if __name__ == "__main__":
    unittest.main()
