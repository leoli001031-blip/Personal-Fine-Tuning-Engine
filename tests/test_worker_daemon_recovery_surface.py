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


class WorkerDaemonRecoverySurfaceTests(unittest.TestCase):
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

    def test_daemon_help_lists_recovery_commands(self) -> None:
        runner = CliRunner()

        help_result = runner.invoke(cli_main.app, ["daemon", "--help"])
        self.assertEqual(help_result.exit_code, 0, msg=help_result.stdout)
        self.assertIn("recover", help_result.stdout)
        self.assertIn("restart", help_result.stdout)

    def test_daemon_recover_and_restart_update_local_recovery_state(self) -> None:
        runner = CliRunner()

        recover_result = runner.invoke(cli_main.app, ["daemon", "recover", "--workspace", "user_default", "--note", "manual_recovery"])
        self.assertEqual(recover_result.exit_code, 0, msg=recover_result.stdout)
        self.assertIn("PFE worker daemon", recover_result.stdout)
        self.assertIn("last_event=recover_blocked", recover_result.stdout)
        self.assertIn("last_reason=daemon_recovery_not_needed", recover_result.stdout)
        self.assertIn("auto_restart_enabled=yes", recover_result.stdout)
        self.assertIn("history_count=1", recover_result.stdout)

        recover_history = runner.invoke(cli_main.app, ["daemon", "history", "--workspace", "user_default", "--limit", "5"])
        self.assertEqual(recover_history.exit_code, 0, msg=recover_history.stdout)
        self.assertIn("event=recover_blocked", recover_history.stdout)
        self.assertIn("reason=daemon_recovery_not_needed", recover_history.stdout)

        restart_result = runner.invoke(cli_main.app, ["daemon", "restart", "--workspace", "user_default", "--note", "manual_restart"])
        self.assertEqual(restart_result.exit_code, 0, msg=restart_result.stdout)
        self.assertIn("requested_action=restart", restart_result.stdout)
        self.assertIn("recovery_state=restarting", restart_result.stdout)
        self.assertIn("restart_attempts=1", restart_result.stdout)
        self.assertIn("history_count=2", restart_result.stdout)

        root_status_result = runner.invoke(cli_main.app, ["status", "--workspace", "user_default"])
        self.assertEqual(root_status_result.exit_code, 0, msg=root_status_result.stdout)
        self.assertIn("DAEMON TIMELINE", root_status_result.stdout)
        self.assertIn("restart_requested", root_status_result.stdout)
        self.assertIn("count:", root_status_result.stdout)

        daemon_history = runner.invoke(cli_main.app, ["daemon", "history", "--workspace", "user_default", "--limit", "5"])
        self.assertEqual(daemon_history.exit_code, 0, msg=daemon_history.stdout)
        self.assertIn("event=recover_blocked", daemon_history.stdout)
        self.assertIn("event=restart_requested", daemon_history.stdout)
        self.assertIn("note=manual_restart", daemon_history.stdout)


if __name__ == "__main__":
    unittest.main()
