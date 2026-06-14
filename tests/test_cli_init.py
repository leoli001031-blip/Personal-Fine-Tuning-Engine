from __future__ import annotations

import unittest
from pathlib import Path

from typer.testing import CliRunner

from pfe_cli import main as cli_main
from pfe_core.config import PFEConfig
from tests.matrix_test_compat import isolated_cwd, strip_ansi


class CLIInitTests(unittest.TestCase):
    def test_init_creates_workspace_dirs_and_default_config(self) -> None:
        runner = CliRunner()
        with isolated_cwd():
            result = runner.invoke(
                cli_main.app,
                [
                    "init",
                    "--workspace",
                    "alice",
                    "--base-model",
                    "/models/qwen-local",
                    "--home",
                    ".pfe",
                ],
            )

            self.assertEqual(result.exit_code, 0, msg=result.stdout)
            home = Path(".pfe")
            config_path = home / "config.toml"
            self.assertTrue((home / "data").is_dir())
            self.assertTrue((home / "adapters" / "alice").is_dir())
            self.assertTrue((home / "cache").is_dir())
            self.assertTrue((home / "logs").is_dir())
            self.assertTrue(config_path.is_file())

            config = PFEConfig.load(home=home)
            self.assertEqual(config.model.base_model, "/models/qwen-local")
            self.assertEqual(config.server.host, "127.0.0.1")
            self.assertEqual(config.server.port, 8921)
            self.assertEqual(config.privacy.mode, "strict_local")
            self.assertTrue(config.privacy.redact_pii)
            self.assertFalse(config.security.allow_remote_access)
            self.assertTrue(config.collector.enabled)
            self.assertEqual(config.trainer.method, "qlora")

            text = result.stdout
            self.assertIn("PFE workspace initialized", text)
            self.assertIn("config path: .pfe/config.toml", text)
            self.assertIn("workspace:   alice", text)
            self.assertIn("base model:  /models/qwen-local", text)
            self.assertIn("pfe doctor --workspace alice", text)
            self.assertIn("pfe next --workspace alice", text)
            self.assertIn("pfe serve --port 8921 --workspace alice --live", text)

    def test_init_help_documents_required_options(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli_main.app, ["init", "--help"])

        self.assertEqual(result.exit_code, 0, msg=result.stdout)
        text = strip_ansi(result.stdout)
        self.assertIn("--workspace", text)
        self.assertIn("--base-model", text)
        self.assertIn("--home", text)
