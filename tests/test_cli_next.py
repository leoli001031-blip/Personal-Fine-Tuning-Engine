from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

from typer.testing import CliRunner

from pfe_cli import main as cli_main
from tests.matrix_test_compat import isolated_cwd


class CLINextTests(unittest.TestCase):
    def setUp(self) -> None:
        self.previous_home = os.environ.get("PFE_HOME")

    def tearDown(self) -> None:
        if self.previous_home is None:
            os.environ.pop("PFE_HOME", None)
        else:
            os.environ["PFE_HOME"] = self.previous_home

    def test_next_points_to_init_when_config_is_missing(self) -> None:
        runner = CliRunner()
        with isolated_cwd():
            os.environ["PFE_HOME"] = str(Path.cwd() / ".pfe")

            result = runner.invoke(cli_main.app, ["next", "--workspace", "alice"])

            self.assertEqual(result.exit_code, 0, msg=result.stdout)
            text = result.stdout
            self.assertIn("PFE next", text)
            self.assertIn("workspace: alice", text)
            self.assertIn("state: init_required", text)
            self.assertIn("pfe init --workspace alice --base-model <path-or-model-id>", text)
            self.assertIn("pfe doctor --workspace alice", text)

    def test_next_guides_initialized_workspace_toward_feedback_loop(self) -> None:
        runner = CliRunner()
        with isolated_cwd():
            os.environ["PFE_HOME"] = str(Path.cwd() / ".pfe")
            model_dir = Path("models/local-base")
            model_dir.mkdir(parents=True)
            (model_dir / "config.json").write_text(
                json.dumps({"architectures": ["GPT2LMHeadModel"], "model_type": "gpt2", "vocab_size": 32}),
                encoding="utf-8",
            )
            init_result = runner.invoke(
                cli_main.app,
                [
                    "init",
                    "--workspace",
                    "alice",
                    "--base-model",
                    "./models/local-base",
                    "--home",
                    ".pfe",
                ],
            )
            self.assertEqual(init_result.exit_code, 0, msg=init_result.stdout)

            result = runner.invoke(cli_main.app, ["next", "--workspace", "alice", "--json"])

            self.assertEqual(result.exit_code, 0, msg=result.stdout)
            plan = json.loads(result.stdout)
            self.assertEqual(plan["state"], "collect_feedback")
            self.assertEqual(plan["workspace"], "alice")
            self.assertIn("pfe generate --scenario life-coach", "\n".join(plan["commands"]))
            self.assertIn("pfe trigger configure --workspace alice", "\n".join(plan["commands"]))
            self.assertIn("pfe collect ingest --workspace alice --help", "\n".join(plan["commands"]))

    def test_next_surfaces_deferred_queue_work_after_feedback_ingest(self) -> None:
        runner = CliRunner()
        with isolated_cwd():
            os.environ["PFE_HOME"] = str(Path.cwd() / ".pfe")
            model_dir = Path("models/local-base")
            model_dir.mkdir(parents=True)
            (model_dir / "config.json").write_text(
                json.dumps({"architectures": ["GPT2LMHeadModel"], "model_type": "gpt2", "vocab_size": 32}),
                encoding="utf-8",
            )
            commands = [
                [
                    "init",
                    "--workspace",
                    "alice",
                    "--base-model",
                    "./models/local-base",
                    "--home",
                    ".pfe",
                ],
                ["generate", "--scenario", "life-coach", "--style", "warm", "--num", "8", "--workspace", "alice"],
                [
                    "trigger",
                    "configure",
                    "--workspace",
                    "alice",
                    "--enable",
                    "--min-new-samples",
                    "1",
                    "--queue-mode",
                    "deferred",
                    "--max-interval-days",
                    "0",
                    "--no-require-confirmation",
                    "--epochs",
                    "1",
                    "--backend",
                    "mock_local",
                ],
                [
                    "collect",
                    "ingest",
                    "--workspace",
                    "alice",
                    "--event-id",
                    "evt-next-feedback-1",
                    "--request-id",
                    "req-next-1",
                    "--session-id",
                    "sess-next-1",
                    "--source-event-id",
                    "evt-next-chat-1",
                    "--user-input",
                    "Help me choose one next step for today.",
                    "--model-output",
                    "Pick one task you can finish in the next 20 minutes.",
                    "--action",
                    "accept",
                    "--scenario",
                    "life-coach",
                ],
            ]
            for command in commands:
                result = runner.invoke(cli_main.app, command)
                self.assertEqual(result.exit_code, 0, msg=result.stdout)

            result = runner.invoke(cli_main.app, ["next", "--workspace", "alice"])

            self.assertEqual(result.exit_code, 0, msg=result.stdout)
            text = result.stdout
            self.assertIn("state: queue_ready", text)
            self.assertIn("process the next deferred auto-train queue item", text)
            self.assertIn("pfe trigger process-next --workspace alice", text)

            process_result = runner.invoke(cli_main.app, ["trigger", "process-next", "--workspace", "alice"])
            self.assertEqual(process_result.exit_code, 0, msg=process_result.stdout)

            result = runner.invoke(cli_main.app, ["next", "--workspace", "alice", "--json"])
            self.assertEqual(result.exit_code, 0, msg=result.stdout)
            plan = json.loads(result.stdout)
            self.assertEqual(plan["state"], "evaluate_candidate")
            self.assertIn("pfe eval --base-model base --adapter", "\n".join(plan["commands"]))
            self.assertIn("pfe adapter promote", "\n".join(plan["commands"]))


if __name__ == "__main__":
    unittest.main()
