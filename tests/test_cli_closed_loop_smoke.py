from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pytest

from tests.matrix_test_compat import strip_ansi

ROOT = Path(__file__).resolve().parents[1]
for package_dir in ("pfe-core", "pfe-cli", "pfe-server"):
    package_path = str(ROOT / package_dir)
    if package_path not in os.sys.path:
        os.sys.path.insert(0, package_path)


@pytest.mark.slow
class CLIClosedLoopSmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.previous_home = os.environ.get("PFE_HOME")
        self.pfe_home = Path(self.tempdir.name) / ".pfe"
        self.pfe_home.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        if self.previous_home is None:
            os.environ.pop("PFE_HOME", None)
        else:
            os.environ["PFE_HOME"] = self.previous_home
        self.tempdir.cleanup()

    def _run_cli(self, *args: str) -> str:
        env = os.environ.copy()
        env["PFE_HOME"] = str(self.pfe_home)
        env["PFE_DISABLE_AUTO_LOCAL_BASE_MODEL"] = "1"
        env["PYTHONPATH"] = os.pathsep.join(
            str(ROOT / package_dir) for package_dir in ("pfe-core", "pfe-cli", "pfe-server")
        )
        completed = subprocess.run(
            [sys.executable, "-m", "pfe_cli.main", *args],
            cwd=str(ROOT),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise AssertionError(
                f"CLI command failed: {' '.join(args)}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
            )
        return completed.stdout

    def test_generate_train_eval_promote_status_and_serve_smoke(self) -> None:
        generate_output = self._run_cli("generate", "--scenario", "life-coach", "--style", "温和", "--num", "8")
        clean_generate_output = strip_ansi(generate_output)
        self.assertIn("Saved 8 distilled sample(s)", clean_generate_output)

        train_output = self._run_cli("train", "--method", "qlora", "--epochs", "1", "--train-type", "sft")
        clean_train_output = strip_ansi(train_output)
        match = re.search(r"version:\s+(\d{8}-\d{3})", clean_train_output)
        self.assertIsNotNone(match, msg=clean_train_output)
        version = match.group(1)
        self.assertIn("[ TRAINING RESULT ]", clean_train_output)
        self.assertIn("backend:", clean_train_output)
        self.assertIn("export:", clean_train_output)

        eval_output = self._run_cli("eval", "--base-model", "base", "--adapter", version, "--num-samples", "3")
        clean_eval_output = strip_ansi(eval_output)
        self.assertIn("[ EVALUATION RESULT ]", clean_eval_output)
        self.assertIn("adapter:", clean_eval_output)
        self.assertIn(f"{version}", clean_eval_output)
        self.assertIn("recommendation:", clean_eval_output)

        status_before_promote = self._run_cli("status")
        clean_status_before_promote = strip_ansi(status_before_promote)
        self.assertIn("[ ADAPTER LIFECYCLE ]", clean_status_before_promote)
        self.assertIn("recent training:", clean_status_before_promote)
        self.assertIn(f"{version}", clean_status_before_promote)

        promote_output = self._run_cli("adapter", "promote", version)
        clean_promote_output = strip_ansi(promote_output)
        self.assertIn("latest:", clean_promote_output)
        self.assertIn(version, clean_promote_output)

        status_after_promote = self._run_cli("status")
        clean_status_after_promote = strip_ansi(status_after_promote)
        self.assertIn("[ ADAPTER LIFECYCLE ]", clean_status_after_promote)
        self.assertIn(f"latest promoted:         {version} | state=promoted", clean_status_after_promote)
        self.assertIn(f"recent training:         {version} | state=promoted", clean_status_after_promote)

        serve_output = self._run_cli("serve", "--host", "127.0.0.1", "--port", "8921")
        clean_serve_output = strip_ansi(serve_output)
        self.assertIn("[ SERVE PREVIEW ]", clean_serve_output)
        self.assertIn("[ LATEST PROMOTED ]", clean_serve_output)
        self.assertIn(f"version:                 {version}", clean_serve_output)
        self.assertIn("[ RECENT TRAINING ]", clean_serve_output)
        self.assertIn(f"version:                 {version}", clean_serve_output)
        self.assertIn("state:                   pending_eval", clean_serve_output)


if __name__ == "__main__":
    unittest.main()
