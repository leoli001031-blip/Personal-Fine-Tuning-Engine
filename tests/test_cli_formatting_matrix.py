"""Updated CLI formatting tests for Matrix theme."""

from __future__ import annotations

import io
import json
import os
import re
from contextlib import redirect_stdout
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

ROOT = Path(__file__).resolve().parents[1]
for package_dir in ("pfe-core", "pfe-cli", "pfe-server"):
    package_path = str(ROOT / package_dir)
    if package_path not in os.sys.path:
        os.sys.path.insert(0, package_path)

from pfe_cli import adapter_commands, main as cli_main
from pfe_cli.main import (
    _format_eval_result,
    _format_eval_result_legacy,
    _format_serve,
    _format_serve_preview,
    _format_status,
    _format_train_preview,
    _format_train_result,
)
from pfe_core.adapter_store.store import AdapterStore
from pfe_core.db.sqlite import resolve_home
from pfe_core.pipeline import PipelineService
from pfe_core.trainer.service import TrainerService
from pfe_server.app import build_serve_plan


def strip_ansi(text: str) -> str:
    """Remove ANSI color codes from text for testing."""
    ansi_pattern = re.compile(r'\x1b\[[0-9;]*m')
    return ansi_pattern.sub('', text)


class CLIFormattingMatrixTests(unittest.TestCase):
    """Test CLI formatting with Matrix theme (default)."""

    def test_resolve_home_prefers_workspace_local_dot_pfe_when_present(self) -> None:
        previous_home = os.environ.get("PFE_HOME")
        current_dir = Path.cwd()
        try:
            with tempfile.TemporaryDirectory() as tempdir:
                os.chdir(tempdir)
                Path(".pfe").mkdir()
                os.environ.pop("PFE_HOME", None)
                self.assertEqual(resolve_home().resolve(), (Path(tempdir) / ".pfe").resolve())
        finally:
            os.chdir(current_dir)
            if previous_home is None:
                os.environ.pop("PFE_HOME", None)
            else:
                os.environ["PFE_HOME"] = previous_home

    def test_status_format_includes_plan_sections(self) -> None:
        payload = {
            "home": "/tmp/pfe",
            "latest_adapter": {
                "version": "20260323-001",
                "state": "promoted",
            },
            "latest_adapter_version": "20260323-001",
            "recent_adapter": {
                "version": "20260323-002",
                "state": "pending_eval",
            },
            "recent_adapter_version": "20260323-002",
            "signal_summary": {"total_signals": 42, "processed_signals": 38},
            "sample_counts": {"train": 10, "val": 1, "test": 2},
        }

        text = _format_status(payload, workspace="/tmp/pfe")
        clean = strip_ansi(text)
        
        # Check Matrix format elements
        self.assertIn("PFE STATUS", clean)
        self.assertIn("20260323-", clean)  # Version may be truncated in display
        self.assertIn("promoted", clean)
        self.assertIn("pending_eval", clean)
        self.assertIn("ADAPTER LIFECYCLE", clean)
        self.assertIn("WORKSPACE: /tmp/pfe", clean)

    def test_train_result_includes_version_and_samples(self) -> None:
        payload = {
            "version": "20260323-001",
            "adapter_path": "/tmp/pfe/adapters/20260323-001",
            "num_samples": 12,
            "backend_plan": {
                "selected_backend": "mock_local",
                "requested_backend": "auto",
                "runtime_device": "cpu",
                "requires_export_step": False,
            },
            "export_runtime": {
                "required": False,
                "target_artifact_format": "peft_lora",
                "dry_run": True,
            },
            "metrics": {
                "num_fresh_samples": 8,
                "num_replay_samples": 4,
            },
        }

        text = _format_train_result(payload)
        clean = strip_ansi(text)
        
        # Check Matrix format elements
        self.assertIn("TRAINING COMPLETE", clean)
        self.assertIn("20260323-001", clean)
        self.assertIn("TRAINING RESULT", clean)

    def test_eval_result_shows_recommendation(self) -> None:
        payload = {
            "adapter_version": "20260323-001",
            "base_model": "base",
            "num_test_samples": 2,
            "scores": {
                "style_preference_hit_rate": 0.85,
                "style_match": 0.75,
                "preference_alignment": 0.72,
            },
            "comparison": "improved",
            "recommendation": "deploy",
        }

        text = _format_eval_result(payload)
        clean = strip_ansi(text)
        
        # Check Matrix format elements
        self.assertIn("EVALUATION RESULT", clean)
        self.assertIn("20260323-001", clean)
        self.assertIn("deploy", clean.lower())
        self.assertIn("style_preference_hit_rate", clean)
        self.assertIn("0.85", clean)

    def test_eval_result_legacy_prioritizes_style_preference_hit_rate(self) -> None:
        payload = {
            "adapter_version": "20260323-001",
            "base_model": "base",
            "num_test_samples": 2,
            "scores": {
                "style_match": 0.75,
                "preference_alignment": 0.72,
                "style_preference_hit_rate": 0.85,
            },
            "comparison": "improved",
            "recommendation": "deploy",
        }

        text = _format_eval_result_legacy(payload)
        self.assertIn("scores: style_preference_hit_rate=", text)
        self.assertLess(
            text.index("style_preference_hit_rate="),
            text.index("style_match="),
        )

    def test_serve_preserves_ready_message(self) -> None:
        text = _format_serve({"ready_message": "server ready on 127.0.0.1:8921"})
        clean = strip_ansi(text)
        self.assertIn("server ready on 127.0.0.1:8921", clean)

    def test_adapter_list_shows_matrix_format(self) -> None:
        """Test that adapter list uses Matrix table format."""
        result = {
            "versions": [
                {"version": "20260323-001", "state": "promoted", "latest": True, "num_training_samples": 4},
                {"version": "20260323-002", "state": "archived", "num_training_samples": 2},
            ]
        }
        
        # Capture output from _echo_result
        from pfe_cli.adapter_commands import _echo_result
        import sys
        from io import StringIO
        
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        _echo_result(result)
        output = sys.stdout.getvalue()
        sys.stdout = old_stdout
        
        clean = strip_ansi(output)
        self.assertIn("ADAPTER VERSIONS", clean)
        self.assertIn("20260323-001", clean)
        self.assertIn("promoted", clean)


if __name__ == "__main__":
    unittest.main()
