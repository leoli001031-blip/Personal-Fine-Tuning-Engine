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

import pfe_cli.main as cli_main
from pfe_cli.main import _format_candidate_timeline
from pfe_core.pipeline import PipelineService


class CandidateTimelineSurfaceTests(unittest.TestCase):
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

    def _build_timeline(self) -> tuple[PipelineService, str, str]:
        service = PipelineService()
        service.generate(scenario="life-coach", style="温和", num_samples=8)
        first = service.train_result(method="qlora", epochs=1, train_type="sft")
        service.promote_candidate(note="ready_for_rollout")

        service.generate(scenario="work-coach", style="direct", num_samples=8)
        second = service.train_result(method="qlora", epochs=1, train_type="sft")
        service.archive_candidate(note="archive_after_review")
        return service, first.version, second.version

    def test_candidate_timeline_formatter_derives_stage_and_label(self) -> None:
        text = _format_candidate_timeline(
            {
                "workspace": "user_default",
                "count": 2,
                "limit": 5,
                "current_stage": "archived",
                "transition_count": 2,
                "last_reason": "archived_after_review",
                "last_candidate_version": "20260326-200",
                "items": [
                    {
                        "timestamp": "2026-03-26T09:00:00+00:00",
                        "action": "promote_candidate",
                        "status": "completed",
                        "reason": "candidate_promoted",
                        "operator_note": "ready_for_rollout",
                        "candidate_version": "20260326-100",
                        "promoted_version": "20260326-100",
                    },
                    {
                        "timestamp": "2026-03-26T10:00:00+00:00",
                        "action": "archive_candidate",
                        "status": "completed",
                        "reason": "archived_after_review",
                        "operator_note": "archive_after_review",
                        "candidate_version": "20260326-200",
                        "archived_version": "20260326-200",
                    },
                ],
            }
        )

        self.assertIn("PFE candidate timeline", text)
        self.assertIn("summary: workspace=user_default | count=2 | limit=5 | current_stage=archived", text)
        self.assertIn("latest timestamp: 2026-03-26T10:00:00+00:00", text)
        self.assertIn("timeline:", text)
        self.assertIn("1. timestamp=2026-03-26T09:00:00+00:00 | stage=promoted | label=promote_candidate:completed", text)
        self.assertIn("2. timestamp=2026-03-26T10:00:00+00:00 | stage=archived | label=archive_candidate:completed", text)

    def test_candidate_timeline_cli_surfaces_stage_and_version_changes(self) -> None:
        service, promoted_version, archived_version = self._build_timeline()
        runner = CliRunner()

        result = runner.invoke(
            cli_main.app,
            ["candidate", "timeline", "--workspace", "user_default", "--limit", "5"],
        )

        self.assertEqual(result.exit_code, 0, msg=result.stdout)
        self.assertIn("PFE candidate timeline", result.stdout)
        self.assertIn("current_stage=archived", result.stdout)
        self.assertIn(f"last_candidate_version={archived_version}", result.stdout)
        self.assertIn("label=promote_candidate:completed", result.stdout)
        self.assertIn("label=archive_candidate:completed", result.stdout)
        self.assertIn(f"candidate_version={promoted_version}", result.stdout)
        self.assertIn(f"candidate_version={archived_version}", result.stdout)


if __name__ == "__main__":
    unittest.main()
