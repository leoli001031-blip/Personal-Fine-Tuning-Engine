from __future__ import annotations

import pathlib
import unittest


class ChatWorkerRunnerTimelineSurfaceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.html = pathlib.Path(
            str(pathlib.Path(__file__).resolve().parents[1] / "pfe-server" / "pfe_server" / "static" / "chat.html")
        ).read_text(encoding="utf-8")

    def test_worker_runner_timeline_surface_is_present(self) -> None:
        expected_fragments = [
            "Worker Loop",
            "Worker Runner Timeline / History",
            "Worker Runner / lease / processed / current focus / recent timeline events",
            "Worker Runner Focus",
            "Worker Runner Latest Anomaly",
            "Worker Runner Suggested Action",
            "PFE worker runner timeline",
            "timeline / history",
            "current focus:",
            "latest anomaly:",
            "current suggested action:",
            "recent runner events:",
            "recent runner events: none",
        ]

        for fragment in expected_fragments:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, self.html)

    def test_worker_runner_timeline_formatter_helpers_are_present(self) -> None:
        expected_fragments = [
            "formatWorkerRunnerTimeline",
            "formatWorkerRunnerHistory",
            "formatWorkerRunnerTimelineFocus",
            "formatWorkerRunnerTimelineAnomaly",
            "formatWorkerRunnerTimelineAction",
            "deriveWorkerRunnerTimelineEvent",
            "deriveWorkerRunnerTimelineAnomaly",
            "deriveWorkerRunnerTimelineSuggestion",
        ]

        for fragment in expected_fragments:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, self.html)


if __name__ == "__main__":
    unittest.main()
