from __future__ import annotations

import pathlib
import unittest


class ChatCandidateTimelineSurfaceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.html = pathlib.Path(
            str(pathlib.Path(__file__).resolve().parents[1] / "pfe-server" / "pfe_server" / "static" / "chat.html")
        ).read_text(encoding="utf-8")

    def test_candidate_history_panel_reads_like_a_timeline(self) -> None:
        expected_fragments = [
            "Candidate History",
            "timeline-preview",
            "PFE candidate timeline",
            "轨迹预览",
            "summary: ",
            "current stage: ",
            "recent transitions:",
            "deriveCandidateTimelineStage",
            "formatCandidateTimelineEntry",
        ]

        for fragment in expected_fragments:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, self.html)

    def test_candidate_history_refresh_still_comes_from_history_api(self) -> None:
        expected_fragments = [
            'fetchWithAuth("/pfe/candidate/history?limit=5")',
            "candidateHistoryDetail.items || candidateHistory.items || []",
            "candidateHistoryValue",
        ]

        for fragment in expected_fragments:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, self.html)


if __name__ == "__main__":
    unittest.main()
