from __future__ import annotations

import pathlib
import unittest


class ChatDaemonRecoveryTimelineSurfaceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.html = pathlib.Path(
            str(pathlib.Path(__file__).resolve().parents[1] / "pfe-server" / "pfe_server" / "static" / "chat.html")
        ).read_text(encoding="utf-8")

    def test_worker_daemon_recovery_timeline_surface_is_present(self) -> None:
        expected_fragments = [
            "Worker Daemon",
            "Daemon History / Recovery Timeline",
            "recovery timeline / history",
            "Daemon Timeline Focus",
            "Daemon Latest Anomaly",
            "Daemon Suggested Action",
            "latest anomaly:",
            "current suggested action:",
            "started | recover_requested | restart_requested | stop_requested | completed | stale_lock_takeover",
            "workerDaemonHistoryValue",
            "recent recovery events:",
            "recent recovery events: none",
        ]

        for fragment in expected_fragments:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, self.html)

    def test_recovery_timeline_formatter_mentions_recent_reason_and_key_events(self) -> None:
        expected_fragments = [
            "formatWorkerDaemonRecoveryTimeline",
            "formatWorkerDaemonRecoveryTimelineFocus",
            "formatWorkerDaemonRecoveryTimelineAnomaly",
            "formatWorkerDaemonRecoveryTimelineAction",
            "deriveDaemonRecoveryTimelineEvent",
            "deriveDaemonRecoveryTimelineAnomaly",
            "deriveDaemonRecoveryTimelineSuggestion",
            "last_event=",
            "last_reason=",
            "last_note=",
            "stale_lock_takeover",
            "recover_requested",
            "restart_requested",
            "stop_requested",
            "completed",
            "started",
        ]

        for fragment in expected_fragments:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, self.html)


if __name__ == "__main__":
    unittest.main()
