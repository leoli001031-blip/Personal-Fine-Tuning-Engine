from __future__ import annotations

import pathlib
import unittest


class ChatDaemonHealthSurfaceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.html = pathlib.Path(
            str(pathlib.Path(__file__).resolve().parents[1] / "pfe-server" / "pfe_server" / "static" / "chat.html")
        ).read_text(encoding="utf-8")

    def test_worker_daemon_surface_exposes_atomic_health_and_recovery_fields(self) -> None:
        expected_fragments = [
            "Worker Daemon",
            "atomic health fields",
            "Daemon Atomic",
            "Daemon History / Recovery Timeline",
            "recovery timeline / history",
            "current focus:",
            "latest anomaly:",
            "current suggested action:",
            "operationsDaemonAtomicValue",
            "workerDaemonAtomicValue",
            "workerDaemonHistoryValue",
            "health_state / lease_state / heartbeat_state / restart_policy_state / recovery_action",
        ]

        for fragment in expected_fragments:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, self.html)

    def test_atomic_formatter_helpers_are_present(self) -> None:
        expected_fragments = [
            "formatDaemonAtomic",
            "formatWorkerDaemonRecoveryTimeline",
            "deriveDaemonRecoveryTimelineAnomaly",
            "deriveDaemonRecoveryTimelineSuggestion",
            "deriveDaemonHealthState",
            "deriveLeaseState",
            "deriveHeartbeatState",
            "deriveRestartPolicyState",
            "deriveRecoveryAction",
            "deriveDaemonRecoveryTimelineEvent",
        ]

        for fragment in expected_fragments:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, self.html)


if __name__ == "__main__":
    unittest.main()
