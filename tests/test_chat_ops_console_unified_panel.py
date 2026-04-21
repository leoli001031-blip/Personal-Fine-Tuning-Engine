import pathlib
import unittest


class ChatOpsConsoleUnifiedPanelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.html = pathlib.Path(
            str(pathlib.Path(__file__).resolve().parents[1] / "pfe-server" / "pfe_server" / "static" / "chat.html")
        ).read_text(encoding="utf-8")

    def test_sidebar_reads_like_a_unified_operations_console(self) -> None:
        expected_fragments = [
            "Operations Console",
            "统一运营面板",
            "Operations Snapshot",
            "Auto Trigger",
            "Operations Event Stream",
            "Severity",
            "Attention",
            "Dashboard Digest",
            "Action Policy",
            "Latest Recovery",
            "Inspection Digest",
            "Current Focus",
            "Priority Action",
            "Handling Mode",
            "Candidate Timeline",
            "Queue History",
            "Worker Runner",
            "Worker Runner Focus",
            "Worker Runner Latest Anomaly",
            "Worker Runner Suggested Action",
            "Daemon Timeline Focus",
            "Daemon Latest Anomaly",
            "Daemon Suggested Action",
            "status-section",
            "ops-intro",
            "status-item full",
            "timeline-preview",
        ]

        for fragment in expected_fragments:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, self.html)

    def test_history_formatters_surface_distinct_operators_views(self) -> None:
        expected_fragments = [
            "PFE operations overview",
            "PFE operations event stream",
            "PFE candidate timeline",
            "PFE queue history",
            "PFE worker runner timeline",
            "formatOperationsEventStream",
            "operationsEventStreamSeverityValue",
            "operationsEventStreamAttentionValue",
            "operationsEventStreamDigestValue",
            "operationsEventStreamPolicyValue",
            "operationsEventStreamRecoveryValue",
            "operationsInspectionDigestValue",
            "operationsCurrentFocusValue",
            "operationsCurrentFocusBadgeValue",
            "operationsPriorityActionValue",
            "operationsPriorityActionBadgeValue",
            "operationsHandlingModeValue",
            "operationsHandlingModeBadgeValue",
            "operationsLatestRecoveryValue",
            "operationsLatestRecoveryBadgeValue",
            "event-stream-chip",
            "inspection-chip",
            "inspection-mini-badge",
            "requires-immediate",
            "requires-review",
            "formatQueueHistory",
            "formatWorkerRunnerHistory",
            "formatWorkerRunnerTimelineFocus",
            "formatWorkerRunnerTimelineAnomaly",
            "formatWorkerRunnerTimelineAction",
            "formatWorkerRunnerTimeline",
            "formatWorkerDaemonRecoveryTimelineFocus",
            "formatWorkerDaemonRecoveryTimelineAnomaly",
            "formatWorkerDaemonRecoveryTimelineAction",
            "current focus:",
            "latest anomaly:",
            "current suggested action:",
            "required_action=",
            "escalation_mode=",
            "latest_recovery=",
            "recent runner events: none",
        ]

        for fragment in expected_fragments:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, self.html)


if __name__ == "__main__":
    unittest.main()
