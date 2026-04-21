import pathlib
import unittest


class ChatOpsConsoleSurfaceTest(unittest.TestCase):
    def test_sidebar_groups_and_history_panels_are_present(self) -> None:
        html = pathlib.Path(
            str(pathlib.Path(__file__).resolve().parents[1] / "pfe-server" / "pfe_server" / "static" / "chat.html")
        ).read_text(encoding="utf-8")

        expected_fragments = [
            "Core State",
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
            "Candidate / Queue",
            "Worker Loop",
            "Worker Runner Focus",
            "Worker Runner Latest Anomaly",
            "Worker Runner Suggested Action",
            "Candidate History",
            "Queue History",
            "Worker Runner",
            "Worker Runner Timeline / History",
            "workerRunnerHistoryValue",
            "formatWorkerRunnerHistory",
            "formatWorkerRunnerTimelineFocus",
            "formatWorkerRunnerTimelineAnomaly",
            "formatWorkerRunnerTimelineAction",
            "formatWorkerRunnerTimeline",
            "latest anomaly:",
            "current suggested action:",
            "Worker Daemon",
            "Daemon Timeline Focus",
            "Daemon Latest Anomaly",
            "Daemon Suggested Action",
            "formatWorkerDaemonRecoveryTimelineFocus",
            "formatWorkerDaemonRecoveryTimelineAnomaly",
            "formatWorkerDaemonRecoveryTimelineAction",
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
            "applyInspectionChip(elements.operationsCurrentFocusValue",
            "applyInspectionChip(elements.operationsPriorityActionValue",
            "applyInspectionChip(elements.operationsHandlingModeValue",
            "applyInspectionChip(elements.operationsLatestRecoveryValue",
            "applyInspectionBadge(elements.operationsCurrentFocusBadgeValue",
            "applyInspectionBadge(elements.operationsPriorityActionBadgeValue",
            "applyInspectionBadge(elements.operationsHandlingModeBadgeValue",
            "applyInspectionBadge(elements.operationsLatestRecoveryBadgeValue",
            "inspection-chip",
            "inspection-mini-badge",
            "required_action=",
            "escalation_mode=",
            "latest_recovery=",
        ]

        for fragment in expected_fragments:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, html)


if __name__ == "__main__":
    unittest.main()
