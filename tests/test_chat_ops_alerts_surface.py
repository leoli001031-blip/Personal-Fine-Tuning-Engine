from __future__ import annotations

import pathlib
import unittest


class ChatOpsAlertsSurfaceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.html = pathlib.Path(
            str(pathlib.Path(__file__).resolve().parents[1] / "pfe-server" / "pfe_server" / "static" / "chat.html")
        ).read_text(encoding="utf-8")

    def test_sidebar_exposes_alert_and_health_blocks(self) -> None:
        expected_fragments = [
            "Alerts / Health",
            "attention_needed / recovery / action policy / next actions / recent recovery",
            "Health",
            "Alerts",
            "Recovery",
            "Current Focus",
            "Priority Action",
            "Handling Mode",
            "Latest Recovery",
            "Action Policy",
            "Inspection Digest",
            "Next Actions",
            "operationsHealthValue",
            "operationsAlertsValue",
            "operationsRecoveryValue",
            "operationsCurrentFocusValue",
            "operationsCurrentFocusBadgeValue",
            "operationsPriorityActionValue",
            "operationsPriorityActionBadgeValue",
            "operationsHandlingModeValue",
            "operationsHandlingModeBadgeValue",
            "operationsLatestRecoveryValue",
            "operationsLatestRecoveryBadgeValue",
            "operationsPolicyDigestValue",
            "operationsInspectionDigestValue",
            "operationsNextActionsValue",
        ]

        for fragment in expected_fragments:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, self.html)

    def test_alert_formatter_functions_are_present(self) -> None:
        expected_fragments = [
            "formatOperationsHealth",
            "formatOperationsAlerts",
            "formatOperationsRecovery",
            "formatOperationsPolicyDigest",
            "formatOperationsInspectionDigest",
            "formatOperationsLatestRecovery",
            "formatOperationsNextActions",
            "deriveInspectionSeverity",
            "deriveInspectionModifiers",
            "applyInspectionChip",
            "formatInspectionBadge",
            "applyInspectionBadge",
            "dashboard.dashboard_digest",
            "daemon-lease=",
            "attention-needed",
            "primary=",
            "inspection-chip",
            "inspection-mini-badge",
            "requires-immediate",
            "requires-review",
        ]

        for fragment in expected_fragments:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, self.html)


if __name__ == "__main__":
    unittest.main()
