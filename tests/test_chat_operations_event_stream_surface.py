import pathlib
import unittest


class ChatOperationsEventStreamSurfaceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.html = pathlib.Path(
            str(pathlib.Path(__file__).resolve().parents[1] / "pfe-server" / "pfe_server" / "static" / "chat.html")
        ).read_text(encoding="utf-8")

    def test_sidebar_exposes_operations_event_stream_panel(self) -> None:
        expected_fragments = [
            "Operations Event Stream",
            "candidate / queue / runner / daemon timeline summary",
            "Severity",
            "Attention",
            "operationsEventStreamValue",
            "operationsEventStreamSeverityValue",
            "operationsEventStreamAttentionValue",
            "operationsEventStreamDigestValue",
            "operationsEventStreamPolicyValue",
            "operationsEventStreamRecoveryValue",
            "operationsEventStreamRecentValue",
            "formatOperationsEventStream",
            "event-stream-chip",
            "structuredStream.severity",
            "attention-needed",
            "Dashboard Digest",
            "Action Policy",
            "required_action=",
            "escalation_mode=",
            "latest_recovery=",
            "summary: ",
            "recent stream: ",
        ]

        for fragment in expected_fragments:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, self.html)


if __name__ == "__main__":
    unittest.main()
