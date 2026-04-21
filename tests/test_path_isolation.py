from __future__ import annotations

import json
import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
for package_dir in ("pfe-core", "pfe-cli", "pfe-server"):
    package_path = str(ROOT / package_dir)
    if package_path not in os.sys.path:
        os.sys.path.insert(0, package_path)

from pfe_core.collector import ChatCollector, ChatInteraction, CollectorConfig
from pfe_core.config import PFEConfig
from pfe_core.pii_audit import PIIAuditLog, PIIWhitelist
from pfe_core.pii_detector import PIIDetectionResult
from pfe_core.router import create_router


class PathIsolationTests(unittest.TestCase):
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

    def test_pii_audit_and_whitelist_follow_pfe_home(self) -> None:
        audit = PIIAuditLog()
        whitelist = PIIWhitelist()

        self.assertEqual(audit.log_dir.resolve(), (self.pfe_home / "audit").resolve())
        self.assertEqual(whitelist.whitelist_path.resolve(), (self.pfe_home / "pii_whitelist.json").resolve())

    def test_chat_collector_writes_audit_logs_under_pfe_home(self) -> None:
        collector = ChatCollector(workspace="test_workspace", config=CollectorConfig(enabled=True))
        interaction = ChatInteraction(
            session_id="session_123",
            request_id="req_456",
            user_message="What is Python?",
            assistant_message="Python is a programming language.",
        )

        signals = collector.on_interaction(interaction, action="continue")

        self.assertEqual(len(signals), 1)
        audit_dir = self.pfe_home / "audit"
        log_files = list(audit_dir.glob("pii_audit_*.jsonl"))
        self.assertTrue(log_files)
        self.assertTrue(all(path.is_file() for path in log_files))

    def test_pii_audit_report_reads_legacy_naive_timestamps(self) -> None:
        audit = PIIAuditLog()
        log_file = audit.log_dir / f"pii_audit_{datetime.now(timezone.utc).strftime('%Y-%m')}.jsonl"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_file.write_text(
            json.dumps(
                {
                    "timestamp": datetime.now(timezone.utc).replace(tzinfo=None, microsecond=0).isoformat(),
                    "source_type": "signal",
                    "source_id": "legacy-signal",
                    "has_pii": False,
                    "pii_types": [],
                    "finding_count": 0,
                    "action_taken": "none",
                    "success": True,
                }
            )
            + "\n",
            encoding="utf-8",
        )

        report = audit.get_report(days=30)

        self.assertEqual(report.total_scanned, 1)
        self.assertEqual(report.pii_detected_count, 0)

    def test_pii_audit_rotation_uses_utc_month(self) -> None:
        audit = PIIAuditLog()
        utc_rollover = datetime(2026, 5, 1, 0, 30, tzinfo=timezone.utc)

        with patch("pfe_core.pii_audit._utc_now", return_value=utc_rollover):
            audit.log_detection(
                "signal",
                "utc-rollover",
                PIIDetectionResult(text_length=12),
            )

        log_file = audit.log_dir / "pii_audit_2026-05.jsonl"
        self.assertTrue(log_file.exists())
        self.assertIn("utc-rollover", log_file.read_text(encoding="utf-8"))

    def test_router_default_config_path_follows_pfe_home(self) -> None:
        router = create_router(config=PFEConfig.load())
        self.assertEqual(router._scenarios_config_path.resolve(), (self.pfe_home / "scenarios.json").resolve())
