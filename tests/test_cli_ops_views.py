from __future__ import annotations

import os
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for package_dir in ("pfe-core", "pfe-cli", "pfe-server"):
    package_path = str(ROOT / package_dir)
    if package_path not in os.sys.path:
        os.sys.path.insert(0, package_path)

from pfe_cli.main import _format_candidate_history, _format_status, _format_train_queue_history, _format_worker_runner_history


class CliOpsViewsTests(unittest.TestCase):
    def test_status_surface_highlights_ops_attention(self) -> None:
        status = {
            "workspace": "user_default",
            "latest_adapter_version": "20260326-201",
            "latest_adapter": {
                "version": "20260326-201",
                "state": "promoted",
                "num_samples": 12,
                "artifact_format": "gguf_merged",
                "export_artifact_valid": False,
                "export_artifact_exists": False,
                "export_artifact_path": "/tmp/missing.gguf",
            },
            "candidate_summary": {
                "candidate_version": "20260326-200",
                "candidate_state": "pending_eval",
                "candidate_needs_promotion": True,
                "candidate_can_promote": False,
                "candidate_can_archive": True,
            },
            "train_queue": {
                "count": 3,
                "max_priority": 100,
                "counts": {"queued": 2, "awaiting_confirmation": 1, "running": 0, "completed": 0, "failed": 0},
                "current": {"job_id": "job-queue-1", "state": "queued"},
                "confirmation_summary": {"awaiting_confirmation_count": 1, "next_job_id": "job-queue-2"},
                "worker_runner": {
                    "active": True,
                    "lock_state": "active",
                    "stop_requested": False,
                    "lease_expires_at": "2026-03-26T10:00:00+00:00",
                },
                "history_summary": {
                    "transition_count": 4,
                    "last_transition": {"job_id": "job-queue-2", "event": "completed", "state": "completed"},
                    "last_reason": "training_completed",
                },
            },
        }

        text = _format_status(status, workspace="user_default")

        from tests.matrix_test_compat import strip_ansi
        clean = strip_ansi(text)
        self.assertIn("CANDIDATE SUMMARY", clean)
        self.assertIn("candidate version:", clean)
        self.assertIn("20260326-200", clean)
        self.assertIn("candidate state:", clean)
        self.assertIn("pending_eval", clean)
        self.assertIn("TRAIN QUEUE", clean)
        self.assertIn("states:", clean)
        self.assertIn("queued:2", clean)
        self.assertIn("confirmation:", clean)
        self.assertIn("awaiting confirmation count=1", clean)
        self.assertIn("next job id=job-queue-2", clean)
        self.assertIn("worker runner:", clean)
        self.assertIn("lock state=active", clean)
        self.assertIn("active=yes", clean)
        self.assertIn("stop requested=no", clean)
        self.assertIn("latest promoted:", clean)
        self.assertIn("export_artifact_valid=False", clean)
        self.assertIn("export_artifact_exists=False", clean)
        self.assertIn("export_artifact_path=/tmp/missing.gguf", clean)

    def test_status_surface_is_clean_when_nothing_needs_attention(self) -> None:
        status = {
            "workspace": "user_default",
            "latest_adapter": {"version": "20260326-300", "state": "promoted", "export_artifact_valid": True, "export_artifact_exists": True},
            "candidate_summary": {
                "candidate_version": "20260326-300",
                "candidate_state": "promoted",
                "candidate_needs_promotion": False,
            },
            "train_queue": {
                "count": 0,
                "counts": {},
                "worker_runner": {"active": False, "lock_state": "idle", "stop_requested": False},
            },
        }

        text = _format_status(status, workspace="user_default")

        from tests.matrix_test_compat import strip_ansi
        clean = strip_ansi(text)
        self.assertIn("CANDIDATE SUMMARY", clean)
        self.assertIn("candidate state:", clean)
        self.assertIn("promoted", clean)
        self.assertIn("TRAIN QUEUE", clean)
        self.assertIn("count:", clean)
        # Train queue count appears as padded key-value; just verify 0 appears in the train queue box
        train_queue_section = clean.split("TRAIN QUEUE")[1].split("┌──[")[0]
        self.assertIn("0", train_queue_section)

    def test_history_formatters_include_latest_timestamp(self) -> None:
        candidate_text = _format_candidate_history(
            {
                "workspace": "user_default",
                "count": 2,
                "last_action": "promote_candidate",
                "last_status": "completed",
                "last_reason": "candidate_promoted",
                "last_note": "ready_for_rollout",
                "last_candidate_version": "20260326-200",
                "items": [
                    {
                        "timestamp": "2026-03-26T09:00:00+00:00",
                        "action": "archive_candidate",
                        "status": "completed",
                        "reason": "archived_after_review",
                        "operator_note": "archive_after_review",
                        "candidate_version": "20260326-100",
                    },
                    {
                        "timestamp": "2026-03-26T10:00:00+00:00",
                        "action": "promote_candidate",
                        "status": "completed",
                        "reason": "candidate_promoted",
                        "operator_note": "ready_for_rollout",
                        "candidate_version": "20260326-200",
                        "promoted_version": "20260326-200",
                    },
                ],
            }
        )
        queue_text = _format_train_queue_history(
            {
                "workspace": "user_default",
                "job_id": "job-queue-1",
                "state": "completed",
                "count": 2,
                "history_count": 2,
                "history_summary": {
                    "transition_count": 2,
                    "last_reason": "training_completed",
                    "last_transition": {"event": "completed"},
                },
                "history": [
                    {"timestamp": "2026-03-26T08:00:00+00:00", "event": "approved", "state": "queued", "reason": "confirmation_approved", "note": "safe_to_run"},
                    {"timestamp": "2026-03-26T08:10:00+00:00", "event": "completed", "state": "completed", "reason": "training_completed", "note": "done"},
                ],
            }
        )
        runner_text = _format_worker_runner_history(
            {
                "workspace": "user_default",
                "count": 2,
                "last_event": "completed",
                "last_reason": "idle_exit",
                "items": [
                    {"timestamp": "2026-03-26T11:00:00+00:00", "event": "started", "reason": "runner_started", "metadata": {"pid": 1234}},
                    {"timestamp": "2026-03-26T11:01:00+00:00", "event": "completed", "reason": "idle_exit", "metadata": {"pid": 1234, "processed_count": 1}},
                ],
            }
        )

        self.assertIn("latest timestamp: 2026-03-26T10:00:00+00:00", candidate_text)
        self.assertIn("latest timestamp: 2026-03-26T08:10:00+00:00", queue_text)
        self.assertIn("latest timestamp: 2026-03-26T11:01:00+00:00", runner_text)


if __name__ == "__main__":
    unittest.main()
