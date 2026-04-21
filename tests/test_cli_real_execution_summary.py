from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for package_dir in ("pfe-core", "pfe-cli", "pfe-server"):
    package_path = str(ROOT / package_dir)
    if package_path not in os.sys.path:
        os.sys.path.insert(0, package_path)

from pfe_cli import main as cli_main
from pfe_cli.main import _format_serve_preview, _format_status


class CLIRealExecutionSummaryTests(unittest.TestCase):
    def test_status_shows_real_execution_and_export_toolchain_from_cli_state(self) -> None:
        previous_home = os.environ.get("PFE_HOME")
        try:
            with tempfile.TemporaryDirectory() as tempdir:
                pfe_home = Path(tempdir) / ".pfe"
                pfe_home.mkdir(parents=True, exist_ok=True)
                (pfe_home / "cli_state.json").write_text(
                    json.dumps(
                        {
                            "recorded_at": "2026-03-23T14:30:44.521066+00:00",
                            "workspace": "user_default",
                            "recent_training": {
                                "version": "20260323-006",
                                "state": "pending_eval",
                                "execution_backend": "mock_local",
                                "executor_mode": "phase0_mock",
                                "job_execution": {
                                    "status": "executed",
                                    "audit": {"runner_status": "prepared"},
                                },
                                "export_execution": {
                                    "status": "not_required",
                                    "attempted": False,
                                    "success": True,
                                    "metadata": {"execution_mode": "not_required"},
                                },
                            },
                        },
                        ensure_ascii=False,
                        indent=2,
                        sort_keys=True,
                    ),
                    encoding="utf-8",
                )
                os.environ["PFE_HOME"] = str(pfe_home)

                payload = {
                    "home": str(pfe_home),
                    "latest_adapter": {"version": "20260323-005", "state": "promoted"},
                    "latest_adapter_version": "20260323-005",
                    "recent_adapter": {"version": "20260323-006", "state": "pending_eval"},
                    "recent_adapter_version": "20260323-006",
                    "adapter_lifecycle": {"counts": {"pending_eval": 4, "promoted": 1, "archived": 1}},
                    "signal_count": 3,
                    "adapter_versions": 6,
                    "sample_counts": {"train": 15, "val": 0, "test": 5},
                    "trainer": {
                        "runtime": {"runtime_device": "cpu"},
                        "plans": {
                            "sft": {
                                "recommended_backend": "mock_local",
                                "requires_export_step": False,
                            }
                        },
                    },
                }

                text = _format_status(payload, workspace="user_default")
        finally:
            if previous_home is None:
                os.environ.pop("PFE_HOME", None)
            else:
                os.environ["PFE_HOME"] = previous_home

        from tests.matrix_test_compat import strip_ansi
        clean = strip_ansi(text)
        self.assertIn("latest promoted:", clean)
        self.assertIn("20260323-005 | state=promoted", clean)
        self.assertIn("recent training:", clean)
        self.assertIn("20260323-006 | state=pending_eval | execution_backend=mock_local | executor_mode=phase0_mock", clean)
        self.assertIn("REAL EXECUTION", clean)
        self.assertIn("status:", clean)
        self.assertIn("executed", clean)
        self.assertIn("audit runner status:", clean)
        self.assertIn("prepared", clean)
        self.assertIn("EXPORT TOOLCHAIN", clean)
        self.assertIn("not_required", clean)
        self.assertIn("attempted:", clean)
        self.assertIn("False", clean)
        self.assertIn("success:", clean)
        self.assertIn("True", clean)
        self.assertIn("meta execution_mode:", clean)
        self.assertIn("not_required", clean)

    def test_serve_preview_shows_real_execution_and_export_toolchain_from_cli_state(self) -> None:
        previous_home = os.environ.get("PFE_HOME")
        original_optional_call = cli_main._optional_module_call
        try:
            with tempfile.TemporaryDirectory() as tempdir:
                pfe_home = Path(tempdir) / ".pfe"
                pfe_home.mkdir(parents=True, exist_ok=True)
                (pfe_home / "cli_state.json").write_text(
                    json.dumps(
                        {
                            "recent_training": {
                                "version": "20260323-006",
                                "state": "pending_eval",
                                "execution_backend": "mock_local",
                                "executor_mode": "phase0_mock",
                                "job_execution": {
                                    "status": "executed",
                                    "audit": {"runner_status": "prepared"},
                                },
                                "export_execution": {
                                    "status": "not_required",
                                    "attempted": False,
                                    "success": True,
                                    "metadata": {"execution_mode": "not_required"},
                                },
                            }
                        },
                        ensure_ascii=False,
                        indent=2,
                        sort_keys=True,
                    ),
                    encoding="utf-8",
                )
                os.environ["PFE_HOME"] = str(pfe_home)

                def fake_optional_module_call(module_name: str, attr_name: str, *args: object, **kwargs: object):
                    if module_name == "pfe_server.app" and attr_name == "build_serve_plan":
                        return {
                            "runtime": {
                                "provider": "core",
                                "dry_run": True,
                                "uvicorn_available": True,
                                "app_target": "pfe_server.app:app",
                                "command": ["/usr/bin/python3", "-m", "uvicorn"],
                            },
                            "uvicorn_module": "uvicorn",
                        }
                    return original_optional_call(module_name, attr_name, *args, **kwargs)

                cli_main._optional_module_call = fake_optional_module_call
                text = _format_serve_preview(
                    port=8921,
                    host="127.0.0.1",
                    adapter="latest",
                    workspace="user_default",
                    api_key=None,
                    real_local=False,
                )
        finally:
            cli_main._optional_module_call = original_optional_call
            if previous_home is None:
                os.environ.pop("PFE_HOME", None)
            else:
                os.environ["PFE_HOME"] = previous_home

        from tests.matrix_test_compat import strip_ansi
        clean = strip_ansi(text)
        self.assertIn("SERVE PREVIEW", clean)
        self.assertIn("RECENT TRAINING", clean)
        self.assertIn("version:", clean)
        self.assertIn("20260323-006", clean)
        self.assertIn("state:", clean)
        self.assertIn("pending_eval", clean)
        self.assertIn("execution backend:", clean)
        self.assertIn("mock_local", clean)
        self.assertIn("executor mode:", clean)
        self.assertIn("phase0_mock", clean)
        self.assertIn("status:", clean)
        self.assertIn("executed", clean)
        self.assertIn("not_required", clean)


if __name__ == "__main__":
    unittest.main()
