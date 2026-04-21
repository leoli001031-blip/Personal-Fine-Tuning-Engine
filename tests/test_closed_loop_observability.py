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
from pfe_core.adapter_store.store import AdapterStore
from pfe_core.pipeline import PipelineService
from tests.matrix_test_compat import strip_ansi


class ClosedLoopObservabilityTests(unittest.TestCase):
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

    def _build_status_payload(self, *, real_execution_kind: str) -> dict[str, object]:
        return {
            "home": str(self.pfe_home),
            "latest_adapter": {"version": "20260324-101", "state": "promoted", "samples": 7, "format": "peft_lora"},
            "latest_adapter_version": "20260324-101",
            "recent_adapter": {"version": "20260324-102", "state": "pending_eval", "samples": 5, "format": "peft_lora"},
            "recent_adapter_version": "20260324-102",
            "adapter_lifecycle": {"counts": {"pending_eval": 1, "promoted": 1}},
            "signal_count": 1,
            "adapter_versions": 2,
            "sample_counts": {"train": 12, "val": 1, "test": 1},
            "real_execution_summary": {
                "status": "completed" if real_execution_kind != "unavailable" else "unavailable",
                "state": "completed" if real_execution_kind != "unavailable" else "unavailable",
                "kind": real_execution_kind,
                "backend": "mock_local",
                "runner_status": "completed" if real_execution_kind != "unavailable" else "unavailable",
                "execution_mode": "real_import" if real_execution_kind == "real_local" else "phase0_mock",
                "attempted": real_execution_kind != "unavailable",
                "available": real_execution_kind != "unavailable",
                "success": real_execution_kind != "unavailable",
                "output_dir": str(self.pfe_home / "adapters" / "user_default" / "20260324-102"),
            },
            "export_toolchain_summary": {
                "status": "not_required",
                "toolchain_status": "not_required",
                "execution_mode": "not_required",
                "attempted": False,
                "success": True,
            },
            "trainer": {
                "runtime": {"runtime_device": "cpu"},
                "plans": {
                    "sft": {
                        "recommended_backend": "mock_local",
                        "requires_export_step": False,
                    }
                },
                "last_run": {
                    "version": "20260324-102",
                    "execution_backend": "mock_local",
                    "job_execution_summary": {
                        "state": "executed",
                        "attempted": True,
                        "success": True,
                        "executor_mode": "phase0_mock",
                    },
                    "real_execution_summary": {
                        "state": "completed" if real_execution_kind != "unavailable" else "unavailable",
                        "kind": real_execution_kind,
                        "backend": "mock_local",
                        "runner_status": "completed" if real_execution_kind != "unavailable" else "unavailable",
                        "execution_mode": "real_import" if real_execution_kind == "real_local" else "phase0_mock",
                        "attempted": real_execution_kind != "unavailable",
                        "available": real_execution_kind != "unavailable",
                        "success": real_execution_kind != "unavailable",
                        "output_dir": str(self.pfe_home / "adapters" / "user_default" / "20260324-102"),
                    },
                    "export_toolchain_summary": {
                        "status": "not_required",
                        "toolchain_status": "not_required",
                        "execution_mode": "not_required",
                        "attempted": False,
                        "success": True,
                    },
                },
            },
        }

    def test_status_surfaces_real_execution_kind_without_mixing_latest_and_recent(self) -> None:
        for kind in ("real_local", "toy_local", "unavailable"):
            with self.subTest(kind=kind):
                text = _format_status(self._build_status_payload(real_execution_kind=kind), workspace="user_default")
                clean = strip_ansi(text)
                self.assertIn("[ ADAPTER LIFECYCLE ]", clean)
                self.assertIn("latest promoted:         20260324-101 | state=promoted", clean)
                self.assertIn("recent training:         20260324-102 | state=pending_eval", clean)
                self.assertIn("[ REAL EXECUTION ]", clean)
                self.assertIn(f"status:                  {'completed' if kind != 'unavailable' else 'unavailable'}", clean)
                self.assertIn(f"kind:                    {kind}", clean)
                self.assertIn("[ EXPORT TOOLCHAIN ]", clean)
                self.assertIn("status:                  not_required", clean)

    def test_serve_preview_surfaces_real_execution_kind_without_mixing_latest_and_recent(self) -> None:
        pipeline = PipelineService()
        pipeline.generate(scenario="life-coach", style="warm", num_samples=8)
        first_result = pipeline.train_result(method="qlora", epochs=1, train_type="sft")
        store = AdapterStore(home=self.pfe_home)
        store.promote(first_result.version)

        pipeline.generate(scenario="work-coach", style="direct", num_samples=8)
        second_result = pipeline.train_result(method="qlora", epochs=1, train_type="sft")

        original_optional_call = cli_main._optional_module_call
        try:
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

            for kind in ("real_local", "toy_local", "unavailable"):
                with self.subTest(kind=kind):
                    cli_state = {
                        "recorded_at": "2026-03-24T12:00:00+08:00",
                        "workspace": "user_default",
                        "recent_training": {
                            "version": second_result.version,
                            "state": "pending_eval",
                            "execution_backend": "mock_local",
                            "executor_mode": "phase0_mock",
                            "job_execution": {
                                "status": "executed",
                                "audit": {"runner_status": "prepared"},
                            },
                            "real_execution_summary": {
                                "status": "completed" if kind != "unavailable" else "unavailable",
                                "state": "completed" if kind != "unavailable" else "unavailable",
                                "kind": kind,
                                "backend": "mock_local",
                                "runner_status": "completed" if kind != "unavailable" else "unavailable",
                                "execution_mode": "real_import" if kind == "real_local" else "phase0_mock",
                                "attempted": kind != "unavailable",
                                "available": kind != "unavailable",
                                "success": kind != "unavailable",
                            },
                            "export_toolchain_summary": {
                                "status": "not_required",
                                "toolchain_status": "not_required",
                                "execution_mode": "not_required",
                                "attempted": False,
                                "success": True,
                            },
                        },
                    }
                    (self.pfe_home / "cli_state.json").write_text(
                        json.dumps(cli_state, ensure_ascii=False, indent=2, sort_keys=True),
                        encoding="utf-8",
                    )

                    text = _format_serve_preview(
                        port=8921,
                        host="127.0.0.1",
                        adapter="latest",
                        workspace="user_default",
                        api_key=None,
                        real_local=False,
                    )
                    clean = strip_ansi(text)

                    self.assertIn("[ LATEST PROMOTED ]", clean)
                    self.assertIn(f"version:                 {first_result.version}", clean)
                    self.assertIn("state:                   promoted", clean)
                    self.assertIn("[ RECENT TRAINING ]", clean)
                    self.assertIn(f"version:                 {second_result.version}", clean)
                    self.assertIn("state:                   pending_eval", clean)
                    self.assertIn(f"status:                  {'completed' if kind != 'unavailable' else 'unavailable'}", clean)
                    self.assertIn(f"kind:                    {kind}", clean)
                    self.assertIn("status:                  not_required", clean)
        finally:
            cli_main._optional_module_call = original_optional_call


if __name__ == "__main__":
    unittest.main()
