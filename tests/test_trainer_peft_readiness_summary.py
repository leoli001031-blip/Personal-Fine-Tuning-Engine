from __future__ import annotations

import importlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
for package_dir in ("pfe-core", "pfe-cli", "pfe-server"):
    package_path = str(ROOT / package_dir)
    if package_path not in os.sys.path:
        os.sys.path.insert(0, package_path)

trainer_executor_module = importlib.import_module("pfe_core.trainer.executors")


class TrainerPeftReadinessSummaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.previous_cwd = os.getcwd()
        os.chdir(self.tempdir.name)

    def tearDown(self) -> None:
        os.chdir(self.previous_cwd)
        self.tempdir.cleanup()

    def _job_spec(self) -> dict[str, object]:
        return {
            "backend": "peft",
            "execution_backend": "peft",
            "execution_executor": "peft",
            "executor_mode": "real_import",
            "ready": True,
            "dry_run": False,
            "recipe": {"training": {"method": "qlora"}},
            "audit": {"import_probe": {"ready": False, "missing_modules": ["peft", "accelerate"]}},
            "training_examples": [
                {
                    "sample_id": "sample-1",
                    "instruction": "hello",
                    "chosen": "world",
                    "rejected": None,
                    "sample_type": "sft",
                }
            ],
        }

    def test_execute_peft_training_reports_readiness_summary_when_local_model_source_is_missing(self) -> None:
        job_spec = self._job_spec()
        job_spec["audit"] = {"import_probe": {"ready": True, "missing_modules": []}}
        fake_result = {
            "backend": "peft",
            "dry_run": False,
            "execution_mode": "real_import",
            "job_spec": dict(job_spec),
            "status": "completed",
            "real_execution": {
                "kind": "real_peft",
                "path": "real_import",
                "dependency_ready": True,
                "source_ready": False,
                "executor_ready": True,
                "blocked_by": [],
                "blocking_reasons": [],
                "success": True,
                "message": "real peft execution completed",
            },
        }
        with patch.object(
            trainer_executor_module,
            "_probe_real_peft_runtime",
            return_value={
                "available": True,
                "dependency_ready": True,
                "missing_modules": [],
                "required_modules": ["torch", "transformers", "peft", "accelerate"],
                "blocked_by": [],
                "blocking_reasons": [],
                "reason": "real peft runtime available",
            },
        ), patch.object(
            trainer_executor_module,
            "_run_real_import_peft_training",
            return_value=fake_result,
        ):
            result = trainer_executor_module.execute_peft_training(job_spec=job_spec, dry_run=False)

        readiness = result["readiness_summary"]
        self.assertTrue(readiness["available"])
        self.assertEqual(readiness["selected_execution_mode"], "real_import")
        self.assertFalse(readiness["real_local"]["available"])
        self.assertTrue(readiness["real_local"]["dependency_ready"])
        self.assertFalse(readiness["real_local"]["source_ready"])
        self.assertIn("no local base model path or config path available", readiness["real_local"]["reason"])
        self.assertIn("missing_local_model_source", readiness["real_local"]["blocked_by"])
        self.assertTrue(readiness["real_import"]["available"])
        self.assertTrue(readiness["real_import"]["dependency_ready"])
        self.assertTrue(readiness["real_import"]["executor_ready"])
        self.assertEqual(result["execution_mode"], "real_import")
        self.assertEqual(result["real_execution"]["kind"], "real_peft")
        self.assertTrue(result["real_execution"]["dependency_ready"])
        self.assertFalse(result["real_execution"]["source_ready"])
        self.assertEqual(result["real_execution"]["blocked_by"], [])

        job_summary = trainer_executor_module.summarize_training_job_execution(
            {"runner_result": result, "status": result["status"]}
        )
        real_summary = trainer_executor_module.summarize_real_training_execution(
            {"runner_result": result, "status": result["status"]}
        )
        self.assertEqual(job_summary["readiness_summary"], readiness)
        self.assertEqual(real_summary["readiness_summary"], readiness)
        self.assertEqual(real_summary["blocked_by"], [])
        self.assertTrue(real_summary["dependency_ready"])
        self.assertFalse(real_summary["source_ready"])

    def test_execute_peft_training_reports_local_model_readiness_when_config_exists(self) -> None:
        local_model_dir = Path(self.tempdir.name) / "local-model"
        local_model_dir.mkdir(parents=True, exist_ok=True)
        (local_model_dir / "config.json").write_text("{}", encoding="utf-8")

        job_spec = self._job_spec()
        job_spec["recipe"] = {
            "training": {
                "method": "qlora",
                "base_model_config_path": str(local_model_dir),
                "local_only": True,
            }
        }

        fake_result = {
            "backend": "peft",
            "dry_run": False,
            "execution_mode": "real_local",
            "job_spec": dict(job_spec),
            "status": "completed",
            "real_execution": {
                "kind": "real_local_peft",
                "path": "real_local",
                "success": True,
                "message": "real local training completed",
            },
        }
        with patch.object(
            trainer_executor_module,
            "_probe_real_local_runtime",
            return_value={
                "available": True,
                "missing_modules": [],
                "required_modules": ["torch", "transformers"],
                "reason": "local transformers runtime available",
            },
        ), patch.object(
            trainer_executor_module,
            "_probe_real_peft_runtime",
            return_value={
                "available": False,
                "missing_modules": ["peft"],
                "required_modules": ["torch", "transformers", "peft", "accelerate"],
                "reason": "missing required peft runtime modules",
            },
        ), patch.object(
            trainer_executor_module,
            "_run_real_local_peft_training",
            return_value=fake_result,
        ):
            result = trainer_executor_module.execute_peft_training(job_spec=job_spec, dry_run=False)

        readiness = result["readiness_summary"]
        self.assertTrue(readiness["available"])
        self.assertEqual(readiness["selected_execution_mode"], "real_local")
        self.assertTrue(readiness["real_local"]["available"])
        self.assertTrue(readiness["real_local"]["config_path"].endswith("local-model"))
        self.assertFalse(readiness["real_import"]["available"])
        self.assertEqual(result["execution_mode"], "real_local")
        self.assertEqual(result["status"], "completed")

    def test_execute_peft_training_prefers_real_import_when_both_execution_paths_are_available(self) -> None:
        local_model_dir = Path(self.tempdir.name) / "local-model"
        local_model_dir.mkdir(parents=True, exist_ok=True)
        (local_model_dir / "config.json").write_text("{}", encoding="utf-8")

        job_spec = self._job_spec()
        job_spec["audit"] = {"import_probe": {"ready": True, "missing_modules": []}}
        job_spec["recipe"] = {
            "training": {
                "method": "qlora",
                "base_model_path": str(local_model_dir),
                "base_model_config_path": str(local_model_dir),
                "local_only": True,
            }
        }
        fake_result = {
            "backend": "peft",
            "dry_run": False,
            "execution_mode": "real_import",
            "job_spec": dict(job_spec),
            "status": "completed",
            "real_execution": {
                "kind": "real_peft",
                "path": "real_import",
                "success": True,
                "message": "real peft execution completed",
            },
        }
        with patch.object(
            trainer_executor_module,
            "_probe_real_local_runtime",
            return_value={
                "available": True,
                "dependency_ready": True,
                "source_ready": True,
                "missing_modules": [],
                "required_modules": ["torch", "transformers"],
                "blocked_by": [],
                "blocking_reasons": [],
                "reason": "local transformers runtime available",
            },
        ), patch.object(
            trainer_executor_module,
            "_probe_real_peft_runtime",
            return_value={
                "available": True,
                "dependency_ready": True,
                "missing_modules": [],
                "required_modules": ["torch", "transformers", "peft", "accelerate"],
                "blocked_by": [],
                "blocking_reasons": [],
                "reason": "real peft runtime available",
            },
        ), patch.object(
            trainer_executor_module,
            "_run_real_import_peft_training",
            return_value=fake_result,
        ) as real_import_patch, patch.object(
            trainer_executor_module,
            "_run_real_local_peft_training",
            side_effect=AssertionError("real_local should not be preferred when real_import is available"),
        ):
            result = trainer_executor_module.execute_peft_training(job_spec=job_spec, dry_run=False)

        readiness = result["readiness_summary"]
        self.assertTrue(readiness["available"])
        self.assertEqual(readiness["selected_execution_mode"], "real_import")
        self.assertTrue(readiness["real_local"]["available"])
        self.assertTrue(readiness["real_import"]["available"])
        self.assertEqual(result["execution_mode"], "real_import")
        self.assertEqual(result["real_execution"]["kind"], "real_peft")
        real_import_patch.assert_called_once()


if __name__ == "__main__":
    unittest.main()
