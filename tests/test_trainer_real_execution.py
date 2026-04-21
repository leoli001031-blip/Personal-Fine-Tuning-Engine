from __future__ import annotations

import importlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
for package_dir in ("pfe-core", "pfe-cli", "pfe-server"):
    package_path = str(ROOT / package_dir)
    if package_path not in os.sys.path:
        os.sys.path.insert(0, package_path)

trainer_executor_module = importlib.import_module("pfe_core.trainer.executors")


class TrainerRealExecutionTests(unittest.TestCase):
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

    def _job_spec(self) -> dict[str, object]:
        return {
            "backend": "peft",
            "execution_backend": "peft",
            "execution_executor": "peft",
            "executor_mode": "real_import",
            "ready": True,
            "dry_run": False,
            "recipe": {"backend": "peft"},
            "audit": {"import_probe": {"ready": True, "missing_modules": []}},
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

    def test_execute_peft_training_reports_structured_real_execution_when_dependencies_missing(self) -> None:
        job_spec = self._job_spec()
        with patch.object(
            trainer_executor_module,
            "_probe_real_peft_runtime",
            return_value={"available": False, "missing_modules": ["peft"], "required_modules": ["torch", "peft"]},
        ):
            result = trainer_executor_module.execute_peft_training(job_spec=job_spec, dry_run=False)

        self.assertEqual(result["backend"], "peft")
        self.assertEqual(result["status"], "ready")
        self.assertIn("real_execution", result)
        self.assertEqual(result["real_execution"]["kind"], "toy_local_peft")
        self.assertFalse(result["real_execution"]["attempted"])
        self.assertFalse(result["real_execution"]["available"])
        self.assertIn("missing_modules", result["real_execution"])
        self.assertIn("peft", result["real_execution"]["missing_modules"])

    def test_execute_peft_training_returns_successful_real_peft_real_execution(self) -> None:
        job_spec = self._job_spec()
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
                "train_loss": 0.123,
                "num_examples": 1,
            },
        }
        with patch.object(
            trainer_executor_module,
            "_probe_real_peft_runtime",
            return_value={"available": True, "missing_modules": [], "required_modules": ["torch", "transformers", "peft", "accelerate"]},
        ), patch.object(
            trainer_executor_module,
            "_run_real_import_peft_training",
            return_value=fake_result,
        ):
            result = trainer_executor_module.execute_peft_training(job_spec=job_spec, dry_run=False)

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["execution_mode"], "real_import")
        self.assertEqual(result["real_execution"]["kind"], "real_peft")
        self.assertTrue(result["real_execution"]["success"])
        self.assertEqual(result["real_execution"]["message"], "real peft execution completed")
        summary = trainer_executor_module.summarize_real_training_execution({"runner_result": result, "status": result["status"]})
        self.assertEqual(summary["path"], "real_import")
        self.assertEqual(summary["kind"], "real_peft")

    def test_run_materialized_training_job_bundle_recovers_result_from_result_json(self) -> None:
        backend_dispatch = {
            "requested_backend": "peft",
            "execution_backend": "peft",
            "execution_executor": "peft",
            "executor_mode": "real_import",
            "reasons": ["using requested backend peft"],
            "requires_export_step": False,
            "export_steps": [],
            "export_format": None,
            "export_backend": None,
            "capability": {"artifact_format": "peft_lora"},
        }
        plan = trainer_executor_module.build_training_execution_recipe(
            backend_dispatch=backend_dispatch,
            runtime={"runtime_device": "cpu"},
            method="qlora",
            epochs=1,
            train_type="sft",
            base_model_name="mock-llama-target",
            num_train_samples=1,
            num_fresh_samples=1,
            num_replay_samples=0,
            replay_ratio=0.0,
            train_examples=[
                {
                    "sample_id": "sample-1",
                    "instruction": "hello",
                    "chosen": "world",
                    "rejected": None,
                    "sample_type": "sft",
                }
            ],
            allow_mock_fallback=True,
        )
        bundle = trainer_executor_module.materialize_training_job_bundle(
            execution_plan=plan,
            output_dir=self.pfe_home / "adapters" / "user_default" / "20260323-777",
        )
        result_json_path = Path(bundle.result_json_path)
        expected_payload = {
            "backend": "peft",
            "status": "completed",
            "real_execution": {
                "kind": "toy_local_peft",
                "success": True,
                "num_examples": 1,
            },
        }
        result_json_path.write_text(json.dumps(expected_payload, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")

        fake_completed = SimpleNamespace(returncode=0, stdout="", stderr="")
        with patch.object(trainer_executor_module.subprocess, "run", return_value=fake_completed):
            run_result = trainer_executor_module.run_materialized_training_job_bundle(bundle, force_dry_run=False)

        self.assertTrue(run_result.attempted)
        self.assertTrue(run_result.success)
        self.assertEqual(run_result.status, "executed")
        self.assertEqual(run_result.runner_result["real_execution"]["kind"], "toy_local_peft")
        self.assertTrue(run_result.runner_result["real_execution"]["success"])
        self.assertEqual(run_result.metadata["result_json_path"], str(result_json_path))
        self.assertEqual(run_result.audit["result_json_path"], str(result_json_path))


if __name__ == "__main__":
    unittest.main()
