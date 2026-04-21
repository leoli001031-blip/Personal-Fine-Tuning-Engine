from __future__ import annotations

import json
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

from pfe_core.trainer.executors import execute_peft_training, summarize_real_training_execution


class TrainerPeftRuntimePathTests(unittest.TestCase):
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
            "recipe": {
                "training": {
                    "method": "qlora",
                    "base_model_config_path": str(Path(self.tempdir.name) / "local-model"),
                    "local_only": True,
                }
            },
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

    def test_execute_peft_training_uses_real_local_config_path_without_peft_dependency(self) -> None:
        local_model_dir = Path(self.tempdir.name) / "local-model"
        local_model_dir.mkdir(parents=True, exist_ok=True)
        (local_model_dir / "config.json").write_text(
            json.dumps(
                {
                    "architectures": ["GPT2LMHeadModel"],
                    "model_type": "gpt2",
                    "vocab_size": 32,
                    "n_positions": 32,
                    "n_ctx": 32,
                    "n_embd": 16,
                    "n_layer": 1,
                    "n_head": 1,
                    "bos_token_id": 1,
                    "eos_token_id": 2,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        artifact_dir = Path(self.tempdir.name) / "real-local-output"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_manifest = artifact_dir / "real_local_job_manifest.json"
        summary_path = artifact_dir / "training_summary.json"
        real_execution_path = artifact_dir / "real_execution.json"
        for file_path in (artifact_manifest, summary_path, real_execution_path):
            file_path.write_text("{}", encoding="utf-8")

        fake_result = {
            "backend": "peft",
            "dry_run": False,
            "execution_mode": "real_local",
            "job_spec": self._job_spec(),
            "status": "completed",
            "real_execution": {
                "kind": "real_local_peft",
                "path": "real_local",
                "source_kind": "config",
                "load_mode": "from_config",
                "artifact_manifest_path": str(artifact_manifest),
                "summary_path": str(summary_path),
                "real_execution_path": str(real_execution_path),
                "success": True,
            },
        }

        with patch(
            "pfe_core.trainer.executors._probe_real_local_runtime",
            return_value={"available": True, "missing_modules": [], "required_modules": ["torch", "transformers"]},
        ), patch(
            "pfe_core.trainer.executors._probe_real_peft_runtime",
            return_value={
                "available": False,
                "dependency_ready": False,
                "missing_modules": ["peft"],
                "required_modules": ["torch", "transformers", "peft", "accelerate"],
                "blocked_by": ["missing_module:peft"],
                "blocking_reasons": ["missing required module: peft"],
                "reason": "missing required peft runtime modules",
            },
        ), patch(
            "pfe_core.trainer.executors._run_real_local_peft_training",
            return_value=fake_result,
        ):
            result = execute_peft_training(job_spec=self._job_spec(), dry_run=False)

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["execution_mode"], "real_local")
        self.assertEqual(result["real_execution"]["kind"], "real_local_peft")
        self.assertEqual(result["real_execution"]["path"], "real_local")
        self.assertEqual(result["real_execution"]["source_kind"], "config")
        self.assertEqual(result["real_execution"]["load_mode"], "from_config")
        self.assertTrue(Path(result["real_execution"]["artifact_manifest_path"]).exists())
        self.assertTrue(Path(result["real_execution"]["summary_path"]).exists())
        self.assertTrue(Path(result["real_execution"]["real_execution_path"]).exists())

        summary = summarize_real_training_execution({"runner_result": result, "status": result["status"]})
        self.assertEqual(summary["path"], "real_local")
        self.assertEqual(summary["kind"], "real_local_peft")
        self.assertEqual(summary["source_kind"], "config")
        self.assertEqual(summary["load_mode"], "from_config")

    def test_execute_peft_training_reports_unavailable_when_no_local_source_and_no_peft(self) -> None:
        job_spec = {
            "backend": "peft",
            "execution_backend": "peft",
            "execution_executor": "peft",
            "executor_mode": "real_import",
            "ready": True,
            "dry_run": False,
            "recipe": {"training": {"method": "qlora"}},
            "audit": {"import_probe": {"ready": False, "missing_modules": ["peft"]}},
            "training_examples": [{"sample_id": "sample-1", "instruction": "hello", "chosen": "world"}],
        }

        with patch(
            "pfe_core.trainer.executors._probe_real_peft_runtime",
            return_value={
                "available": False,
                "missing_modules": ["peft"],
                "required_modules": ["torch", "transformers", "peft", "accelerate"],
                "reason": "missing required peft runtime modules",
            },
        ):
            result = execute_peft_training(job_spec=job_spec, dry_run=False)

        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["real_execution"]["path"], "unavailable")
        self.assertEqual(result["real_execution"]["kind"], "toy_local_peft")
        self.assertFalse(result["real_execution"]["available"])
        self.assertIn("missing_modules", result["real_execution"])


if __name__ == "__main__":
    unittest.main()
