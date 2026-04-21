from __future__ import annotations

import importlib
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

trainer_executor_module = importlib.import_module("pfe_core.trainer.executors")


class TrainerRealPeftJobTests(unittest.TestCase):
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
                    "base_model": "mock-base",
                    "num_fresh_samples": 1,
                }
            },
            "preference_reinforced_fresh_sample_count": 1,
            "preference_reinforced_fresh_sample_ids": ["sample-1"],
            "preference_reinforced_fresh_sample_ratio": 1.0,
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

    def test_materialize_toy_peft_job_artifacts_writes_expected_files(self) -> None:
        bundle = trainer_executor_module._materialize_toy_peft_job_artifacts(
            output_dir=Path("artifacts"),
            job_spec=self._job_spec(),
            training_examples=[{"sample_id": "sample-1"}],
            train_loss=0.125,
            execution_mode="real_import",
            run_status="completed",
        )

        artifact_dir = Path(bundle["artifact_dir"])
        self.assertTrue(artifact_dir.exists())
        self.assertTrue((artifact_dir / "adapter_model.safetensors").exists())
        self.assertTrue((artifact_dir / "adapter_config.json").exists())
        self.assertTrue((Path(bundle["trainer_state_path"])).exists())
        self.assertTrue((Path(bundle["summary_path"])).exists())
        self.assertTrue((Path(bundle["real_execution_path"])).exists())
        self.assertTrue((Path(bundle["artifact_manifest_path"])).exists())
        self.assertIn("adapter_model", bundle["artifacts"])
        self.assertEqual(bundle["metrics"]["num_examples"], 1)
        self.assertEqual(bundle["metrics"]["loss"], 0.125)
        self.assertEqual(bundle["metrics"]["preference_reinforced_fresh_sample_count"], 1)
        self.assertEqual(bundle["metrics"]["preference_reinforced_fresh_sample_ids"], ["sample-1"])
        self.assertEqual(bundle["metrics"]["preference_reinforced_fresh_sample_ratio"], 1.0)

        metrics_path = Path(bundle["metrics_path"])
        self.assertTrue(metrics_path.exists())
        metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        self.assertEqual(metrics_payload["loss"], 0.125)
        self.assertEqual(metrics_payload["num_examples"], 1)
        self.assertEqual(metrics_payload["preference_reinforced_fresh_sample_count"], 1)
        self.assertEqual(metrics_payload["preference_reinforced_fresh_sample_ids"], ["sample-1"])

        summary = json.loads(Path(bundle["summary_path"]).read_text(encoding="utf-8"))
        self.assertEqual(summary["backend"], "peft")
        self.assertEqual(summary["status"], "completed")
        self.assertIn("artifact_dir", summary)
        self.assertEqual(summary["preference_reinforced_fresh_sample_count"], 1)
        self.assertEqual(summary["preference_reinforced_fresh_sample_ids"], ["sample-1"])
        self.assertEqual(summary["preference_reinforced_fresh_sample_ratio"], 1.0)

        real_execution = json.loads(Path(bundle["real_execution_path"]).read_text(encoding="utf-8"))
        self.assertEqual(real_execution["status"], "completed")
        self.assertEqual(real_execution["num_examples"], 1)
        self.assertEqual(real_execution["preference_reinforced_fresh_sample_count"], 1)
        self.assertEqual(real_execution["preference_reinforced_fresh_sample_ids"], ["sample-1"])
        self.assertEqual(real_execution["preference_reinforced_fresh_sample_ratio"], 1.0)

        manifest = json.loads(Path(bundle["artifact_manifest_path"]).read_text(encoding="utf-8"))
        self.assertEqual(manifest["metadata"]["training"]["preference_reinforced_fresh_sample_count"], 1)
        self.assertEqual(manifest["metadata"]["training"]["preference_reinforced_fresh_sample_ids"], ["sample-1"])
        self.assertEqual(manifest["metadata"]["training"]["preference_reinforced_fresh_sample_ratio"], 1.0)

    def test_materialize_toy_peft_job_artifacts_preserves_existing_adapter_files(self) -> None:
        artifact_dir = Path("artifacts") / "peft_lora"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        existing_model = artifact_dir / "adapter_model.safetensors"
        existing_config = artifact_dir / "adapter_config.json"
        existing_model.write_bytes(b"real-adapter-binary")
        existing_config.write_text('{"real": true}\n', encoding="utf-8")

        bundle = trainer_executor_module._materialize_toy_peft_job_artifacts(
            output_dir=Path("artifacts"),
            job_spec=self._job_spec(),
            training_examples=[{"sample_id": "sample-1"}],
            train_loss=0.25,
            execution_mode="real_import",
            run_status="completed",
            preserve_existing_adapter_files=True,
        )

        self.assertEqual(existing_model.read_bytes(), b"real-adapter-binary")
        self.assertEqual(existing_config.read_text(encoding="utf-8"), '{"real": true}\n')
        self.assertEqual(Path(bundle["artifact_dir"]), artifact_dir)

    def test_execute_peft_training_surfaces_artifact_bundle_metadata(self) -> None:
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
                "num_examples": 1,
                "train_loss": 0.125,
                "output_dir": "/tmp/pfe-real-peft",
                "artifact_dir": "/tmp/pfe-real-peft/peft_lora",
                "artifact_manifest_path": "/tmp/pfe-real-peft/real_peft_job_manifest.json",
                "summary_path": "/tmp/pfe-real-peft/training_summary.json",
                "metrics": {"loss": 0.125, "num_examples": 1},
                "artifacts": {"adapter_model": "/tmp/pfe-real-peft/peft_lora/adapter_model.safetensors"},
                "success": True,
                "message": "real peft execution completed",
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
        self.assertEqual(result["real_execution"]["artifact_dir"], "/tmp/pfe-real-peft/peft_lora")
        self.assertIn("artifact_manifest_path", result["real_execution"])

    def test_run_real_local_peft_training_keeps_artifact_dir_adapter_only(self) -> None:
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
        job_spec = {
            "backend": "peft",
            "execution_backend": "peft",
            "execution_executor": "peft",
            "executor_mode": "real_local",
            "ready": True,
            "dry_run": False,
            "recipe": {
                "training": {
                    "method": "qlora",
                    "base_model_config_path": str(local_model_dir),
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

        result = trainer_executor_module._run_real_local_peft_training(job_spec)

        artifact_dir = Path(result["real_execution"]["artifact_dir"])
        self.assertTrue((artifact_dir / "adapter_config.json").exists())
        self.assertTrue((artifact_dir / "real_local_model.safetensors").exists())
        self.assertFalse(any(artifact_dir.glob("model-*.safetensors")))
        self.assertFalse((artifact_dir / "config.json").exists())
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["real_execution"]["kind"], "real_local_peft")


if __name__ == "__main__":
    unittest.main()
