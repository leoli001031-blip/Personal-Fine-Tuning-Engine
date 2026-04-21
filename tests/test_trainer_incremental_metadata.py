from __future__ import annotations

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

from pfe_core.trainer.service import TrainerService


class _IncrementalStore:
    def __init__(self, parent_path: Path):
        self.parent_path = parent_path
        self.load_calls: list[str] = []

    def load(self, version: str) -> str:
        self.load_calls.append(version)
        return str(self.parent_path)


class TrainerIncrementalMetadataTests(unittest.TestCase):
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

    def test_train_incremental_resolves_parent_adapter_context_explicitly(self) -> None:
        parent_dir = self.pfe_home / "adapters" / "user_default" / "20260324-010"
        parent_dir.mkdir(parents=True, exist_ok=True)
        (parent_dir / "adapter_manifest.json").write_text(
            """
            {
              "version": "20260324-010",
              "workspace": "user_default",
              "base_model": "Qwen/Test-Base",
              "artifact_format": "peft_lora",
              "state": "promoted",
              "num_samples": 12,
              "training_run_id": "20260324-010"
            }
            """.strip()
            + "\n",
            encoding="utf-8",
        )

        store = _IncrementalStore(parent_dir)
        trainer = TrainerService(store=store)

        context = trainer._resolve_incremental_parent_context(base_adapter="20260324-010", workspace="user_default")
        self.assertEqual(context["requested_base_adapter"], "20260324-010")
        self.assertEqual(context["parent_adapter_version"], "20260324-010")
        self.assertEqual(context["parent_adapter_path"], str(parent_dir))
        self.assertEqual(context["parent_base_model"], "Qwen/Test-Base")
        self.assertEqual(context["parent_artifact_format"], "peft_lora")
        self.assertEqual(context["parent_state"], "promoted")

        store.load_calls.clear()
        with patch.object(
            trainer,
            "train_result",
            return_value=SimpleNamespace(version="20260324-011", metrics={}),
        ) as train_result_mock:
            trainer.train_incremental(base_adapter="20260324-010", method="qlora", epochs=2, workspace="user_default")

        train_result_mock.assert_called_once()
        kwargs = train_result_mock.call_args.kwargs
        self.assertEqual(kwargs["base_model"], "Qwen/Test-Base")
        self.assertEqual(kwargs["workspace"], "user_default")
        self.assertEqual(kwargs["train_type"], "sft")
        self.assertIn("incremental_context", kwargs)
        self.assertEqual(kwargs["incremental_context"]["parent_adapter_version"], "20260324-010")
        self.assertEqual(kwargs["incremental_context"]["parent_adapter_path"], str(parent_dir))
        self.assertEqual(kwargs["incremental_context"]["source_adapter_version"], "20260324-010")
        self.assertEqual(kwargs["incremental_context"]["source_model"], "Qwen/Test-Base")
        self.assertEqual(store.load_calls, ["20260324-010"])


if __name__ == "__main__":
    unittest.main()
