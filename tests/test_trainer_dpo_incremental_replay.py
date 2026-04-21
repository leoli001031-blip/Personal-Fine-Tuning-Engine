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

from pfe_core.config import PFEConfig, TrainerConfig
from pfe_core.errors import TrainingError
from pfe_core.trainer.service import TrainerService

trainer_service_module = __import__("pfe_core.trainer.service", fromlist=["TrainerService"])


class _TrainerStore:
    def __init__(self, home: Path, version_dir: Path | None = None, latest_version: str | None = None):
        self.home = home
        self.version_dir = version_dir
        self.latest_version = latest_version
        self.created_training_config: dict[str, object] | None = None
        self.load_calls: list[str] = []

    def load(self, version: str) -> str:
        self.load_calls.append(version)
        if version in {"latest", self.latest_version} and self.version_dir is not None:
            return str(self.version_dir)
        raise RuntimeError(f"missing adapter version: {version}")

    def current_latest_version(self) -> str | None:
        return self.latest_version

    def list_version_records(self, limit: int = 20) -> list[dict[str, object]]:
        del limit
        if self.latest_version is None:
            return []
        return [{"version": self.latest_version}]

    def create_training_version(
        self,
        *,
        base_model: str,
        training_config: dict[str, object],
        artifact_format: str = "peft_lora",
    ) -> dict[str, object]:
        del base_model, artifact_format
        self.created_training_config = dict(training_config)
        assert self.version_dir is not None
        self.version_dir.mkdir(parents=True, exist_ok=True)
        return {"version": "20260401-001", "path": str(self.version_dir), "manifest": {"version": "20260401-001"}}

    def mark_pending_eval(self, version: str, *, num_samples: int, metrics: dict[str, object] | None = None) -> None:
        del version, num_samples, metrics


class TrainerDpoIncrementalReplayTests(unittest.TestCase):
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

    def _sample(
        self,
        sample_id: str,
        *,
        sample_type: str = "dpo",
        used_in_version: str | None = None,
    ) -> dict[str, object]:
        return {
            "sample_id": sample_id,
            "sample_type": sample_type,
            "instruction": f"instruction-{sample_id}",
            "chosen": f"chosen-{sample_id}",
            "rejected": f"rejected-{sample_id}",
            "score": 0.9,
            "source": "signal",
            "source_event_ids": [sample_id],
            "source_adapter_version": "20260331-001",
            "created_at": "2026-04-01T00:00:00+00:00",
            "used_in_version": used_in_version,
            "metadata": {"dataset_split": "train"},
        }

    def _config(
        self,
        *,
        replay_ratio: float = 0.5,
        dpo_replay_ratio: float = 0.75,
        replay_history_limit: int = 20,
        replay_min_samples: int = 1,
        incremental_parent_selector: str = "promoted_or_latest",
    ) -> PFEConfig:
        return PFEConfig(
            trainer=TrainerConfig(
                replay_ratio=replay_ratio,
                dpo_replay_ratio=dpo_replay_ratio,
                replay_history_limit=replay_history_limit,
                replay_min_samples=replay_min_samples,
                incremental_parent_selector=incremental_parent_selector,
            )
        )

    def test_build_dataset_uses_recent_history_tail_and_dpo_ratio(self) -> None:
        trainer = TrainerService()
        fresh_rows = [self._sample("dpo-new-1"), self._sample("dpo-new-2")]
        history_rows = [
            self._sample("dpo-old-1", used_in_version="20260401-001"),
            self._sample("dpo-old-2", used_in_version="20260401-002"),
            self._sample("dpo-old-3", used_in_version="20260401-003"),
            self._sample("dpo-old-4", used_in_version="20260401-004"),
        ]

        def fake_list_samples(*, sample_type: str | None = None, dataset_split: str | None = None, include_used: bool = True, **_: object):
            del dataset_split
            if sample_type != "dpo":
                return []
            if include_used:
                return [*fresh_rows, *history_rows]
            return list(fresh_rows)

        with patch.object(trainer_service_module, "list_samples", side_effect=fake_list_samples), patch.object(
            trainer_service_module.PFEConfig,
            "load",
            return_value=self._config(dpo_replay_ratio=0.75, replay_history_limit=2),
        ):
            fresh, replay, dataset_plan = trainer._build_dataset(train_type="dpo")

        self.assertEqual([sample["sample_id"] for sample in fresh], ["dpo-new-1", "dpo-new-2"])
        self.assertEqual([sample["sample_id"] for sample in replay], ["dpo-old-3", "dpo-old-4"])
        self.assertEqual(dataset_plan["sample_type"], "dpo")
        self.assertEqual(dataset_plan["history_limit"], 2)
        self.assertEqual(dataset_plan["configured_replay_ratio"], 0.75)
        self.assertEqual(dataset_plan["selected_replay_count"], 2)
        self.assertEqual(dataset_plan["replay_strategy"], "recent_history_tail")
        self.assertEqual(dataset_plan["replay_sample_ids"], ["dpo-old-3", "dpo-old-4"])

    def test_incremental_parent_context_falls_back_to_latest_promoted_adapter(self) -> None:
        latest_dir = self.pfe_home / "adapters" / "user_default" / "20260401-009"
        latest_dir.mkdir(parents=True, exist_ok=True)
        (latest_dir / "adapter_manifest.json").write_text(
            json.dumps(
                {
                    "version": "20260401-009",
                    "workspace": "user_default",
                    "base_model": "Qwen/Test-Base",
                    "artifact_format": "peft_lora",
                    "state": "promoted",
                    "num_samples": 12,
                    "training_run_id": "20260401-009",
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        store = _TrainerStore(self.pfe_home, version_dir=latest_dir, latest_version="20260401-009")
        trainer = TrainerService(store=store)

        with patch.object(
            trainer_service_module.PFEConfig,
            "load",
            return_value=self._config(incremental_parent_selector="promoted_or_latest"),
        ):
            context = trainer._resolve_incremental_parent_context(base_adapter="missing-parent", workspace="user_default")

        self.assertEqual(context["parent_adapter_version"], "20260401-009")
        self.assertEqual(context["resolved_base_model"], "Qwen/Test-Base")
        self.assertEqual(context["parent_selection_policy"], "promoted_or_latest")
        self.assertEqual(context["parent_selection_mode"], "fallback_promoted_or_latest")
        self.assertIn("using promoted_or_latest candidate 20260401-009", context["parent_selection_reason"])
        self.assertEqual(store.load_calls[-1], "20260401-009")

    def test_train_result_records_dpo_dataset_plan_and_lineage(self) -> None:
        fresh_rows = [self._sample("dpo-new-1"), self._sample("dpo-new-2")]
        history_rows = [
            self._sample("dpo-old-1", used_in_version="20260401-001"),
            self._sample("dpo-old-2", used_in_version="20260401-002"),
            self._sample("dpo-old-3", used_in_version="20260401-003"),
            self._sample("dpo-old-4", used_in_version="20260401-004"),
        ]
        version_dir = self.pfe_home / "adapters" / "user_default" / "20260401-010"
        store = _TrainerStore(self.pfe_home, version_dir=version_dir, latest_version="20260401-009")
        trainer = TrainerService(store=store)
        marked_samples: list[list[str]] = []

        def fake_list_samples(*, sample_type: str | None = None, dataset_split: str | None = None, include_used: bool = True, **_: object):
            del dataset_split
            if sample_type != "dpo":
                return []
            if include_used:
                return [*fresh_rows, *history_rows]
            return list(fresh_rows)

        def fake_mark_samples_used(sample_ids, version: str, home: str | Path | None = None):
            del version, home
            marked_samples.append(list(sample_ids))

        with patch.object(trainer_service_module, "list_samples", side_effect=fake_list_samples), patch.object(
            trainer_service_module,
            "mark_samples_used",
            side_effect=fake_mark_samples_used,
        ), patch.object(
            trainer_service_module.PFEConfig,
            "load",
            return_value=self._config(dpo_replay_ratio=0.75, replay_history_limit=2),
        ):
            result = trainer.train_result(method="qlora", epochs=1, base_model="mock-llama-target", train_type="dpo")

        self.assertEqual(result.training_config["train_type"], "dpo")
        self.assertEqual(result.training_config["dataset_plan"]["sample_type"], "dpo")
        self.assertEqual(result.training_config["dataset_plan"]["selected_replay_count"], 2)
        self.assertEqual(result.training_config["dataset_plan"]["replay_sample_ids"], ["dpo-old-3", "dpo-old-4"])
        self.assertEqual(result.training_config["configured_replay_ratio"], 0.75)
        self.assertEqual(result.training_config["dpo_beta"], 0.1)
        self.assertEqual(result.metrics["dataset_plan"]["selected_replay_count"], 2)
        self.assertEqual(result.metrics["num_fresh_samples"], 2)
        self.assertEqual(result.metrics["num_replay_samples"], 2)
        self.assertEqual(marked_samples, [["dpo-new-1", "dpo-new-2"]])
        self.assertEqual(store.created_training_config["dataset_plan"]["sample_type"], "dpo")
        self.assertEqual(store.created_training_config["configured_replay_ratio"], 0.75)
        self.assertIn("incremental_context", result.audit_info or {})

    def test_train_result_prioritizes_preference_reinforced_fresh_samples(self) -> None:
        fresh_rows = [
            self._sample("sft-new-1"),
            self._sample("sft-new-2"),
            self._sample("sft-new-3"),
        ]
        fresh_rows[1]["metadata"]["explicit_response_preference_reinforced"] = True
        fresh_rows[1]["metadata"]["training_signal_category"] = "preference_reinforced"
        version_dir = self.pfe_home / "adapters" / "user_default" / "20260401-011"
        store = _TrainerStore(self.pfe_home, version_dir=version_dir, latest_version="20260401-010")
        trainer = TrainerService(store=store)
        marked_samples: list[list[str]] = []

        def fake_list_samples(*, sample_type: str | None = None, dataset_split: str | None = None, include_used: bool = True, **_: object):
            del dataset_split
            if sample_type != "sft":
                return []
            if include_used:
                return list(fresh_rows)
            return list(fresh_rows)

        def fake_mark_samples_used(sample_ids, version: str, home: str | Path | None = None):
            del version, home
            marked_samples.append(list(sample_ids))

        with patch.object(trainer_service_module, "list_samples", side_effect=fake_list_samples), patch.object(
            trainer_service_module,
            "mark_samples_used",
            side_effect=fake_mark_samples_used,
        ), patch.object(
            trainer_service_module.PFEConfig,
            "load",
            return_value=self._config(),
        ):
            result = trainer.train_result(method="qlora", epochs=1, base_model="mock-llama-target", train_type="sft")

        self.assertEqual(result.training_config["dataset_plan"]["fresh_sample_ids"], [
            "sft-new-2",
            "sft-new-1",
            "sft-new-3",
        ])
        self.assertEqual(result.training_config["dataset_plan"]["preference_reinforced_fresh_sample_count"], 1)
        self.assertEqual(result.training_config["dataset_plan"]["preference_reinforced_fresh_sample_ids"], ["sft-new-2"])
        self.assertEqual(
            result.training_config["execution_recipe"]["backend_recipe"]["training"]["preference_reinforced_fresh_sample_count"],
            1,
        )
        self.assertEqual(
            result.training_config["execution_recipe"]["backend_recipe"]["training"]["preference_reinforced_fresh_sample_ids"],
            ["sft-new-2"],
        )
        self.assertEqual(
            result.training_config["execution_recipe"]["job_spec"]["preference_reinforced_fresh_sample_count"],
            1,
        )
        self.assertEqual(
            result.training_config["execution_recipe"]["job_spec"]["preference_reinforced_fresh_sample_ids"],
            ["sft-new-2"],
        )
        self.assertEqual(result.metrics["preference_reinforced_fresh_sample_count"], 1)
        self.assertEqual(result.metrics["preference_reinforced_fresh_sample_ids"], ["sft-new-2"])
        self.assertEqual(result.metrics["num_fresh_samples"], 3)
        self.assertEqual(
            result.metrics["execution_recipe"]["backend_recipe"]["training"]["preference_reinforced_fresh_sample_count"],
            1,
        )
        self.assertEqual(
            result.metrics["job_spec"]["preference_reinforced_fresh_sample_count"],
            1,
        )
        self.assertEqual(marked_samples, [["sft-new-2", "sft-new-1", "sft-new-3"]])
        self.assertEqual(store.created_training_config["dataset_plan"]["preference_reinforced_fresh_sample_count"], 1)
        self.assertEqual(store.created_training_config["dataset_plan"]["preference_reinforced_fresh_sample_ids"], ["sft-new-2"])

    def test_dpo_validation_rejects_incomplete_pairs(self) -> None:
        trainer = TrainerService()
        with self.assertRaises(TrainingError):
            trainer._validate_train_samples(
                [
                    {
                        "sample_id": "dpo-bad-1",
                        "instruction": "prompt",
                        "chosen": "chosen",
                        "rejected": "",
                        "metadata": {"dataset_split": "train"},
                    }
                ],
                train_type="dpo",
            )


if __name__ == "__main__":
    unittest.main()
