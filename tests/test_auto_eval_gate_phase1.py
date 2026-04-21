from __future__ import annotations

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

from pfe_core.config import PFEConfig
from pfe_core.pipeline import PipelineService


class _FakeAdapterStore:
    def __init__(self) -> None:
        self.promoted_versions: list[str] = []

    def list_version_records(self, limit: int = 20) -> list[dict[str, object]]:
        del limit
        return []

    def current_latest_version(self) -> str | None:
        return None

    def promote(self, version: str, workspace: str | None = None) -> str:
        del workspace
        self.promoted_versions.append(version)
        return version


class AutoEvalGatePhase1Tests(unittest.TestCase):
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

    def _seed_signal_ready_state(self) -> PipelineService:
        service = PipelineService()
        service.generate(scenario="life-coach", style="温和", num_samples=8)
        return service

    def _signal_payload(self, event_id: str, request_id: str, session_id: str) -> dict[str, object]:
        return {
            "event_id": event_id,
            "request_id": request_id,
            "session_id": session_id,
            "source_event_id": f"{event_id}-source",
            "source_event_ids": [f"{event_id}-source", event_id],
            "event_type": "accept",
            "user_input": "帮我把今天的安排理顺一下",
            "model_output": "先列出今天最重要的三件事。",
            "user_action": {"type": "accept"},
        }

    def test_auto_eval_gate_defaults_to_safe_disabled_behavior(self) -> None:
        PFEConfig().save(home=self.pfe_home)
        service = self._seed_signal_ready_state()

        with (
            patch.object(service, "train_result") as train_result_mock,
            patch.object(service, "evaluate") as evaluate_mock,
        ):
            result = service.signal(self._signal_payload("evt-auto-disabled", "req-auto-disabled", "sess-auto-disabled"))

        auto_train = result["auto_train"]
        self.assertFalse(auto_train["enabled"])
        self.assertFalse(auto_train["ready"])
        self.assertFalse(auto_train["triggered"])
        self.assertIn("trigger_disabled", auto_train["blocked_reasons"])
        train_result_mock.assert_not_called()
        evaluate_mock.assert_not_called()

    def test_auto_eval_gate_enabled_runs_eval_but_not_promote(self) -> None:
        config = PFEConfig()
        config.trainer.trigger.enabled = True
        config.trainer.trigger.min_new_samples = 1
        config.trainer.trigger.max_interval_days = 0
        config.trainer.trigger.auto_evaluate = True
        config.trainer.trigger.auto_promote = False
        config.trainer.trigger.eval_num_samples = 2
        config.trainer.epochs = 1
        config.save(home=self.pfe_home)

        service = self._seed_signal_ready_state()
        fake_store = _FakeAdapterStore()

        class _FakeTrainResult:
            version = "20260325-996"
            metrics = {
                "state": "pending_eval",
                "num_fresh_samples": 1,
                "num_replay_samples": 0,
            }

        with (
            patch.object(service, "train_result", return_value=_FakeTrainResult()) as train_result_mock,
            patch.object(
                service,
                "evaluate",
                return_value='{"recommendation":"deploy","comparison":"improved"}',
            ) as evaluate_mock,
            patch("pfe_core.pipeline.create_adapter_store", return_value=fake_store),
        ):
            result = service.signal(self._signal_payload("evt-auto-eval", "req-auto-eval", "sess-auto-eval"))

        auto_train = result["auto_train"]
        train_result_mock.assert_called_once()
        evaluate_mock.assert_called_once()
        self.assertTrue(auto_train["enabled"])
        self.assertTrue(auto_train["ready"])
        self.assertTrue(auto_train["triggered"])
        self.assertTrue(auto_train["eval_triggered"])
        self.assertFalse(auto_train["promote_triggered"])
        self.assertEqual(auto_train["eval_recommendation"], "deploy")
        self.assertEqual(auto_train["eval_comparison"], "improved")
        self.assertEqual(auto_train["execution_policy"]["review_gate"], "not_required")
        self.assertEqual(auto_train["execution_policy"]["evaluation_mode"], "auto_evaluate")
        self.assertTrue(auto_train["execution_policy"]["auto_evaluate_enabled"])
        self.assertIsNone(auto_train["execution_policy"]["evaluation_gate_reason"])
        self.assertFalse(auto_train["execution_policy"]["auto_promote_enabled"])
        self.assertEqual(auto_train["execution_policy"]["promotion_mode"], "manual_promote")
        self.assertEqual(fake_store.promoted_versions, [])

    def test_auto_eval_gate_enabled_promotes_only_when_eval_deploys(self) -> None:
        config = PFEConfig()
        config.trainer.trigger.enabled = True
        config.trainer.trigger.min_new_samples = 1
        config.trainer.trigger.max_interval_days = 0
        config.trainer.trigger.auto_evaluate = True
        config.trainer.trigger.auto_promote = True
        config.trainer.trigger.eval_num_samples = 2
        config.trainer.epochs = 1
        config.save(home=self.pfe_home)

        service = self._seed_signal_ready_state()
        fake_store = _FakeAdapterStore()

        class _FakeTrainResult:
            version = "20260325-995"
            metrics = {
                "state": "pending_eval",
                "num_fresh_samples": 1,
                "num_replay_samples": 0,
            }

        with (
            patch.object(service, "train_result", return_value=_FakeTrainResult()) as train_result_mock,
            patch.object(
                service,
                "evaluate",
                return_value='{"recommendation":"deploy","comparison":"improved"}',
            ) as evaluate_mock,
            patch("pfe_core.pipeline.create_adapter_store", return_value=fake_store),
        ):
            result = service.signal(self._signal_payload("evt-auto-promote", "req-auto-promote", "sess-auto-promote"))

        auto_train = result["auto_train"]
        train_result_mock.assert_called_once()
        evaluate_mock.assert_called_once()
        self.assertTrue(auto_train["enabled"])
        self.assertTrue(auto_train["ready"])
        self.assertTrue(auto_train["triggered"])
        self.assertTrue(auto_train["eval_triggered"])
        self.assertTrue(auto_train["promote_triggered"])
        self.assertEqual(auto_train["promoted_version"], "20260325-995")
        self.assertEqual(auto_train["triggered_state"], "promoted")
        self.assertEqual(auto_train["execution_policy"]["evaluation_mode"], "auto_evaluate")
        self.assertEqual(auto_train["execution_policy"]["promotion_mode"], "auto_promote")
        self.assertTrue(auto_train["execution_policy"]["auto_promote_enabled"])
        self.assertEqual(auto_train["execution_policy"]["promotion_requirement"], "deploy_eval_recommendation")
        self.assertEqual(auto_train["execution_policy"]["stop_stage"], "promote")
        self.assertEqual(fake_store.promoted_versions, ["20260325-995"])

    def test_auto_promote_requires_auto_evaluate_policy(self) -> None:
        config = PFEConfig()
        config.trainer.trigger.enabled = True
        config.trainer.trigger.min_new_samples = 1
        config.trainer.trigger.max_interval_days = 0
        config.trainer.trigger.auto_evaluate = False
        config.trainer.trigger.auto_promote = True
        config.trainer.epochs = 1
        config.save(home=self.pfe_home)

        service = self._seed_signal_ready_state()

        class _FakeTrainResult:
            version = "20260325-994"
            metrics = {
                "state": "pending_eval",
                "num_fresh_samples": 1,
                "num_replay_samples": 0,
            }

        with (
            patch.object(service, "train_result", return_value=_FakeTrainResult()) as train_result_mock,
            patch.object(service, "evaluate") as evaluate_mock,
        ):
            result = service.signal(self._signal_payload("evt-auto-promote-gate", "req-auto-promote-gate", "sess-auto-promote-gate"))

        auto_train = result["auto_train"]
        train_result_mock.assert_called_once()
        evaluate_mock.assert_not_called()
        self.assertTrue(auto_train["triggered"])
        self.assertFalse(auto_train["eval_triggered"])
        self.assertFalse(auto_train["promote_triggered"])
        self.assertEqual(auto_train["promote_reason"], "policy_requires_auto_evaluate")
        self.assertEqual(auto_train["execution_policy"]["evaluation_mode"], "skip_evaluate")
        self.assertEqual(auto_train["execution_policy"]["evaluation_gate_reason"], "policy_skips_auto_evaluate")
        self.assertEqual(auto_train["execution_policy"]["evaluation_gate_action"], "manual_evaluate_after_training")
        self.assertTrue(auto_train["execution_policy"]["auto_promote_requested"])
        self.assertEqual(auto_train["execution_policy"]["promotion_mode"], "manual_promote")
        self.assertFalse(auto_train["execution_policy"]["auto_promote_enabled"])
        self.assertEqual(auto_train["execution_policy"]["promote_gate_reason"], "policy_requires_auto_evaluate")
        self.assertEqual(auto_train["execution_policy"]["promote_gate_action"], "enable_auto_evaluate")


if __name__ == "__main__":
    unittest.main()
