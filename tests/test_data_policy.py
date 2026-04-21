from __future__ import annotations

import os
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for package_dir in ("pfe-core", "pfe-cli", "pfe-server"):
    package_path = str(ROOT / package_dir)
    if package_path not in os.sys.path:
        os.sys.path.insert(0, package_path)

from pfe_core.data_policy import (
    UserDatum,
    extract_user_data_candidates,
    route_signal_for_training,
    route_user_datum,
)
from pfe_core.models import ImplicitSignal, SignalQuality


class DataPolicyTests(unittest.TestCase):
    def test_extract_candidates_finds_name_role_and_style_preference(self) -> None:
        message = "我叫小王，我是程序员，我希望你以后回答更温和、更鼓励式。"
        candidates = extract_user_data_candidates(message)

        self.assertTrue(any(item.kind == "identity_fact" and item.key == "名字" for item in candidates))
        self.assertTrue(any(item.kind == "role_fact" and item.key == "职业" for item in candidates))
        self.assertTrue(any(item.kind == "response_preference" for item in candidates))

    def test_identity_fact_routes_to_memory_not_training(self) -> None:
        decision = route_user_datum(
            UserDatum(key="名字", value="小王", kind="identity_fact", pii_types=["person_name"])
        )

        self.assertEqual(decision.primary_lane, "memory")
        self.assertIn("prompt_context", decision.additional_lanes)
        self.assertFalse(decision.eligible_for_training)
        self.assertFalse(decision.pii_blocked)

    def test_response_preference_is_profile_first_until_reinforced(self) -> None:
        decision = route_user_datum(
            UserDatum(key="回答风格偏好", value="请更温和一点", kind="response_preference")
        )

        self.assertEqual(decision.primary_lane, "profile")
        self.assertIn("signal", decision.additional_lanes)
        self.assertFalse(decision.eligible_for_training)
        self.assertEqual(decision.training_target, "preference_only")

    def test_reinforced_response_preference_becomes_trainable(self) -> None:
        decision = route_user_datum(
            UserDatum(
                key="回答风格偏好",
                value="请更温和一点",
                kind="response_preference",
                metadata={"confirmed_by_feedback": True},
            )
        )

        self.assertEqual(decision.primary_lane, "profile")
        self.assertTrue(decision.eligible_for_training)
        self.assertEqual(decision.training_target, "sft")

    def test_high_risk_pii_is_blocked(self) -> None:
        decision = route_user_datum(
            UserDatum(key="联系方式", value="13800000000", kind="identity_fact", pii_types=["phone"])
        )

        self.assertEqual(decision.primary_lane, "discard")
        self.assertTrue(decision.pii_blocked)
        self.assertEqual(decision.training_target, "blocked")

    def test_accept_signal_becomes_sft_candidate(self) -> None:
        signal = ImplicitSignal(
            signal_type="accept",
            event_type="accept",
            confidence=0.92,
            request_id="req-1",
            session_id="sess-1",
            context="帮我写一段周报",
            model_output="这是周报草稿",
            source_event_ids=["evt-1", "evt-2"],
            signal_quality=SignalQuality(
                reply_style="accepted",
                confidence=0.92,
                provenance={"event_chain_complete": True},
            ),
        )

        decision = route_signal_for_training(signal)
        self.assertTrue(decision.eligible)
        self.assertEqual(decision.primary_target, "sft_candidate")
        self.assertEqual(decision.recommended_sample_type, "sft")

    def test_edit_signal_prefers_dpo_when_chain_is_complete(self) -> None:
        signal = ImplicitSignal(
            signal_type="edit",
            event_type="edit",
            confidence=0.88,
            request_id="req-2",
            session_id="sess-2",
            context="帮我回复这封邮件",
            model_output="原始回复",
            user_action={"edited_text": "用户修改后的回复"},
            source_event_ids=["evt-3", "evt-4"],
            signal_quality=SignalQuality(
                reply_style="edited",
                confidence=0.88,
                provenance={"event_chain_complete": True},
            ),
        )

        decision = route_signal_for_training(signal)
        self.assertTrue(decision.eligible)
        self.assertEqual(decision.primary_target, "dpo_candidate")
        self.assertEqual(decision.recommended_sample_type, "dpo")

    def test_reject_signal_requires_pairing(self) -> None:
        signal = ImplicitSignal(
            signal_type="reject",
            event_type="reject",
            confidence=0.9,
            request_id="req-3",
            session_id="sess-3",
            context="给我一个方案",
            model_output="不喜欢的回复",
            source_event_ids=["evt-5", "evt-6"],
            signal_quality=SignalQuality(
                reply_style="rejected",
                confidence=0.9,
                provenance={"event_chain_complete": True},
            ),
        )

        decision = route_signal_for_training(signal)
        self.assertFalse(decision.eligible)
        self.assertEqual(decision.primary_target, "dpo_rejected_only")
        self.assertTrue(decision.requires_pairing)


if __name__ == "__main__":
    unittest.main()
