"""Tests for LLM-based profile extraction and preference drift detection."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from pfe_core.profile.llm_extractor import LLMProfileExtractor
from pfe_core.user_profile import PreferenceScore, UserProfile


class TestLLMProfileExtractor:
    """Unit tests for LLMProfileExtractor."""

    def test_build_prompt_includes_messages(self) -> None:
        extractor = LLMProfileExtractor()
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        prompt = extractor._build_prompt(messages)
        assert "user: Hello" in prompt
        assert "assistant: Hi there" in prompt
        assert "JSON" in prompt

    def test_parse_json_direct(self) -> None:
        extractor = LLMProfileExtractor()
        data = {"identity": {"name": "Alice"}}
        assert extractor._parse_json(json.dumps(data)) == data

    def test_parse_json_markdown_block(self) -> None:
        extractor = LLMProfileExtractor()
        data = {"identity": {"name": "Alice"}}
        text = f"```json\n{json.dumps(data)}\n```"
        assert extractor._parse_json(text) == data

    def test_parse_json_plain_block(self) -> None:
        extractor = LLMProfileExtractor()
        data = {"identity": {"name": "Alice"}}
        text = f"Some intro\n```\n{json.dumps(data)}\n```\nOutro"
        assert extractor._parse_json(text) == data

    def test_validate_output_fills_defaults(self) -> None:
        extractor = LLMProfileExtractor()
        validated = extractor._validate_output({})
        assert validated["identity"] == {"name": "", "role": "", "location": ""}
        assert validated["stable_preferences"] == []
        assert validated["response_preferences"] == []
        assert validated["style_indicators"] == []
        assert validated["domain_interests"] == []
        assert validated["temporal_notes"] == []

    def test_extract_from_conversation_empty_messages(self) -> None:
        extractor = LLMProfileExtractor()
        result = extractor.extract_from_conversation([])
        assert result["identity"] == {"name": "", "role": "", "location": ""}
        meta = result["_extraction_meta"]
        assert meta["success"] is True  # empty is valid

    def test_extract_from_conversation_llm_success(self) -> None:
        extractor = LLMProfileExtractor()
        fake_response = json.dumps({
            "identity": {"name": "Alice", "role": "engineer", "location": "Beijing"},
            "stable_preferences": [{"category": "tech", "preference": "python", "confidence": 0.9}],
            "response_preferences": [{"aspect": "length", "preference": "concise", "confidence": 0.8}],
            "style_indicators": [{"trait": "direct", "evidence": "uses short sentences"}],
            "domain_interests": [{"domain": "programming", "level": 0.85}],
            "temporal_notes": [{"note": "started new job", "date_hint": "2024-01"}],
        })

        with patch.object(extractor, "_call_llm", return_value=fake_response):
            result = extractor.extract_from_conversation([{"role": "user", "content": "Hi"}])

        assert result["identity"]["name"] == "Alice"
        assert len(result["stable_preferences"]) == 1
        assert result["domain_interests"][0]["level"] == 0.85
        assert result["_extraction_meta"]["success"] is True
        assert result["_extraction_meta"]["method"] == "llm"

    def test_extract_from_conversation_llm_failure_fallback(self) -> None:
        extractor = LLMProfileExtractor()

        with patch.object(extractor, "_call_llm", side_effect=RuntimeError("timeout")):
            result = extractor.extract_from_conversation([{"role": "user", "content": "Hi"}])

        assert result["_extraction_meta"]["success"] is False
        assert "error" in result["_extraction_meta"]
        assert result["identity"] == {"name": "", "role": "", "location": ""}

    def test_call_llm_makes_post_request(self) -> None:
        extractor = LLMProfileExtractor(api_base="http://localhost:8000", api_key="sk-test")
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": '{"identity": {}}'}}]
        }

        with patch("requests.post", return_value=mock_response) as mock_post:
            text = extractor._call_llm("prompt")

        assert text == '{"identity": {}}'
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert kwargs["json"]["model"] == extractor.model
        assert kwargs["headers"]["Authorization"] == "Bearer sk-test"


class TestPreferenceDriftDetection:
    """Unit tests for preference drift detection on PreferenceScore and UserProfile."""

    def test_detect_drift_no_history(self) -> None:
        pref = PreferenceScore(value="formal")
        assert pref.detect_drift(threshold=0.3) is None

    def test_detect_drift_insufficient_history(self) -> None:
        pref = PreferenceScore(value="formal", score_history=[0.5, 0.6, 0.7])
        assert pref.detect_drift(threshold=0.3) is None

    def test_detect_drift_increase(self) -> None:
        pref = PreferenceScore(value="formal", score_history=[0.2, 0.2, 0.2, 0.8, 0.8, 0.8])
        alert = pref.detect_drift(threshold=0.3)
        assert alert is not None
        assert alert["drift_direction"] == "increase"
        assert alert["severity"] == "high"
        assert alert["old_avg"] == pytest.approx(0.2)
        assert alert["new_avg"] == pytest.approx(0.8)

    def test_detect_drift_decrease(self) -> None:
        pref = PreferenceScore(value="casual", score_history=[0.9, 0.9, 0.9, 0.3, 0.3, 0.3])
        alert = pref.detect_drift(threshold=0.3)
        assert alert is not None
        assert alert["drift_direction"] == "decrease"
        assert alert["severity"] == "high"

    def test_detect_drift_below_threshold(self) -> None:
        pref = PreferenceScore(value="concise", score_history=[0.5, 0.5, 0.5, 0.6, 0.6, 0.6])
        assert pref.detect_drift(threshold=0.3) is None

    def test_detect_drift_medium_severity(self) -> None:
        pref = PreferenceScore(value="technical", score_history=[0.5, 0.5, 0.5, 0.9, 0.9, 0.9])
        alert = pref.detect_drift(threshold=0.3)
        assert alert is not None
        assert alert["severity"] == "medium"

    def test_user_profile_detect_preference_drift(self) -> None:
        profile = UserProfile(user_id="u1")
        profile.update_style_preference("formal", 0.2)
        profile.update_style_preference("formal", 0.2)
        profile.update_style_preference("formal", 0.8)
        profile.update_style_preference("formal", 0.8)
        # Ensure history has enough entries
        profile.style_preferences["formal"].score_history = [0.2, 0.2, 0.8, 0.8]
        alerts = profile.detect_preference_drift(threshold=0.3)
        assert len(alerts) == 1
        assert alerts[0]["preference_key"] == "formal"

    def test_user_profile_no_drift(self) -> None:
        profile = UserProfile(user_id="u1")
        profile.update_style_preference("casual", 0.5)
        profile.style_preferences["casual"].score_history = [0.5, 0.5, 0.5, 0.5]
        alerts = profile.detect_preference_drift(threshold=0.3)
        assert alerts == []

    def test_preference_score_update_maintains_window(self) -> None:
        pref = PreferenceScore(value="x", history_window_size=3)
        pref.update(0.1, "s1")
        pref.update(0.2, "s2")
        pref.update(0.3, "s3")
        pref.update(0.4, "s4")
        assert pref.score_history == [0.2, 0.3, 0.4]

    def test_preference_score_serialization_roundtrip(self) -> None:
        pref = PreferenceScore(value="formal", score_history=[0.1, 0.2, 0.3])
        data = pref.to_dict()
        restored = PreferenceScore.from_dict(data)
        assert restored.value == pref.value
        assert restored.score_history == pref.score_history
        assert restored.history_window_size == pref.history_window_size


class TestUserProfileLLMFields:
    """Unit tests for new LLM-related fields on UserProfile."""

    def test_llm_fields_defaults(self) -> None:
        profile = UserProfile(user_id="u1")
        assert profile.extracted_by == "rule_based"
        assert profile.llm_extraction_confidence == 0.0
        assert profile.llm_extracted_at is None
        assert profile.llm_identity == {}
        assert profile.llm_stable_preferences == []

    def test_to_dict_includes_llm_fields(self) -> None:
        profile = UserProfile(user_id="u1")
        profile.extracted_by = "llm"
        profile.llm_extraction_confidence = 0.8
        profile.llm_identity = {"name": "Alice"}
        data = profile.to_dict()
        assert data["extracted_by"] == "llm"
        assert data["llm_extraction_confidence"] == 0.8
        assert data["llm_identity"] == {"name": "Alice"}

    def test_from_dict_restores_llm_fields(self) -> None:
        profile = UserProfile(user_id="u1")
        profile.extracted_by = "llm"
        profile.llm_extraction_confidence = 0.8
        profile.llm_identity = {"name": "Alice"}
        profile.llm_stable_preferences = [{"category": "tech", "preference": "python"}]
        data = profile.to_dict()
        restored = UserProfile.from_dict(data)
        assert restored.extracted_by == "llm"
        assert restored.llm_extraction_confidence == 0.8
        assert restored.llm_identity == {"name": "Alice"}
        assert restored.llm_stable_preferences == [{"category": "tech", "preference": "python"}]
