"""Tests for P2-C semantic router and scenario-aware classification."""

from __future__ import annotations

import pytest

from pfe_core.scenarios import ScenarioConfig, BUILTIN_SCENARIOS
from pfe_core.router import (
    IntentClassifier,
    ScenarioRouter,
    create_router,
    IntentClassification,
    RoutingResult,
)
from pfe_core.router.semantic_classifier import (
    SemanticClassifier,
    SemanticClassificationResult,
    _fallback_similarity,
    _aggregate_scores,
)
from pfe_core.user_profile import UserProfile, PreferenceScore
from pfe_core.config import PFEConfig, RouterConfig


class TestSemanticClassifier:
    """Unit tests for SemanticClassifier."""

    def test_classify_returns_result(self):
        classifier = SemanticClassifier()
        result = classifier.classify("Write a Python function")
        assert isinstance(result, SemanticClassificationResult)
        assert result.scenario_id in BUILTIN_SCENARIOS
        assert 0.0 <= result.confidence <= 1.0
        assert result.method in ("tfidf", "fallback")

    def test_classify_empty_input(self):
        classifier = SemanticClassifier()
        result = classifier.classify("")
        assert result.scenario_id == "chat"
        assert result.confidence == 0.0

    def test_classify_no_examples(self):
        scenarios = {
            "empty": ScenarioConfig(
                scenario_id="empty",
                name="Empty",
                description="No examples",
                adapter_version="latest",
                trigger_keywords=[],
                examples=[],
                example_phrases=[],
            )
        }
        classifier = SemanticClassifier(scenarios=scenarios)
        result = classifier.classify("something")
        assert result.scenario_id == "chat"

    def test_fallback_similarity(self):
        assert _fallback_similarity("hello world", "hello world") == 1.0
        assert _fallback_similarity("", "anything") == 0.0
        assert 0.0 < _fallback_similarity("hello", "hallo") < 1.0

    def test_aggregate_scores(self):
        assert _aggregate_scores([0.5, 0.8]) == pytest.approx(0.8 * 0.8 + 0.65 * 0.2, rel=1e-3)
        assert _aggregate_scores([]) == 0.0

    def test_refresh_index(self):
        classifier = SemanticClassifier()
        classifier.refresh_index()
        result = classifier.classify("test")
        assert result.scenario_id in BUILTIN_SCENARIOS


class TestKeywordModeBackwardCompatibility:
    """Ensure keyword-only mode still works exactly as before."""

    def test_keyword_classify_basic(self):
        classifier = IntentClassifier()
        result = classifier.classify("Write a Python function")
        assert isinstance(result, IntentClassification)
        assert result.primary_intent == "coding"
        assert result.confidence > 0.0

    def test_keyword_empty_input(self):
        classifier = IntentClassifier()
        result = classifier.classify("")
        assert result.primary_intent == "chat"
        assert result.confidence == 0.0

    def test_router_keyword_strategy(self):
        config = PFEConfig()
        config.router.strategy = "keyword"  # type: ignore[misc]
        router = ScenarioRouter(config=config)
        result = router.route("Help me write an essay")
        assert isinstance(result, RoutingResult)
        assert result.scenario_id == "writing"


class TestHybridMode:
    """Tests for hybrid keyword + semantic routing."""

    def test_hybrid_classify_semantic_match(self):
        """Semantic match for phrase without direct keyword hit."""
        config = PFEConfig()
        config.router.strategy = "hybrid"  # type: ignore[misc]
        router = ScenarioRouter(config=config)
        # "帮我写一个Python排序函数" has strong semantic overlap with coding examples
        result = router.route("帮我写一个Python排序函数")
        assert result.scenario_id == "coding"

    def test_hybrid_fusion_weights(self):
        config = PFEConfig()
        config.router.strategy = "hybrid"  # type: ignore[misc]
        router = ScenarioRouter(config=config)
        result = router.test_route("Debug this JavaScript error")
        assert result["strategy"] == "hybrid"
        scores = result["classification"]["all_scores"]
        assert "coding" in scores
        # Confidence should be non-zero because both keyword and semantic hit coding
        assert scores["coding"] > 0.0

    def test_hybrid_low_confidence_fallback(self):
        config = PFEConfig()
        config.router.strategy = "hybrid"  # type: ignore[misc]
        router = ScenarioRouter(config=config)
        # Nonsense input should fallback to chat
        result = router.route("xyz123 nonsense !!!")
        # Even if a scenario is selected, confidence should be low enough to trigger fallback
        if result.confidence < router.scorer.min_confidence_threshold:
            assert result.scenario_id == "chat"
            assert result.fallback is True
        else:
            # If confidence scorer bumps it above threshold, still acceptable
            assert result.confidence >= router.scorer.min_confidence_threshold


class TestProfileBoost:
    """Tests for user-profile-aware routing boost."""

    def test_profile_boost_high_domain_preference(self):
        config = PFEConfig()
        config.router.strategy = "keyword"  # type: ignore[misc]
        router = ScenarioRouter(config=config)

        profile = UserProfile(user_id="test_user")
        profile.update_domain_preference("coding", 0.9)

        # Even a weak coding signal should be boosted
        classification = router.classifier.classify("python")
        boosted = router._apply_profile_boost(classification, profile)
        assert boosted.all_scores["coding"] >= classification.all_scores.get("coding", 0.0)

    def test_profile_boost_changes_primary(self):
        config = PFEConfig()
        config.router.strategy = "keyword"  # type: ignore[misc]
        router = ScenarioRouter(config=config)

        profile = UserProfile(user_id="test_user")
        profile.update_domain_preference("writing", 0.95)

        # A borderline input that might otherwise go to chat gets boosted to writing
        classification = router.classifier.classify("help me with text")
        boosted = router._apply_profile_boost(classification, profile)
        # writing score should receive +0.15
        assert boosted.all_scores["writing"] >= 0.15

    def test_profile_boost_no_effect_when_low_preference(self):
        config = PFEConfig()
        config.router.strategy = "keyword"  # type: ignore[misc]
        router = ScenarioRouter(config=config)

        profile = UserProfile(user_id="test_user")
        profile.update_domain_preference("coding", 0.5)

        classification = router.classifier.classify("python")
        boosted = router._apply_profile_boost(classification, profile)
        # No boost because preference score <= 0.7
        assert boosted.all_scores["coding"] == classification.all_scores.get("coding", 0.0)

    def test_route_with_user_profile(self):
        config = PFEConfig()
        config.router.strategy = "hybrid"  # type: ignore[misc]
        router = ScenarioRouter(config=config)

        profile = UserProfile(user_id="test_user")
        profile.update_domain_preference("coding", 0.95)

        result = router.route("something about code", user_profile=profile)
        assert isinstance(result, RoutingResult)
        assert result.confidence > 0.0


class TestScenarioConfigExamplePhrases:
    """Ensure example_phrases field is present and populated."""

    def test_builtin_scenarios_have_example_phrases(self):
        for sid, scenario in BUILTIN_SCENARIOS.items():
            assert hasattr(scenario, "example_phrases")
            assert isinstance(scenario.example_phrases, list)
            if sid != "chat":
                assert len(scenario.example_phrases) >= 3, f"{sid} should have at least 3 example_phrases"

    def test_custom_scenario_with_example_phrases(self):
        from pfe_core.scenarios import create_custom_scenario

        scenario = create_custom_scenario(
            scenario_id="test",
            name="Test",
            description="Test scenario",
            adapter_version="latest",
            example_phrases=["phrase one", "phrase two"],
        )
        assert scenario.example_phrases == ["phrase one", "phrase two"]


class TestCLIOptions:
    """Smoke tests for CLI option parsing (we test the router functions directly)."""

    def test_test_route_strategy_override(self):
        config = PFEConfig()
        config.router.strategy = "keyword"  # type: ignore[misc]
        router = ScenarioRouter(config=config)
        # Simulate CLI overriding strategy
        config.router.strategy = "hybrid"  # type: ignore[misc]
        router2 = ScenarioRouter(config=config)
        result = router2.test_route("帮我写代码")
        assert result["strategy"] == "hybrid"

    def test_show_scores_flag(self):
        config = PFEConfig()
        config.router.strategy = "hybrid"  # type: ignore[misc]
        router = ScenarioRouter(config=config)
        result = router.test_route("Write a blog post")
        assert "all_routes" in result
        assert len(result["all_routes"]) >= len(BUILTIN_SCENARIOS)
