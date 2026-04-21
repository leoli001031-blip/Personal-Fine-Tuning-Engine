"""Personalized Evaluator for PFE Phase 2.

This module implements a personalized evaluation framework that assesses
"whether the model output is more like what this specific user wants",
rather than just generic quality.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from difflib import SequenceMatcher
from enum import Enum
from statistics import mean
from typing import Any, Literal, Optional, Sequence

from ..profile_extractor import ProfileExtractor
from ..user_profile import PreferenceScore, UserProfile


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _normalize(text: Optional[str]) -> str:
    if not text:
        return ""
    return " ".join(text.lower().split())


def _similarity(left: Optional[str], right: Optional[str]) -> float:
    return SequenceMatcher(None, _normalize(left), _normalize(right)).ratio()


def _keyword_density(text: Optional[str], keywords: Sequence[str]) -> float:
    if not text:
        return 0.0
    lowered = text.lower()
    hits = sum(1 for keyword in keywords if keyword.lower() in lowered)
    return hits / max(1, len(keywords))


class JudgeBackend(str, Enum):
    """Supported judge backends."""

    RULE = "rule"
    LLM = "llm"
    HYBRID = "hybrid"


@dataclass
class JudgeResult:
    """Result from a single judge evaluation."""

    score: float = 0.0
    passed: bool = False
    rationale: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PersonalizedEvalConfig:
    """Configuration for personalized evaluation."""

    user_id: str | None = None
    judge_backend: JudgeBackend = JudgeBackend.RULE
    style_keywords: tuple[str, ...] = (
        "理解",
        "感受",
        "支持",
        "共情",
        "温和",
        "一起",
        "慢慢",
        "可以",
    )
    # Weights for personalized metrics when computing overall personalized score
    style_preference_weight: float = 0.25
    profile_aware_weight: float = 0.25
    preference_alignment_weight: float = 0.25
    consistency_weight: float = 0.25
    # Thresholds
    min_style_hit_rate: float = 0.5
    min_profile_aware_score: float = 0.5
    min_preference_alignment: float = 0.5
    min_consistency_score: float = 0.5
    # LLM judge settings (mockable)
    llm_judge_model: str = "mock"
    llm_judge_prompt_version: str = "v1"
    enable_generic_quality_comparison: bool = True


@dataclass
class EvalReport:
    """Personalized evaluation report."""

    adapter_version: str = ""
    base_model: str = ""
    user_id: str | None = None
    num_test_samples: int = 0
    # Personalized metrics
    style_preference_hit_rate: float = 0.0
    profile_aware_accuracy: float = 0.0
    preference_alignment: float = 0.0
    consistency_score: float = 0.0
    # Generic quality metrics (for comparison)
    generic_quality_score: float = 0.0
    fluency_score: float = 0.0
    relevance_score: float = 0.0
    # Overall scores
    overall_personalized_score: float = 0.0
    overall_generic_score: float = 0.0
    # Summary and explanation
    personalization_summary: str = ""
    generic_quality_summary: str = ""
    differentiation_note: str = ""
    recommendation: Literal["deploy", "keep_previous", "needs_more_data"] = "needs_more_data"
    # Per-sample details
    details: list[dict[str, Any]] = field(default_factory=list)
    # Metadata
    judge_backend: str = "rule"
    judge_model: str = "local-heuristic"
    created_at: datetime = field(default_factory=_utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter_version": self.adapter_version,
            "base_model": self.base_model,
            "user_id": self.user_id,
            "num_test_samples": self.num_test_samples,
            "style_preference_hit_rate": round(self.style_preference_hit_rate, 4),
            "profile_aware_accuracy": round(self.profile_aware_accuracy, 4),
            "preference_alignment": round(self.preference_alignment, 4),
            "consistency_score": round(self.consistency_score, 4),
            "generic_quality_score": round(self.generic_quality_score, 4),
            "fluency_score": round(self.fluency_score, 4),
            "relevance_score": round(self.relevance_score, 4),
            "overall_personalized_score": round(self.overall_personalized_score, 4),
            "overall_generic_score": round(self.overall_generic_score, 4),
            "personalization_summary": self.personalization_summary,
            "generic_quality_summary": self.generic_quality_summary,
            "differentiation_note": self.differentiation_note,
            "recommendation": self.recommendation,
            "details": self.details,
            "judge_backend": self.judge_backend,
            "judge_model": self.judge_model,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class PersonalizedVsGenericComparison:
    """Comparison result distinguishing personalized vs generic improvements."""

    personalized_delta: float = 0.0
    generic_delta: float = 0.0
    personalization_wins: bool = False
    generic_quality_wins: bool = False
    summary: str = ""
    tradeoff_warning: bool = False
    recommendation: Literal["deploy", "keep_previous", "needs_more_data"] = "needs_more_data"

    def to_dict(self) -> dict[str, Any]:
        return {
            "personalized_delta": round(self.personalized_delta, 4),
            "generic_delta": round(self.generic_delta, 4),
            "personalization_wins": self.personalization_wins,
            "generic_quality_wins": self.generic_quality_wins,
            "summary": self.summary,
            "tradeoff_warning": self.tradeoff_warning,
            "recommendation": self.recommendation,
        }


# ---------------------------------------------------------------------------
# Style preference hint mapping (shared with evaluator/auto.py)
# ---------------------------------------------------------------------------
_STYLE_PREFERENCE_HINTS: dict[str, tuple[str, ...]] = {
    "empathetic": ("温和", "共情", "理解", "支持", "鼓励", "关怀", "耐心", "empathetic", "warm"),
    "concise": ("简洁", "简短", "精炼", "直接", "结论", "概要", "concise", "brief"),
    "detailed": ("详细", "具体", "深入", "展开", "步骤", "全面", "detailed", "thorough"),
    "formal": ("正式", "礼貌", "客气", "正式一点", "formal"),
    "friendly": ("轻松", "随和", "自然", "口语", "朋友", "friendly", "casual"),
    "technical": ("技术", "专业", "实现", "原理", "架构", "代码", "technical"),
    "direct": ("直接", "明确", "不绕弯", "先说结论", "direct"),
}

_STYLE_PREFERENCE_METADATA_KEYS: tuple[str, ...] = (
    "explicit_response_style_preference",
    "explicit_style_preference",
    "response_style_preference",
    "response_style_preferences",
    "style_preference",
    "style_preferences",
    "preferred_style",
    "preferred_styles",
)


def _coerce_text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, dict):
        result: list[str] = []
        for key, item in value.items():
            if isinstance(item, (int, float)):
                if float(item) > 0:
                    result.append(str(key))
                continue
            result.extend(_coerce_text_list(item))
        if result:
            return result
        return [str(key).strip() for key in value.keys() if str(key).strip()]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        result = []
        for item in value:
            result.extend(_coerce_text_list(item))
        return result
    text = str(value).strip()
    return [text] if text else []


def _extract_style_preference_values(metadata: Any) -> list[str]:
    if not isinstance(metadata, dict):
        return []
    values: list[str] = []
    for key in _STYLE_PREFERENCE_METADATA_KEYS:
        values.extend(_coerce_text_list(metadata.get(key)))
    profile = metadata.get("profile")
    if isinstance(profile, dict):
        values.extend(_coerce_text_list(profile.get("style_preferences")))
    deduped: list[str] = []
    for value in values:
        normalized = value.strip()
        if normalized and normalized not in deduped:
            deduped.append(normalized)
    return deduped


def _style_preference_hints(preference: str) -> tuple[str, ...]:
    preference_normalized = _normalize(preference)
    if not preference_normalized:
        return ()
    for style_name, hints in _STYLE_PREFERENCE_HINTS.items():
        if style_name in preference_normalized or any(hint in preference_normalized for hint in hints):
            return hints
    return tuple(token for token in re.split(r"[,\s、/|;；，]+", preference_normalized) if token)


def _style_preference_hit(preference: str, output: str) -> bool:
    output_normalized = _normalize(output)
    if not output_normalized:
        return False
    hints = _style_preference_hints(preference)
    if not hints:
        return False
    return any(hint.lower() in output_normalized for hint in hints)


# ---------------------------------------------------------------------------
# Judge protocol
# ---------------------------------------------------------------------------
class BaseJudge(ABC):
    """Abstract base for personalized judges."""

    @abstractmethod
    def evaluate_style_preference_hit_rate(
        self,
        outputs: Sequence[str],
        metadata_list: Sequence[dict[str, Any]],
        profile: UserProfile | None,
    ) -> JudgeResult:
        """Evaluate style preference hit rate."""

    @abstractmethod
    def evaluate_profile_aware_accuracy(
        self,
        prompts: Sequence[str],
        outputs: Sequence[str],
        profile: UserProfile | None,
    ) -> JudgeResult:
        """Evaluate profile-aware context understanding accuracy."""

    @abstractmethod
    def evaluate_preference_alignment(
        self,
        outputs: Sequence[str],
        preferred_outputs: Sequence[str] | None,
        profile: UserProfile | None,
    ) -> JudgeResult:
        """Evaluate alignment with user historical preferences."""

    @abstractmethod
    def evaluate_consistency(
        self,
        outputs: Sequence[str],
        repeated_prompts: Sequence[str] | None = None,
    ) -> JudgeResult:
        """Evaluate consistency across multiple answers to same/similar prompts."""

    @abstractmethod
    def evaluate_generic_quality(
        self,
        prompts: Sequence[str],
        outputs: Sequence[str],
        references: Sequence[str | None] | None,
    ) -> JudgeResult:
        """Evaluate generic quality (fluency, relevance, etc.)."""


class RuleBasedJudge(BaseJudge):
    """Rule-based judge using keyword/pattern matching."""

    def __init__(self, config: PersonalizedEvalConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Style preference hit rate
    # ------------------------------------------------------------------
    def evaluate_style_preference_hit_rate(
        self,
        outputs: Sequence[str],
        metadata_list: Sequence[dict[str, Any]],
        profile: UserProfile | None,
    ) -> JudgeResult:
        if not outputs:
            return JudgeResult(score=0.0, passed=False, rationale="No outputs provided")

        hits: list[float] = []
        per_sample: list[dict[str, Any]] = []

        for idx, output in enumerate(outputs):
            meta = metadata_list[idx] if idx < len(metadata_list) else {}
            preferences = _extract_style_preference_values(meta)

            # Also include profile style preferences if available
            if profile is not None:
                for style, pref in profile.style_preferences.items():
                    if pref.score * pref.confidence >= 0.3:
                        preferences.append(style)

            if not preferences:
                # No preference declared -> neutral score for this sample
                hits.append(0.5)
                per_sample.append({"index": idx, "hit": 0.5, "preferences": [], "reason": "no_preference_declared"})
                continue

            matched = any(_style_preference_hit(pref, output) for pref in preferences)
            score = 1.0 if matched else 0.0
            hits.append(score)
            per_sample.append({"index": idx, "hit": score, "preferences": preferences, "reason": "matched" if matched else "missed"})

        avg_hit = mean(hits) if hits else 0.0
        passed = avg_hit >= self.config.min_style_hit_rate
        rationale = (
            f"Style preference hit rate: {avg_hit:.3f} "
            f"({sum(h for h in hits if h == 1.0)}/{len(hits)} hits)"
        )
        return JudgeResult(
            score=avg_hit,
            passed=passed,
            rationale=rationale,
            metadata={"per_sample": per_sample, "total_samples": len(outputs)},
        )

    # ------------------------------------------------------------------
    # Profile-aware accuracy
    # ------------------------------------------------------------------
    def evaluate_profile_aware_accuracy(
        self,
        prompts: Sequence[str],
        outputs: Sequence[str],
        profile: UserProfile | None,
    ) -> JudgeResult:
        if not outputs:
            return JudgeResult(score=0.0, passed=False, rationale="No outputs provided")

        if profile is None or not any([
            profile.style_preferences,
            profile.domain_preferences,
            profile.interaction_patterns,
        ]):
            return JudgeResult(
                score=0.5,
                passed=True,
                rationale="No user profile available; returning neutral score",
                metadata={"profile_available": False},
            )

        scores: list[float] = []
        per_sample: list[dict[str, Any]] = []

        for idx, (prompt, output) in enumerate(zip(prompts, outputs)):
            combined_text = f"{prompt} {output}".lower()
            match_scores: list[float] = []

            # Domain alignment
            for domain, pref in profile.domain_preferences.items():
                domain_keywords = {
                    "programming": ["code", "programming", "function", "代码", "编程"],
                    "writing": ["write", "essay", "article", "写作", "文章"],
                    "learning": ["learn", "tutorial", "学习", "教程"],
                    "analysis": ["analyze", "data", "分析", "数据"],
                    "creative": ["creative", "design", "创意", "设计"],
                    "business": ["business", "product", "商业", "产品"],
                }
                keywords = domain_keywords.get(domain, [domain])
                if any(kw in combined_text for kw in keywords):
                    match_scores.append(pref.score * pref.confidence)

            # Style alignment
            for style, pref in profile.style_preferences.items():
                style_keywords = {
                    "formal": ["please", "thank you", "would you", "请", "您好", "谢谢"],
                    "casual": ["hey", "hi", "yeah", "嘿", "嗨", "咋"],
                    "concise": ["brief", "short", "summary", "简短", "简洁", "概要"],
                    "detailed": ["detailed", "comprehensive", "详细", "具体", "深入"],
                    "technical": ["technical", "implementation", "技术", "实现", "架构"],
                    "non_technical": ["simple", "plain", "简单", "通俗", "易懂"],
                }
                keywords = style_keywords.get(style, [style])
                if any(kw in output.lower() for kw in keywords):
                    match_scores.append(pref.score * pref.confidence)

            # Interaction pattern alignment
            for pattern, pref in profile.interaction_patterns.items():
                pattern_keywords = {
                    "likes_examples": ["example", "for instance", "such as", "例子", "示例"],
                    "prefers_direct": ["direct", "straight", "直接", "干脆"],
                    "wants_reasoning": ["because", "reason", "因此", "原因"],
                    "prefers_code": ["code", "implement", "代码", "实现"],
                }
                keywords = pattern_keywords.get(pattern, [pattern])
                if any(kw in output.lower() for kw in keywords):
                    match_scores.append(pref.score * pref.confidence)

            sample_score = mean(match_scores) if match_scores else 0.3
            scores.append(sample_score)
            per_sample.append({"index": idx, "score": sample_score, "match_count": len(match_scores)})

        avg_score = mean(scores) if scores else 0.0
        passed = avg_score >= self.config.min_profile_aware_score
        rationale = f"Profile-aware accuracy: {avg_score:.3f} (matches across {len(scores)} samples)"
        return JudgeResult(
            score=avg_score,
            passed=passed,
            rationale=rationale,
            metadata={"per_sample": per_sample, "profile_available": True},
        )

    # ------------------------------------------------------------------
    # Preference alignment
    # ------------------------------------------------------------------
    def evaluate_preference_alignment(
        self,
        outputs: Sequence[str],
        preferred_outputs: Sequence[str] | None,
        profile: UserProfile | None,
    ) -> JudgeResult:
        if not outputs:
            return JudgeResult(score=0.0, passed=False, rationale="No outputs provided")

        if preferred_outputs is not None and len(preferred_outputs) == len(outputs):
            # Direct comparison with preferred outputs
            similarities = [_similarity(out, pref) for out, pref in zip(outputs, preferred_outputs)]
            avg_sim = mean(similarities) if similarities else 0.0
            passed = avg_sim >= self.config.min_preference_alignment
            return JudgeResult(
                score=avg_sim,
                passed=passed,
                rationale=f"Preference alignment (reference-based): {avg_sim:.3f}",
                metadata={"method": "reference_similarity", "sample_count": len(outputs)},
            )

        # Profile-based alignment: measure how well output matches user preference vector
        if profile is not None:
            pref_vector = profile.get_preference_vector()
            if pref_vector:
                scores: list[float] = []
                for output in outputs:
                    output_lower = output.lower()
                    matches = 0
                    total_weight = 0.0
                    for key, weight in pref_vector.items():
                        total_weight += weight
                        # Simple keyword presence as proxy
                        category, pref_name = key.split(".", 1) if "." in key else ("", key)
                        if pref_name.lower() in output_lower:
                            matches += weight
                    score = matches / max(total_weight, 0.01) if total_weight > 0 else 0.0
                    scores.append(min(1.0, score))
                avg_score = mean(scores) if scores else 0.0
                passed = avg_score >= self.config.min_preference_alignment
                return JudgeResult(
                    score=avg_score,
                    passed=passed,
                    rationale=f"Preference alignment (profile-based): {avg_score:.3f}",
                    metadata={"method": "profile_vector", "sample_count": len(outputs)},
                )

        return JudgeResult(
            score=0.5,
            passed=True,
            rationale="No preference reference or profile available; returning neutral score",
            metadata={"method": "neutral_fallback"},
        )

    # ------------------------------------------------------------------
    # Consistency
    # ------------------------------------------------------------------
    def evaluate_consistency(
        self,
        outputs: Sequence[str],
        repeated_prompts: Sequence[str] | None = None,
    ) -> JudgeResult:
        if not outputs:
            return JudgeResult(score=0.0, passed=False, rationale="No outputs provided")
        if len(outputs) < 2:
            return JudgeResult(
                score=0.5,
                passed=True,
                rationale="Only one output; consistency not measurable, returning neutral",
                metadata={"sample_count": 1},
            )

        # If repeated_prompts provided, group by prompt and compute intra-group similarity
        if repeated_prompts is not None and len(repeated_prompts) == len(outputs):
            groups: dict[str, list[str]] = {}
            for prompt, output in zip(repeated_prompts, outputs):
                groups.setdefault(prompt, []).append(output)

            group_scores: list[float] = []
            for prompt, group_outputs in groups.items():
                if len(group_outputs) < 2:
                    continue
                # Average pairwise similarity within group
                sims: list[float] = []
                for i in range(len(group_outputs)):
                    for j in range(i + 1, len(group_outputs)):
                        sims.append(_similarity(group_outputs[i], group_outputs[j]))
                group_scores.append(mean(sims) if sims else 0.0)

            avg_score = mean(group_scores) if group_scores else 0.0
        else:
            # Global pairwise similarity across all outputs
            sims: list[float] = []
            for i in range(len(outputs)):
                for j in range(i + 1, len(outputs)):
                    sims.append(_similarity(outputs[i], outputs[j]))
            avg_score = mean(sims) if sims else 0.0

        passed = avg_score >= self.config.min_consistency_score
        rationale = f"Consistency score: {avg_score:.3f} (across {len(outputs)} outputs)"
        return JudgeResult(
            score=avg_score,
            passed=passed,
            rationale=rationale,
            metadata={"sample_count": len(outputs), "pairwise_comparisons": len(sims) if 'sims' in locals() else None},
        )

    # ------------------------------------------------------------------
    # Generic quality
    # ------------------------------------------------------------------
    def evaluate_generic_quality(
        self,
        prompts: Sequence[str],
        outputs: Sequence[str],
        references: Sequence[str | None] | None,
    ) -> JudgeResult:
        if not outputs:
            return JudgeResult(score=0.0, passed=False, rationale="No outputs provided")

        fluency_scores: list[float] = []
        relevance_scores: list[float] = []

        for idx, (prompt, output) in enumerate(zip(prompts, outputs)):
            # Fluency proxy: length-normalized punctuation ratio + word count
            words = output.split()
            word_count = len(words)
            if word_count == 0:
                fluency_scores.append(0.0)
                relevance_scores.append(0.0)
                continue

            # Simple fluency: reasonable length and sentence structure
            sentences = re.split(r"[.!?。！？]", output)
            avg_sentence_len = word_count / max(len([s for s in sentences if s.strip()]), 1)
            fluency = 0.5
            if 3 <= avg_sentence_len <= 30:
                fluency = 0.8
            elif avg_sentence_len > 0:
                fluency = 0.6
            if word_count < 5:
                fluency = 0.3
            fluency_scores.append(fluency)

            # Relevance proxy: prompt-output similarity
            relevance = _similarity(prompt, output)
            # Boost if reference is available and matched
            if references is not None and idx < len(references) and references[idx]:
                ref_sim = _similarity(output, references[idx])
                relevance = max(relevance, ref_sim * 0.8)
            relevance_scores.append(relevance)

        avg_fluency = mean(fluency_scores) if fluency_scores else 0.0
        avg_relevance = mean(relevance_scores) if relevance_scores else 0.0
        overall = (avg_fluency + avg_relevance) / 2.0

        return JudgeResult(
            score=overall,
            passed=overall >= 0.5,
            rationale=f"Generic quality: fluency={avg_fluency:.3f}, relevance={avg_relevance:.3f}",
            metadata={
                "fluency": avg_fluency,
                "relevance": avg_relevance,
                "sample_count": len(outputs),
            },
        )


class LLMJudge(BaseJudge):
    """LLM-based judge (mockable interface)."""

    def __init__(self, config: PersonalizedEvalConfig) -> None:
        self.config = config

    def _mock_evaluate(self, dimension: str, outputs: Sequence[str], **kwargs: Any) -> JudgeResult:
        """Mock evaluation for testing and fallback."""
        # Simple heuristic fallback when LLM is not available
        if dimension == "style_preference_hit_rate":
            return RuleBasedJudge(self.config).evaluate_style_preference_hit_rate(
                outputs, kwargs.get("metadata_list", []), kwargs.get("profile")
            )
        if dimension == "profile_aware_accuracy":
            return RuleBasedJudge(self.config).evaluate_profile_aware_accuracy(
                kwargs.get("prompts", []), outputs, kwargs.get("profile")
            )
        if dimension == "preference_alignment":
            return RuleBasedJudge(self.config).evaluate_preference_alignment(
                outputs, kwargs.get("preferred_outputs"), kwargs.get("profile")
            )
        if dimension == "consistency":
            return RuleBasedJudge(self.config).evaluate_consistency(
                outputs, kwargs.get("repeated_prompts")
            )
        if dimension == "generic_quality":
            return RuleBasedJudge(self.config).evaluate_generic_quality(
                kwargs.get("prompts", []), outputs, kwargs.get("references")
            )
        return JudgeResult(score=0.5, passed=True, rationale=f"Mock {dimension}: neutral fallback")

    def evaluate_style_preference_hit_rate(
        self,
        outputs: Sequence[str],
        metadata_list: Sequence[dict[str, Any]],
        profile: UserProfile | None,
    ) -> JudgeResult:
        if self.config.llm_judge_model == "mock":
            return self._mock_evaluate("style_preference_hit_rate", outputs, metadata_list=metadata_list, profile=profile)
        # Real LLM evaluation would go here
        raise NotImplementedError("Real LLM judge not yet implemented")

    def evaluate_profile_aware_accuracy(
        self,
        prompts: Sequence[str],
        outputs: Sequence[str],
        profile: UserProfile | None,
    ) -> JudgeResult:
        if self.config.llm_judge_model == "mock":
            return self._mock_evaluate("profile_aware_accuracy", outputs, prompts=prompts, profile=profile)
        raise NotImplementedError("Real LLM judge not yet implemented")

    def evaluate_preference_alignment(
        self,
        outputs: Sequence[str],
        preferred_outputs: Sequence[str] | None,
        profile: UserProfile | None,
    ) -> JudgeResult:
        if self.config.llm_judge_model == "mock":
            return self._mock_evaluate("preference_alignment", outputs, preferred_outputs=preferred_outputs, profile=profile)
        raise NotImplementedError("Real LLM judge not yet implemented")

    def evaluate_consistency(
        self,
        outputs: Sequence[str],
        repeated_prompts: Sequence[str] | None = None,
    ) -> JudgeResult:
        if self.config.llm_judge_model == "mock":
            return self._mock_evaluate("consistency", outputs, repeated_prompts=repeated_prompts)
        raise NotImplementedError("Real LLM judge not yet implemented")

    def evaluate_generic_quality(
        self,
        prompts: Sequence[str],
        outputs: Sequence[str],
        references: Sequence[str | None] | None,
    ) -> JudgeResult:
        if self.config.llm_judge_model == "mock":
            return self._mock_evaluate("generic_quality", outputs, prompts=prompts, references=references)
        raise NotImplementedError("Real LLM judge not yet implemented")


class HybridJudge(BaseJudge):
    """Hybrid judge combining rule-based and LLM-based evaluation."""

    def __init__(self, config: PersonalizedEvalConfig) -> None:
        self.config = config
        self.rule_judge = RuleBasedJudge(config)
        self.llm_judge = LLMJudge(config)

    def _combine(self, rule_result: JudgeResult, llm_result: JudgeResult, rule_weight: float = 0.6) -> JudgeResult:
        combined_score = rule_weight * rule_result.score + (1 - rule_weight) * llm_result.score
        return JudgeResult(
            score=combined_score,
            passed=combined_score >= 0.5,
            rationale=f"Hybrid: rule={rule_result.score:.3f}, llm={llm_result.score:.3f}, combined={combined_score:.3f}",
            metadata={
                "rule": {"score": rule_result.score, "rationale": rule_result.rationale},
                "llm": {"score": llm_result.score, "rationale": llm_result.rationale},
                "rule_weight": rule_weight,
            },
        )

    def evaluate_style_preference_hit_rate(
        self,
        outputs: Sequence[str],
        metadata_list: Sequence[dict[str, Any]],
        profile: UserProfile | None,
    ) -> JudgeResult:
        rule = self.rule_judge.evaluate_style_preference_hit_rate(outputs, metadata_list, profile)
        llm = self.llm_judge.evaluate_style_preference_hit_rate(outputs, metadata_list, profile)
        return self._combine(rule, llm)

    def evaluate_profile_aware_accuracy(
        self,
        prompts: Sequence[str],
        outputs: Sequence[str],
        profile: UserProfile | None,
    ) -> JudgeResult:
        rule = self.rule_judge.evaluate_profile_aware_accuracy(prompts, outputs, profile)
        llm = self.llm_judge.evaluate_profile_aware_accuracy(prompts, outputs, profile)
        return self._combine(rule, llm)

    def evaluate_preference_alignment(
        self,
        outputs: Sequence[str],
        preferred_outputs: Sequence[str] | None,
        profile: UserProfile | None,
    ) -> JudgeResult:
        rule = self.rule_judge.evaluate_preference_alignment(outputs, preferred_outputs, profile)
        llm = self.llm_judge.evaluate_preference_alignment(outputs, preferred_outputs, profile)
        return self._combine(rule, llm)

    def evaluate_consistency(
        self,
        outputs: Sequence[str],
        repeated_prompts: Sequence[str] | None = None,
    ) -> JudgeResult:
        rule = self.rule_judge.evaluate_consistency(outputs, repeated_prompts)
        llm = self.llm_judge.evaluate_consistency(outputs, repeated_prompts)
        return self._combine(rule, llm)

    def evaluate_generic_quality(
        self,
        prompts: Sequence[str],
        outputs: Sequence[str],
        references: Sequence[str | None] | None,
    ) -> JudgeResult:
        rule = self.rule_judge.evaluate_generic_quality(prompts, outputs, references)
        llm = self.llm_judge.evaluate_generic_quality(prompts, outputs, references)
        return self._combine(rule, llm)


# ---------------------------------------------------------------------------
# PersonalizedEvaluator
# ---------------------------------------------------------------------------
class PersonalizedEvaluator:
    """Main evaluator for personalized model quality assessment."""

    def __init__(self, config: Optional[PersonalizedEvalConfig] = None, home: str | None = None) -> None:
        self.config = config or PersonalizedEvalConfig()
        self.home = home
        self._judge = self._create_judge()
        self._profile: UserProfile | None = None
        if self.config.user_id:
            self._load_profile(self.config.user_id)

    def _create_judge(self) -> BaseJudge:
        if self.config.judge_backend == JudgeBackend.RULE:
            return RuleBasedJudge(self.config)
        if self.config.judge_backend == JudgeBackend.LLM:
            return LLMJudge(self.config)
        return HybridJudge(self.config)

    def _load_profile(self, user_id: str) -> None:
        try:
            extractor = ProfileExtractor(user_id, home=self.home)
            self._profile = extractor.profile
        except Exception:
            self._profile = None

    @property
    def profile(self) -> UserProfile | None:
        return self._profile

    def evaluate(
        self,
        prompts: Sequence[str],
        outputs: Sequence[str],
        *,
        metadata_list: Sequence[dict[str, Any]] | None = None,
        preferred_outputs: Sequence[str] | None = None,
        references: Sequence[str | None] | None = None,
        repeated_prompts: Sequence[str] | None = None,
        adapter_version: str = "latest",
        base_model: str = "base",
    ) -> EvalReport:
        """Run full personalized evaluation on a set of model outputs."""
        if not outputs:
            raise ValueError("No outputs provided for evaluation")

        meta = list(metadata_list) if metadata_list is not None else [{} for _ in outputs]

        # Personalized metrics
        style_result = self._judge.evaluate_style_preference_hit_rate(outputs, meta, self._profile)
        profile_result = self._judge.evaluate_profile_aware_accuracy(prompts, outputs, self._profile)
        alignment_result = self._judge.evaluate_preference_alignment(outputs, preferred_outputs, self._profile)
        consistency_result = self._judge.evaluate_consistency(outputs, repeated_prompts)

        # Generic quality metrics
        generic_result = self._judge.evaluate_generic_quality(prompts, outputs, references)
        fluency = generic_result.metadata.get("fluency", 0.0)
        relevance = generic_result.metadata.get("relevance", 0.0)

        # Weighted overall personalized score
        overall_personalized = (
            self.config.style_preference_weight * style_result.score
            + self.config.profile_aware_weight * profile_result.score
            + self.config.preference_alignment_weight * alignment_result.score
            + self.config.consistency_weight * consistency_result.score
        )

        overall_generic = generic_result.score

        # Build differentiation note
        personalization_summary = self._build_personalization_summary(
            style_result, profile_result, alignment_result, consistency_result
        )
        generic_quality_summary = self._build_generic_quality_summary(generic_result)
        differentiation_note = self._build_differentiation_note(overall_personalized, overall_generic)

        # Recommendation
        recommendation = self._make_recommendation(
            overall_personalized, overall_generic, style_result, profile_result, alignment_result, consistency_result
        )

        # Per-sample details
        details: list[dict[str, Any]] = []
        for idx, (prompt, output) in enumerate(zip(prompts, outputs)):
            details.append({
                "index": idx,
                "prompt": prompt,
                "output": output,
                "reference": references[idx] if references is not None and idx < len(references) else None,
                "metadata": meta[idx] if idx < len(meta) else {},
            })

        return EvalReport(
            adapter_version=adapter_version,
            base_model=base_model,
            user_id=self.config.user_id,
            num_test_samples=len(outputs),
            style_preference_hit_rate=style_result.score,
            profile_aware_accuracy=profile_result.score,
            preference_alignment=alignment_result.score,
            consistency_score=consistency_result.score,
            generic_quality_score=overall_generic,
            fluency_score=fluency,
            relevance_score=relevance,
            overall_personalized_score=overall_personalized,
            overall_generic_score=overall_generic,
            personalization_summary=personalization_summary,
            generic_quality_summary=generic_quality_summary,
            differentiation_note=differentiation_note,
            recommendation=recommendation,
            details=details,
            judge_backend=self.config.judge_backend.value,
            judge_model=self.config.llm_judge_model,
            metadata={
                "style_result": {"score": style_result.score, "passed": style_result.passed, "rationale": style_result.rationale},
                "profile_result": {"score": profile_result.score, "passed": profile_result.passed, "rationale": profile_result.rationale},
                "alignment_result": {"score": alignment_result.score, "passed": alignment_result.passed, "rationale": alignment_result.rationale},
                "consistency_result": {"score": consistency_result.score, "passed": consistency_result.passed, "rationale": consistency_result.rationale},
                "generic_result": {"score": generic_result.score, "passed": generic_result.passed, "rationale": generic_result.rationale},
            },
        )

    def _build_personalization_summary(
        self,
        style: JudgeResult,
        profile: JudgeResult,
        alignment: JudgeResult,
        consistency: JudgeResult,
    ) -> str:
        parts = []
        if style.passed:
            parts.append(f"style hit rate OK ({style.score:.2f})")
        else:
            parts.append(f"style hit rate low ({style.score:.2f})")
        if profile.passed:
            parts.append(f"profile-aware accuracy OK ({profile.score:.2f})")
        else:
            parts.append(f"profile-aware accuracy low ({profile.score:.2f})")
        if alignment.passed:
            parts.append(f"preference alignment OK ({alignment.score:.2f})")
        else:
            parts.append(f"preference alignment low ({alignment.score:.2f})")
        if consistency.passed:
            parts.append(f"consistency OK ({consistency.score:.2f})")
        else:
            parts.append(f"consistency low ({consistency.score:.2f})")
        return "; ".join(parts)

    def _build_generic_quality_summary(self, generic: JudgeResult) -> str:
        fluency = generic.metadata.get("fluency", 0.0)
        relevance = generic.metadata.get("relevance", 0.0)
        return f"Generic quality: fluency={fluency:.2f}, relevance={relevance:.2f}, overall={generic.score:.2f}"

    def _build_differentiation_note(self, personalized_score: float, generic_score: float) -> str:
        gap = personalized_score - generic_score
        if gap > 0.15:
            return (
                f"Personalization score ({personalized_score:.2f}) is significantly higher than "
                f"generic quality score ({generic_score:.2f}). The model is delivering strong personalized value."
            )
        if gap < -0.15:
            return (
                f"Generic quality score ({generic_score:.2f}) is significantly higher than "
                f"personalization score ({personalized_score:.2f}). The model may be generic but not personalized."
            )
        return (
            f"Personalization score ({personalized_score:.2f}) and generic quality score ({generic_score:.2f}) "
            f"are closely aligned. The model is balanced but not strongly differentiated."
        )

    def _make_recommendation(
        self,
        overall_personalized: float,
        overall_generic: float,
        style: JudgeResult,
        profile: JudgeResult,
        alignment: JudgeResult,
        consistency: JudgeResult,
    ) -> Literal["deploy", "keep_previous", "needs_more_data"]:
        # Require both personalization and generic quality to be acceptable
        if overall_generic < 0.4:
            return "needs_more_data"
        personalized_pass = all([
            style.passed,
            profile.passed,
            alignment.passed,
            consistency.passed,
        ])
        if personalized_pass and overall_personalized >= 0.6:
            return "deploy"
        if overall_personalized >= 0.5 and overall_generic >= 0.5:
            return "keep_previous"
        return "needs_more_data"

    def compare_personalized_vs_generic(
        self,
        personalized_report: EvalReport | dict[str, Any],
        generic_report: EvalReport | dict[str, Any],
        *,
        delta_threshold: float = 0.05,
    ) -> PersonalizedVsGenericComparison:
        """Compare two reports to explicitly distinguish personalized vs generic quality improvements.

        The ``personalized_report`` should come from the personalized (adapted) model,
        and ``generic_report`` from the base/generic model.
        """
        left = self._coerce_report(personalized_report)
        right = self._coerce_report(generic_report)

        personalized_metrics = ["style_preference_hit_rate", "profile_aware_accuracy", "preference_alignment", "consistency_score"]
        generic_metrics = ["generic_quality_score", "fluency_score", "relevance_score"]

        personalized_delta = self._compute_metric_delta(left, right, personalized_metrics)
        generic_delta = self._compute_metric_delta(left, right, generic_metrics)

        personalization_wins = personalized_delta > delta_threshold
        generic_quality_wins = generic_delta > delta_threshold
        tradeoff = (personalized_delta < -delta_threshold and generic_delta > delta_threshold) or (
            personalized_delta > delta_threshold and generic_delta < -delta_threshold
        )

        if personalization_wins and not tradeoff:
            recommendation: Literal["deploy", "keep_previous", "needs_more_data"] = "deploy"
            summary = f"Personalization improved by {personalized_delta:.3f} with acceptable generic quality change ({generic_delta:+.3f})"
        elif generic_quality_wins and not personalization_wins:
            recommendation = "keep_previous"
            summary = f"Generic quality improved by {generic_delta:.3f} but personalization did not improve significantly ({personalized_delta:+.3f})"
        elif tradeoff:
            recommendation = "needs_more_data"
            summary = (
                f"Trade-off detected: personalization {personalized_delta:+.3f}, "
                f"generic quality {generic_delta:+.3f}. Needs more data to decide."
            )
        else:
            recommendation = "needs_more_data"
            summary = f"No significant change: personalization {personalized_delta:+.3f}, generic quality {generic_delta:+.3f}"

        return PersonalizedVsGenericComparison(
            personalized_delta=personalized_delta,
            generic_delta=generic_delta,
            personalization_wins=personalization_wins,
            generic_quality_wins=generic_quality_wins,
            summary=summary,
            tradeoff_warning=tradeoff,
            recommendation=recommendation,
        )

    @staticmethod
    def _coerce_report(report: EvalReport | dict[str, Any]) -> dict[str, float]:
        if isinstance(report, EvalReport):
            return {
                "style_preference_hit_rate": report.style_preference_hit_rate,
                "profile_aware_accuracy": report.profile_aware_accuracy,
                "preference_alignment": report.preference_alignment,
                "consistency_score": report.consistency_score,
                "generic_quality_score": report.generic_quality_score,
                "fluency_score": report.fluency_score,
                "relevance_score": report.relevance_score,
                "overall_personalized_score": report.overall_personalized_score,
                "overall_generic_score": report.overall_generic_score,
            }
        data = dict(report)
        return {
            "style_preference_hit_rate": float(data.get("style_preference_hit_rate", 0.0)),
            "profile_aware_accuracy": float(data.get("profile_aware_accuracy", 0.0)),
            "preference_alignment": float(data.get("preference_alignment", 0.0)),
            "consistency_score": float(data.get("consistency_score", 0.0)),
            "generic_quality_score": float(data.get("generic_quality_score", 0.0)),
            "fluency_score": float(data.get("fluency_score", 0.0)),
            "relevance_score": float(data.get("relevance_score", 0.0)),
            "overall_personalized_score": float(data.get("overall_personalized_score", 0.0)),
            "overall_generic_score": float(data.get("overall_generic_score", 0.0)),
        }

    @staticmethod
    def _compute_metric_delta(left: dict[str, float], right: dict[str, float], metrics: Sequence[str]) -> float:
        deltas = [left.get(m, 0.0) - right.get(m, 0.0) for m in metrics if m in left or m in right]
        if not deltas:
            return 0.0
        return sum(deltas) / len(deltas)
