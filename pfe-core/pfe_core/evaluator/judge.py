"""Judge protocol and local evaluator for PFE."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Any, Literal, Optional, Sequence

from ..curator.datasets import sample_dataset_split, select_holdout_samples
from ..curator.distillation import TrainingSample
from ..profile_extractor import get_user_profile_store

try:  # pragma: no cover - optional dependency during early bootstrap
    from pydantic import BaseModel, ConfigDict, Field
except Exception:  # pragma: no cover
    class BaseModel:  # type: ignore[override]
        def __init__(self, **data: Any) -> None:
            for key, value in data.items():
                setattr(self, key, value)

        def model_dump(self) -> dict[str, Any]:
            return dict(self.__dict__)

        @classmethod
        def model_validate(cls, data: Any) -> "BaseModel":
            if isinstance(data, dict):
                return cls(**data)
            if hasattr(data, "model_dump"):
                return cls(**data.model_dump())
            return cls(**dict(data))

    class ConfigDict(dict):  # type: ignore[override]
        pass

    def Field(default: Any = None, *, default_factory: Optional[Any] = None, **_: Any) -> Any:  # type: ignore[override]
        if default_factory is not None:
            return default_factory()
        return default


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


@dataclass
class EvalDetail:
    sample_id: str
    dataset_split: str
    prompt: str
    base_output: str
    adapted_output: str
    reference_output: Optional[str]
    scores: dict[str, float]
    winner: str
    reason: str


class EvalDetailSchema(BaseModel):
    model_config = ConfigDict(extra="allow")

    sample_id: str
    dataset_split: str
    prompt: str
    base_output: str
    adapted_output: str
    reference_output: Optional[str] = None
    scores: dict[str, float] = Field(default_factory=dict)
    winner: str = "tie"
    reason: str = ""


@dataclass
class EvalReport:
    adapter_version: str
    base_model: str
    num_test_samples: int
    scores: dict[str, float]
    comparison: str
    recommendation: str
    details: list[EvalDetail]
    judge_model: str = "local-heuristic"
    judge_prompt_version: str = "v1"
    created_at: str = field(default_factory=_now)
    metadata: dict[str, Any] = field(default_factory=dict)


class EvalReportSchema(BaseModel):
    model_config = ConfigDict(extra="allow")

    adapter_version: str
    base_model: str
    num_test_samples: int
    scores: dict[str, float] = Field(default_factory=dict)
    comparison: str = "neutral"
    recommendation: str = "needs_more_data"
    details: list[EvalDetailSchema] = Field(default_factory=list)
    judge_model: str = "local-heuristic"
    judge_prompt_version: str = "v1"
    created_at: str = Field(default_factory=_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


@dataclass
class EvalComparisonDetail:
    metric: str
    left_score: float
    right_score: float
    delta: float
    preferred: Literal["left", "right", "tie"]
    reason: str


@dataclass
class EvalComparisonDimensionSummary:
    dimension: Literal["quality", "personalization", "other"]
    metrics: list[str] = field(default_factory=list)
    left_score: float = 0.0
    right_score: float = 0.0
    delta: float = 0.0
    preferred: Literal["left", "right", "tie"] = "tie"
    reason: str = ""


@dataclass
class EvalComparisonReport:
    left_adapter_version: str
    right_adapter_version: str
    base_model: str
    comparison: Literal["left_better", "right_better", "neutral"]
    winner: Literal["left", "right", "tie"]
    recommendation: Literal["keep_left", "keep_right", "needs_more_data"]
    overall_delta: float
    left_scores: dict[str, float] = field(default_factory=dict)
    right_scores: dict[str, float] = field(default_factory=dict)
    score_deltas: dict[str, float] = field(default_factory=dict)
    details: list[EvalComparisonDetail] = field(default_factory=list)
    dimension_summaries: list[EvalComparisonDimensionSummary] = field(default_factory=list)
    personalization_summary: str = ""
    quality_summary: str = ""
    summary_line: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_now)


@dataclass
class JudgeConfig:
    judge_model: str = "local-heuristic"
    judge_prompt_version: str = "v1"
    require_holdout_split: bool = True
    forbid_teacher_test_overlap: bool = True
    allowed_eval_splits: tuple[str, ...] = ("val", "test")
    style_keywords: tuple[str, ...] = ("理解", "感受", "支持", "共情", "温和", "一起", "慢慢", "可以")
    preferred_eval_splits: tuple[str, ...] = ("test", "val")
    max_eval_samples: int = 200
    # Profile-aware evaluation settings
    user_id: str | None = None
    enable_personalization_eval: bool = True
    personalization_weight: float = 0.3  # Weight for personalization in overall score


class JudgeConfigSchema(BaseModel):
    model_config = ConfigDict(extra="allow")

    judge_model: str = "local-heuristic"
    judge_prompt_version: str = "v1"
    require_holdout_split: bool = True
    forbid_teacher_test_overlap: bool = True
    allowed_eval_splits: tuple[str, ...] = ("val", "test")
    style_keywords: tuple[str, ...] = ("理解", "感受", "支持", "共情", "温和", "一起", "慢慢", "可以")
    preferred_eval_splits: tuple[str, ...] = ("test", "val")
    max_eval_samples: int = 200


class JudgeProtocol(ABC):
    @abstractmethod
    def prepare_eval_set(self, samples: Sequence[TrainingSample]) -> list[TrainingSample]:
        """Select only holdout/test samples for evaluation."""

    @abstractmethod
    def compare(self, base_output: str, adapted_output: str, reference: Optional[str]) -> dict[str, Any]:
        """Return per-sample comparison details."""


def _coerce_eval_report_payload(report: EvalReport | EvalReportSchema | dict[str, Any]) -> dict[str, Any]:
    if is_dataclass(report):
        return asdict(report)
    if hasattr(report, "model_dump"):
        return dict(report.model_dump())
    if isinstance(report, dict):
        return dict(report)
    raise TypeError(f"unsupported eval report type: {type(report)!r}")


def _comparison_dimension(metric: str) -> Literal["quality", "personalization", "other"]:
    if metric in {"style_match", "preference_alignment", "personality_consistency", "style_preference_hit_rate"}:
        return "personalization"
    if metric == "quality_preservation":
        return "quality"
    return "other"


def _summarize_dimension(
    dimension: Literal["quality", "personalization", "other"],
    metrics: Sequence[str],
    left_scores: dict[str, float],
    right_scores: dict[str, float],
    *,
    delta_threshold: float,
) -> EvalComparisonDimensionSummary | None:
    selected_metrics = [metric for metric in metrics if metric in left_scores or metric in right_scores]
    if not selected_metrics:
        return None

    left_average = sum(float(left_scores.get(metric, 0.0)) for metric in selected_metrics) / len(selected_metrics)
    right_average = sum(float(right_scores.get(metric, 0.0)) for metric in selected_metrics) / len(selected_metrics)
    delta = right_average - left_average
    if delta > delta_threshold:
        preferred: Literal["left", "right", "tie"] = "right"
        reason = f"right improved {dimension} by {delta:.3f}"
    elif delta < -delta_threshold:
        preferred = "left"
        reason = f"left improved {dimension} by {abs(delta):.3f}"
    else:
        preferred = "tie"
        reason = f"{dimension} stayed within +/-{delta_threshold:.3f}"

    return EvalComparisonDimensionSummary(
        dimension=dimension,
        metrics=selected_metrics,
        left_score=left_average,
        right_score=right_average,
        delta=delta,
        preferred=preferred,
        reason=reason,
    )


def compare_eval_reports(
    left_report: EvalReport | EvalReportSchema | dict[str, Any],
    right_report: EvalReport | EvalReportSchema | dict[str, Any],
    *,
    left_label: Optional[str] = None,
    right_label: Optional[str] = None,
    delta_threshold: float = 0.05,
) -> EvalComparisonReport:
    """Build a reusable version-to-version comparison report from two evaluation reports."""

    left = _coerce_eval_report_payload(left_report)
    right = _coerce_eval_report_payload(right_report)
    left_scores = {str(key): float(value) for key, value in dict(left.get("scores") or {}).items()}
    right_scores = {str(key): float(value) for key, value in dict(right.get("scores") or {}).items()}
    metric_names = sorted(set(left_scores) | set(right_scores))
    score_deltas: dict[str, float] = {}
    details: list[EvalComparisonDetail] = []

    for metric in metric_names:
        left_score = float(left_scores.get(metric, 0.0))
        right_score = float(right_scores.get(metric, 0.0))
        delta = right_score - left_score
        score_deltas[metric] = delta
        if delta > delta_threshold:
            preferred: Literal["left", "right", "tie"] = "right"
            reason = f"right improved {metric} by {delta:.3f}"
        elif delta < -delta_threshold:
            preferred = "left"
            reason = f"left improved {metric} by {abs(delta):.3f}"
        else:
            preferred = "tie"
            reason = f"{metric} stayed within +/-{delta_threshold:.3f}"
        details.append(
            EvalComparisonDetail(
                metric=metric,
                left_score=left_score,
                right_score=right_score,
                delta=delta,
                preferred=preferred,
                reason=reason,
            )
        )

    overall_delta = sum(score_deltas.values()) / max(len(score_deltas), 1)
    dimension_summaries: list[EvalComparisonDimensionSummary] = []
    for dimension, metrics in (
        ("personalization", ("style_match", "preference_alignment", "personality_consistency", "style_preference_hit_rate")),
        ("quality", ("quality_preservation",)),
    ):
        summary = _summarize_dimension(dimension, metrics, left_scores, right_scores, delta_threshold=delta_threshold)
        if summary is not None:
            dimension_summaries.append(summary)

    dimension_summary_map = {summary.dimension: summary for summary in dimension_summaries}
    personalization_summary = dimension_summary_map.get("personalization")
    quality_summary = dimension_summary_map.get("quality")
    personalization_delta = personalization_summary.delta if personalization_summary is not None else 0.0
    quality_delta = quality_summary.delta if quality_summary is not None else 0.0

    if overall_delta > delta_threshold:
        comparison: Literal["left_better", "right_better", "neutral"] = "right_better"
        winner: Literal["left", "right", "tie"] = "right"
        recommendation: Literal["keep_left", "keep_right", "needs_more_data"] = (
            "keep_right"
            if personalization_delta > 0.0 and quality_delta >= -delta_threshold
            else "needs_more_data"
        )
    elif overall_delta < -delta_threshold:
        comparison = "left_better"
        winner = "left"
        recommendation = "keep_left" if personalization_delta < 0.0 and quality_delta <= delta_threshold else "needs_more_data"
    else:
        comparison = "neutral"
        winner = "tie"
        recommendation = "needs_more_data"

    left_label_value = left_label or str(left.get("adapter_version") or "left")
    right_label_value = right_label or str(right.get("adapter_version") or "right")
    metadata = {
        "left_report": {
            "adapter_version": left.get("adapter_version"),
            "base_model": left.get("base_model"),
            "comparison": left.get("comparison"),
            "recommendation": left.get("recommendation"),
            "num_test_samples": left.get("num_test_samples"),
        },
        "right_report": {
            "adapter_version": right.get("adapter_version"),
            "base_model": right.get("base_model"),
            "comparison": right.get("comparison"),
            "recommendation": right.get("recommendation"),
            "num_test_samples": right.get("num_test_samples"),
        },
        "delta_threshold": delta_threshold,
        "shared_metrics": metric_names,
        "left_label": left_label_value,
        "right_label": right_label_value,
        "dimension_summaries": [asdict(summary) for summary in dimension_summaries],
        "personalization_delta": personalization_delta,
        "quality_delta": quality_delta,
    }
    if left.get("base_model") and right.get("base_model") and left.get("base_model") != right.get("base_model"):
        metadata["base_model_mismatch"] = {
            "left": left.get("base_model"),
            "right": right.get("base_model"),
        }

    summary_bits: list[str] = []
    if personalization_summary is not None:
        summary_bits.append(
            "personalization="
            + ", ".join(f"{metric}:{_format_score_delta(score_deltas.get(metric, 0.0))}" for metric in personalization_summary.metrics)
        )
    if quality_summary is not None:
        summary_bits.append(
            "quality="
            + ", ".join(f"{metric}:{_format_score_delta(score_deltas.get(metric, 0.0))}" for metric in quality_summary.metrics)
        )
    if not summary_bits:
        summary_bits.append("no comparable dimensions")
    summary_line = "; ".join(summary_bits)

    return EvalComparisonReport(
        left_adapter_version=left_label_value,
        right_adapter_version=right_label_value,
        base_model=str(left.get("base_model") or right.get("base_model") or ""),
        comparison=comparison,
        winner=winner,
        recommendation=recommendation,
        overall_delta=overall_delta,
        left_scores=left_scores,
        right_scores=right_scores,
        score_deltas=score_deltas,
        details=details,
        dimension_summaries=dimension_summaries,
        personalization_summary=_format_dimension_summary(personalization_summary),
        quality_summary=_format_dimension_summary(quality_summary),
        summary_line=summary_line,
        metadata=metadata,
    )


def _format_score_delta(value: float) -> str:
    if value >= 0:
        return f"+{value:.3f}"
    return f"{value:.3f}"


def _format_dimension_summary(summary: Optional[EvalComparisonDimensionSummary]) -> str:
    if summary is None:
        return ""
    metric_bits = ", ".join(summary.metrics)
    return (
        f"{summary.dimension}: left={summary.left_score:.3f} | right={summary.right_score:.3f} | "
        f"delta={_format_score_delta(summary.delta)} | preferred={summary.preferred} | metrics={metric_bits} | reason={summary.reason}"
    )


class LocalJudge(JudgeProtocol):
    """Deterministic rule-based judge used for Phase 0."""

    def __init__(self, config: Optional[JudgeConfig] = None, home: str | None = None) -> None:
        self.config = config or JudgeConfig()
        self.home = home
        self._profile = None
        if self.config.user_id and self.config.enable_personalization_eval:
            try:
                store = get_user_profile_store(home)
                self._profile = store.get_profile(self.config.user_id)
            except Exception:
                pass

    def _compute_profile_match_score(self, output: str) -> float:
        """Compute how well output matches user profile preferences."""
        if not self._profile:
            return 0.5  # Neutral if no profile

        output_lower = output.lower()
        match_scores = []

        # Check style alignment
        top_styles = self._profile.get_top_style_preferences(2)
        style_indicators = {
            "formal": ["please", "thank you", "would you", "请", "您好", "谢谢"],
            "casual": ["hey", "hi", "yeah", "嘿", "嗨", "咋"],
            "concise": ["brief", "short", "summary", "简短", "简洁", "概要"],
            "detailed": ["detailed", "comprehensive", "详细", "具体", "深入"],
            "technical": ["technical", "implementation", "技术", "实现", "架构"],
            "non_technical": ["simple", "plain", "简单", "通俗", "易懂"],
        }
        for style, score in top_styles:
            indicators = style_indicators.get(style, [style])
            if any(ind in output_lower for ind in indicators):
                match_scores.append(score)

        # Check domain alignment
        top_domains = self._profile.get_top_domains(2)
        domain_keywords = {
            "programming": ["code", "programming", "function", "代码", "编程"],
            "writing": ["write", "essay", "article", "写作", "文章"],
            "learning": ["learn", "tutorial", "学习", "教程"],
            "analysis": ["analyze", "data", "分析", "数据"],
            "creative": ["creative", "design", "创意", "设计"],
            "business": ["business", "product", "商业", "产品"],
        }
        for domain, score in top_domains:
            keywords = domain_keywords.get(domain, [domain])
            if any(kw in output_lower for kw in keywords):
                match_scores.append(score)

        # Check interaction pattern alignment
        top_patterns = self._profile.get_top_interaction_patterns(2)
        pattern_indicators = {
            "likes_examples": ["example", "for instance", "such as", "例子", "示例"],
            "prefers_direct": ["direct", "straight", "直接", "干脆"],
            "wants_reasoning": ["because", "reason", "因此", "原因"],
            "prefers_code": ["code", "implement", "代码", "实现"],
        }
        for pattern, score in top_patterns:
            indicators = pattern_indicators.get(pattern, [pattern])
            if any(ind in output_lower for ind in indicators):
                match_scores.append(score)

        if match_scores:
            return sum(match_scores) / len(match_scores)
        return 0.5

    def prepare_eval_set(self, samples: Sequence[TrainingSample]) -> list[TrainingSample]:
        holdout = select_holdout_samples(
            samples,
            preferred_splits=self.config.preferred_eval_splits,
            max_samples=self.config.max_eval_samples,
        )
        prepared: list[TrainingSample] = []
        for sample in holdout:
            split = sample_dataset_split(sample)
            if split not in self.config.allowed_eval_splits:
                continue
            if self.config.forbid_teacher_test_overlap and sample.source == "teacher" and split == "test":
                continue
            prepared.append(sample)

        if self.config.require_holdout_split and not prepared:
            raise ValueError("no eligible holdout/test samples for evaluation")
        return prepared

    def compare(self, base_output: str, adapted_output: str, reference: Optional[str]) -> dict[str, Any]:
        style_keywords = self.config.style_keywords
        adapted_style = _keyword_density(adapted_output, style_keywords)
        base_style = _keyword_density(base_output, style_keywords)
        reference_overlap = _similarity(adapted_output, reference) if reference else _similarity(adapted_output, base_output)
        base_reference_overlap = _similarity(base_output, reference) if reference else 0.5
        quality_preservation = 0.55 + 0.45 * _similarity(base_output, adapted_output)
        personality_consistency = 0.55 + 0.45 * (
            0.6 * adapted_style + 0.4 * reference_overlap
        )

        scores = {
            "style_match": min(1.0, 0.35 + 0.65 * adapted_style),
            "preference_alignment": min(1.0, 0.4 + 0.6 * reference_overlap),
            "quality_preservation": min(1.0, quality_preservation),
            "personality_consistency": min(1.0, personality_consistency),
        }

        # Add profile-based personalization score if enabled
        if self.config.enable_personalization_eval and self._profile:
            adapted_profile_match = self._compute_profile_match_score(adapted_output)
            base_profile_match = self._compute_profile_match_score(base_output)
            scores["profile_match"] = adapted_profile_match
            scores["profile_match_delta"] = adapted_profile_match - base_profile_match

            # Adjust personalization scores with profile match
            personalization_boost = self.config.personalization_weight * adapted_profile_match
            scores["style_match"] = min(1.0, scores["style_match"] + personalization_boost * 0.1)
            scores["preference_alignment"] = min(1.0, scores["preference_alignment"] + personalization_boost * 0.1)

        adapted_total = scores["style_match"] + scores["preference_alignment"] + scores["quality_preservation"] + scores["personality_consistency"]
        base_total = (
            min(1.0, 0.35 + 0.65 * base_style)
            + min(1.0, 0.4 + 0.6 * base_reference_overlap)
            + min(1.0, 0.55 + 0.45 * _similarity(base_output, base_output))
            + min(1.0, 0.55 + 0.45 * (0.6 * base_style + 0.4 * base_reference_overlap))
        )

        if adapted_total > base_total + 0.15:
            comparison = "improved"
            winner = "adapted"
        elif adapted_total < base_total - 0.15:
            comparison = "degraded"
            winner = "base"
        else:
            comparison = "neutral"
            winner = "tie"

        reason = (
            f"personalization(style={scores['style_match']:.2f}, "
            f"preference={scores['preference_alignment']:.2f}, "
            f"consistency={scores['personality_consistency']:.2f}); "
            f"quality={scores['quality_preservation']:.2f}; "
            f"adapted_total={adapted_total:.2f}; base_total={base_total:.2f}"
        )

        if self.config.enable_personalization_eval and self._profile:
            reason += f"; profile_match={scores.get('profile_match', 0):.2f}"

        result = {
            "scores": scores,
            "comparison": comparison,
            "winner": winner,
            "reason": reason,
            "summary": reason,
            "dimension_breakdown": {
                "personalization": {
                    "style_match": scores["style_match"],
                    "preference_alignment": scores["preference_alignment"],
                    "personality_consistency": scores["personality_consistency"],
                },
                "quality": {
                    "quality_preservation": scores["quality_preservation"],
                },
            },
        }

        if self.config.enable_personalization_eval and self._profile:
            result["dimension_breakdown"]["personalization"]["profile_match"] = scores.get("profile_match", 0)
            result["profile"] = {
                "user_id": self.config.user_id,
                "dominant_style": self._profile.dominant_style,
                "dominant_domains": self._profile.dominant_domains,
            }

        return result


def make_recommendation(report: EvalReport, prev_report: Optional[EvalReport] = None) -> str:
    if report.scores.get("quality_preservation", 0.0) < 0.8:
        return "keep_previous"
    if prev_report is None:
        return "deploy"
    if report.scores.get("style_match", 0.0) > prev_report.scores.get("style_match", 0.0):
        return "deploy"
    if report.scores.get("style_preference_hit_rate", 0.0) > prev_report.scores.get("style_preference_hit_rate", 0.0):
        return "deploy"
    if report.scores.get("preference_alignment", 0.0) > prev_report.scores.get("preference_alignment", 0.0):
        return "deploy"
    if report.scores.get("personality_consistency", 0.0) > prev_report.scores.get("personality_consistency", 0.0):
        return "deploy"
    return "needs_more_data"
