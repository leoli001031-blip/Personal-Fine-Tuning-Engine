"""Automatic evaluation pipeline for PFE."""

from __future__ import annotations

import re
from statistics import mean
from typing import Any, Optional, Sequence

from ..curator.datasets import sample_dataset_split
from ..curator.distillation import TrainingSample
from .judge import (
    EvalComparisonReport,
    EvalDetail,
    EvalReport,
    JudgeConfig,
    JudgeProtocol,
    LocalJudge,
    compare_eval_reports,
    make_recommendation,
)


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

_STYLE_PREFERENCE_HINTS: dict[str, tuple[str, ...]] = {
    "empathetic": ("温和", "共情", "理解", "支持", "鼓励", "关怀", "耐心", "empathetic", "warm"),
    "concise": ("简洁", "简短", "精炼", "直接", "结论", "概要", "concise", "brief"),
    "detailed": ("详细", "具体", "深入", "展开", "步骤", "全面", "detailed", "thorough"),
    "formal": ("正式", "礼貌", "客气", "正式一点", "formal"),
    "friendly": ("轻松", "随和", "自然", "口语", "朋友", "friendly", "casual"),
    "technical": ("技术", "专业", "实现", "原理", "架构", "代码", "technical"),
    "direct": ("直接", "明确", "不绕弯", "先说结论", "direct"),
}


def _normalize(text: Optional[str]) -> str:
    if not text:
        return ""
    return " ".join(text.lower().split())


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
        result: list[str] = []
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


def _style_preference_hit_rate(sample: TrainingSample, adapted_output: str) -> Optional[float]:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    preferences = _extract_style_preference_values(metadata)
    if not preferences:
        return None
    return 1.0 if any(_style_preference_hit(preference, adapted_output) for preference in preferences) else 0.0


class AutoEvaluator:
    """Run a deterministic local judge over holdout/test samples."""

    def __init__(self, config: Optional[JudgeConfig] = None, *, judge: Optional[JudgeProtocol] = None) -> None:
        self.config = config or JudgeConfig()
        self.judge = judge or LocalJudge(self.config)

    def evaluate(
        self,
        samples: Optional[Sequence[TrainingSample]] = None,
        *,
        test_prompts: Optional[Sequence[str]] = None,
        base_responses: Optional[Sequence[str]] = None,
        adapted_responses: Optional[Sequence[str]] = None,
        reference_responses: Optional[Sequence[str]] = None,
        base_model: str = "base",
        adapter_version: str = "latest",
        prev_report: Optional[EvalReport] = None,
    ) -> EvalReport:
        provided_samples = list(samples or [])
        requested_split_counts = self._split_counts(provided_samples)
        if provided_samples:
            prepared = self.judge.prepare_eval_set(provided_samples)
        elif test_prompts is not None:
            prepared = [
                TrainingSample(
                    sample_id=f"eval_{index}",
                    sample_type="sft",
                    instruction=prompt,
                    chosen="",
                    rejected=None,
                    score=1.0,
                    source="manual",
                    source_event_ids=[f"eval:{index}"],
                    metadata={"dataset_split": "test"},
                )
                for index, prompt in enumerate(test_prompts)
            ]
        else:
            prepared = self.judge.prepare_eval_set([])
        selected_split_counts = self._split_counts(prepared)
        eval_split_source = self._infer_eval_split_source(requested_split_counts, selected_split_counts, test_prompts is not None)

        prompts = list(test_prompts) if test_prompts is not None else [sample.instruction for sample in prepared]
        base_outputs = list(base_responses) if base_responses is not None else [sample.chosen for sample in prepared]
        adapted_outputs = list(adapted_responses) if adapted_responses is not None else [sample.chosen for sample in prepared]
        references = list(reference_responses) if reference_responses is not None else [
            sample.rejected if sample.sample_type == "dpo" else None for sample in prepared
        ]

        size = min(len(prepared), len(prompts), len(base_outputs), len(adapted_outputs), len(references))
        if size <= 0:
            raise ValueError("no evaluation samples or response pairs provided")

        details: list[EvalDetail] = []
        aggregates: dict[str, list[float]] = {
            "style_match": [],
            "preference_alignment": [],
            "quality_preservation": [],
            "personality_consistency": [],
        }
        style_preference_hit_sample_ids: list[str] = []

        for index in range(size):
            sample = prepared[index]
            comparison = self.judge.compare(base_outputs[index], adapted_outputs[index], references[index])
            scores = dict(comparison["scores"])
            style_preference_hit = _style_preference_hit_rate(sample, adapted_outputs[index])
            if style_preference_hit is not None:
                scores["style_preference_hit_rate"] = style_preference_hit
                aggregates.setdefault("style_preference_hit_rate", []).append(float(style_preference_hit))
                style_preference_hit_sample_ids.append(sample.sample_id)
            for key, value in scores.items():
                if key == "style_preference_hit_rate" and style_preference_hit is not None:
                    continue
                aggregates.setdefault(key, []).append(float(value))

            details.append(
                EvalDetail(
                    sample_id=sample.sample_id,
                    dataset_split=sample_dataset_split(sample) or "unknown",
                    prompt=prompts[index],
                    base_output=base_outputs[index],
                    adapted_output=adapted_outputs[index],
                    reference_output=references[index],
                    scores=scores,
                    winner=comparison["winner"],
                    reason=comparison["reason"],
                )
            )

        scores = {key: mean(values) if values else 0.0 for key, values in aggregates.items()}
        comparison = "improved" if mean(scores.values()) >= 0.7 else "neutral"
        if scores["quality_preservation"] < 0.8:
            comparison = "degraded"

        report = EvalReport(
            adapter_version=adapter_version,
            base_model=base_model,
            num_test_samples=size,
            scores=scores,
            comparison=comparison,
            recommendation="needs_more_data",
            details=details,
            judge_model=self.config.judge_model,
            judge_prompt_version=self.config.judge_prompt_version,
            metadata={
                "allowed_eval_splits": self.config.allowed_eval_splits,
                "forbid_teacher_test_overlap": self.config.forbid_teacher_test_overlap,
                "eval_scope": "holdout_only",
                "eval_split_source": eval_split_source,
                "requested_split_counts": requested_split_counts,
                "selected_split_counts": selected_split_counts,
                "prepared_sample_ids": [sample.sample_id for sample in prepared],
                "prepared_sample_splits": [sample_dataset_split(sample) or "unknown" for sample in prepared],
                "style_preference_hit_rate_sample_count": len(style_preference_hit_sample_ids),
                "style_preference_hit_rate_sample_ids": style_preference_hit_sample_ids,
                "train_samples_excluded": max(0, requested_split_counts.get("train", 0) - selected_split_counts.get("train", 0)),
            },
        )
        report.recommendation = make_recommendation(report, prev_report)
        return report

    @staticmethod
    def _report_adapter_version(report: Any) -> Optional[str]:
        if hasattr(report, "adapter_version"):
            version = getattr(report, "adapter_version", None)
            if version is not None:
                return str(version)
        if isinstance(report, dict):
            version = report.get("adapter_version")
            if version is not None:
                return str(version)
        if hasattr(report, "model_dump"):
            try:
                payload = dict(report.model_dump())
            except Exception:
                return None
            version = payload.get("adapter_version")
            if version is not None:
                return str(version)
        return None

    def compare_reports(
        self,
        left_report: EvalReport | dict[str, Any],
        right_report: EvalReport | dict[str, Any],
        *,
        left_label: Optional[str] = None,
        right_label: Optional[str] = None,
        delta_threshold: float = 0.05,
    ) -> EvalComparisonReport:
        return compare_eval_reports(
            left_report,
            right_report,
            left_label=left_label,
            right_label=right_label,
            delta_threshold=delta_threshold,
        )

    def compare_versions(
        self,
        left_report: EvalReport | dict[str, Any],
        right_report: EvalReport | dict[str, Any],
        *,
        delta_threshold: float = 0.05,
    ) -> EvalComparisonReport:
        return self.compare_reports(
            left_report,
            right_report,
            left_label=self._report_adapter_version(left_report),
            right_label=self._report_adapter_version(right_report),
            delta_threshold=delta_threshold,
        )

    @staticmethod
    def _split_counts(samples: Sequence[TrainingSample]) -> dict[str, int]:
        counts = {"train": 0, "val": 0, "test": 0, "unknown": 0}
        for sample in samples:
            split = sample_dataset_split(sample) or "unknown"
            counts[split if split in counts else "unknown"] += 1
        return counts

    def evaluate_preference_alignment(
        self,
        samples: Optional[Sequence[TrainingSample]] = None,
        *,
        test_prompts: Optional[Sequence[str]] = None,
        chosen_responses: Optional[Sequence[str]] = None,
        rejected_responses: Optional[Sequence[str]] = None,
        model_outputs: Optional[Sequence[str]] = None,
        base_model: str = "base",
        adapter_version: str = "latest",
    ) -> dict[str, Any]:
        """Evaluate preference alignment for DPO-trained models.

        This method specifically evaluates how well the model's outputs align
        with preferred (chosen) vs non-preferred (rejected) responses.

        Args:
            samples: Training samples with chosen/rejected pairs
            test_prompts: Test prompts
            chosen_responses: Preferred responses
            rejected_responses: Non-preferred responses
            model_outputs: Model's actual outputs to evaluate
            base_model: Base model identifier
            adapter_version: Adapter version being evaluated

        Returns:
            Dictionary with preference alignment metrics
        """
        provided_samples = list(samples or [])

        if provided_samples:
            # Filter to DPO samples only
            dpo_samples = [s for s in provided_samples if s.sample_type == "dpo"]
            if not dpo_samples:
                return {
                    "adapter_version": adapter_version,
                    "base_model": base_model,
                    "preference_alignment_score": 0.0,
                    "num_samples": 0,
                    "error": "No DPO samples provided",
                }
            prompts = [s.instruction for s in dpo_samples]
            chosen = [s.chosen for s in dpo_samples]
            rejected = [s.rejected for s in dpo_samples if s.rejected]
        elif test_prompts is not None:
            prompts = list(test_prompts)
            chosen = list(chosen_responses or [])
            rejected = list(rejected_responses or [])
        else:
            return {
                "adapter_version": adapter_version,
                "base_model": base_model,
                "preference_alignment_score": 0.0,
                "num_samples": 0,
                "error": "No evaluation data provided",
            }

        outputs = list(model_outputs or [])

        if not outputs or len(outputs) != len(prompts):
            return {
                "adapter_version": adapter_version,
                "base_model": base_model,
                "preference_alignment_score": 0.0,
                "num_samples": len(prompts),
                "error": "Model outputs missing or mismatched",
            }

        # Calculate preference alignment scores
        alignment_scores: list[float] = []
        for idx, output in enumerate(outputs):
            if idx >= len(chosen) or idx >= len(rejected):
                continue

            # Compare output similarity to chosen vs rejected
            chosen_sim = self._text_similarity(output, chosen[idx])
            rejected_sim = self._text_similarity(output, rejected[idx])

            # Alignment score: how much more similar to chosen than rejected
            if chosen_sim + rejected_sim > 0:
                alignment = chosen_sim / (chosen_sim + rejected_sim)
            else:
                alignment = 0.5

            alignment_scores.append(alignment)

        if not alignment_scores:
            return {
                "adapter_version": adapter_version,
                "base_model": base_model,
                "preference_alignment_score": 0.0,
                "num_samples": len(prompts),
                "error": "Could not calculate alignment scores",
            }

        avg_alignment = mean(alignment_scores)

        return {
            "adapter_version": adapter_version,
            "base_model": base_model,
            "preference_alignment_score": round(avg_alignment, 4),
            "num_samples": len(alignment_scores),
            "alignment_scores": alignment_scores,
            "interpretation": (
                "strong_preference" if avg_alignment >= 0.7
                else "moderate_preference" if avg_alignment >= 0.6
                else "weak_preference" if avg_alignment >= 0.5
                else "misaligned"
            ),
        }

    @staticmethod
    def _text_similarity(text1: str, text2: str) -> float:
        """Calculate simple text similarity using sequence matching.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity ratio between 0 and 1
        """
        from difflib import SequenceMatcher
        return SequenceMatcher(None, text1, text2).ratio()

    @staticmethod
    def _infer_eval_split_source(
        requested_split_counts: dict[str, int],
        selected_split_counts: dict[str, int],
        manual_prompts: bool,
    ) -> str:
        if manual_prompts:
            return "manual_test_prompts"
        if selected_split_counts.get("test", 0) and selected_split_counts.get("val", 0):
            return "val+test"
        if selected_split_counts.get("test", 0):
            return "test"
        if selected_split_counts.get("val", 0):
            return "val"
        if requested_split_counts.get("train", 0):
            return "provided_mixed_inputs"
        return "manual"
