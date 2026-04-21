"""Personalized evaluation framework for PFE Phase 2.

Provides rule-based and LLM-based judges for assessing personalized model
quality, distinguishing generic quality improvements from personalization hits.
"""

from __future__ import annotations

from .personalized_evaluator import (
    EvalReport,
    JudgeBackend,
    JudgeResult,
    PersonalizedEvaluator,
    PersonalizedEvalConfig,
    PersonalizedVsGenericComparison,
)

__all__ = [
    "EvalReport",
    "JudgeBackend",
    "JudgeResult",
    "PersonalizedEvaluator",
    "PersonalizedEvalConfig",
    "PersonalizedVsGenericComparison",
]
