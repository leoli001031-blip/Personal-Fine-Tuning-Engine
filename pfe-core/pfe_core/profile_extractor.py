"""Profile extractor for building user profiles from implicit signals.

This module extracts user preferences, style patterns, and interaction behaviors
from implicit signals and conversation history to build comprehensive user profiles.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from .collector.chat_collector import ChatCollector
from .models import ImplicitSignal, normalize_utc_datetime, parse_utc_datetime
from .storage import list_signals
from .user_profile import PreferenceScore, UserProfile, get_user_profile_store


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


# Domain keywords for classification
DOMAIN_KEYWORDS = {
    "programming": [
        "代码", "编程", "程序", "函数", "类", "变量", "bug", "debug", "python", "javascript",
        "code", "programming", "function", "class", "variable", "error", "compile", "runtime",
        "api", "database", "sql", "git", "github", "algorithm", "数据结构", "框架", "库"
    ],
    "writing": [
        "写作", "文章", "故事", "小说", "文案", "博客", "邮件", "报告", "摘要",
        "writing", "essay", "story", "novel", "copywriting", "blog", "email", "report",
        "summary", "draft", "edit", "revise", "grammar", "style", "tone"
    ],
    "learning": [
        "学习", "教程", "课程", "知识", "概念", "理解", "解释", "例子", "练习",
        "learning", "tutorial", "course", "knowledge", "concept", "understand", "explain",
        "example", "exercise", "study", "homework", "assignment", "exam", "quiz"
    ],
    "analysis": [
        "分析", "数据", "统计", "图表", "趋势", "预测", "模型", "评估", "比较",
        "analysis", "data", "statistics", "chart", "graph", "trend", "forecast", "model",
        "evaluate", "compare", "metric", "kpi", "performance", "insight"
    ],
    "creative": [
        "创意", "设计", "想法", "灵感", "头脑风暴", "概念", "原型", "迭代",
        "creative", "design", "idea", "inspiration", "brainstorm", "concept", "prototype",
        "iterate", "innovation", "art", "visual", "aesthetic", "ui", "ux"
    ],
    "business": [
        "商业", "业务", "产品", "市场", "客户", "销售", "策略", "计划", "预算",
        "business", "product", "market", "customer", "sales", "strategy", "plan", "budget",
        "revenue", "profit", "roi", "marketing", "operations", "management"
    ],
}

# Style indicators
STYLE_INDICATORS = {
    "formal": [
        "请", "您好", "谢谢", "抱歉", "能否", "是否可以", "麻烦", "感谢",
        "please", "thank you", "would you", "could you", "may I", "appreciate",
        "kindly", "regards", "sincerely", "formally"
    ],
    "casual": [
        "嘿", "嗨", "咋", "呗", "啦", "呢", "吧", "哈",
        "hey", "hi", "yeah", "nah", "cool", "awesome", "btw", "lol", "haha"
    ],
    "concise": [
        "简短", "简洁", "概要", "总结", "要点", "一句话", "简要",
        "brief", "short", "concise", "summary", "key points", "tl;dr", "in short"
    ],
    "detailed": [
        "详细", "具体", "深入", "全面", "完整", "详细说明", "展开", "解释清楚",
        "detailed", "specific", "in-depth", "comprehensive", "complete", "elaborate",
        "thorough", "explain clearly", "step by step"
    ],
    "technical": [
        "技术", "实现", "架构", "原理", "机制", "底层", "性能", "优化",
        "technical", "implementation", "architecture", "principle", "mechanism",
        "low-level", "performance", "optimization", "efficiency"
    ],
    "non_technical": [
        "简单", "通俗", "易懂", "白话", "不用术语", "普通人",
        "simple", "plain", "easy", "layman", "non-technical", "no jargon", "beginner"
    ],
}

# Interaction pattern indicators
INTERACTION_PATTERNS = {
    "likes_examples": [
        "例子", "示例", "举例", "比如", "例如", "sample", "example", "for instance",
        "such as", "like", "demonstrate", "show me"
    ],
    "prefers_direct": [
        "直接", "干脆", "不要废话", "直奔主题", "开门见山", "少说废话",
        "direct", "straight", "no fluff", "to the point", "get to it", "concise"
    ],
    "wants_reasoning": [
        "为什么", "原因", "理由", "逻辑", "思路", "怎么想到", "如何得出",
        "why", "reason", "logic", "thinking", "how did you", "rationale", "explain your"
    ],
    "prefers_code": [
        "代码", "实现", "写个", "程序", "脚本", "function", "code", "implement",
        "write a", "script", "program", "show code", "code example"
    ],
    "wants_alternatives": [
        "其他", "另外", "还有", "或者", "备选", "方案", "alternative", "other",
        "another", "or", "different", "option", "choice", "compare"
    ],
    "seeks_validation": [
        "对吗", "是否正确", "你觉得", "你怎么看", "确认", "检查",
        "right", "correct", "what do you think", "validate", "check", "verify"
    ],
}


@dataclass
class SignalAnalysisResult:
    """Result of analyzing a single signal."""
    signal_id: str
    timestamp: datetime
    domains: dict[str, float] = field(default_factory=dict)
    styles: dict[str, float] = field(default_factory=dict)
    patterns: dict[str, float] = field(default_factory=dict)
    confidence: float = 0.5


class ProfileExtractor:
    """Extracts user profile information from implicit signals."""

    def __init__(self, user_id: str, home: str | None = None):
        self.user_id = user_id
        self.home = home
        self.store = get_user_profile_store(home)
        self.profile = self.store.get_profile(user_id)

    def _classify_domain(self, text: str) -> dict[str, float]:
        """Classify text into domains based on keyword matching."""
        text_lower = text.lower()
        scores = {}

        for domain, keywords in DOMAIN_KEYWORDS.items():
            matches = sum(1 for kw in keywords if kw.lower() in text_lower)
            if matches > 0:
                # Normalize by keyword count and text length
                score = min(1.0, matches / len(keywords) * 5)  # Scale factor
                scores[domain] = score

        # Normalize to sum to 1.0
        total = sum(scores.values())
        if total > 0:
            scores = {k: v / total for k, v in scores.items()}

        return scores

    def _detect_style(self, text: str) -> dict[str, float]:
        """Detect communication style from text."""
        text_lower = text.lower()
        scores = {}

        for style, indicators in STYLE_INDICATORS.items():
            matches = sum(1 for ind in indicators if ind.lower() in text_lower)
            if matches > 0:
                scores[style] = min(1.0, matches * 0.3)

        # Special heuristics
        word_count = len(text.split())
        if word_count < 20:
            scores["concise"] = scores.get("concise", 0) + 0.3
        elif word_count > 100:
            scores["detailed"] = scores.get("detailed", 0) + 0.3

        # Normalize
        total = sum(scores.values())
        if total > 0:
            scores = {k: min(1.0, v / total * 2) for k, v in scores.items()}

        return scores

    def _detect_interaction_patterns(self, text: str) -> dict[str, float]:
        """Detect interaction patterns from text."""
        text_lower = text.lower()
        scores = {}

        for pattern, indicators in INTERACTION_PATTERNS.items():
            matches = sum(1 for ind in indicators if ind.lower() in text_lower)
            if matches > 0:
                scores[pattern] = min(1.0, matches * 0.4)

        return scores

    def _analyze_signal(self, signal: ImplicitSignal) -> SignalAnalysisResult:
        """Analyze a single implicit signal."""
        # Combine context and user action for analysis
        text = f"{signal.context} "
        if isinstance(signal.user_action, dict):
            if "next_message" in signal.user_action:
                text += str(signal.user_action["next_message"])
            elif "edited_text" in signal.user_action:
                text += str(signal.user_action["edited_text"])

        result = SignalAnalysisResult(
            signal_id=signal.signal_id,
            timestamp=signal.timestamp if isinstance(signal.timestamp, datetime) else _utc_now(),
            confidence=signal.confidence,
        )

        # Classify domains from user context
        result.domains = self._classify_domain(text)

        # Detect style from user message
        result.styles = self._detect_style(signal.context)

        # Detect interaction patterns
        result.patterns = self._detect_interaction_patterns(text)

        return result

    def _update_profile_from_analysis(self, analysis: SignalAnalysisResult) -> None:
        """Update profile from a single signal analysis."""
        # Update domain preferences
        for domain, score in analysis.domains.items():
            weighted_score = score * analysis.confidence
            self.profile.update_domain_preference(domain, weighted_score, analysis.signal_id)

        # Update style preferences
        for style, score in analysis.styles.items():
            weighted_score = score * analysis.confidence
            self.profile.update_style_preference(style, weighted_score, analysis.signal_id)

        # Update interaction patterns
        for pattern, score in analysis.patterns.items():
            weighted_score = score * analysis.confidence
            self.profile.update_interaction_pattern(pattern, weighted_score, analysis.signal_id)

    def extract_from_signals(
        self,
        signals: list[ImplicitSignal] | None = None,
        since: datetime | None = None,
        incremental: bool = True,
    ) -> dict[str, Any]:
        """Extract profile information from signals.

        Args:
            signals: List of signals to analyze (if None, loads from storage)
            since: Only analyze signals since this time
            incremental: If True, only analyze new signals since last analysis

        Returns:
            Summary of extraction results
        """
        if signals is None:
            # Load signals from storage
            signal_dicts = list_signals(user_id=self.user_id, home=self.home)
            signals = []
            for s in signal_dicts:
                try:
                    # Convert dict to ImplicitSignal
                    signal = ImplicitSignal(
                        signal_id=s.get("event_id", ""),
                        source_event_id=s.get("source_event_id", ""),
                        request_id=s.get("request_id", ""),
                        session_id=s.get("session_id", ""),
                        adapter_version=s.get("adapter_version"),
                        event_type=s.get("event_type", "accept"),
                        timestamp=parse_utc_datetime(s.get("timestamp"), default=_utc_now()),
                        context=s.get("context", ""),
                        model_output=s.get("model_output", ""),
                        user_action=s.get("user_action", {}),
                        signal_type=s.get("metadata", {}).get("signal_type", "accept"),
                        confidence=s.get("metadata", {}).get("confidence", 0.5),
                        extraction_rule=s.get("metadata", {}).get("extraction_rule", ""),
                    )
                    signals.append(signal)
                except Exception:
                    continue

        # Filter by time if needed
        if incremental and self.profile.last_analysis_at:
            since = self.profile.last_analysis_at

        if since:
            since = normalize_utc_datetime(since)
            signals = [
                s for s in signals
                if isinstance(s.timestamp, datetime) and normalize_utc_datetime(s.timestamp) > since
            ]

        # Analyze each signal
        analyses: list[SignalAnalysisResult] = []
        for signal in signals:
            try:
                analysis = self._analyze_signal(signal)
                analyses.append(analysis)
                self._update_profile_from_analysis(analysis)
            except Exception:
                continue

        # Update metadata
        self.profile.analysis_count += len(analyses)
        self.profile.last_analysis_at = _utc_now()

        # Compute dominant traits
        self.profile.compute_dominant_traits()

        # Save profile
        self.store.save_profile(self.user_id)

        return {
            "signals_analyzed": len(analyses),
            "domains_found": list(set().union(*[a.domains.keys() for a in analyses])),
            "styles_found": list(set().union(*[a.styles.keys() for a in analyses])),
            "patterns_found": list(set().union(*[a.patterns.keys() for a in analyses])),
            "profile_updated": self.profile.updated_at.isoformat(),
        }

    def generate_profile_summary(self, use_llm: bool = False) -> str:
        """Generate a human-readable summary of the user profile.

        Args:
            use_llm: If True, uses LLM to generate summary and merge structured fields

        Returns:
            Profile summary string
        """
        if use_llm:
            try:
                from .profile.llm_extractor import LLMProfileExtractor
                # Build pseudo-conversation from recent signals if available,
                # otherwise fall back to an empty list which triggers empty fallback.
                messages = self._build_conversation_from_profile()
                extractor = LLMProfileExtractor()
                result = extractor.extract_from_conversation(messages)
                meta = result.get("_extraction_meta", {})
                if meta.get("success"):
                    self._merge_llm_result(result)
                    self.profile.extracted_by = "llm"
                    self.profile.llm_extracted_at = _utc_now()
                    # Compute confidence as average of domain interest levels
                    interests = result.get("domain_interests", [])
                    if interests:
                        avg_conf = sum(d.get("level", 0.0) for d in interests) / len(interests)
                        self.profile.llm_extraction_confidence = round(min(1.0, avg_conf), 4)
                    else:
                        self.profile.llm_extraction_confidence = 0.5
                    # Build summary from LLM fields
                    summary_parts = []
                    identity = result.get("identity", {})
                    if identity.get("name") or identity.get("role"):
                        summary_parts.append(
                            f"用户{identity.get('name', '')}({identity.get('role', '')})"
                        )
                    top_domains = [
                        d["domain"] for d in result.get("domain_interests", [])
                        if d.get("level", 0) > 0.3
                    ][:2]
                    if top_domains:
                        summary_parts.append(f"关注领域：{'、'.join(top_domains)}")
                    styles = [
                        s["trait"] for s in result.get("style_indicators", [])
                    ][:2]
                    if styles:
                        summary_parts.append(f"风格特征：{'、'.join(styles)}")
                    summary = "；".join(summary_parts) if summary_parts else "用户画像数据不足"
                    self.profile.profile_summary = summary
                    self.store.save_profile(self.user_id)
                    return summary
            except Exception as exc:
                # LLM path failed, fall through to rule-based
                import logging
                logging.getLogger(__name__).warning("LLM profile summary failed: %s", exc)

        # Rule-based summary generation (default or fallback)
        parts = []

        # Style summary
        top_styles = self.profile.get_top_style_preferences(2)
        if top_styles:
            style_desc = "、".join([s[0] for s in top_styles])
            parts.append(f"偏好{style_desc}的沟通风格")

        # Domain summary
        top_domains = self.profile.get_top_domains(2)
        if top_domains:
            domain_desc = "、".join([d[0] for d in top_domains])
            parts.append(f"主要关注{domain_desc}领域")

        # Pattern summary
        top_patterns = self.profile.get_top_interaction_patterns(2)
        if top_patterns:
            pattern_names = {
                "likes_examples": "喜欢例子",
                "prefers_direct": "偏好直接回答",
                "wants_reasoning": "关注推理过程",
                "prefers_code": "偏好代码",
                "wants_alternatives": "喜欢多种方案",
                "seeks_validation": "寻求确认",
            }
            pattern_desc = "、".join([pattern_names.get(p[0], p[0]) for p in top_patterns])
            parts.append(f"交互特点：{pattern_desc}")

        summary = "；".join(parts) if parts else "用户画像数据不足"
        self.profile.profile_summary = summary
        self.store.save_profile(self.user_id)
        return summary

    def _build_conversation_from_profile(self) -> list[dict]:
        """Build a pseudo-conversation from stored signals for LLM extraction.

        Loads recent implicit signals and converts them into message dicts.
        """
        try:
            signal_dicts = list_signals(user_id=self.user_id, home=self.home)
        except Exception:
            return []
        messages: list[dict] = []
        for s in signal_dicts[-20:]:  # Last 20 signals
            ctx = s.get("context", "").strip()
            model_out = s.get("model_output", "").strip()
            if ctx:
                messages.append({"role": "user", "content": ctx})
            if model_out:
                messages.append({"role": "assistant", "content": model_out})
        return messages

    def _merge_llm_result(self, result: dict[str, Any]) -> None:
        """Merge LLM extraction result into the existing UserProfile."""
        self.profile.llm_identity = result.get("identity", {})
        self.profile.llm_stable_preferences = result.get("stable_preferences", [])
        self.profile.llm_response_preferences = result.get("response_preferences", [])
        self.profile.llm_style_indicators = result.get("style_indicators", [])
        self.profile.llm_domain_interests = result.get("domain_interests", [])
        self.profile.llm_temporal_notes = result.get("temporal_notes", [])

        # Also update rule-based domain preferences from LLM domain interests
        for item in result.get("domain_interests", []):
            domain = item.get("domain")
            level = item.get("level", 0.0)
            if domain:
                self.profile.update_domain_preference(domain, level)

        # Update style preferences from style indicators
        for item in result.get("style_indicators", []):
            trait = item.get("trait")
            if trait:
                self.profile.update_style_preference(trait, 0.7)

    def get_profile_for_training(self) -> dict[str, Any]:
        """Get profile formatted for training configuration."""
        return {
            "user_id": self.user_id,
            "dominant_style": self.profile.dominant_style,
            "dominant_domains": self.profile.dominant_domains,
            "style_preferences": {
                k: {"score": v.score, "confidence": v.confidence}
                for k, v in self.profile.style_preferences.items()
            },
            "domain_preferences": {
                k: {"score": v.score, "confidence": v.confidence}
                for k, v in self.profile.domain_preferences.items()
            },
            "interaction_patterns": {
                k: {"score": v.score, "confidence": v.confidence}
                for k, v in self.profile.interaction_patterns.items()
            },
            "preference_vector": self.profile.get_preference_vector(),
        }

    def get_profile_for_curation(self) -> dict[str, list[str]]:
        """Get profile formatted for data curation priorities."""
        return {
            "priority_domains": [d[0] for d in self.profile.get_top_domains(5)],
            "priority_styles": [s[0] for s in self.profile.get_top_style_preferences(3)],
            "priority_patterns": [p[0] for p in self.profile.get_top_interaction_patterns(3)],
        }


def extract_profile_for_user(
    user_id: str,
    signals: list[ImplicitSignal] | None = None,
    incremental: bool = True,
    home: str | None = None,
) -> dict[str, Any]:
    """Convenience function to extract profile for a user.

    Args:
        user_id: User identifier
        signals: Optional list of signals to analyze
        incremental: If True, only analyze new signals
        home: Optional PFE home directory

    Returns:
        Extraction summary
    """
    extractor = ProfileExtractor(user_id, home=home)
    result = extractor.extract_from_signals(signals=signals, incremental=incremental)
    summary = extractor.generate_profile_summary()
    result["profile_summary"] = summary
    return result


def get_user_profile(user_id: str, home: str | None = None) -> UserProfile:
    """Get user profile (convenience function)."""
    store = get_user_profile_store(home)
    return store.get_profile(user_id)


def analyze_user_conversation(
    user_id: str,
    conversation: list[dict[str, str]],
    home: str | None = None,
) -> dict[str, Any]:
    """Analyze a conversation and update user profile.

    Args:
        user_id: User identifier
        conversation: List of message dicts with 'role' and 'content'
        home: Optional PFE home directory

    Returns:
        Analysis results
    """
    extractor = ProfileExtractor(user_id, home=home)

    # Convert conversation to pseudo-signals for analysis
    pseudo_signals = []
    for i, msg in enumerate(conversation):
        if msg.get("role") == "user":
            # Create a pseudo-signal from user message
            pseudo_signal = ImplicitSignal(
                signal_id=f"conv_{i}",
                source_event_id=f"conv_{i}",
                request_id=f"conv_{i}",
                session_id="conversation",
                context=msg.get("content", ""),
                model_output=conversation[i + 1].get("content", "") if i + 1 < len(conversation) else "",
                user_action={"next_message": msg.get("content", "")},
                signal_type="accept",
                confidence=0.7,
            )
            pseudo_signals.append(pseudo_signal)

    result = extractor.extract_from_signals(signals=pseudo_signals, incremental=False)
    summary = extractor.generate_profile_summary()
    result["profile_summary"] = summary
    return result
