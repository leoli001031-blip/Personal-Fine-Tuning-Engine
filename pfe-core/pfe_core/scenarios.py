"""Scenario definitions for multi-adapter routing system.

This module defines the scenario configuration and built-in scenarios
for the Phase 2 Router system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ScenarioConfig:
    """Configuration for a single scenario.

    Attributes:
        scenario_id: Unique identifier for the scenario (e.g., "coding", "writing")
        name: Human-readable name for the scenario
        description: Detailed description of what this scenario covers
        adapter_version: The adapter version to use for this scenario
        trigger_keywords: Keywords that trigger this scenario
        examples: Example inputs that should match this scenario
        priority: Priority for this scenario (higher = more preferred)
        confidence_boost: Additional confidence boost when matched
    """

    scenario_id: str
    name: str
    description: str
    adapter_version: str
    trigger_keywords: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)
    example_phrases: list[str] = field(default_factory=list)
    priority: int = 0
    confidence_boost: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "scenario_id": self.scenario_id,
            "name": self.name,
            "description": self.description,
            "adapter_version": self.adapter_version,
            "trigger_keywords": self.trigger_keywords,
            "examples": self.examples,
            "example_phrases": self.example_phrases,
            "priority": self.priority,
            "confidence_boost": self.confidence_boost,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScenarioConfig":
        """Create from dictionary representation."""
        return cls(
            scenario_id=data["scenario_id"],
            name=data["name"],
            description=data.get("description", ""),
            adapter_version=data["adapter_version"],
            trigger_keywords=list(data.get("trigger_keywords", [])),
            examples=list(data.get("examples", [])),
            example_phrases=list(data.get("example_phrases", [])),
            priority=int(data.get("priority", 0)),
            confidence_boost=float(data.get("confidence_boost", 0.0)),
        )


# Built-in scenario definitions with comprehensive keyword coverage
BUILTIN_SCENARIOS: dict[str, ScenarioConfig] = {
    "coding": ScenarioConfig(
        scenario_id="coding",
        name="Programming & Code Generation",
        description="Code writing, debugging, refactoring, and technical implementation tasks",
        adapter_version="latest",
        trigger_keywords=[
            # Programming languages
            "python", "javascript", "typescript", "java", "c++", "c#", "go", "rust",
            "ruby", "php", "swift", "kotlin", "scala", "r", "matlab", "perl",
            # Code concepts
            "code", "coding", "programming", "function", "class", "method",
            "variable", "algorithm", "data structure", "api", "library",
            "framework", "debug", "debugging", "refactor", "optimization",
            "compile", "runtime", "syntax", "error", "exception", "bug",
            "git", "github", "repository", "commit", "pull request",
            # Web development
            "html", "css", "react", "vue", "angular", "node", "django", "flask",
            "database", "sql", "nosql", "mongodb", "postgresql", "mysql",
            # DevOps/Cloud
            "docker", "kubernetes", "aws", "azure", "gcp", "cloud", "serverless",
            "ci/cd", "pipeline", "deployment", "infrastructure",
        ],
        examples=[
            "Write a Python function to sort a list",
            "Debug this JavaScript error",
            "How do I implement a binary search tree?",
            "Refactor this code to use async/await",
            "Create a REST API endpoint in Flask",
            "Explain this regex pattern",
            "How to optimize this SQL query?",
        ],
        example_phrases=[
            "Write a Python function to sort a list",
            "Debug this JavaScript error",
            "How do I implement a binary search tree?",
            "Refactor this code to use async/await",
            "Create a REST API endpoint in Flask",
            "帮我写一个Python排序函数",
            "这段代码怎么调试",
            "如何实现二叉搜索树",
            "优化这个SQL查询",
        ],
        priority=10,
        confidence_boost=0.1,
    ),
    "writing": ScenarioConfig(
        scenario_id="writing",
        name="Writing & Content Creation",
        description="Creative writing, copywriting, editing, and content generation",
        adapter_version="latest",
        trigger_keywords=[
            # Writing types
            "write", "writing", "essay", "article", "blog", "post", "content",
            "copywriting", "creative writing", "story", "fiction", "novel",
            "poem", "poetry", "script", "screenplay", "dialogue",
            # Editing
            "edit", "editing", "proofread", "revise", "rewrite", "polish",
            "grammar", "spelling", "punctuation", "style", "tone",
            # Content types
            "email", "letter", "memo", "report", "proposal", "resume",
            "cover letter", "linkedin", "bio", "profile", "summary",
            # Marketing
            "marketing", "advertising", "campaign", "slogan", "headline",
            "product description", "landing page", "newsletter",
            # Academic
            "academic", "thesis", "dissertation", "research paper", "abstract",
            "citation", "bibliography", "literature review",
        ],
        examples=[
            "Help me write a cover letter",
            "Edit this paragraph for clarity",
            "Write a product description for my website",
            "Create an engaging blog post about AI",
            "Proofread my essay",
            "Write a professional email to my boss",
            "Help me craft a compelling story opening",
        ],
        example_phrases=[
            "Help me write a cover letter",
            "Edit this paragraph for clarity",
            "Write a product description for my website",
            "Create an engaging blog post about AI",
            "Proofread my essay",
            "帮我写一封求职信",
            "润色这段文字",
            "为我的网站写产品描述",
            "写一篇关于AI的博客",
        ],
        priority=10,
        confidence_boost=0.1,
    ),
    "learning": ScenarioConfig(
        scenario_id="learning",
        name="Learning & Education",
        description="Educational explanations, tutoring, and knowledge transfer",
        adapter_version="latest",
        trigger_keywords=[
            # Learning concepts
            "learn", "learning", "study", "studying", "education", "tutorial",
            "course", "lesson", "lecture", "homework", "assignment", "exam",
            "test", "quiz", "practice", "exercise", "problem", "solution",
            # Explanation requests
            "explain", "how does", "how do", "what is", "what are", "why is",
            "why does", "teach me", "help me understand", "i don't understand",
            "confused about", "clarify", "simplify", "break down",
            # Subjects
            "math", "mathematics", "physics", "chemistry", "biology", "science",
            "history", "geography", "economics", "finance", "accounting",
            "statistics", "calculus", "algebra", "geometry", "trigonometry",
            "language", "english", "chinese", "spanish", "french", "german",
            # Study methods
            "memorize", "remember", "technique", "strategy", "tip", "advice",
            "resource", "material", "textbook", "reference", "guide",
        ],
        examples=[
            "Explain quantum mechanics simply",
            "Help me understand calculus derivatives",
            "How do I solve this math problem?",
            "Teach me about the French Revolution",
            "What is machine learning?",
            "Study tips for my upcoming exam",
            "Explain photosynthesis step by step",
        ],
        example_phrases=[
            "Explain quantum mechanics simply",
            "Help me understand calculus derivatives",
            "How do I solve this math problem?",
            "Teach me about the French Revolution",
            "What is machine learning?",
            "简单解释量子力学",
            "帮我理解微积分导数",
            "怎么解这道数学题",
            "教我法国大革命",
            "机器学习是什么",
        ],
        priority=8,
        confidence_boost=0.05,
    ),
    "creative": ScenarioConfig(
        scenario_id="creative",
        name="Creative & Brainstorming",
        description="Ideas generation, creative projects, and artistic inspiration",
        adapter_version="latest",
        trigger_keywords=[
            # Creative concepts
            "creative", "creativity", "brainstorm", "brainstorming", "idea",
            "ideas", "inspiration", "imagine", "imagination", "design",
            "art", "artistic", "aesthetic", "visual", "music", "musical",
            # Generation
            "generate", "come up with", "think of", "suggest", "recommend",
            "ideas for", "options", "possibilities", "alternatives",
            # Projects
            "project", "diy", "craft", "handmade", "creation", "make",
            "build", "create", "produce", "develop",
            # Entertainment
            "game", "gaming", "movie", "film", "book", "novel", "character",
            "plot", "storyline", "setting", "worldbuilding", "fanfiction",
            # Art forms
            "drawing", "painting", "sketch", "illustration", "photography",
            "animation", "graphic design", "logo", "branding", "fashion",
            "interior design", "architecture", "sculpture", "pottery",
        ],
        examples=[
            "Brainstorm ideas for my novel",
            "Help me design a logo for my brand",
            "Generate creative writing prompts",
            "Ideas for a birthday party theme",
            "Suggest color palettes for my room",
            "Help me worldbuild for my fantasy story",
            "Creative gift ideas for my friend",
        ],
        example_phrases=[
            "Brainstorm ideas for my novel",
            "Help me design a logo for my brand",
            "Generate creative writing prompts",
            "Ideas for a birthday party theme",
            "Suggest color palettes for my room",
            "帮我想想小说创意",
            "帮我设计一个品牌logo",
            "生成一些创意写作提示",
            "生日派对主题建议",
        ],
        priority=8,
        confidence_boost=0.05,
    ),
    "chat": ScenarioConfig(
        scenario_id="chat",
        name="General Conversation",
        description="Casual chat, general questions, and everyday assistance",
        adapter_version="latest",
        trigger_keywords=[
            # Greetings
            "hello", "hi", "hey", "good morning", "good afternoon", "good evening",
            "how are you", "what's up", "how's it going", "nice to meet you",
            # Conversation
            "chat", "talk", "conversation", "discuss", "tell me about",
            "share your thoughts", "opinion", "thoughts on", "what do you think",
            # General help
            "help", "assist", "support", "advice", "suggestion", "recommendation",
            "question", "wondering", "curious", "interested in",
            # Personal
            "i feel", "i think", "i want", "i need", "should i", "would you",
            "can you", "could you", "please", "thank", "thanks",
            # Daily life
            "plan", "schedule", "organize", "reminder", "advice on", "tips for",
            "recommend", "suggest", "what should", "how should",
        ],
        examples=[
            "Hello, how are you today?",
            "Can we chat about life?",
            "I need some advice",
            "What do you think about this?",
            "Tell me about yourself",
            "Help me organize my day",
            "Just wanted to talk",
        ],
        example_phrases=[
            "Hello, how are you today?",
            "Can we chat about life?",
            "I need some advice",
            "What do you think about this?",
            "Tell me about yourself",
            "你好，今天过得怎么样",
            "能聊聊生活吗",
            "我需要一些建议",
            "你觉得这个怎么样",
        ],
        priority=5,
        confidence_boost=0.0,
    ),
}


def get_builtin_scenario(scenario_id: str) -> ScenarioConfig | None:
    """Get a built-in scenario by ID."""
    return BUILTIN_SCENARIOS.get(scenario_id)


def list_builtin_scenarios() -> list[ScenarioConfig]:
    """List all built-in scenarios."""
    return list(BUILTIN_SCENARIOS.values())


def create_custom_scenario(
    scenario_id: str,
    name: str,
    description: str,
    adapter_version: str,
    trigger_keywords: list[str] | None = None,
    examples: list[str] | None = None,
    example_phrases: list[str] | None = None,
    priority: int = 0,
    confidence_boost: float = 0.0,
) -> ScenarioConfig:
    """Create a custom scenario configuration."""
    return ScenarioConfig(
        scenario_id=scenario_id,
        name=name,
        description=description,
        adapter_version=adapter_version,
        trigger_keywords=trigger_keywords or [],
        examples=examples or [],
        example_phrases=example_phrases or [],
        priority=priority,
        confidence_boost=confidence_boost,
    )
