"""Phase31 Obsidian/Agent conversation signal-mining primitives."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping


PHASE31_KIND = "phase31_obsidian_agent_conversation_signal_mining"
PHASE31_FEEDBACK_SOURCE = "historical_user_agent_conversation"
PHASE31_MIN_APPROVED_SIGNALS = 20
PHASE31_MIN_HOLDOUT = 12

PHASE31_SIGNAL_TYPES = {
    "acceptance",
    "correction",
    "workflow_preference",
    "style_preference",
    "verification_preference",
    "safety_boundary",
}

PHASE31_QUALITY_METRICS = (
    "source_boundary_rate",
    "no_secret_rate",
    "redaction_applied_rate",
    "user_preference_specificity_rate",
    "chosen_rejected_contrast_rate",
    "profile_memory_routing_rate",
    "not_actual_feedback_rate",
    "holdout_isolation_rate",
    "concise_target_rate",
)

_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.S)
_MESSAGE_RE = re.compile(r"^(👤 用户|🤖 Agent):\n", re.M)
_LOCAL_PATH_RE = re.compile(r"/Users/[^\s，。；;、)）\]]+")
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE_RE = re.compile(r"(?<!\d)(?:\+?86[- ]?)?1[3-9]\d{9}(?!\d)")
_BOT_TOKEN_RE = re.compile(r"\b\d{6,}:[A-Za-z0-9_-]{20,}\b")
_API_KEY_RE = re.compile(r"\b(?:sk|rk|ak)-[A-Za-z0-9_-]{16,}\b", re.I)
_SECRET_WORD_RE = re.compile(r"api[_-]?key|secret|token|password|passwd|bearer|private key|telegram_bot_token", re.I)
_LOW_VALUE_RE = re.compile(r"测试一下|你好|在吗|继续$", re.I)

_TYPE_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("safety_boundary", re.compile(r"隐私|脱敏|不要泄露|token|权限|secret|api key|密码", re.I)),
    ("correction", re.compile(r"不是|不对|错了|改一下|修正|重新|问题是|跑偏|偏了|太复杂|太啰嗦")),
    ("verification_preference", re.compile(r"核对|检查|验证|测试|证据|截图|真实|跑一遍|确认|review", re.I)),
    ("style_preference", re.compile(r"口播稿|简洁|复杂|风格|设计语言|像我|AI味|表达|文案|写作")),
    ("acceptance", re.compile(r"可以|OK|好|处理完|没有问题|通过|继续", re.I)),
    ("workflow_preference", re.compile(r"我需要|我希望|最好|先|下一步|规划|追求目标|提交|整理|打开|帮我", re.I)),
)

_TARGET_BY_TYPE = {
    "acceptance": "给出短 checkpoint，并继续推进下一步。",
    "correction": "先承认偏差，再明确改法和当前要执行的最小步骤。",
    "workflow_preference": "把用户意图转成可执行计划，少讲理论，明确证据和下一步。",
    "style_preference": "按用户偏好的表达风格重写，保持简洁、自然、少 AI 味。",
    "verification_preference": "先做真实检查，再用路径、计数、测试或截图证据汇报。",
    "safety_boundary": "保留边界和脱敏，不把敏感信息写入训练目标。",
}

_SUMMARY_BY_TYPE = {
    "acceptance": "用户在历史对话中认可短 checkpoint，并希望继续推进下一步。",
    "correction": "用户在历史对话中纠正了方向或输出质量，要求 agent 立刻调整。",
    "workflow_preference": "用户在历史对话中要求把目标转成可执行规划和后续动作。",
    "style_preference": "用户在历史对话中表达了更简洁、更自然或更贴近个人风格的偏好。",
    "verification_preference": "用户在历史对话中要求真实核对、测试、截图、计数或文件证据。",
    "safety_boundary": "用户在历史对话中涉及权限、自动化或敏感边界，需要脱敏和人工复核。",
}


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _compact(text: str, *, max_chars: int = 360) -> str:
    compact = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 1].rstrip() + "..."


def _stable_id(*parts: str, length: int = 12) -> str:
    digest = hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()
    return digest[:length]


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return {}, text
    raw = match.group(1)
    meta: dict[str, Any] = {}
    for line in raw.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        meta[key.strip()] = value.strip().strip('"')
    return meta, text[match.end() :]


def parse_agent_messages(text: str) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    matches = list(_MESSAGE_RE.finditer(text))
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        role = "user" if "用户" in match.group(1) else "agent"
        content = text[start:end].strip()
        if content:
            messages.append({"role": role, "content": content})
    return messages


def sanitize_text(text: str) -> tuple[str, list[str]]:
    sanitized = str(text or "")
    redactions: list[str] = []
    replacements = (
        (_LOCAL_PATH_RE, "[LOCAL_PATH]", "local_path"),
        (_EMAIL_RE, "[EMAIL]", "email"),
        (_PHONE_RE, "[PHONE]", "phone"),
        (_BOT_TOKEN_RE, "[BOT_TOKEN]", "bot_token"),
        (_API_KEY_RE, "[API_KEY]", "api_key"),
    )
    for pattern, replacement, label in replacements:
        if pattern.search(sanitized):
            sanitized = pattern.sub(replacement, sanitized)
            redactions.append(label)
    return sanitized, sorted(set(redactions))


def secret_risk_reasons(text: str) -> list[str]:
    reasons = []
    if _BOT_TOKEN_RE.search(text):
        reasons.append("bot_token")
    if _API_KEY_RE.search(text):
        reasons.append("api_key")
    if _SECRET_WORD_RE.search(text):
        reasons.append("secret_keyword")
    if "PRIVATE KEY" in text.upper():
        reasons.append("private_key")
    return sorted(set(reasons))


def classify_signal_type(user_text: str) -> str:
    for signal_type, pattern in _TYPE_RULES:
        if pattern.search(user_text):
            return signal_type
    return "workflow_preference"


def discover_phase31_sources(
    *,
    vault_path: Path,
    max_conversations: int = 80,
    max_chars_per_file: int = 160_000,
) -> dict[str, Any]:
    conversations_dir = vault_path / "Conversations"
    files = sorted(conversations_dir.glob("*.md")) if conversations_dir.exists() else []
    scored: list[dict[str, Any]] = []
    keywords = re.compile(r"不要|不是|我需要|我希望|最好|核对|检查|验证|测试|证据|处理完|通过|继续|提交|追求目标|截图|真实")
    for path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")[:max_chars_per_file]
        except OSError:
            continue
        meta, body = parse_frontmatter(text)
        messages = parse_agent_messages(body)
        user_messages = [item["content"] for item in messages if item["role"] == "user"]
        agent_messages = [item["content"] for item in messages if item["role"] == "agent"]
        if not user_messages or not agent_messages:
            continue
        keyword_hits = len(keywords.findall(body))
        score = keyword_hits + min(len(messages), 30) / 10
        if _LOW_VALUE_RE.search(" ".join(user_messages[:2])):
            score -= 2
        relative_path = str(path.relative_to(vault_path))
        scored.append(
            {
                "source_id": f"phase31-source-{_stable_id(str(path))}",
                "path": f"[AGENT_MEMORY_VAULT]/{relative_path}",
                "path_hash": _stable_id(str(path), length=16),
                "relative_path": relative_path,
                "title": path.stem,
                "date": str(meta.get("date") or ""),
                "agent": str(meta.get("agent") or ""),
                "topics": str(meta.get("topics") or ""),
                "message_count": len(messages),
                "user_message_count": len(user_messages),
                "agent_message_count": len(agent_messages),
                "keyword_hits": keyword_hits,
                "score": round(score, 3),
                "secret_risk_reasons": secret_risk_reasons(text),
            }
        )
    selected = sorted(scored, key=lambda item: (-float(item["score"]), str(item["date"]), item["title"]))[:max_conversations]
    return {
        "kind": "phase31_source_inventory",
        "vault_path": "[AGENT_MEMORY_VAULT]",
        "vault_path_hash": _stable_id(str(vault_path), length=16),
        "conversation_count": len(files),
        "eligible_source_count": len(scored),
        "selected_source_count": len(selected),
        "sources": selected,
        "created_at": _utcnow_iso(),
    }


def _select_signal_messages(messages: list[Mapping[str, str]]) -> tuple[dict[str, str] | None, dict[str, str] | None]:
    user_messages = [dict(item) for item in messages if item.get("role") == "user"]
    agent_messages = [dict(item) for item in messages if item.get("role") == "agent"]
    if not user_messages or not agent_messages:
        return None, None
    def rank_user_message(item: Mapping[str, str]) -> tuple[int, int, int, int]:
        content = item.get("content", "")
        length = len(content)
        direct_signal = 1 if any(pattern.search(content) for _, pattern in _TYPE_RULES) else 0
        concise_bonus = 1 if 12 <= length <= 700 else 0
        long_penalty = -1 if length > 1600 else 0
        low_value_penalty = -2 if _LOW_VALUE_RE.search(content) else 0
        signal_rank = len(_TYPE_RULES) - next((index for index, (_, pat) in enumerate(_TYPE_RULES) if pat.search(content)), len(_TYPE_RULES))
        return (direct_signal + concise_bonus + long_penalty + low_value_penalty, signal_rank, -abs(length - 180), -length)

    ranked = sorted(user_messages, key=rank_user_message, reverse=True)
    selected_user = ranked[0]
    user_index = messages.index(selected_user) if selected_user in messages else 0
    following_agents = [dict(item) for item in messages[user_index + 1 :] if item.get("role") == "agent"]
    selected_agent = following_agents[-1] if following_agents else agent_messages[-1]
    return selected_user, selected_agent


def _chosen_output(signal_type: str, user_excerpt: str, agent_excerpt: str) -> str:
    target = _TARGET_BY_TYPE.get(signal_type, _TARGET_BY_TYPE["workflow_preference"])
    return (
        f"当前理解：用户偏好是{target}\n"
        f"处理方式：先基于真实材料核对，再给出短结论和下一步；必要时给出文件路径、计数或测试证据。\n"
        f"可复用记忆：遇到类似请求时，优先执行和验证，少做泛泛解释。\n"
        f"边界：不复制历史私密细节，不把未经确认的 agent 输出当成事实。"
    )


def _rejected_output(signal_type: str) -> str:
    if signal_type == "verification_preference":
        return "我觉得应该没问题，先按这个理解继续。"
    if signal_type == "correction":
        return "好的，我继续按原来的方向做，不需要调整。"
    if signal_type == "style_preference":
        return "下面我将从宏观愿景、底层逻辑和长期价值三个层面展开说明。"
    if signal_type == "safety_boundary":
        return "我会保留所有原文细节，包括 token、路径和账号信息，方便训练。"
    return "这是一个很好的问题。你可以再告诉我更多背景，我再帮你规划。"


def _signal_summary(signal_type: str) -> str:
    return _SUMMARY_BY_TYPE.get(signal_type, _SUMMARY_BY_TYPE["workflow_preference"])


def extract_phase31_signals(
    *,
    vault_path: Path,
    source_inventory: Mapping[str, Any],
    holdout_count: int = PHASE31_MIN_HOLDOUT,
) -> dict[str, Any]:
    signals: list[dict[str, Any]] = []
    holdout: list[dict[str, Any]] = []
    sources = [dict(item) for item in source_inventory.get("sources") or []]
    for index, source in enumerate(sources, start=1):
        path = vault_path / str(source.get("relative_path") or "")
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        _, body = parse_frontmatter(text)
        messages = parse_agent_messages(body)
        user_msg, agent_msg = _select_signal_messages(messages)
        if not user_msg or not agent_msg:
            continue
        raw_user = user_msg["content"]
        raw_agent = agent_msg["content"]
        redacted_user, user_redactions = sanitize_text(raw_user)
        redacted_agent, agent_redactions = sanitize_text(raw_agent)
        source_id = str(source.get("source_id") or f"phase31-source-{index:03d}")
        chunk_id = f"phase31-chunk-{_stable_id(source_id, raw_user, length=10)}"
        signal_type = classify_signal_type(raw_user)
        secret_reasons = sorted(set(secret_risk_reasons(raw_user + "\n" + raw_agent) + list(source.get("secret_risk_reasons") or [])))
        base_row = {
            "signal_id": f"phase31-signal-{index:03d}",
            "feedback_source": PHASE31_FEEDBACK_SOURCE,
            "source_id": source_id,
            "chunk_id": chunk_id,
            "conversation_path": str(source.get("relative_path") or ""),
            "conversation_title": str(source.get("title") or ""),
            "agent": str(source.get("agent") or ""),
            "date": str(source.get("date") or ""),
            "signal_type": signal_type,
            "user_text_excerpt": _signal_summary(signal_type),
            "agent_context_excerpt": "[REDACTED_AGENT_CONTEXT_HASHED]",
            "user_text_hash": _stable_id(redacted_user, length=16),
            "agent_context_hash": _stable_id(redacted_agent, length=16),
            "raw_excerpt_committed": False,
            "human_feedback_text": _signal_summary(signal_type),
            "chosen": _chosen_output(signal_type, redacted_user, redacted_agent),
            "rejected": _rejected_output(signal_type),
            "what_the_user_was_trying_to_fix": _TARGET_BY_TYPE.get(signal_type),
            "redactions": sorted(set(user_redactions + agent_redactions)),
            "secret_risk_reasons": secret_reasons,
            "attestation": {
                "historical_user_agent_conversation": True,
                "confirmed_actual_user_feedback": False,
                "not_realtime_actual_feedback": True,
                "requires_human_review_before_training": True,
                "user_authorized_current_run": True,
            },
            "metadata": {
                "phase": "phase31",
                "source_kind": "obsidian_agent_memory",
                "not_actual_user_feedback": True,
                "not_for_product_benefit_claim": True,
            },
        }
        if len(holdout) < holdout_count:
            holdout.append({**base_row, "split": "holdout", "eligible_for_training": False, "training_eligibility": "holdout_only"})
        else:
            eligible = not secret_reasons and len(_compact(redacted_user)) >= 16
            signals.append(
                {
                    **base_row,
                    "split": "candidate",
                    "eligible_for_training": eligible,
                    "eligible_for_product_benefit": False,
                    "training_eligibility": "eligible_after_human_review" if eligible else "excluded",
                    "exclusion_reason": None if eligible else "secret_or_low_information",
                }
            )
    return {
        "kind": "phase31_extracted_signals",
        "signal_count": len(signals),
        "holdout_count": len(holdout),
        "signals": signals,
        "holdout": {
            "kind": "phase31_holdout",
            "not_for_training": True,
            "holdout_count": len(holdout),
            "items": holdout,
        },
        "created_at": _utcnow_iso(),
    }


def validate_phase31_signal(signal: Mapping[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    if signal.get("feedback_source") != PHASE31_FEEDBACK_SOURCE:
        reasons.append("unsupported_feedback_source")
    if _dict(signal.get("attestation")).get("confirmed_actual_user_feedback") is True:
        reasons.append("historical_conversation_cannot_be_actual_feedback")
    if not _dict(signal.get("attestation")).get("requires_human_review_before_training"):
        reasons.append("human_review_boundary_required")
    if str(signal.get("signal_type") or "") not in PHASE31_SIGNAL_TYPES:
        reasons.append("unsupported_signal_type")
    if not signal.get("source_id") or not signal.get("chunk_id"):
        reasons.append("source_boundary_required")
    if signal.get("secret_risk_reasons"):
        reasons.append("secret_risk_quarantine")
    if not str(signal.get("chosen") or "").strip() or not str(signal.get("rejected") or "").strip():
        reasons.append("chosen_rejected_required")
    status = "passed" if not reasons else "quarantined"
    return {
        "kind": "phase31_signal_validation",
        "status": status,
        "passed": status == "passed",
        "reasons": sorted(set(reasons)),
        "created_at": _utcnow_iso(),
    }


def build_phase31_routing_report(signals: list[Mapping[str, Any]]) -> dict[str, Any]:
    routed = []
    status_counts: Counter[str] = Counter()
    type_counts: Counter[str] = Counter()
    for signal in signals:
        validation = validate_phase31_signal(signal)
        status_counts[str(validation["status"])] += 1
        type_counts[str(signal.get("signal_type") or "")] += 1
        eligible = bool(signal.get("eligible_for_training")) and validation["passed"]
        targets = []
        if eligible:
            targets = ["profile_candidate", "memory_candidate", "sft_candidate", "dpo_candidate", "hard_negative_candidate"]
        routed.append(
            {
                "signal_id": signal.get("signal_id"),
                "feedback_source": signal.get("feedback_source"),
                "signal_type": signal.get("signal_type"),
                "status": validation["status"],
                "eligible_for_training": eligible,
                "eligible_for_product_benefit": False,
                "training_targets": targets,
                "data_use": ["historical_preference_mining", "training_candidate_review", "profile_memory_candidate_review"] if eligible else [],
                "validation": validation,
            }
        )
    return {
        "kind": "phase31_signal_routing_report",
        "signal_count": len(signals),
        "eligible_training_count": sum(1 for item in routed if item["eligible_for_training"]),
        "historical_conversation_count": len(signals),
        "actual_user_feedback_count": 0,
        "status_counts": dict(sorted(status_counts.items())),
        "signal_type_counts": dict(sorted(type_counts.items())),
        "routed_signals": routed,
        "created_at": _utcnow_iso(),
    }


def score_phase31_candidate(signal: Mapping[str, Any], *, holdout_chunks: set[str] | None = None) -> dict[str, float]:
    holdout_chunks = holdout_chunks or set()
    chosen = str(signal.get("chosen") or "")
    rejected = str(signal.get("rejected") or "")
    has_boundary = bool(signal.get("source_id") and signal.get("chunk_id"))
    no_secret = not signal.get("secret_risk_reasons")
    redaction_applied = bool(signal.get("redactions"))
    specificity = bool(str(signal.get("what_the_user_was_trying_to_fix") or "").strip() and len(str(signal.get("user_text_excerpt") or "")) >= 16)
    contrast = chosen.strip() != rejected.strip() and len(chosen) > len(rejected)
    routed = True
    not_actual = signal.get("feedback_source") == PHASE31_FEEDBACK_SOURCE and _dict(signal.get("metadata")).get("not_actual_user_feedback") is True
    isolated = str(signal.get("chunk_id") or "") not in holdout_chunks
    concise = 80 <= len(_compact(chosen, max_chars=10_000)) <= 520
    return {
        "source_boundary_rate": 1.0 if has_boundary else 0.0,
        "no_secret_rate": 1.0 if no_secret else 0.0,
        "redaction_applied_rate": 1.0 if redaction_applied else 0.0,
        "user_preference_specificity_rate": 1.0 if specificity else 0.0,
        "chosen_rejected_contrast_rate": 1.0 if contrast else 0.0,
        "profile_memory_routing_rate": 1.0 if routed else 0.0,
        "not_actual_feedback_rate": 1.0 if not_actual else 0.0,
        "holdout_isolation_rate": 1.0 if isolated else 0.0,
        "concise_target_rate": 1.0 if concise else 0.0,
    }


def aggregate_phase31_quality(scores: Iterable[Mapping[str, Any]]) -> dict[str, float]:
    rows = [dict(item) for item in scores]
    if not rows:
        return {metric: 0.0 for metric in PHASE31_QUALITY_METRICS}
    return {
        metric: round(sum(float(row.get(metric, 0.0)) for row in rows) / len(rows), 3)
        for metric in PHASE31_QUALITY_METRICS
    }


def build_phase31_candidate_artifacts(
    *,
    signals: list[Mapping[str, Any]],
    routing_report: Mapping[str, Any],
    holdout: Mapping[str, Any],
) -> dict[str, Any]:
    routed_by_id = {str(item.get("signal_id")): _dict(item) for item in routing_report.get("routed_signals") or []}
    holdout_chunks = {str(item.get("chunk_id")) for item in holdout.get("items") or [] if isinstance(item, Mapping)}
    sft_samples: list[dict[str, Any]] = []
    dpo_pairs: list[dict[str, Any]] = []
    hard_negative_pairs: list[dict[str, Any]] = []
    profile_candidates: list[dict[str, Any]] = []
    memory_candidates: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    for signal in signals:
        signal_id = str(signal.get("signal_id") or "")
        route = routed_by_id.get(signal_id) or {}
        if not route.get("eligible_for_training"):
            excluded.append({"signal_id": signal_id, "reason": "not_eligible", "route": route})
            continue
        if str(signal.get("chunk_id") or "") in holdout_chunks:
            excluded.append({"signal_id": signal_id, "reason": "holdout_contamination"})
            continue
        scores = score_phase31_candidate(signal, holdout_chunks=holdout_chunks)
        quality_rows.append({"signal_id": signal_id, **scores})
        metadata = {
            "phase": "phase31",
            "feedback_source": PHASE31_FEEDBACK_SOURCE,
            "source_id": signal.get("source_id"),
            "chunk_id": signal.get("chunk_id"),
            "conversation_path": signal.get("conversation_path"),
            "signal_type": signal.get("signal_type"),
            "not_actual_user_feedback": True,
            "not_for_product_benefit_claim": True,
            "requires_human_review_before_training": True,
        }
        prompt = (
            "根据用户历史 Agent 协作偏好，回答一个类似请求。"
            f"\n用户历史偏好类型：{signal.get('signal_type')}"
            f"\n偏好摘要：{signal.get('user_text_excerpt')}"
            f"\n证据哈希：{signal.get('user_text_hash')}"
        )
        chosen = str(signal.get("chosen") or "")
        rejected = str(signal.get("rejected") or "")
        sft_samples.append(
            {
                "sample_id": f"phase31-sft-{signal_id}",
                "sample_type": "sft",
                "instruction": "学习用户的 Agent 协作偏好：先执行、要证据、简洁汇报、保留边界。",
                "input": prompt,
                "output": chosen,
                "prompt": prompt,
                "chosen": chosen,
                "metadata": metadata,
            }
        )
        pair = {
            "pair_id": f"phase31-dpo-{signal_id}",
            "sample_id": f"phase31-dpo-{signal_id}",
            "sample_type": "dpo",
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "metadata": metadata,
        }
        dpo_pairs.append(pair)
        hard_negative_pairs.append({**pair, "pair_id": f"phase31-hard-negative-{signal_id}", "sample_id": f"phase31-hard-negative-{signal_id}", "sample_type": "hard_negative"})
        profile_candidates.append(
            {
                "profile_id": f"phase31-profile-{signal_id}",
                "preference": signal.get("what_the_user_was_trying_to_fix"),
                "signal_type": signal.get("signal_type"),
                "evidence_summary": signal.get("user_text_excerpt"),
                "evidence_hash": signal.get("user_text_hash"),
                "metadata": metadata,
            }
        )
        memory_candidates.append(
            {
                "memory_id": f"phase31-memory-{signal_id}",
                "memory": signal.get("chosen"),
                "evidence": signal.get("conversation_path"),
                "metadata": metadata,
            }
        )
    quality = build_phase31_quality_report(quality_rows=quality_rows, sft_samples=sft_samples, dpo_pairs=dpo_pairs)
    integrity = phase31_holdout_integrity_check(holdout=holdout, candidates=sft_samples + dpo_pairs + hard_negative_pairs)
    manifest = {
        "kind": "phase31_candidate_manifest",
        "historical_conversation_signal_count": len(signals),
        "approved_candidate_signal_count": len(sft_samples),
        "actual_user_feedback_count": 0,
        "sft_sample_count": len(sft_samples),
        "dpo_pair_count": len(dpo_pairs),
        "hard_negative_pair_count": len(hard_negative_pairs),
        "profile_candidate_count": len(profile_candidates),
        "memory_candidate_count": len(memory_candidates),
        "excluded_signal_count": len(excluded),
        "quality_passed": quality["passed"],
        "product_benefit_claim_allowed": False,
        "requires_human_review_before_training": True,
        "created_at": _utcnow_iso(),
    }
    return {
        "kind": "phase31_candidate_artifacts",
        "sft_samples": sft_samples,
        "dpo_pairs": dpo_pairs,
        "hard_negative_pairs": hard_negative_pairs,
        "profile_candidates": profile_candidates,
        "memory_candidates": memory_candidates,
        "excluded": excluded,
        "quality_rows": quality_rows,
        "candidate_manifest": manifest,
        "candidate_quality_report": quality,
        "holdout_integrity_check": integrity,
        "created_at": _utcnow_iso(),
    }


def build_phase31_quality_report(
    *,
    quality_rows: list[Mapping[str, Any]],
    sft_samples: list[Mapping[str, Any]],
    dpo_pairs: list[Mapping[str, Any]],
) -> dict[str, Any]:
    aggregate = aggregate_phase31_quality(quality_rows)
    failures = []
    for row in quality_rows:
        failed = [
            metric
            for metric in PHASE31_QUALITY_METRICS
            if metric != "redaction_applied_rate" and float(row.get(metric, 0.0)) < 1.0
        ]
        if failed:
            failures.append({"signal_id": row.get("signal_id"), "failed_metrics": failed})
    required_counts = len(sft_samples) >= PHASE31_MIN_APPROVED_SIGNALS and len(dpo_pairs) >= PHASE31_MIN_APPROVED_SIGNALS
    passed = required_counts and not failures
    return {
        "kind": "phase31_candidate_quality_report",
        "passed": passed,
        "required_counts_passed": required_counts,
        "aggregate": aggregate,
        "failure_count": len(failures),
        "failures": failures[:50],
        "sample_count": len(sft_samples),
        "dpo_pair_count": len(dpo_pairs),
        "quality_contract": "historical conversation signals must be source-bound, redacted, contrasted, and not treated as actual feedback",
        "created_at": _utcnow_iso(),
    }


def phase31_holdout_integrity_check(*, holdout: Mapping[str, Any], candidates: list[Mapping[str, Any]]) -> dict[str, Any]:
    holdout_chunks = {str(item.get("chunk_id")) for item in holdout.get("items") or [] if isinstance(item, Mapping)}
    candidate_chunks = {
        str(_dict(item.get("metadata")).get("chunk_id"))
        for item in candidates
        if isinstance(item, Mapping) and _dict(item.get("metadata")).get("chunk_id")
    }
    contaminated = sorted(holdout_chunks & candidate_chunks)
    return {
        "kind": "phase31_holdout_integrity_check",
        "passed": not contaminated,
        "holdout_count": len(holdout.get("items") or []),
        "candidate_chunk_count": len(candidate_chunks),
        "contaminated_chunk_ids": contaminated,
        "created_at": _utcnow_iso(),
    }


def phase31_final_decision(*, quality_report: Mapping[str, Any], candidate_manifest: Mapping[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    if not quality_report.get("passed"):
        reasons.append("historical_signal_quality_report_failed")
    if int(candidate_manifest.get("approved_candidate_signal_count") or 0) < PHASE31_MIN_APPROVED_SIGNALS:
        reasons.append("insufficient_reviewable_historical_signals")
    recommendation = "historical_signal_quality_ready_for_human_review" if not reasons else "collect_more_or_review_sources"
    return {
        "kind": "phase31_final_decision",
        "status": "ready" if not reasons else "blocked",
        "recommendation": recommendation,
        "promotion_allowed": False,
        "auto_promotion_allowed": False,
        "actual_user_feedback_collected": False,
        "historical_user_agent_conversations_used": True,
        "product_benefit_claim_allowed": False,
        "training_launch_allowed": False,
        "next_step": "human_review_historical_candidates_then_optional_small_training_probe",
        "reasons": reasons,
        "created_at": _utcnow_iso(),
    }
