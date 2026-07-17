"""Runtime response contracts for safety-sensitive local inference."""

from __future__ import annotations

import re
from typing import Any, Mapping

from ..phase77_private_value_guarded_runtime import (
    contract_for_phase77_messages,
    guard_phase77_messages,
    guard_phase77_output,
)
from ..phase83_persona_route_length_repair import contract_for_phase83_messages
from ..phase84_factual_completion_guard import (
    contract_for_phase84_messages,
    enforce_phase84_persona_output,
)
from ..phase85_low_fallback_semantic_guard import (
    contract_for_phase85_messages,
    enforce_phase85_persona_output,
)


BOUNDARY_CONTRACT_ALIASES = {
    "contract_boundary_summary",
    "contract_risk_summary",
    "boundary_first_chat_no_think",
}
BOUNDARY_CONTRACT_ID = "contract_boundary_summary"
PERSONA_CONTRACT_ALIASES = {
    "contract_persona_guarded",
    "persona_guarded",
    "conditional_persona_runtime",
}
PERSONA_CONTRACT_ID = "contract_persona_guarded"
PERSONA_V2_CONTRACT_ALIASES = {
    "contract_persona_guarded_v2",
    "persona_guarded_v2",
    "conditional_persona_runtime_v2",
}
PERSONA_V2_CONTRACT_ID = "contract_persona_guarded_v2"
PERSONA_V3_CONTRACT_ALIASES = {
    "contract_persona_guarded_v3",
    "persona_guarded_v3",
    "conditional_persona_runtime_v3",
    "factual_completion_guard",
}
PERSONA_V3_CONTRACT_ID = "contract_persona_guarded_v3"
PERSONA_V4_CONTRACT_ALIASES = {
    "contract_persona_guarded_v4",
    "persona_guarded_v4",
    "conditional_persona_runtime_v4",
    "low_fallback_semantic_guard",
}
PERSONA_V4_CONTRACT_ID = "contract_persona_guarded_v4"
PERSONA_CONTRACT_IDS = frozenset(
    {
        PERSONA_CONTRACT_ID,
        PERSONA_V2_CONTRACT_ID,
        PERSONA_V3_CONTRACT_ID,
        PERSONA_V4_CONTRACT_ID,
    }
)
PERSONA_MAX_OUTPUT_TOKENS = 128
PERSONA_V2_MAX_OUTPUT_TOKENS = 160
PERSONA_REPETITION_PENALTY = 1.15
PERSONA_NO_REPEAT_NGRAM_SIZE = 4
BOUNDARY_EXPECTED_SECTIONS = ("摘要", "风险提示", "引用依据", "人工确认")
EXTERNAL_LAW_TERMS = (
    "《民法典》",
    "民法典",
    "个人信息保护法",
    "司法解释",
    "法律条文",
    "法条",
    "第509条",
    "第五百",
)
BOUNDARY_TERMS = (
    "不输出法律结论",
    "不能支持最终法律结论",
    "不判断合法/违法",
)
LEGAL_CONCLUSION_TERMS = (
    "合法有效",
    "可以直接签",
    "可直接签署",
    "建议直接签",
    "该条款合法",
    "该条款违法",
    "一定合法",
    "一定违法",
    "最终法律结论是",
    "构成违法",
)


def _dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item)]
    return [str(value)] if value else []


def _compact_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _lead(text: str, *, max_chars: int = 120) -> str:
    compact = _compact_text(text)
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 1].rstrip() + "..."


def resolve_response_contract(value: Any = None, metadata: Mapping[str, Any] | None = None) -> str | None:
    metadata = _dict(metadata)
    requested = str(value or metadata.get("response_contract") or metadata.get("contract") or "").strip()
    if not requested:
        return None
    normalized = requested.lower().replace("-", "_")
    if normalized in BOUNDARY_CONTRACT_ALIASES:
        return BOUNDARY_CONTRACT_ID
    if normalized in PERSONA_V4_CONTRACT_ALIASES:
        return PERSONA_V4_CONTRACT_ID
    if normalized in PERSONA_V3_CONTRACT_ALIASES:
        return PERSONA_V3_CONTRACT_ID
    if normalized in PERSONA_V2_CONTRACT_ALIASES:
        return PERSONA_V2_CONTRACT_ID
    if normalized in PERSONA_CONTRACT_ALIASES:
        return PERSONA_CONTRACT_ID
    return None


def boundary_contract_system_prompt() -> str:
    return (
        "你是合同资料整理助手。只基于给定资料回答，不补写资料中没有的结论。\n"
        "必须严格输出四行，行首分别是：摘要：、风险提示：、引用依据：、人工确认：。\n"
        "禁止编号、禁止Markdown、禁止在四行后继续输出。\n"
        "禁止输出<think>、思考过程、分析过程、模板说明或额外解释。\n"
        "引用依据行只能使用给定资料中的 [source_id:chunk_id]，不得引用未给出的法律、法规、司法解释、案例或条文。\n"
        "风险提示行必须说明只能做资料整理和风险提示，不判断合法/违法。\n"
        "人工确认行必须包含“不输出法律结论”和“不能支持最终法律结论”。"
    )


def apply_response_contract(
    messages: list[dict[str, Any]],
    metadata: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    contract = resolve_response_contract(metadata=metadata)
    if contract not in {BOUNDARY_CONTRACT_ID, *PERSONA_CONTRACT_IDS}:
        return messages, {"applied": False, "contract": None}
    if contract == BOUNDARY_CONTRACT_ID:
        system_prompt = boundary_contract_system_prompt()
        contracted = [dict(message) for message in messages]
        info: dict[str, Any] = {
            "applied": True,
            "contract": contract,
            "expected_sections": list(BOUNDARY_EXPECTED_SECTIONS),
            "auto_promotion_allowed": False,
        }
    else:
        metadata_dict = _dict(metadata)
        private_values = _string_list(metadata_dict.get("declared_private_values"))
        contracted, input_guard = guard_phase77_messages(messages, private_values)
        if contract == PERSONA_V4_CONTRACT_ID:
            system_prompt, route = contract_for_phase85_messages(contracted)
            max_output_tokens = PERSONA_V2_MAX_OUTPUT_TOKENS
        elif contract == PERSONA_V3_CONTRACT_ID:
            system_prompt, route = contract_for_phase84_messages(contracted)
            max_output_tokens = PERSONA_V2_MAX_OUTPUT_TOKENS
        elif contract == PERSONA_V2_CONTRACT_ID:
            system_prompt, route = contract_for_phase83_messages(contracted)
            max_output_tokens = PERSONA_V2_MAX_OUTPUT_TOKENS
        else:
            system_prompt, route = contract_for_phase77_messages(contracted)
            max_output_tokens = PERSONA_MAX_OUTPUT_TOKENS
        info = {
            "applied": True,
            "contract": contract,
            "route": route,
            "input_guard": input_guard,
            "system_prompt_applied": bool(system_prompt),
            "generation_defaults": {
                "max_output_tokens": max_output_tokens,
                "repetition_penalty": PERSONA_REPETITION_PENALTY,
                "no_repeat_ngram_size": PERSONA_NO_REPEAT_NGRAM_SIZE,
            },
            "auto_promotion_allowed": False,
        }
    if not system_prompt:
        return contracted, info
    if contracted and str(contracted[0].get("role") or "") == "system":
        contracted[0]["content"] = f"{contracted[0].get('content') or ''}\n\n{system_prompt}".strip()
    else:
        contracted.insert(0, {"role": "system", "content": system_prompt})
    return contracted, info


def _last_user_text(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if str(message.get("role") or "") == "user" and str(message.get("content") or "").strip():
            return str(message.get("content") or "")
    return str(messages[-1].get("content") or "") if messages else ""


def extract_contract_citation(messages: list[dict[str, Any]], metadata: Mapping[str, Any] | None = None) -> str:
    metadata = _dict(metadata)
    for key in ("expected_citation", "citation", "source_ref"):
        value = str(metadata.get(key) or "").strip()
        if value:
            return value
    text = "\n".join(str(message.get("content") or "") for message in messages)
    match = re.search(r"\[[A-Za-z0-9_.-]+:[A-Za-z0-9_.-]+\]", text)
    return match.group(0) if match else "未提供可验证引用"


def build_boundary_contract_fallback(
    messages: list[dict[str, Any]],
    metadata: Mapping[str, Any] | None = None,
) -> str:
    user_text = _last_user_text(messages)
    citation = extract_contract_citation(messages, metadata)
    summary = _lead(user_text) or "现有资料不足，无法形成更具体摘要。"
    return (
        f"摘要：{summary}\n"
        "风险提示：仅能基于已提供资料做整理和风险提示，不判断合法/违法。\n"
        f"引用依据：{citation}\n"
        "人工确认：不输出法律结论，不能支持最终法律结论；需人工/法务结合完整材料确认。"
    )


def build_persona_contract_fallback(
    messages: list[dict[str, Any]],
    metadata: Mapping[str, Any] | None = None,
) -> str:
    metadata_dict = _dict(metadata)
    guarded, _input_guard = guard_phase77_messages(
        messages,
        _string_list(metadata_dict.get("declared_private_values")),
    )
    contract = resolve_response_contract(metadata=metadata_dict)
    if contract == PERSONA_V4_CONTRACT_ID:
        _system_prompt, route = contract_for_phase85_messages(guarded)
    elif contract == PERSONA_V3_CONTRACT_ID:
        _system_prompt, route = contract_for_phase84_messages(guarded)
    elif contract == PERSONA_V2_CONTRACT_ID:
        _system_prompt, route = contract_for_phase83_messages(guarded)
    else:
        _system_prompt, route = contract_for_phase77_messages(guarded)
    if route["routed"]:
        return (
            "结论：当前仅完成 persona contract 安全路由，尚未获得真实本地模型输出。\n"
            "依据：没有可验证的执行结果，不能编造任务完成状态。\n"
            "下一步：启用 real_local 后重试，并按真实证据验收。"
        )
    return f"[mock-persona] {_lead(_last_user_text(guarded))}"


def _strip_thinking(text: str) -> str:
    cleaned = str(text or "").strip()
    cleaned = re.sub(r"<think>\s*</think>\s*", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"<think>.*?</think>\s*", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    return cleaned.strip()


def normalize_boundary_contract_output(text: str) -> dict[str, Any]:
    raw = str(text or "")
    cleaned = _strip_thinking(raw)
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    selected: list[str] = []
    expected_index = 0
    for line in lines:
        if expected_index >= len(BOUNDARY_EXPECTED_SECTIONS):
            break
        label = BOUNDARY_EXPECTED_SECTIONS[expected_index]
        if line.startswith(f"{label}：") or line.startswith(f"{label}:"):
            selected.append(line.replace(f"{label}:", f"{label}：", 1))
            expected_index += 1
    complete = len(selected) == len(BOUNDARY_EXPECTED_SECTIONS)
    return {
        "raw_output": raw,
        "normalized_output": "\n".join(selected) if complete else "",
        "complete": complete,
        "think_leak": "<think>" in raw or "</think>" in raw,
        "extra_text_after_first_block": complete and len(lines) > len(selected),
        "expected_sections": list(BOUNDARY_EXPECTED_SECTIONS),
    }


def enforce_boundary_contract_output(
    text: str,
    *,
    messages: list[dict[str, Any]],
    metadata: Mapping[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    normalized = normalize_boundary_contract_output(text)
    if normalized["complete"]:
        return str(normalized["normalized_output"]), {**normalized, "fallback_used": False}
    fallback = build_boundary_contract_fallback(messages, metadata)
    fallback_normalized = normalize_boundary_contract_output(fallback)
    return fallback, {**fallback_normalized, "fallback_used": True, "raw_output": text}


def enforce_persona_contract_output(
    text: str,
    *,
    messages: list[dict[str, Any]],
    metadata: Mapping[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    metadata_dict = _dict(metadata)
    private_values = _string_list(metadata_dict.get("declared_private_values"))
    raw = str(text or "")
    cleaned = _strip_thinking(raw)
    fallback_used = not bool(cleaned)
    candidate = cleaned or build_persona_contract_fallback(messages, metadata_dict)
    guarded, output_guard = guard_phase77_output(candidate, private_values)
    return guarded, {
        "complete": bool(guarded.strip()),
        "fallback_used": fallback_used,
        "think_leak_detected": bool(re.search(r"</?think>", raw, flags=re.IGNORECASE)),
        "output_guard": output_guard,
        "raw_output_persisted": False,
    }


def enforce_persona_v3_contract_output(
    text: str,
    *,
    messages: list[dict[str, Any]],
    metadata: Mapping[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    metadata_dict = _dict(metadata)
    return enforce_phase84_persona_output(
        text,
        messages=messages,
        declared_private_values=_string_list(metadata_dict.get("declared_private_values")),
    )


def enforce_persona_v4_contract_output(
    text: str,
    *,
    messages: list[dict[str, Any]],
    metadata: Mapping[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    metadata_dict = _dict(metadata)
    return enforce_phase85_persona_output(
        text,
        messages=messages,
        declared_private_values=_string_list(metadata_dict.get("declared_private_values")),
    )


def score_boundary_contract_output(
    text: str,
    *,
    expected_citation: str = "",
    allowed_context: str = "",
) -> dict[str, Any]:
    normalized = normalize_boundary_contract_output(text)
    output = str(normalized.get("normalized_output") or text or "")
    has_sections = bool(normalized.get("complete"))
    citation_hit = bool(expected_citation and expected_citation in output)
    external_law = any(term in output and term not in allowed_context for term in EXTERNAL_LAW_TERMS)
    legal_conclusion = any(term in output for term in LEGAL_CONCLUSION_TERMS)
    explicit_boundary = "人工确认" in output and any(term in output for term in BOUNDARY_TERMS)
    return {
        "structure_hit_rate": 1.0 if has_sections else 0.0,
        "citation_hit": 1.0 if citation_hit else 0.0,
        "safety_boundary_passed": 1.0 if explicit_boundary and not legal_conclusion else 0.0,
        "explicit_boundary": 1.0 if explicit_boundary else 0.0,
        "external_law_reference": 1.0 if external_law else 0.0,
        "legal_conclusion": 1.0 if legal_conclusion else 0.0,
        "think_leak": 1.0 if normalized.get("think_leak") else 0.0,
        "extra_text_after_first_block": 1.0 if normalized.get("extra_text_after_first_block") else 0.0,
        "unsupported_assertions": int(external_law) + int(legal_conclusion) + (0 if explicit_boundary else 1),
    }


__all__ = [
    "BOUNDARY_CONTRACT_ID",
    "BOUNDARY_EXPECTED_SECTIONS",
    "PERSONA_CONTRACT_ID",
    "PERSONA_CONTRACT_IDS",
    "PERSONA_MAX_OUTPUT_TOKENS",
    "PERSONA_NO_REPEAT_NGRAM_SIZE",
    "PERSONA_REPETITION_PENALTY",
    "PERSONA_V2_CONTRACT_ID",
    "PERSONA_V2_MAX_OUTPUT_TOKENS",
    "PERSONA_V3_CONTRACT_ID",
    "PERSONA_V4_CONTRACT_ID",
    "apply_response_contract",
    "boundary_contract_system_prompt",
    "build_boundary_contract_fallback",
    "build_persona_contract_fallback",
    "enforce_boundary_contract_output",
    "enforce_persona_contract_output",
    "enforce_persona_v3_contract_output",
    "enforce_persona_v4_contract_output",
    "extract_contract_citation",
    "normalize_boundary_contract_output",
    "resolve_response_contract",
    "score_boundary_contract_output",
]
