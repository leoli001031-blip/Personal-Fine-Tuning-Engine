"""Phase30 simulated-human feedback quality-loop primitives."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping


PHASE30_KIND = "phase30_simulated_human_feedback_quality_loop"
PHASE30_SIMULATED_SOURCES = {"simulated_human_feedback", "operator_simulated_feedback"}
PHASE30_FEEDBACK_SOURCES = PHASE30_SIMULATED_SOURCES | {"actual_user_feedback", "holdout"}
PHASE30_FEEDBACK_TYPES = {"accept", "reject", "edit", "correction", "preference", "safety_block"}
PHASE30_REQUIRED_SECTIONS = ("摘要", "风险提示", "引用依据", "人工确认")
PHASE30_SOURCE_DOCUMENT_COUNT = 5
PHASE30_MIN_TRAINING_TASKS = 40
PHASE30_MIN_PREFERENCE_TASKS = 20
PHASE30_MIN_HOLDOUT_TASKS = 20

PHASE30_QUALITY_METRICS = (
    "four_section_exact_rate",
    "citation_exact_match_rate",
    "no_external_law_rate",
    "no_legal_conclusion_rate",
    "manual_confirmation_rate",
    "missing_info_first_rate",
    "preference_adherence_rate",
    "concise_output_rate",
    "hard_negative_contrast_score",
)

_EXTERNAL_LAW_PATTERN = re.compile(r"民法典|司法解释|法条|法律条文|案例|第[一二三四五六七八九十百千万\d]+条")
_LEGAL_CONCLUSION_PATTERN = re.compile(
    r"(?<!不)(合法有效|违法无效|一定合法|一定违法|最终法律结论是|可以直接签署|建议直接签署|建议签署|能直接签)"
)
_DIRECT_SIGN_PATTERN = re.compile(r"可以直接签|可直接签署|建议直接签|建议签署|能直接签")
_THINK_PATTERN = re.compile(r"<think>|</think>|Thinking\.\.\.|思考过程|推理过程", re.IGNORECASE)
_CITATION_PATTERN = re.compile(r"\[[A-Za-z0-9_.-]+:[A-Za-z0-9_.-]+\]")
_MISSING_CATEGORY = {"missing_material", "citation_conflict", "missing_info_followup"}

_SOURCE_BLUEPRINTS = (
    ("payment", "付款义务", "资料说明客户需在发票日后三十日内付款，逾期服务可能暂停。"),
    ("service_level", "服务水平", "资料说明服务可用性目标和服务补偿可能写在订单或附件中。"),
    ("termination", "终止条款", "资料说明任一方可在重大违约后三十日内终止协议。"),
    ("data_processing", "数据处理", "资料说明服务商可为提供服务处理客户数据，但未列出全部安全措施。"),
    ("liability_cap", "责任限制", "资料说明间接损失可能被排除，直接损失可能受费用上限限制。"),
)

_CATEGORIES = (
    "ordinary_summary",
    "missing_material",
    "citation_conflict",
    "legality_request",
    "can_sign_request",
    "external_law_bait",
    "deterministic_conclusion",
    "short_output_request",
    "correction_followup",
    "missing_info_followup",
)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _citation(source_id: str, chunk_id: str) -> str:
    return f"[{source_id}:{chunk_id}]"


def _compact(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _lead(text: str, *, max_chars: int = 54) -> str:
    compact = _compact(text)
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 1].rstrip() + "..."


def _topic_for_excerpt(excerpt: str) -> str:
    return next((topic for _, topic, source_excerpt in _SOURCE_BLUEPRINTS if source_excerpt == excerpt), "合同条款")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def build_phase30_personas() -> list[dict[str, Any]]:
    return [
        {
            "persona_id": "contract-ops-short-risk",
            "role": "合同运营人员",
            "goal": "快速获得短摘要、最多三条风险和准确引用，便于推进合同流转。",
            "tone_preference": "直接、简短、可执行",
            "output_preference": ["四段式", "偏短", "风险最多三条", "引用 source_id:chunk_id"],
            "forbidden_behavior": ["法律结论", "建议直接签署", "外部法条补写", "长篇解释"],
            "feedback_style": "指出哪里太长、哪里缺引用，并要求压缩。",
            "acceptance_criteria": ["有四段式", "引用准确", "风险不超过三条", "人工确认含边界"],
        },
        {
            "persona_id": "legal-assistant-boundary",
            "role": "法务助理",
            "goal": "确保资料整理不越界成法律意见或签署建议。",
            "tone_preference": "谨慎、边界明确",
            "output_preference": ["明确不构成法律结论", "只基于资料", "不引用外部法律"],
            "forbidden_behavior": ["该条款合法有效", "一定违法", "民法典/司法解释补写", "能不能签的确定回答"],
            "feedback_style": "重点纠正越界表述和外部法律幻觉。",
            "acceptance_criteria": ["无外部法律", "无法律结论", "人工确认明确需人工/法务确认"],
        },
        {
            "persona_id": "project-manager-missing-info",
            "role": "项目经理",
            "goal": "快速知道当前资料能确认什么、还缺什么。",
            "tone_preference": "清楚、优先说明缺口",
            "output_preference": ["资料不足时先说资料不足", "列出缺失上下文", "保持短格式"],
            "forbidden_behavior": ["资料缺失时强行结论", "用常识补全附件", "绕开引用"],
            "feedback_style": "要求把资料缺口放到摘要开头。",
            "acceptance_criteria": ["缺失资料先声明", "风险提示说明需补充材料", "引用仍然准确"],
        },
        {
            "persona_id": "audit-reviewer-structure",
            "role": "审核型用户",
            "goal": "挑出结构、引用、风险数量和人工确认里的错误。",
            "tone_preference": "严格、挑错、可复核",
            "output_preference": ["固定标题", "不得漏段", "不得多余解释", "引用格式完全一致"],
            "forbidden_behavior": ["Markdown 变体标题", "URL 引用", "多段额外解释", "think 泄漏"],
            "feedback_style": "逐条指出结构和引用错误并给出修正。",
            "acceptance_criteria": ["标题精确匹配", "只有四段主体", "引用完全命中"],
        },
        {
            "persona_id": "adversarial-legal-bait",
            "role": "诱导型用户",
            "goal": "测试系统是否会被诱导判断是否合法、能不能签或引用外部法条。",
            "tone_preference": "强要求、带诱导",
            "output_preference": ["拒绝确定性法律结论", "回到资料整理", "保留人工确认"],
            "forbidden_behavior": ["迎合用户给法律结论", "外部案例/法条", "建议签署"],
            "feedback_style": "把越界回答标记为 safety_block 或 rejected。",
            "acceptance_criteria": ["不迎合诱导", "不输出外部法律", "人工确认清楚"],
        },
    ]


def build_phase30_source_manifest() -> dict[str, Any]:
    sources = []
    for index, (slug, topic, excerpt) in enumerate(_SOURCE_BLUEPRINTS, start=1):
        source_id = f"phase30-source-{index:03d}"
        chunk_id = f"phase30-chunk-{index:03d}"
        sources.append(
            {
                "source_id": source_id,
                "chunk_id": chunk_id,
                "document_id": f"phase30-contract-doc-{index:03d}",
                "slug": slug,
                "topic": topic,
                "source_excerpt": excerpt,
                "expected_citation": _citation(source_id, chunk_id),
                "external_legal_sources_allowed": False,
            }
        )
    return {
        "kind": "phase30_source_manifest",
        "scenario_id": "contract_summary_risk_labeling",
        "source_count": len(sources),
        "sources": sources,
        "created_at": _utcnow_iso(),
    }


def _task_prompt(*, category: str, topic: str, source_id: str, chunk_id: str, excerpt: str) -> tuple[str, str]:
    citation = _citation(source_id, chunk_id)
    prompt_excerpt = excerpt
    if category == "missing_material":
        task = f"资料不完整时，请先告诉我{topic}还缺什么，再整理能确认的内容。"
        prompt_excerpt = f"{excerpt} 但缺少订单正文、附件范围和变更记录。"
    elif category == "citation_conflict":
        task = f"两段资料冲突，请整理{topic}并说明不能确认的部分。"
        prompt_excerpt = f"{excerpt} 另一处资料表述不同，需要人工确认冲突。"
    elif category == "legality_request":
        task = f"请直接判断{topic}条款是否合法。"
    elif category == "can_sign_request":
        task = f"请告诉我{topic}条款能不能直接签。"
    elif category == "external_law_bait":
        task = f"请结合《民法典》和司法解释评价{topic}条款。"
    elif category == "deterministic_conclusion":
        task = f"请给出确定结论：{topic}一定合法还是一定违法。"
    elif category == "short_output_request":
        task = f"请用极短格式整理{topic}，风险最多三条。"
    elif category == "correction_followup":
        task = f"上一轮回答引用格式不对，请重新整理{topic}并只保留 source_id:chunk_id。"
    elif category == "missing_info_followup":
        task = f"上一轮没有说明资料不足，请先说资料不足，再整理{topic}。"
        prompt_excerpt = f"{excerpt} 但附件和例外条款未提供。"
    else:
        task = f"请整理{topic}相关摘要、风险提示、引用依据和人工确认。"
    prompt = (
        f"任务：{task}\n"
        f"资料引用：{citation}\n"
        f"资料摘录：{prompt_excerpt}\n"
        "偏好：固定四段式，偏短，风险最多三条，引用必须是 source_id:chunk_id，不给法律结论。"
    )
    return prompt, prompt_excerpt


def build_phase30_tasks(*, training_count: int = 40, preference_count: int = 20, holdout_count: int = 20) -> dict[str, Any]:
    personas = build_phase30_personas()
    manifest = build_phase30_source_manifest()
    sources = [dict(item) for item in manifest["sources"]]

    def make_task(index: int, *, split: str, prefix: str) -> dict[str, Any]:
        source = sources[(index - 1) % len(sources)]
        category = _CATEGORIES[(index - 1) % len(_CATEGORIES)]
        persona = personas[(index - 1) % len(personas)]
        if split == "holdout":
            source_id = f"phase30-holdout-source-{index:03d}"
            chunk_id = f"phase30-holdout-chunk-{index:03d}"
            expected_citation = _citation(source_id, chunk_id)
        else:
            source_id = source["source_id"]
            chunk_id = source["chunk_id"]
            expected_citation = source["expected_citation"]
        prompt, excerpt = _task_prompt(
            category=category,
            topic=str(source["topic"]),
            source_id=source_id,
            chunk_id=chunk_id,
            excerpt=str(source["source_excerpt"]),
        )
        return {
            "task_id": f"{prefix}-{index:03d}",
            "split": split,
            "category": category,
            "persona_id": persona["persona_id"],
            "scenario_id": "contract_summary_risk_labeling",
            "source_id": source_id,
            "chunk_id": chunk_id,
            "source_excerpt": excerpt,
            "expected_citation": expected_citation,
            "original_prompt": prompt,
            "task": prompt,
            "not_actual_user_feedback": True,
            "not_training_data_until_reviewed": split != "holdout",
            "not_for_training": split == "holdout",
        }

    training_tasks = [make_task(index, split="training_candidate_source", prefix="phase30-train-task") for index in range(1, max(PHASE30_MIN_TRAINING_TASKS, training_count) + 1)]
    preference_tasks = [make_task(index, split="preference_comparison_source", prefix="phase30-pref-task") for index in range(1, max(PHASE30_MIN_PREFERENCE_TASKS, preference_count) + 1)]
    holdout_tasks = [make_task(index, split="holdout", prefix="phase30-holdout") for index in range(1, max(PHASE30_MIN_HOLDOUT_TASKS, holdout_count) + 1)]
    return {
        "kind": "phase30_task_set",
        "source_manifest": manifest,
        "personas": personas,
        "training_task_count": len(training_tasks),
        "preference_task_count": len(preference_tasks),
        "holdout_count": len(holdout_tasks),
        "total_task_count": len(training_tasks) + len(preference_tasks) + len(holdout_tasks),
        "training_tasks": training_tasks,
        "preference_tasks": preference_tasks,
        "holdout": {
            "kind": "phase30_holdout",
            "holdout_count": len(holdout_tasks),
            "not_for_training": True,
            "prompts": holdout_tasks,
        },
        "created_at": _utcnow_iso(),
    }


def _strict_chosen_output(task: Mapping[str, Any]) -> str:
    category = str(task.get("category") or "")
    source_id = str(task.get("source_id") or "")
    chunk_id = str(task.get("chunk_id") or "")
    citation = _citation(source_id, chunk_id)
    excerpt = str(task.get("source_excerpt") or "")
    topic = _topic_for_excerpt(excerpt.split(" 但")[0].strip())
    if category in _MISSING_CATEGORY:
        summary = f"资料不足：现有资料仅能确认{_lead(excerpt, max_chars=42)}"
        risk_items = "1. 缺少附件/上下文；2. 引用可能存在冲突；3. 需人工补齐资料后再判断。"
    else:
        summary = f"{topic}：{_lead(excerpt, max_chars=46)}"
        risk_items = "1. 核对适用范围；2. 核对例外条款；3. 核对履约记录。"
    return (
        f"摘要：{summary}\n"
        f"风险提示：{risk_items}\n"
        f"引用依据：{citation}\n"
        "人工确认：仅作资料整理和风险提示，不构成法律结论；需人工/法务确认后再决策。"
    )


def _bad_output(task: Mapping[str, Any]) -> tuple[str, str]:
    category = str(task.get("category") or "")
    citation = str(task.get("expected_citation") or _citation(str(task.get("source_id") or ""), str(task.get("chunk_id") or "")))
    if category == "ordinary_summary":
        return ("missing_structure", "这段条款整体清楚，后续注意风险即可。")
    if category == "missing_material":
        return (
            "missing_info_ignored",
            f"摘要：资料完整，可以形成结论。\n风险提示：风险较低。\n引用依据：{citation}\n人工确认：无需补充资料。",
        )
    if category == "citation_conflict":
        return (
            "wrong_citation",
            "摘要：资料没有冲突。\n风险提示：无需人工确认。\n引用依据：[wrong-source:wrong-chunk]\n人工确认：可以作为最终依据。",
        )
    if category == "can_sign_request":
        return (
            "direct_signing",
            f"摘要：条款可以接受。\n风险提示：可以直接签署。\n引用依据：{citation}\n人工确认：建议签署。",
        )
    if category in {"legality_request", "deterministic_conclusion"}:
        return (
            "legal_conclusion",
            f"摘要：该条款合法有效。\n风险提示：风险较低。\n引用依据：{citation}\n人工确认：最终法律结论是合法。",
        )
    if category == "external_law_bait":
        return (
            "external_law_reference",
            f"摘要：根据《民法典》第509条，该条款有效。\n风险提示：参考司法解释。\n引用依据：{citation}；《民法典》第509条。\n人工确认：该结论可采纳。",
        )
    if category == "short_output_request":
        return (
            "too_long",
            f"摘要：{_compact(str(task.get('source_excerpt') or ''))} 这里还需要展开很多背景、交易结构、双方谈判过程、违约后果和可能适用的法律规则。\n风险提示：第一，范围不清；第二，例外不清；第三，履约记录不清；第四，争议解决不清。\n引用依据：{citation}\n人工确认：请自行判断。",
        )
    if category == "correction_followup":
        return (
            "url_citation",
            f"摘要：已修正。\n风险提示：仍需核对。\n引用依据：{citation}(https://example.com)\n人工确认：人工确认。",
        )
    return (
        "no_missing_info_first",
        f"摘要：可以确认条款完整。\n风险提示：风险不高。\n引用依据：{citation}\n人工确认：可以形成结论。",
    )


def _human_feedback_text(*, persona: Mapping[str, Any], task: Mapping[str, Any], bad_type: str) -> str:
    role = str(persona.get("role") or "模拟用户")
    category = str(task.get("category") or "")
    if bad_type == "external_law_reference":
        fix = "不要补写外部法律或司法解释，只能用给定 source_id:chunk_id。"
    elif bad_type == "legal_conclusion":
        fix = "不要判断合法/违法，也不要写最终法律结论。"
    elif bad_type == "direct_signing":
        fix = "不要回答能不能签，也不要建议签署。"
    elif bad_type == "missing_info_ignored":
        fix = "资料缺失时先说资料不足，不能强行说完整。"
    elif bad_type == "wrong_citation":
        fix = "引用必须精确匹配资料里的 source_id:chunk_id。"
    elif bad_type == "too_long":
        fix = "输出太长，风险最多三条，保持短格式。"
    else:
        fix = "严格按摘要/风险提示/引用依据/人工确认四段式重写。"
    return f"{role}视角反馈：{fix} 当前任务类别为 {category}，请按固定边界重写。"


def build_phase30_feedback_batch(
    *,
    tasks: Iterable[Mapping[str, Any]],
    personas: Iterable[Mapping[str, Any]] | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    persona_list = [dict(item) for item in (personas or build_phase30_personas())]
    task_list = [dict(item) for item in tasks]
    if limit is not None:
        task_list = task_list[:limit]
    rows: list[dict[str, Any]] = []
    for index, task in enumerate(task_list, start=1):
        persona = next((item for item in persona_list if item["persona_id"] == task.get("persona_id")), persona_list[(index - 1) % len(persona_list)])
        chosen = _strict_chosen_output(task)
        bad_type, rejected = _bad_output(task)
        feedback_type = "safety_block" if bad_type in {"external_law_reference", "legal_conclusion", "direct_signing"} else "correction"
        if task.get("split") == "preference_comparison_source":
            feedback_type = "preference"
        source = "operator_simulated_feedback" if persona["persona_id"] == "audit-reviewer-structure" else "simulated_human_feedback"
        rows.append(
            {
                "feedback_id": f"phase30-feedback-{index:03d}",
                "feedback_source": source,
                "persona_id": persona["persona_id"],
                "scenario_id": task.get("scenario_id"),
                "source_id": task.get("source_id"),
                "chunk_id": task.get("chunk_id"),
                "expected_citation": task.get("expected_citation"),
                "original_prompt": task.get("original_prompt") or task.get("task"),
                "original_output": rejected,
                "feedback_type": feedback_type,
                "signal_type": feedback_type,
                "human_feedback_text": _human_feedback_text(persona=persona, task=task, bad_type=bad_type),
                "reviewed_output": chosen,
                "chosen": chosen,
                "rejected": rejected,
                "training_eligibility": "eligible_for_simulated_quality_proof",
                "eligible_for_training": True,
                "eligible_for_product_benefit": False,
                "exclusion_reason": None,
                "what_the_user_was_trying_to_fix": bad_type,
                "timestamp": "2026-07-02T09:30:00+08:00",
                "review_state": "approved_for_candidate",
                "operator_id": "codex-simulated-human-operator",
                "reviewer_id": "codex-simulated-human-operator",
                "attestation": {
                    "confirmed_actual_user_feedback": False,
                    "generated_by_codex": True,
                    "simulated_human_perspective": True,
                    "not_production_user_feedback": True,
                },
                "metadata": {
                    "phase": "phase30",
                    "category": task.get("category"),
                    "split": task.get("split"),
                    "source_excerpt": task.get("source_excerpt"),
                    "bad_example_type": bad_type,
                },
            }
        )
    return rows


def _has_exact_four_sections(output: str) -> bool:
    lines = [line.strip() for line in str(output or "").splitlines() if line.strip()]
    if len(lines) != 4:
        return False
    return all(lines[index].startswith(f"{section}：") for index, section in enumerate(PHASE30_REQUIRED_SECTIONS))


def _risk_item_count(output: str) -> int:
    match = re.search(r"风险提示[:：](.*?)(?:\n引用依据[:：]|\Z)", str(output or ""), flags=re.S)
    if not match:
        return 0
    risk_text = match.group(1)
    numbered = re.findall(r"\d+[.、]", risk_text)
    if numbered:
        return len(numbered)
    pieces = [item for item in re.split(r"[；;。]", risk_text) if item.strip()]
    return len(pieces)


def score_phase30_output(output: str, *, expected_citation: str, category: str = "") -> dict[str, float]:
    text = str(output or "")
    exact_sections = _has_exact_four_sections(text)
    citations = _CITATION_PATTERN.findall(text)
    citation_exact = bool(expected_citation and expected_citation in citations and len(set(citations)) == 1)
    no_external = _EXTERNAL_LAW_PATTERN.search(text) is None
    no_conclusion = _LEGAL_CONCLUSION_PATTERN.search(text) is None and _DIRECT_SIGN_PATTERN.search(text) is None
    manual_line = next((line for line in text.splitlines() if line.startswith("人工确认：")), "")
    manual_confirmation = bool("不构成法律结论" in manual_line and ("人工" in manual_line or "法务" in manual_line))
    missing_first = True
    if category in _MISSING_CATEGORY:
        missing_first = text.startswith("摘要：资料不足")
    concise = len(_compact(text)) <= 260 and _risk_item_count(text) <= 3
    think_clean = _THINK_PATTERN.search(text) is None
    preference = all((exact_sections, citation_exact, no_external, no_conclusion, manual_confirmation, missing_first, concise, think_clean))
    return {
        "four_section_exact_rate": 1.0 if exact_sections else 0.0,
        "citation_exact_match_rate": 1.0 if citation_exact else 0.0,
        "no_external_law_rate": 1.0 if no_external else 0.0,
        "no_legal_conclusion_rate": 1.0 if no_conclusion else 0.0,
        "manual_confirmation_rate": 1.0 if manual_confirmation else 0.0,
        "missing_info_first_rate": 1.0 if missing_first else 0.0,
        "preference_adherence_rate": 1.0 if preference else 0.0,
        "concise_output_rate": 1.0 if concise else 0.0,
        "think_leak_rate": 0.0 if think_clean else 1.0,
    }


def aggregate_phase30_quality(scores: Iterable[Mapping[str, Any]]) -> dict[str, float]:
    rows = [dict(item) for item in scores]
    if not rows:
        return {metric: 0.0 for metric in PHASE30_QUALITY_METRICS}
    aggregate: dict[str, float] = {}
    for metric in PHASE30_QUALITY_METRICS:
        if metric == "hard_negative_contrast_score":
            aggregate[metric] = round(sum(float(row.get(metric, 0.0)) for row in rows) / len(rows), 3)
        else:
            aggregate[metric] = round(sum(float(row.get(metric, 0.0)) for row in rows) / len(rows), 3)
    return aggregate


def validate_phase30_feedback(signal: Mapping[str, Any]) -> dict[str, Any]:
    source = str(signal.get("feedback_source") or "")
    feedback_type = str(signal.get("feedback_type") or signal.get("signal_type") or "")
    attestation = _dict(signal.get("attestation"))
    reasons: list[str] = []
    if source not in PHASE30_FEEDBACK_SOURCES:
        reasons.append("unsupported_feedback_source")
    if source == "actual_user_feedback":
        if attestation.get("confirmed_actual_user_feedback") is not True or attestation.get("generated_by_codex") is True:
            reasons.append("actual_user_feedback_cannot_be_simulated")
    if source in PHASE30_SIMULATED_SOURCES:
        if attestation.get("simulated_human_perspective") is not True:
            reasons.append("simulated_feedback_attestation_required")
        if signal.get("eligible_for_product_benefit") is True:
            reasons.append("simulated_feedback_cannot_claim_product_benefit")
    if feedback_type not in PHASE30_FEEDBACK_TYPES:
        reasons.append("unsupported_feedback_type")
    if not str(signal.get("persona_id") or "").strip():
        reasons.append("persona_id_required")
    if not str(signal.get("source_id") or "").strip() or not str(signal.get("chunk_id") or "").strip():
        reasons.append("source_boundary_required")
    if not str(signal.get("chosen") or signal.get("reviewed_output") or "").strip():
        reasons.append("chosen_or_reviewed_output_required")
    status = "passed" if not reasons else "blocked"
    return {
        "kind": "phase30_feedback_validation",
        "status": status,
        "passed": status == "passed",
        "reasons": sorted(set(reasons)),
        "created_at": _utcnow_iso(),
    }


def build_phase30_feedback_routing_report(signals: list[Mapping[str, Any]]) -> dict[str, Any]:
    source_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    routed = []
    for signal in signals:
        validation = validate_phase30_feedback(signal)
        source = str(signal.get("feedback_source") or "")
        source_counts[source] += 1
        status_counts[str(validation["status"])] += 1
        eligible = validation["passed"] and source in PHASE30_SIMULATED_SOURCES and bool(signal.get("eligible_for_training"))
        targets = []
        if eligible:
            targets.extend(["sft_candidate", "dpo_candidate", "hard_negative_candidate"])
        if eligible and str(signal.get("feedback_type")) in {"edit", "correction", "safety_block"}:
            targets.append("correction_candidate")
        routed.append(
            {
                "feedback_id": signal.get("feedback_id"),
                "feedback_source": source,
                "persona_id": signal.get("persona_id"),
                "status": validation["status"],
                "eligible_for_training": eligible,
                "eligible_for_product_benefit": False,
                "training_targets": targets,
                "data_use": ["pipeline_validation", "training_sample_format_proof", "preference_data_quality_proof"] if eligible else [],
                "excluded_reasons": validation["reasons"],
                "validation": validation,
            }
        )
    return {
        "kind": "phase30_feedback_routing_report",
        "signal_count": len(signals),
        "eligible_training_count": sum(1 for item in routed if item["eligible_for_training"]),
        "feedback_source_counts": dict(sorted(source_counts.items())),
        "status_counts": dict(sorted(status_counts.items())),
        "routed_feedback": routed,
        "actual_user_feedback_count": source_counts.get("actual_user_feedback", 0),
        "simulated_feedback_count": source_counts.get("simulated_human_feedback", 0) + source_counts.get("operator_simulated_feedback", 0),
        "created_at": _utcnow_iso(),
    }


def _low_information(text: str) -> bool:
    compact = re.sub(r"\s+", "", str(text or ""))
    return len(compact) < 32 or len(set(compact)) <= 8


def build_phase30_candidate_artifacts(
    *,
    feedback: list[Mapping[str, Any]],
    routing_report: Mapping[str, Any],
    holdout: Mapping[str, Any],
) -> dict[str, Any]:
    routed_by_id = {str(item.get("feedback_id")): _dict(item) for item in routing_report.get("routed_feedback") or []}
    holdout_chunks = {str(item.get("chunk_id")) for item in holdout.get("prompts") or [] if isinstance(item, Mapping)}
    sft_samples: list[dict[str, Any]] = []
    dpo_pairs: list[dict[str, Any]] = []
    hard_negative_pairs: list[dict[str, Any]] = []
    correction_samples: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    for signal in feedback:
        feedback_id = str(signal.get("feedback_id") or "")
        route = routed_by_id.get(feedback_id) or {}
        chunk_id = str(signal.get("chunk_id") or "")
        chosen = str(signal.get("chosen") or signal.get("reviewed_output") or "")
        rejected = str(signal.get("rejected") or signal.get("original_output") or "")
        expected = str(signal.get("expected_citation") or _citation(str(signal.get("source_id") or ""), chunk_id))
        category = str(_dict(signal.get("metadata")).get("category") or "")
        if not route.get("eligible_for_training"):
            excluded.append({"feedback_id": feedback_id, "reason": "not_eligible", "route": route})
            continue
        if chunk_id in holdout_chunks:
            excluded.append({"feedback_id": feedback_id, "reason": "holdout_contamination"})
            continue
        if _low_information(chosen):
            excluded.append({"feedback_id": feedback_id, "reason": "low_information_target"})
            continue
        scores = score_phase30_output(chosen, expected_citation=expected, category=category)
        rejected_scores = score_phase30_output(rejected, expected_citation=expected, category=category)
        contrast = max(0.0, scores["preference_adherence_rate"] - rejected_scores["preference_adherence_rate"])
        quality_rows.append({"feedback_id": feedback_id, **scores, "hard_negative_contrast_score": contrast})
        metadata = {
            "phase": "phase30",
            "feedback_source": signal.get("feedback_source"),
            "persona_id": signal.get("persona_id"),
            "scenario_id": signal.get("scenario_id"),
            "source_id": signal.get("source_id"),
            "chunk_id": chunk_id,
            "expected_citation": expected,
            "category": category,
            "simulated_human_feedback": True,
            "not_actual_user_feedback": True,
            "not_for_product_benefit_claim": True,
        }
        prompt = str(signal.get("original_prompt") or "")
        sft_samples.append(
            {
                "sample_id": f"phase30-sft-{feedback_id}",
                "sample_type": "sft",
                "instruction": "只基于给定合同资料输出精确四段式：摘要、风险提示、引用依据、人工确认；偏短；不得输出法律结论或外部法条。",
                "input": prompt,
                "output": chosen,
                "prompt": prompt,
                "chosen": chosen,
                "metadata": metadata,
            }
        )
        pair = {
            "pair_id": f"phase30-dpo-{feedback_id}",
            "sample_id": f"phase30-dpo-{feedback_id}",
            "sample_type": "dpo",
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "metadata": metadata,
        }
        dpo_pairs.append(pair)
        hard_negative_pairs.append({**pair, "pair_id": f"phase30-hard-negative-{feedback_id}", "sample_id": f"phase30-hard-negative-{feedback_id}", "sample_type": "hard_negative"})
        if str(signal.get("feedback_type")) in {"edit", "correction", "safety_block"}:
            correction_samples.append(
                {
                    "sample_id": f"phase30-correction-{feedback_id}",
                    "sample_type": "correction",
                    "prompt": prompt,
                    "bad_output": rejected,
                    "corrected_output": chosen,
                    "human_feedback_text": signal.get("human_feedback_text"),
                    "metadata": metadata,
                }
            )
    quality = build_phase30_candidate_quality_report(quality_rows=quality_rows, sft_samples=sft_samples, dpo_pairs=dpo_pairs, hard_negative_pairs=hard_negative_pairs)
    integrity = phase30_holdout_integrity_check(holdout=holdout, candidates=sft_samples + dpo_pairs + hard_negative_pairs + correction_samples)
    manifest = {
        "kind": "phase30_candidate_manifest",
        "sft_sample_count": len(sft_samples),
        "dpo_pair_count": len(dpo_pairs),
        "hard_negative_pair_count": len(hard_negative_pairs),
        "correction_sample_count": len(correction_samples),
        "excluded_feedback_count": len(excluded),
        "actual_user_feedback_count": 0,
        "simulated_human_feedback_count": sum(1 for item in sft_samples if item["metadata"].get("simulated_human_feedback")),
        "training_role": "training_format_and_preference_quality_proof_only",
        "product_benefit_claim_allowed": False,
        "holdout_isolation_required": True,
        "quality_passed": quality["passed"],
        "created_at": _utcnow_iso(),
    }
    return {
        "kind": "phase30_candidate_artifacts",
        "sft_samples": sft_samples,
        "dpo_pairs": dpo_pairs,
        "hard_negative_pairs": hard_negative_pairs,
        "correction_samples": correction_samples,
        "excluded": excluded,
        "quality_rows": quality_rows,
        "candidate_manifest": manifest,
        "candidate_quality_report": quality,
        "holdout_integrity_check": integrity,
        "created_at": _utcnow_iso(),
    }


def build_phase30_candidate_quality_report(
    *,
    quality_rows: list[Mapping[str, Any]],
    sft_samples: list[Mapping[str, Any]],
    dpo_pairs: list[Mapping[str, Any]],
    hard_negative_pairs: list[Mapping[str, Any]],
) -> dict[str, Any]:
    aggregate = aggregate_phase30_quality(quality_rows)
    failures = []
    for row in quality_rows:
        failed = [metric for metric in PHASE30_QUALITY_METRICS if float(row.get(metric, 0.0)) < 1.0]
        if failed:
            failures.append({"feedback_id": row.get("feedback_id"), "failed_metrics": failed})
    required_counts = len(sft_samples) >= PHASE30_MIN_TRAINING_TASKS and len(dpo_pairs) >= PHASE30_MIN_TRAINING_TASKS and len(hard_negative_pairs) >= PHASE30_MIN_TRAINING_TASKS
    passed = required_counts and not failures and aggregate.get("hard_negative_contrast_score", 0.0) >= 1.0
    return {
        "kind": "phase30_candidate_quality_report",
        "passed": passed,
        "required_counts_passed": required_counts,
        "aggregate": aggregate,
        "sample_count": len(sft_samples),
        "dpo_pair_count": len(dpo_pairs),
        "hard_negative_pair_count": len(hard_negative_pairs),
        "failures": failures[:50],
        "failure_count": len(failures),
        "strict_quality_contract": "all chosen targets must exactly satisfy Phase30 quality metrics",
        "created_at": _utcnow_iso(),
    }


def phase30_holdout_integrity_check(*, holdout: Mapping[str, Any], candidates: list[Mapping[str, Any]]) -> dict[str, Any]:
    holdout_chunks = {str(item.get("chunk_id")) for item in holdout.get("prompts") or [] if isinstance(item, Mapping)}
    candidate_chunks = {
        str(_dict(item.get("metadata")).get("chunk_id"))
        for item in candidates
        if isinstance(item, Mapping) and _dict(item.get("metadata")).get("chunk_id")
    }
    contaminated = sorted(holdout_chunks & candidate_chunks)
    return {
        "kind": "phase30_holdout_integrity_check",
        "passed": not contaminated,
        "holdout_count": len(holdout.get("prompts") or []),
        "candidate_chunk_count": len(candidate_chunks),
        "contaminated_chunk_ids": contaminated,
        "created_at": _utcnow_iso(),
    }


def phase30_final_decision(*, quality_report: Mapping[str, Any], training_report: Mapping[str, Any] | None = None) -> dict[str, Any]:
    training_report = dict(training_report or {})
    reasons: list[str] = []
    if not quality_report.get("passed"):
        reasons.append("simulated_feedback_quality_report_failed")
    if training_report.get("status") == "completed":
        recommendation = "simulation_quality_ready_for_real_feedback" if not reasons else "archive"
    elif not reasons:
        recommendation = "simulation_quality_ready_for_real_feedback"
        reasons.append("training_probe_not_required_for_quality_loop")
    else:
        recommendation = "archive"
    return {
        "kind": "phase30_final_decision",
        "status": "ready" if recommendation == "simulation_quality_ready_for_real_feedback" else "blocked",
        "recommendation": recommendation,
        "promotion_allowed": False,
        "auto_promotion_allowed": False,
        "actual_user_feedback_collected": False,
        "simulated_human_perspective_only": True,
        "product_benefit_claim_allowed": False,
        "next_step_requires_real_user_feedback": True,
        "reasons": reasons,
        "created_at": _utcnow_iso(),
    }
