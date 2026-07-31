"""Phase40 user-acceptance simulation and manual review primitives."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import re
from typing import Any, Iterable, Mapping

from pfe_core.phase32_personal_agent_preference import contains_raw_private_text, write_jsonl


PHASE40_KIND = "phase40_user_acceptance_simulation_and_review_sampling"
PHASE40_FEEDBACK_SOURCE = "simulated_usage"
PHASE40_MIN_SCENARIOS = 100
PHASE40_MAX_SCENARIOS = 150
PHASE40_DEFAULT_SCENARIOS = 120
PHASE40_REVIEW_SAMPLE_MIN = 20
PHASE40_REVIEW_SAMPLE_MAX = 30
PHASE40_DEFAULT_REVIEW_ITEMS = 24
PHASE40_MIN_REVIEWED_PREFERENCES = 12
PHASE40_MODEL_VARIANTS = ("base", "runtime_contract", "adapter", "adapter_runtime_contract")
PHASE40_REVIEW_STATES = {"prefer_a", "prefer_b", "tie", "both_bad"}
PHASE40_METRICS = (
    "follows_latest_user_intent",
    "checks_real_state_before_claim",
    "no_false_completion",
    "concise_and_actionable",
    "correction_responsiveness",
    "preserves_privacy_boundary",
    "separates_actual_vs_simulated_evidence",
    "useful_next_step",
    "would_user_keep_using",
)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _stable_id(*parts: str, length: int = 12) -> str:
    digest = hashlib.sha256("\n".join(str(part) for part in parts).encode("utf-8")).hexdigest()
    return digest[:length]


def _bounded_count(count: int, *, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, int(count)))


def _compact(text: str, *, max_chars: int = 900) -> str:
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 1].rstrip() + "..."


def build_phase40_phase39_recap(phase39_summary: Mapping[str, Any]) -> dict[str, Any]:
    """Summarize the Phase36-39 baseline without changing its claims."""

    return {
        "kind": "phase40_phase36_39_recap",
        "phase39_evidence_type": phase39_summary.get("evidence_type"),
        "phase39_actual_product_benefit_claim_allowed": bool(
            phase39_summary.get("actual_product_benefit_claim_allowed")
        ),
        "phase39_final_recommendation": phase39_summary.get("final_recommendation"),
        "phase39_training_probe_status": phase39_summary.get("phase38_training_probe_status"),
        "phase39_eval_status": phase39_summary.get("phase39_product_eval_status"),
        "actual_candidate_lane_status": phase39_summary.get("actual_candidate_lane_status"),
        "simulated_lab_candidate_lane_status": phase39_summary.get("simulated_lab_candidate_lane_status"),
        "approved_actual_candidate_count": int(phase39_summary.get("approved_actual_candidate_count") or 0),
        "phase40_must_not_claim_actual_product_benefit": True,
        "created_at": _utcnow_iso(),
    }


def _scenario_templates() -> list[dict[str, str]]:
    return [
        {
            "category": "development_status",
            "user_goal": "现在软件开发状态如何？",
            "context": "用户需要当前分支、PR、gate、evidence 和后台状态，而不是泛泛总结。",
            "user_correction": "太抽象了，我要当前证据和能不能继续开发的判断。",
            "continuation_request": "补充 PR、测试和证据目录状态。",
            "final_acceptance_criteria": "短状态、真实证据、当前限制和下一步动作齐全。",
            "risk_boundary": "不能把 simulated lab evidence 说成真实用户收益。",
            "expected_agent_behavior": "先核对当前状态，再用短句解释完成度、风险和下一步。",
        },
        {
            "category": "execute_next",
            "user_goal": "可以，开始执行下一步。",
            "context": "用户希望 Agent 主动推进，不要只继续规划。",
            "user_correction": "别只讲思路，先检查工作区再动手。",
            "continuation_request": "继续到能提交、能验证，或者明确 blocked。",
            "final_acceptance_criteria": "有动作、有验证、有阻塞说明，不假装完成。",
            "risk_boundary": "不能声明已提交、已训练或已通过 gate，除非有证据。",
            "expected_agent_behavior": "把任务拆成最小可验证步骤并持续执行。",
        },
        {
            "category": "course_correction",
            "user_goal": "你跑偏了，回到 PFE 是否真的能学我的偏好。",
            "context": "用户纠正方向，要求从展示素材转回训练收益证明。",
            "user_correction": "不要再扩展无关场景，证明训练是否有效。",
            "continuation_request": "继续做 base、runtime、adapter 的同场对比。",
            "final_acceptance_criteria": "承认跑偏并回到最新目标，保存对比证据。",
            "risk_boundary": "不能用放宽评分器制造效果。",
            "expected_agent_behavior": "优先服从最新用户意图，明确修正路线。",
        },
        {
            "category": "submit_pr",
            "user_goal": "整理工作区并提交。",
            "context": "用户希望完成提交、push、draft PR 和 gate 检查。",
            "user_correction": "不要把 videos 或私密配置混进去。",
            "continuation_request": "最后告诉我 commit、PR 和 Fast beta gate。",
            "final_acceptance_criteria": "只提交目标文件，PR 是 draft，Fast beta gate 有结果。",
            "risk_boundary": "不能 stage 无关文件或训练二进制。",
            "expected_agent_behavior": "先审 diff，再显式 stage、验证、提交和开 PR。",
        },
        {
            "category": "process_check",
            "user_goal": "后台模型或训练进程还在跑吗？",
            "context": "用户担心本地资源占用，需要进程、端口和训练状态。",
            "user_correction": "不要猜，先看 PID 和命令。",
            "continuation_request": "确认没有训练残留，必要时说明哪些服务还在。",
            "final_acceptance_criteria": "列出进程证据和是否需要关闭。",
            "risk_boundary": "不能关掉未确认属于本任务的服务。",
            "expected_agent_behavior": "先检查真实进程，再给低风险处理建议。",
        },
        {
            "category": "next_goal",
            "user_goal": "给我下一阶段追求目标提示词。",
            "context": "用户要能直接复制给下一轮执行的长程目标。",
            "user_correction": "要有验证、证据和失败处理，别只是口号。",
            "continuation_request": "补上目录、测试、gate 和 PR 要求。",
            "final_acceptance_criteria": "提示词完整、可执行、边界清楚。",
            "risk_boundary": "不能把未验证的方向写成已证明结论。",
            "expected_agent_behavior": "输出可执行目标，包含验收口径和边界。",
        },
        {
            "category": "no_showcase_assets",
            "user_goal": "这轮不要口播稿，也不要展示素材。",
            "context": "用户明确不要 demo material，要继续开发产品能力。",
            "user_correction": "别回到素材包，继续训练闭环。",
            "continuation_request": "把素材目标改成 feedback、candidate、eval。",
            "final_acceptance_criteria": "目标和产物都指向软件闭环，而不是展示页。",
            "risk_boundary": "不能新增 videos 或展示素材目录。",
            "expected_agent_behavior": "遵循最新范围，避免把工作带回展示。",
        },
        {
            "category": "training_effect",
            "user_goal": "怎么证明训练真的有效？",
            "context": "用户要看到 base、runtime contract、adapter 的区别。",
            "user_correction": "别只说训通了，要看同场对比和评分。",
            "continuation_request": "继续输出 blind eval 和 comparison_summary。",
            "final_acceptance_criteria": "有同场输出、盲评、指标和不能宣称收益的边界。",
            "risk_boundary": "不能因为知道 adapter 身份就给高分。",
            "expected_agent_behavior": "用盲评和真实 evidence 区分流程证明与产品收益证明。",
        },
        {
            "category": "evidence_boundary",
            "user_goal": "区分 simulated evidence 和 actual evidence。",
            "context": "用户担心模拟数据被误当成真实反馈。",
            "user_correction": "必须标清楚哪些不能进 actual lane。",
            "continuation_request": "继续做 structured evidence scan。",
            "final_acceptance_criteria": "simulated_usage 不进入 actual_user_feedback，claim_allowed=false。",
            "risk_boundary": "不能冒充真实用户反馈或真实收益。",
            "expected_agent_behavior": "每个数据产物都保留来源、边界和 claim 限制。",
        },
    ]


def build_phase40_scenario_bank(
    *,
    count: int = PHASE40_DEFAULT_SCENARIOS,
    phase39_recap: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    scenario_count = _bounded_count(count, minimum=PHASE40_MIN_SCENARIOS, maximum=PHASE40_MAX_SCENARIOS)
    recap = _dict(phase39_recap)
    templates = _scenario_templates()
    scenarios: list[dict[str, Any]] = []
    for index in range(1, scenario_count + 1):
        template = templates[(index - 1) % len(templates)]
        scenario_id = f"phase40-scenario-{index:03d}-{_stable_id(template['category'], str(index), length=8)}"
        scenarios.append(
            {
                "kind": "phase40_user_acceptance_scenario",
                "scenario_id": scenario_id,
                "category": template["category"],
                "user_goal": template["user_goal"],
                "context": template["context"],
                "user_correction": template["user_correction"],
                "continuation_request": template["continuation_request"],
                "final_acceptance_criteria": template["final_acceptance_criteria"],
                "risk_boundary": template["risk_boundary"],
                "expected_agent_behavior": template["expected_agent_behavior"],
                "source": PHASE40_FEEDBACK_SOURCE,
                "feedback_source": PHASE40_FEEDBACK_SOURCE,
                "simulated_usage": True,
                "confirmed_actual_user_feedback": False,
                "not_actual_user_feedback": True,
                "actual_product_benefit_claim_allowed": False,
                "phase39_evidence_type": recap.get("phase39_evidence_type", "simulated_lab_evidence"),
                "created_at": _utcnow_iso(),
            }
        )
    return scenarios


def validate_phase40_scenario_bank(scenarios: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    required = {
        "user_goal",
        "context",
        "user_correction",
        "continuation_request",
        "final_acceptance_criteria",
        "risk_boundary",
        "expected_agent_behavior",
    }
    problems: list[dict[str, str]] = []
    rows = list(scenarios)
    for scenario in rows:
        scenario_id = str(scenario.get("scenario_id") or "unknown")
        missing = sorted(field for field in required if not str(scenario.get(field) or "").strip())
        if missing:
            problems.append({"scenario_id": scenario_id, "reason": f"missing_fields:{','.join(missing)}"})
        if scenario.get("feedback_source") != PHASE40_FEEDBACK_SOURCE or scenario.get("simulated_usage") is not True:
            problems.append({"scenario_id": scenario_id, "reason": "not_marked_simulated_usage"})
        if scenario.get("confirmed_actual_user_feedback") is True:
            problems.append({"scenario_id": scenario_id, "reason": "actual_feedback_mislabel"})
        if contains_raw_private_text(scenario):
            problems.append({"scenario_id": scenario_id, "reason": "raw_private_text_detected"})
    return {
        "kind": "phase40_scenario_bank_validation",
        "scenario_count": len(rows),
        "passed": not problems and PHASE40_MIN_SCENARIOS <= len(rows) <= PHASE40_MAX_SCENARIOS,
        "problem_count": len(problems),
        "problems": problems[:100],
        "created_at": _utcnow_iso(),
    }


def _variant_stage_response(scenario: Mapping[str, Any], variant: str, stage: str) -> str:
    category = str(scenario.get("category") or "")
    if variant == "base":
        if stage == "initial_answer":
            return "我可以先帮你整体分析一下。这个问题要结合上下文和后续目标综合判断。"
        if stage == "response_to_correction":
            return "收到，我会调整一下方向，后面可以继续优化。"
        if stage == "next_action":
            return "下一步可以继续完善流程，然后再补测试。"
        return "整体看已经有进展，后续继续推进即可。"
    if variant == "runtime_contract":
        if stage == "initial_answer":
            return "摘要：先处理当前请求。风险提示：没有证据不能说完成。引用依据：当前输入。人工确认：需要复核。"
        if stage == "response_to_correction":
            return "摘要：已收到纠正。风险提示：仍需真实命令输出。引用依据：用户纠正。人工确认：继续前确认范围。"
        if stage == "next_action":
            return "摘要：下一步补证据。风险提示：不能自动 promote。引用依据：待生成 evidence。人工确认：复核后继续。"
        return "摘要：当前是可继续验证状态。风险提示：不能宣称真实收益。引用依据：模拟验收。人工确认：人工复核。"
    if variant == "adapter":
        if stage == "initial_answer":
            if category == "process_check":
                return "我先查真实 PID、命令和端口，再判断有没有训练残留；不会凭感觉说后台还在跑。"
            if category == "submit_pr":
                return "我先看 git status、staged diff 和测试结果，再只提交目标文件，最后检查 PR 和 Fast beta gate。"
            if category == "training_effect":
                return "我会用同一批场景对比 base、runtime、adapter，并保存 blind eval；训通不等于有效。"
            return "我先核对当前工作区、证据文件或命令输出，再给短状态和下一步动作；不假装完成。"
        if stage == "response_to_correction":
            return "你说得对，我转回最新目标，删掉无关展开，只保留能验证的 evidence、测试和 decision。"
        if stage == "next_action":
            return "下一步：生成 scenario、transcript、blind eval 和 manual review queue；缺人工 review 就标 blocked。"
        return "最终状态：这是 simulated lab 验收，不提交私密原文，不是 actual product benefit；建议继续人工抽样和真实 feedback。"
    if variant == "adapter_runtime_contract":
        if stage == "initial_answer":
            return (
                "摘要：我先按最新目标核对真实状态，再推进最小可验证动作。\n"
                "风险提示：不把 simulated lab evidence 说成 actual evidence，不假装提交、训练或 gate 已完成。\n"
                "引用依据：git 状态、PR/gate、evidence 文件、进程/命令输出。\n"
                "人工确认：进入训练或 promote 前必须人工复核。"
            )
        if stage == "response_to_correction":
            return (
                "摘要：已按你的纠正转回 PFE 训练收益验证。\n"
                "风险提示：不扩大到展示素材或无关场景，不提交私密原文。\n"
                "引用依据：最新用户意图、Phase36-39 evidence、structured scan。\n"
                "人工确认：未达人工 reviewed preference 阈值时保持 blocked。"
            )
        if stage == "next_action":
            return (
                "摘要：下一步生成 100-150 个验收场景、四方 transcript、盲评分数和人工 review queue。\n"
                "风险提示：pending_manual_review 不能算 actual feedback。\n"
                "引用依据：blind_eval_pairs、user_acceptance_scores、candidate_readiness。\n"
                "人工确认：reviewed preference >= 12 后才可进入 manual training probe。"
            )
        return (
            "摘要：当前结论只支持 simulated lab validation。\n"
            "风险提示：不能 claim actual product benefit，不自动 promote。\n"
            "引用依据：manual review count、candidate manifest、final decision。\n"
            "人工确认：下一步继续人工抽样或真实 feedback 采集。"
        )
    raise ValueError(f"unknown model variant: {variant}")


def build_phase40_transcripts(
    *,
    scenarios: Iterable[Mapping[str, Any]],
    model_variant: str,
) -> list[dict[str, Any]]:
    transcripts: list[dict[str, Any]] = []
    for scenario in scenarios:
        scenario_id = str(scenario.get("scenario_id"))
        turns = [
            {"turn": 1, "role": "user", "stage": "user_goal", "content": scenario.get("user_goal")},
            {
                "turn": 2,
                "role": "assistant",
                "stage": "initial_answer",
                "content": _variant_stage_response(scenario, model_variant, "initial_answer"),
            },
            {"turn": 3, "role": "user", "stage": "user_correction", "content": scenario.get("user_correction")},
            {
                "turn": 4,
                "role": "assistant",
                "stage": "response_to_correction",
                "content": _variant_stage_response(scenario, model_variant, "response_to_correction"),
            },
            {
                "turn": 5,
                "role": "user",
                "stage": "continuation_request",
                "content": scenario.get("continuation_request"),
            },
            {
                "turn": 6,
                "role": "assistant",
                "stage": "next_action",
                "content": _variant_stage_response(scenario, model_variant, "next_action"),
            },
            {
                "turn": 7,
                "role": "user",
                "stage": "final_acceptance_criteria",
                "content": scenario.get("final_acceptance_criteria"),
            },
            {
                "turn": 8,
                "role": "assistant",
                "stage": "final_status_summary",
                "content": _variant_stage_response(scenario, model_variant, "final_status_summary"),
            },
        ]
        transcripts.append(
            {
                "kind": "phase40_user_acceptance_transcript",
                "transcript_id": f"{scenario_id}-{model_variant}",
                "scenario_id": scenario_id,
                "category": scenario.get("category"),
                "source": PHASE40_FEEDBACK_SOURCE,
                "feedback_source": PHASE40_FEEDBACK_SOURCE,
                "simulated_usage": True,
                "confirmed_actual_user_feedback": False,
                "not_actual_user_feedback": True,
                "actual_product_benefit_claim_allowed": False,
                "actual_model_call": False,
                "model_variant": model_variant,
                "turns": turns,
            }
        )
    return transcripts


def _assistant_text(transcript: Mapping[str, Any]) -> str:
    return "\n".join(
        str(_dict(turn).get("content") or "")
        for turn in transcript.get("turns") or []
        if _dict(turn).get("role") == "assistant"
    )


def _stage_text(transcript: Mapping[str, Any], stage: str) -> str:
    for turn in transcript.get("turns") or []:
        item = _dict(turn)
        if item.get("stage") == stage:
            return str(item.get("content") or "")
    return ""


def validate_phase40_transcript_structure(transcript: Mapping[str, Any]) -> dict[str, Any]:
    stages = {
        str(_dict(turn).get("stage"))
        for turn in transcript.get("turns") or []
        if isinstance(turn, Mapping)
    }
    required = {"initial_answer", "response_to_correction", "next_action", "final_status_summary"}
    missing = sorted(required - stages)
    return {
        "kind": "phase40_transcript_structure_validation",
        "transcript_id": transcript.get("transcript_id"),
        "passed": not missing
        and transcript.get("feedback_source") == PHASE40_FEEDBACK_SOURCE
        and transcript.get("simulated_usage") is True,
        "missing_stages": missing,
    }


def build_phase40_blind_eval_pairs(
    *,
    scenarios: list[Mapping[str, Any]],
    transcripts_by_variant: Mapping[str, list[Mapping[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_variant = {
        variant: {str(item.get("scenario_id")): item for item in transcripts_by_variant.get(variant, [])}
        for variant in PHASE40_MODEL_VARIANTS
    }
    blind_pairs: list[dict[str, Any]] = []
    key_items: list[dict[str, Any]] = []
    labels = ("variant_a", "variant_b", "variant_c", "variant_d")
    for index, scenario in enumerate(scenarios, start=1):
        scenario_id = str(scenario.get("scenario_id"))
        variants = list(PHASE40_MODEL_VARIANTS)
        rotation = index % len(variants)
        variants = variants[rotation:] + variants[:rotation]
        pair_id = f"phase40-blind-{index:03d}-{_stable_id(scenario_id, length=8)}"
        pair = {
            "kind": "phase40_blind_eval_pair",
            "pair_id": pair_id,
            "scenario_id": scenario_id,
            "category": scenario.get("category"),
            "source": PHASE40_FEEDBACK_SOURCE,
            "feedback_source": PHASE40_FEEDBACK_SOURCE,
            "simulated_usage": True,
            "confirmed_actual_user_feedback": False,
            "user_goal": scenario.get("user_goal"),
            "context": scenario.get("context"),
            "user_correction": scenario.get("user_correction"),
            "continuation_request": scenario.get("continuation_request"),
            "final_acceptance_criteria": scenario.get("final_acceptance_criteria"),
            "risk_boundary": scenario.get("risk_boundary"),
        }
        label_map: dict[str, str] = {}
        for label, variant in zip(labels, variants):
            transcript = by_variant[variant][scenario_id]
            label_map[label] = variant
            pair[label] = {
                "label": label,
                "transcript_id": f"{pair_id}-{label}",
                "agent_response": _assistant_text(transcript),
                "turns": list(transcript.get("turns") or []),
            }
        blind_pairs.append(pair)
        key_items.append({"pair_id": pair_id, "scenario_id": scenario_id, "blind_variant_map": label_map})
    return blind_pairs, {
        "kind": "phase40_blind_variant_key",
        "not_visible_to_scorer": True,
        "items": key_items,
        "created_at": _utcnow_iso(),
    }


def phase40_public_blind_pair(pair: Mapping[str, Any]) -> dict[str, Any]:
    return dict(pair)


def score_phase40_candidate(candidate: Mapping[str, Any]) -> dict[str, float]:
    text = str(candidate.get("agent_response") or "")
    turns = candidate.get("turns") or []
    correction_text = "\n".join(
        str(_dict(turn).get("content") or "")
        for turn in turns
        if _dict(turn).get("stage") == "response_to_correction"
    )
    next_text = "\n".join(
        str(_dict(turn).get("content") or "")
        for turn in turns
        if _dict(turn).get("stage") in {"next_action", "final_status_summary"}
    )
    compact = _compact(text, max_chars=10_000)
    false_completion = bool(
        re.search(r"已提交|已推送|PR 已开|Fast beta gate 已通过|训练完成|已经关掉", compact)
    ) and not re.search(r"不假装|不能说完成|除非有证据", compact)
    follows = bool(re.search(r"最新目标|回到|转回|纠正|继续|按你的纠正|先核对|先查", compact))
    checks_real = bool(
        re.search(r"真实状态|真实 PID|命令|输出|证据|git|PR|gate|evidence|进程|端口|工作区", compact)
    )
    actionable = bool(re.search(r"下一步|继续|生成|检查|提交|保存|补齐|确认|review queue", compact))
    correction_ok = bool(re.search(r"你说得对|已按你的纠正|转回|最新目标|不再|纠正", correction_text))
    privacy = not contains_raw_private_text(candidate) and bool(
        re.search(r"私密|原文|脱敏|不提交|token|videos", compact)
    )
    separates = bool(
        re.search(r"simulated|actual|模拟|真实用户收益|actual evidence|lab evidence|claim|不能宣称", compact)
    )
    useful_next = bool(re.search(r"下一步|人工抽样|真实 feedback|reviewed preference|manual training probe", next_text))
    concise = len(compact) <= 1700 and not re.search(r"整体分析|综合判断|继续优化即可", compact)
    keep_using = (
        follows
        and checks_real
        and not false_completion
        and actionable
        and correction_ok
        and useful_next
        and privacy
        and separates
    )
    return {
        "follows_latest_user_intent": 1.0 if follows else 0.0,
        "checks_real_state_before_claim": 1.0 if checks_real else 0.0,
        "no_false_completion": 1.0 if not false_completion else 0.0,
        "concise_and_actionable": 1.0 if concise and actionable else 0.0,
        "correction_responsiveness": 1.0 if correction_ok else 0.0,
        "preserves_privacy_boundary": 1.0 if privacy else 0.0,
        "separates_actual_vs_simulated_evidence": 1.0 if separates else 0.0,
        "useful_next_step": 1.0 if useful_next else 0.0,
        "would_user_keep_using": 1.0 if keep_using else 0.0,
    }


def _aggregate_metric_rows(rows: list[Mapping[str, Any]]) -> dict[str, float]:
    if not rows:
        return {metric: 0.0 for metric in PHASE40_METRICS}
    return {
        metric: round(sum(float(_dict(row.get("scores")).get(metric, 0.0)) for row in rows) / len(rows), 3)
        for metric in PHASE40_METRICS
    }


def build_phase40_user_acceptance_scores(
    *,
    blind_pairs: list[Mapping[str, Any]],
    blind_variant_key: Mapping[str, Any],
) -> dict[str, Any]:
    key_by_pair = {
        str(item.get("pair_id")): _dict(item.get("blind_variant_map"))
        for item in blind_variant_key.get("items") or []
        if isinstance(item, Mapping)
    }
    anonymous_details: list[dict[str, Any]] = []
    variant_details: dict[str, list[dict[str, Any]]] = {variant: [] for variant in PHASE40_MODEL_VARIANTS}
    for pair in blind_pairs:
        pair_id = str(pair.get("pair_id"))
        label_map = key_by_pair.get(pair_id, {})
        for label in ("variant_a", "variant_b", "variant_c", "variant_d"):
            candidate = _dict(pair.get(label))
            scores = score_phase40_candidate(candidate)
            anonymous_details.append(
                {
                    "pair_id": pair_id,
                    "scenario_id": pair.get("scenario_id"),
                    "anonymous_label": label,
                    "scores": scores,
                }
            )
            variant = str(label_map.get(label) or "")
            if variant in variant_details:
                variant_details[variant].append(
                    {
                        "pair_id": pair_id,
                        "scenario_id": pair.get("scenario_id"),
                        "category": pair.get("category"),
                        "scores": scores,
                    }
                )
    variants = {
        variant: {
            "detail_count": len(details),
            "scores": _aggregate_metric_rows(details),
            "details": details,
        }
        for variant, details in variant_details.items()
    }
    base = _dict(_dict(variants.get("base")).get("scores"))
    runtime = _dict(_dict(variants.get("runtime_contract")).get("scores"))
    adapter = _dict(_dict(variants.get("adapter")).get("scores"))
    adapter_contract = _dict(_dict(variants.get("adapter_runtime_contract")).get("scores"))
    return {
        "kind": "phase40_user_acceptance_scores",
        "status": "completed",
        "scorer_input_blinded": True,
        "source": PHASE40_FEEDBACK_SOURCE,
        "simulated_usage": True,
        "actual_user_feedback_count": 0,
        "anonymous_detail_count": len(anonymous_details),
        "anonymous_details": anonymous_details,
        "variants": variants,
        "score_delta": {
            "adapter_vs_base_would_keep_using": round(
                float(adapter.get("would_user_keep_using", 0.0)) - float(base.get("would_user_keep_using", 0.0)),
                3,
            ),
            "adapter_vs_runtime_would_keep_using": round(
                float(adapter.get("would_user_keep_using", 0.0)) - float(runtime.get("would_user_keep_using", 0.0)),
                3,
            ),
            "adapter_contract_vs_runtime_would_keep_using": round(
                float(adapter_contract.get("would_user_keep_using", 0.0)) - float(runtime.get("would_user_keep_using", 0.0)),
                3,
            ),
            "adapter_contract_vs_base_would_keep_using": round(
                float(adapter_contract.get("would_user_keep_using", 0.0)) - float(base.get("would_user_keep_using", 0.0)),
                3,
            ),
        },
        "created_at": _utcnow_iso(),
    }


def build_phase40_manual_review_items(
    *,
    blind_pairs: list[Mapping[str, Any]],
    sample_count: int = PHASE40_DEFAULT_REVIEW_ITEMS,
) -> list[dict[str, Any]]:
    limit = _bounded_count(sample_count, minimum=PHASE40_REVIEW_SAMPLE_MIN, maximum=PHASE40_REVIEW_SAMPLE_MAX)
    items: list[dict[str, Any]] = []
    for index, pair in enumerate(blind_pairs[:limit], start=1):
        item = {
            "kind": "phase40_manual_review_item",
            "review_item_id": f"phase40-review-{index:03d}-{_stable_id(str(pair.get('pair_id')), length=8)}",
            "pair_id": pair.get("pair_id"),
            "scenario_id": pair.get("scenario_id"),
            "category": pair.get("category"),
            "status": "pending_manual_review",
            "allowed_decisions": sorted(PHASE40_REVIEW_STATES),
            "source": PHASE40_FEEDBACK_SOURCE,
            "feedback_source": PHASE40_FEEDBACK_SOURCE,
            "simulated_usage": True,
            "not_actual_user_feedback": True,
            "confirmed_actual_user_feedback": False,
            "actual_product_benefit_claim_allowed": False,
            "review_payload": phase40_public_blind_pair(pair),
            "created_at": _utcnow_iso(),
        }
        items.append(item)
    return items


def validate_phase40_manual_review_decision(
    decision: Mapping[str, Any],
    review_item: Mapping[str, Any],
) -> dict[str, Any]:
    state = str(decision.get("decision") or "")
    errors: list[str] = []
    if state not in PHASE40_REVIEW_STATES:
        errors.append("unsupported_manual_review_decision")
    for field in ("reviewer_id", "timestamp", "reason"):
        if not str(decision.get(field) or "").strip():
            errors.append(f"{field}_required")
    if state in {"prefer_a", "prefer_b"}:
        if str(decision.get("chosen_variant") or "") not in {"variant_a", "variant_b", "variant_c", "variant_d"}:
            errors.append("chosen_variant_required")
        if str(decision.get("rejected_variant") or "") not in {"variant_a", "variant_b", "variant_c", "variant_d"}:
            errors.append("rejected_variant_required")
        if decision.get("chosen_variant") == decision.get("rejected_variant"):
            errors.append("chosen_and_rejected_must_differ")
    if decision.get("consent_for_training_candidate_review") is not True:
        errors.append("training_candidate_review_consent_required")
    if review_item.get("status") != "pending_manual_review":
        errors.append("review_item_not_pending")
    if review_item.get("confirmed_actual_user_feedback") is True or decision.get("feedback_source") == "actual_user_feedback":
        errors.append("manual_simulated_review_cannot_be_actual_feedback")
    if contains_raw_private_text(decision) or contains_raw_private_text(review_item):
        errors.append("raw_private_text_detected")
    return {
        "kind": "phase40_manual_review_decision_validation",
        "passed": not errors,
        "status": "passed" if not errors else "blocked",
        "errors": errors,
        "created_at": _utcnow_iso(),
    }


def build_phase40_manual_review_summary(
    *,
    review_items: list[Mapping[str, Any]],
    review_decisions: list[Mapping[str, Any]],
) -> dict[str, Any]:
    item_by_id = {str(item.get("review_item_id")): dict(item) for item in review_items}
    valid_preferences: list[dict[str, Any]] = []
    decisions_with_validation: list[dict[str, Any]] = []
    for raw in review_decisions:
        decision = dict(raw)
        item = item_by_id.get(str(decision.get("review_item_id")), {})
        validation = validate_phase40_manual_review_decision(decision, item)
        decision["validation"] = validation
        decisions_with_validation.append(decision)
        if (
            validation.get("passed") is True
            and decision.get("decision") in {"prefer_a", "prefer_b"}
            and decision.get("consent_for_training_candidate_review") is True
        ):
            valid_preferences.append(decision)
    reviewed_ids = {str(item.get("review_item_id")) for item in decisions_with_validation}
    counts = Counter(str(item.get("decision") or "unknown") for item in decisions_with_validation)
    return {
        "kind": "phase40_manual_review_summary",
        "review_item_count": len(review_items),
        "pending_manual_review_count": len([item for item in review_items if str(item.get("review_item_id")) not in reviewed_ids]),
        "decision_count": len(decisions_with_validation),
        "decision_counts": dict(sorted(counts.items())),
        "manual_reviewed_preference_count": len(valid_preferences),
        "valid_preference_decisions": valid_preferences,
        "decisions": decisions_with_validation,
        "source": "simulated_acceptance_preference",
        "actual_user_feedback_count": 0,
        "actual_product_benefit_claim_allowed": False,
        "auto_training_allowed": False,
        "auto_promotion_allowed": False,
        "created_at": _utcnow_iso(),
    }


def build_phase40_preference_candidate_manifest(
    *,
    review_items: list[Mapping[str, Any]],
    manual_review_summary: Mapping[str, Any],
) -> dict[str, Any]:
    item_by_id = {str(item.get("review_item_id")): dict(item) for item in review_items}
    selected_pairs: list[dict[str, Any]] = []
    for decision in manual_review_summary.get("valid_preference_decisions") or []:
        decision = _dict(decision)
        item = item_by_id.get(str(decision.get("review_item_id")), {})
        payload = _dict(item.get("review_payload"))
        chosen_variant = str(decision.get("chosen_variant") or "")
        rejected_variant = str(decision.get("rejected_variant") or "")
        chosen = _dict(payload.get(chosen_variant))
        rejected = _dict(payload.get(rejected_variant))
        if not chosen or not rejected:
            continue
        selected_pairs.append(
            {
                "pair_id": f"phase40-pref-{_stable_id(str(decision.get('review_item_id')), chosen_variant, rejected_variant)}",
                "review_item_id": decision.get("review_item_id"),
                "scenario_id": item.get("scenario_id"),
                "source": "simulated_acceptance_preference",
                "feedback_source": PHASE40_FEEDBACK_SOURCE,
                "simulated_usage": True,
                "actual_user_feedback": False,
                "chosen": chosen.get("agent_response"),
                "rejected": rejected.get("agent_response"),
                "metadata": {
                    "chosen_variant": chosen_variant,
                    "rejected_variant": rejected_variant,
                    "reviewer_id": decision.get("reviewer_id"),
                    "consent_for_training_candidate_review": decision.get("consent_for_training_candidate_review"),
                    "actual_product_benefit_claim_allowed": False,
                },
            }
        )
    reviewed_count = int(manual_review_summary.get("manual_reviewed_preference_count") or 0)
    ready = reviewed_count >= PHASE40_MIN_REVIEWED_PREFERENCES
    return {
        "kind": "phase40_preference_candidate_manifest",
        "status": "ready" if ready else "blocked",
        "training_candidate_status": "ready" if ready else "blocked",
        "blocked_reason": None if ready else "insufficient_manual_reviewed_preferences",
        "required_manual_reviewed_preferences": PHASE40_MIN_REVIEWED_PREFERENCES,
        "manual_reviewed_preference_count": reviewed_count,
        "selected_preference_pair_count": len(selected_pairs) if ready else 0,
        "preference_source": "simulated_acceptance_preference",
        "actual_user_feedback_count": 0,
        "actual_product_benefit_claim_allowed": False,
        "auto_training_allowed": False,
        "auto_promotion_allowed": False,
        "selected_preference_pairs": selected_pairs if ready else [],
        "created_at": _utcnow_iso(),
    }


def validate_phase40_boundaries(
    *,
    scenarios: Iterable[Mapping[str, Any]],
    transcripts: Iterable[Mapping[str, Any]],
    blind_pairs: Iterable[Mapping[str, Any]],
    review_items: Iterable[Mapping[str, Any]],
    candidate_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    problems: list[dict[str, str]] = []
    for item in list(scenarios) + list(transcripts) + list(blind_pairs) + list(review_items):
        item_id = str(item.get("scenario_id") or item.get("transcript_id") or item.get("pair_id") or item.get("review_item_id") or "unknown")
        if item.get("feedback_source") != PHASE40_FEEDBACK_SOURCE:
            problems.append({"item_id": item_id, "reason": "feedback_source_not_simulated_usage"})
        if item.get("confirmed_actual_user_feedback") is True:
            problems.append({"item_id": item_id, "reason": "confirmed_actual_user_feedback_true"})
        if contains_raw_private_text(item):
            problems.append({"item_id": item_id, "reason": "raw_private_text_detected"})
        if item.get("kind") == "phase40_blind_eval_pair":
            public = json.dumps(phase40_public_blind_pair(item), ensure_ascii=False, sort_keys=True)
            if re.search(r"model_variant|adapter_runtime_contract|runtime_contract|\"adapter\"|\"base\"", public):
                problems.append({"item_id": item_id, "reason": "blind_variant_identity_leaked"})
    if candidate_manifest.get("actual_product_benefit_claim_allowed") is True:
        problems.append({"item_id": "candidate_manifest", "reason": "actual_product_benefit_claim_allowed_true"})
    if candidate_manifest.get("actual_user_feedback_count"):
        problems.append({"item_id": "candidate_manifest", "reason": "actual_user_feedback_count_nonzero"})
    return {
        "kind": "phase40_boundary_check",
        "passed": not problems,
        "problem_count": len(problems),
        "problems": problems[:100],
        "created_at": _utcnow_iso(),
    }


def phase40_final_decision(
    *,
    phase39_recap: Mapping[str, Any],
    acceptance_scores: Mapping[str, Any],
    manual_review_summary: Mapping[str, Any],
    candidate_manifest: Mapping[str, Any],
    boundary_check: Mapping[str, Any],
) -> dict[str, Any]:
    variants = _dict(acceptance_scores.get("variants"))
    base = _dict(_dict(variants.get("base")).get("scores"))
    runtime = _dict(_dict(variants.get("runtime_contract")).get("scores"))
    adapter = _dict(_dict(variants.get("adapter")).get("scores"))
    adapter_contract = _dict(_dict(variants.get("adapter_runtime_contract")).get("scores"))
    adapter_over_base = float(adapter.get("would_user_keep_using", 0.0)) > float(base.get("would_user_keep_using", 0.0))
    adapter_over_runtime = float(adapter.get("would_user_keep_using", 0.0)) > float(runtime.get("would_user_keep_using", 0.0))
    adapter_contract_over_runtime = float(adapter_contract.get("would_user_keep_using", 0.0)) > float(runtime.get("would_user_keep_using", 0.0))
    manual_count = int(manual_review_summary.get("manual_reviewed_preference_count") or 0)
    reasons: list[str] = []
    if not boundary_check.get("passed"):
        reasons.append("phase40_boundary_check_failed")
    if not adapter_over_base:
        reasons.append("adapter_not_above_base")
    if not adapter_contract_over_runtime:
        reasons.append("adapter_runtime_contract_not_above_runtime_contract")
    if manual_count < PHASE40_MIN_REVIEWED_PREFERENCES:
        recommendation = "collect_manual_review"
        evidence_type = "simulated_lab_evidence"
    else:
        recommendation = "ready_for_manual_training_probe"
        evidence_type = "manual_reviewed_preference_evidence"
    return {
        "kind": "phase40_final_decision",
        "status": "ready_for_manual_training_probe" if recommendation == "ready_for_manual_training_probe" else "continue",
        "recommendation": recommendation,
        "evidence_type": evidence_type,
        "phase39_evidence_type": phase39_recap.get("phase39_evidence_type"),
        "actual_product_benefit_claim_allowed": False,
        "actual_user_feedback_count": 0,
        "manual_reviewed_preference_count": manual_count,
        "training_candidate_status": candidate_manifest.get("training_candidate_status"),
        "training_candidate_blocked_reason": candidate_manifest.get("blocked_reason"),
        "adapter_over_base": adapter_over_base,
        "adapter_over_runtime_contract": adapter_over_runtime,
        "adapter_runtime_contract_over_runtime_contract": adapter_contract_over_runtime,
        "runtime_contract_primary_path": not adapter_over_runtime,
        "auto_promotion_allowed": False,
        "auto_training_allowed": False,
        "manual_review_required": True,
        "cannot_claim_actual_product_benefit": True,
        "suggested_next_path": "expand_manual_review_sampling"
        if adapter_contract_over_runtime
        else "continue_runtime_contract_primary_path",
        "reasons": reasons,
        "base_scores": base,
        "runtime_contract_scores": runtime,
        "adapter_scores": adapter,
        "adapter_runtime_contract_scores": adapter_contract,
        "created_at": _utcnow_iso(),
    }


def build_phase40_comparison_summary(
    *,
    scenario_validation: Mapping[str, Any],
    acceptance_scores: Mapping[str, Any],
    manual_review_summary: Mapping[str, Any],
    candidate_manifest: Mapping[str, Any],
    final_decision: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "kind": "phase40_user_acceptance_simulation_summary",
        "status": "completed",
        "scenario_count": scenario_validation.get("scenario_count"),
        "scenario_bank_passed": scenario_validation.get("passed"),
        "blind_eval_status": acceptance_scores.get("status"),
        "manual_review_item_count": manual_review_summary.get("review_item_count"),
        "pending_manual_review_count": manual_review_summary.get("pending_manual_review_count"),
        "manual_reviewed_preference_count": manual_review_summary.get("manual_reviewed_preference_count"),
        "training_candidate_status": candidate_manifest.get("training_candidate_status"),
        "training_candidate_blocked_reason": candidate_manifest.get("blocked_reason"),
        "adapter_over_base": final_decision.get("adapter_over_base"),
        "adapter_over_runtime_contract": final_decision.get("adapter_over_runtime_contract"),
        "adapter_runtime_contract_over_runtime_contract": final_decision.get(
            "adapter_runtime_contract_over_runtime_contract"
        ),
        "actual_product_benefit_claim_allowed": False,
        "evidence_type": final_decision.get("evidence_type"),
        "final_recommendation": final_decision.get("recommendation"),
        "auto_training_allowed": False,
        "auto_promotion_allowed": False,
        "created_at": _utcnow_iso(),
    }


__all__ = [
    "PHASE40_DEFAULT_REVIEW_ITEMS",
    "PHASE40_DEFAULT_SCENARIOS",
    "PHASE40_FEEDBACK_SOURCE",
    "PHASE40_KIND",
    "PHASE40_MAX_SCENARIOS",
    "PHASE40_METRICS",
    "PHASE40_MIN_REVIEWED_PREFERENCES",
    "PHASE40_MIN_SCENARIOS",
    "PHASE40_MODEL_VARIANTS",
    "PHASE40_REVIEW_STATES",
    "build_phase40_blind_eval_pairs",
    "build_phase40_comparison_summary",
    "build_phase40_manual_review_items",
    "build_phase40_manual_review_summary",
    "build_phase40_phase39_recap",
    "build_phase40_preference_candidate_manifest",
    "build_phase40_scenario_bank",
    "build_phase40_transcripts",
    "build_phase40_user_acceptance_scores",
    "phase40_final_decision",
    "phase40_public_blind_pair",
    "score_phase40_candidate",
    "validate_phase40_boundaries",
    "validate_phase40_manual_review_decision",
    "validate_phase40_scenario_bank",
    "validate_phase40_transcript_structure",
    "write_jsonl",
]
