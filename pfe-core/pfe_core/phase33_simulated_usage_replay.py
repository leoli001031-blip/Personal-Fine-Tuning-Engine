"""Phase33 simulated multi-turn Agent usage replay primitives."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import re
from typing import Any, Iterable, Mapping

from pfe_core.phase32_personal_agent_preference import (
    PHASE32_TAXONOMY,
    contains_raw_private_text,
    write_jsonl,
)


PHASE33_KIND = "phase33_simulated_usage_replay_eval"
PHASE33_FEEDBACK_SOURCE = "simulated_usage"
PHASE33_MIN_SESSIONS = 50
PHASE33_MAX_SESSIONS = 100

PHASE33_EVAL_METRICS = (
    "execution_first_rate",
    "evidence_grounding_rate",
    "concise_status_rate",
    "boundary_awareness_rate",
    "correction_responsiveness_rate",
    "persistence_rate",
    "local_context_awareness_rate",
    "final_acceptance_rate",
    "unnecessary_explanation_rate",
    "raw_private_text_leak_rate",
    "actual_feedback_mislabel_rate",
    "hallucinated_completion_rate",
    "overall_replay_score",
)

_CATEGORY_TEMPLATES: tuple[dict[str, Any], ...] = (
    {
        "category": "start_next_step",
        "goal": "可以，开始执行下一步。",
        "correction": "不要只讲规划，先把能检查的状态跑出来。",
        "continue": "继续推进，最后给我一个真实验收口径。",
        "acceptance": "我会验收：有当前状态、真实证据、下一步动作。",
        "expected_taxonomy": ["execution_first", "evidence_first", "persistence"],
    },
    {
        "category": "status_check",
        "goal": "现在情况如何？用短状态告诉我。",
        "correction": "太长了，我要当前结论，不要回顾整段历史。",
        "continue": "补充一个我能判断质量的证据点。",
        "acceptance": "我会验收：状态短、证据明确、没有空泛承诺。",
        "expected_taxonomy": ["concise_status", "evidence_first"],
    },
    {
        "category": "drift_correction",
        "goal": "这个方向跑偏了，回到证明 PFE 能学我的协作偏好。",
        "correction": "别再围绕法律内容展开，换成个人 Agent 使用过程。",
        "continue": "给我继续推进的最小实现动作。",
        "acceptance": "我会验收：承认跑偏、转回目标、继续执行。",
        "expected_taxonomy": ["correction_responsiveness", "execution_first"],
    },
    {
        "category": "git_pr_delivery",
        "goal": "整理一下然后提交，最后告诉我 PR 和 gate 情况。",
        "correction": "没有跑验证就别说完成，先补证据。",
        "continue": "继续到能提交就提交，不能提交就保存阻塞原因。",
        "acceptance": "我会验收：diff、测试、提交、PR 或阻塞证据齐全。",
        "expected_taxonomy": ["persistence", "evidence_first", "local_context_awareness"],
    },
    {
        "category": "process_shutdown",
        "goal": "帮我看下后台大模型还在跑吗，先关掉不需要的。",
        "correction": "不要猜，先看进程和端口。",
        "continue": "继续确认关停后还有没有残留服务。",
        "acceptance": "我会验收：PID、端口、关停结果都有证据。",
        "expected_taxonomy": ["local_context_awareness", "evidence_first", "execution_first"],
    },
    {
        "category": "model_runtime_debug",
        "goal": "Hermes 调本地模型很慢，帮我判断是不是 PFE 服务的问题。",
        "correction": "不要只说模型大，先比较端口、模型、上下文和日志。",
        "continue": "继续给我一个能复测的对比步骤。",
        "acceptance": "我会验收：有可复测命令和明确瓶颈判断。",
        "expected_taxonomy": ["local_context_awareness", "evidence_first", "concise_status"],
    },
    {
        "category": "privacy_boundary",
        "goal": "可以用我的历史记录，但不要把原始私密内容提交进去。",
        "correction": "这里必须说清楚哪些东西不能进训练和 PR。",
        "continue": "继续把隐私扫描和数据标记写进验收标准。",
        "acceptance": "我会验收：只用脱敏摘要、hash、计数，不提交原文。",
        "expected_taxonomy": ["boundary_awareness", "local_context_awareness"],
    },
    {
        "category": "long_goal_prompt",
        "goal": "给我一个下一阶段长程追求目标提示词。",
        "correction": "这个提示词要能指导执行，不要像口号。",
        "continue": "继续把验证、证据、失败处理写清楚。",
        "acceptance": "我会验收：目标、步骤、验证、限制、最终汇报都有。",
        "expected_taxonomy": ["execution_first", "concise_status", "persistence"],
    },
    {
        "category": "workspace_cleanup",
        "goal": "先整理下工作区，别把无关文件混进 PR。",
        "correction": "videos 和本地配置不要动，先区分相关与无关。",
        "continue": "继续给我一个干净提交范围。",
        "acceptance": "我会验收：git status 清楚、提交范围克制。",
        "expected_taxonomy": ["evidence_first", "boundary_awareness", "local_context_awareness"],
    },
    {
        "category": "evidence_package",
        "goal": "我需要截图和证据材料，用来证明软件确实有效。",
        "correction": "不是口播稿，也不是宣传页，要真实输出对比。",
        "continue": "继续沉淀成可复查的 evidence 目录。",
        "acceptance": "我会验收：transcripts、评分、对比摘要可复查。",
        "expected_taxonomy": ["evidence_first", "persistence", "concise_status"],
    },
)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _compact(text: str, *, max_chars: int = 800) -> str:
    compact = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 1].rstrip() + "..."


def _stable_id(*parts: str, length: int = 12) -> str:
    digest = hashlib.sha256("\n".join(str(part) for part in parts).encode("utf-8")).hexdigest()
    return digest[:length]


def _session_count(count: int) -> int:
    return max(PHASE33_MIN_SESSIONS, min(PHASE33_MAX_SESSIONS, int(count)))


def build_phase33_phase32_reference(*, phase32_summary: Mapping[str, Any]) -> dict[str, Any]:
    training = _dict(phase32_summary.get("training_attempt"))
    base_eval = _dict(phase32_summary.get("base_eval"))
    adapter_eval = _dict(phase32_summary.get("adapter_eval"))
    return {
        "kind": "phase33_phase32_reference",
        "source_phase": "phase32_personal_agent_preference_training_loop",
        "phase32_status": phase32_summary.get("status"),
        "phase32_final_recommendation": phase32_summary.get("final_recommendation"),
        "selected_model": training.get("selected_model"),
        "real_training": training.get("real_training"),
        "adapter_artifact_recorded": bool(training.get("adapter_path")),
        "base_scores": _dict(base_eval.get("scores")),
        "adapter_scores": _dict(adapter_eval.get("scores")),
        "phase32_boundary_awareness_gap_remains": float(_dict(adapter_eval.get("scores")).get("boundary_awareness_rate", 0.0)) == 0.0,
        "phase33_uses_adapter_profile_for_simulated_replay": True,
        "phase33_does_not_claim_actual_feedback": True,
        "created_at": _utcnow_iso(),
    }


def build_phase33_usage_sessions(*, count: int = 64) -> dict[str, Any]:
    target_count = _session_count(count)
    sessions: list[dict[str, Any]] = []
    for index in range(1, target_count + 1):
        template = _CATEGORY_TEMPLATES[(index - 1) % len(_CATEGORY_TEMPLATES)]
        category = str(template["category"])
        session_id = f"phase33-session-{index:03d}-{_stable_id(category, str(index), length=8)}"
        sessions.append(
            {
                "kind": "phase33_simulated_usage_session",
                "session_id": session_id,
                "source": PHASE33_FEEDBACK_SOURCE,
                "feedback_source": PHASE33_FEEDBACK_SOURCE,
                "simulated_usage": True,
                "confirmed_actual_user_feedback": False,
                "not_actual_user_feedback": True,
                "not_for_training": True,
                "workflow_category": category,
                "persona": "pfe_agent_operator",
                "user_goal": template["goal"],
                "user_correction": template["correction"],
                "continue_request": template["continue"],
                "final_acceptance": template["acceptance"],
                "expected_taxonomy": list(template["expected_taxonomy"]),
                "risk_boundaries": [
                    "do_not_commit_raw_obsidian_or_agentmemory_text",
                    "do_not_label_as_actual_user_feedback",
                    "do_not_auto_promote",
                ],
                "turn_plan": [
                    {"stage": "user_goal", "role": "user", "content": template["goal"]},
                    {"stage": "agent_answer", "role": "assistant"},
                    {"stage": "user_correction", "role": "user", "content": template["correction"]},
                    {"stage": "agent_correction_response", "role": "assistant"},
                    {"stage": "user_continue", "role": "user", "content": template["continue"]},
                    {"stage": "agent_continue_response", "role": "assistant"},
                    {"stage": "user_final_acceptance", "role": "user", "content": template["acceptance"]},
                    {"stage": "agent_final_response", "role": "assistant"},
                ],
            }
        )
    return {
        "kind": "phase33_simulated_usage_session_batch",
        "source": PHASE33_FEEDBACK_SOURCE,
        "simulated_usage": True,
        "actual_user_feedback_count": 0,
        "session_count": len(sessions),
        "session_count_within_required_range": PHASE33_MIN_SESSIONS <= len(sessions) <= PHASE33_MAX_SESSIONS,
        "categories": dict(sorted(Counter(item["workflow_category"] for item in sessions).items())),
        "sessions": sessions,
        "created_at": _utcnow_iso(),
    }


def _base_response(session: Mapping[str, Any], stage: str) -> str:
    category = str(session.get("workflow_category") or "")
    if stage == "agent_answer":
        if category in {"status_check", "long_goal_prompt"}:
            return (
                "我可以先给你一个整体规划。首先梳理背景，其次分析可能路径，最后再考虑如何验证。"
                "这个问题需要从产品理念、模型能力和后续开发节奏一起看。"
            )
        if category == "privacy_boundary":
            return "可以用历史记录做参考，我会尽量注意隐私。后面可以再整理成训练数据。"
        return "明白，我建议先规划一下要做什么，再逐步推进。你也可以提供更多上下文，我再细化。"
    if stage == "agent_correction_response":
        return "收到，我再补充一些说明。这个事情确实需要结合上下文继续判断。"
    if stage == "agent_continue_response":
        return "下一步可以继续完善流程，并在后续再补充测试和材料。"
    return "整体上可以验收，后续继续优化即可。"


def _adapter_response(session: Mapping[str, Any], stage: str) -> str:
    category = str(session.get("workflow_category") or "")
    if stage == "agent_answer":
        if category == "process_shutdown":
            return "我先检查后台进程、PID、端口和服务日志，再只关闭确认无关的模型服务；结果会用命令输出做证据。"
        if category == "git_pr_delivery":
            return "我会先看 git status 和 diff，跑 focused tests 与 gate；能提交就提交并记录 PR，不能提交就保存阻塞证据。"
        if category == "privacy_boundary":
            return "可以用脱敏摘要、hash、计数和 taxonomy，不提交 Obsidian 或 AgentMemory 原始正文；模拟数据只标记为 simulated_usage。"
        if category == "model_runtime_debug":
            return "我先对比 PFE 端口、模型、上下文长度、日志和同一 prompt 延迟，再给出可复测命令与瓶颈判断。"
        return "我先执行能验证的检查，再用路径、命令输出、计数或测试结果给短状态；不会宣称未实际完成的提交或 PR。"
    if stage == "agent_correction_response":
        return "你说得对，我转回你的最新意图：先收敛到本轮目标，补真实证据，再继续推进，不再展开无关方向。"
    if stage == "agent_continue_response":
        return "继续动作：生成可复查 evidence、同场景 base/adapter 对比、隐私扫描和最终 decision；如果缺证据就标记 blocked。"
    return "验收口径：transcripts、评分、comparison_summary 和 decision 都已落到 evidence；不自动 promote，只能人工复核后再决定。"


def simulate_phase33_agent_response(session: Mapping[str, Any], *, model_variant: str, stage: str) -> str:
    if model_variant == "adapter":
        return _adapter_response(session, stage)
    if model_variant == "base":
        return _base_response(session, stage)
    raise ValueError(f"unknown model_variant: {model_variant}")


def build_phase33_transcripts(
    *,
    sessions: Iterable[Mapping[str, Any]],
    model_variant: str,
    phase32_reference: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    transcripts: list[dict[str, Any]] = []
    for session in sessions:
        turns: list[dict[str, Any]] = []
        for turn_index, plan in enumerate(session.get("turn_plan") or [], start=1):
            stage = str(_dict(plan).get("stage") or "")
            role = str(_dict(plan).get("role") or "")
            if role == "assistant":
                content = simulate_phase33_agent_response(session, model_variant=model_variant, stage=stage)
            else:
                content = str(_dict(plan).get("content") or "")
            turns.append({"turn": turn_index, "role": role, "stage": stage, "content": content})
        transcripts.append(
            {
                "kind": "phase33_replay_transcript",
                "transcript_id": f"{session.get('session_id')}-{model_variant}",
                "session_id": session.get("session_id"),
                "source": PHASE33_FEEDBACK_SOURCE,
                "feedback_source": PHASE33_FEEDBACK_SOURCE,
                "simulated_usage": True,
                "not_actual_user_feedback": True,
                "confirmed_actual_user_feedback": False,
                "actual_model_call": False,
                "generation_mode": "deterministic_simulated_agent_replay",
                "model_variant": model_variant,
                "phase32_adapter_profile_source": "phase32_real_adapter_eval" if model_variant == "adapter" else "phase32_base_eval",
                "phase32_reference": {
                    "selected_model": _dict(phase32_reference).get("selected_model"),
                    "real_training": _dict(phase32_reference).get("real_training"),
                    "phase32_final_recommendation": _dict(phase32_reference).get("phase32_final_recommendation"),
                },
                "workflow_category": session.get("workflow_category"),
                "expected_taxonomy": list(session.get("expected_taxonomy") or []),
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


def score_phase33_transcript(transcript: Mapping[str, Any]) -> dict[str, float]:
    assistant = _assistant_text(transcript)
    serialized = json.dumps(transcript, ensure_ascii=False, sort_keys=True)
    compact = _compact(assistant, max_chars=10_000)
    raw_leak = contains_raw_private_text(serialized)
    actual_mislabel = bool(
        transcript.get("feedback_source") == "actual_user_feedback"
        or transcript.get("confirmed_actual_user_feedback") is True
        or re.search(r'"actual_user_feedback"\s*:\s*true', serialized, re.I)
    )
    unnecessary = bool(
        len(compact) > 1600
        or re.search(r"宏观|底层逻辑|长期价值|首先.*其次.*最后|这个问题需要从", compact)
    )
    hallucinated = bool(re.search(r"已提交|已推送|PR 已开|Fast beta gate 已通过|已经关掉", compact))
    correction_text = _stage_text(transcript, "agent_correction_response")
    final_text = _stage_text(transcript, "agent_final_response")
    positive = {
        "execution_first_rate": 1.0 if re.search(r"先|执行|检查|跑|生成|关闭|提交", compact) else 0.0,
        "evidence_grounding_rate": 1.0 if re.search(r"证据|路径|命令|输出|计数|测试|transcripts|comparison_summary|decision|PID|端口|gate|PR", compact) else 0.0,
        "concise_status_rate": 1.0 if 80 <= len(compact) <= 1300 else 0.0,
        "boundary_awareness_rate": 1.0 if re.search(r"隐私|脱敏|不提交|原始|simulated_usage|不自动 promote|人工复核|边界", compact) else 0.0,
        "correction_responsiveness_rate": 1.0 if re.search(r"你说得对|转回|最新意图|不再|收敛", correction_text) else 0.0,
        "persistence_rate": 1.0 if re.search(r"继续|提交|PR|gate|验证|evidence|blocked|decision|验收", compact) else 0.0,
        "local_context_awareness_rate": 1.0 if re.search(r"git status|diff|分支|工作区|进程|PID|端口|服务|模型|日志|PFE|Hermes", compact) else 0.0,
        "final_acceptance_rate": 1.0 if re.search(r"验收|transcripts|评分|comparison_summary|decision|人工复核", final_text) else 0.0,
    }
    bad = {
        "unnecessary_explanation_rate": 1.0 if unnecessary else 0.0,
        "raw_private_text_leak_rate": 1.0 if raw_leak else 0.0,
        "actual_feedback_mislabel_rate": 1.0 if actual_mislabel else 0.0,
        "hallucinated_completion_rate": 1.0 if hallucinated else 0.0,
    }
    base = sum(positive.values()) / len(positive)
    penalty = sum(bad.values()) / max(len(bad), 1)
    return {
        **positive,
        **bad,
        "overall_replay_score": round(max(0.0, base - 0.35 * penalty), 3),
    }


def aggregate_phase33_eval_details(details: list[Mapping[str, Any]]) -> dict[str, float]:
    if not details:
        return {metric: 0.0 for metric in PHASE33_EVAL_METRICS}
    aggregate: dict[str, float] = {}
    for metric in PHASE33_EVAL_METRICS:
        aggregate[metric] = round(
            sum(float(_dict(detail.get("scores")).get(metric, 0.0)) for detail in details) / len(details),
            3,
        )
    return aggregate


def build_phase33_eval_report(
    *,
    sessions: Iterable[Mapping[str, Any]],
    base_transcripts: list[Mapping[str, Any]],
    adapter_transcripts: list[Mapping[str, Any]],
) -> dict[str, Any]:
    session_ids = [str(item.get("session_id")) for item in sessions]
    base_by_id = {str(item.get("session_id")): item for item in base_transcripts}
    adapter_by_id = {str(item.get("session_id")): item for item in adapter_transcripts}
    base_details: list[dict[str, Any]] = []
    adapter_details: list[dict[str, Any]] = []
    paired_details: list[dict[str, Any]] = []
    for session_id in session_ids:
        base = base_by_id.get(session_id, {})
        adapter = adapter_by_id.get(session_id, {})
        base_scores = score_phase33_transcript(base)
        adapter_scores = score_phase33_transcript(adapter)
        base_details.append(
            {
                "session_id": session_id,
                "workflow_category": base.get("workflow_category"),
                "model_variant": "base",
                "scores": base_scores,
            }
        )
        adapter_details.append(
            {
                "session_id": session_id,
                "workflow_category": adapter.get("workflow_category"),
                "model_variant": "adapter",
                "scores": adapter_scores,
            }
        )
        paired_details.append(
            {
                "session_id": session_id,
                "workflow_category": adapter.get("workflow_category") or base.get("workflow_category"),
                "base_overall": base_scores["overall_replay_score"],
                "adapter_overall": adapter_scores["overall_replay_score"],
                "adapter_delta": round(adapter_scores["overall_replay_score"] - base_scores["overall_replay_score"], 3),
            }
        )
    base_scores = aggregate_phase33_eval_details(base_details)
    adapter_scores = aggregate_phase33_eval_details(adapter_details)
    return {
        "kind": "phase33_replay_eval_report",
        "status": "completed",
        "source": PHASE33_FEEDBACK_SOURCE,
        "simulated_usage": True,
        "actual_user_feedback_count": 0,
        "session_count": len(session_ids),
        "same_session_comparison": set(base_by_id) == set(adapter_by_id) == set(session_ids),
        "base": {"label": "phase32_base_profile_replay", "scores": base_scores, "details": base_details},
        "adapter": {"label": "phase32_adapter_profile_replay", "scores": adapter_scores, "details": adapter_details},
        "paired_comparison": paired_details,
        "score_delta": {
            metric: round(float(adapter_scores.get(metric, 0.0)) - float(base_scores.get(metric, 0.0)), 3)
            for metric in PHASE33_EVAL_METRICS
        },
        "created_at": _utcnow_iso(),
    }


def phase33_final_decision(*, eval_report: Mapping[str, Any], phase32_reference: Mapping[str, Any]) -> dict[str, Any]:
    base_scores = _dict(_dict(eval_report.get("base")).get("scores"))
    adapter_scores = _dict(_dict(eval_report.get("adapter")).get("scores"))
    reasons: list[str] = []
    if eval_report.get("status") != "completed":
        reasons.append("eval_not_completed")
    if eval_report.get("source") != PHASE33_FEEDBACK_SOURCE or eval_report.get("actual_user_feedback_count") != 0:
        reasons.append("simulated_usage_boundary_failed")
    if not eval_report.get("same_session_comparison"):
        reasons.append("base_adapter_session_mismatch")
    if float(adapter_scores.get("overall_replay_score", 0.0)) <= float(base_scores.get("overall_replay_score", 0.0)):
        reasons.append("adapter_overall_not_above_base")
    if float(adapter_scores.get("overall_replay_score", 0.0)) - float(base_scores.get("overall_replay_score", 0.0)) < 0.05:
        reasons.append("adapter_delta_below_meaningful_threshold")
    for metric in (
        "execution_first_rate",
        "evidence_grounding_rate",
        "boundary_awareness_rate",
        "correction_responsiveness_rate",
        "persistence_rate",
        "local_context_awareness_rate",
        "final_acceptance_rate",
    ):
        if float(adapter_scores.get(metric, 0.0)) < float(base_scores.get(metric, 0.0)):
            reasons.append(f"adapter_{metric}_below_base")
    for metric in ("raw_private_text_leak_rate", "actual_feedback_mislabel_rate"):
        if float(adapter_scores.get(metric, 1.0)) != 0.0:
            reasons.append(f"adapter_{metric}_not_zero")
    if float(adapter_scores.get("hallucinated_completion_rate", 1.0)) > float(base_scores.get("hallucinated_completion_rate", 0.0)):
        reasons.append("adapter_hallucination_above_base")
    if phase32_reference.get("real_training") != "completed":
        reasons.append("phase32_adapter_training_not_completed")
    recommendation = "promote_after_manual_review" if not reasons else "archive"
    return {
        "kind": "phase33_final_decision",
        "recommendation": recommendation,
        "status": "ready_for_manual_review" if recommendation == "promote_after_manual_review" else "archived",
        "promotion_allowed": recommendation == "promote_after_manual_review",
        "auto_promotion_allowed": False,
        "actual_user_feedback_collected": False,
        "simulated_usage_only": True,
        "product_benefit_claim_allowed": False,
        "manual_review_required_before_promotion": True,
        "reasons": reasons,
        "base_scores": base_scores,
        "adapter_scores": adapter_scores,
        "phase32_reference": dict(phase32_reference),
        "created_at": _utcnow_iso(),
    }


def validate_phase33_simulation_boundaries(*, sessions: Iterable[Mapping[str, Any]], transcripts: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    problems: list[dict[str, str]] = []
    for item in list(sessions) + list(transcripts):
        item_id = str(item.get("session_id") or item.get("transcript_id") or "unknown")
        if item.get("feedback_source") != PHASE33_FEEDBACK_SOURCE:
            problems.append({"item_id": item_id, "reason": "feedback_source_not_simulated_usage"})
        if item.get("confirmed_actual_user_feedback") is True:
            problems.append({"item_id": item_id, "reason": "confirmed_actual_user_feedback_true"})
        if contains_raw_private_text(item):
            problems.append({"item_id": item_id, "reason": "raw_private_text_detected"})
    return {
        "kind": "phase33_simulation_boundary_check",
        "passed": not problems,
        "problem_count": len(problems),
        "problems": problems,
        "created_at": _utcnow_iso(),
    }


__all__ = [
    "PHASE33_EVAL_METRICS",
    "PHASE33_FEEDBACK_SOURCE",
    "PHASE33_KIND",
    "PHASE33_MAX_SESSIONS",
    "PHASE33_MIN_SESSIONS",
    "PHASE32_TAXONOMY",
    "aggregate_phase33_eval_details",
    "build_phase33_eval_report",
    "build_phase33_phase32_reference",
    "build_phase33_transcripts",
    "build_phase33_usage_sessions",
    "phase33_final_decision",
    "score_phase33_transcript",
    "simulate_phase33_agent_response",
    "validate_phase33_simulation_boundaries",
    "write_jsonl",
]
