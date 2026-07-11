"""Phase42 reliability evidence and candidate-regeneration primitives."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Iterable, Mapping

from .candidate_quality import assess_preference_candidate_quality


PHASE42_KIND = "phase42_trustworthy_training_runtime_hardening"
PHASE42_MIN_HOLDOUT_COUNT = 20
PHASE42_MIN_PREFERENCE_COUNT = 12

PHASE42_GENERIC_HOLDOUTS: tuple[dict[str, Any], ...] = (
    {"id": "generic-01", "prompt": "用一句话解释什么是本地模型。", "keywords": ["本地", "设备", "电脑"]},
    {"id": "generic-02", "prompt": "列出验证软件质量的三个步骤。", "keywords": ["测试", "验证", "检查"]},
    {"id": "generic-03", "prompt": "把 hello world 翻译成中文。", "keywords": ["你好", "世界"]},
    {"id": "generic-04", "prompt": "解释 API 和用户界面的区别。", "keywords": ["接口", "界面", "api"]},
    {"id": "generic-05", "prompt": "给出一个不超过两句的项目风险提示。", "keywords": ["风险", "可能", "注意"]},
    {"id": "generic-06", "prompt": "为什么测试通过不一定代表产品可用？", "keywords": ["测试", "场景", "用户"]},
    {"id": "generic-07", "prompt": "如何确认一个后台进程仍在运行？", "keywords": ["进程", "pid", "端口"]},
    {"id": "generic-08", "prompt": "Git 提交前通常应该检查什么？", "keywords": ["git", "测试", "diff", "状态"]},
    {"id": "generic-09", "prompt": "什么是训练集和测试集隔离？", "keywords": ["训练", "测试", "隔离"]},
    {"id": "generic-10", "prompt": "用一句话说明什么是隐私同意。", "keywords": ["同意", "授权", "隐私"]},
    {"id": "generic-11", "prompt": "如何判断模型回答出现了重复？", "keywords": ["重复", "相同", "比较"]},
    {"id": "generic-12", "prompt": "请给出今天完成一项任务的简短计划。", "keywords": ["任务", "完成", "检查"]},
    {"id": "generic-13", "prompt": "什么是流式 API 响应？", "keywords": ["流", "逐步", "分块"]},
    {"id": "generic-14", "prompt": "为什么模型输出需要引用真实证据？", "keywords": ["证据", "验证", "来源"]},
    {"id": "generic-15", "prompt": "解释 base model 和 adapter 的区别。", "keywords": ["基础", "adapter", "适配"]},
    {"id": "generic-16", "prompt": "什么时候应该把训练结果标记为 blocked？", "keywords": ["失败", "条件", "证据", "blocked"]},
    {"id": "generic-17", "prompt": "如何避免把模拟反馈说成真实反馈？", "keywords": ["模拟", "真实", "标记"]},
    {"id": "generic-18", "prompt": "用一句话解释上下文窗口。", "keywords": ["上下文", "token", "文本"]},
    {"id": "generic-19", "prompt": "为什么上线前需要人工复核？", "keywords": ["人工", "风险", "复核"]},
    {"id": "generic-20", "prompt": "给出一个保存实验失败证据的方法。", "keywords": ["日志", "输出", "保存", "证据"]},
)

_SIMULATION_FOCUSES = (
    "当前分支和未提交改动", "最近提交与目标分支", "Draft PR 状态", "Fast beta gate",
    "focused tests 结果", "完整 smoke 结果", "evidence 文件路径", "后台 PID 与端口",
    "训练产物是否可解析", "参数是否真实更新", "base 与 adapter 同场输出", "holdout 隔离",
    "simulated 与 actual 标签", "私密原文排除", "失败原因持久化", "自动 promote 禁止",
    "用户最新意图", "纠正后的执行方向", "下一条可执行命令", "完成标准",
    "模型上下文预算", "SSE 结束事件", "adapter 生命周期", "人工复核边界",
)


def _normalized(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).lower()


def evaluate_adapter_generic_holdout(
    outputs: Iterable[Mapping[str, Any]],
    *,
    leakage_terms: Iterable[str] = ("未量化零点五b", "深蓝闭环"),
) -> dict[str, Any]:
    rows = [dict(row) for row in outputs]
    responses = [_normalized(row.get("response")) for row in rows]
    counts = Counter(responses)
    duplicate_count = sum(amount - 1 for text, amount in counts.items() if text and amount > 1)
    unique_ratio = round(len(set(responses)) / len(rows), 4) if rows else 0.0
    duplicate_rate = round(duplicate_count / len(rows), 4) if rows else 1.0
    leakage_terms_normalized = [_normalized(term) for term in leakage_terms if _normalized(term)]
    leakage_hits = [
        {"holdout_id": row.get("holdout_id"), "term": term}
        for row, response in zip(rows, responses)
        for term in leakage_terms_normalized
        if term in response
    ]
    relevance_hits = 0
    for row, response in zip(rows, responses):
        keywords = [_normalized(term) for term in row.get("expected_keywords") or []]
        if any(keyword and keyword in response for keyword in keywords):
            relevance_hits += 1
    count = len(rows)
    relevance_rate = round(relevance_hits / count, 4) if count else 0.0
    nonempty_rate = round(sum(bool(response) for response in responses) / count, 4) if count else 0.0
    reasons: list[str] = []
    if count < PHASE42_MIN_HOLDOUT_COUNT:
        reasons.append("insufficient_generic_holdout")
    if leakage_hits:
        reasons.append("training_leakage_detected")
    if unique_ratio < 0.7:
        reasons.append("response_unique_ratio_below_threshold")
    if duplicate_rate > 0.15:
        reasons.append("response_duplicate_rate_above_threshold")
    if relevance_rate < 0.4:
        reasons.append("task_relevance_rate_below_threshold")
    if nonempty_rate < 1.0:
        reasons.append("empty_holdout_response")
    passed = not reasons
    return {
        "kind": "phase42_adapter_generic_holdout_report",
        "passed": passed,
        "holdout": {"count": count, "passed": passed},
        "holdout_count": count,
        "training_leakage_detected": bool(leakage_hits),
        "leakage_hits": leakage_hits,
        "scores": {
            "response_unique_ratio": unique_ratio,
            "exact_duplicate_rate": duplicate_rate,
            "task_relevance_rate": relevance_rate,
            "nonempty_rate": nonempty_rate,
        },
        "thresholds": {
            "minimum_holdout_count": PHASE42_MIN_HOLDOUT_COUNT,
            "minimum_unique_ratio": 0.7,
            "maximum_duplicate_rate": 0.15,
            "minimum_task_relevance_rate": 0.4,
            "required_nonempty_rate": 1.0,
        },
        "reasons": reasons,
    }


def build_phase41_v2_simulated_candidates(
    *,
    review_items: Iterable[Mapping[str, Any]],
    review_decisions: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    item_by_id = {str(item.get("review_item_id")): dict(item) for item in review_items}
    pairs: list[dict[str, Any]] = []
    for index, decision in enumerate(review_decisions):
        if decision.get("decision") not in {"prefer_a", "prefer_b"}:
            continue
        item = item_by_id.get(str(decision.get("review_item_id")), {})
        payload = item.get("review_payload") if isinstance(item.get("review_payload"), Mapping) else {}
        goal = str(payload.get("user_goal") or "继续当前 PFE 任务").strip()
        correction = str(payload.get("user_correction") or "先核对证据再继续").strip()
        continuation = str(payload.get("continuation_request") or "给出下一步动作").strip()
        focus = _SIMULATION_FOCUSES[index % len(_SIMULATION_FOCUSES)]
        instruction = f"用户目标：{goal}\n用户纠正：{correction}\n继续要求：{continuation}\n本轮核验重点：{focus}"
        chosen = (
            f"我会先核对{focus}，再回答“{goal}”。收到纠正后，按“{correction}”收缩范围；"
            f"随后执行“{continuation}”。只报告真实命令、路径或计数，缺证据就标 blocked；"
            "该记录是 simulated_usage，不是 actual_user_feedback，也不允许自动 promote。"
        )
        rejected = (
            f"关于“{goal}”，我会先做整体分析，{focus}以后再看。"
            "目前可以认为已经基本完成，后续继续优化即可。"
        )
        pairs.append(
            {
                "pair_id": f"phase41-v2-{index + 1:03d}",
                "scenario_id": item.get("scenario_id"),
                "instruction": instruction,
                "chosen": chosen,
                "rejected": rejected,
                "simulation_focus": focus,
                "source": "phase42_scenario_specific_simulation",
                "feedback_source": "simulated_usage",
                "simulated_usage": True,
                "actual_model_call": False,
                "confirmed_actual_user_feedback": False,
                "not_actual_user_feedback": True,
                "actual_product_benefit_claim_allowed": False,
                "auto_training_allowed": False,
                "auto_promotion_allowed": False,
            }
        )
    quality = assess_preference_candidate_quality(pairs)
    enough = len(pairs) >= PHASE42_MIN_PREFERENCE_COUNT
    return {
        "kind": "phase41_v2_scenario_specific_simulated_candidate_manifest",
        "status": "quality_ready_for_future_manual_review" if enough and quality["passed"] else "blocked",
        "candidate_count": len(pairs),
        "minimum_candidate_count": PHASE42_MIN_PREFERENCE_COUNT,
        "candidate_quality": quality,
        "selected_preference_pairs": pairs,
        "simulated_usage": True,
        "actual_model_call": False,
        "actual_user_feedback_count": 0,
        "actual_product_benefit_claim_allowed": False,
        "manual_training_probe_allowed": False,
        "auto_training_allowed": False,
        "auto_promotion_allowed": False,
    }


def build_phase42_final_decision(
    *,
    adapter_report: Mapping[str, Any],
    lifecycle_decision: Mapping[str, Any],
    training_attempt: Mapping[str, Any],
    context_smoke: Mapping[str, Any],
    hermes_streaming_passed: bool,
    security_tests_passed: bool,
    full_validation_passed: bool,
    phase41_current: Mapping[str, Any],
    phase41_v2: Mapping[str, Any],
) -> dict[str, Any]:
    checks = {
        "bad_adapter_detected": adapter_report.get("passed") is False,
        "bad_adapter_archived_without_deletion": (
            lifecycle_decision.get("action") == "archived"
            and lifecycle_decision.get("artifact_retained") is True
        ),
        "real_local_training_completed": (
            training_attempt.get("real_training") is True
            and (training_attempt.get("adapter_validation") or {}).get("valid") is True
            and (training_attempt.get("execution") or {}).get("parameters_updated") is True
        ),
        "hermes_openai_streaming_passed": hermes_streaming_passed,
        "real_context_budget_passed": context_smoke.get("passed") is True,
        "security_and_privacy_tests_passed": security_tests_passed,
        "full_validation_gate_passed": full_validation_passed,
        "phase41_duplicate_candidate_blocked": phase41_current.get("training_candidate_status") == "blocked",
        "phase41_v2_quality_passed": (
            phase41_v2.get("status") == "quality_ready_for_future_manual_review"
            and (phase41_v2.get("candidate_quality") or {}).get("passed") is True
        ),
    }
    reliability_passed = all(checks.values())
    return {
        "kind": "phase42_final_decision",
        "status": "reliability_gate_passed" if reliability_passed else "blocked",
        "recommendation": (
            "return_to_personal_preference_training_after_manual_data_review"
            if reliability_passed
            else "fix_remaining_phase42_reliability_failures"
        ),
        "reliability_gate_passed": reliability_passed,
        "checks": checks,
        "adapter_version_reviewed": adapter_report.get("version"),
        "adapter_action": lifecycle_decision.get("action"),
        "actual_product_benefit_claim_allowed": False,
        "simulated_candidate_product_benefit_claim_allowed": False,
        "auto_training_allowed": False,
        "auto_promotion_allowed": False,
        "next_gate": "manual_review_of_diverse_preference_candidates_before_any_training_probe",
    }


__all__ = [
    "PHASE42_GENERIC_HOLDOUTS",
    "PHASE42_KIND",
    "build_phase41_v2_simulated_candidates",
    "build_phase42_final_decision",
    "evaluate_adapter_generic_holdout",
]
