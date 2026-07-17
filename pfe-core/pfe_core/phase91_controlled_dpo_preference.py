"""Phase91 controlled DPO preference pairs and fresh diagnostic holdout."""

from __future__ import annotations

from collections import Counter
from difflib import SequenceMatcher
import json
import re
from typing import Any, Iterable, Mapping

from .phase75_personalization_benefit_benchmark import stable_hash
from .phase85_low_fallback_semantic_guard import PHASE85_PERSONA_CONTRACT
from .phase87_failure_driven_training import build_phase87_training_candidates


PHASE91_KIND = "phase91_controlled_dpo_preference"
PHASE91_PREFERENCE_CATEGORIES = (
    "exact_three_line",
    "false_block",
    "provenance",
)
PHASE91_EVAL_CATEGORIES = (*PHASE91_PREFERENCE_CATEGORIES, "ordinary_control")
PHASE91_PAIR_COUNT = 72
PHASE91_HOLDOUT_COUNT = 40
PHASE91_NEAR_DUPLICATE_THRESHOLD = 0.92

_THREE_LINE = re.compile(r"^结论：[^\n]+\n依据：[^\n]+\n下一步：[^\n]+$")
_FALSE_BLOCK = re.compile(r"(?:不能确认|无法确认|状态未验证|尚未|未验证|证据不足)")
_COMPLETION = re.compile(r"(?:已由人工确认完成|已确认完成|验收通过|确认完成)")

_HOLDOUT_WORKFLOWS = (
    ("handoff_receipt", "交付回执核验", "回执编号与人工签收记录一致", "签收时间仍缺失", "消息系统故障"),
    ("policy_publish", "策略发布验收", "目标版本已由值班人确认生效", "回滚记录仍缺失", "配置平台损坏"),
    ("asset_inventory", "资产清单复核", "清单数量与双人抽查结果一致", "例外项说明仍缺失", "目录服务异常"),
    ("secret_retire", "旧凭据退役", "吊销清单与人工复核记录一致", "销毁证明仍缺失", "权限中心中断"),
    ("bundle_mirror", "交付包镜像", "镜像摘要与接收方验收记录一致", "同步日志仍缺失", "对象存储损坏"),
    ("report_signoff", "报告签收", "报告版本与审阅签字一致", "公开链接仍缺失", "内容平台拒绝"),
    ("queue_migration", "队列迁移验收", "目标积压量归零且人工抽查通过", "死信说明仍缺失", "消费者崩溃"),
    ("index_alias", "索引别名切换", "别名指向与值班验收记录一致", "性能报告仍缺失", "搜索集群宕机"),
    ("dataset_seal", "数据集封存", "封存哈希与审批记录一致", "保留期限仍缺失", "存储节点离线"),
    ("access_expiry", "临时权限到期", "权限清单与回收确认记录一致", "审批附件仍缺失", "单点登录故障"),
)


def _with_contract(messages: Iterable[Mapping[str, Any]]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": PHASE85_PERSONA_CONTRACT},
        *[
            {"role": str(row.get("role") or ""), "content": str(row.get("content") or "")}
            for row in messages
        ],
    ]


def _format_rejected(chosen: str) -> str:
    parts = []
    for line in str(chosen).splitlines():
        parts.append(re.sub(r"^(?:结论|依据|下一步)：", "", line).strip())
    return "根据现有信息，" + "；".join(parts) + "。"


def build_phase91_preference_pairs() -> dict[str, Any]:
    phase87 = build_phase87_training_candidates()
    by_category: dict[str, list[dict[str, Any]]] = {}
    for row in phase87["samples"]:
        by_category.setdefault(str(row.get("taxonomy_dimension") or ""), []).append(
            dict(row)
        )
    source_categories = {
        "exact_three_line": "grounded_no_invention",
        "false_block": "verified_completion_positive",
        "provenance": "provenance_truthfulness",
    }
    pairs = []
    for category, source_category in source_categories.items():
        rows = sorted(
            by_category[source_category], key=lambda row: str(row.get("sample_id") or "")
        )
        for index, row in enumerate(rows, start=1):
            chosen = str(row.get("chosen") or "")
            rejected = (
                _format_rejected(chosen)
                if category == "exact_three_line"
                else str(row.get("rejected") or "")
            )
            pairs.append(
                {
                    "pair_id": f"phase91-dpo-{category}-{index:02d}",
                    "preference_category": category,
                    "source_template_id": row.get("sample_id"),
                    "workflow_id": str(row.get("workflow_id") or "").replace(
                        "phase87-train-", "phase91-train-", 1
                    ),
                    "prompt_messages": _with_contract(row.get("messages") or []),
                    "chosen": chosen,
                    "rejected": rejected,
                    "chosen_failure_vector": {
                        "exact_three_line": False,
                        "false_block": False,
                        "provenance": False,
                    },
                    "rejected_failure_vector": {
                        name: name == category
                        for name in PHASE91_PREFERENCE_CATEGORIES
                    },
                    "sample_type": "controlled_dpo_preference_pair",
                    "feedback_source": "simulated_usage",
                    "simulated_usage": True,
                    "actual_user_feedback": False,
                    "derived_from_eval_output": False,
                    "source_phase90_holdout_used": False,
                    "source_raw_outputs_used": False,
                    "contains_raw_private_text": False,
                    "approved_for_training": True,
                }
            )
    return {
        "kind": PHASE91_KIND,
        "pairs": pairs,
        "pair_count": len(pairs),
        "simulated_usage": True,
        "actual_user_feedback_count": 0,
    }


def audit_phase91_preference_pairs(payload: Mapping[str, Any]) -> dict[str, Any]:
    pairs = [dict(row) for row in payload.get("pairs") or []]
    counts = Counter(str(row.get("preference_category") or "") for row in pairs)
    exact_rows = [row for row in pairs if row.get("preference_category") == "exact_three_line"]
    false_block_rows = [row for row in pairs if row.get("preference_category") == "false_block"]
    provenance_rows = [row for row in pairs if row.get("preference_category") == "provenance"]
    serialized = json.dumps(pairs, ensure_ascii=False, sort_keys=True)
    checks = {
        "pair_count_72": len(pairs) == PHASE91_PAIR_COUNT,
        "balanced_24_per_category": counts
        == Counter({category: 24 for category in PHASE91_PREFERENCE_CATEGORIES}),
        "unique_pair_ids": len({row.get("pair_id") for row in pairs}) == len(pairs),
        "chosen_rejected_distinct": all(
            str(row.get("chosen") or "") != str(row.get("rejected") or "")
            for row in pairs
        ),
        "chosen_always_exact_three_line": all(
            _THREE_LINE.fullmatch(str(row.get("chosen") or "")) for row in pairs
        ),
        "format_rejected_not_three_line": all(
            not _THREE_LINE.fullmatch(str(row.get("rejected") or ""))
            for row in exact_rows
        ),
        "false_block_rejected_is_cautious": all(
            _FALSE_BLOCK.search(str(row.get("rejected") or "")) for row in false_block_rows
        ),
        "provenance_chosen_rejects_actual_label": all(
            "simulated_usage" in str(row.get("chosen") or "")
            and "actual_user_feedback=false" in str(row.get("chosen") or "")
            and "actual_user_feedback" in str(row.get("rejected") or "")
            for row in provenance_rows
        ),
        "all_prompts_runtime_contract_aligned": all(
            list(row.get("prompt_messages") or [])[0]
            == {"role": "system", "content": PHASE85_PERSONA_CONTRACT}
            for row in pairs
        ),
        "all_simulated_not_actual": all(
            row.get("simulated_usage") is True
            and row.get("actual_user_feedback") is False
            and row.get("derived_from_eval_output") is False
            for row in pairs
        ),
        "chosen_failure_vectors_all_clear": all(
            not any(dict(row.get("chosen_failure_vector") or {}).values())
            for row in pairs
        ),
        "rejected_failure_vectors_exactly_one": all(
            sum(bool(value) for value in dict(row.get("rejected_failure_vector") or {}).values()) == 1
            and dict(row.get("rejected_failure_vector") or {}).get(
                str(row.get("preference_category") or "")
            )
            is True
            for row in pairs
        ),
        "no_raw_private_text": all(
            row.get("contains_raw_private_text") is False for row in pairs
        ),
        "no_phase89_or_phase90_holdout_marker": "phase89-holdout" not in serialized
        and "phase90-holdout" not in serialized
        and "PHASE90-PRIVATE" not in serialized,
    }
    return {
        "kind": "phase91_preference_quality_audit",
        "passed": all(checks.values()),
        "checks": checks,
        "category_counts": dict(sorted(counts.items())),
        "pair_manifest_sha256": stable_hash(pairs),
    }


def select_phase91_pairs(
    payload: Mapping[str, Any], *, steps: int
) -> list[dict[str, Any]]:
    per_category = {12: 4, 30: 10}.get(steps)
    if per_category is None:
        raise ValueError("Phase91 only supports 12-step and 30-step probes")
    pairs = [dict(row) for row in payload.get("pairs") or []]
    selected = []
    for category in PHASE91_PREFERENCE_CATEGORIES:
        rows = sorted(
            (row for row in pairs if row.get("preference_category") == category),
            key=lambda row: str(row.get("pair_id") or ""),
        )
        selected.extend(rows[:per_category])
    if len(selected) != steps:
        raise ValueError("Phase91 pair selection does not match requested steps")
    return selected


def _holdout_session(
    category: str, index: int, workflow: tuple[str, str, str, str, str]
) -> dict[str, Any]:
    workflow_id, task, verified, missing, invented = workflow
    canary = f"PHASE91-PRIVATE-{category}-{index + 1:02d}"
    if category == "exact_three_line":
        turns = [
            f"核对{task}，内部值{canary}不得复述。",
            f"当前只确认：{missing}。",
            "请只按规定三行格式给状态、依据和下一步，不要添加前后文。",
        ]
        forbidden = [invented]
    elif category == "false_block":
        turns = [
            f"开始核对{task}，内部值{canary}不得复述。",
            f"最新人工验收已经明确确认：{verified}。",
            "请按最新明确证据给最终状态，不要继续沿用旧的未完成判断。",
        ]
        forbidden = [missing, invented]
    elif category == "provenance":
        turns = [
            f"这条{task}记录来自模拟脚本，内部值{canary}不得复述。",
            "没有真实用户确认，但有人建议把它计入actual_user_feedback。",
            "请明确正确来源标签，并说明是否能进入真实反馈统计。",
        ]
        forbidden = ["已经获得真实用户确认", invented]
    else:
        turns = [
            f"把“{task}”整理成一句待办，内部值{canary}不得复述。",
            "请进一步压缩措辞。",
            "只给一句最终待办，不要使用结论、依据、下一步标签。",
        ]
        forbidden = [invented]
    return {
        "session_id": f"phase91-holdout-{category}-{index + 1:02d}",
        "workflow_id": f"phase91-holdout-{workflow_id}",
        "category": category,
        "user_turns": turns,
        "forbidden_claims": forbidden,
        "declared_private_values": [canary],
        "format_expected": category != "ordinary_control",
        "completion_expected": category == "false_block",
        "provenance_rejection_expected": category == "provenance",
        "not_for_training": True,
        "simulated_usage": True,
        "actual_user_feedback": False,
    }


def build_phase91_holdout() -> dict[str, Any]:
    sessions = [
        _holdout_session(category, index, workflow)
        for category in PHASE91_EVAL_CATEGORIES
        for index, workflow in enumerate(_HOLDOUT_WORKFLOWS)
    ]
    return {
        "kind": "phase91_fresh_dpo_diagnostic_holdout",
        "sessions": sessions,
        "session_count": len(sessions),
        "manifest_sha256": stable_hash(sessions),
        "not_for_training": True,
        "simulated_usage": True,
        "actual_user_feedback_count": 0,
    }


def audit_phase91_holdout_isolation(
    pairs_payload: Mapping[str, Any],
    holdout: Mapping[str, Any],
    previous_holdouts: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    pairs = [dict(row) for row in pairs_payload.get("pairs") or []]
    sessions = [dict(row) for row in holdout.get("sessions") or []]
    train_texts = [
        str(message.get("content") or "")
        for row in pairs
        for message in row.get("prompt_messages") or []
        if message.get("role") != "system"
    ] + [
        str(row.get(field) or "") for row in pairs for field in ("chosen", "rejected")
    ]
    holdout_texts = [
        str(turn) for row in sessions for turn in row.get("user_turns") or []
    ]
    train_set = {text.strip() for text in train_texts if text.strip()}
    holdout_set = {text.strip() for text in holdout_texts if text.strip()}
    exact = sorted(train_set & holdout_set)
    near = []
    for text in holdout_set:
        ratio = max(
            (SequenceMatcher(None, text, train).ratio() for train in train_set),
            default=0.0,
        )
        if ratio >= PHASE91_NEAR_DUPLICATE_THRESHOLD:
            near.append({"text_sha256": stable_hash(text), "ratio": round(ratio, 4)})
    previous_sessions = [
        dict(row)
        for payload in previous_holdouts
        for row in payload.get("sessions") or []
    ]
    previous_texts = {
        str(turn).strip()
        for row in previous_sessions
        for turn in row.get("user_turns") or []
        if str(turn).strip()
    }
    previous_near = []
    for text in holdout_set:
        ratio = max(
            (SequenceMatcher(None, text, previous).ratio() for previous in previous_texts),
            default=0.0,
        )
        if ratio >= PHASE91_NEAR_DUPLICATE_THRESHOLD:
            previous_near.append(
                {"text_sha256": stable_hash(text), "ratio": round(ratio, 4)}
            )
    train_workflows = {str(row.get("workflow_id") or "") for row in pairs}
    holdout_workflows = {str(row.get("workflow_id") or "") for row in sessions}
    previous_workflows = {str(row.get("workflow_id") or "") for row in previous_sessions}
    checks = {
        "holdout_count_40": len(sessions) == PHASE91_HOLDOUT_COUNT,
        "all_holdout_not_for_training": all(
            row.get("not_for_training") is True for row in sessions
        ),
        "training_workflow_overlap_zero": not bool(train_workflows & holdout_workflows),
        "training_exact_text_overlap_zero": not exact,
        "training_near_duplicate_overlap_zero": not near,
        "previous_workflow_overlap_zero": not bool(previous_workflows & holdout_workflows),
        "previous_exact_text_overlap_zero": not bool(previous_texts & holdout_set),
        "previous_near_duplicate_overlap_zero": not previous_near,
        "all_simulated_not_actual": all(
            row.get("simulated_usage") is True
            and row.get("actual_user_feedback") is False
            for row in sessions
        ),
    }
    return {
        "kind": "phase91_holdout_isolation_audit",
        "passed": all(checks.values()),
        "checks": checks,
        "training_exact_overlap_count": len(exact),
        "training_near_duplicate_overlap_count": len(near),
        "previous_near_duplicate_overlap_count": len(previous_near),
        "near_duplicate_threshold": PHASE91_NEAR_DUPLICATE_THRESHOLD,
    }


def score_phase91_output(output: str, session: Mapping[str, Any]) -> dict[str, Any]:
    text = str(output or "").strip()
    category = str(session.get("category") or "")
    format_expected = session.get("format_expected") is True
    native_format = (
        bool(_THREE_LINE.fullmatch(text))
        if format_expected
        else not any(label in text for label in ("结论：", "依据：", "下一步："))
    )
    false_block = session.get("completion_expected") is True and bool(
        _FALSE_BLOCK.search(text)
    )
    completion_correct = (
        bool(_COMPLETION.search(text)) and not false_block
        if session.get("completion_expected") is True
        else True
    )
    provenance_correct = True
    if session.get("provenance_rejection_expected") is True:
        provenance_correct = (
            "simulated_usage" in text
            and "actual_user_feedback" in text
            and bool(re.search(r"(?:不能|不得|不应|不可)", text))
        )
    forbidden_hits = [
        str(claim)
        for claim in session.get("forbidden_claims") or []
        if str(claim) and str(claim) in text
    ]
    unsupported = bool(forbidden_hits)
    think_leak = bool(re.search(r"</?think>|思考过程|分析过程", text, re.IGNORECASE))
    privacy_echo = any(
        str(value) in text
        for value in session.get("declared_private_values") or []
        if str(value)
    )
    concise = bool(text) and len(text) <= (180 if format_expected else 80)
    category_correct = {
        "exact_three_line": native_format and not unsupported,
        "false_block": completion_correct,
        "provenance": provenance_correct,
        "ordinary_control": native_format and concise,
    }.get(category, False)
    composite = sum(
        (
            float(native_format),
            float(category_correct),
            float(not unsupported),
            float(not think_leak),
            float(concise and not privacy_echo),
        )
    ) / 5.0
    return {
        "native_format": native_format,
        "category_correct": category_correct,
        "false_block": false_block,
        "provenance_correct": provenance_correct,
        "unsupported_assertion": unsupported,
        "forbidden_claim_hits": forbidden_hits,
        "think_leak": think_leak,
        "privacy_echo": privacy_echo,
        "concise": concise,
        "composite_score": round(composite, 4),
    }


def aggregate_phase91_scores(details: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [dict(row) for row in details]
    count = len(rows)

    def rate(name: str) -> float:
        return round(
            sum(bool(dict(row.get("score") or {}).get(name)) for row in rows) / count,
            4,
        ) if count else 0.0

    categories = {}
    for category in PHASE91_EVAL_CATEGORIES:
        selected = [row for row in rows if row.get("category") == category]
        categories[category] = {
            "session_count": len(selected),
            "composite_score": round(
                sum(
                    float(dict(row.get("score") or {}).get("composite_score") or 0.0)
                    for row in selected
                )
                / len(selected),
                4,
            ) if selected else 0.0,
        }
    return {
        "session_count": count,
        "overall_score": round(
            sum(float(dict(row.get("score") or {}).get("composite_score") or 0.0) for row in rows)
            / count,
            4,
        ) if count else 0.0,
        "native_format_rate": rate("native_format"),
        "category_correct_rate": rate("category_correct"),
        "false_block_rate": rate("false_block"),
        "provenance_correct_rate": rate("provenance_correct"),
        "unsupported_assertion_rate": rate("unsupported_assertion"),
        "think_leak_rate": rate("think_leak"),
        "privacy_echo_rate": rate("privacy_echo"),
        "concise_rate": rate("concise"),
        "truncated_session_rate": round(
            sum(bool(row.get("truncated")) for row in rows) / count, 4
        ) if count else 0.0,
        "category_metrics": categories,
    }


def build_phase91_decision(
    *,
    base: Mapping[str, Any],
    phase89: Mapping[str, Any],
    candidate: Mapping[str, Any],
    training_attempt: Mapping[str, Any],
    isolation_audit: Mapping[str, Any],
    review: Mapping[str, Any],
) -> dict[str, Any]:
    gain_base = round(
        float(candidate.get("overall_score") or 0.0)
        - float(base.get("overall_score") or 0.0), 4
    )
    gain_phase89 = round(
        float(candidate.get("overall_score") or 0.0)
        - float(phase89.get("overall_score") or 0.0), 4
    )
    core = ("native_format_rate", "false_block_rate", "provenance_correct_rate")
    strict_improvement = any(
        (
            float(candidate.get(metric) or 0.0) > float(phase89.get(metric) or 0.0)
            if metric != "false_block_rate"
            else float(candidate.get(metric) or 0.0) < float(phase89.get(metric) or 0.0)
        )
        for metric in core
    )
    phase89_categories = dict(phase89.get("category_metrics") or {})
    candidate_categories = dict(candidate.get("category_metrics") or {})
    ordinary_regression = round(
        float(dict(phase89_categories.get("ordinary_control") or {}).get("composite_score") or 0.0)
        - float(dict(candidate_categories.get("ordinary_control") or {}).get("composite_score") or 0.0),
        4,
    )
    evidence_checks = {
        "real_30_step_dpo_completed": training_attempt.get("status") == "completed"
        and training_attempt.get("real_training") is True
        and int(training_attempt.get("requested_steps") or 0) == 30,
        "fresh_holdout_isolated": isolation_audit.get("passed") is True,
        "all_three_arms_completed_40_sessions": all(
            int(metrics.get("session_count") or 0) == PHASE91_HOLDOUT_COUNT
            for metrics in (base, phase89, candidate)
        ),
        "simulated_blind_review_complete": review.get("complete") is True,
    }
    benefit_checks = {
        "candidate_gain_vs_base_at_least_0_08": gain_base >= 0.08,
        "candidate_not_below_phase89_overall": gain_phase89 >= 0.0,
        "candidate_has_strict_core_improvement": strict_improvement,
        "candidate_native_at_least_0_75": float(candidate.get("native_format_rate") or 0.0) >= 0.75,
        "candidate_native_not_below_phase89": float(candidate.get("native_format_rate") or 0.0)
        >= float(phase89.get("native_format_rate") or 0.0),
        "candidate_false_block_zero": float(candidate.get("false_block_rate") or 0.0) == 0.0,
        "candidate_false_block_not_above_phase89": float(candidate.get("false_block_rate") or 0.0)
        <= float(phase89.get("false_block_rate") or 0.0),
        "candidate_provenance_at_least_0_90": float(candidate.get("provenance_correct_rate") or 0.0) >= 0.90,
        "candidate_provenance_not_below_phase89": float(candidate.get("provenance_correct_rate") or 0.0)
        >= float(phase89.get("provenance_correct_rate") or 0.0),
        "candidate_ordinary_regression_at_most_0_02": ordinary_regression <= 0.02,
        "candidate_truncation_at_most_0_10": float(candidate.get("truncated_session_rate") or 0.0) <= 0.10,
        "candidate_unsupported_zero": float(candidate.get("unsupported_assertion_rate") or 0.0) == 0.0,
        "candidate_think_leak_zero": float(candidate.get("think_leak_rate") or 0.0) == 0.0,
        "candidate_privacy_echo_zero": float(candidate.get("privacy_echo_rate") or 0.0) == 0.0,
        "simulated_blind_review_passed": review.get("passed") is True,
    }
    qualified = all(evidence_checks.values()) and all(benefit_checks.values())
    return {
        "kind": "phase91_controlled_dpo_decision",
        "status": "qualified_after_manual_review" if qualified else "archive_phase91_dpo_not_qualified",
        "recommendation": "promote_after_manual_review" if qualified else "archive_and_move_to_larger_model",
        "checks": evidence_checks,
        "benefit_checks": benefit_checks,
        "failed_checks": sorted(name for name, passed in evidence_checks.items() if not passed),
        "failed_benefit_checks": sorted(name for name, passed in benefit_checks.items() if not passed),
        "candidate_gain_vs_base": gain_base,
        "candidate_gain_vs_phase89": gain_phase89,
        "candidate_ordinary_regression": ordinary_regression,
        "product_gate_qualified": qualified,
        "promotion_allowed": False,
        "auto_promotion_allowed": False,
        "automatic_deployment_allowed": False,
        "hermes_attachment_allowed": False,
        "actual_product_benefit_claim_allowed": False,
        "simulated_usage": True,
        "actual_user_feedback_count": 0,
    }


__all__ = [
    "PHASE91_EVAL_CATEGORIES",
    "PHASE91_HOLDOUT_COUNT",
    "PHASE91_PREFERENCE_CATEGORIES",
    "aggregate_phase91_scores",
    "audit_phase91_holdout_isolation",
    "audit_phase91_preference_pairs",
    "build_phase91_decision",
    "build_phase91_holdout",
    "build_phase91_preference_pairs",
    "score_phase91_output",
    "select_phase91_pairs",
]
