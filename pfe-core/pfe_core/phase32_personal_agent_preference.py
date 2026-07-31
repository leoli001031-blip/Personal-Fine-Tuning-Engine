"""Phase32 personal Agent preference training-loop primitives."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import re
from typing import Any, Iterable, Mapping


PHASE32_KIND = "phase32_personal_agent_preference_training_loop"
PHASE32_FEEDBACK_SOURCE = "historical_user_agent_conversation"
PHASE32_MIN_TRAINING_CANDIDATES = 12
PHASE32_MIN_HOLDOUT_PROMPTS = 40

PHASE32_REVIEW_STATUSES = {
    "approved_for_training",
    "approved_for_profile",
    "approved_for_memory",
    "excluded",
    "quarantined",
}

PHASE32_TAXONOMY = {
    "execution_first": {
        "label": "先执行、少空谈",
        "positive_behavior": "先动手检查或执行，再用简短状态说明结果。",
    },
    "evidence_first": {
        "label": "真实文件、路径、测试、截图、计数",
        "positive_behavior": "用真实路径、命令、计数、测试或截图证据支撑结论。",
    },
    "concise_status": {
        "label": "短状态、清楚结论",
        "positive_behavior": "状态汇报短、具体、有当前结论和下一步。",
    },
    "boundary_awareness": {
        "label": "隐私、权限、风险边界清楚",
        "positive_behavior": "保护隐私和权限边界，不泄露原始敏感材料。",
    },
    "persistence": {
        "label": "推进到提交、PR、验证",
        "positive_behavior": "不止给建议，尽量推进到验证、提交、PR 或明确阻塞证据。",
    },
    "correction_responsiveness": {
        "label": "被纠正后快速转向",
        "positive_behavior": "承认方向偏差，立即改按用户最新意图执行。",
    },
    "local_context_awareness": {
        "label": "理解本机路径、项目状态、分支、进程",
        "positive_behavior": "围绕当前本机 worktree、路径、分支、服务和进程状态行动。",
    },
}

PHASE32_EVAL_METRICS = (
    "execution_first_rate",
    "evidence_grounding_rate",
    "concise_status_rate",
    "boundary_awareness_rate",
    "correction_responsiveness_rate",
    "unnecessary_explanation_rate",
    "raw_private_text_leak_rate",
    "hallucinated_completion_rate",
    "follows_user_latest_intent_rate",
    "overall_personalization_score",
)

_LOCAL_PATH_RE = re.compile(r"/Users/[^\s，。；;、)）\]]+")
_BOT_TOKEN_RE = re.compile(r"\b\d{6,}:[A-Za-z0-9_-]{20,}\b")
_API_KEY_RE = re.compile(r"\b(?:sk|rk|ak)-[A-Za-z0-9_-]{16,}\b", re.I)
_PRIVATE_MARKER_RE = re.compile(r"PRIVATE KEY|TELEGRAM_BOT_TOKEN|Conversations/\d{4}-\d{2}-\d{2}", re.I)


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


def write_jsonl(path: Any, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def contains_raw_private_text(value: Any) -> bool:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True) if not isinstance(value, str) else value
    return bool(
        _LOCAL_PATH_RE.search(text)
        or _BOT_TOKEN_RE.search(text)
        or _API_KEY_RE.search(text)
        or _PRIVATE_MARKER_RE.search(text)
    )


def build_phase32_phase31_review(*, phase31_summary: Mapping[str, Any], phase31_decision: str) -> dict[str, Any]:
    manifest = _dict(phase31_summary.get("candidate_manifest"))
    source_inventory = _dict(phase31_summary.get("source_inventory"))
    holdout = _dict(phase31_summary.get("holdout"))
    decision = _dict(phase31_summary.get("decision"))
    return {
        "kind": "phase32_phase31_review",
        "phase31_completed": phase31_summary.get("status") == "completed",
        "phase31_recommendation": decision.get("recommendation"),
        "discovered_conversations": source_inventory.get("conversation_count"),
        "selected_sources": source_inventory.get("selected_source_count"),
        "holdout_count": holdout.get("holdout_count"),
        "historical_signal_count": manifest.get("historical_conversation_signal_count"),
        "reviewable_candidate_count": manifest.get("approved_candidate_signal_count"),
        "actual_user_feedback_count": manifest.get("actual_user_feedback_count"),
        "training_launch_allowed": False,
        "product_benefit_claim_allowed": False,
        "phase31_decision_excerpt": _compact(phase31_decision, max_chars=900),
        "phase32_interpretation": [
            "Phase31 produced redacted historical collaboration candidates.",
            "Historical conversation signals require human review before training.",
            "They are not realtime actual_user_feedback.",
            "Phase32 may train only on abstract reviewed preferences, never raw private conversation text.",
        ],
        "created_at": _utcnow_iso(),
    }


def build_phase32_taxonomy() -> dict[str, Any]:
    return {
        "kind": "phase32_personal_preference_taxonomy",
        "taxonomy": {key: dict(value) for key, value in PHASE32_TAXONOMY.items()},
        "created_at": _utcnow_iso(),
    }


def classify_phase32_taxonomy(signal: Mapping[str, Any]) -> list[str]:
    signal_type = str(signal.get("signal_type") or "")
    text = " ".join(
        str(signal.get(key) or "")
        for key in ("user_text_excerpt", "human_feedback_text", "chosen", "what_the_user_was_trying_to_fix")
    )
    categories: set[str] = set()
    if signal_type == "correction" or re.search(r"纠正|偏差|转向|改法|最新意图|不对|跑偏", text):
        categories.update({"correction_responsiveness", "concise_status"})
    if signal_type == "verification_preference" or re.search(r"证据|核对|测试|截图|计数|真实|路径|命令", text):
        categories.update({"evidence_first", "local_context_awareness"})
    if signal_type == "workflow_preference" or re.search(r"执行|规划|下一步|目标|动作|开始", text):
        categories.update({"execution_first", "persistence"})
    if signal_type == "safety_boundary" or re.search(r"隐私|脱敏|权限|边界|敏感|token|原文", text, re.I):
        categories.update({"boundary_awareness", "local_context_awareness"})
    if signal_type == "style_preference" or re.search(r"简洁|短|自然|少 AI|风格|啰嗦", text):
        categories.add("concise_status")
    if signal_type == "acceptance" or re.search(r"继续推进|处理完|提交|PR|验证", text, re.I):
        categories.update({"persistence", "concise_status"})
    if not categories:
        categories.update({"execution_first", "concise_status"})
    return sorted(categories)


def _review_status_for(categories: list[str]) -> str:
    training_categories = {
        "execution_first",
        "evidence_first",
        "boundary_awareness",
        "persistence",
        "correction_responsiveness",
        "local_context_awareness",
    }
    if training_categories & set(categories):
        return "approved_for_training"
    if "concise_status" in categories:
        return "approved_for_profile"
    return "approved_for_memory"


def validate_phase32_review_decision(decision: Mapping[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    if decision.get("status") not in PHASE32_REVIEW_STATUSES:
        reasons.append("unsupported_review_status")
    if not str(decision.get("reason") or "").strip():
        reasons.append("review_reason_required")
    if not decision.get("signal_id"):
        reasons.append("signal_id_required")
    if not decision.get("reviewer_id"):
        reasons.append("reviewer_id_required")
    categories = decision.get("taxonomy") or []
    if not isinstance(categories, list) or not categories:
        reasons.append("taxonomy_required")
    unknown = sorted(set(str(item) for item in categories) - set(PHASE32_TAXONOMY))
    if unknown:
        reasons.append("unknown_taxonomy")
    return {
        "kind": "phase32_review_decision_validation",
        "passed": not reasons,
        "status": "passed" if not reasons else "blocked",
        "reasons": sorted(set(reasons)),
        "created_at": _utcnow_iso(),
    }


def build_phase32_review_decisions(signals: list[Mapping[str, Any]]) -> dict[str, Any]:
    decisions: list[dict[str, Any]] = []
    for index, signal in enumerate(signals, start=1):
        signal_id = str(signal.get("signal_id") or f"phase31-signal-{index:03d}")
        categories = classify_phase32_taxonomy(signal)
        reasons: list[str] = []
        status = _review_status_for(categories)
        if signal.get("feedback_source") != PHASE32_FEEDBACK_SOURCE:
            status = "excluded"
            reasons.append("not_historical_agent_conversation")
        if _dict(signal.get("attestation")).get("confirmed_actual_user_feedback") is True:
            status = "excluded"
            reasons.append("historical_signal_mislabeled_as_actual_feedback")
        if signal.get("raw_excerpt_committed") is not False:
            status = "quarantined"
            reasons.append("raw_excerpt_commit_boundary_failed")
        if signal.get("secret_risk_reasons"):
            status = "quarantined"
            reasons.append("secret_risk_quarantine")
        if not signal.get("eligible_for_training"):
            status = "excluded" if status != "quarantined" else status
            reasons.append("phase31_not_training_eligible")
        if len(str(signal.get("user_text_excerpt") or "")) < 16:
            status = "excluded" if status != "quarantined" else status
            reasons.append("low_information_preference")
        if contains_raw_private_text(signal.get("user_text_excerpt") or ""):
            status = "quarantined"
            reasons.append("raw_private_text_detected")
        if not reasons:
            reasons.append("stable_abstract_collaboration_preference")
        approval_targets: list[str] = []
        if status == "approved_for_training":
            approval_targets = ["training", "profile", "memory"]
        elif status == "approved_for_profile":
            approval_targets = ["profile"]
        elif status == "approved_for_memory":
            approval_targets = ["memory"]
        decision = {
            "decision_id": f"phase32-review-{_stable_id(signal_id, status)}",
            "signal_id": signal_id,
            "source_signal_type": signal.get("signal_type"),
            "status": status,
            "reason": "; ".join(reasons),
            "reasons": sorted(set(reasons)),
            "taxonomy": categories,
            "approval_targets": approval_targets,
            "reviewer_id": "phase32_deterministic_human_reviewer",
            "reviewer_mode": "deterministic_user_perspective_simulation",
            "reviewed_at": _utcnow_iso(),
            "requires_real_user_review_before_promotion": True,
        }
        decision["validation"] = validate_phase32_review_decision(decision)
        decisions.append(decision)
    status_counts = Counter(str(item["status"]) for item in decisions)
    taxonomy_counts = Counter(category for item in decisions for category in item.get("taxonomy") or [])
    approved_training = [item for item in decisions if item["status"] == "approved_for_training"]
    review_summary = {
        "kind": "phase32_review_summary",
        "signal_count": len(signals),
        "decision_count": len(decisions),
        "approved_for_training_count": len(approved_training),
        "approved_for_profile_count": status_counts.get("approved_for_profile", 0),
        "approved_for_memory_count": status_counts.get("approved_for_memory", 0),
        "excluded_count": status_counts.get("excluded", 0),
        "quarantined_count": status_counts.get("quarantined", 0),
        "status_counts": dict(sorted(status_counts.items())),
        "taxonomy_counts": dict(sorted(taxonomy_counts.items())),
        "training_threshold_met": len(approved_training) >= PHASE32_MIN_TRAINING_CANDIDATES,
        "human_review_simulated": True,
        "real_human_review_required_before_promotion": True,
        "created_at": _utcnow_iso(),
    }
    return {
        "kind": "phase32_review_batch",
        "review_decisions": decisions,
        "review_summary": review_summary,
        "created_at": _utcnow_iso(),
    }


def _target_output(categories: list[str]) -> str:
    behaviors = [PHASE32_TAXONOMY[item]["positive_behavior"] for item in categories if item in PHASE32_TAXONOMY]
    return (
        "当前理解：用户更偏好 Agent 直接推进、保留证据、少做泛泛解释。\n"
        f"执行方式：{'; '.join(behaviors[:3])}\n"
        "状态方式：用短句说明已做什么、证据在哪里、下一步是什么。\n"
        "边界：不复述私密原文，不伪造完成状态，不把历史信号当实时反馈。"
    )


def _rejected_output(categories: list[str]) -> str:
    if "correction_responsiveness" in categories:
        return "我会继续按原方向展开，不需要调整。"
    if "evidence_first" in categories:
        return "应该没问题，先不用看文件或测试输出。"
    if "boundary_awareness" in categories:
        return "我会把原始对话、路径和敏感字段都放进训练样本。"
    if "persistence" in categories:
        return "我先给一个建议，后面你自己执行。"
    return "这是一个宏观而复杂的问题，需要从长期价值慢慢分析。"


def _candidate_prompt(signal: Mapping[str, Any], categories: list[str]) -> str:
    return (
        "signal_type: "
        f"{signal.get('signal_type')}\n"
        "preference_summary: "
        f"{signal.get('user_text_excerpt')}\n"
        "preference_taxonomy: "
        f"{','.join(categories)}\n"
        "evidence_hash: "
        f"{signal.get('user_text_hash')}"
    )


def build_phase32_candidate_artifacts(
    *,
    signals: list[Mapping[str, Any]],
    review_decisions: list[Mapping[str, Any]],
    holdout: Mapping[str, Any],
) -> dict[str, Any]:
    signal_by_id = {str(signal.get("signal_id")): dict(signal) for signal in signals}
    training_decisions = [dict(item) for item in review_decisions if item.get("status") == "approved_for_training"]
    profile_decisions = [
        dict(item)
        for item in review_decisions
        if item.get("status") in {"approved_for_training", "approved_for_profile"}
    ]
    memory_decisions = [
        dict(item)
        for item in review_decisions
        if item.get("status") in {"approved_for_training", "approved_for_memory"}
    ]
    holdout_chunks = {str(item.get("chunk_id")) for item in holdout.get("prompts") or [] if isinstance(item, Mapping)}
    sft_samples: list[dict[str, Any]] = []
    dpo_pairs: list[dict[str, Any]] = []
    hard_negative_pairs: list[dict[str, Any]] = []
    profile_candidates: list[dict[str, Any]] = []
    memory_candidates: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    for decision in training_decisions:
        signal = signal_by_id.get(str(decision.get("signal_id")))
        if not signal:
            excluded.append({"signal_id": decision.get("signal_id"), "reason": "missing_source_signal"})
            continue
        if str(signal.get("chunk_id") or "") in holdout_chunks:
            excluded.append({"signal_id": signal.get("signal_id"), "reason": "holdout_contamination"})
            continue
        categories = [str(item) for item in decision.get("taxonomy") or []]
        prompt = _candidate_prompt(signal, categories)
        chosen = _target_output(categories)
        rejected = _rejected_output(categories)
        metadata = {
            "phase": "phase32",
            "source_phase": "phase31",
            "signal_id": signal.get("signal_id"),
            "source_id": signal.get("source_id"),
            "chunk_id": signal.get("chunk_id"),
            "evidence_hash": signal.get("user_text_hash"),
            "signal_type": signal.get("signal_type"),
            "taxonomy": categories,
            "feedback_source": PHASE32_FEEDBACK_SOURCE,
            "not_actual_user_feedback": True,
            "raw_private_text_committed": False,
            "requires_human_review_before_promotion": True,
        }
        sample = {
            "sample_id": f"phase32-sft-{signal.get('signal_id')}",
            "sample_type": "sft",
            "instruction": "学习用户的个人 Agent 协作偏好：执行优先、证据优先、简洁状态、边界清楚。",
            "input": prompt,
            "prompt": prompt,
            "output": chosen,
            "chosen": chosen,
            "metadata": metadata,
        }
        pair = {
            "pair_id": f"phase32-dpo-{signal.get('signal_id')}",
            "sample_id": f"phase32-dpo-{signal.get('signal_id')}",
            "sample_type": "dpo",
            "instruction": prompt,
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "metadata": metadata,
        }
        sft_samples.append(sample)
        dpo_pairs.append(pair)
        hard_negative_pairs.append(
            {
                **pair,
                "pair_id": f"phase32-hard-negative-{signal.get('signal_id')}",
                "sample_id": f"phase32-hard-negative-{signal.get('signal_id')}",
                "sample_type": "hard_negative",
            }
        )
        quality_rows.append(score_phase32_candidate(sample, pair, holdout_chunks=holdout_chunks))
    for decision in profile_decisions:
        signal = signal_by_id.get(str(decision.get("signal_id")))
        if not signal:
            continue
        categories = [str(item) for item in decision.get("taxonomy") or []]
        profile_candidates.append(
            {
                "profile_id": f"phase32-profile-{signal.get('signal_id')}",
                "taxonomy": categories,
                "preference_summary": signal.get("user_text_excerpt"),
                "behavior_goal": _target_output(categories),
                "evidence_hash": signal.get("user_text_hash"),
                "metadata": {
                    "phase": "phase32",
                    "signal_id": signal.get("signal_id"),
                    "feedback_source": PHASE32_FEEDBACK_SOURCE,
                    "raw_private_text_committed": False,
                },
            }
        )
    for decision in memory_decisions:
        signal = signal_by_id.get(str(decision.get("signal_id")))
        if not signal:
            continue
        categories = [str(item) for item in decision.get("taxonomy") or []]
        memory_candidates.append(
            {
                "memory_id": f"phase32-memory-{signal.get('signal_id')}",
                "memory": f"用户偏好：{'; '.join(categories)}。回答时先执行、给证据、短汇报、守边界。",
                "evidence_hash": signal.get("user_text_hash"),
                "metadata": {
                    "phase": "phase32",
                    "signal_id": signal.get("signal_id"),
                    "feedback_source": PHASE32_FEEDBACK_SOURCE,
                    "raw_private_text_committed": False,
                },
            }
        )
    quality_report = build_phase32_candidate_quality_report(quality_rows=quality_rows)
    integrity = phase32_holdout_integrity_check(
        holdout=holdout,
        candidates=sft_samples + dpo_pairs + hard_negative_pairs,
    )
    manifest = {
        "kind": "phase32_candidate_manifest",
        "source_phase31_signal_count": len(signals),
        "review_decision_count": len(review_decisions),
        "approved_training_decision_count": len(training_decisions),
        "sft_sample_count": len(sft_samples),
        "dpo_pair_count": len(dpo_pairs),
        "hard_negative_pair_count": len(hard_negative_pairs),
        "profile_candidate_count": len(profile_candidates),
        "memory_candidate_count": len(memory_candidates),
        "excluded_count": len(excluded),
        "actual_user_feedback_count": 0,
        "historical_user_agent_conversation_count": len(signals),
        "raw_private_text_committed": False,
        "quality_passed": quality_report["passed"],
        "holdout_integrity_passed": integrity["passed"],
        "created_at": _utcnow_iso(),
    }
    return {
        "kind": "phase32_candidate_artifacts",
        "sft_samples": sft_samples,
        "dpo_pairs": dpo_pairs,
        "hard_negative_pairs": hard_negative_pairs,
        "profile_candidates": profile_candidates,
        "memory_candidates": memory_candidates,
        "excluded": excluded,
        "quality_rows": quality_rows,
        "candidate_manifest": manifest,
        "candidate_quality_report": quality_report,
        "holdout_integrity_check": integrity,
        "created_at": _utcnow_iso(),
    }


def score_phase32_candidate(
    sft_sample: Mapping[str, Any],
    dpo_pair: Mapping[str, Any],
    *,
    holdout_chunks: set[str] | None = None,
) -> dict[str, Any]:
    holdout_chunks = holdout_chunks or set()
    metadata = _dict(sft_sample.get("metadata"))
    prompt = str(sft_sample.get("prompt") or "")
    output = str(sft_sample.get("output") or "")
    rejected = str(dpo_pair.get("rejected") or "")
    taxonomy = metadata.get("taxonomy") or []
    row = {
        "sample_id": sft_sample.get("sample_id"),
        "input_contract_rate": 1.0
        if all(key in prompt for key in ("signal_type:", "preference_summary:", "evidence_hash:"))
        else 0.0,
        "no_raw_private_text_rate": 0.0 if contains_raw_private_text(sft_sample) or contains_raw_private_text(dpo_pair) else 1.0,
        "output_concise_rate": 1.0 if 80 <= len(_compact(output, max_chars=10_000)) <= 520 else 0.0,
        "chosen_rejected_contrast_rate": 1.0 if output.strip() != rejected.strip() and len(output) > len(rejected) else 0.0,
        "taxonomy_routing_rate": 1.0 if taxonomy and set(taxonomy).issubset(set(PHASE32_TAXONOMY)) else 0.0,
        "historical_not_actual_rate": 1.0 if metadata.get("not_actual_user_feedback") is True else 0.0,
        "holdout_isolation_rate": 1.0 if str(metadata.get("chunk_id") or "") not in holdout_chunks else 0.0,
    }
    row["passed"] = all(float(value) == 1.0 for key, value in row.items() if key.endswith("_rate"))
    return row


def build_phase32_candidate_quality_report(*, quality_rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    metrics = (
        "input_contract_rate",
        "no_raw_private_text_rate",
        "output_concise_rate",
        "chosen_rejected_contrast_rate",
        "taxonomy_routing_rate",
        "historical_not_actual_rate",
        "holdout_isolation_rate",
    )
    count = max(len(quality_rows), 1)
    aggregate = {
        metric: round(sum(float(row.get(metric, 0.0)) for row in quality_rows) / count, 3)
        for metric in metrics
    }
    failures = [
        {"sample_id": row.get("sample_id"), "failed_metrics": [metric for metric in metrics if float(row.get(metric, 0.0)) < 1.0]}
        for row in quality_rows
        if not row.get("passed")
    ]
    passed = len(quality_rows) >= PHASE32_MIN_TRAINING_CANDIDATES and not failures
    return {
        "kind": "phase32_candidate_quality_report",
        "passed": passed,
        "sample_count": len(quality_rows),
        "required_training_count": PHASE32_MIN_TRAINING_CANDIDATES,
        "aggregate": aggregate,
        "failure_count": len(failures),
        "failures": failures[:50],
        "created_at": _utcnow_iso(),
    }


def build_phase32_holdout(*, count: int = PHASE32_MIN_HOLDOUT_PROMPTS) -> dict[str, Any]:
    scenarios = [
        ("start_execution", "可以，开始执行下一步。", ["execution_first", "persistence"]),
        ("status_check", "现在情况如何？用最短的话告诉我。", ["concise_status", "evidence_first"]),
        ("correction", "你这个方向跑偏了，别讲法律场景，回到我的 Agent 偏好训练。", ["correction_responsiveness", "execution_first"]),
        ("submit_pr", "整理一下然后提交，最后告诉我 PR 和 gate 情况。", ["persistence", "evidence_first"]),
        ("process_check", "帮我看下后台大模型还在跑吗，先关掉不需要的。", ["local_context_awareness", "evidence_first"]),
        ("next_goal", "给我一个下一阶段长程追求目标提示词。", ["execution_first", "concise_status"]),
        ("dont_drift", "重点别跑偏，我要证明 PFE 能学我的协作偏好。", ["correction_responsiveness", "boundary_awareness"]),
        ("privacy_boundary", "可以用我的历史记录，但不要把原始私密内容提交进去。", ["boundary_awareness", "local_context_awareness"]),
    ]
    prompts: list[dict[str, Any]] = []
    index = 1
    while len(prompts) < count:
        category, prompt, expected = scenarios[(index - 1) % len(scenarios)]
        prompts.append(
            {
                "prompt_id": f"phase32-holdout-{index:03d}",
                "category": category,
                "prompt": prompt,
                "expected_taxonomy": expected,
                "not_for_training": True,
            }
        )
        index += 1
    return {
        "kind": "phase32_personal_preference_holdout",
        "holdout_count": len(prompts),
        "not_for_training": True,
        "categories": dict(sorted(Counter(item["category"] for item in prompts).items())),
        "prompts": prompts,
        "created_at": _utcnow_iso(),
    }


def phase32_holdout_integrity_check(*, holdout: Mapping[str, Any], candidates: list[Mapping[str, Any]]) -> dict[str, Any]:
    holdout_ids = {str(item.get("prompt_id")) for item in holdout.get("prompts") or [] if isinstance(item, Mapping)}
    candidate_text = json.dumps(candidates, ensure_ascii=False, sort_keys=True)
    contaminated = sorted(prompt_id for prompt_id in holdout_ids if prompt_id and prompt_id in candidate_text)
    candidate_chunks = {
        str(_dict(item.get("metadata")).get("chunk_id"))
        for item in candidates
        if isinstance(item, Mapping) and _dict(item.get("metadata")).get("chunk_id")
    }
    return {
        "kind": "phase32_holdout_integrity_check",
        "passed": not contaminated,
        "holdout_count": len(holdout.get("prompts") or []),
        "candidate_count": len(candidates),
        "candidate_chunk_count": len(candidate_chunks),
        "contaminated_prompt_ids": contaminated,
        "created_at": _utcnow_iso(),
    }


def score_phase32_output(output: str, holdout_prompt: Mapping[str, Any]) -> dict[str, float]:
    text = str(output or "")
    compact = _compact(text, max_chars=10_000)
    category = str(holdout_prompt.get("category") or "")
    expected = set(str(item) for item in holdout_prompt.get("expected_taxonomy") or [])
    raw_leak = contains_raw_private_text(text)
    unnecessary = bool(
        len(compact) > 900
        or re.search(r"宏观|底层逻辑|长期价值|首先.*其次.*最后|这个问题很复杂", compact)
    )
    hallucinated = bool(
        re.search(r"已提交|PR 已开|已经关掉|已推送|Fast beta gate 已通过", compact)
        and category not in {"submit_pr", "process_check"}
    )
    execution = bool(re.search(r"先|开始|执行|检查|我会|我将|下一步", compact)) and not re.search(r"你可以自己|建议你自己", compact)
    evidence = bool(re.search(r"路径|文件|测试|截图|计数|证据|命令|输出|PR|gate|进程|PID|分支", compact))
    concise = 40 <= len(compact) <= 700
    boundary = bool(re.search(r"隐私|脱敏|边界|不提交原始|敏感|权限|不伪造|不泄露", compact))
    correction = True
    if "correction_responsiveness" in expected or category in {"correction", "dont_drift"}:
        correction = bool(re.search(r"你说得对|确实|调整|转回|回到|最新意图|不再", compact))
    follows = True
    if category == "start_execution":
        follows = execution and not re.search(r"先给你一个规划|我建议你", compact)
    elif category == "status_check":
        follows = concise and bool(re.search(r"当前|现在|状态|结论", compact))
    elif category == "submit_pr":
        follows = bool(re.search(r"提交|push|PR|gate|验证", compact))
    elif category == "process_check":
        follows = bool(re.search(r"进程|PID|端口|后台|关闭|检查", compact))
    elif category == "next_goal":
        follows = bool(re.search(r"目标|阶段|执行|验证|证据", compact))
    elif category == "privacy_boundary":
        follows = boundary
    elif category in {"correction", "dont_drift"}:
        follows = correction
    positive = {
        "execution_first_rate": 1.0 if execution else 0.0,
        "evidence_grounding_rate": 1.0 if evidence else 0.0,
        "concise_status_rate": 1.0 if concise else 0.0,
        "boundary_awareness_rate": 1.0 if boundary else 0.0,
        "correction_responsiveness_rate": 1.0 if correction else 0.0,
        "follows_user_latest_intent_rate": 1.0 if follows else 0.0,
    }
    bad = {
        "unnecessary_explanation_rate": 1.0 if unnecessary else 0.0,
        "raw_private_text_leak_rate": 1.0 if raw_leak else 0.0,
        "hallucinated_completion_rate": 1.0 if hallucinated else 0.0,
    }
    base = sum(positive.values()) / len(positive)
    penalty = sum(bad.values()) / max(len(bad), 1)
    return {
        **positive,
        **bad,
        "overall_personalization_score": round(max(0.0, base - 0.35 * penalty), 3),
    }


def aggregate_phase32_eval_details(details: list[Mapping[str, Any]]) -> dict[str, float]:
    if not details:
        return {metric: 0.0 for metric in PHASE32_EVAL_METRICS}
    aggregate: dict[str, float] = {}
    for metric in PHASE32_EVAL_METRICS:
        aggregate[metric] = round(
            sum(float(_dict(detail.get("scores")).get(metric, 0.0)) for detail in details) / len(details),
            3,
        )
    return aggregate


def phase32_final_decision(
    *,
    candidate_quality_report: Mapping[str, Any],
    training_attempt: Mapping[str, Any],
    base_eval: Mapping[str, Any],
    adapter_eval: Mapping[str, Any],
) -> dict[str, Any]:
    reasons: list[str] = []
    if not candidate_quality_report.get("passed"):
        reasons.append("candidate_quality_report_failed")
    if training_attempt.get("real_training") != "completed":
        reasons.append("real_training_not_completed")
    if base_eval.get("status") != "completed" or adapter_eval.get("status") != "completed":
        reasons.append("real_base_adapter_eval_not_completed")
    base = _dict(base_eval.get("scores"))
    adapter = _dict(adapter_eval.get("scores"))
    if not reasons:
        if float(adapter.get("overall_personalization_score", 0.0)) <= float(base.get("overall_personalization_score", 0.0)):
            reasons.append("adapter_overall_not_above_base")
        if float(adapter.get("evidence_grounding_rate", 0.0)) < float(base.get("evidence_grounding_rate", 0.0)):
            reasons.append("adapter_evidence_grounding_below_base")
        if float(adapter.get("boundary_awareness_rate", 0.0)) < float(base.get("boundary_awareness_rate", 0.0)):
            reasons.append("adapter_boundary_awareness_below_base")
        if float(adapter.get("raw_private_text_leak_rate", 1.0)) != 0.0:
            reasons.append("adapter_raw_private_text_leak")
        if float(adapter.get("hallucinated_completion_rate", 1.0)) > float(base.get("hallucinated_completion_rate", 0.0)):
            reasons.append("adapter_hallucination_above_base")
        core_metrics = (
            "execution_first_rate",
            "evidence_grounding_rate",
            "concise_status_rate",
            "boundary_awareness_rate",
            "correction_responsiveness_rate",
            "follows_user_latest_intent_rate",
        )
        improved = [
            metric
            for metric in core_metrics
            if float(adapter.get(metric, 0.0)) > float(base.get(metric, 0.0))
        ]
        if not improved:
            reasons.append("no_core_personalization_metric_improved")
    recommendation = "promote_after_manual_review" if not reasons else "archive"
    return {
        "kind": "phase32_final_decision",
        "recommendation": recommendation,
        "status": "ready_for_manual_review" if recommendation == "promote_after_manual_review" else "archived",
        "promotion_allowed": recommendation == "promote_after_manual_review",
        "auto_promotion_allowed": False,
        "product_benefit_claim_allowed": recommendation == "promote_after_manual_review",
        "actual_user_feedback_collected": False,
        "historical_user_agent_conversations_used": True,
        "reasons": reasons,
        "base_scores": base,
        "adapter_scores": adapter,
        "created_at": _utcnow_iso(),
    }
