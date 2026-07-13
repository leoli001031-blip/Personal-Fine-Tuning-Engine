"""Phase47 simulated-user review primitives for Phase46 correction candidates."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import re
from typing import Any, Iterable, Mapping


PHASE47_KIND = "phase47_simulated_user_review"
PHASE47_REVIEWER_ID = "phase47-codex-simulated-user"
PHASE47_EXPECTED_CANDIDATE_COUNT = 48

_RAW_PRIVATE_RE = re.compile(
    r"\bsk-[A-Za-z0-9_-]{16,}\b|\b\d{8,12}:[A-Za-z0-9_-]{24,}\b|\bPFE\d+_(?:HOLDOUT_)?SECRET_[A-Za-z0-9_]+\b",
    flags=re.IGNORECASE,
)
_PLACEHOLDER_RE = re.compile(r"<\s*PRIVATE_[A-Z0-9_ -]+>", flags=re.IGNORECASE)


_EDIT_OVERRIDES: dict[str, tuple[str, str]] = {
    "phase46-curated-001": (
        "我只检查 8927 是否监听；在拿到当前 `lsof` 输出前，状态是未确认。",
        "删掉重复说明，直接给当前状态边界和唯一核验动作。",
    ),
    "phase46-curated-002": (
        "我只核验 adapter 文件是否存在；未读取目录前，结果是未确认。",
        "用户只要求核验文件存在性，不额外扩大到校验结果。",
    ),
    "phase46-curated-003": (
        "我只读取当前 Git 分支，不比较页面，也不启动 Studio。",
        "改成更直接的当前动作，并明确两个旧任务均已停止。",
    ),
    "phase46-curated-004": (
        "目前没有验收结果可核对。待确认项是测试结果、签名状态和人工验收记录。",
        "原目标把缺失项写得过于确定，修订后先交代证据现状。",
    ),
    "phase46-curated-005": (
        "我只报告实际目录中的 shard 数量；未读取目录前无法给出数量。",
        "去掉过程性措辞，直接回答为什么当前不能报告数量。",
    ),
    "phase46-curated-006": (
        "我只运行 `git status --short` 检查工作区，不修改文件或提交。",
        "把核验动作放在前面，使执行边界更清楚。",
    ),
    "phase46-curated-008": (
        "会议通知：下周二下午 3 点开会，请准时参加。",
        "保留用户原来的相对时间表达，通知读起来更自然。",
    ),
    "phase46-curated-011": (
        "谢谢你帮我一起排查这个问题。",
        "原文暗示排查已完成，修订后只表达感谢，不补写状态。",
    ),
    "phase46-curated-021": (
        "当前只完成 70/80，不能按 80 条汇总；保留已有结果并续跑剩余 10 条。",
        "去掉不必要的英文状态词，直接说明完成度和恢复动作。",
    ),
    "phase46-curated-025": (
        "本地分支领先远端 1 个 commit，说明这条提交尚未 push。",
        "已有 ahead 证据足以判断未推送，不应继续写成无法确认。",
    ),
    "phase46-curated-031": (
        "不能。模拟回放只说明实验条件下的表现，真实用户收益仍需实际反馈验证。",
        "保留来源边界，同时缩短成用户容易理解的结论。",
    ),
    "phase46-curated-039": (
        "持久化日志不应包含凭证正文；审计只保留凭证类型、hash 和出现次数。",
        "直接回答可保存的证据边界，删除泛化的命令语气。",
    ),
    "phase46-curated-042": (
        "记录一次预清洗泄漏尝试和输出 hash；只持久化脱敏结果，不保存原始输出。",
        "压缩表达，并把不得保存原始输出写成明确边界。",
    ),
}


_ACCEPT_REASONS = {
    "latest_goal_replacement": "回答已停止冲突旧目标，只处理最新请求，状态边界明确。",
    "ordinary_direct_task": "回答自然直接，没有把普通任务强行套入工程流程。",
    "evidence_status": "回答区分了已有证据与当前状态，没有补写看不到的结论。",
    "failure_truthfulness": "回答没有把部分进度包装成成功，并给出可执行恢复动作。",
    "git_process_truthfulness": "回答只依据当前 Git、进程或端口证据，不虚构执行完成。",
    "provenance_boundary": "回答清楚区分模拟证据、真实反馈和产品收益。",
    "privacy_non_echo": "回答保留审计所需类型和计数，同时不复述私密值或占位符。",
    "nonsecret_preservation": "回答保留公开标识，没有因为隐私规则而过度删除。",
}


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalized(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).lower()


def _sentence_count(value: str) -> int:
    return len([part for part in re.split(r"[。！？!?]+", str(value).strip()) if part.strip()])


def _sha256_text(value: Any) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()


def _required_fields_present(candidate: Mapping[str, Any]) -> bool:
    messages = candidate.get("messages") or []
    return (
        bool(candidate.get("pair_id"))
        and bool(candidate.get("category"))
        and bool(candidate.get("chosen"))
        and bool(candidate.get("rejected"))
        and isinstance(messages, list)
        and len(messages) == 3
        and [str(row.get("role") or "") for row in messages if isinstance(row, Mapping)] == ["user", "assistant", "user"]
    )


def _target_checks(text: str) -> dict[str, bool]:
    normalized = _normalized(text)
    return {
        "non_empty": bool(normalized),
        "concise": len(text) <= 160 and 1 <= _sentence_count(text) <= 3,
        "no_raw_private_value": not bool(_RAW_PRIVATE_RE.search(text)),
        "no_internal_placeholder": not bool(_PLACEHOLDER_RE.search(text)),
        "no_false_completion_phrase": not bool(re.search(r"已经完成|已成功|可以直接发布|可以合并", text)),
    }


def review_phase47_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    pair_id = str(candidate.get("pair_id") or "")
    category = str(candidate.get("category") or "")
    original = str(candidate.get("chosen") or "").strip()
    if not _required_fields_present(candidate):
        decision = "reject"
        reviewed = ""
        reason = "候选缺少三轮边界、chosen/rejected 或稳定标识，无法形成可审计偏好。"
    elif pair_id in _EDIT_OVERRIDES:
        decision = "edit"
        reviewed, reason = _EDIT_OVERRIDES[pair_id]
    else:
        decision = "accept"
        reviewed = original
        reason = _ACCEPT_REASONS.get(category, "回答满足最新用户要求且没有明显真实性或隐私问题。")
    checks = _target_checks(reviewed) if reviewed else {
        "non_empty": False,
        "concise": False,
        "no_raw_private_value": True,
        "no_internal_placeholder": True,
        "no_false_completion_phrase": True,
    }
    if decision != "reject" and not all(checks.values()):
        decision = "reject"
        reason = "模拟审核后的目标仍未通过简洁性、隐私或虚假完成边界。"
        reviewed = ""
    return {
        "kind": "phase47_simulated_user_review_decision",
        "decision_id": f"phase47-review-{pair_id.removeprefix('phase46-curated-') or 'invalid'}",
        "pair_id": pair_id,
        "sample_id": candidate.get("sample_id"),
        "category": category,
        "decision": decision,
        "reason": reason,
        "original_chosen": original or None,
        "original_chosen_sha256": _sha256_text(original),
        "reviewed_chosen": reviewed or None,
        "reviewed_chosen_sha256": _sha256_text(reviewed) if reviewed else None,
        "changed": decision == "edit",
        "review_checks": checks,
        "would_keep_response": decision in {"accept", "edit"},
        "reviewer_id": PHASE47_REVIEWER_ID,
        "reviewer_mode": "codex_simulated_real_user_perspective",
        "feedback_source": "simulated_usage",
        "simulated_usage": True,
        "simulated_human_review": True,
        "actual_human_review": False,
        "actual_user_feedback": False,
        "confirmed_actual_user_feedback": False,
        "actual_product_benefit_claim_allowed": False,
        "eligible_for_simulated_lab_candidate": decision in {"accept", "edit"},
        "eligible_for_production_training": False,
        "auto_training_allowed": False,
        "auto_promotion_allowed": False,
        "required_human_action": "actual_user_accept_edit_or_reject",
        "created_at": _utcnow(),
    }


def build_phase47_simulated_review(candidates: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [dict(row) for row in candidates]
    decisions = [review_phase47_candidate(row) for row in rows]
    counts = Counter(str(row.get("decision") or "unknown") for row in decisions)
    return {
        "kind": "phase47_simulated_user_review_batch",
        "source_candidate_count": len(rows),
        "reviewed_count": len(decisions),
        "decision_counts": dict(sorted(counts.items())),
        "edit_rate": round(counts["edit"] / len(decisions), 4) if decisions else 0.0,
        "reviewer_id": PHASE47_REVIEWER_ID,
        "reviewer_mode": "codex_simulated_real_user_perspective",
        "decisions": decisions,
        "simulated_usage": True,
        "actual_human_review_count": 0,
        "actual_user_feedback_count": 0,
        "actual_product_benefit_claim_allowed": False,
        "eligible_for_production_training": False,
        "auto_training_allowed": False,
        "auto_promotion_allowed": False,
        "created_at": _utcnow(),
    }


def build_phase47_reviewed_candidates(
    candidates: Iterable[Mapping[str, Any]],
    decisions: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    decision_by_id = {str(row.get("pair_id")): dict(row) for row in decisions}
    reviewed: list[dict[str, Any]] = []
    for source in candidates:
        row = dict(source)
        decision = decision_by_id.get(str(row.get("pair_id") or ""), {})
        if decision.get("decision") not in {"accept", "edit"}:
            continue
        row["chosen"] = str(decision.get("reviewed_chosen") or "")
        row.update(
            {
                "kind": "phase47_simulated_reviewed_preference_candidate",
                "phase47_review_decision": decision.get("decision"),
                "phase47_review_decision_id": decision.get("decision_id"),
                "reviewer_id": PHASE47_REVIEWER_ID,
                "reviewer_mode": "codex_simulated_real_user_perspective",
                "review_status": "simulated_review_only",
                "feedback_source": "simulated_usage",
                "simulated_usage": True,
                "simulated_human_review": True,
                "actual_human_review": False,
                "actual_user_feedback": False,
                "confirmed_actual_user_feedback": False,
                "actual_product_benefit_claim_allowed": False,
                "eligible_for_simulated_lab_candidate": True,
                "eligible_for_training": False,
                "eligible_for_production_training": False,
                "manual_user_review_required": True,
                "training_blocker": "pending_actual_human_confirmation",
                "auto_training_allowed": False,
                "auto_promotion_allowed": False,
            }
        )
        reviewed.append(row)
    return reviewed


def audit_phase47_review(
    *,
    source_candidates: Iterable[Mapping[str, Any]],
    decisions: Iterable[Mapping[str, Any]],
    reviewed_candidates: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    source = [dict(row) for row in source_candidates]
    review = [dict(row) for row in decisions]
    selected = [dict(row) for row in reviewed_candidates]
    source_ids = [str(row.get("pair_id") or "") for row in source]
    review_ids = [str(row.get("pair_id") or "") for row in review]
    selected_ids = [str(row.get("pair_id") or "") for row in selected]
    counts = Counter(str(row.get("decision") or "unknown") for row in review)
    categories = Counter(str(row.get("category") or "unknown") for row in review)
    target_problems = [
        str(row.get("pair_id") or "")
        for row in selected
        if not all(_target_checks(str(row.get("chosen") or "")).values())
    ]
    provenance_problems = [
        str(row.get("pair_id") or "")
        for row in selected
        if row.get("actual_human_review") is not False
        or row.get("actual_user_feedback") is not False
        or row.get("eligible_for_training") is not False
        or row.get("eligible_for_production_training") is not False
        or row.get("manual_user_review_required") is not True
    ]
    chosen = [_normalized(row.get("chosen")) for row in selected]
    reasons: list[str] = []
    if len(source) != PHASE47_EXPECTED_CANDIDATE_COUNT:
        reasons.append("source_candidate_count_mismatch")
    if source_ids != review_ids or len(set(review_ids)) != len(review_ids):
        reasons.append("review_coverage_or_identity_failed")
    if sorted(selected_ids) != sorted(row["pair_id"] for row in review if row.get("decision") in {"accept", "edit"}):
        reasons.append("selected_candidate_mapping_failed")
    if counts["edit"] == 0:
        reasons.append("no_meaningful_edits_recorded")
    if target_problems:
        reasons.append("reviewed_target_quality_failed")
    if provenance_problems:
        reasons.append("simulated_actual_boundary_failed")
    if len(set(chosen)) != len(chosen):
        reasons.append("duplicate_reviewed_targets")
    return {
        "kind": "phase47_simulated_user_review_audit",
        "passed": not reasons,
        "source_candidate_count": len(source),
        "reviewed_decision_count": len(review),
        "selected_simulated_lab_candidate_count": len(selected),
        "decision_counts": dict(sorted(counts.items())),
        "category_counts": dict(sorted(categories.items())),
        "review_coverage_complete": source_ids == review_ids,
        "unique_review_id_count": len(set(review_ids)),
        "edited_candidate_count": counts["edit"],
        "rejected_candidate_count": counts["reject"],
        "unique_reviewed_target_ratio": round(len(set(chosen)) / len(chosen), 4) if chosen else 0.0,
        "target_problem_ids": target_problems,
        "provenance_problem_ids": provenance_problems,
        "actual_human_review_count": 0,
        "actual_user_feedback_count": 0,
        "eligible_for_production_training_count": 0,
        "actual_product_benefit_claim_allowed": False,
        "reasons": reasons,
    }


def build_phase47_decision(*, audit: Mapping[str, Any]) -> dict[str, Any]:
    passed = audit.get("passed") is True
    recommendation = (
        "use_simulated_reviewed_pack_for_fresh_runtime_ablation"
        if passed
        else "repair_simulated_review_pack_before_experiment"
    )
    return {
        "kind": "phase47_final_decision",
        "status": "ready_for_simulated_runtime_experiment" if passed else "blocked",
        "recommendation": recommendation,
        "review_audit_passed": passed,
        "reviewed_candidate_count": audit.get("selected_simulated_lab_candidate_count"),
        "edited_candidate_count": audit.get("edited_candidate_count"),
        "rejected_candidate_count": audit.get("rejected_candidate_count"),
        "actual_human_review_completed": False,
        "actual_user_feedback_count": 0,
        "training_status": "blocked",
        "training_blocker": "pending_actual_human_confirmation",
        "new_training_allowed": False,
        "runtime_experiment_allowed": passed,
        "actual_product_benefit_claim_allowed": False,
        "auto_training_allowed": False,
        "auto_promotion_allowed": False,
        "hermes_attachment_allowed": False,
        "next_action": "design_compact_intent_runtime_and_freeze_fresh_holdout" if passed else "repair_review_failures",
    }


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = [
    "PHASE47_EXPECTED_CANDIDATE_COUNT",
    "PHASE47_KIND",
    "PHASE47_REVIEWER_ID",
    "audit_phase47_review",
    "build_phase47_decision",
    "build_phase47_reviewed_candidates",
    "build_phase47_simulated_review",
    "review_phase47_candidate",
    "stable_hash",
]
