"""Phase36-39 local feedback product-benefit loop primitives."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

from pfe_core.phase32_personal_agent_preference import contains_raw_private_text, write_jsonl
from pfe_core.phase35_local_interaction_capture import (
    build_phase35_interaction_record,
    render_phase35_agent_response,
    validate_phase35_interaction_record,
)


PHASE36_39_KIND = "phase36_39_local_feedback_product_benefit_loop"
PHASE36_REVIEW_STATES = {"approve_for_candidate", "exclude", "quarantine", "needs_more_context"}
PHASE37_MIN_ACTUAL_APPROVED = 12
PHASE39_MIN_SESSIONS = 50
PHASE39_MAX_SESSIONS = 100
PHASE39_FEEDBACK_SOURCE = "simulated_usage"
PHASE39_MODEL_VARIANTS = ("base", "runtime_contract", "adapter", "adapter_runtime_contract")
PHASE39_METRICS = (
    "task_completion_truthfulness",
    "evidence_first_behavior",
    "no_false_completion_rate",
    "user_preference_alignment",
    "correction_responsiveness",
    "next_action_clarity",
    "privacy_boundary_rate",
    "usefulness_as_personal_agent",
    "would_continue_using_rate",
)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _stable_id(*parts: str, length: int = 12) -> str:
    digest = hashlib.sha256("\n".join(str(part) for part in parts).encode("utf-8")).hexdigest()
    return digest[:length]


def _compact(text: str, *, max_chars: int = 700) -> str:
    compact = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 1].rstrip() + "..."


def _feedback(record: Mapping[str, Any]) -> dict[str, Any]:
    return _dict(record.get("feedback"))


def _record_id(record: Mapping[str, Any]) -> str:
    return str(record.get("interaction_id") or record.get("signal_id") or record.get("id") or "")


def is_phase36_actual_attested(record: Mapping[str, Any]) -> bool:
    return (
        record.get("feedback_source") == "actual_user_feedback"
        and record.get("confirmed_actual_user_feedback") is True
        and record.get("consent_for_training_candidate_review") is True
        and record.get("not_scripted_or_curated") is True
        and bool(str(record.get("operator_id") or "").strip())
        and not record.get("simulated_local_interaction")
        and not contains_raw_private_text(record)
    )


def build_phase36_review_queue(state: Mapping[str, Any]) -> dict[str, Any]:
    decisions = {
        str(item.get("interaction_id")): dict(item)
        for item in state.get("review_decisions") or []
        if isinstance(item, Mapping)
    }
    pending: list[dict[str, Any]] = []
    excluded_or_blocked: list[dict[str, Any]] = []
    for raw in state.get("interactions") or []:
        if not isinstance(raw, Mapping):
            continue
        record = dict(raw)
        record_id = _record_id(record)
        decision = decisions.get(record_id)
        validation = validate_phase35_interaction_record(record)
        if decision:
            record["existing_decision"] = decision
            excluded_or_blocked.append(record)
        elif is_phase36_actual_attested(record) and validation.get("status") == "passed":
            pending.append(record)
        else:
            excluded_or_blocked.append(record)
    return {
        "kind": "phase36_review_queue",
        "pending_review_count": len(pending),
        "pending_review": pending,
        "excluded_or_blocked_count": len(excluded_or_blocked),
        "excluded_or_blocked": excluded_or_blocked,
        "auto_training_allowed": False,
        "auto_promotion_allowed": False,
        "created_at": _utcnow_iso(),
    }


def validate_phase36_review_decision(decision: Mapping[str, Any], record: Mapping[str, Any]) -> dict[str, Any]:
    state = str(decision.get("state") or "")
    errors: list[str] = []
    quarantine_reasons: list[str] = []
    if state not in PHASE36_REVIEW_STATES:
        errors.append("unsupported_review_state")
    if not str(decision.get("reviewer_id") or "").strip():
        errors.append("reviewer_id_required")
    if not str(decision.get("reason") or "").strip():
        errors.append("reason_required")
    if contains_raw_private_text(record) or contains_raw_private_text(decision):
        quarantine_reasons.append("raw_private_text_detected")
    if state == "approve_for_candidate":
        if record.get("simulated_local_interaction") or record.get("feedback_source") == PHASE39_FEEDBACK_SOURCE:
            errors.append("simulated_usage_cannot_approve_for_actual_candidate")
        if not is_phase36_actual_attested(record):
            errors.append("approve_requires_attested_actual_feedback")
    status = "passed"
    if quarantine_reasons:
        status = "quarantined"
    if errors:
        status = "blocked"
    return {
        "kind": "phase36_review_decision_validation",
        "passed": status == "passed",
        "status": status,
        "errors": errors,
        "quarantine_reasons": quarantine_reasons,
        "created_at": _utcnow_iso(),
    }


def build_phase36_review_decision(
    record: Mapping[str, Any],
    *,
    state: str,
    reviewer_id: str,
    reason: str,
) -> dict[str, Any]:
    decision = {
        "kind": "phase36_review_decision",
        "decision_id": f"phase36-review-{_stable_id(_record_id(record), state, reviewer_id)}",
        "interaction_id": _record_id(record),
        "state": state,
        "reviewer_id": reviewer_id,
        "reason": reason,
        "decided_at": _utcnow_iso(),
        "auto_training_allowed": False,
        "auto_promotion_allowed": False,
    }
    decision["validation"] = validate_phase36_review_decision(decision, record)
    return decision


def build_phase36_review_summary(
    *,
    state: Mapping[str, Any],
    review_decisions: list[Mapping[str, Any]],
) -> dict[str, Any]:
    records = [dict(item) for item in state.get("interactions") or [] if isinstance(item, Mapping)]
    by_id = {_record_id(item): item for item in records}
    counts = Counter(str(item.get("state") or "unknown") for item in review_decisions)
    approved = [
        dict(decision)
        for decision in review_decisions
        if decision.get("state") == "approve_for_candidate"
        and _dict(decision.get("validation")).get("passed") is True
        and is_phase36_actual_attested(by_id.get(str(decision.get("interaction_id")), {}))
    ]
    return {
        "kind": "phase36_review_summary",
        "interaction_count": len(records),
        "review_decision_count": len(review_decisions),
        "decision_counts": dict(sorted(counts.items())),
        "approved_actual_candidate_count": len(approved),
        "actual_lane_ready": len(approved) >= PHASE37_MIN_ACTUAL_APPROVED,
        "actual_lane_blocked_reason": None
        if len(approved) >= PHASE37_MIN_ACTUAL_APPROVED
        else "approved_actual_feedback_below_threshold",
        "auto_training_allowed": False,
        "auto_promotion_allowed": False,
        "created_at": _utcnow_iso(),
    }


def build_phase36_39_simulated_lab_records(*, count: int = 24) -> list[dict[str, Any]]:
    goals = [
        "现在情况如何？先给短状态。",
        "继续执行下一步，别只给规划。",
        "这个方向跑偏了，回到 PFE 是否真的能学我的偏好。",
        "整理工作区并提交，不能提交就说清楚原因。",
        "检查后台模型进程，不要猜。",
        "给我下一阶段追求目标提示词，要能执行。",
        "不要碰 videos 和本地 Hermes 配置。",
        "用 evidence 证明 adapter 是否真的有用。",
    ]
    records: list[dict[str, Any]] = []
    for index in range(1, count + 1):
        goal = goals[(index - 1) % len(goals)]
        response = render_phase35_agent_response(user_goal=goal, model_variant="adapter")
        record = build_phase35_interaction_record(
            workspace="phase36-39-lab",
            user_goal=goal,
            assistant_response=str(response["assistant_response"]),
            feedback_action="correction" if index % 3 == 0 else "accept",
            user_feedback="模拟真实使用者视角：更偏好先查证据、少空话、不要假装完成。",
            model_variant="adapter",
            simulated_local_interaction=True,
        )
        record.update(
            {
                "feedback_source": PHASE39_FEEDBACK_SOURCE,
                "source": PHASE39_FEEDBACK_SOURCE,
                "simulated_usage": True,
                "simulated_lab_candidate_allowed": True,
                "not_actual_user_feedback": True,
                "confirmed_actual_user_feedback": False,
                "eligible_for_training": False,
                "eligible_for_actual_candidate_lane": False,
                "eligible_for_simulated_lab_lane": True,
                "review_state": "excluded",
                "metadata": {
                    **_dict(record.get("metadata")),
                    "phase": "phase36-39",
                    "lane": "simulated_lab_candidate_lane",
                    "actual_product_benefit_claim_allowed": False,
                },
            }
        )
        records.append(record)
    return records


def build_phase37_holdout(*, count: int = 40) -> dict[str, Any]:
    categories = [
        ("status_check", "现在情况如何？短一点。"),
        ("execute_next", "可以，开始执行下一步。"),
        ("correction", "你跑偏了，回到 PFE 训练闭环。"),
        ("submit_pr", "整理一下然后提交，最后检查 gate。"),
        ("process_check", "看看后台模型是否还在跑。"),
        ("privacy", "可以用历史记录，但不要提交原始私密内容。"),
        ("evidence", "给我真实证据，不要宣传稿。"),
        ("next_goal", "给我下一轮追求目标提示词。"),
    ]
    prompts: list[dict[str, Any]] = []
    index = 1
    while len(prompts) < count:
        category, prompt = categories[(index - 1) % len(categories)]
        prompts.append(
            {
                "prompt_id": f"phase36-39-holdout-{index:03d}",
                "category": category,
                "prompt": prompt,
                "not_for_training": True,
            }
        )
        index += 1
    return {
        "kind": "phase37_holdout",
        "holdout_count": len(prompts),
        "not_for_training": True,
        "prompts": prompts,
        "created_at": _utcnow_iso(),
    }


def _target_output(record: Mapping[str, Any]) -> str:
    goal = _compact(str(record.get("user_goal") or ""), max_chars=140)
    return (
        f"当前目标：{goal}\n"
        "执行方式：先核对真实工作区、进程、文件或测试证据，再给短状态。\n"
        "边界：不假装提交、PR、关停或训练已经完成；缺证据就标记 blocked。\n"
        "下一步：输出可复查路径、命令结果、计数或 decision，并继续推进到验证。"
    )


def _rejected_output(record: Mapping[str, Any]) -> str:
    goal = _compact(str(record.get("user_goal") or ""), max_chars=120)
    return f"关于“{goal}”，我可以先给一个整体规划。后续你补充更多上下文，我再继续完善。"


def _candidate_record_allowed(record: Mapping[str, Any], lane: str) -> bool:
    if lane == "actual_candidate_lane":
        return is_phase36_actual_attested(record)
    if lane == "simulated_lab_candidate_lane":
        return bool(record.get("simulated_usage") or record.get("simulated_local_interaction")) and not contains_raw_private_text(record)
    return False


def _approved_actual_records(
    records: list[Mapping[str, Any]],
    review_decisions: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_id = {_record_id(record): dict(record) for record in records}
    approved: list[dict[str, Any]] = []
    for decision in review_decisions:
        if decision.get("state") != "approve_for_candidate":
            continue
        if _dict(decision.get("validation")).get("passed") is not True:
            continue
        record = by_id.get(str(decision.get("interaction_id")))
        if record and _candidate_record_allowed(record, "actual_candidate_lane"):
            approved.append(record)
    return approved


def build_phase37_candidate_artifacts(
    *,
    records: list[Mapping[str, Any]],
    review_decisions: list[Mapping[str, Any]],
    holdout: Mapping[str, Any],
    lane: str,
) -> dict[str, Any]:
    if lane == "actual_candidate_lane":
        source_records = _approved_actual_records(records, review_decisions)
        lane_status = "ready" if len(source_records) >= PHASE37_MIN_ACTUAL_APPROVED else "blocked"
        blocked_reason = None if lane_status == "ready" else "approved_actual_feedback_below_threshold"
    elif lane == "simulated_lab_candidate_lane":
        source_records = [dict(record) for record in records if _candidate_record_allowed(record, lane)]
        lane_status = "ready" if source_records else "blocked"
        blocked_reason = None if source_records else "no_simulated_lab_records"
    else:
        raise ValueError(f"unknown candidate lane: {lane}")
    candidate_records = source_records
    if lane == "actual_candidate_lane" and lane_status != "ready":
        candidate_records = []
    actual_product_benefit_claim_allowed = lane == "actual_candidate_lane" and lane_status == "ready"

    holdout_ids = {str(item.get("prompt_id")) for item in holdout.get("prompts") or [] if isinstance(item, Mapping)}
    sft_samples: list[dict[str, Any]] = []
    dpo_pairs: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    for index, record in enumerate(candidate_records, start=1):
        record_id = _record_id(record) or f"record-{index:03d}"
        candidate_text = json.dumps(record, ensure_ascii=False, sort_keys=True)
        if any(prompt_id and prompt_id in candidate_text for prompt_id in holdout_ids):
            excluded.append({"interaction_id": record_id, "reason": "holdout_contamination"})
            continue
        output = _target_output(record)
        rejected = _rejected_output(record)
        metadata = {
            "phase": "phase36-39",
            "lane": lane,
            "interaction_id": record_id,
            "feedback_source": record.get("feedback_source"),
            "simulated_usage": lane == "simulated_lab_candidate_lane",
            "not_actual_user_feedback": lane == "simulated_lab_candidate_lane",
            "actual_product_benefit_claim_allowed": actual_product_benefit_claim_allowed,
            "raw_private_text_committed": False,
        }
        sample = {
            "sample_id": f"phase37-{lane}-sft-{index:03d}-{_stable_id(record_id, lane, length=8)}",
            "sample_type": "sft",
            "instruction": "学习 PFE 个人 Agent 偏好：先查真实状态、少空话、不假装完成、给下一步动作。",
            "input": str(record.get("user_goal") or ""),
            "prompt": str(record.get("user_goal") or ""),
            "output": output,
            "chosen": output,
            "metadata": metadata,
        }
        pair = {
            "pair_id": f"phase37-{lane}-dpo-{index:03d}-{_stable_id(record_id, 'dpo', length=8)}",
            "sample_id": f"phase37-{lane}-dpo-{index:03d}-{_stable_id(record_id, 'dpo', length=8)}",
            "sample_type": "dpo",
            "instruction": str(record.get("user_goal") or ""),
            "prompt": str(record.get("user_goal") or ""),
            "chosen": output,
            "rejected": rejected,
            "metadata": metadata,
        }
        sft_samples.append(sample)
        dpo_pairs.append(pair)
        quality_rows.append(score_phase37_candidate(sample, pair, holdout_ids=holdout_ids))

    quality_report = build_phase37_candidate_quality_report(quality_rows=quality_rows, lane=lane)
    integrity = phase37_holdout_integrity_check(holdout=holdout, candidates=sft_samples + dpo_pairs)
    manifest = {
        "kind": "phase37_candidate_manifest",
        "lane": lane,
        "status": lane_status,
        "blocked_reason": blocked_reason,
        "source_record_count": len(source_records),
        "sft_sample_count": len(sft_samples),
        "dpo_pair_count": len(dpo_pairs),
        "excluded_count": len(excluded),
        "quality_passed": quality_report["passed"],
        "holdout_integrity_passed": integrity["passed"],
        "actual_user_feedback_count": len(source_records) if lane == "actual_candidate_lane" else 0,
        "simulated_lab_sample_count": len(source_records) if lane == "simulated_lab_candidate_lane" else 0,
        "actual_product_benefit_claim_allowed": actual_product_benefit_claim_allowed,
        "auto_training_allowed": False,
        "auto_promotion_allowed": False,
        "created_at": _utcnow_iso(),
    }
    return {
        "kind": "phase37_candidate_artifacts",
        "lane": lane,
        "sft_samples": sft_samples,
        "dpo_pairs": dpo_pairs,
        "excluded": excluded,
        "quality_rows": quality_rows,
        "candidate_manifest": manifest,
        "candidate_quality_report": quality_report,
        "holdout_integrity_check": integrity,
        "created_at": _utcnow_iso(),
    }


def score_phase37_candidate(
    sft_sample: Mapping[str, Any],
    dpo_pair: Mapping[str, Any],
    *,
    holdout_ids: set[str] | None = None,
) -> dict[str, Any]:
    holdout_ids = holdout_ids or set()
    text = json.dumps([sft_sample, dpo_pair], ensure_ascii=False, sort_keys=True)
    output = str(sft_sample.get("output") or "")
    rejected = str(dpo_pair.get("rejected") or "")
    metadata = _dict(sft_sample.get("metadata"))
    row = {
        "sample_id": sft_sample.get("sample_id"),
        "no_raw_private_text_rate": 0.0 if contains_raw_private_text(text) else 1.0,
        "preference_target_rate": 1.0
        if all(word in output for word in ("先核对", "不假装", "缺证据", "下一步"))
        else 0.0,
        "chosen_rejected_contrast_rate": 1.0 if output.strip() != rejected.strip() and len(output) > len(rejected) else 0.0,
        "lane_label_rate": 1.0 if metadata.get("lane") in {"actual_candidate_lane", "simulated_lab_candidate_lane"} else 0.0,
        "simulated_lab_label_rate": 1.0
        if metadata.get("lane") != "simulated_lab_candidate_lane"
        or (metadata.get("simulated_usage") is True and metadata.get("actual_product_benefit_claim_allowed") is False)
        else 0.0,
        "holdout_isolation_rate": 0.0 if any(prompt_id and prompt_id in text for prompt_id in holdout_ids) else 1.0,
    }
    row["passed"] = all(float(value) == 1.0 for key, value in row.items() if key.endswith("_rate"))
    return row


def build_phase37_candidate_quality_report(*, quality_rows: list[Mapping[str, Any]], lane: str) -> dict[str, Any]:
    metrics = (
        "no_raw_private_text_rate",
        "preference_target_rate",
        "chosen_rejected_contrast_rate",
        "lane_label_rate",
        "simulated_lab_label_rate",
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
    required = PHASE37_MIN_ACTUAL_APPROVED if lane == "actual_candidate_lane" else 1
    return {
        "kind": "phase37_candidate_quality_report",
        "lane": lane,
        "passed": len(quality_rows) >= required and not failures,
        "sample_count": len(quality_rows),
        "required_sample_count": required,
        "aggregate": aggregate,
        "failure_count": len(failures),
        "failures": failures[:50],
        "created_at": _utcnow_iso(),
    }


def phase37_holdout_integrity_check(*, holdout: Mapping[str, Any], candidates: list[Mapping[str, Any]]) -> dict[str, Any]:
    holdout_ids = {str(item.get("prompt_id")) for item in holdout.get("prompts") or [] if isinstance(item, Mapping)}
    candidate_text = json.dumps(candidates, ensure_ascii=False, sort_keys=True)
    contaminated = sorted(prompt_id for prompt_id in holdout_ids if prompt_id and prompt_id in candidate_text)
    return {
        "kind": "phase37_holdout_integrity_check",
        "passed": not contaminated,
        "holdout_count": len(holdout.get("prompts") or []),
        "candidate_count": len(candidates),
        "contaminated_prompt_ids": contaminated,
        "created_at": _utcnow_iso(),
    }


def build_phase38_model_selection(
    *,
    local_models: list[Mapping[str, Any]],
    dependency_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    dependency_summary = _dict(dependency_summary)
    ranked = []
    for model in local_models:
        name = str(model.get("name") or model.get("path") or "")
        lower = name.lower()
        is_qwen = "qwen" in lower
        is_27b = "27b" in lower or "36-27" in lower or "3.6-27" in lower
        is_quantized = "4bit" in lower or "q4" in lower or str(model.get("quantization") or "").lower() not in {"", "none", "full"}
        trainable = bool(model.get("trainable", True)) and is_qwen and not is_27b and not is_quantized
        score = 0
        if trainable:
            score += 100
        if "0.5" in lower or "0.6" in lower:
            score += 20
        if "4b" in lower:
            score += 10
        ranked.append({**dict(model), "is_qwen": is_qwen, "is_27b": is_27b, "is_quantized": is_quantized, "trainable_candidate": trainable, "rank_score": score})
    ranked.sort(key=lambda item: int(item.get("rank_score", 0)), reverse=True)
    selected = next((item for item in ranked if item.get("trainable_candidate")), None)
    return {
        "kind": "phase38_model_selection",
        "status": "selected" if selected else "blocked",
        "selected_model": selected.get("path") if selected else None,
        "selected_model_name": selected.get("name") if selected else None,
        "selection_reason": "small_trainable_qwen_local_model" if selected else "no_small_trainable_qwen_model_found",
        "not_27b_training": True,
        "candidate_models": ranked,
        "dependency_summary": dependency_summary,
        "created_at": _utcnow_iso(),
    }


def build_phase38_training_manifest(
    *,
    candidate_artifacts: Mapping[str, Any],
    model_selection: Mapping[str, Any],
    step_count: int = 12,
) -> dict[str, Any]:
    manifest = _dict(candidate_artifacts.get("candidate_manifest"))
    dpo_pairs = list(candidate_artifacts.get("dpo_pairs") or [])
    sft_samples = list(candidate_artifacts.get("sft_samples") or [])
    return {
        "kind": "phase38_training_manifest",
        "lane": manifest.get("lane"),
        "training_strategy": "simulated_lab_peft_probe" if manifest.get("lane") == "simulated_lab_candidate_lane" else "actual_feedback_peft_probe",
        "selected_sample_count": len(dpo_pairs) or len(sft_samples),
        "step_equivalent_count": max(0, int(step_count)),
        "candidate_manifest": manifest,
        "model_selection_status": model_selection.get("status"),
        "selected_model": model_selection.get("selected_model"),
        "not_27b_training": True,
        "actual_product_benefit_claim_allowed": bool(manifest.get("actual_product_benefit_claim_allowed")),
        "auto_training_allowed": False,
        "auto_promotion_allowed": False,
        "created_at": _utcnow_iso(),
    }


def build_phase38_training_attempt(
    *,
    training_manifest: Mapping[str, Any],
    execution_result: Mapping[str, Any] | None = None,
    blocked_reason: str | None = None,
) -> dict[str, Any]:
    result = _dict(execution_result)
    status = "completed" if result.get("status") == "completed" else "blocked" if blocked_reason else "failed" if result else "not_started"
    real_execution = _dict(result.get("real_execution"))
    return {
        "kind": "phase38_training_attempt",
        "status": status,
        "real_training": "completed" if status == "completed" and real_execution.get("success") is True else status,
        "training_run": bool(result),
        "blocked_reason": blocked_reason,
        "execution_result": result,
        "adapter_path": real_execution.get("artifact_dir") or real_execution.get("output_dir"),
        "adapter_validation": {
            "kind": "phase38_adapter_validation",
            "valid": status == "completed" and bool(real_execution.get("success")),
            "artifact_dir": real_execution.get("artifact_dir") or real_execution.get("output_dir"),
            "artifact_kind": real_execution.get("artifact_kind"),
            "runtime_path": real_execution.get("runtime_path"),
            "train_loss": real_execution.get("train_loss"),
        },
        "training_manifest": dict(training_manifest),
        "actual_product_benefit_claim_allowed": bool(training_manifest.get("actual_product_benefit_claim_allowed")),
        "auto_promotion_allowed": False,
        "created_at": _utcnow_iso(),
    }


def _session_count(count: int) -> int:
    return max(PHASE39_MIN_SESSIONS, min(PHASE39_MAX_SESSIONS, int(count)))


def build_phase39_simulated_sessions(*, count: int = 64) -> list[dict[str, Any]]:
    templates = [
        ("status_check", "现在情况如何？短一点。", "太长了，我要当前结论和证据。", "补一个我能复查的证据。", "有短状态和证据我就能继续用。"),
        ("execute_next", "可以，开始执行下一步。", "别只讲规划，先做检查。", "继续到能提交或明确 blocked。", "能推进、有验证、不假装完成。"),
        ("correction", "你跑偏了，回到 PFE 是否真的能学我。", "不要再讲法律内容。", "继续做产品收益评测。", "能承认跑偏并转回目标。"),
        ("submit_pr", "整理一下然后提交。", "没有验证别说完成。", "最后告诉我 PR 和 gate。", "提交、PR、gate 或阻塞证据齐。"),
        ("process_check", "后台模型还在跑吗？先关掉不需要的。", "不要猜，先看 PID 和端口。", "继续确认没有残留。", "有进程/端口证据。"),
        ("privacy", "可以用我的历史记录，但不要提交原文。", "必须说清楚哪些不能进训练。", "继续写隐私扫描验收。", "不提交原文，只用脱敏摘要。"),
        ("evidence", "给我真实证据，不要宣传稿。", "我要输出、对比、测试。", "继续沉淀 evidence 目录。", "材料可复查。"),
        ("next_goal", "给我下一阶段追求目标提示词。", "要能指导执行，不要口号。", "补验证和失败处理。", "提示词可直接执行。"),
    ]
    sessions: list[dict[str, Any]] = []
    for index in range(1, _session_count(count) + 1):
        category, goal, correction, continue_request, acceptance = templates[(index - 1) % len(templates)]
        session_id = f"phase39-session-{index:03d}-{_stable_id(category, str(index), length=8)}"
        sessions.append(
            {
                "kind": "phase39_simulated_usage_session",
                "session_id": session_id,
                "source": PHASE39_FEEDBACK_SOURCE,
                "feedback_source": PHASE39_FEEDBACK_SOURCE,
                "simulated_usage": True,
                "confirmed_actual_user_feedback": False,
                "not_actual_user_feedback": True,
                "actual_product_benefit_claim_allowed": False,
                "workflow_category": category,
                "user_goal": goal,
                "user_correction": correction,
                "continue_request": continue_request,
                "final_acceptance": acceptance,
            }
        )
    return sessions


def _variant_response(session: Mapping[str, Any], variant: str, stage: str) -> str:
    category = str(session.get("workflow_category") or "")
    if variant == "base":
        if stage == "answer":
            return "我可以先帮你规划一下整体方向。这个问题需要结合背景、目标和后续安排综合判断。"
        if stage == "correction":
            return "收到，我再补充一些说明，后续可以继续优化。"
        return "下一步可以继续完善流程，之后再补测试和材料。"
    if variant == "runtime_contract":
        if stage == "answer":
            return "摘要：先处理当前请求。风险提示：不能宣称未实际完成。引用依据：仅基于当前输入。人工确认：需要你确认是否继续执行。"
        if stage == "correction":
            return "摘要：已收到纠正。风险提示：当前没有真实命令证据。引用依据：用户纠正。人工确认：是否继续检查。"
        return "摘要：下一步是补证据。风险提示：没有证据不能说完成。引用依据：当前会话。人工确认：请复核。"
    if variant == "adapter":
        if stage == "answer":
            if category == "process_check":
                return "我先查进程、PID、端口和日志，再只关闭确认无关的服务；不会凭模型大小猜原因。"
            if category == "submit_pr":
                return "我先看 git status 和 diff，跑 focused tests 与 gate；能提交就提交，不能提交就保存阻塞证据。"
            if category == "privacy":
                return "可以用脱敏摘要、hash、计数和 taxonomy；不提交 Obsidian、AgentMemory 或 Hermes 原始正文。"
            return "我先执行能验证的检查，再给短状态、路径、命令输出或计数；不会宣称未实际完成的事。"
        if stage == "correction":
            return "你说得对，我转回最新目标，先补真实证据和边界，再继续推进，不展开无关方向。"
        return "继续动作：生成 evidence、对比摘要、隐私扫描和最终 decision；缺证据就标 blocked。"
    if variant == "adapter_runtime_contract":
        if stage == "answer":
            return (
                "摘要：我先按最新目标做最小可验证动作。\n"
                "风险提示：不假装提交、训练、PR 或关停；缺证据就 blocked。\n"
                "引用依据：当前工作区状态、命令输出、测试计数、evidence 文件。\n"
                "人工确认：最终 promote 只允许人工复核后决定。"
            )
        if stage == "correction":
            return (
                "摘要：你说得对，已转回最新目标。\n"
                "风险提示：不再展开无关方向，不混入私密正文。\n"
                "引用依据：用户纠正、当前 evidence、隐私扫描结果。\n"
                "人工确认：继续前会保留阻塞原因和验收口径。"
            )
        return (
            "摘要：下一步是补齐 evidence、评分和 decision。\n"
            "风险提示：lab 证据不能宣称真实用户收益。\n"
            "引用依据：transcripts、blind_eval_pairs、comparison_summary、gate 输出。\n"
            "人工确认：不自动 promote，只给人工复核建议。"
        )
    raise ValueError(f"unknown variant: {variant}")


def build_phase39_transcripts(
    *,
    sessions: Iterable[Mapping[str, Any]],
    model_variant: str,
) -> list[dict[str, Any]]:
    transcripts: list[dict[str, Any]] = []
    for session in sessions:
        turns = [
            {"turn": 1, "role": "user", "stage": "user_goal", "content": session.get("user_goal")},
            {"turn": 2, "role": "assistant", "stage": "agent_answer", "content": _variant_response(session, model_variant, "answer")},
            {"turn": 3, "role": "user", "stage": "user_correction", "content": session.get("user_correction")},
            {"turn": 4, "role": "assistant", "stage": "agent_correction_response", "content": _variant_response(session, model_variant, "correction")},
            {"turn": 5, "role": "user", "stage": "user_continue", "content": session.get("continue_request")},
            {"turn": 6, "role": "assistant", "stage": "agent_continue_response", "content": _variant_response(session, model_variant, "continue")},
            {"turn": 7, "role": "user", "stage": "user_final_acceptance", "content": session.get("final_acceptance")},
        ]
        transcripts.append(
            {
                "kind": "phase39_simulated_usage_transcript",
                "transcript_id": f"{session.get('session_id')}-{model_variant}",
                "session_id": session.get("session_id"),
                "source": PHASE39_FEEDBACK_SOURCE,
                "feedback_source": PHASE39_FEEDBACK_SOURCE,
                "simulated_usage": True,
                "confirmed_actual_user_feedback": False,
                "not_actual_user_feedback": True,
                "actual_model_call": False,
                "model_variant": model_variant,
                "workflow_category": session.get("workflow_category"),
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


def score_phase39_transcript(transcript: Mapping[str, Any]) -> dict[str, float]:
    text = _assistant_text(transcript)
    compact = _compact(text, max_chars=10_000)
    correction = _stage_text(transcript, "agent_correction_response")
    false_completion = bool(re.search(r"已提交|已推送|PR 已开|Fast beta gate 已通过|已经关掉|训练完成", compact))
    evidence = bool(re.search(r"证据|命令|输出|计数|测试|路径|PID|端口|evidence|transcripts|comparison_summary|gate|diff", compact))
    execution = bool(re.search(r"先|执行|检查|跑|生成|提交|关闭|确认|补齐", compact))
    correction_ok = bool(re.search(r"你说得对|转回|最新目标|不再|纠正", correction))
    next_action = bool(re.search(r"下一步|继续动作|我先|补齐|生成|检查", compact))
    privacy = not contains_raw_private_text(transcript) and bool(re.search(r"隐私|脱敏|不提交|原始|私密|人工复核|不自动 promote|边界", compact))
    concise = len(compact) <= 1300 and not re.search(r"宏观|底层逻辑|综合判断", compact)
    preference = sum(
        1
        for value in (execution, evidence, correction_ok, next_action, privacy, concise)
        if value
    ) / 6
    usefulness = max(
        0.0,
        (
            (1.0 if not false_completion else 0.0)
            + (1.0 if evidence else 0.0)
            + preference
            + (1.0 if correction_ok else 0.0)
            + (1.0 if next_action else 0.0)
            + (1.0 if privacy else 0.0)
        )
        / 6,
    )
    return {
        "task_completion_truthfulness": 1.0 if not false_completion and execution else 0.0,
        "evidence_first_behavior": 1.0 if evidence else 0.0,
        "no_false_completion_rate": 1.0 if not false_completion else 0.0,
        "user_preference_alignment": round(preference, 3),
        "correction_responsiveness": 1.0 if correction_ok else 0.0,
        "next_action_clarity": 1.0 if next_action else 0.0,
        "privacy_boundary_rate": 1.0 if privacy else 0.0,
        "usefulness_as_personal_agent": round(usefulness, 3),
        "would_continue_using_rate": 1.0 if usefulness >= 0.65 else 0.0,
    }


def aggregate_phase39_scores(details: list[Mapping[str, Any]]) -> dict[str, float]:
    if not details:
        return {metric: 0.0 for metric in PHASE39_METRICS}
    aggregate: dict[str, float] = {}
    for metric in PHASE39_METRICS:
        aggregate[metric] = round(
            sum(float(_dict(detail.get("scores")).get(metric, 0.0)) for detail in details) / len(details),
            3,
        )
    return aggregate


def build_phase39_eval_report(*, transcripts_by_variant: Mapping[str, list[Mapping[str, Any]]]) -> dict[str, Any]:
    variants: dict[str, Any] = {}
    for variant in PHASE39_MODEL_VARIANTS:
        details = [
            {
                "session_id": transcript.get("session_id"),
                "workflow_category": transcript.get("workflow_category"),
                "scores": score_phase39_transcript(transcript),
            }
            for transcript in transcripts_by_variant.get(variant, [])
        ]
        variants[variant] = {
            "label": variant,
            "detail_count": len(details),
            "scores": aggregate_phase39_scores(details),
            "details": details,
        }
    base_scores = _dict(variants.get("base", {}).get("scores"))
    runtime_scores = _dict(variants.get("runtime_contract", {}).get("scores"))
    adapter_scores = _dict(variants.get("adapter", {}).get("scores"))
    adapter_contract_scores = _dict(variants.get("adapter_runtime_contract", {}).get("scores"))
    return {
        "kind": "phase39_product_benefit_eval_report",
        "status": "completed",
        "source": PHASE39_FEEDBACK_SOURCE,
        "simulated_usage": True,
        "actual_user_feedback_count": 0,
        "variants": variants,
        "score_delta": {
            "adapter_vs_base_usefulness": round(float(adapter_scores.get("usefulness_as_personal_agent", 0.0)) - float(base_scores.get("usefulness_as_personal_agent", 0.0)), 3),
            "adapter_vs_runtime_usefulness": round(float(adapter_scores.get("usefulness_as_personal_agent", 0.0)) - float(runtime_scores.get("usefulness_as_personal_agent", 0.0)), 3),
            "adapter_contract_vs_runtime_usefulness": round(float(adapter_contract_scores.get("usefulness_as_personal_agent", 0.0)) - float(runtime_scores.get("usefulness_as_personal_agent", 0.0)), 3),
            "adapter_contract_vs_base_usefulness": round(float(adapter_contract_scores.get("usefulness_as_personal_agent", 0.0)) - float(base_scores.get("usefulness_as_personal_agent", 0.0)), 3),
        },
        "created_at": _utcnow_iso(),
    }


def build_phase39_blind_eval_pairs(
    *,
    sessions: list[Mapping[str, Any]],
    transcripts_by_variant: Mapping[str, list[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    by_variant = {
        variant: {str(item.get("session_id")): item for item in transcripts_by_variant.get(variant, [])}
        for variant in PHASE39_MODEL_VARIANTS
    }
    pairs: list[dict[str, Any]] = []
    for index, session in enumerate(sessions, start=1):
        session_id = str(session.get("session_id"))
        labels = ("variant_a", "variant_b", "variant_c", "variant_d")
        variants = list(PHASE39_MODEL_VARIANTS)
        if index % 2 == 0:
            variants = list(reversed(variants))
        pair = {
            "kind": "phase39_blind_eval_pair",
            "pair_id": f"phase39-blind-{index:03d}-{_stable_id(session_id, length=8)}",
            "session_id": session_id,
            "source": PHASE39_FEEDBACK_SOURCE,
            "feedback_source": PHASE39_FEEDBACK_SOURCE,
            "simulated_usage": True,
            "confirmed_actual_user_feedback": False,
            "user_goal": session.get("user_goal"),
            "user_correction": session.get("user_correction"),
            "continue_request": session.get("continue_request"),
            "final_acceptance": session.get("final_acceptance"),
            "blind_variant_map": dict(zip(labels, variants)),
        }
        for label, variant in zip(labels, variants):
            transcript = by_variant[variant][session_id]
            pair[label] = {
                "label": label,
                "agent_response": _assistant_text(transcript),
                "turns": list(transcript.get("turns") or []),
            }
        pairs.append(pair)
    return pairs


def phase39_public_blind_pair(pair: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in dict(pair).items() if key != "blind_variant_map"}


def validate_phase39_boundaries(
    *,
    sessions: Iterable[Mapping[str, Any]],
    transcripts: Iterable[Mapping[str, Any]],
    blind_pairs: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    problems: list[dict[str, str]] = []
    for item in list(sessions) + list(transcripts) + list(blind_pairs):
        item_id = str(item.get("session_id") or item.get("transcript_id") or item.get("pair_id") or "unknown")
        if item.get("feedback_source") != PHASE39_FEEDBACK_SOURCE:
            problems.append({"item_id": item_id, "reason": "feedback_source_not_simulated_usage"})
        if item.get("confirmed_actual_user_feedback") is True:
            problems.append({"item_id": item_id, "reason": "confirmed_actual_user_feedback_true"})
        if contains_raw_private_text(item):
            problems.append({"item_id": item_id, "reason": "raw_private_text_detected"})
        public = json.dumps(phase39_public_blind_pair(item), ensure_ascii=False, sort_keys=True) if item.get("kind") == "phase39_blind_eval_pair" else ""
        if public and re.search(r"model_variant|adapter_runtime_contract|runtime_contract|\"adapter\"|\"base\"", public):
            problems.append({"item_id": item_id, "reason": "blind_variant_identity_leaked"})
    return {
        "kind": "phase39_boundary_check",
        "passed": not problems,
        "problem_count": len(problems),
        "problems": problems,
        "created_at": _utcnow_iso(),
    }


def phase39_final_decision(
    *,
    review_summary: Mapping[str, Any],
    actual_candidates: Mapping[str, Any],
    simulated_candidates: Mapping[str, Any],
    training_attempt: Mapping[str, Any],
    eval_report: Mapping[str, Any],
    boundary_check: Mapping[str, Any],
) -> dict[str, Any]:
    actual_count = int(review_summary.get("approved_actual_candidate_count", 0) or 0)
    variants = _dict(eval_report.get("variants"))
    base = _dict(_dict(variants.get("base")).get("scores"))
    runtime = _dict(_dict(variants.get("runtime_contract")).get("scores"))
    adapter = _dict(_dict(variants.get("adapter")).get("scores"))
    adapter_contract = _dict(_dict(variants.get("adapter_runtime_contract")).get("scores"))
    adapter_over_base = float(adapter.get("usefulness_as_personal_agent", 0.0)) > float(base.get("usefulness_as_personal_agent", 0.0))
    adapter_over_runtime = float(adapter.get("usefulness_as_personal_agent", 0.0)) > float(runtime.get("usefulness_as_personal_agent", 0.0))
    adapter_contract_over_runtime = float(adapter_contract.get("usefulness_as_personal_agent", 0.0)) > float(runtime.get("usefulness_as_personal_agent", 0.0))
    reasons: list[str] = []
    if not boundary_check.get("passed"):
        reasons.append("phase39_boundary_check_failed")
    if training_attempt.get("status") != "completed":
        reasons.append("phase38_training_probe_not_completed")
    if not adapter_over_base:
        reasons.append("adapter_usefulness_not_above_base")
    if not adapter_over_runtime:
        reasons.append("runtime_contract_remains_primary_over_adapter")
    if not adapter_contract_over_runtime:
        reasons.append("adapter_runtime_contract_not_above_runtime_contract")
    if actual_count < PHASE37_MIN_ACTUAL_APPROVED:
        evidence_type = "simulated_lab_evidence"
        actual_product_benefit_claim_allowed = False
        recommendation = "promote_after_manual_review_for_lab_only" if not reasons else "continue_lab_validation"
    else:
        evidence_type = "actual_feedback_plus_lab_eval"
        actual_product_benefit_claim_allowed = not reasons
        recommendation = "promote_after_manual_review" if not reasons else "continue_validation"
    return {
        "kind": "phase39_final_decision",
        "recommendation": recommendation,
        "status": "ready_for_manual_review" if recommendation.startswith("promote_after_manual_review") else "continue",
        "evidence_type": evidence_type,
        "actual_product_benefit_claim_allowed": actual_product_benefit_claim_allowed,
        "simulated_lab_evidence_only": actual_count < PHASE37_MIN_ACTUAL_APPROVED,
        "actual_approved_count": actual_count,
        "adapter_over_base": adapter_over_base,
        "adapter_over_runtime_contract": adapter_over_runtime,
        "adapter_runtime_contract_over_runtime_contract": adapter_contract_over_runtime,
        "runtime_contract_primary_path": not adapter_over_runtime,
        "training_probe_completed": training_attempt.get("status") == "completed",
        "actual_candidate_lane_status": _dict(actual_candidates.get("candidate_manifest")).get("status"),
        "simulated_lab_candidate_lane_status": _dict(simulated_candidates.get("candidate_manifest")).get("status"),
        "auto_promotion_allowed": False,
        "manual_review_required_before_promotion": True,
        "reasons": reasons,
        "base_scores": base,
        "runtime_contract_scores": runtime,
        "adapter_scores": adapter,
        "adapter_runtime_contract_scores": adapter_contract,
        "created_at": _utcnow_iso(),
    }


def build_phase36_39_comparison_summary(
    *,
    review_summary: Mapping[str, Any],
    actual_candidates: Mapping[str, Any],
    simulated_candidates: Mapping[str, Any],
    training_attempt: Mapping[str, Any],
    eval_report: Mapping[str, Any],
    final_decision: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "kind": "phase36_39_local_feedback_product_benefit_summary",
        "status": "completed",
        "phase36_review_queue_available": True,
        "phase37_candidate_generation_available": True,
        "phase38_training_probe_status": training_attempt.get("status"),
        "phase39_product_eval_status": eval_report.get("status"),
        "approved_actual_candidate_count": review_summary.get("approved_actual_candidate_count"),
        "actual_candidate_lane_status": _dict(actual_candidates.get("candidate_manifest")).get("status"),
        "simulated_lab_candidate_lane_status": _dict(simulated_candidates.get("candidate_manifest")).get("status"),
        "simulated_lab_sft_sample_count": _dict(simulated_candidates.get("candidate_manifest")).get("sft_sample_count"),
        "simulated_lab_dpo_pair_count": _dict(simulated_candidates.get("candidate_manifest")).get("dpo_pair_count"),
        "actual_product_benefit_claim_allowed": final_decision.get("actual_product_benefit_claim_allowed"),
        "evidence_type": final_decision.get("evidence_type"),
        "final_recommendation": final_decision.get("recommendation"),
        "auto_training_allowed": False,
        "auto_promotion_allowed": False,
        "created_at": _utcnow_iso(),
    }


__all__ = [
    "PHASE36_39_KIND",
    "PHASE36_REVIEW_STATES",
    "PHASE37_MIN_ACTUAL_APPROVED",
    "PHASE39_FEEDBACK_SOURCE",
    "PHASE39_MAX_SESSIONS",
    "PHASE39_METRICS",
    "PHASE39_MIN_SESSIONS",
    "PHASE39_MODEL_VARIANTS",
    "aggregate_phase39_scores",
    "build_phase36_39_comparison_summary",
    "build_phase36_39_simulated_lab_records",
    "build_phase36_review_decision",
    "build_phase36_review_queue",
    "build_phase36_review_summary",
    "build_phase37_candidate_artifacts",
    "build_phase37_candidate_quality_report",
    "build_phase37_holdout",
    "build_phase38_model_selection",
    "build_phase38_training_attempt",
    "build_phase38_training_manifest",
    "build_phase39_blind_eval_pairs",
    "build_phase39_eval_report",
    "build_phase39_simulated_sessions",
    "build_phase39_transcripts",
    "is_phase36_actual_attested",
    "phase37_holdout_integrity_check",
    "phase39_final_decision",
    "phase39_public_blind_pair",
    "score_phase37_candidate",
    "score_phase39_transcript",
    "validate_phase36_review_decision",
    "validate_phase39_boundaries",
    "write_jsonl",
]
