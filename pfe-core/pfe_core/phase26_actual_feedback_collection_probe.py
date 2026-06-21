"""Phase26 actual-feedback collection and training-probe readiness primitives.

Phase26 makes the next product step executable without fabricating user data:
it prepares a reviewable collection pack, validates attested actual-feedback
batches, and only opens the Qwen training-probe path after enough approved
actual-user corrections exist.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Any, Mapping

from .inference.contracts import BOUNDARY_CONTRACT_ID
from .phase23_runtime_contract_loop import build_runtime_contract_response
from .phase24_real_signal_review_candidate_value import (
    build_phase24_candidate_artifacts,
    build_phase24_holdout,
    build_phase24_model_selection,
    evaluate_phase24_runtime_contract_holdout,
    phase24_holdout_integrity_check,
    phase24_runtime_product_decision,
)
from .phase25_actual_user_feedback_loop import (
    PHASE25_MIN_APPROVED_ACTUAL_CANDIDATES,
    apply_phase25_review_decisions,
    build_phase25_actual_feedback_signal,
    build_phase25_attestation_template,
    build_phase25_review_queue,
    build_phase25_routing_report,
    build_phase25_training_job_specs,
    build_phase25_training_readiness_report,
)


PHASE26_KIND = "phase26_actual_feedback_collection_training_probe"
PHASE26_COLLECTION_COUNT = PHASE25_MIN_APPROVED_ACTUAL_CANDIDATES

_COLLECTION_BLUEPRINTS = (
    ("付款义务", "资料说明客户需在发票日后三十日内付款，逾期服务可能暂停。"),
    ("服务水平", "资料说明服务可用性目标和服务补偿可能写在订单或附件中。"),
    ("终止条款", "资料说明任一方可在重大违约后三十日内终止协议。"),
    ("自动续约", "资料说明协议可能按年度自动续约，除非提前通知终止。"),
    ("审计权利", "资料说明客户可能有审计权，但审计频率、范围和保密要求需核对。"),
    ("数据处理", "资料说明服务商可为提供服务处理客户数据，但未列出全部安全措施。"),
    ("责任限制", "资料说明间接损失可能被排除，直接损失可能受费用上限限制。"),
    ("知识产权", "资料说明客户保留输入资料权利，服务商保留平台技术权利。"),
    ("保密义务", "资料说明双方需保护非公开信息，并限制未授权披露。"),
    ("违约责任", "资料说明逾期交付可能触发违约金，但未列明上限和豁免情形。"),
    ("资料缺失", "资料只说明存在补充订单，但没有给出订单正文、附件和适用范围。"),
    ("引用冲突", "资料 A 说明提前三十日通知终止，资料 B 说明提前六十日通知终止，需要人工确认冲突。"),
)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _collection_prompt(*, focus: str, citation: str, excerpt: str) -> str:
    return (
        f"任务：请整理{focus}相关摘要、风险提示、引用依据和人工确认项。\n"
        f"资料引用：{citation}\n"
        f"资料摘录：{excerpt}\n"
        "只基于给定资料回答，不输出法律结论。"
    )


def _suggested_target(*, citation: str, excerpt: str) -> str:
    return (
        f"摘要：资料显示：{excerpt}\n"
        "风险提示：需核对资料完整性、适用范围、附件和冲突条款；只做资料整理和风险提示，不判断合法/违法。\n"
        f"引用依据：{citation}\n"
        "人工确认：不输出法律结论，不能支持最终法律结论；需人工/法务结合完整材料确认。"
    )


def build_phase26_collection_pack(*, count: int = PHASE26_COLLECTION_COUNT) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    for index in range(max(PHASE26_COLLECTION_COUNT, count)):
        focus, excerpt = _COLLECTION_BLUEPRINTS[index % len(_COLLECTION_BLUEPRINTS)]
        source_id = f"phase26-collection-source-{index + 1:03d}"
        chunk_id = f"phase26-collection-chunk-{index + 1:03d}"
        citation = f"[{source_id}:{chunk_id}]"
        prompt = _collection_prompt(focus=focus, citation=citation, excerpt=excerpt)
        metadata = {
            "response_contract": BOUNDARY_CONTRACT_ID,
            "expected_citation": citation,
            "source_excerpt": excerpt,
            "source_id": source_id,
            "chunk_id": chunk_id,
            "phase": "phase26",
            "collection_item": True,
        }
        runtime_response = build_runtime_contract_response(
            messages=[{"role": "user", "content": prompt}],
            metadata=metadata,
            mode=BOUNDARY_CONTRACT_ID,
        )
        items.append(
            {
                "collection_id": f"phase26-collection-{index + 1:03d}",
                "prompt": prompt,
                "messages": [{"role": "user", "content": prompt}],
                "metadata": metadata,
                "runtime_output": runtime_response["output"],
                "runtime_scores": runtime_response["scores"],
                "suggested_target_template": _suggested_target(citation=citation, excerpt=excerpt),
                "actual_feedback_required": True,
                "not_training_data_until_attested_and_approved": True,
            }
        )
    return {
        "kind": "phase26_actual_feedback_collection_pack",
        "collection_count": len(items),
        "minimum_required_approved_actual_candidates": PHASE25_MIN_APPROVED_ACTUAL_CANDIDATES,
        "attestation_template": build_phase25_attestation_template(),
        "items": items,
        "created_at": _utcnow_iso(),
    }


def build_phase26_feedback_batch(payloads: list[Mapping[str, Any]]) -> dict[str, Any]:
    intakes: list[dict[str, Any]] = []
    accepted_signals: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    for index, payload in enumerate(payloads):
        intake = build_phase25_actual_feedback_signal(payload)
        intakes.append({"batch_index": index, **intake})
        if intake.get("status") == "accepted_pending_review" and isinstance(intake.get("signal"), Mapping):
            accepted_signals.append(dict(intake["signal"]))
        else:
            blocked.append(
                {
                    "batch_index": index,
                    "errors": _dict(intake.get("validation")).get("errors") or ["blocked"],
                }
            )
    return {
        "kind": "phase26_actual_feedback_batch",
        "payload_count": len(payloads),
        "accepted_pending_review_count": len(accepted_signals),
        "blocked_count": len(blocked),
        "intakes": intakes,
        "accepted_signals": accepted_signals,
        "blocked": blocked,
        "created_at": _utcnow_iso(),
    }


def build_phase26_probe_readiness(
    *,
    signals: list[Mapping[str, Any]],
    approved_signal_ids: set[str] | None = None,
    excluded_signal_ids: set[str] | None = None,
    local_models: list[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    holdout = build_phase24_holdout(regression_count=50, hard_count=50)
    queue = build_phase25_review_queue(signals)
    reviewed = apply_phase25_review_decisions(
        queue,
        signals,
        approved_signal_ids=approved_signal_ids,
        excluded_signal_ids=excluded_signal_ids,
    )
    routing = build_phase25_routing_report(reviewed, signals)
    holdout_chunk_ids = {str(item.get("chunk_id")) for item in holdout["prompts"] if item.get("chunk_id")}
    candidates = build_phase24_candidate_artifacts(
        signals=signals,
        reviewed=reviewed,
        routing_report=routing,
        holdout_chunk_ids=holdout_chunk_ids,
    )
    integrity = phase24_holdout_integrity_check(
        holdout=holdout,
        sft_samples=candidates["sft_samples"],
        dpo_pairs=candidates["dpo_pairs"],
    )
    runtime_eval = evaluate_phase24_runtime_contract_holdout(holdout)
    runtime_decision = phase24_runtime_product_decision(runtime_eval)
    model_selection = build_phase24_model_selection(local_models=local_models or [])
    readiness = build_phase25_training_readiness_report(
        reviewed=reviewed,
        routing_report=routing,
        candidate_manifest=candidates["candidate_manifest"],
        candidate_quality_report=candidates["quality_report"],
        holdout_integrity=integrity,
        runtime_decision=runtime_decision,
        model_selection=model_selection,
    )
    job_specs = build_phase25_training_job_specs(readiness, model_selection)
    return {
        "kind": "phase26_training_probe_readiness",
        "actual_feedback_count": len(signals),
        "approved_signal_count": len(approved_signal_ids or set()),
        "queue": queue,
        "reviewed": reviewed,
        "routing_report": routing,
        "candidate_artifacts": candidates,
        "holdout_integrity_check": integrity,
        "runtime_eval": runtime_eval,
        "runtime_decision": runtime_decision,
        "model_selection": model_selection,
        "training_readiness": readiness,
        "training_job_specs": job_specs,
        "created_at": _utcnow_iso(),
    }


def build_phase26_empty_state(*, local_models: list[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    collection_pack = build_phase26_collection_pack()
    readiness = build_phase26_probe_readiness(signals=[], local_models=local_models or [])
    return {
        "kind": "phase26_empty_actual_feedback_collection_state",
        "collection_pack": collection_pack,
        "feedback_batch": build_phase26_feedback_batch([]),
        "probe_readiness": readiness,
        "created_at": _utcnow_iso(),
    }


def build_phase26_comparison_summary(state: Mapping[str, Any]) -> dict[str, Any]:
    readiness_payload = _dict(state.get("probe_readiness"))
    readiness = _dict(readiness_payload.get("training_readiness"))
    runtime_eval = _dict(readiness_payload.get("runtime_eval"))
    return {
        "kind": "phase26_comparison_summary",
        "status": "completed",
        "collection_count": _dict(state.get("collection_pack")).get("collection_count", 0),
        "actual_feedback_count": readiness_payload.get("actual_feedback_count", 0),
        "approved_actual_candidate_count": readiness.get("approved_actual_candidate_count", 0),
        "runtime_contract_eval": {
            "holdout_count": runtime_eval.get("holdout_count"),
            "scores": runtime_eval.get("scores"),
            "decision": readiness_payload.get("runtime_decision"),
        },
        "training_readiness": readiness,
        "training_job_specs": readiness_payload.get("training_job_specs"),
        "final_recommendation": (
            "run_qwen3_4b_training_probe"
            if readiness.get("status") == "ready_for_real_training_probe"
            else "collect_and_review_actual_user_feedback"
        ),
        "auto_promotion_allowed": False,
        "feedback_source_policy": "actual user feedback must be attested, manually reviewed, and never synthesized",
        "created_at": _utcnow_iso(),
    }


def summarize_phase26_batch(batch: Mapping[str, Any], readiness: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "kind": "phase26_batch_summary",
        "payload_count": batch.get("payload_count", 0),
        "accepted_pending_review_count": batch.get("accepted_pending_review_count", 0),
        "blocked_count": batch.get("blocked_count", 0),
        "readiness_status": _dict(readiness.get("training_readiness")).get("status"),
        "readiness_blockers": _dict(readiness.get("training_readiness")).get("blockers") or [],
        "state_counts": _dict(readiness.get("reviewed")).get("state_counts") or {},
        "created_at": _utcnow_iso(),
    }


__all__ = [
    "PHASE26_COLLECTION_COUNT",
    "PHASE26_KIND",
    "build_phase26_collection_pack",
    "build_phase26_comparison_summary",
    "build_phase26_empty_state",
    "build_phase26_feedback_batch",
    "build_phase26_probe_readiness",
    "summarize_phase26_batch",
]
