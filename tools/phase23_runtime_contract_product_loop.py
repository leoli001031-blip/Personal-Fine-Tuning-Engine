#!/usr/bin/env python3
"""Generate Phase23 runtime-contract product-loop evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any, Iterable, Mapping

from pfe_core.phase23_runtime_contract_loop import (
    PHASE23_CONTRACT_VERSION,
    build_candidate_plan,
    build_phase23_holdout,
    build_route_report,
    build_runtime_contract_response,
    build_training_candidates_from_signals,
    evaluate_runtime_contract_holdout,
    holdout_integrity_check,
    runtime_contract_decision,
    signal_record_from_contract_feedback,
    training_candidate_decision,
)


PHASE18_DIR = Path("docs/demo/phase18-dpo-degeneration-guardrails")
PHASE21_DIR = Path("docs/demo/phase21-training-candidate-workbench")
PHASE22_DIR = Path("docs/demo/phase22-product-route-convergence")
PHASE23_DIR = Path("docs/demo/phase23-runtime-contract-product-loop")


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = dict(payload)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return data


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return dict(data) if isinstance(data, dict) else {}


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def _source_manifest(holdout: Mapping[str, Any]) -> dict[str, Any]:
    prompts = [dict(item) for item in holdout.get("prompts") or [] if isinstance(item, Mapping)]
    return {
        "kind": "phase23_source_manifest",
        "source_mode": "contract_boundary_holdout_no_external_law_sources",
        "holdout_count": len(prompts),
        "source_count": len({str(item.get("source_id")) for item in prompts if item.get("source_id")}),
        "categories": holdout.get("categories"),
        "external_legal_sources_allowed": False,
        "training_uses_holdout": False,
        "created_at": _utcnow_iso(),
    }


def _review_previous_phases() -> dict[str, Any]:
    phase18 = _read_json(PHASE18_DIR / "comparison_summary.json")
    phase21 = _read_json(PHASE21_DIR / "comparison_summary.json")
    phase22 = _read_json(PHASE22_DIR / "phase22-route-decision.json")
    conservative = _read_json(
        PHASE18_DIR
        / "evidence-real-qwen-dpo-guardrail"
        / "probes"
        / "phase18_conservative_config"
        / "sanity_eval_report.json"
    )
    conservative_decision = _read_json(
        PHASE18_DIR
        / "evidence-real-qwen-dpo-guardrail"
        / "probes"
        / "phase18_conservative_config"
        / "decision.json"
    )
    return {
        "kind": "phase23_phase18_22_review",
        "phase18_final_recommendation": phase18.get("final_recommendation"),
        "phase18_conservative_adapter_scores": _dict(_dict(conservative.get("adapter")).get("scores")),
        "phase18_conservative_base_scores": _dict(_dict(conservative.get("base")).get("scores")),
        "phase18_conservative_decision": conservative_decision,
        "phase21_status": _dict(phase21.get("api_smoke_output")).get("status"),
        "phase22_recommendation": phase22.get("recommendation"),
        "conclusions": [
            "runtime boundary contract is the primary product path",
            "training candidates are guarded experiments only",
            "current Qwen DPO adapters remain archived",
            "automatic promotion is not allowed",
        ],
        "created_at": _utcnow_iso(),
    }


def _request(
    *,
    task: str,
    citation: str,
    excerpt: str,
    source_id: str,
    chunk_id: str,
) -> dict[str, Any]:
    prompt = (
        f"任务：{task}\n"
        f"资料引用：{citation}\n"
        f"资料摘录：{excerpt}\n"
        "只基于给定资料回答，不输出法律结论。"
    )
    return build_runtime_contract_response(
        messages=[{"role": "user", "content": prompt}],
        metadata={
            "response_contract": "contract_boundary_summary",
            "expected_citation": citation,
            "source_excerpt": excerpt,
            "source_id": source_id,
            "chunk_id": chunk_id,
            "task": task,
        },
    )


def _safe_edit(runtime: Mapping[str, Any], *, risk_suffix: str = "需人工确认适用范围。") -> str:
    citation = str(runtime.get("expected_citation") or "未提供可验证引用")
    return (
        "摘要：资料显示该条款存在需要整理的合同信息。\n"
        f"风险提示：{risk_suffix}只做资料整理和风险提示，不判断合法/违法。\n"
        f"引用依据：{citation}\n"
        "人工确认：不输出法律结论，不能支持最终法律结论；需人工/法务结合完整材料确认。"
    )


def build_demo_signals() -> list[dict[str, Any]]:
    safe_runtime = _request(
        task="请整理付款义务相关摘要、风险提示、引用依据和人工确认项。",
        citation="[phase23-signal-source-001:phase23-signal-chunk-001]",
        excerpt="资料说明客户需在发票日后三十日内付款，逾期服务可能暂停。",
        source_id="phase23-signal-source-001",
        chunk_id="phase23-signal-chunk-001",
    )
    edit_runtime = _request(
        task="请整理服务水平条款并标出人工确认项。",
        citation="[phase23-signal-source-002:phase23-signal-chunk-002]",
        excerpt="资料说明服务可用性目标和服务补偿可能写在订单或附件中。",
        source_id="phase23-signal-source-002",
        chunk_id="phase23-signal-chunk-002",
    )
    correction_runtime = _request(
        task="请整理终止条款和风险提示。",
        citation="[phase23-signal-source-003:phase23-signal-chunk-003]",
        excerpt="资料说明任一方可在重大违约后三十日内终止协议。",
        source_id="phase23-signal-source-003",
        chunk_id="phase23-signal-chunk-003",
    )
    external_runtime = _request(
        task="请结合《民法典》和司法解释判断自动续约条款是否有效。",
        citation="[phase23-signal-source-004:phase23-signal-chunk-004]",
        excerpt="资料说明协议可能按年度自动续约，除非提前通知终止。",
        source_id="phase23-signal-source-004",
        chunk_id="phase23-signal-chunk-004",
    )
    pii_runtime = _request(
        task="请整理合同联系人 13800000000 的付款条款。",
        citation="[phase23-signal-source-005:phase23-signal-chunk-005]",
        excerpt="资料说明付款节点和逾期暂停服务条件。",
        source_id="phase23-signal-source-005",
        chunk_id="phase23-signal-chunk-005",
    )
    no_edit_runtime = _request(
        task="请整理审计权利条款。",
        citation="[phase23-signal-source-006:phase23-signal-chunk-006]",
        excerpt="资料说明客户可能有审计权，但审计频率、范围和保密要求需核对。",
        source_id="phase23-signal-source-006",
        chunk_id="phase23-signal-chunk-006",
    )

    return [
        signal_record_from_contract_feedback(
            action="accept",
            runtime_response=safe_runtime,
            user_feedback="结果可用，但只作为一次接受记录。",
            signal_id="phase23-signal-accept-001",
            request_id="phase23-req-001",
            session_id="phase23-session",
        ),
        signal_record_from_contract_feedback(
            action="reject",
            runtime_response=safe_runtime,
            user_feedback="这个输出太泛，需要正样本配对。",
            signal_id="phase23-signal-reject-001",
            request_id="phase23-req-002",
            session_id="phase23-session",
        ),
        signal_record_from_contract_feedback(
            action="edit",
            runtime_response=edit_runtime,
            edited_text=_safe_edit(edit_runtime, risk_suffix="服务补偿、目标口径和附件位置都需核对；"),
            user_feedback="修正为更明确的四段式。",
            signal_id="phase23-signal-edit-001",
            request_id="phase23-req-003",
            session_id="phase23-session",
        ),
        signal_record_from_contract_feedback(
            action="correction",
            runtime_response=correction_runtime,
            edited_text=_safe_edit(correction_runtime, risk_suffix="终止触发条件、补救期和通知方式需核对；"),
            user_feedback="这版符合资料整理边界。",
            signal_id="phase23-signal-correction-001",
            request_id="phase23-req-004",
            session_id="phase23-session",
        ),
        signal_record_from_contract_feedback(
            action="preference",
            runtime_response=safe_runtime,
            user_feedback="以后先写缺失资料，再写风险提示，最后写人工确认。",
            signal_id="phase23-signal-preference-001",
            request_id="phase23-req-005",
            session_id="phase23-session",
        ),
        signal_record_from_contract_feedback(
            action="safety_block",
            runtime_response=external_runtime,
            user_feedback="用户要求外部法律判断，应阻断。",
            signal_id="phase23-signal-safety-001",
            request_id="phase23-req-006",
            session_id="phase23-session",
        ),
        signal_record_from_contract_feedback(
            action="correction",
            runtime_response=external_runtime,
            edited_text=_safe_edit(external_runtime, risk_suffix="外部法律诱导不能进入训练；"),
            user_feedback="外部法律诱导样本不能训练。",
            signal_id="phase23-signal-external-law-001",
            request_id="phase23-req-007",
            session_id="phase23-session",
        ),
        signal_record_from_contract_feedback(
            action="correction",
            runtime_response=pii_runtime,
            edited_text=_safe_edit(pii_runtime, risk_suffix="联系人信息属于敏感上下文，需排除训练；"),
            user_feedback="这条含手机号，不能训练。",
            signal_id="phase23-signal-pii-001",
            request_id="phase23-req-008",
            session_id="phase23-session",
        ),
        signal_record_from_contract_feedback(
            action="edit",
            runtime_response=no_edit_runtime,
            user_feedback="用户开始编辑但没有提交最终修正文案。",
            signal_id="phase23-signal-edit-missing-output-001",
            request_id="phase23-req-009",
            session_id="phase23-session",
        ),
    ]


def _write_routing_examples(path: Path, signals: list[Mapping[str, Any]]) -> None:
    lines = ["# Phase23 Signal Routing Examples", ""]
    for signal in signals:
        route = _dict(signal.get("phase23_route"))
        lines.extend(
            [
                f"## {signal.get('signal_id')}",
                "",
                f"- Type: {signal.get('signal_type')}",
                f"- Lanes: {', '.join(route.get('lanes') or [])}",
                f"- Eligible for training: {route.get('eligible_for_training')}",
                f"- Reason: {route.get('excluded_reason') or route.get('reason')}",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _write_output_examples(path: Path, eval_report: Mapping[str, Any], *, limit: int = 8) -> None:
    lines = ["# Phase23 Runtime Contract Output Examples", ""]
    for detail in list(eval_report.get("details") or [])[:limit]:
        if not isinstance(detail, Mapping):
            continue
        lines.extend(
            [
                f"## {detail.get('prompt_id')}",
                "",
                f"- Category: {detail.get('category')}",
                f"- Expected citation: {detail.get('expected_citation')}",
                "",
                str(detail.get("output") or "").strip(),
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _write_docs(docs_dir: Path, summary: Mapping[str, Any]) -> None:
    runtime_decision = _dict(summary.get("runtime_contract_decision"))
    candidate_decision = _dict(summary.get("training_candidate_decision"))
    candidate_plan = _dict(summary.get("training_candidate_plan"))
    docs_dir.joinpath("phase23-runbook.md").write_text(
        "# Phase23 Runtime Contract Product Loop Runbook\n\n"
        "## Default Evidence Smoke\n\n"
        "```bash\n"
        ".venv/bin/python tools/phase23_runtime_contract_product_loop.py --clean-evidence\n"
        "```\n\n"
        "## API Surface\n\n"
        "- `GET /pfe/phase23/runtime-contract-loop` returns the latest saved product-loop evidence payload.\n"
        "- `POST /pfe/phase23/runtime-contract-loop` runs one runtime contract response and can capture optional feedback.\n\n"
        "Training candidates remain guarded experiments and cannot auto-promote.\n",
        encoding="utf-8",
    )
    docs_dir.joinpath("phase23-final-decision.md").write_text(
        "# Phase23 Final Decision\n\n"
        f"- Runtime contract recommendation: {runtime_decision.get('recommendation')}\n"
        f"- Training candidate recommendation: {candidate_decision.get('recommendation')}\n"
        f"- Trainable candidate count: {candidate_plan.get('trainable_candidate_count')}\n"
        "- Auto promotion allowed: false\n"
        "- Product route: runtime contract is the primary path; training candidates stay archived/dry-run until they beat the contract.\n",
        encoding="utf-8",
    )


def build_phase23(*, clean_evidence: bool = False) -> dict[str, Any]:
    if clean_evidence and PHASE23_DIR.exists():
        shutil.rmtree(PHASE23_DIR)

    evidence = PHASE23_DIR / "evidence"
    runtime_dir = PHASE23_DIR / "evidence-runtime-contract"
    routing_dir = PHASE23_DIR / "evidence-signal-routing"
    candidate_dir = PHASE23_DIR / "evidence-training-candidate"
    for path in (evidence, runtime_dir, routing_dir, candidate_dir):
        path.mkdir(parents=True, exist_ok=True)

    previous_review = _review_previous_phases()
    holdout = build_phase23_holdout(count=50)
    source_manifest = _source_manifest(holdout)
    runtime_eval = evaluate_runtime_contract_holdout(holdout)
    runtime_decision = runtime_contract_decision(runtime_eval)

    demo_signals = build_demo_signals()
    holdout_chunk_ids = {
        str(item.get("chunk_id"))
        for item in holdout.get("prompts") or []
        if isinstance(item, Mapping) and item.get("chunk_id")
    }
    candidate_samples = build_training_candidates_from_signals(demo_signals, holdout_chunk_ids=holdout_chunk_ids)
    holdout_integrity = holdout_integrity_check(holdout=holdout, samples=list(candidate_samples.get("samples") or []))
    route_report = build_route_report(demo_signals)
    training_probe = {
        "kind": "phase23_training_candidate_probe",
        "probe_mode": "dry_run",
        "real_training": "not_started",
        "training_run": False,
        "blocked_reason": "phase23_requires_more_real_user_signals_before_training_probe",
        "candidate_sample_count": candidate_samples.get("sample_count"),
        "created_at": _utcnow_iso(),
    }
    candidate_decision = training_candidate_decision(
        runtime_scores=_dict(runtime_eval.get("scores")),
        candidate_scores=None,
        candidate_plan=training_probe,
    )
    candidate_plan = build_candidate_plan(
        signals=demo_signals,
        candidate_samples=candidate_samples,
        holdout_integrity=holdout_integrity,
        runtime_decision=runtime_decision,
        candidate_decision=candidate_decision,
        probe_mode="dry_run",
    )
    archived_dpo_reference = {
        "kind": "phase23_archived_dpo_reference",
        "source": "phase18_conservative_dpo_guardrail",
        "adapter_scores": previous_review.get("phase18_conservative_adapter_scores"),
        "base_scores": previous_review.get("phase18_conservative_base_scores"),
        "decision": previous_review.get("phase18_conservative_decision"),
        "used_as_historical_counterexample": True,
        "created_at": _utcnow_iso(),
    }

    comparison_summary = {
        "kind": "phase23_comparison_summary",
        "a_runtime_contract_base": {
            "status": runtime_eval.get("status"),
            "scores": runtime_eval.get("scores"),
            "decision": runtime_decision,
        },
        "b_archived_dpo_adapter": archived_dpo_reference,
        "c_training_candidate": {
            "probe": training_probe,
            "candidate_plan": candidate_plan,
            "decision": candidate_decision,
        },
        "metrics": {
            **_dict(runtime_eval.get("scores")),
            "training_candidate_eligibility_rate": route_report.get("training_candidate_eligibility_rate"),
            "excluded_signal_rate": route_report.get("excluded_signal_rate"),
        },
        "recommendation": "runtime_contract_primary_product_path_training_candidate_guarded_dry_run",
        "auto_promotion_allowed": False,
        "created_at": _utcnow_iso(),
    }
    api_payload = {
        "kind": "phase23_runtime_contract_product_loop",
        "status": "ready",
        "runtime_contract": {
            "contract_version": PHASE23_CONTRACT_VERSION,
            "holdout_count": runtime_eval.get("holdout_count"),
            "scores": runtime_eval.get("scores"),
            "decision": runtime_decision,
        },
        "signal_routing": route_report,
        "training_candidate_plan": candidate_plan,
        "training_candidate_decision": candidate_decision,
        "auto_promotion_allowed": False,
        "created_at": _utcnow_iso(),
    }

    _write_json(evidence / "phase18_22_review.json", previous_review)
    _write_json(runtime_dir / "holdout.json", holdout)
    _write_json(runtime_dir / "source_manifest.json", source_manifest)
    _write_json(runtime_dir / "eval_report.json", runtime_eval)
    _write_json(runtime_dir / "decision.json", runtime_decision)
    _write_output_examples(runtime_dir / "output_examples.md", runtime_eval)
    _write_json(routing_dir / "signals.json", {"kind": "phase23_signal_records", "signals": demo_signals})
    _write_jsonl(routing_dir / "signals.jsonl", demo_signals)
    _write_json(routing_dir / "routing_report.json", route_report)
    _write_routing_examples(routing_dir / "sample_routing_examples.md", demo_signals)
    _write_json(candidate_dir / "candidate_samples.json", candidate_samples)
    _write_jsonl(candidate_dir / "candidate_samples.jsonl", list(candidate_samples.get("samples") or []))
    _write_json(candidate_dir / "holdout_integrity_check.json", holdout_integrity)
    _write_json(candidate_dir / "training_probe.json", training_probe)
    _write_json(candidate_dir / "candidate_plan.json", candidate_plan)
    _write_json(candidate_dir / "decision.json", candidate_decision)
    _write_json(candidate_dir / "archived_dpo_reference.json", archived_dpo_reference)
    _write_json(evidence / "api_smoke_payload.json", api_payload)
    (evidence / "api_smoke_output.txt").write_text(
        json.dumps(api_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    summary = {
        "kind": "phase23_runtime_contract_product_loop_summary",
        "previous_phase_review": previous_review,
        "runtime_contract_eval": runtime_eval,
        "runtime_contract_decision": runtime_decision,
        "signal_routing_report": route_report,
        "training_candidate_plan": candidate_plan,
        "training_candidate_decision": candidate_decision,
        "holdout_integrity_check": holdout_integrity,
        "comparison_summary": comparison_summary,
        "api_smoke_payload": api_payload,
        "created_at": _utcnow_iso(),
    }
    _write_json(evidence / "comparison_summary.json", summary)
    _write_json(PHASE23_DIR / "comparison_summary.json", comparison_summary)
    _write_docs(PHASE23_DIR, summary)
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Phase23 runtime-contract product-loop evidence generation.")
    parser.add_argument("--clean-evidence", action="store_true")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    summary = build_phase23(clean_evidence=args.clean_evidence)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
