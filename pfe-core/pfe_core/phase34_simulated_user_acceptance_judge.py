"""Phase34 simulated real-user acceptance judge primitives."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import re
from typing import Any, Iterable, Mapping

from pfe_core.phase32_personal_agent_preference import contains_raw_private_text, write_jsonl
from pfe_core.phase33_simulated_usage_replay import (
    PHASE33_FEEDBACK_SOURCE,
    build_phase33_transcripts,
    build_phase33_usage_sessions,
)


PHASE34_KIND = "phase34_simulated_real_user_acceptance_judge"
PHASE34_FEEDBACK_SOURCE = "simulated_user_judgement"
PHASE34_MIN_SCENARIOS = 80
PHASE34_MAX_SCENARIOS = 120

PHASE34_VALUE_METRICS = (
    "acceptance_rate",
    "adapter_win_rate",
    "base_win_rate",
    "correction_recovery_rate",
    "evidence_trust_rate",
    "user_effort_reduction_rate",
    "frustration_reduction_rate",
    "false_completion_penalty_rate",
    "privacy_boundary_trust_rate",
    "would_continue_using_rate",
    "overall_product_value_score",
)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _stable_id(*parts: str, length: int = 12) -> str:
    digest = hashlib.sha256("\n".join(str(part) for part in parts).encode("utf-8")).hexdigest()
    return digest[:length]


def _scenario_count(count: int) -> int:
    return max(PHASE34_MIN_SCENARIOS, min(PHASE34_MAX_SCENARIOS, int(count)))


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


def _score_response(transcript: Mapping[str, Any]) -> dict[str, float]:
    text = _assistant_text(transcript)
    correction = _stage_text(transcript, "agent_correction_response")
    final = _stage_text(transcript, "agent_final_response")
    raw_leak = contains_raw_private_text(transcript)
    false_completion = bool(re.search(r"已提交|已推送|PR 已开|Fast beta gate 已通过|已经关掉", text))
    verbose = bool(len(text) > 1600 or re.search(r"首先.*其次.*最后|宏观|底层逻辑|这个问题需要从", text))
    evidence = bool(re.search(r"证据|命令|输出|计数|测试|transcripts|comparison_summary|decision|PID|端口|gate|PR|diff", text))
    execution = bool(re.search(r"先|检查|执行|跑|生成|提交|关闭|确认", text))
    correction_recovery = bool(re.search(r"你说得对|转回|最新意图|收敛|不再", correction))
    boundary = bool(re.search(r"隐私|脱敏|不提交|原始|simulated_usage|人工复核|不自动 promote|边界", text))
    local = bool(re.search(r"git status|diff|工作区|分支|进程|PID|端口|服务|模型|日志|PFE|Hermes", text))
    final_acceptance = bool(re.search(r"验收|transcripts|评分|comparison_summary|decision|人工复核|blocked", final))
    concise = 120 <= len(text) <= 1300 and not verbose
    positive = {
        "understood_intent_score": 1.0 if execution or correction_recovery else 0.0,
        "conciseness_score": 1.0 if concise else 0.0,
        "execution_progress_score": 1.0 if execution else 0.0,
        "evidence_trust_score": 1.0 if evidence else 0.0,
        "correction_recovery_score": 1.0 if correction_recovery else 0.0,
        "privacy_boundary_score": 1.0 if boundary else 0.0,
        "local_context_score": 1.0 if local else 0.0,
        "final_acceptance_score": 1.0 if final_acceptance else 0.0,
    }
    bad = {
        "false_completion_penalty": 1.0 if false_completion else 0.0,
        "raw_private_text_leak": 1.0 if raw_leak else 0.0,
        "verbosity_penalty": 1.0 if verbose else 0.0,
    }
    perceived = sum(positive.values()) / len(positive)
    trust = (
        positive["evidence_trust_score"]
        + positive["privacy_boundary_score"]
        + (1.0 - bad["false_completion_penalty"])
        + (1.0 - bad["raw_private_text_leak"])
    ) / 4
    effort_reduction = (
        positive["execution_progress_score"]
        + positive["correction_recovery_score"]
        + positive["final_acceptance_score"]
        + positive["conciseness_score"]
    ) / 4
    frustration = (
        (1.0 - positive["correction_recovery_score"])
        + bad["verbosity_penalty"]
        + bad["false_completion_penalty"]
    ) / 3
    overall = max(0.0, (perceived + trust + effort_reduction + (1.0 - frustration)) / 4 - 0.25 * bad["raw_private_text_leak"])
    return {
        **positive,
        **bad,
        "perceived_value_score": round(perceived, 3),
        "trust_score": round(trust, 3),
        "user_effort_reduction_score": round(effort_reduction, 3),
        "frustration_score": round(frustration, 3),
        "overall_product_value_score": round(overall, 3),
        "would_continue_using": 1.0 if overall >= 0.65 and not raw_leak else 0.0,
    }


def build_phase34_phase33_review(*, phase33_summary: Mapping[str, Any], phase33_decision_text: str = "") -> dict[str, Any]:
    eval_report = _dict(phase33_summary.get("eval_report"))
    return {
        "kind": "phase34_phase33_review",
        "phase33_completed": phase33_summary.get("status") == "completed",
        "phase33_session_count": phase33_summary.get("session_count"),
        "phase33_actual_user_feedback_count": phase33_summary.get("actual_user_feedback_count"),
        "phase33_final_recommendation": phase33_summary.get("final_recommendation"),
        "phase33_base_scores": _dict(_dict(eval_report.get("base")).get("scores")),
        "phase33_adapter_scores": _dict(_dict(eval_report.get("adapter")).get("scores")),
        "phase33_score_delta": _dict(eval_report.get("score_delta")),
        "phase33_decision_summary": phase33_decision_text[:1800],
        "phase34_scope": "simulated_user_judgement_only_no_training_no_actual_feedback",
        "phase34_does_not_train": True,
        "phase34_does_not_auto_promote": True,
        "phase34_does_not_collect_actual_feedback": True,
        "created_at": _utcnow_iso(),
    }


def build_phase34_acceptance_scenarios(*, count: int = 100, phase33_reference: Mapping[str, Any] | None = None) -> dict[str, Any]:
    target_count = _scenario_count(count)
    phase33_batch = build_phase33_usage_sessions(count=target_count)
    scenarios: list[dict[str, Any]] = []
    for index, session in enumerate(phase33_batch["sessions"], start=1):
        scenario_id = f"phase34-scenario-{index:03d}-{_stable_id(session['session_id'], str(index), length=8)}"
        scenarios.append(
            {
                "kind": "phase34_simulated_acceptance_scenario",
                "scenario_id": scenario_id,
                "source_session_id": session["session_id"],
                "source": PHASE34_FEEDBACK_SOURCE,
                "feedback_source": PHASE34_FEEDBACK_SOURCE,
                "simulated_user_judgement": True,
                "confirmed_actual_user_feedback": False,
                "not_actual_user_feedback": True,
                "not_for_training": True,
                "workflow_category": session["workflow_category"],
                "user_intent": session["user_goal"],
                "expected_outcome": session["final_acceptance"],
                "user_correction": session["user_correction"],
                "continuation_need": session["continue_request"],
                "expected_taxonomy": list(session.get("expected_taxonomy") or []),
                "acceptance_lens": [
                    "understood_user_intent",
                    "less_verbose",
                    "made_progress",
                    "evidence_first",
                    "correction_recovery",
                    "privacy_boundary",
                    "reduced_supervision_cost",
                    "no_false_completion",
                ],
                "source_phase33_reference": {
                    "phase33_final_recommendation": _dict(phase33_reference).get("final_recommendation"),
                    "phase33_session_count": _dict(phase33_reference).get("session_count"),
                },
            }
        )
    return {
        "kind": "phase34_acceptance_scenario_batch",
        "source": PHASE34_FEEDBACK_SOURCE,
        "simulated_user_judgement": True,
        "actual_user_feedback_count": 0,
        "scenario_count": len(scenarios),
        "scenario_count_within_required_range": PHASE34_MIN_SCENARIOS <= len(scenarios) <= PHASE34_MAX_SCENARIOS,
        "categories": dict(sorted(Counter(item["workflow_category"] for item in scenarios).items())),
        "scenarios": scenarios,
        "created_at": _utcnow_iso(),
    }


def build_phase34_blind_eval_pairs(
    *,
    scenarios: Iterable[Mapping[str, Any]],
    base_transcripts: Iterable[Mapping[str, Any]],
    adapter_transcripts: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    base_by_session = {str(item.get("session_id")): item for item in base_transcripts}
    adapter_by_session = {str(item.get("session_id")): item for item in adapter_transcripts}
    pairs: list[dict[str, Any]] = []
    for index, scenario in enumerate(scenarios, start=1):
        session_id = str(scenario.get("source_session_id"))
        base = base_by_session[session_id]
        adapter = adapter_by_session[session_id]
        adapter_is_a = index % 2 == 0
        variant_a = adapter if adapter_is_a else base
        variant_b = base if adapter_is_a else adapter
        pairs.append(
            {
                "kind": "phase34_blind_eval_pair",
                "pair_id": f"phase34-blind-pair-{index:03d}-{_stable_id(session_id, str(index), length=8)}",
                "source": PHASE34_FEEDBACK_SOURCE,
                "feedback_source": PHASE34_FEEDBACK_SOURCE,
                "simulated_user_judgement": True,
                "confirmed_actual_user_feedback": False,
                "not_actual_user_feedback": True,
                "scenario_id": scenario.get("scenario_id"),
                "source_session_id": session_id,
                "workflow_category": scenario.get("workflow_category"),
                "user_intent": scenario.get("user_intent"),
                "expected_outcome": scenario.get("expected_outcome"),
                "user_correction": scenario.get("user_correction"),
                "continuation_need": scenario.get("continuation_need"),
                "variant_a": {
                    "label": "variant_a",
                    "agent_response": _assistant_text(variant_a),
                    "turns": list(variant_a.get("turns") or []),
                },
                "variant_b": {
                    "label": "variant_b",
                    "agent_response": _assistant_text(variant_b),
                    "turns": list(variant_b.get("turns") or []),
                },
                "blind_variant_map": {
                    "variant_a": "adapter" if adapter_is_a else "base",
                    "variant_b": "base" if adapter_is_a else "adapter",
                },
            }
        )
    return pairs


def blind_pair_public_view(pair: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in dict(pair).items()
        if key != "blind_variant_map"
    }


def validate_phase34_blind_pair(pair: Mapping[str, Any]) -> dict[str, Any]:
    public = json.dumps(blind_pair_public_view(pair), ensure_ascii=False, sort_keys=True)
    leaked = bool(
        re.search(
            r'"model_variant"|phase32_adapter_profile_source|phase32_base_eval|phase32_real_adapter_eval|"preferred_model_after_unblind"',
            public,
            re.I,
        )
    )
    return {
        "kind": "phase34_blind_pair_validation",
        "passed": not leaked and pair.get("feedback_source") == PHASE34_FEEDBACK_SOURCE,
        "identity_leaked_to_judge": leaked,
        "created_at": _utcnow_iso(),
    }


def judge_phase34_blind_pair(pair: Mapping[str, Any]) -> dict[str, Any]:
    public_pair = blind_pair_public_view(pair)
    variant_a = _dict(public_pair.get("variant_a"))
    variant_b = _dict(public_pair.get("variant_b"))
    score_a = _score_response({"turns": variant_a.get("turns") or []})
    score_b = _score_response({"turns": variant_b.get("turns") or []})
    preferred = "variant_a"
    if score_b["overall_product_value_score"] > score_a["overall_product_value_score"]:
        preferred = "variant_b"
    elif score_b["overall_product_value_score"] == score_a["overall_product_value_score"]:
        preferred = "tie"
    chosen_score = score_a if preferred == "variant_a" else score_b if preferred == "variant_b" else score_a
    acceptance_decision = "accept"
    if chosen_score["raw_private_text_leak"]:
        acceptance_decision = "blocked"
    elif chosen_score["overall_product_value_score"] < 0.45:
        acceptance_decision = "reject"
    elif chosen_score["overall_product_value_score"] < 0.65:
        acceptance_decision = "needs_edit"
    preferred_identity = _dict(pair.get("blind_variant_map")).get(preferred, "tie") if preferred != "tie" else "tie"
    reason = "更少监督成本、纠正后能转向，并且有证据/边界意识。"
    if preferred_identity == "base":
        reason = "两个回答都可用，但 base 在该场景没有明显更差。"
    if preferred == "tie":
        reason = "两个回答的用户价值评分一致。"
    return {
        "kind": "phase34_simulated_user_judgement",
        "judge_id": "phase34-simulated-real-user-judge",
        "pair_id": pair.get("pair_id"),
        "scenario_id": pair.get("scenario_id"),
        "source": PHASE34_FEEDBACK_SOURCE,
        "feedback_source": PHASE34_FEEDBACK_SOURCE,
        "simulated_user_judgement": True,
        "confirmed_actual_user_feedback": False,
        "not_actual_user_feedback": True,
        "not_for_training": True,
        "user_intent": pair.get("user_intent"),
        "expected_outcome": pair.get("expected_outcome"),
        "agent_response": {
            "variant_a": variant_a.get("agent_response"),
            "variant_b": variant_b.get("agent_response"),
        },
        "user_correction": pair.get("user_correction"),
        "continuation_need": pair.get("continuation_need"),
        "preferred_variant": preferred,
        "preferred_model_after_unblind": preferred_identity,
        "acceptance_decision": acceptance_decision,
        "acceptance_reason": reason,
        "variant_scores": {
            "variant_a": score_a,
            "variant_b": score_b,
        },
        "perceived_value_score": chosen_score["perceived_value_score"],
        "trust_score": chosen_score["trust_score"],
        "user_effort_reduction_score": chosen_score["user_effort_reduction_score"],
        "frustration_score": chosen_score["frustration_score"],
        "would_continue_using": bool(chosen_score["would_continue_using"]),
        "created_at": _utcnow_iso(),
    }


def aggregate_phase34_judgements(judgements: list[Mapping[str, Any]]) -> dict[str, Any]:
    if not judgements:
        return {"kind": "phase34_acceptance_scores", **{metric: 0.0 for metric in PHASE34_VALUE_METRICS}}
    model_rows = {"base": [], "adapter": []}
    preferred_counts = Counter(str(item.get("preferred_model_after_unblind")) for item in judgements)
    for judgement in judgements:
        variant_scores = _dict(judgement.get("variant_scores"))
        pair_map = {}
        preferred = str(judgement.get("preferred_variant"))
        if preferred != "tie":
            pair_map[preferred] = str(judgement.get("preferred_model_after_unblind"))
            pair_map["variant_b" if preferred == "variant_a" else "variant_a"] = "base" if pair_map[preferred] == "adapter" else "adapter"
        else:
            # Alternating blind pairs make this fallback deterministic enough for ties.
            pair_map = {"variant_a": "base", "variant_b": "adapter"}
        for variant, scores in variant_scores.items():
            model = pair_map.get(variant)
            if model in model_rows:
                model_rows[model].append(_dict(scores))
    def avg(model: str, metric: str) -> float:
        rows = model_rows[model]
        if not rows:
            return 0.0
        return round(sum(float(row.get(metric, 0.0)) for row in rows) / len(rows), 3)
    def rate(items: Iterable[Mapping[str, Any]], predicate: Any) -> float:
        rows = list(items)
        if not rows:
            return 0.0
        return round(sum(1 for row in rows if predicate(row)) / len(rows), 3)
    base_rows = model_rows["base"]
    adapter_rows = model_rows["adapter"]
    base_scores = {
        "acceptance_rate": rate(base_rows, lambda row: float(row.get("overall_product_value_score", 0.0)) >= 0.65),
        "correction_recovery_rate": avg("base", "correction_recovery_score"),
        "evidence_trust_rate": avg("base", "evidence_trust_score"),
        "user_effort_reduction_rate": avg("base", "user_effort_reduction_score"),
        "frustration_score": avg("base", "frustration_score"),
        "frustration_reduction_rate": round(1.0 - avg("base", "frustration_score"), 3),
        "false_completion_penalty_rate": avg("base", "false_completion_penalty"),
        "privacy_boundary_trust_rate": avg("base", "privacy_boundary_score"),
        "would_continue_using_rate": avg("base", "would_continue_using"),
        "overall_product_value_score": avg("base", "overall_product_value_score"),
    }
    adapter_scores = {
        "acceptance_rate": rate(adapter_rows, lambda row: float(row.get("overall_product_value_score", 0.0)) >= 0.65),
        "correction_recovery_rate": avg("adapter", "correction_recovery_score"),
        "evidence_trust_rate": avg("adapter", "evidence_trust_score"),
        "user_effort_reduction_rate": avg("adapter", "user_effort_reduction_score"),
        "frustration_score": avg("adapter", "frustration_score"),
        "frustration_reduction_rate": round(1.0 - avg("adapter", "frustration_score"), 3),
        "false_completion_penalty_rate": avg("adapter", "false_completion_penalty"),
        "privacy_boundary_trust_rate": avg("adapter", "privacy_boundary_score"),
        "would_continue_using_rate": avg("adapter", "would_continue_using"),
        "overall_product_value_score": avg("adapter", "overall_product_value_score"),
    }
    adapter_win_rate = round(preferred_counts["adapter"] / len(judgements), 3)
    base_win_rate = round(preferred_counts["base"] / len(judgements), 3)
    return {
        "kind": "phase34_acceptance_scores",
        "judgement_count": len(judgements),
        "preferred_counts": dict(sorted(preferred_counts.items())),
        "adapter_win_rate": adapter_win_rate,
        "base_win_rate": base_win_rate,
        "tie_rate": round(preferred_counts["tie"] / len(judgements), 3),
        "base": base_scores,
        "adapter": adapter_scores,
        "score_delta": {
            "acceptance_rate": round(adapter_scores["acceptance_rate"] - base_scores["acceptance_rate"], 3),
            "adapter_win_rate": round(adapter_win_rate - base_win_rate, 3),
            "correction_recovery_rate": round(adapter_scores["correction_recovery_rate"] - base_scores["correction_recovery_rate"], 3),
            "evidence_trust_rate": round(adapter_scores["evidence_trust_rate"] - base_scores["evidence_trust_rate"], 3),
            "user_effort_reduction_rate": round(adapter_scores["user_effort_reduction_rate"] - base_scores["user_effort_reduction_rate"], 3),
            "frustration_score": round(adapter_scores["frustration_score"] - base_scores["frustration_score"], 3),
            "frustration_reduction_rate": round(adapter_scores["frustration_reduction_rate"] - base_scores["frustration_reduction_rate"], 3),
            "false_completion_penalty_rate": round(adapter_scores["false_completion_penalty_rate"] - base_scores["false_completion_penalty_rate"], 3),
            "privacy_boundary_trust_rate": round(adapter_scores["privacy_boundary_trust_rate"] - base_scores["privacy_boundary_trust_rate"], 3),
            "would_continue_using_rate": round(adapter_scores["would_continue_using_rate"] - base_scores["would_continue_using_rate"], 3),
            "overall_product_value_score": round(adapter_scores["overall_product_value_score"] - base_scores["overall_product_value_score"], 3),
        },
    }


def phase34_final_decision(*, acceptance_scores: Mapping[str, Any], boundary_check: Mapping[str, Any]) -> dict[str, Any]:
    base = _dict(acceptance_scores.get("base"))
    adapter = _dict(acceptance_scores.get("adapter"))
    reasons: list[str] = []
    if not boundary_check.get("passed"):
        reasons.append("simulation_boundary_failed")
    if float(acceptance_scores.get("adapter_win_rate", 0.0)) <= float(acceptance_scores.get("base_win_rate", 0.0)):
        reasons.append("adapter_win_rate_not_above_base")
    if float(adapter.get("acceptance_rate", 0.0)) < float(base.get("acceptance_rate", 0.0)):
        reasons.append("adapter_acceptance_below_base")
    if float(adapter.get("user_effort_reduction_rate", 0.0)) <= float(base.get("user_effort_reduction_rate", 0.0)):
        reasons.append("adapter_effort_reduction_not_above_base")
    if float(adapter.get("frustration_score", 1.0)) >= float(base.get("frustration_score", 0.0)):
        reasons.append("adapter_frustration_not_lower_than_base")
    if float(adapter.get("false_completion_penalty_rate", 1.0)) > float(base.get("false_completion_penalty_rate", 0.0)):
        reasons.append("adapter_false_completion_above_base")
    if float(adapter.get("privacy_boundary_trust_rate", 0.0)) < float(base.get("privacy_boundary_trust_rate", 0.0)):
        reasons.append("adapter_privacy_boundary_trust_below_base")
    if float(adapter.get("would_continue_using_rate", 0.0)) <= float(base.get("would_continue_using_rate", 0.0)):
        reasons.append("adapter_would_continue_using_not_above_base")
    core_improvements = [
        metric
        for metric in (
            "acceptance_rate",
            "user_effort_reduction_rate",
            "frustration_reduction_rate",
            "privacy_boundary_trust_rate",
            "would_continue_using_rate",
            "overall_product_value_score",
        )
        if float(adapter.get(metric, 0.0)) > float(base.get(metric, 0.0))
    ]
    if not core_improvements:
        reasons.append("no_core_user_value_metric_improved")
    recommendation = "promote_after_manual_review" if not reasons else "archive"
    return {
        "kind": "phase34_final_decision",
        "recommendation": recommendation,
        "status": "ready_for_manual_review" if recommendation == "promote_after_manual_review" else "archived",
        "promotion_allowed": recommendation == "promote_after_manual_review",
        "auto_promotion_allowed": False,
        "product_benefit_claim_allowed": False,
        "actual_user_feedback_collected": False,
        "simulated_user_judgement_only": True,
        "manual_review_required_before_promotion": True,
        "core_improvements": core_improvements,
        "reasons": reasons,
        "base_scores": base,
        "adapter_scores": adapter,
        "created_at": _utcnow_iso(),
    }


def validate_phase34_simulation_boundaries(*, scenarios: Iterable[Mapping[str, Any]], pairs: Iterable[Mapping[str, Any]], judgements: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    problems: list[dict[str, str]] = []
    for item in list(scenarios) + list(pairs) + list(judgements):
        item_id = str(item.get("scenario_id") or item.get("pair_id") or "unknown")
        if item.get("feedback_source") != PHASE34_FEEDBACK_SOURCE:
            problems.append({"item_id": item_id, "reason": "feedback_source_not_simulated_user_judgement"})
        if item.get("confirmed_actual_user_feedback") is True:
            problems.append({"item_id": item_id, "reason": "confirmed_actual_user_feedback_true"})
        if contains_raw_private_text(item):
            problems.append({"item_id": item_id, "reason": "raw_private_text_detected"})
    return {
        "kind": "phase34_simulation_boundary_check",
        "passed": not problems,
        "problem_count": len(problems),
        "problems": problems,
        "created_at": _utcnow_iso(),
    }


def build_phase34_default_inputs(*, scenario_count: int, phase33_summary: Mapping[str, Any]) -> dict[str, Any]:
    scenario_batch = build_phase34_acceptance_scenarios(count=scenario_count, phase33_reference=phase33_summary)
    source_sessions = []
    # Rebuild source Phase33 sessions with the same count so source_session_id matches.
    phase33_sessions = build_phase33_usage_sessions(count=scenario_batch["scenario_count"])["sessions"]
    by_id = {str(item["session_id"]): item for item in phase33_sessions}
    for scenario in scenario_batch["scenarios"]:
        source_sessions.append(by_id[str(scenario["source_session_id"])])
    base_transcripts = build_phase33_transcripts(sessions=source_sessions, model_variant="base")
    adapter_transcripts = build_phase33_transcripts(sessions=source_sessions, model_variant="adapter")
    pairs = build_phase34_blind_eval_pairs(
        scenarios=scenario_batch["scenarios"],
        base_transcripts=base_transcripts,
        adapter_transcripts=adapter_transcripts,
    )
    judgements = [judge_phase34_blind_pair(pair) for pair in pairs]
    boundary_check = validate_phase34_simulation_boundaries(
        scenarios=scenario_batch["scenarios"],
        pairs=pairs,
        judgements=judgements,
    )
    scores = aggregate_phase34_judgements(judgements)
    decision = phase34_final_decision(acceptance_scores=scores, boundary_check=boundary_check)
    return {
        "scenario_batch": scenario_batch,
        "base_transcripts": base_transcripts,
        "adapter_transcripts": adapter_transcripts,
        "blind_eval_pairs": pairs,
        "judgements": judgements,
        "boundary_check": boundary_check,
        "acceptance_scores": scores,
        "decision": decision,
    }


__all__ = [
    "PHASE34_FEEDBACK_SOURCE",
    "PHASE34_KIND",
    "PHASE34_MAX_SCENARIOS",
    "PHASE34_MIN_SCENARIOS",
    "PHASE34_VALUE_METRICS",
    "aggregate_phase34_judgements",
    "blind_pair_public_view",
    "build_phase34_acceptance_scenarios",
    "build_phase34_blind_eval_pairs",
    "build_phase34_default_inputs",
    "build_phase34_phase33_review",
    "judge_phase34_blind_pair",
    "phase34_final_decision",
    "validate_phase34_blind_pair",
    "validate_phase34_simulation_boundaries",
    "write_jsonl",
]
