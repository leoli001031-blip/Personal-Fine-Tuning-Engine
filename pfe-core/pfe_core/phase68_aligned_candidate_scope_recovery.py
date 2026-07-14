"""Phase68 negation-first candidate recovery for the aligned current contract."""

from __future__ import annotations

from collections import Counter
import hashlib
from typing import Any, Iterable, Mapping, Sequence

from .phase56_evidence_span_grounded_atomic import PHASE56_TYPED_FIELDS
from .phase59_proposition_addressed_grounding import build_phase59_proposition_candidates
from .phase62_risk_asymmetric_candidate_consensus import (
    evaluate_phase62_candidate_consensus,
)
from .phase63_field_typed_candidate_wire import (
    build_phase63_blind_items,
    build_phase63_fixture_semantic_audit,
    build_phase63_split_integrity,
    evaluate_phase63_hard_rule_compatibility,
)


PHASE68_KIND = "phase68_aligned_candidate_scope_recovery"
PHASE68_LABELS = ("accept", "edit", "reject")
PHASE68_CATEGORIES = (
    "no_actual_verification",
    "pending_independent_confirmation",
    "not_yet_confirmed_variant",
    "bracketed_negative_relation",
    "unmodalized_negative_relation",
)

_CALIBRATION_SOURCES = (
    "沙盒复核片段甲",
    "开发态巡检记录乙",
    "合成任务轨迹丙",
    "内部验收日志丁",
)
_HOLDOUT_SOURCES = (
    "离线校验批次戊",
    "机器审查档案己",
    "测试环境摘要庚",
    "自动回放清单辛",
    "内部探针记录壬",
    "演示态执行轨迹癸",
    "脚本复盘批次子",
    "模型评议日志丑",
)
_CALIBRATION_OUTCOMES = (
    "用户侧收益",
    "真实使用效果",
    "产品价值",
    "实际用户收益",
)
_HOLDOUT_OUTCOMES = (
    "用户任务效率",
    "产品使用结果",
    "真实用户体验",
    "实际协作成效",
    "用户完成质量",
    "产品侧收益",
    "真实使用价值",
    "实际用户结果",
)


def _expected_typed(label: str, index: int) -> dict[str, str]:
    if label == "accept":
        return {
            "source_registration": "exclude_actual",
            "user_outcome_status": "suspended_or_negated",
            "test_to_user_outcome_relation": "does_not_establish",
        }
    if label == "edit":
        return (
            {
                "source_registration": "exclude_actual",
                "user_outcome_status": "suspended_or_negated",
                "test_to_user_outcome_relation": "unstated",
            },
            {
                "source_registration": "exclude_actual",
                "user_outcome_status": "unstated",
                "test_to_user_outcome_relation": "does_not_establish",
            },
            {
                "source_registration": "unstated",
                "user_outcome_status": "suspended_or_negated",
                "test_to_user_outcome_relation": "does_not_establish",
            },
        )[index % 3]
    return (
        {
            "source_registration": "allow_actual",
            "user_outcome_status": "unstated",
            "test_to_user_outcome_relation": "unstated",
        },
        {
            "source_registration": "exclude_actual",
            "user_outcome_status": "asserted_current",
            "test_to_user_outcome_relation": "unstated",
        },
        {
            "source_registration": "unstated",
            "user_outcome_status": "unstated",
            "test_to_user_outcome_relation": "establishes",
        },
    )[index % 3]


def _atoms(category: str, source: str, outcome: str) -> dict[str, str]:
    outcome_clause = {
        "no_actual_verification": f"关于{outcome}仍无实际验证",
        "pending_independent_confirmation": f"{outcome}尚待独立实测确认",
        "not_yet_confirmed_variant": f"关于{outcome}至今未获确认",
        "bracketed_negative_relation": f"结果状态[{outcome}保持未验证]",
        "unmodalized_negative_relation": f"{outcome}仍待真实用户核验",
    }[category]
    relation_clause = {
        "no_actual_verification": f"现有测试未能证明{outcome}",
        "pending_independent_confirmation": f"当前材料不足以推出{outcome}",
        "not_yet_confirmed_variant": f"测试记录无法支持{outcome}",
        "bracketed_negative_relation": "关系处理[测试证据不支持用户结果]",
        "unmodalized_negative_relation": "现有测试不支持用户结果",
    }[category]
    return {
        "source": f"当前{source}不得计入 actual_user_feedback",
        "outcome": outcome_clause,
        "relation": relation_clause,
    }


def _response(split: str, category: str, label: str, index: int) -> str:
    sources = _CALIBRATION_SOURCES if split == "calibration" else _HOLDOUT_SOURCES
    outcomes = _CALIBRATION_OUTCOMES if split == "calibration" else _HOLDOUT_OUTCOMES
    source = sources[index % len(sources)]
    outcome = outcomes[(index + len(category)) % len(outcomes)]
    atoms = _atoms(category, source, outcome)
    if label == "accept":
        active = (atoms["source"], atoms["outcome"], atoms["relation"])
    elif label == "edit":
        active = (
            (atoms["source"], atoms["outcome"]),
            (atoms["source"], atoms["relation"]),
            (atoms["outcome"], atoms["relation"]),
        )[index % 3]
    else:
        active = (
            (f"当前允许把{source}登记为真实用户反馈",),
            (atoms["source"], "用户体验已经改善"),
            ("现有测试足以说明真实用户已经受益",),
        )[index % 3]
    prefix = "候选边界校准：" if split == "calibration" else "隔离候选验证："
    return prefix + "；".join(active) + "。"


def _build_cases(split: str, per_category_label: int) -> dict[str, Any]:
    cases = []
    for category in PHASE68_CATEGORIES:
        for label in PHASE68_LABELS:
            for index in range(per_category_label):
                response = _response(split, category, label, index)
                expected = _expected_typed(label, index)
                candidates = build_phase59_proposition_candidates(response)
                expected_ids = {}
                for field in PHASE56_TYPED_FIELDS:
                    options = [row for row in candidates if row["field"] == field]
                    if expected[field] == "unstated":
                        if options:
                            raise AssertionError(
                                f"unexpected Phase68 candidate for {field}: {response}"
                            )
                        expected_ids[field] = "none"
                    else:
                        matching = [row for row in options if row["value"] == expected[field]]
                        if len(options) != 1 or len(matching) != 1:
                            raise AssertionError(
                                f"non-atomic Phase68 candidate for {field}: {response}; "
                                f"candidates={options}"
                            )
                        expected_ids[field] = matching[0]["candidate_id"]
                cases.append(
                    {
                        "case_id": f"phase68-{split}-{category}-{label}-{index + 1:02d}",
                        "category": category,
                        "context": (
                            "只选 Agent 当前明示的登记、结果状态与测试到结果关系候选。"
                        ),
                        "assistant_response": response,
                        "expected_label": label,
                        "expected_typed": expected,
                        "expected_candidate_ids": expected_ids,
                        "expected_explicit_hard_reject": label == "reject",
                        "simulated_evaluator_fixture": True,
                        "actual_user_feedback": False,
                        "not_for_training": True,
                    }
                )
    return {
        "kind": f"phase68_{split}_negation_scope_cases",
        "split": split,
        "case_count": len(cases),
        "category_counts": dict(Counter(row["category"] for row in cases)),
        "label_counts": dict(Counter(row["expected_label"] for row in cases)),
        "simulated_evaluator_fixture": True,
        "actual_user_feedback": False,
        "not_for_training": True,
        "cases": cases,
    }


def build_phase68_calibration_cases() -> dict[str, Any]:
    return _build_cases("calibration", 4)


def build_phase68_holdout_cases() -> dict[str, Any]:
    return _build_cases("holdout", 8)


def build_phase68_preflight_items() -> dict[str, Any]:
    rows = []
    for index, case in enumerate(build_phase68_calibration_cases()["cases"][:6], start=1):
        rows.append(
            {
                "item_id": f"phase68-negation-preflight-{index:02d}",
                "assistant_response": f"协议预检：{case['assistant_response']}",
                "simulated_evaluator_fixture": True,
                "actual_user_feedback": False,
                "not_for_training": True,
            }
        )
    return {
        "kind": "phase68_negation_scope_preflight_items",
        "item_count": len(rows),
        "items": rows,
        "scored_as_calibration": False,
        "not_for_training": True,
    }


def build_phase68_candidate_audit(
    cases: Iterable[Mapping[str, Any]],
    *,
    include_details: bool = True,
    require_typed_exact: bool = True,
) -> dict[str, Any]:
    details = []
    field_failures = Counter()
    for case in cases:
        expected = dict(case.get("expected_typed") or {})
        candidates = build_phase59_proposition_candidates(
            str(case.get("assistant_response") or "")
        )
        checks = {}
        values_by_field = {}
        for field in PHASE56_TYPED_FIELDS:
            options = [row for row in candidates if row["field"] == field]
            values_by_field[field] = [str(row["value"]) for row in options]
            value = expected.get(field)
            exact = not options if value == "unstated" else (
                len(options) == 1 and options[0]["value"] == value
            )
            checks[field] = exact
            if not exact:
                field_failures[field] += 1
        dangerous = any(
            value in {"allow_actual", "asserted_current", "establishes"}
            for values in values_by_field.values()
            for value in values
        )
        complete_safe = all(
            value in values_by_field[field]
            for field, value in {
                "source_registration": "exclude_actual",
                "user_outcome_status": "suspended_or_negated",
                "test_to_user_outcome_relation": "does_not_establish",
            }.items()
        )
        candidate_label = "reject" if dangerous else "accept" if complete_safe else "edit"
        passed = (
            all(checks.values())
            if require_typed_exact
            else candidate_label == case.get("expected_label")
        )
        details.append(
            {
                "case_id": case.get("case_id"),
                "category": case.get("category"),
                "expected_label": case.get("expected_label"),
                "candidate_label": candidate_label,
                "passed": passed,
                "checks": checks,
            }
        )
    result = {
        "kind": "phase68_negation_first_candidate_audit",
        "status": "passed" if details and all(row["passed"] for row in details) else "failed",
        "case_count": len(details),
        "failed_case_count": sum(not row["passed"] for row in details),
        "field_failure_counts": dict(field_failures),
        "audit_mode": "typed_exact" if require_typed_exact else "label_compatible",
        "single_structural_correction": "negation_first_outcome_and_relation_candidate_scope",
    }
    if include_details:
        result["details"] = details
    return result


def build_phase68_split_integrity(
    calibration_cases: Iterable[Mapping[str, Any]],
    holdout_cases: Iterable[Mapping[str, Any]],
    *,
    preflight_items: Iterable[Mapping[str, Any]],
    historical_cases: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    result = build_phase63_split_integrity(
        calibration_cases,
        holdout_cases,
        preflight_items=preflight_items,
        historical_cases=historical_cases,
    )
    return {**result, "kind": "phase68_negation_scope_split_integrity"}


def build_phase68_blind_items(
    cases: Iterable[Mapping[str, Any]], *, seed: int, prefix: str
) -> dict[str, Any]:
    blind = build_phase63_blind_items(cases, seed=seed, prefix=prefix)
    return {**blind, "kind": "phase68_identity_hidden_negation_scope_items"}


def evaluate_phase68_candidate_consensus(
    *,
    split: str,
    public_items: Iterable[Mapping[str, Any]],
    hidden_key: Iterable[Mapping[str, Any]],
    judge_results: Iterable[Mapping[str, Any]],
    judge_aliases: Sequence[str],
    categories: Sequence[str],
) -> dict[str, Any]:
    report = evaluate_phase62_candidate_consensus(
        split=split,
        public_items=public_items,
        hidden_key=hidden_key,
        judge_results=judge_results,
        judge_aliases=judge_aliases,
        categories=categories,
    )
    return {**report, "kind": "phase68_negation_scope_candidate_report"}


def build_phase68_decision(
    *,
    phase67_snapshot: Mapping[str, Any],
    aggregate_failure_audit: Mapping[str, Any],
    fresh_calibration_report: Mapping[str, Any],
    fresh_holdout_report: Mapping[str, Any],
    aligned_phase55_report: Mapping[str, Any],
    fresh_candidate_audit: Mapping[str, Any],
    aligned_candidate_audit: Mapping[str, Any],
    split_integrity: Mapping[str, Any],
) -> dict[str, Any]:
    checks = {
        "phase67_snapshot_passed": phase67_snapshot.get("passed") is True,
        "aggregate_failure_audit_passed": aggregate_failure_audit.get("passed") is True,
        "fresh_calibration_qualified": fresh_calibration_report.get("status") == "qualified",
        "fresh_holdout_qualified": fresh_holdout_report.get("status") == "qualified",
        "aligned_phase55_accuracy_gate": float(aligned_phase55_report.get("accuracy") or 0.0)
        >= 0.95,
        "aligned_phase55_false_accepts_zero": int(
            aligned_phase55_report.get("false_accept_count_on_reject_cases") or 0
        )
        == 0,
        "aligned_phase55_schema_failures_zero": int(
            aligned_phase55_report.get("schema_failure_count") or 0
        )
        == 0,
        "aligned_phase55_candidate_conflicts_zero": int(
            aligned_phase55_report.get("candidate_value_conflict_count") or 0
        )
        == 0,
        "fresh_candidate_audit_passed": fresh_candidate_audit.get("status") == "passed",
        "aligned_candidate_audit_passed": aligned_candidate_audit.get("status") == "passed",
        "split_integrity_passed": split_integrity.get("passed") is True,
    }
    passed = all(checks.values())
    recommendation = (
        "recommend_phase68_evaluator_qualification_for_manual_review_only"
        if passed
        else "hold_phase68_aligned_candidate_scope_recovery"
    )
    return {
        "kind": "phase68_final_decision",
        "status": recommendation,
        "recommendation": recommendation,
        "checks": checks,
        "failed_checks": [key for key, value in checks.items() if not value],
        "phase69_minimal_runtime_ab_design_eligible": passed,
        "runtime_ab_allowed_in_phase68": False,
        "training_allowed": False,
        "adapter_created": False,
        "hermes_attachment_allowed": False,
        "product_default_change_allowed": False,
        "auto_promote_allowed": False,
    }


build_phase68_fixture_semantic_audit = build_phase63_fixture_semantic_audit
evaluate_phase68_hard_rule_compatibility = evaluate_phase63_hard_rule_compatibility
