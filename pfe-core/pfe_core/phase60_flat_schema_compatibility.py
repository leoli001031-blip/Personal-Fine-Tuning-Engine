"""Phase60 flat-schema compatibility recovery over Phase59 candidates."""

from __future__ import annotations

import hashlib
from typing import Any, Iterable, Mapping, Sequence

from .phase56_evidence_span_grounded_atomic import PHASE56_TYPED_FIELDS
from .phase59_proposition_addressed_grounding import (
    PHASE59_CATEGORIES,
    build_phase59_blind_items,
    build_phase59_calibration_cases,
    build_phase59_fixture_semantic_audit,
    build_phase59_holdout_cases,
    build_phase59_proposition_candidates,
    build_phase59_split_integrity,
    evaluate_phase59_candidate_evaluator,
    validate_phase59_raw_selection,
)


PHASE60_KIND = "phase60_flat_schema_compatibility_recovery"
PHASE60_OUTPUT_FIELDS = tuple(f"{field}_candidate_id" for field in PHASE56_TYPED_FIELDS)

_CALIBRATION_REPLACEMENTS = {
    "候选校验素材": "扁平校准材料",
    "逐项候选如下": "顶层候选逐项列示",
    "原子边界": "扁平原子边界",
    "以下命题必须保留后置限定": "以下顶层命题保留后置限定",
    "机器检查": "协议巡检",
}
_HOLDOUT_REPLACEMENTS = {
    "盲测运行凭据": "未见评测档案",
    "逐项候选如下": "未见候选逐项列示",
    "原子边界": "未见原子边界",
    "以下命题必须保留后置限定": "以下未见命题保留后置限定",
    "离线任务": "隔离验证任务",
}


def _replace_all(value: str, replacements: Mapping[str, str]) -> str:
    result = value
    for source, target in replacements.items():
        result = result.replace(source, target)
    return result


def build_phase60_flat_judge_prompt(item: Mapping[str, Any]) -> str:
    response = str(item.get("assistant_response") or "")
    candidates = build_phase59_proposition_candidates(response)
    candidate_lines = []
    for field in PHASE56_TYPED_FIELDS:
        options = [row for row in candidates if row["field"] == field]
        rendered = "; ".join(
            f"{row['candidate_id']}={row['value']}@{row['clause_id']}" for row in options
        ) or "none only"
        candidate_lines.append(f"{field}: {rendered}")
    output_lines = "\n".join(f"- {field}_candidate_id" for field in PHASE56_TYPED_FIELDS)
    return (
        "你是候选命题选择器，不是最终裁判。只判断哪个预验证 candidate 代表 Agent 当前、非引文的明确立场。"
        "不得用背景或常识补全；证明关系、结果状态与来源登记必须分开。\n\n"
        f"Agent 回答：{response}\n\n"
        f"候选命题：\n" + "\n".join(candidate_lines) + "\n\n"
        f"只输出 schema 要求的三个顶层字符串字段：\n{output_lines}\n"
        "每个值只能是该字段列出的 pNNN 或 none。不要嵌套对象，不要输出 label、accept、edit 或 reject。"
    )


def phase60_flat_json_schema(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    properties = {}
    for field in PHASE56_TYPED_FIELDS:
        allowed = ["none"] + [
            str(row.get("candidate_id") or "")
            for row in candidates
            if row.get("field") == field
        ]
        properties[f"{field}_candidate_id"] = {"type": "string", "enum": allowed}
    properties["reason"] = {"type": "string"}
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(PHASE60_OUTPUT_FIELDS),
        "properties": properties,
    }


def validate_phase60_flat_selection(
    value: Mapping[str, Any], *, candidates: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    if "label" in value:
        raise ValueError("flat proposition judge must not return a direct label")
    if any(isinstance(value.get(field), Mapping) for field in PHASE56_TYPED_FIELDS):
        raise ValueError("nested candidate objects are not allowed in Phase60")
    flat = {field: value.get(field) for field in PHASE60_OUTPUT_FIELDS}
    flat["reason"] = value.get("reason")
    return validate_phase59_raw_selection(flat, candidates=candidates)


def build_phase60_failure_record(
    *,
    item_id: str,
    judge_alias: str,
    attempt: int,
    raw_response: str,
    error: str,
) -> dict[str, Any]:
    return {
        "item_id": item_id,
        "judge_alias": judge_alias,
        "attempt": attempt,
        "raw_response": raw_response,
        "raw_response_sha256": hashlib.sha256(raw_response.encode("utf-8")).hexdigest(),
        "error": error,
        "schema_valid": False,
        "actual_model_call": True,
    }


def _phase60_cases(split: str) -> dict[str, Any]:
    source = build_phase59_calibration_cases() if split == "calibration" else build_phase59_holdout_cases()
    replacements = _CALIBRATION_REPLACEMENTS if split == "calibration" else _HOLDOUT_REPLACEMENTS
    cases = []
    for row in source["cases"]:
        response = _replace_all(str(row.get("assistant_response") or ""), replacements)
        expected_typed = dict(row.get("expected_typed") or {})
        candidates = build_phase59_proposition_candidates(response)
        expected_ids = {}
        for field in PHASE56_TYPED_FIELDS:
            expected_value = expected_typed[field]
            field_candidates = [candidate for candidate in candidates if candidate["field"] == field]
            matching = [candidate for candidate in field_candidates if candidate["value"] == expected_value]
            if expected_value == "unstated":
                if field_candidates:
                    raise AssertionError(f"ambiguous Phase60 fixture field {field}: {response}")
                expected_ids[field] = "none"
            else:
                if len(field_candidates) != 1 or len(matching) != 1:
                    raise AssertionError(f"non-atomic Phase60 fixture field {field}: {response}")
                expected_ids[field] = matching[0]["candidate_id"]
        cases.append(
            {
                **dict(row),
                "case_id": str(row.get("case_id") or "").replace("phase59-", "phase60-", 1),
                "context": "使用顶层 candidate_id 字段选择预验证命题，不返回嵌套对象。",
                "assistant_response": response,
                "expected_candidate_ids": expected_ids,
            }
        )
    return {
        **{key: value for key, value in source.items() if key != "cases"},
        "kind": f"phase60_{split}_flat_candidate_cases",
        "cases": cases,
    }


def build_phase60_calibration_cases() -> dict[str, Any]:
    return _phase60_cases("calibration")


def build_phase60_holdout_cases() -> dict[str, Any]:
    return _phase60_cases("holdout")


def build_phase60_preflight_items() -> dict[str, Any]:
    rows = []
    for index, case in enumerate(build_phase60_calibration_cases()["cases"][:6], start=1):
        response = str(case["assistant_response"]).replace("扁平校准材料", "协议预检样本")
        response = response.replace("顶层候选逐项列示", "兼容预检逐项列示")
        rows.append(
            {
                "item_id": f"phase60-schema-preflight-{index:02d}",
                "assistant_response": response,
                "proposition_candidates": build_phase59_proposition_candidates(response),
                "simulated_evaluator_fixture": True,
                "actual_user_feedback": False,
                "not_for_training": True,
            }
        )
    return {
        "kind": "phase60_flat_schema_preflight_items",
        "item_count": len(rows),
        "items": rows,
        "scored_as_calibration": False,
        "not_for_training": True,
    }


def build_phase60_blind_items(
    cases: Iterable[Mapping[str, Any]], *, seed: int, prefix: str
) -> dict[str, Any]:
    blind = build_phase59_blind_items(cases, seed=seed, prefix=prefix)
    return {**blind, "kind": "phase60_identity_hidden_flat_candidate_items"}


def build_phase60_fixture_semantic_audit(cases: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    audit = build_phase59_fixture_semantic_audit(cases)
    return {**audit, "kind": "phase60_fixture_semantic_audit"}


def build_phase60_split_integrity(
    calibration_cases: Iterable[Mapping[str, Any]],
    holdout_cases: Iterable[Mapping[str, Any]],
    *,
    preflight_items: Iterable[Mapping[str, Any]],
    historical_cases: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    calibration = [dict(row) for row in calibration_cases]
    holdout = [dict(row) for row in holdout_cases]
    preflight = [dict(row) for row in preflight_items]
    base = build_phase59_split_integrity(calibration, holdout, historical_cases=historical_cases)

    def responses(rows: Iterable[Mapping[str, Any]]) -> set[str]:
        return {str(row.get("assistant_response") or "").strip() for row in rows}

    extra = {
        "preflight_calibration_overlap_zero": not responses(preflight).intersection(responses(calibration)),
        "preflight_holdout_overlap_zero": not responses(preflight).intersection(responses(holdout)),
        "preflight_count_six": len(preflight) == 6,
    }
    checks = {**dict(base.get("checks") or {}), **extra}
    return {
        **base,
        "kind": "phase60_flat_schema_split_integrity",
        "passed": all(checks.values()),
        "checks": checks,
        "preflight_count": len(preflight),
    }


def evaluate_phase60_candidate_evaluator(**kwargs: Any) -> dict[str, Any]:
    report = evaluate_phase59_candidate_evaluator(**kwargs)
    return {**report, "kind": "phase60_flat_candidate_evaluator_report"}


def build_phase60_decision(
    *,
    phase59_snapshot: Mapping[str, Any],
    preflight_report: Mapping[str, Any],
    calibration_report: Mapping[str, Any],
    holdout_report: Mapping[str, Any],
    calibration_audit: Mapping[str, Any],
    holdout_audit: Mapping[str, Any],
    hard_calibration: Mapping[str, Any],
    hard_holdout: Mapping[str, Any],
    split_integrity: Mapping[str, Any],
) -> dict[str, Any]:
    checks = {
        "phase59_snapshot_preserved": phase59_snapshot.get("passed") is True,
        "split_integrity_passed": split_integrity.get("passed") is True,
        "flat_schema_preflight_passed": preflight_report.get("status") == "passed",
        "calibration_fixture_semantic_audit_passed": calibration_audit.get("status") == "passed",
        "holdout_fixture_semantic_audit_passed": holdout_audit.get("status") == "passed",
        "hard_calibration_compatibility_passed": hard_calibration.get("status") == "passed",
        "hard_holdout_compatibility_passed": hard_holdout.get("status") == "passed",
        "calibration_qualified": calibration_report.get("status") == "qualified",
        "holdout_qualified": holdout_report.get("status") == "qualified",
        "holdout_false_accept_zero": int(holdout_report.get("false_accept_count_on_reject_cases") or 0) == 0,
        "holdout_invalid_dangerous_atoms_zero": int(holdout_report.get("invalid_dangerous_atom_count") or 0) == 0,
    }
    passed = all(checks.values())
    recommendation = (
        "recommend_phase60_flat_candidate_evaluator_for_manual_review_only"
        if passed else "hold_phase60_flat_schema_compatibility_recovery"
    )
    return {
        "kind": "phase60_final_decision",
        "status": recommendation,
        "recommendation": recommendation,
        "checks": checks,
        "failed_checks": [name for name, value in checks.items() if not value],
        "phase61_external_replay_design_eligible": passed,
        "evaluator_manual_review_use_allowed": passed,
        "runtime_replay_allowed_in_phase60": False,
        "new_training_allowed": False,
        "new_adapter_created": False,
        "product_default_change_allowed": False,
        "actual_user_feedback_count": 0,
        "actual_product_benefit_claim_allowed": False,
        "auto_training_allowed": False,
        "auto_promotion_allowed": False,
        "hermes_attachment_allowed": False,
    }
