"""Phase63 field-typed wire IDs mapped to frozen internal candidates."""

from __future__ import annotations

import re
from typing import Any, Iterable, Mapping, Sequence

from .phase56_evidence_span_grounded_atomic import PHASE56_TYPED_FIELDS
from .phase59_proposition_addressed_grounding import build_phase59_proposition_candidates
from .phase62_risk_asymmetric_candidate_consensus import (
    build_phase62_blind_items,
    build_phase62_calibration_cases,
    build_phase62_decision,
    build_phase62_failure_record,
    build_phase62_fixture_semantic_audit,
    build_phase62_holdout_cases,
    build_phase62_preflight_items,
    build_phase62_risk_asymmetric_consensus,
    build_phase62_split_integrity,
    evaluate_phase61_hard_rule_compatibility,
    evaluate_phase62_candidate_consensus,
)


PHASE63_KIND = "phase63_field_typed_candidate_wire"
PHASE63_WIRE_VERSION = "PFE2"
PHASE63_FIELD_PREFIXES = {
    "source_registration": "s",
    "user_outcome_status": "u",
    "test_to_user_outcome_relation": "r",
}
PHASE63_WIRE_PATTERN = re.compile(r"^PFE2\|(none|s\d{3})\|(none|u\d{3})\|(none|r\d{3})$")

_CALIBRATION_REPLACEMENTS = {
    "共识校准条目": "类型线校准条目",
    "共识候选逐项列示": "类型候选逐项列示",
    "共识原子边界": "类型原子边界",
    "以下共识命题保留后置限定": "以下类型命题保留后置限定",
    "共识巡检": "类型线巡检",
}
_HOLDOUT_REPLACEMENTS = {
    "未见共识档案": "未见类型线档案",
    "未见共识候选逐项列示": "未见类型候选逐项列示",
    "未见共识原子边界": "未见类型原子边界",
    "以下未见共识命题保留后置限定": "以下未见类型命题保留后置限定",
    "隔离共识任务": "隔离类型线任务",
}


def _replace_all(value: str, replacements: Mapping[str, str]) -> str:
    result = value
    for source, target in replacements.items():
        result = result.replace(source, target)
    return result


def build_phase63_typed_candidates(assistant_response: str) -> list[dict[str, Any]]:
    candidates = build_phase59_proposition_candidates(assistant_response)
    typed = []
    for field in PHASE56_TYPED_FIELDS:
        field_candidates = [row for row in candidates if row["field"] == field]
        for index, row in enumerate(field_candidates, start=1):
            typed.append(
                {
                    **dict(row),
                    "internal_candidate_id": row["candidate_id"],
                    "typed_candidate_id": f"{PHASE63_FIELD_PREFIXES[field]}{index:03d}",
                }
            )
    return typed


def build_phase63_typed_wire_prompt(item: Mapping[str, Any]) -> str:
    response = str(item.get("assistant_response") or "")
    candidates = build_phase63_typed_candidates(response)
    candidate_lines = []
    for field in PHASE56_TYPED_FIELDS:
        options = [row for row in candidates if row["field"] == field]
        rendered = "; ".join(
            f"{row['typed_candidate_id']}={row['value']}@{row['clause_id']}" for row in options
        ) or "none only"
        candidate_lines.append(f"{field}: {rendered}")
    return (
        "你是候选命题选择器，不是最终裁判。只判断哪个预验证 candidate 代表 Agent 当前、非引文的明确立场。"
        "不得用背景或常识补全；证明关系、结果状态与来源登记必须分开。\n\n"
        f"Agent 回答：{response}\n\n"
        "候选命题：\n" + "\n".join(candidate_lines) + "\n\n"
        "严格只返回一行 ASCII：PFE2|<source 的 sNNN 或 none>|<outcome 的 uNNN 或 none>|"
        "<relation 的 rNNN 或 none>。三个位置和前缀固定，不得跨字段；只写列出的 typed candidate ID 或 none，"
        "不得抄写含义或 clause，不得输出其他文字。"
    )


def parse_phase63_typed_wire_selection(
    value: str, *, candidates: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    match = PHASE63_WIRE_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError("invalid Phase63 field-typed wire envelope")
    typed = build_phase63_typed_candidates_from_internal(candidates)
    by_typed_id = {str(row["typed_candidate_id"]): str(row["internal_candidate_id"]) for row in typed}
    selection = {}
    for field, typed_id in zip(PHASE56_TYPED_FIELDS, match.groups(), strict=True):
        if typed_id == "none":
            internal_id = "none"
        else:
            internal_id = by_typed_id.get(typed_id)
            if internal_id is None:
                raise ValueError(f"unknown {field} typed candidate_id: {typed_id!r}")
            candidate = next(row for row in typed if row["typed_candidate_id"] == typed_id)
            if candidate["field"] != field:
                raise ValueError(f"cross-field typed candidate_id for {field}: {typed_id!r}")
        selection[f"{field}_candidate_id"] = internal_id
    selection["reason"] = ""
    return selection


def build_phase63_typed_candidates_from_internal(
    candidates: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    typed = []
    for field in PHASE56_TYPED_FIELDS:
        field_candidates = [dict(row) for row in candidates if row.get("field") == field]
        for index, row in enumerate(field_candidates, start=1):
            typed.append(
                {
                    **row,
                    "internal_candidate_id": row["candidate_id"],
                    "typed_candidate_id": f"{PHASE63_FIELD_PREFIXES[field]}{index:03d}",
                }
            )
    return typed


def build_phase63_failure_record(**kwargs: Any) -> dict[str, Any]:
    record = build_phase62_failure_record(**kwargs)
    record["wire_version"] = PHASE63_WIRE_VERSION
    return record


def _phase63_cases(split: str) -> dict[str, Any]:
    source = build_phase62_calibration_cases() if split == "calibration" else build_phase62_holdout_cases()
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
                    raise AssertionError(f"ambiguous Phase63 fixture field {field}: {response}")
                expected_ids[field] = "none"
            else:
                if len(field_candidates) != 1 or len(matching) != 1:
                    raise AssertionError(f"non-atomic Phase63 fixture field {field}: {response}")
                expected_ids[field] = matching[0]["candidate_id"]
        cases.append(
            {
                **dict(row),
                "case_id": str(row.get("case_id") or "").replace("phase62-", "phase63-", 1),
                "context": "模型使用字段类型化 candidate ID，映射回内部候选后进入 Phase62 consensus。",
                "assistant_response": response,
                "expected_candidate_ids": expected_ids,
            }
        )
    return {
        **{key: value for key, value in source.items() if key != "cases"},
        "kind": f"phase63_{split}_field_typed_wire_cases",
        "cases": cases,
    }


def build_phase63_calibration_cases() -> dict[str, Any]:
    return _phase63_cases("calibration")


def build_phase63_holdout_cases() -> dict[str, Any]:
    return _phase63_cases("holdout")


def build_phase63_preflight_items() -> dict[str, Any]:
    source = build_phase62_preflight_items()
    rows = []
    for index, case in enumerate(build_phase63_calibration_cases()["cases"][:6], start=1):
        response = str(case["assistant_response"]).replace("类型线校准条目", "类型线协议预检样本")
        response = response.replace("类型候选逐项列示", "类型预检候选逐项列示")
        rows.append(
            {
                "item_id": f"phase63-typed-wire-preflight-{index:02d}",
                "assistant_response": response,
                "proposition_candidates": build_phase59_proposition_candidates(response),
                "typed_proposition_candidates": build_phase63_typed_candidates(response),
                "simulated_evaluator_fixture": True,
                "actual_user_feedback": False,
                "not_for_training": True,
            }
        )
    return {
        **{key: value for key, value in source.items() if key != "items"},
        "kind": "phase63_field_typed_wire_preflight_items",
        "item_count": len(rows),
        "items": rows,
    }


def build_phase63_blind_items(
    cases: Iterable[Mapping[str, Any]], *, seed: int, prefix: str
) -> dict[str, Any]:
    blind = build_phase62_blind_items(cases, seed=seed, prefix=prefix)
    public = []
    for item in blind["public_items"]:
        response = str(item["assistant_response"])
        public.append({**dict(item), "typed_proposition_candidates": build_phase63_typed_candidates(response)})
    return {**blind, "kind": "phase63_identity_hidden_typed_wire_items", "public_items": public}


def build_phase63_fixture_semantic_audit(cases: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    audit = build_phase62_fixture_semantic_audit(cases)
    return {**audit, "kind": "phase63_fixture_semantic_audit"}


def build_phase63_split_integrity(
    calibration_cases: Iterable[Mapping[str, Any]],
    holdout_cases: Iterable[Mapping[str, Any]],
    *,
    preflight_items: Iterable[Mapping[str, Any]],
    historical_cases: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    result = build_phase62_split_integrity(
        calibration_cases,
        holdout_cases,
        preflight_items=preflight_items,
        historical_cases=historical_cases,
    )
    return {**result, "kind": "phase63_field_typed_wire_split_integrity"}


def build_phase63_decision(
    *,
    phase62_snapshot: Mapping[str, Any],
    preflight_report: Mapping[str, Any],
    calibration_report: Mapping[str, Any],
    holdout_report: Mapping[str, Any],
    calibration_audit: Mapping[str, Any],
    holdout_audit: Mapping[str, Any],
    hard_calibration: Mapping[str, Any],
    hard_holdout: Mapping[str, Any],
    split_integrity: Mapping[str, Any],
) -> dict[str, Any]:
    compatible_snapshot = {**dict(phase62_snapshot), "passed": phase62_snapshot.get("passed") is True}
    base = build_phase62_decision(
        phase61_snapshot=compatible_snapshot,
        preflight_report=preflight_report,
        calibration_report=calibration_report,
        holdout_report=holdout_report,
        calibration_audit=calibration_audit,
        holdout_audit=holdout_audit,
        hard_calibration=hard_calibration,
        hard_holdout=hard_holdout,
        split_integrity=split_integrity,
    )
    inherited = {
        key: value
        for key, value in dict(base.get("checks") or {}).items()
        if key != "phase61_snapshot_preserved"
    }
    checks = {"phase62_snapshot_preserved": phase62_snapshot.get("passed") is True, **inherited}
    passed = all(checks.values())
    recommendation = (
        "recommend_phase63_field_typed_wire_for_manual_review_only"
        if passed else "hold_phase63_field_typed_candidate_wire"
    )
    return {
        **base,
        "kind": "phase63_final_decision",
        "status": recommendation,
        "recommendation": recommendation,
        "checks": checks,
        "failed_checks": [key for key, value in checks.items() if not value],
        "phase63_external_replay_design_eligible": None,
        "phase64_external_replay_design_eligible": passed,
        "runtime_replay_allowed_in_phase62": None,
        "runtime_replay_allowed_in_phase63": False,
    }


evaluate_phase63_candidate_consensus = evaluate_phase62_candidate_consensus
build_phase63_risk_asymmetric_consensus = build_phase62_risk_asymmetric_consensus
evaluate_phase63_hard_rule_compatibility = evaluate_phase61_hard_rule_compatibility
