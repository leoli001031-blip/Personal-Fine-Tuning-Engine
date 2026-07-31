"""Phase61 strict compact wire protocol over frozen Phase59 candidates."""

from __future__ import annotations

import re
from typing import Any, Iterable, Mapping, Sequence

from .phase56_evidence_span_grounded_atomic import PHASE56_TYPED_FIELDS
from .phase59_proposition_addressed_grounding import (
    build_phase59_proposition_candidates,
    evaluate_phase59_hard_rule_compatibility,
)
from .phase60_flat_schema_compatibility import (
    build_phase60_blind_items,
    build_phase60_calibration_cases,
    build_phase60_decision,
    build_phase60_failure_record,
    build_phase60_fixture_semantic_audit,
    build_phase60_flat_judge_prompt,
    build_phase60_holdout_cases,
    build_phase60_split_integrity,
    evaluate_phase60_candidate_evaluator,
    validate_phase60_flat_selection,
)


PHASE61_KIND = "phase61_compact_candidate_wire_protocol"
PHASE61_WIRE_VERSION = "PFE1"
PHASE61_WIRE_PATTERN = re.compile(r"^PFE1\|(none|p\d{3})\|(none|p\d{3})\|(none|p\d{3})$")

_CALIBRATION_REPLACEMENTS = {
    "扁平校准材料": "线协议校准记录",
    "顶层候选逐项列示": "定序候选逐项列示",
    "扁平原子边界": "定序原子边界",
    "以下顶层命题保留后置限定": "以下定序命题保留后置限定",
    "协议巡检": "编码巡检",
}
_HOLDOUT_REPLACEMENTS = {
    "未见评测档案": "隔离盲测记录",
    "未见候选逐项列示": "隔离候选逐项列示",
    "未见原子边界": "隔离原子边界",
    "以下未见命题保留后置限定": "以下隔离命题保留后置限定",
    "隔离验证任务": "封闭验证任务",
}


def _replace_all(value: str, replacements: Mapping[str, str]) -> str:
    result = value
    for source, target in replacements.items():
        result = result.replace(source, target)
    return result


def build_phase61_wire_judge_prompt(item: Mapping[str, Any]) -> str:
    phase60_prompt = build_phase60_flat_judge_prompt(item)
    semantic_prompt = phase60_prompt.split("只输出 schema 要求", 1)[0]
    return (
        semantic_prompt
        + "严格只返回一行 ASCII：PFE1|<source candidate id 或 none>|<outcome candidate id 或 none>|"
        "<relation candidate id 或 none>。三个位置顺序固定。只写 pNNN 或 none，不得抄写含义或 clause，"
        "不得输出 JSON、字段名、解释、空格或其他文字。"
    )


def parse_phase61_wire_selection(
    value: str, *, candidates: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    match = PHASE61_WIRE_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError("invalid Phase61 compact wire envelope")
    flat = {
        f"{field}_candidate_id": candidate_id
        for field, candidate_id in zip(PHASE56_TYPED_FIELDS, match.groups(), strict=True)
    }
    return validate_phase60_flat_selection(flat, candidates=candidates)


def build_phase61_failure_record(**kwargs: Any) -> dict[str, Any]:
    return {
        **build_phase60_failure_record(**kwargs),
        "wire_version": PHASE61_WIRE_VERSION,
    }


def _phase61_cases(split: str) -> dict[str, Any]:
    source = build_phase60_calibration_cases() if split == "calibration" else build_phase60_holdout_cases()
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
                    raise AssertionError(f"ambiguous Phase61 fixture field {field}: {response}")
                expected_ids[field] = "none"
            else:
                if len(field_candidates) != 1 or len(matching) != 1:
                    raise AssertionError(f"non-atomic Phase61 fixture field {field}: {response}")
                expected_ids[field] = matching[0]["candidate_id"]
        cases.append(
            {
                **dict(row),
                "case_id": str(row.get("case_id") or "").replace("phase60-", "phase61-", 1),
                "context": "使用固定顺序单行 wire protocol 选择预验证命题，不返回 JSON 或自由文本。",
                "assistant_response": response,
                "expected_candidate_ids": expected_ids,
            }
        )
    return {
        **{key: value for key, value in source.items() if key != "cases"},
        "kind": f"phase61_{split}_compact_wire_cases",
        "cases": cases,
    }


def build_phase61_calibration_cases() -> dict[str, Any]:
    return _phase61_cases("calibration")


def build_phase61_holdout_cases() -> dict[str, Any]:
    return _phase61_cases("holdout")


def build_phase61_preflight_items() -> dict[str, Any]:
    rows = []
    for index, case in enumerate(build_phase61_calibration_cases()["cases"][:6], start=1):
        response = str(case["assistant_response"]).replace("线协议校准记录", "线协议预检样本")
        response = response.replace("定序候选逐项列示", "预检候选逐项列示")
        rows.append(
            {
                "item_id": f"phase61-wire-preflight-{index:02d}",
                "assistant_response": response,
                "proposition_candidates": build_phase59_proposition_candidates(response),
                "simulated_evaluator_fixture": True,
                "actual_user_feedback": False,
                "not_for_training": True,
            }
        )
    return {
        "kind": "phase61_compact_wire_preflight_items",
        "item_count": len(rows),
        "items": rows,
        "scored_as_calibration": False,
        "not_for_training": True,
    }


def build_phase61_blind_items(
    cases: Iterable[Mapping[str, Any]], *, seed: int, prefix: str
) -> dict[str, Any]:
    blind = build_phase60_blind_items(cases, seed=seed, prefix=prefix)
    return {**blind, "kind": "phase61_identity_hidden_compact_wire_items"}


def build_phase61_fixture_semantic_audit(cases: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    audit = build_phase60_fixture_semantic_audit(cases)
    return {**audit, "kind": "phase61_fixture_semantic_audit"}


def build_phase61_split_integrity(
    calibration_cases: Iterable[Mapping[str, Any]],
    holdout_cases: Iterable[Mapping[str, Any]],
    *,
    preflight_items: Iterable[Mapping[str, Any]],
    historical_cases: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    result = build_phase60_split_integrity(
        calibration_cases,
        holdout_cases,
        preflight_items=preflight_items,
        historical_cases=historical_cases,
    )
    return {**result, "kind": "phase61_compact_wire_split_integrity"}


def evaluate_phase61_candidate_evaluator(**kwargs: Any) -> dict[str, Any]:
    report = evaluate_phase60_candidate_evaluator(**kwargs)
    return {**report, "kind": "phase61_compact_wire_candidate_evaluator_report"}


def evaluate_phase61_hard_rule_compatibility(cases: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    return evaluate_phase59_hard_rule_compatibility(cases)


def build_phase61_decision(
    *,
    phase60_snapshot: Mapping[str, Any],
    preflight_report: Mapping[str, Any],
    calibration_report: Mapping[str, Any],
    holdout_report: Mapping[str, Any],
    calibration_audit: Mapping[str, Any],
    holdout_audit: Mapping[str, Any],
    hard_calibration: Mapping[str, Any],
    hard_holdout: Mapping[str, Any],
    split_integrity: Mapping[str, Any],
) -> dict[str, Any]:
    phase60_compatible_snapshot = {
        **dict(phase60_snapshot),
        "passed": phase60_snapshot.get("passed") is True,
    }
    base = build_phase60_decision(
        phase59_snapshot=phase60_compatible_snapshot,
        preflight_report=preflight_report,
        calibration_report=calibration_report,
        holdout_report=holdout_report,
        calibration_audit=calibration_audit,
        holdout_audit=holdout_audit,
        hard_calibration=hard_calibration,
        hard_holdout=hard_holdout,
        split_integrity=split_integrity,
    )
    passed = all(dict(base.get("checks") or {}).values())
    recommendation = (
        "recommend_phase61_compact_wire_evaluator_for_manual_review_only"
        if passed else "hold_phase61_compact_candidate_wire_protocol"
    )
    return {
        **base,
        "kind": "phase61_final_decision",
        "status": recommendation,
        "recommendation": recommendation,
        "checks": {
            "phase60_snapshot_preserved": phase60_snapshot.get("passed") is True,
            **{key: value for key, value in dict(base.get("checks") or {}).items() if key != "phase59_snapshot_preserved"},
        },
        "failed_checks": [
            key
            for key, value in {
                "phase60_snapshot_preserved": phase60_snapshot.get("passed") is True,
                **{key: value for key, value in dict(base.get("checks") or {}).items() if key != "phase59_snapshot_preserved"},
            }.items()
            if not value
        ],
        "phase61_external_replay_design_eligible": None,
        "phase62_external_replay_design_eligible": passed,
        "runtime_replay_allowed_in_phase60": None,
        "runtime_replay_allowed_in_phase61": False,
    }
