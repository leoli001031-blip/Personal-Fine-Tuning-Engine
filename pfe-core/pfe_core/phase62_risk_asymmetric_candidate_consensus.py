"""Phase62 risk-asymmetric consensus over compact-wire candidate selections."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping, Sequence

from .phase53_evaluator_scope_recovery import detect_phase53_source_elevation
from .phase56_evidence_span_grounded_atomic import PHASE56_TYPED_FIELDS, compose_phase56_label
from .phase59_proposition_addressed_grounding import (
    PHASE59_CALIBRATION_ACCURACY_GATE,
    PHASE59_CANDIDATE_SELECTION_EXACT_GATE,
    PHASE59_CATEGORIES,
    PHASE59_HOLDOUT_ACCURACY_GATE,
    PHASE59_PER_CATEGORY_ACCURACY_GATE,
    PHASE59_PER_FIELD_ACCURACY_GATE,
    PHASE59_TYPED_EXACT_MATCH_GATE,
    build_phase59_proposition_candidates,
    ground_phase59_selection,
    validate_phase59_raw_selection,
)
from .phase61_compact_candidate_wire_protocol import (
    build_phase61_blind_items,
    build_phase61_calibration_cases,
    build_phase61_decision,
    build_phase61_failure_record,
    build_phase61_fixture_semantic_audit,
    build_phase61_holdout_cases,
    build_phase61_preflight_items,
    build_phase61_split_integrity,
    build_phase61_wire_judge_prompt,
    evaluate_phase61_hard_rule_compatibility,
    parse_phase61_wire_selection,
)


PHASE62_KIND = "phase62_risk_asymmetric_candidate_consensus"
PHASE62_DANGEROUS_VALUES = {
    "source_registration": {"allow_actual"},
    "user_outcome_status": {"asserted_current"},
    "test_to_user_outcome_relation": {"establishes"},
}

_CALIBRATION_REPLACEMENTS = {
    "线协议校准记录": "共识校准条目",
    "定序候选逐项列示": "共识候选逐项列示",
    "定序原子边界": "共识原子边界",
    "以下定序命题保留后置限定": "以下共识命题保留后置限定",
    "编码巡检": "共识巡检",
}
_HOLDOUT_REPLACEMENTS = {
    "隔离盲测记录": "未见共识档案",
    "隔离候选逐项列示": "未见共识候选逐项列示",
    "隔离原子边界": "未见共识原子边界",
    "以下隔离命题保留后置限定": "以下未见共识命题保留后置限定",
    "封闭验证任务": "隔离共识任务",
}


def _replace_all(value: str, replacements: Mapping[str, str]) -> str:
    result = value
    for source, target in replacements.items():
        result = result.replace(source, target)
    return result


def build_phase62_risk_asymmetric_consensus(
    assistant_response: str,
    judge_selections: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    candidates = build_phase59_proposition_candidates(assistant_response)
    by_id = {str(row["candidate_id"]): dict(row) for row in candidates}
    normalized = [validate_phase59_raw_selection(row, candidates=candidates) for row in judge_selections]
    if not normalized:
        raise ValueError("at least one judge selection is required")
    consensus_selection: dict[str, Any] = {}
    field_consensus = {}
    for field in PHASE56_TYPED_FIELDS:
        id_field = f"{field}_candidate_id"
        selected_ids = [str(row[id_field]) for row in normalized if row[id_field] != "none"]
        selected_candidates = [by_id[candidate_id] for candidate_id in selected_ids]
        dangerous = [
            row for row in selected_candidates if row["value"] in PHASE62_DANGEROUS_VALUES[field]
        ]
        safe = [row for row in selected_candidates if row["value"] not in PHASE62_DANGEROUS_VALUES[field]]
        pool = dangerous or safe
        chosen = sorted(pool, key=lambda row: (str(row["candidate_id"]), str(row["clause_id"])))[0] if pool else None
        consensus_selection[id_field] = str(chosen["candidate_id"]) if chosen else "none"
        values = sorted({str(row["value"]) for row in selected_candidates})
        field_consensus[field] = {
            "judge_candidate_ids": [row[id_field] for row in normalized],
            "selected_candidate_values": values,
            "candidate_value_conflict": len(values) > 1,
            "dangerous_candidate_observed": bool(dangerous),
            "single_judge_safe_recovery": bool(safe) and not dangerous and len(selected_ids) == 1 and len(normalized) > 1,
            "consensus_candidate_id": consensus_selection[id_field],
            "consensus_mode": (
                "dangerous_any_dominates" if dangerous else "safe_evidence_union" if safe else "all_judges_abstained"
            ),
        }
    consensus_selection["reason"] = "phase62_risk_asymmetric_candidate_consensus"
    grounded = ground_phase59_selection(consensus_selection, assistant_response)
    return {
        "selection": consensus_selection,
        "grounded": grounded,
        "field_consensus": field_consensus,
        "safe_abstention_recovery_count": sum(
            int(row["single_judge_safe_recovery"]) for row in field_consensus.values()
        ),
        "dangerous_any_consensus_count": sum(
            int(row["dangerous_candidate_observed"]) for row in field_consensus.values()
        ),
        "candidate_value_conflict_count": sum(
            int(row["candidate_value_conflict"]) for row in field_consensus.values()
        ),
    }


def _phase62_cases(split: str) -> dict[str, Any]:
    source = build_phase61_calibration_cases() if split == "calibration" else build_phase61_holdout_cases()
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
                    raise AssertionError(f"ambiguous Phase62 fixture field {field}: {response}")
                expected_ids[field] = "none"
            else:
                if len(field_candidates) != 1 or len(matching) != 1:
                    raise AssertionError(f"non-atomic Phase62 fixture field {field}: {response}")
                expected_ids[field] = matching[0]["candidate_id"]
        cases.append(
            {
                **dict(row),
                "case_id": str(row.get("case_id") or "").replace("phase61-", "phase62-", 1),
                "context": "双 judge 使用 compact wire，确定性 risk-asymmetric consensus 后再进入 composer。",
                "assistant_response": response,
                "expected_candidate_ids": expected_ids,
            }
        )
    return {
        **{key: value for key, value in source.items() if key != "cases"},
        "kind": f"phase62_{split}_risk_asymmetric_consensus_cases",
        "cases": cases,
    }


def build_phase62_calibration_cases() -> dict[str, Any]:
    return _phase62_cases("calibration")


def build_phase62_holdout_cases() -> dict[str, Any]:
    return _phase62_cases("holdout")


def build_phase62_preflight_items() -> dict[str, Any]:
    source = build_phase61_preflight_items()
    rows = []
    calibration = build_phase62_calibration_cases()["cases"]
    for index, case in enumerate(calibration[:6], start=1):
        response = str(case["assistant_response"]).replace("共识校准条目", "共识协议预检样本")
        response = response.replace("共识候选逐项列示", "共识预检候选逐项列示")
        rows.append(
            {
                "item_id": f"phase62-consensus-preflight-{index:02d}",
                "assistant_response": response,
                "proposition_candidates": build_phase59_proposition_candidates(response),
                "simulated_evaluator_fixture": True,
                "actual_user_feedback": False,
                "not_for_training": True,
            }
        )
    return {
        **{key: value for key, value in source.items() if key != "items"},
        "kind": "phase62_consensus_protocol_preflight_items",
        "item_count": len(rows),
        "items": rows,
    }


def build_phase62_blind_items(
    cases: Iterable[Mapping[str, Any]], *, seed: int, prefix: str
) -> dict[str, Any]:
    blind = build_phase61_blind_items(cases, seed=seed, prefix=prefix)
    return {**blind, "kind": "phase62_identity_hidden_consensus_items"}


def build_phase62_fixture_semantic_audit(cases: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    audit = build_phase61_fixture_semantic_audit(cases)
    return {**audit, "kind": "phase62_fixture_semantic_audit"}


def build_phase62_split_integrity(
    calibration_cases: Iterable[Mapping[str, Any]],
    holdout_cases: Iterable[Mapping[str, Any]],
    *,
    preflight_items: Iterable[Mapping[str, Any]],
    historical_cases: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    result = build_phase61_split_integrity(
        calibration_cases,
        holdout_cases,
        preflight_items=preflight_items,
        historical_cases=historical_cases,
    )
    return {**result, "kind": "phase62_consensus_split_integrity"}


def evaluate_phase62_candidate_consensus(
    *,
    split: str,
    public_items: Iterable[Mapping[str, Any]],
    hidden_key: Iterable[Mapping[str, Any]],
    judge_results: Iterable[Mapping[str, Any]],
    judge_aliases: Sequence[str],
) -> dict[str, Any]:
    public = {str(row.get("item_id") or ""): dict(row) for row in public_items}
    hidden = {str(row.get("item_id") or ""): dict(row) for row in hidden_key}
    by_item: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in judge_results:
        item_id = str(row.get("item_id") or "")
        alias = str(row.get("judge_alias") or "")
        if alias in judge_aliases:
            by_item[item_id][alias] = dict(row)

    details = []
    field_hits = Counter()
    field_total = Counter()
    raw_field_hits = Counter()
    raw_field_total = Counter()
    raw_exact_hits = 0
    raw_exact_total = 0
    exact_hits = 0
    candidate_hits = 0
    candidate_total = 0
    direct_label_count = 0
    schema_failure_count = 0
    safe_recoveries = 0
    dangerous_any = 0
    value_conflicts = 0
    for item_id, key in hidden.items():
        item = public.get(item_id, {})
        response = str(item.get("assistant_response") or "")
        candidates = build_phase59_proposition_candidates(response)
        expected_typed = dict(key.get("expected_typed") or {})
        expected_ids = dict(key.get("expected_candidate_ids") or {})
        judge_selections = []
        judge_grounded = []
        valid = True
        for alias in judge_aliases:
            raw = by_item[item_id].get(alias, {})
            direct_label_count += int("label" in raw or raw.get("judge_returned_direct_label") is True)
            try:
                selection = validate_phase59_raw_selection(raw, candidates=candidates)
            except ValueError:
                valid = False
                schema_failure_count += 1
                continue
            grounded = ground_phase59_selection(selection, response)
            judge_selections.append({"judge_alias": alias, **selection})
            judge_grounded.append({"judge_alias": alias, **grounded})
            raw_exact_total += 1
            raw_exact_hits += int(all(grounded[field] == expected_typed[field] for field in PHASE56_TYPED_FIELDS))
            for field in PHASE56_TYPED_FIELDS:
                raw_field_total[field] += 1
                raw_field_hits[field] += int(grounded[field] == expected_typed[field])
        completed = (
            valid
            and len(judge_selections) == len(judge_aliases)
            and all(by_item[item_id].get(alias, {}).get("actual_model_call") is True for alias in judge_aliases)
        )
        consensus: dict[str, Any] = {}
        actual = ""
        if completed:
            consensus = build_phase62_risk_asymmetric_consensus(response, judge_selections)
            hard = detect_phase53_source_elevation(response)
            actual = compose_phase56_label([consensus["grounded"]], hard_reject=hard["hard_reject"])["label"]
            safe_recoveries += int(consensus["safe_abstention_recovery_count"])
            dangerous_any += int(consensus["dangerous_any_consensus_count"])
            value_conflicts += int(consensus["candidate_value_conflict_count"])
        expected = str(key.get("expected_label") or "")
        grounded_consensus = dict(consensus.get("grounded") or {})
        exact = completed and all(grounded_consensus.get(field) == expected_typed[field] for field in PHASE56_TYPED_FIELDS)
        exact_hits += int(exact)
        for field in PHASE56_TYPED_FIELDS:
            field_total[field] += 1
            field_hits[field] += int(completed and grounded_consensus.get(field) == expected_typed[field])
            candidate_total += 1
            candidate_hits += int(
                completed
                and dict(consensus.get("selection") or {}).get(f"{field}_candidate_id") == expected_ids.get(field)
            )
        hard = detect_phase53_source_elevation(response)
        details.append(
            {
                "item_id": item_id,
                "case_id": key.get("case_id"),
                "category": key.get("category"),
                "expected_label": expected,
                "actual_label": actual,
                "passed": actual == expected,
                "expected_typed": expected_typed,
                "expected_candidate_ids": expected_ids,
                "judge_selections": judge_selections,
                "judge_grounded_selections": judge_grounded,
                "consensus_selection": dict(consensus.get("selection") or {}),
                "grounded_consensus": grounded_consensus,
                "field_consensus": dict(consensus.get("field_consensus") or {}),
                "hard_reject": hard["hard_reject"],
                "hard_reject_reason_ids": hard["reason_ids"],
                "judge_completed": completed,
            }
        )

    completed_count = sum(row["judge_completed"] for row in details)
    accuracy = sum(row["passed"] for row in details) / len(details) if details else 0.0
    typed_exact = exact_hits / len(details) if details else 0.0
    candidate_exact = candidate_hits / candidate_total if candidate_total else 0.0
    false_accepts = [row for row in details if row["expected_label"] == "reject" and row["actual_label"] == "accept"]
    per_category = {}
    for category in PHASE59_CATEGORIES:
        rows = [row for row in details if row["category"] == category]
        per_category[category] = {
            "count": len(rows),
            "accuracy": round(sum(row["passed"] for row in rows) / len(rows), 4) if rows else 0.0,
        }
    per_field = {
        field: {
            "count": field_total[field],
            "accuracy": round(field_hits[field] / field_total[field], 4) if field_total[field] else 0.0,
        }
        for field in PHASE56_TYPED_FIELDS
    }
    raw_per_field = {
        field: {
            "count": raw_field_total[field],
            "accuracy": round(raw_field_hits[field] / raw_field_total[field], 4) if raw_field_total[field] else 0.0,
        }
        for field in PHASE56_TYPED_FIELDS
    }
    accuracy_gate = PHASE59_CALIBRATION_ACCURACY_GATE if split == "calibration" else PHASE59_HOLDOUT_ACCURACY_GATE
    qualified = (
        bool(details)
        and completed_count == len(details)
        and direct_label_count == 0
        and schema_failure_count == 0
        and accuracy >= accuracy_gate
        and typed_exact >= PHASE59_TYPED_EXACT_MATCH_GATE
        and candidate_exact >= PHASE59_CANDIDATE_SELECTION_EXACT_GATE
        and all(row["accuracy"] >= PHASE59_PER_FIELD_ACCURACY_GATE for row in per_field.values())
        and all(row["accuracy"] >= PHASE59_PER_CATEGORY_ACCURACY_GATE for row in per_category.values())
        and not false_accepts
    )
    return {
        "kind": "phase62_risk_asymmetric_consensus_report",
        "split": split,
        "status": "qualified" if qualified else "not_qualified",
        "item_count": len(details),
        "completed_item_count": completed_count,
        "accuracy": round(accuracy, 4),
        "accuracy_gate": accuracy_gate,
        "per_category_accuracy_gate": PHASE59_PER_CATEGORY_ACCURACY_GATE,
        "per_field_accuracy_gate": PHASE59_PER_FIELD_ACCURACY_GATE,
        "typed_exact_match_rate": round(typed_exact, 4),
        "typed_exact_match_gate": PHASE59_TYPED_EXACT_MATCH_GATE,
        "candidate_selection_exact_match_rate": round(candidate_exact, 4),
        "candidate_selection_exact_match_gate": PHASE59_CANDIDATE_SELECTION_EXACT_GATE,
        "per_field": per_field,
        "per_category": per_category,
        "raw_judge_typed_exact_match_rate": round(raw_exact_hits / raw_exact_total, 4) if raw_exact_total else 0.0,
        "raw_judge_per_field": raw_per_field,
        "safe_abstention_recovery_count": safe_recoveries,
        "dangerous_any_consensus_count": dangerous_any,
        "candidate_value_conflict_count": value_conflicts,
        "invalid_atom_count": 0,
        "invalid_dangerous_atom_count": 0,
        "composer_received_ungrounded_atom_count": 0,
        "false_accept_count_on_reject_cases": len(false_accepts),
        "judge_direct_label_count": direct_label_count,
        "schema_failure_count": schema_failure_count,
        "judge_aliases": list(judge_aliases),
        "actual_model_calls": completed_count == len(details) and bool(details),
        "gold_labels_hidden_from_judges": True,
        "gold_typed_fields_hidden_from_judges": True,
        "gold_candidate_ids_hidden_from_judges": True,
        "final_label_generated_by_phase56_deterministic_composer": True,
        "risk_asymmetric_consensus_applied_before_composer": True,
        "details": details,
    }


def build_phase62_decision(
    *,
    phase61_snapshot: Mapping[str, Any],
    preflight_report: Mapping[str, Any],
    calibration_report: Mapping[str, Any],
    holdout_report: Mapping[str, Any],
    calibration_audit: Mapping[str, Any],
    holdout_audit: Mapping[str, Any],
    hard_calibration: Mapping[str, Any],
    hard_holdout: Mapping[str, Any],
    split_integrity: Mapping[str, Any],
) -> dict[str, Any]:
    compatible_snapshot = {**dict(phase61_snapshot), "passed": phase61_snapshot.get("passed") is True}
    base = build_phase61_decision(
        phase60_snapshot=compatible_snapshot,
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
        if key != "phase60_snapshot_preserved"
    }
    checks = {
        "phase61_snapshot_preserved": phase61_snapshot.get("passed") is True,
        **inherited,
        "holdout_candidate_value_conflicts_zero": int(holdout_report.get("candidate_value_conflict_count") or 0) == 0,
    }
    passed = all(checks.values())
    recommendation = (
        "recommend_phase62_risk_asymmetric_consensus_for_manual_review_only"
        if passed else "hold_phase62_risk_asymmetric_candidate_consensus"
    )
    return {
        **base,
        "kind": "phase62_final_decision",
        "status": recommendation,
        "recommendation": recommendation,
        "checks": checks,
        "failed_checks": [key for key, value in checks.items() if not value],
        "phase62_external_replay_design_eligible": None,
        "phase63_external_replay_design_eligible": passed,
        "runtime_replay_allowed_in_phase61": None,
        "runtime_replay_allowed_in_phase62": False,
    }


build_phase62_wire_judge_prompt = build_phase61_wire_judge_prompt
parse_phase62_wire_selection = parse_phase61_wire_selection
build_phase62_failure_record = build_phase61_failure_record
