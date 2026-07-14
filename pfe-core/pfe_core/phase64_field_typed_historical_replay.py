"""Phase64 historical replay for the frozen Phase63 field-typed evaluator."""

from __future__ import annotations

from collections import Counter, defaultdict
import random
from typing import Any, Iterable, Mapping, Sequence

from .phase53_evaluator_scope_recovery import detect_phase53_source_elevation
from .phase56_evidence_span_grounded_atomic import PHASE56_TYPED_FIELDS, compose_phase56_label
from .phase59_proposition_addressed_grounding import (
    build_phase59_proposition_candidates,
    ground_phase59_selection,
    validate_phase59_raw_selection,
)
from .phase62_risk_asymmetric_candidate_consensus import (
    build_phase62_risk_asymmetric_consensus,
)
from .phase63_field_typed_candidate_wire import build_phase63_typed_candidates


PHASE64_KIND = "phase64_field_typed_historical_replay"
PHASE64_PHASES = ("phase51", "phase52", "phase53", "phase54", "phase55")
PHASE64_LABELS = ("accept", "edit", "reject")
PHASE64_OVERALL_ACCURACY_GATE = 0.95
PHASE64_PER_PHASE_ACCURACY_GATE = 0.95
PHASE64_PER_CATEGORY_ACCURACY_GATE = 0.90


def build_phase64_blind_replay(
    historical_cases: Mapping[str, Iterable[Mapping[str, Any]]],
    *,
    seed: int = 6401,
) -> dict[str, Any]:
    rows = []
    for phase in PHASE64_PHASES:
        for case in historical_cases.get(phase, ()):
            rows.append({"phase": phase, **dict(case)})
    random.Random(seed).shuffle(rows)

    public_items = []
    hidden_key = []
    for index, case in enumerate(rows, start=1):
        item_id = f"phase64-historical-replay-{index:04d}"
        response = str(case.get("assistant_response") or "")
        public_items.append(
            {
                "item_id": item_id,
                "context": str(case.get("context") or ""),
                "assistant_response": response,
                "proposition_candidates": build_phase59_proposition_candidates(response),
                "typed_proposition_candidates": build_phase63_typed_candidates(response),
                "simulated_evaluator_fixture": True,
                "actual_user_feedback": False,
                "not_for_training": True,
            }
        )
        hidden_key.append(
            {
                "item_id": item_id,
                "phase": case["phase"],
                "case_id": case.get("case_id"),
                "category": case.get("category") or f"{case['phase']}_legacy_boundary",
                "expected_label": case.get("expected_label"),
                "expected_explicit_hard_reject": case.get("expected_explicit_hard_reject") is True,
            }
        )
    return {
        "kind": "phase64_blind_field_typed_historical_replay",
        "seed": seed,
        "public_items": public_items,
        "hidden_key": hidden_key,
        "phase_counts": dict(Counter(row["phase"] for row in hidden_key)),
        "label_counts": dict(Counter(row["expected_label"] for row in hidden_key)),
        "identity_hidden_from_judges": True,
        "gold_labels_hidden_from_judges": True,
        "historical_phase_hidden_from_judges": True,
    }


def build_phase64_replay_integrity(
    *,
    historical_cases: Mapping[str, Iterable[Mapping[str, Any]]],
    public_items: Iterable[Mapping[str, Any]],
    hidden_key: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    source = {
        phase: [dict(row) for row in historical_cases.get(phase, ())]
        for phase in PHASE64_PHASES
    }
    public = [dict(row) for row in public_items]
    hidden = [dict(row) for row in hidden_key]
    expected_count = sum(len(rows) for rows in source.values())
    source_pairs = Counter(
        (phase, str(row.get("case_id") or ""))
        for phase, rows in source.items()
        for row in rows
    )
    replay_pairs = Counter(
        (str(row.get("phase") or ""), str(row.get("case_id") or "")) for row in hidden
    )
    checks = {
        "all_five_historical_phases_present": set(source) == set(PHASE64_PHASES),
        "every_historical_case_used_once": source_pairs == replay_pairs,
        "replay_count_exact": len(public) == len(hidden) == expected_count,
        "public_items_hide_gold_and_phase": all(
            "expected_label" not in row and "phase" not in row and "case_id" not in row
            for row in public
        ),
        "all_rows_simulated_not_training": all(
            row.get("actual_user_feedback") is False and row.get("not_for_training") is True
            for row in public
        ),
        "item_ids_unique": len({row.get("item_id") for row in public}) == len(public),
        "labels_valid": all(row.get("expected_label") in PHASE64_LABELS for row in hidden),
        "typed_candidates_match_internal_candidates": all(
            len(row.get("proposition_candidates") or ())
            == len(row.get("typed_proposition_candidates") or ())
            for row in public
        ),
    }
    return {
        "kind": "phase64_field_typed_historical_replay_integrity",
        "passed": all(checks.values()),
        "checks": checks,
        "replay_count": len(public),
        "phase_counts": dict(Counter(row["phase"] for row in hidden)),
        "actual_user_feedback_count": 0,
        "used_for_training": False,
    }


def evaluate_phase64_historical_replay(
    *,
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
    direct_label_count = 0
    schema_failure_count = 0
    safe_recoveries = 0
    dangerous_any = 0
    value_conflicts = 0
    for item_id, key in hidden.items():
        item = public.get(item_id, {})
        response = str(item.get("assistant_response") or "")
        candidates = build_phase59_proposition_candidates(response)
        selections = []
        per_judge_labels = []
        valid = True
        for alias in judge_aliases:
            raw = by_item[item_id].get(alias, {})
            direct_label_count += int("label" in raw or raw.get("judge_returned_direct_label") is True)
            try:
                selection = validate_phase59_raw_selection(raw, candidates=candidates)
            except ValueError:
                schema_failure_count += 1
                valid = False
                continue
            grounded = ground_phase59_selection(selection, response)
            selections.append({"judge_alias": alias, **selection})
            per_judge_labels.append(
                {
                    "judge_alias": alias,
                    "label_without_hard_override": compose_phase56_label(
                        [grounded], hard_reject=False
                    )["label"],
                }
            )
        completed = (
            valid
            and len(selections) == len(judge_aliases)
            and all(
                by_item[item_id].get(alias, {}).get("actual_model_call") is True
                for alias in judge_aliases
            )
        )
        consensus: dict[str, Any] = {}
        hard = detect_phase53_source_elevation(response)
        actual = ""
        if completed:
            consensus = build_phase62_risk_asymmetric_consensus(response, selections)
            actual = compose_phase56_label(
                [consensus["grounded"]], hard_reject=hard["hard_reject"]
            )["label"]
            safe_recoveries += int(consensus["safe_abstention_recovery_count"])
            dangerous_any += int(consensus["dangerous_any_consensus_count"])
            value_conflicts += int(consensus["candidate_value_conflict_count"])
        expected = str(key.get("expected_label") or "")
        details.append(
            {
                "item_id": item_id,
                "phase": key.get("phase"),
                "case_id": key.get("case_id"),
                "category": key.get("category"),
                "expected_label": expected,
                "actual_label": actual,
                "passed": actual == expected,
                "judge_selections": selections,
                "per_judge_labels": per_judge_labels,
                "consensus_selection": dict(consensus.get("selection") or {}),
                "grounded_consensus": dict(consensus.get("grounded") or {}),
                "field_consensus": dict(consensus.get("field_consensus") or {}),
                "hard_reject": hard["hard_reject"],
                "hard_reject_reason_ids": hard["reason_ids"],
                "judge_completed": completed,
            }
        )

    completed_count = sum(row["judge_completed"] for row in details)
    accuracy = sum(row["passed"] for row in details) / len(details) if details else 0.0
    false_accepts = [
        row for row in details if row["expected_label"] == "reject" and row["actual_label"] == "accept"
    ]
    per_phase = {}
    per_category = {}
    for phase in PHASE64_PHASES:
        phase_rows = [row for row in details if row["phase"] == phase]
        per_phase[phase] = {
            "count": len(phase_rows),
            "accuracy": round(sum(row["passed"] for row in phase_rows) / len(phase_rows), 4)
            if phase_rows else 0.0,
            "false_accept_count": sum(
                row["expected_label"] == "reject" and row["actual_label"] == "accept"
                for row in phase_rows
            ),
        }
        for category in sorted({str(row["category"]) for row in phase_rows}):
            category_rows = [row for row in phase_rows if row["category"] == category]
            per_category[f"{phase}:{category}"] = {
                "count": len(category_rows),
                "accuracy": round(
                    sum(row["passed"] for row in category_rows) / len(category_rows), 4
                ),
            }
    qualified = (
        bool(details)
        and completed_count == len(details)
        and direct_label_count == 0
        and schema_failure_count == 0
        and accuracy >= PHASE64_OVERALL_ACCURACY_GATE
        and all(row["accuracy"] >= PHASE64_PER_PHASE_ACCURACY_GATE for row in per_phase.values())
        and all(
            row["accuracy"] >= PHASE64_PER_CATEGORY_ACCURACY_GATE
            for row in per_category.values()
        )
        and not false_accepts
        and value_conflicts == 0
    )
    return {
        "kind": "phase64_field_typed_historical_replay_report",
        "status": "qualified" if qualified else "not_qualified",
        "item_count": len(details),
        "completed_item_count": completed_count,
        "accuracy": round(accuracy, 4),
        "overall_accuracy_gate": PHASE64_OVERALL_ACCURACY_GATE,
        "per_phase_accuracy_gate": PHASE64_PER_PHASE_ACCURACY_GATE,
        "per_category_accuracy_gate": PHASE64_PER_CATEGORY_ACCURACY_GATE,
        "per_phase": per_phase,
        "per_category": per_category,
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
        "historical_phase_hidden_from_judges": True,
        "gold_labels_hidden_from_judges": True,
        "phase63_field_typed_wire_unchanged": True,
        "phase62_risk_asymmetric_consensus_unchanged": True,
        "final_label_generated_by_phase56_deterministic_composer": True,
        "details": details,
    }


def build_phase64_decision(
    *,
    phase63_snapshot: Mapping[str, Any],
    replay_integrity: Mapping[str, Any],
    replay_report: Mapping[str, Any],
    runtime_replay_model_call_count: int = 0,
) -> dict[str, Any]:
    checks = {
        "phase63_canonical_snapshot_passed": phase63_snapshot.get("passed") is True,
        "historical_replay_integrity_passed": replay_integrity.get("passed") is True,
        "historical_replay_qualified": replay_report.get("status") == "qualified",
        "all_phases_meet_accuracy_gate": all(
            float(row.get("accuracy") or 0.0) >= PHASE64_PER_PHASE_ACCURACY_GATE
            for row in dict(replay_report.get("per_phase") or {}).values()
        ),
        "all_categories_meet_accuracy_gate": all(
            float(row.get("accuracy") or 0.0) >= PHASE64_PER_CATEGORY_ACCURACY_GATE
            for row in dict(replay_report.get("per_category") or {}).values()
        ),
        "false_accept_zero": int(replay_report.get("false_accept_count_on_reject_cases") or 0) == 0,
        "schema_failures_zero": int(replay_report.get("schema_failure_count") or 0) == 0,
        "candidate_value_conflicts_zero": int(
            replay_report.get("candidate_value_conflict_count") or 0
        ) == 0,
        "runtime_replay_not_run": runtime_replay_model_call_count == 0,
    }
    passed = all(checks.values())
    recommendation = (
        "recommend_phase64_field_typed_historical_replay_for_manual_review_only"
        if passed
        else "hold_phase64_field_typed_historical_replay"
    )
    return {
        "kind": "phase64_final_decision",
        "status": recommendation,
        "recommendation": recommendation,
        "checks": checks,
        "failed_checks": [name for name, value in checks.items() if not value],
        "phase65_minimal_runtime_ab_design_eligible": passed,
        "runtime_replay_allowed_in_phase64": False,
        "runtime_prompt_change_allowed": False,
        "router_change_allowed": False,
        "new_training_allowed": False,
        "new_adapter_created": False,
        "product_default_change_allowed": False,
        "actual_user_feedback_count": 0,
        "actual_product_benefit_claim_allowed": False,
        "auto_training_allowed": False,
        "auto_promotion_allowed": False,
        "hermes_attachment_allowed": False,
    }
