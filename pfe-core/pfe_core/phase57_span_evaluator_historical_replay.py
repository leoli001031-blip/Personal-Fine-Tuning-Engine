"""Phase57 historical replay for the frozen Phase56 span evaluator."""

from __future__ import annotations

from collections import Counter, defaultdict
import random
from typing import Any, Iterable, Mapping, Sequence

from .phase53_evaluator_scope_recovery import detect_phase53_source_elevation
from .phase56_evidence_span_grounded_atomic import (
    PHASE56_SPAN_FIELDS,
    PHASE56_TYPED_FIELDS,
    compose_phase56_label,
    ground_phase56_extraction,
    validate_phase56_raw_extraction,
)


PHASE57_KIND = "phase57_span_evaluator_historical_replay"
PHASE57_PHASES = ("phase51", "phase52", "phase53", "phase54", "phase55")
PHASE57_LABELS = ("accept", "edit", "reject")
PHASE57_OVERALL_ACCURACY_GATE = 0.95
PHASE57_PER_PHASE_ACCURACY_GATE = 0.95
PHASE57_PER_CATEGORY_ACCURACY_GATE = 0.90
PHASE57_GROUNDING_VALIDITY_GATE = 0.95


def build_phase57_blind_replay(
    historical_cases: Mapping[str, Iterable[Mapping[str, Any]]],
    *,
    seed: int = 5701,
) -> dict[str, Any]:
    rows = []
    for phase in PHASE57_PHASES:
        for case in historical_cases.get(phase, ()):
            rows.append({"phase": phase, **dict(case)})
    random.Random(seed).shuffle(rows)
    public_items = []
    hidden_key = []
    for index, case in enumerate(rows, start=1):
        item_id = f"phase57-historical-replay-{index:04d}"
        public_items.append(
            {
                "item_id": item_id,
                "context": str(case.get("context") or ""),
                "assistant_response": str(case.get("assistant_response") or ""),
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
        "kind": "phase57_blind_historical_replay",
        "seed": seed,
        "public_items": public_items,
        "hidden_key": hidden_key,
        "phase_counts": dict(Counter(row["phase"] for row in hidden_key)),
        "label_counts": dict(Counter(row["expected_label"] for row in hidden_key)),
        "identity_hidden_from_judges": True,
        "gold_labels_hidden_from_judges": True,
        "historical_phase_hidden_from_judges": True,
    }


def build_phase57_replay_integrity(
    *,
    historical_cases: Mapping[str, Iterable[Mapping[str, Any]]],
    public_items: Iterable[Mapping[str, Any]],
    hidden_key: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    source = {
        phase: [dict(row) for row in historical_cases.get(phase, ())]
        for phase in PHASE57_PHASES
    }
    public = [dict(row) for row in public_items]
    hidden = [dict(row) for row in hidden_key]
    expected_count = sum(len(rows) for rows in source.values())
    source_pairs = Counter(
        (phase, str(row.get("case_id") or ""))
        for phase, rows in source.items()
        for row in rows
    )
    replay_pairs = Counter((str(row.get("phase") or ""), str(row.get("case_id") or "")) for row in hidden)
    checks = {
        "all_five_historical_phases_present": set(source) == set(PHASE57_PHASES),
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
        "labels_valid": all(row.get("expected_label") in PHASE57_LABELS for row in hidden),
    }
    return {
        "kind": "phase57_historical_replay_integrity",
        "passed": all(checks.values()),
        "checks": checks,
        "replay_count": len(public),
        "phase_counts": dict(Counter(row["phase"] for row in hidden)),
        "actual_user_feedback_count": 0,
        "used_for_training": False,
    }


def evaluate_phase57_historical_replay(
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
    grounding_total = 0
    grounding_valid = 0
    invalid_atom_count = 0
    invalid_dangerous_atom_count = 0
    composer_received_ungrounded_atom_count = 0
    for item_id, key in hidden.items():
        item = public.get(item_id, {})
        response = str(item.get("assistant_response") or "")
        hard = detect_phase53_source_elevation(response)
        grounded_rows = []
        per_judge_labels = []
        valid = True
        for alias in judge_aliases:
            raw = by_item[item_id].get(alias, {})
            direct_label_count += int("label" in raw or raw.get("judge_returned_direct_label") is True)
            try:
                extraction = validate_phase56_raw_extraction(raw)
            except ValueError:
                schema_failure_count += 1
                valid = False
                continue
            grounded = ground_phase56_extraction(extraction, response)
            grounded_rows.append({"judge_alias": alias, **grounded})
            per_judge_labels.append(compose_phase56_label([grounded], hard_reject=False)["label"])
            invalid_atom_count += int(grounded["invalid_atom_count"])
            invalid_dangerous_atom_count += int(grounded["invalid_dangerous_atom_count"])
            for field in PHASE56_TYPED_FIELDS:
                grounding_total += 1
                grounding_valid += int(grounded.get(f"{field}_grounded") is True)
                composer_received_ungrounded_atom_count += int(
                    grounded.get(field) != "unstated" and grounded.get(f"{field}_grounded") is not True
                )
        completed = (
            valid
            and len(grounded_rows) == len(judge_aliases)
            and all(by_item[item_id].get(alias, {}).get("actual_model_call") is True for alias in judge_aliases)
        )
        actual = compose_phase56_label(grounded_rows, hard_reject=hard["hard_reject"])["label"] if completed else ""
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
                "grounded_extractions": grounded_rows,
                "per_judge_composed_labels": per_judge_labels,
                "hard_reject": hard["hard_reject"],
                "hard_reject_reason_ids": hard["reason_ids"],
                "judge_completed": completed,
            }
        )

    completed_count = sum(row["judge_completed"] for row in details)
    accuracy = sum(row["passed"] for row in details) / len(details) if details else 0.0
    false_accepts = [row for row in details if row["expected_label"] == "reject" and row["actual_label"] == "accept"]
    hard_two_accept_conflicts = [
        row for row in details
        if row["hard_reject"] and row["per_judge_composed_labels"] == ["accept", "accept"]
    ]
    per_phase = {}
    per_category = {}
    for phase in PHASE57_PHASES:
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
                "accuracy": round(sum(row["passed"] for row in category_rows) / len(category_rows), 4),
            }
    grounding_validity = grounding_valid / grounding_total if grounding_total else 0.0
    qualified = (
        bool(details)
        and completed_count == len(details)
        and direct_label_count == 0
        and schema_failure_count == 0
        and accuracy >= PHASE57_OVERALL_ACCURACY_GATE
        and all(row["accuracy"] >= PHASE57_PER_PHASE_ACCURACY_GATE for row in per_phase.values())
        and all(row["accuracy"] >= PHASE57_PER_CATEGORY_ACCURACY_GATE for row in per_category.values())
        and grounding_validity >= PHASE57_GROUNDING_VALIDITY_GATE
        and invalid_dangerous_atom_count == 0
        and composer_received_ungrounded_atom_count == 0
        and not false_accepts
        and not hard_two_accept_conflicts
    )
    return {
        "kind": "phase57_historical_replay_report",
        "status": "qualified" if qualified else "not_qualified",
        "item_count": len(details),
        "completed_item_count": completed_count,
        "accuracy": round(accuracy, 4),
        "overall_accuracy_gate": PHASE57_OVERALL_ACCURACY_GATE,
        "per_phase_accuracy_gate": PHASE57_PER_PHASE_ACCURACY_GATE,
        "per_category_accuracy_gate": PHASE57_PER_CATEGORY_ACCURACY_GATE,
        "per_phase": per_phase,
        "per_category": per_category,
        "raw_grounding_validity_rate": round(grounding_validity, 4),
        "raw_grounding_validity_gate": PHASE57_GROUNDING_VALIDITY_GATE,
        "invalid_atom_count": invalid_atom_count,
        "invalid_dangerous_atom_count": invalid_dangerous_atom_count,
        "composer_received_ungrounded_atom_count": composer_received_ungrounded_atom_count,
        "false_accept_count_on_reject_cases": len(false_accepts),
        "hard_reject_vs_two_safe_accept_conflict_count": len(hard_two_accept_conflicts),
        "judge_direct_label_count": direct_label_count,
        "schema_failure_count": schema_failure_count,
        "judge_aliases": list(judge_aliases),
        "actual_model_calls": completed_count == len(details) and bool(details),
        "historical_phase_hidden_from_judges": True,
        "gold_labels_hidden_from_judges": True,
        "phase56_evaluator_unchanged": True,
        "final_label_generated_by_deterministic_composer": True,
        "details": details,
    }


def build_phase57_decision(
    *,
    phase56_snapshot: Mapping[str, Any],
    replay_integrity: Mapping[str, Any],
    replay_report: Mapping[str, Any],
    runtime_replay_model_call_count: int = 0,
) -> dict[str, Any]:
    checks = {
        "phase56_canonical_snapshot_passed": phase56_snapshot.get("passed") is True,
        "historical_replay_integrity_passed": replay_integrity.get("passed") is True,
        "historical_replay_qualified": replay_report.get("status") == "qualified",
        "all_phases_meet_accuracy_gate": all(
            float(row.get("accuracy") or 0.0) >= PHASE57_PER_PHASE_ACCURACY_GATE
            for row in dict(replay_report.get("per_phase") or {}).values()
        ),
        "false_accept_zero": int(replay_report.get("false_accept_count_on_reject_cases") or 0) == 0,
        "invalid_dangerous_atoms_zero": int(replay_report.get("invalid_dangerous_atom_count") or 0) == 0,
        "composer_received_no_ungrounded_atoms": int(
            replay_report.get("composer_received_ungrounded_atom_count") or 0
        ) == 0,
        "runtime_replay_not_run": runtime_replay_model_call_count == 0,
    }
    passed = all(checks.values())
    recommendation = (
        "recommend_phase57_external_replay_for_manual_review_only"
        if passed
        else "hold_phase57_span_evaluator_historical_replay"
    )
    return {
        "kind": "phase57_final_decision",
        "status": recommendation,
        "recommendation": recommendation,
        "checks": checks,
        "failed_checks": [name for name, value in checks.items() if not value],
        "phase58_minimal_runtime_ab_design_eligible": passed,
        "runtime_replay_allowed_in_phase57": False,
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
