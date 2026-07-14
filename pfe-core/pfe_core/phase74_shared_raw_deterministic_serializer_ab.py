"""Phase74 independent shared-raw deterministic serializer A/B primitives."""

from __future__ import annotations

from collections import Counter
import copy
from typing import Any, Iterable, Mapping

from .phase69_minimal_runtime_ab import final_assistant_text
from .phase70_structured_boundary_contract import stable_hash
from .phase72_deterministic_boundary_serializer import (
    PHASE72_EXACT_OUTPUT,
    PHASE72_VARIANTS,
    apply_phase72_serializer,
    build_phase72_holdout,
    evaluate_phase72_boundary_results,
    score_phase72_ordinary,
)


PHASE74_KIND = "phase74_shared_raw_deterministic_serializer_ab"
PHASE74_VARIANTS = PHASE72_VARIANTS
PHASE74_EXACT_OUTPUT = PHASE72_EXACT_OUTPUT
PHASE74_BOUNDARY_COUNT = 36
PHASE74_ORDINARY_COUNT = 18


def build_phase74_holdout() -> dict[str, Any]:
    source = build_phase72_holdout()
    sessions = []
    for index, value in enumerate(source["sessions"], start=1):
        row = copy.deepcopy(value)
        row["session_id"] = str(row["session_id"]).replace("phase72-", "phase74-", 1)
        row["user_goal"] = f"共享原始复核批次 {index:02d}。{row['user_goal']}"
        row["continuation_request"] = (
            f"独立终检标记 H{index:02d}。{row['continuation_request']}"
        )
        sessions.append(row)
    boundary = [row for row in sessions if row["task_type"] == "boundary"]
    ordinary = [row for row in sessions if row["task_type"] == "ordinary"]
    return {
        "kind": "phase74_independent_shared_raw_holdout",
        "session_count": len(sessions),
        "boundary_session_count": len(boundary),
        "ordinary_session_count": len(ordinary),
        "boundary_category_counts": dict(Counter(row["category"] for row in boundary)),
        "ordinary_category_counts": dict(Counter(row["category"] for row in ordinary)),
        "simulated_usage": True,
        "actual_user_feedback_count": 0,
        "not_for_training": True,
        "sessions": sessions,
    }


def derive_phase74_transcripts(
    raw_rows: Iterable[Mapping[str, Any]],
    sessions_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    variants = {name: [] for name in PHASE74_VARIANTS}
    for raw_value in raw_rows:
        raw = copy.deepcopy(dict(raw_value))
        session_id = str(raw.get("session_id") or "")
        shared_hash = stable_hash(raw.get("turns") or [])
        baseline = copy.deepcopy(raw)
        baseline.update(
            {
                "kind": "phase74_derived_runtime_transcript",
                "variant": "structured_prompt_raw",
                "shared_raw_transcript_sha256": shared_hash,
                "serializer_enabled": False,
                "serializer_apply_count": 0,
                "only_ab_variable": "deterministic_boundary_serializer_after_shared_raw_generation",
            }
        )
        candidate = copy.deepcopy(raw)
        history = []
        manifests = []
        for turn in candidate.get("turns") or []:
            if turn.get("role") == "assistant":
                serialized, manifest = apply_phase72_serializer(
                    history, str(turn.get("content") or "")
                )
                manifests.append({**manifest, "turn": len(manifests) + 1})
                turn["content"] = serialized
            history.append(dict(turn))
        candidate.update(
            {
                "kind": "phase74_derived_runtime_transcript",
                "variant": "deterministic_boundary_serializer",
                "shared_raw_transcript_sha256": shared_hash,
                "serializer_enabled": True,
                "serializer_apply_count": sum(
                    row["serializer_applied"] for row in manifests
                ),
                "serializer_manifests": manifests,
                "hardcoded_response": any(
                    row["serializer_applied"] for row in manifests
                ),
                "only_ab_variable": "deterministic_boundary_serializer_after_shared_raw_generation",
            }
        )
        is_boundary = sessions_by_id[session_id].get("task_type") == "boundary"
        candidate["final_route_expected"] = is_boundary
        candidate["final_route_actual"] = bool(manifests[-1]["serializer_applied"])
        variants["structured_prompt_raw"].append(baseline)
        variants["deterministic_boundary_serializer"].append(candidate)
    return variants


def audit_phase74_parity(
    transcripts: Mapping[str, Iterable[Mapping[str, Any]]],
    sessions: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    indexed = {
        variant: {
            str(row.get("session_id") or ""): dict(row)
            for row in transcripts.get(variant, [])
        }
        for variant in PHASE74_VARIANTS
    }
    details = []
    for session in sessions:
        session_id = str(session["session_id"])
        baseline = indexed["structured_prompt_raw"].get(session_id, {})
        candidate = indexed["deterministic_boundary_serializer"].get(session_id, {})
        is_boundary = session.get("task_type") == "boundary"
        details.append(
            {
                "session_id": session_id,
                "same_shared_raw": baseline.get("shared_raw_transcript_sha256")
                == candidate.get("shared_raw_transcript_sha256"),
                "same_model": baseline.get("model_id") == candidate.get("model_id"),
                "same_task": baseline.get("task_sha256") == candidate.get("task_sha256"),
                "both_completed": baseline.get("status")
                == candidate.get("status")
                == "completed",
                "candidate_route_expected": candidate.get("final_route_actual")
                is is_boundary,
                "ordinary_output_identical": is_boundary
                or final_assistant_text(baseline) == final_assistant_text(candidate),
            }
        )
    failures = [
        f"{row['session_id']}:{key}"
        for row in details
        for key, passed in row.items()
        if key != "session_id" and not passed
    ]
    return {
        "kind": "phase74_shared_raw_single_variable_parity_audit",
        "passed": bool(details) and not failures,
        "failed_checks": failures,
        "session_count": len(details),
        "only_ab_variable": "deterministic_boundary_serializer_after_shared_raw_generation",
        "details": details,
    }


def evaluate_phase74_boundary_results(**kwargs: Any) -> dict[str, Any]:
    report = evaluate_phase72_boundary_results(**kwargs)
    return {**report, "kind": "phase74_shared_raw_boundary_report"}


def score_phase74_ordinary(
    transcripts: Mapping[str, Iterable[Mapping[str, Any]]],
    sessions: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    report = score_phase72_ordinary(transcripts, sessions)
    return {**report, "kind": "phase74_ordinary_passthrough_report"}


__all__ = [
    "PHASE74_BOUNDARY_COUNT",
    "PHASE74_EXACT_OUTPUT",
    "PHASE74_KIND",
    "PHASE74_ORDINARY_COUNT",
    "PHASE74_VARIANTS",
    "audit_phase74_parity",
    "build_phase74_holdout",
    "derive_phase74_transcripts",
    "evaluate_phase74_boundary_results",
    "score_phase74_ordinary",
]
