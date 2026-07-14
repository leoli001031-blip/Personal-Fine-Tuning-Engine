#!/usr/bin/env python3
"""Freeze blinded Phase69 runtime outputs before qualified-judge calls."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = REPO_ROOT / "pfe-core"
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

from pfe_core.phase59_proposition_addressed_grounding import build_phase59_proposition_candidates
from pfe_core.phase63_field_typed_candidate_wire import build_phase63_typed_candidates
from pfe_core.phase69_minimal_runtime_ab import (
    PHASE69_VARIANTS,
    audit_phase69_ab_parity,
    final_assistant_text,
    score_phase69_ordinary_transcripts,
    stable_hash,
)


EVIDENCE_ROOT = REPO_ROOT / "docs/demo/phase69-minimal-runtime-ab"


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return dict(value) if isinstance(value, Mapping) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    holdout = _read_json(EVIDENCE_ROOT / "evidence-holdout/holdout.json")
    sessions = [dict(row) for row in holdout.get("sessions") or []]
    protocol = _read_json(EVIDENCE_ROOT / "runtime_ab_protocol.json")
    prefreeze = _read_json(EVIDENCE_ROOT / "pre_model_call_freeze.json")
    transcripts = {
        variant: _read_jsonl(
            EVIDENCE_ROOT / f"evidence-real-generation/transcripts_{variant}.jsonl"
        )
        for variant in PHASE69_VARIANTS
    }
    metrics = {
        variant: _read_json(
            EVIDENCE_ROOT / f"evidence-real-generation/metrics_{variant}.json"
        )
        for variant in PHASE69_VARIANTS
    }
    generation_freezes = {
        variant: _read_json(
            EVIDENCE_ROOT / f"evidence-real-generation/freeze_check_{variant}.json"
        )
        for variant in PHASE69_VARIANTS
    }
    source_paths = {
        "phase69_core": REPO_ROOT / "pfe-core/pfe_core/phase69_minimal_runtime_ab.py",
        "phase69_prepare_eval": Path(__file__).resolve(),
        "phase69_execute_eval": REPO_ROOT / "tools/phase69_execute_eval.py",
        "phase69_finalize": REPO_ROOT / "tools/phase69_finalize_evidence.py",
    }
    source_checks = {
        name: prefreeze.get("source_sha256", {}).get(name) == _sha256(path)
        for name, path in source_paths.items()
    }
    protocol_copy = {key: value for key, value in protocol.items() if key != "protocol_sha256"}
    input_checks = {
        "source_hashes_preserved": all(source_checks.values()),
        "protocol_preserved": stable_hash(protocol_copy)
        == protocol.get("protocol_sha256")
        == prefreeze.get("protocol_sha256"),
        "holdout_preserved": stable_hash(holdout) == prefreeze.get("holdout_sha256"),
        "both_generation_freezes_passed": all(
            row.get("passed") is True for row in generation_freezes.values()
        ),
        "both_arms_have_48_completed_sessions": all(
            row.get("completed_count") == 48 and row.get("failed_count") == 0
            for row in metrics.values()
        ),
        "all_generation_calls_real": all(
            row.get("actual_model_session_count") == 48
            and row.get("actual_generation_call_count") == 144
            for row in metrics.values()
        ),
        "zero_truncation_think_or_privacy_failure": all(
            row.get("truncated_session_count") == 0
            and row.get("think_leak_session_count") == 0
            and row.get("privacy_persistence_failure_count") == 0
            for row in metrics.values()
        ),
    }
    parity = audit_phase69_ab_parity(transcripts, sessions)
    ordinary = score_phase69_ordinary_transcripts(transcripts, sessions)
    input_checks["single_variable_parity_passed"] = parity["passed"]

    session_by_id = {str(row.get("session_id")): row for row in sessions}
    blinded = []
    for variant in PHASE69_VARIANTS:
        for transcript in transcripts[variant]:
            session_id = str(transcript.get("session_id") or "")
            session = session_by_id.get(session_id, {})
            if session.get("task_type") != "boundary":
                continue
            output = final_assistant_text(transcript)
            blinded.append(
                {
                    "variant": variant,
                    "session_id": session_id,
                    "category": session.get("category"),
                    "assistant_response": output,
                }
            )
    random.Random(6901).shuffle(blinded)
    public_items = []
    hidden_items = []
    candidate_audit_rows = []
    for index, row in enumerate(blinded, start=1):
        item_id = f"phase69-blind-{index:03d}"
        response = str(row["assistant_response"])
        candidates = build_phase59_proposition_candidates(response)
        public_items.append(
            {
                "item_id": item_id,
                "assistant_response": response,
                "typed_proposition_candidates": build_phase63_typed_candidates(response),
                "simulated_evaluator_fixture": False,
                "simulated_usage": True,
                "actual_user_feedback": False,
                "not_for_training": True,
            }
        )
        hidden_items.append(
            {
                "item_id": item_id,
                "variant": row["variant"],
                "session_id": row["session_id"],
                "category": row["category"],
                "expected_label": "accept",
                "expected_typed": {
                    "source_registration": "exclude_actual",
                    "user_outcome_status": "suspended_or_negated",
                    "test_to_user_outcome_relation": "does_not_establish",
                },
            }
        )
        candidate_audit_rows.append(
            {
                "item_id": item_id,
                "candidate_count": len(candidates),
                "candidate_values": sorted(
                    {f"{candidate['field']}={candidate['value']}" for candidate in candidates}
                ),
            }
        )
    input_checks["boundary_blind_item_count_exact"] = len(public_items) == 72
    input_checks["identity_removed_from_public_items"] = all(
        "variant" not in row and "session_id" not in row and "category" not in row
        for row in public_items
    )
    ready = all(input_checks.values())
    eval_dir = EVIDENCE_ROOT / "evidence-qualified-evaluator"
    _write_jsonl(eval_dir / "blind_items_public.jsonl", public_items)
    _write_json(
        eval_dir / "blind_hidden_key.json",
        {
            "kind": "phase69_blind_hidden_key",
            "item_count": len(hidden_items),
            "items": hidden_items,
            "hidden_from_judges": True,
        },
    )
    _write_json(eval_dir / "prejudge_candidate_audit.json", {"items": candidate_audit_rows})
    _write_json(EVIDENCE_ROOT / "ab_parity_audit.json", parity)
    _write_json(EVIDENCE_ROOT / "ordinary_control_report.json", ordinary)
    eval_freeze = {
        "kind": "phase69_pre_judge_freeze",
        "public_items_sha256": stable_hash(public_items),
        "hidden_key_sha256": stable_hash(hidden_items),
        "protocol_sha256": protocol.get("protocol_sha256"),
        "source_sha256": {
            name: _sha256(path) for name, path in source_paths.items()
        },
        "frozen_before_judge_calls": True,
        "created_at": _utcnow(),
    }
    _write_json(eval_dir / "pre_judge_freeze.json", eval_freeze)
    decision = {
        "kind": "phase69_eval_preparation_decision",
        "status": "ready_for_qualified_evaluator" if ready else "blocked",
        "checks": input_checks,
        "failed_checks": [key for key, value in input_checks.items() if not value],
        "judge_calls_executed": False,
        "created_at": _utcnow(),
    }
    _write_json(eval_dir / "eval_preparation_decision.json", decision)
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
