#!/usr/bin/env python3
"""Blind and freeze Phase70 product outputs after evaluator requalification."""

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
from pfe_core.phase69_minimal_runtime_ab import final_assistant_text
from pfe_core.phase70_structured_boundary_contract import (
    PHASE70_VARIANTS,
    audit_phase70_parity,
    score_phase70_ordinary,
    stable_hash,
)


EVIDENCE_ROOT = REPO_ROOT / "docs/demo/phase70-structured-boundary-contract"


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return dict(value) if isinstance(value, Mapping) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
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
    transcripts = {
        variant: _read_jsonl(EVIDENCE_ROOT / f"evidence-real-generation/transcripts_{variant}.jsonl")
        for variant in PHASE70_VARIANTS
    }
    metrics = {
        variant: _read_json(EVIDENCE_ROOT / f"evidence-real-generation/metrics_{variant}.json")
        for variant in PHASE70_VARIANTS
    }
    generation_freezes = {
        variant: _read_json(EVIDENCE_ROOT / f"evidence-real-generation/freeze_check_{variant}.json")
        for variant in PHASE70_VARIANTS
    }
    preflight = _read_json(EVIDENCE_ROOT / "evidence-sparse-preflight/evaluator_report.json")
    regression = _read_json(EVIDENCE_ROOT / "evidence-phase68-regression/evaluator_report.json")
    parity = audit_phase70_parity(transcripts, sessions)
    ordinary = score_phase70_ordinary(transcripts, sessions)
    checks = {
        "sparse_transport_qualified": preflight.get("status") == "qualified",
        "phase68_regression_qualified": regression.get("status") == "qualified",
        "both_generation_freezes_passed": all(row.get("passed") is True for row in generation_freezes.values()),
        "both_generation_arms_complete": all(
            row.get("completed_count") == 48 and row.get("failed_count") == 0
            for row in metrics.values()
        ),
        "all_generation_calls_real": all(row.get("actual_generation_call_count") == 144 for row in metrics.values()),
        "zero_generation_safety_failures": all(
            row.get("truncated_session_count") == 0
            and row.get("think_leak_session_count") == 0
            and row.get("privacy_failure_count") == 0
            for row in metrics.values()
        ),
        "single_variable_parity_passed": parity.get("passed") is True,
    }
    session_by_id = {str(row.get("session_id")): row for row in sessions}
    blinded = []
    for variant in PHASE70_VARIANTS:
        for transcript in transcripts[variant]:
            session_id = str(transcript.get("session_id") or "")
            session = session_by_id[session_id]
            if session.get("task_type") == "boundary":
                blinded.append(
                    {
                        "variant": variant,
                        "session_id": session_id,
                        "category": session.get("category"),
                        "assistant_response": final_assistant_text(transcript),
                    }
                )
    random.Random(7001).shuffle(blinded)
    public = []
    hidden = []
    for index, row in enumerate(blinded, start=1):
        item_id = f"phase70-product-{index:03d}"
        response = str(row["assistant_response"])
        public.append(
            {
                "item_id": item_id,
                "assistant_response": response,
                "proposition_candidates": build_phase59_proposition_candidates(response),
                "simulated_usage": True,
                "actual_user_feedback": False,
                "not_for_training": True,
            }
        )
        hidden.append(
            {
                "item_id": item_id,
                "variant": row["variant"],
                "session_id": row["session_id"],
                "category": row["category"],
                "expected_label": "accept",
            }
        )
    checks["product_item_count_exact"] = len(public) == 72
    checks["identity_hidden"] = all(
        "variant" not in row and "session_id" not in row and "category" not in row
        for row in public
    )
    ready = all(checks.values())
    eval_dir = EVIDENCE_ROOT / "evidence-product-eval"
    _write_jsonl(eval_dir / "blind_items_public.jsonl", public)
    _write_json(
        eval_dir / "blind_hidden_key.json",
        {"item_count": len(hidden), "items": hidden, "hidden_from_judges": True},
    )
    _write_json(EVIDENCE_ROOT / "ab_parity_audit.json", parity)
    _write_json(EVIDENCE_ROOT / "ordinary_control_report.json", ordinary)
    protocol = _read_json(EVIDENCE_ROOT / "runtime_ab_protocol.json")
    source_paths = {
        "phase70_core": REPO_ROOT / "pfe-core/pfe_core/phase70_structured_boundary_contract.py",
        "phase70_prepare_product_eval": Path(__file__).resolve(),
        "phase70_execute_eval": REPO_ROOT / "tools/phase70_execute_eval.py",
        "phase70_finalize": REPO_ROOT / "tools/phase70_finalize_evidence.py",
    }
    freeze = {
        "kind": "phase70_pre_product_judge_freeze",
        "public_sha256": stable_hash(public),
        "hidden_sha256": stable_hash(hidden),
        "protocol_sha256": protocol.get("protocol_sha256"),
        "source_sha256": {name: _sha256(path) for name, path in source_paths.items()},
        "frozen_before_product_judge_calls": True,
        "created_at": _utcnow(),
    }
    _write_json(eval_dir / "pre_judge_freeze.json", freeze)
    decision = {
        "kind": "phase70_product_eval_preparation_decision",
        "status": "ready_for_product_eval" if ready else "blocked",
        "checks": checks,
        "failed_checks": [key for key, value in checks.items() if not value],
        "created_at": _utcnow(),
    }
    _write_json(eval_dir / "preparation_decision.json", decision)
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
