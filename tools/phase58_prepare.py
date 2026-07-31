#!/usr/bin/env python3
"""Freeze Phase58 clause-addressed calibration and holdout before model calls."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = REPO_ROOT / "pfe-core"
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

from pfe_core.phase46_runtime_first_latest_intent import stable_hash
from pfe_core.phase56_evidence_span_grounded_atomic import evaluate_phase56_hard_reject_cases
from pfe_core.phase58_clause_addressed_grounding import (
    PHASE58_CALIBRATION_ACCURACY_GATE,
    PHASE58_CATEGORIES,
    PHASE58_EXTRACTION_RUBRIC,
    PHASE58_GROUNDING_VALIDITY_GATE,
    PHASE58_HOLDOUT_ACCURACY_GATE,
    PHASE58_PER_CATEGORY_ACCURACY_GATE,
    PHASE58_PER_FIELD_ACCURACY_GATE,
    PHASE58_TYPED_EXACT_MATCH_GATE,
    build_phase58_blind_items,
    build_phase58_calibration_cases,
    build_phase58_holdout_cases,
    build_phase58_split_integrity,
    phase58_ollama_json_schema,
)


EVIDENCE_ROOT = REPO_ROOT / "docs/demo/phase58-clause-addressed-grounding"
PHASE57_ROOT = REPO_ROOT / "docs/demo/phase57-span-evaluator-historical-replay"
PHASE53_SOURCE = CORE_ROOT / "pfe_core/phase53_evaluator_scope_recovery.py"
PHASE56_SOURCE = CORE_ROOT / "pfe_core/phase56_evidence_span_grounded_atomic.py"
PHASE57_SOURCE = CORE_ROOT / "pfe_core/phase57_span_evaluator_historical_replay.py"
PHASE58_SOURCE = CORE_ROOT / "pfe_core/phase58_clause_addressed_grounding.py"
JUDGE_ALIASES = ("semantic_judge_alpha", "semantic_judge_beta")
JUDGE_MODELS = {
    "semantic_judge_alpha": "gemma4:31b",
    "semantic_judge_beta": "qwen3.6:latest",
}


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return dict(value) if isinstance(value, Mapping) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
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


def _verify_manifest(root: Path, manifest: Mapping[str, Any]) -> bool:
    files = list(manifest.get("files") or [])
    return bool(files) and all(
        (REPO_ROOT / str(row.get("path") or "")).is_file()
        and _sha256(REPO_ROOT / str(row.get("path") or "")) == row.get("sha256")
        for row in files
    )


def _phase57_snapshot() -> dict[str, Any]:
    decision = _read_json(PHASE57_ROOT / "phase57-final-decision.json")
    report = _read_json(PHASE57_ROOT / "evidence-historical-replay/historical_replay_report.json")
    failures = _read_json(PHASE57_ROOT / "evidence-historical-replay/failure_analysis.json")
    manifest = _read_json(PHASE57_ROOT / "evidence_manifest.json")
    manifest_ok = _verify_manifest(PHASE57_ROOT, manifest)
    passed = (
        decision.get("recommendation") == "hold_phase57_span_evaluator_historical_replay"
        and decision.get("phase58_minimal_runtime_ab_design_eligible") is False
        and report.get("status") == "not_qualified"
        and manifest_ok
    )
    return {
        "kind": "phase58_phase57_canonical_snapshot",
        "passed": passed,
        "phase57_recommendation": decision.get("recommendation"),
        "phase57_status": report.get("status"),
        "phase57_accuracy": report.get("accuracy"),
        "phase57_grounding_validity_rate": report.get("raw_grounding_validity_rate"),
        "phase57_invalid_atom_count": report.get("invalid_atom_count"),
        "phase57_invalid_dangerous_atom_count": report.get("invalid_dangerous_atom_count"),
        "phase57_label_failure_count": failures.get("label_failure_count"),
        "phase57_invalid_grounding_count": failures.get("invalid_grounding_count"),
        "phase57_manifest_sha256": manifest.get("manifest_sha256"),
        "phase57_manifest_verified": manifest_ok,
        "phase57_source_modified": False,
    }


def _historical_cases() -> list[dict[str, Any]]:
    cases = []
    for split in ("calibration", "holdout"):
        filename = "calibration_labeled.json" if split == "calibration" else "holdout_labeled.json"
        cases.extend(
            dict(row)
            for row in _read_json(
                REPO_ROOT
                / f"docs/demo/phase56-evidence-span-grounded-atomic/evidence-evaluator-{split}/{filename}"
            ).get("cases") or []
        )
    cases.extend(_read_jsonl(PHASE57_ROOT / "evidence-historical-replay/blind_items_public.jsonl"))
    return cases


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-evidence", action="store_true")
    args = parser.parse_args()
    if args.clean_evidence and EVIDENCE_ROOT.exists():
        shutil.rmtree(EVIDENCE_ROOT)

    baseline = _phase57_snapshot()
    historical = _historical_cases()
    calibration = build_phase58_calibration_cases()
    holdout = build_phase58_holdout_cases()
    split = build_phase58_split_integrity(
        calibration["cases"], holdout["cases"], historical_cases=historical
    )
    calibration_blind = build_phase58_blind_items(
        calibration["cases"], seed=5801, prefix="phase58-calibration-blind"
    )
    holdout_blind = build_phase58_blind_items(
        holdout["cases"], seed=5802, prefix="phase58-holdout-blind"
    )
    hard_calibration = evaluate_phase56_hard_reject_cases(calibration["cases"])
    hard_holdout = evaluate_phase56_hard_reject_cases(holdout["cases"])
    schema_template = phase58_ollama_json_schema(["__CLAUSE_ID__"])
    protocol = {
        "kind": "phase58_frozen_clause_addressed_grounding_protocol",
        "structural_change": "replace_free_form_evidence_span_with_immutable_clause_id",
        "aggregate_phase57_failure_basis": {
            "invalid_atom_count": baseline.get("phase57_invalid_atom_count"),
            "invalid_grounding_count": baseline.get("phase57_invalid_grounding_count"),
            "dominant_reason": "span_does_not_support_atom",
            "dominant_reason_count": 697,
            "individual_historical_rows_used_for_tuning": False,
        },
        "categories": list(PHASE58_CATEGORIES),
        "semantic_judge_aliases": list(JUDGE_ALIASES),
        "semantic_judge_models_private": dict(JUDGE_MODELS),
        "judges_receive_other_judge_identity": False,
        "judges_receive_gold_label": False,
        "judges_receive_gold_typed_fields": False,
        "judges_receive_gold_clause_ids": False,
        "judges_return_direct_label": False,
        "ollama_json_schema_template": schema_template,
        "calibration_accuracy_gate": PHASE58_CALIBRATION_ACCURACY_GATE,
        "holdout_accuracy_gate": PHASE58_HOLDOUT_ACCURACY_GATE,
        "per_category_accuracy_gate": PHASE58_PER_CATEGORY_ACCURACY_GATE,
        "per_field_accuracy_gate": PHASE58_PER_FIELD_ACCURACY_GATE,
        "typed_exact_match_gate": PHASE58_TYPED_EXACT_MATCH_GATE,
        "grounding_validity_gate": PHASE58_GROUNDING_VALIDITY_GATE,
        "invalid_dangerous_atom_gate": 0,
        "composer_received_ungrounded_atom_gate": 0,
        "false_accept_gate_on_reject_cases": 0,
        "temperature": 0,
        "think": False,
        "num_ctx": 4096,
        "num_predict": 256,
        "parallel_worker_count": 4,
        "one_independent_call_per_item_per_judge": True,
        "holdout_allowed_only_after_calibration_qualification": True,
        "holdout_may_be_called_once": True,
        "post_calibration_tuning_allowed": False,
        "runtime_replay_allowed": False,
        "runtime_prompt_or_router_change_allowed": False,
        "training_allowed": False,
        "hermes_attachment_allowed": False,
        "product_default_change_allowed": False,
    }
    protocol["protocol_sha256"] = stable_hash(protocol)
    freeze = {
        "kind": "phase58_pre_model_call_freeze",
        "phase57_canonical_snapshot_passed": baseline["passed"],
        "phase53_hard_detector_source_sha256": _sha256(PHASE53_SOURCE),
        "phase56_evaluator_source_sha256": _sha256(PHASE56_SOURCE),
        "phase57_replay_source_sha256": _sha256(PHASE57_SOURCE),
        "phase58_evaluator_source_sha256": _sha256(PHASE58_SOURCE),
        "phase58_extraction_rubric_sha256": stable_hash(PHASE58_EXTRACTION_RUBRIC),
        "phase58_schema_template_sha256": stable_hash(schema_template),
        "calibration_public_sha256": stable_hash(calibration_blind["public_items"]),
        "calibration_hidden_sha256": stable_hash(calibration_blind["hidden_key"]),
        "holdout_public_sha256": stable_hash(holdout_blind["public_items"]),
        "holdout_hidden_sha256": stable_hash(holdout_blind["hidden_key"]),
        "historical_cases_sha256": stable_hash(historical),
        "protocol_sha256": protocol["protocol_sha256"],
        "frozen_before_calibration_calls": True,
        "frozen_before_holdout_calls": True,
        "created_at": _utcnow(),
    }
    source_manifest = {
        "kind": "phase58_source_boundary_manifest",
        "calibration_count": calibration["case_count"],
        "holdout_count": holdout["case_count"],
        "historical_fixture_count": len(historical),
        "simulated_evaluator_fixture": True,
        "actual_user_feedback_count": 0,
        "actual_human_review_count": 0,
        "not_for_training": True,
        "private_user_material_used": False,
        "phase57_rows_used_for_training": False,
        "phase57_individual_rows_used_for_tuning": False,
    }
    preparation = {
        "kind": "phase58_preparation_decision",
        "status": "ready_for_calibration" if all(
            (
                baseline["passed"],
                split["passed"],
                hard_calibration["status"] == "passed",
                hard_holdout["status"] == "passed",
            )
        ) else "blocked",
        "phase57_canonical_snapshot_passed": baseline["passed"],
        "split_integrity_passed": split["passed"],
        "hard_calibration_passed": hard_calibration["status"] == "passed",
        "hard_holdout_passed": hard_holdout["status"] == "passed",
        "holdout_allowed_only_after_calibration_qualification": True,
    }
    runtime = {
        "kind": "phase58_runtime_status",
        "runtime_replay_status": "not_requested_in_phase58",
        "runtime_replay_model_call_count": 0,
        "runtime_prompt_changed": False,
        "router_changed": False,
    }
    training = {
        "kind": "phase58_training_attempt",
        "status": "not_requested",
        "training_executed": False,
        "adapter_created": False,
        "auto_training_allowed": False,
    }

    calibration_dir = EVIDENCE_ROOT / "evidence-evaluator-calibration"
    holdout_dir = EVIDENCE_ROOT / "evidence-evaluator-holdout"
    _write_json(EVIDENCE_ROOT / "evidence-baseline/phase57_canonical_snapshot.json", baseline)
    _write_json(calibration_dir / "calibration_labeled.json", calibration)
    _write_jsonl(calibration_dir / "blind_items_public.jsonl", calibration_blind["public_items"])
    _write_json(calibration_dir / "blind_hidden_key.json", {"items": calibration_blind["hidden_key"]})
    _write_json(calibration_dir / "hard_reject_report.json", hard_calibration)
    _write_json(holdout_dir / "holdout_labeled.json", holdout)
    _write_jsonl(holdout_dir / "blind_items_public.jsonl", holdout_blind["public_items"])
    _write_json(holdout_dir / "blind_hidden_key.json", {"items": holdout_blind["hidden_key"]})
    _write_json(holdout_dir / "hard_reject_report.json", hard_holdout)
    _write_json(holdout_dir / "split_integrity.json", split)
    _write_json(EVIDENCE_ROOT / "evidence-no-runtime/runtime_status.json", runtime)
    _write_json(EVIDENCE_ROOT / "evidence-no-training/training_attempt.json", training)
    _write_json(EVIDENCE_ROOT / "evaluator_protocol.json", protocol)
    _write_json(EVIDENCE_ROOT / "pre_model_call_freeze.json", freeze)
    _write_json(EVIDENCE_ROOT / "source_manifest.json", source_manifest)
    _write_json(EVIDENCE_ROOT / "preparation_decision.json", preparation)
    print(json.dumps(preparation, ensure_ascii=False, indent=2))
    return 0 if preparation["status"] == "ready_for_calibration" else 1


if __name__ == "__main__":
    raise SystemExit(main())
