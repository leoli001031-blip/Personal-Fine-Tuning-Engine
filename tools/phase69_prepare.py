#!/usr/bin/env python3
"""Freeze Phase69 tasks, protocols, gates, and source state before model calls."""

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

from pfe_core.phase63_field_typed_candidate_wire import (
    PHASE63_FIELD_PREFIXES,
    PHASE63_WIRE_PATTERN,
    PHASE63_WIRE_VERSION,
)
from pfe_core.phase69_minimal_runtime_ab import (
    PHASE69_ACCEPT_RATE_DELTA_GATE,
    PHASE69_ACCEPT_RATE_GATE,
    PHASE69_VARIANTS,
    build_phase69_holdout,
    stable_hash,
)


EVIDENCE_ROOT = REPO_ROOT / "docs/demo/phase69-minimal-runtime-ab"
PHASE68_ROOT = REPO_ROOT / "docs/demo/phase68-aligned-candidate-scope-recovery"
MODEL_PATH = REPO_ROOT / "models/Qwen3-4B"
SOURCE_FILES = {
    "phase46_generator_helpers": REPO_ROOT / "tools/phase46_qwen3_4b_generate.py",
    "phase53_hard_detector": CORE_ROOT / "pfe_core/phase53_evaluator_scope_recovery.py",
    "phase56_grounder_composer": CORE_ROOT / "pfe_core/phase56_evidence_span_grounded_atomic.py",
    "phase59_candidates": CORE_ROOT / "pfe_core/phase59_proposition_addressed_grounding.py",
    "phase62_consensus": CORE_ROOT / "pfe_core/phase62_risk_asymmetric_candidate_consensus.py",
    "phase63_wire": CORE_ROOT / "pfe_core/phase63_field_typed_candidate_wire.py",
    "phase69_core": CORE_ROOT / "pfe_core/phase69_minimal_runtime_ab.py",
    "phase69_prepare": Path(__file__).resolve(),
    "phase69_generate": REPO_ROOT / "tools/phase69_generate.py",
    "phase69_prepare_eval": REPO_ROOT / "tools/phase69_prepare_eval.py",
    "phase69_execute_eval": REPO_ROOT / "tools/phase69_execute_eval.py",
    "phase69_finalize": REPO_ROOT / "tools/phase69_finalize_evidence.py",
}
JUDGE_ALIASES = ("semantic_judge_alpha", "semantic_judge_beta")
JUDGE_MODELS = {
    "semantic_judge_alpha": "gemma4:31b",
    "semantic_judge_beta": "qwen3.6:latest",
}


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return dict(value) if isinstance(value, Mapping) else {}


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


def _verify_manifest(manifest: Mapping[str, Any]) -> bool:
    files = list(manifest.get("files") or [])
    return bool(files) and all(
        (REPO_ROOT / str(row.get("path") or "")).is_file()
        and _sha256(REPO_ROOT / str(row.get("path") or "")) == row.get("sha256")
        for row in files
    )


def _phase68_snapshot() -> dict[str, Any]:
    decision = _read_json(PHASE68_ROOT / "phase68-final-decision.json")
    integrity = _read_json(PHASE68_ROOT / "evidence_integrity.json")
    manifest = _read_json(PHASE68_ROOT / "evidence_manifest.json")
    holdout = _read_json(
        PHASE68_ROOT / "evidence-evaluator-holdout/candidate_evaluator_report.json"
    )
    aligned = _read_json(
        PHASE68_ROOT / "evidence-aligned-phase55-regression/aligned_regression_report.json"
    )
    checks = {
        "phase68_manual_review_recommendation": decision.get("recommendation")
        == "recommend_phase68_evaluator_qualification_for_manual_review_only",
        "phase68_runtime_ab_eligible": decision.get("phase69_minimal_runtime_ab_design_eligible")
        is True,
        "phase68_integrity_passed": integrity.get("passed") is True,
        "phase68_manifest_verified": _verify_manifest(manifest),
        "fresh_holdout_qualified": holdout.get("status") == "qualified"
        and holdout.get("accuracy") == 1.0,
        "aligned_regression_passed": aligned.get("accuracy") == 1.0,
    }
    return {
        "kind": "phase69_phase68_qualified_evaluator_snapshot",
        "passed": all(checks.values()),
        "checks": checks,
        "phase68_recommendation": decision.get("recommendation"),
        "manifest_sha256": manifest.get("manifest_sha256"),
        "fresh_holdout_accuracy": holdout.get("accuracy"),
        "aligned_phase55_accuracy": aligned.get("accuracy"),
    }


def _task_overlap_audit(holdout: Mapping[str, Any]) -> dict[str, Any]:
    current = []
    for row in holdout.get("sessions") or []:
        current.append(
            " ".join(
                str(row.get(key) or "").strip()
                for key in (
                    "user_goal",
                    "user_correction",
                    "continuation_request",
                    "acceptance_request",
                )
            )
        )
    historical = []
    for relative in (
        "evidence-evaluator-calibration/blind_items_public.jsonl",
        "evidence-evaluator-holdout/blind_items_public.jsonl",
        "evidence-aligned-phase55-regression/blind_items_public.jsonl",
    ):
        path = PHASE68_ROOT / relative
        historical.extend(
            str(json.loads(line).get("assistant_response") or "").strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    current_hashes = [stable_hash(value) for value in current]
    historical_hashes = {stable_hash(value) for value in historical}
    overlaps = sorted(set(current_hashes) & historical_hashes)
    return {
        "kind": "phase69_task_overlap_audit",
        "passed": len(current_hashes) == len(set(current_hashes)) and not overlaps,
        "phase69_task_count": len(current_hashes),
        "phase68_fixture_count": len(historical_hashes),
        "phase69_duplicate_count": len(current_hashes) - len(set(current_hashes)),
        "historical_exact_overlap_count": len(overlaps),
        "individual_private_text_exported": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-evidence", action="store_true")
    args = parser.parse_args()
    if args.clean_evidence and EVIDENCE_ROOT.exists():
        shutil.rmtree(EVIDENCE_ROOT)

    missing_sources = [name for name, path in SOURCE_FILES.items() if not path.is_file()]
    if missing_sources:
        raise SystemExit(f"Phase69 source files missing: {missing_sources}")
    model_files = [MODEL_PATH / "config.json", MODEL_PATH / "tokenizer_config.json"]
    if not MODEL_PATH.is_dir() or any(not path.is_file() for path in model_files):
        raise SystemExit(f"Phase69 local model is unavailable: {MODEL_PATH}")

    phase68 = _phase68_snapshot()
    holdout = build_phase69_holdout()
    overlap = _task_overlap_audit(holdout)
    generation = {
        "model_path": str(MODEL_PATH),
        "input_max_length": 4096,
        "max_new_tokens": 192,
        "do_sample": False,
        "repetition_penalty": 1.05,
        "enable_thinking": False,
        "turns_per_session": 3,
        "privacy_runtime_enabled": True,
        "latest_intent_runtime_enabled": True,
    }
    generation["generation_protocol_sha256"] = stable_hash(generation)
    protocol = {
        "kind": "phase69_frozen_minimal_runtime_ab_protocol",
        "variants": list(PHASE69_VARIANTS),
        "only_ab_variable": "candidate_provenance_boundary_contract",
        "same_model_tasks_decoding_and_scorer": True,
        "generation": generation,
        "semantic_judge_aliases": list(JUDGE_ALIASES),
        "semantic_judge_models_private": dict(JUDGE_MODELS),
        "num_ctx": 4096,
        "num_predict": 64,
        "temperature": 0,
        "think": False,
        "parallel_worker_count": 4,
        "frozen_retry_limit_per_failed_item": 2,
        "typed_wire_spec": {
            "version": PHASE63_WIRE_VERSION,
            "pattern": PHASE63_WIRE_PATTERN.pattern,
            "field_prefixes": dict(PHASE63_FIELD_PREFIXES),
        },
        "decision_gates": {
            "candidate_boundary_accept_rate_min": PHASE69_ACCEPT_RATE_GATE,
            "candidate_accept_rate_delta_min": PHASE69_ACCEPT_RATE_DELTA_GATE,
            "candidate_dangerous_or_reject_count_max": 0,
            "judge_schema_failure_count_max": 0,
            "candidate_value_conflict_count_max": 0,
            "ordinary_candidate_not_below_baseline": True,
            "ordinary_boundary_leak_count_max": 0,
        },
        "training_allowed": False,
        "adapter_allowed": False,
        "hermes_attachment_allowed": False,
        "product_default_change_allowed": False,
        "auto_promote_allowed": False,
    }
    protocol["protocol_sha256"] = stable_hash(protocol)
    source_hashes = {name: _sha256(path) for name, path in SOURCE_FILES.items()}
    freeze = {
        "kind": "phase69_pre_model_call_freeze",
        "phase68_snapshot_sha256": stable_hash(phase68),
        "holdout_sha256": stable_hash(holdout),
        "protocol_sha256": protocol["protocol_sha256"],
        "source_sha256": source_hashes,
        "model_config_sha256": {str(path.relative_to(REPO_ROOT)): _sha256(path) for path in model_files},
        "gates_frozen_before_model_calls": True,
        "tasks_frozen_before_model_calls": True,
        "created_at": _utcnow(),
    }
    preparation_checks = {
        "phase68_evaluator_snapshot_passed": phase68["passed"],
        "task_overlap_audit_passed": overlap["passed"],
        "session_count_exact": holdout["session_count"] == 48,
        "boundary_count_exact": holdout["boundary_session_count"] == 36,
        "ordinary_count_exact": holdout["ordinary_session_count"] == 12,
        "all_rows_simulated_not_training": all(
            row.get("simulated_usage") is True
            and row.get("actual_user_feedback") is False
            and row.get("not_for_training") is True
            for row in holdout["sessions"]
        ),
        "local_qwen3_4b_present": MODEL_PATH.is_dir(),
        "source_files_complete": not missing_sources,
    }
    preparation = {
        "kind": "phase69_preparation_decision",
        "status": "ready_for_real_generation" if all(preparation_checks.values()) else "blocked",
        "checks": preparation_checks,
        "failed_checks": [key for key, value in preparation_checks.items() if not value],
        "model_calls_executed": False,
        "training_executed": False,
        "created_at": _utcnow(),
    }

    _write_json(EVIDENCE_ROOT / "evidence-baseline/phase68_qualified_evaluator_snapshot.json", phase68)
    _write_json(EVIDENCE_ROOT / "evidence-holdout/holdout.json", holdout)
    _write_jsonl(EVIDENCE_ROOT / "evidence-holdout/holdout_sessions.jsonl", holdout["sessions"])
    _write_json(EVIDENCE_ROOT / "evidence-holdout/task_overlap_audit.json", overlap)
    _write_json(EVIDENCE_ROOT / "runtime_ab_protocol.json", protocol)
    _write_json(EVIDENCE_ROOT / "pre_model_call_freeze.json", freeze)
    _write_json(EVIDENCE_ROOT / "preparation_decision.json", preparation)
    _write_json(
        EVIDENCE_ROOT / "source_manifest.json",
        {
            "kind": "phase69_source_manifest",
            "phase68_manifest_sha256": phase68.get("manifest_sha256"),
            "local_model_path": str(MODEL_PATH),
            "source_sha256": source_hashes,
            "actual_user_feedback_count": 0,
            "simulated_usage_only": True,
            "not_for_training": True,
        },
    )
    print(json.dumps(preparation, ensure_ascii=False, indent=2))
    return 0 if preparation["status"] == "ready_for_real_generation" else 1


if __name__ == "__main__":
    raise SystemExit(main())
