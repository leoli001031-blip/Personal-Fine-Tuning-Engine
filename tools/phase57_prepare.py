#!/usr/bin/env python3
"""Freeze Phase57 historical replay evidence before any model calls."""

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
from pfe_core.phase56_evidence_span_grounded_atomic import (
    PHASE56_EXTRACTION_RUBRIC,
    phase56_ollama_json_schema,
)
from pfe_core.phase57_span_evaluator_historical_replay import (
    PHASE57_GROUNDING_VALIDITY_GATE,
    PHASE57_OVERALL_ACCURACY_GATE,
    PHASE57_PER_CATEGORY_ACCURACY_GATE,
    PHASE57_PER_PHASE_ACCURACY_GATE,
    PHASE57_PHASES,
    build_phase57_blind_replay,
    build_phase57_replay_integrity,
)


EVIDENCE_ROOT = REPO_ROOT / "docs/demo/phase57-span-evaluator-historical-replay"
PHASE56_ROOT = REPO_ROOT / "docs/demo/phase56-evidence-span-grounded-atomic"
PHASE53_SOURCE = CORE_ROOT / "pfe_core/phase53_evaluator_scope_recovery.py"
PHASE56_SOURCE = CORE_ROOT / "pfe_core/phase56_evidence_span_grounded_atomic.py"
PHASE57_SOURCE = CORE_ROOT / "pfe_core/phase57_span_evaluator_historical_replay.py"
HISTORICAL_ROOTS = {
    "phase51": REPO_ROOT / "docs/demo/phase51-dual-evaluator-hardening",
    "phase52": REPO_ROOT / "docs/demo/phase52-adversarial-evaluator-generalization",
    "phase53": REPO_ROOT / "docs/demo/phase53-evaluator-scope-recovery",
    "phase54": REPO_ROOT / "docs/demo/phase54-typed-proposition-evaluator",
    "phase55": REPO_ROOT / "docs/demo/phase55-atomic-boundary-composition",
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


def _phase56_snapshot() -> dict[str, Any]:
    manifest = _read_json(PHASE56_ROOT / "evidence_manifest.json")
    mismatches = []
    for item in manifest.get("files") or []:
        path = REPO_ROOT / str(item.get("path") or "")
        current = _sha256(path) if path.exists() else None
        if current != item.get("sha256"):
            mismatches.append({"path": item.get("path"), "expected": item.get("sha256"), "current": current})
    decision = _read_json(PHASE56_ROOT / "phase56-final-decision.json")
    integrity = _read_json(PHASE56_ROOT / "evidence_integrity.json")
    passed = (
        not mismatches
        and integrity.get("passed") is True
        and decision.get("recommendation") == "recommend_phase56_span_evaluator_for_manual_review_only"
    )
    return {
        "kind": "phase57_phase56_canonical_snapshot",
        "passed": passed,
        "phase56_recommendation": decision.get("recommendation"),
        "phase56_manifest_sha256": manifest.get("manifest_sha256"),
        "phase56_manifest_file_count": manifest.get("file_count"),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "created_at": _utcnow(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-evidence", action="store_true")
    args = parser.parse_args()
    if args.clean_evidence and EVIDENCE_ROOT.exists():
        debug_root = EVIDENCE_ROOT / "evidence-replay-debug"
        debug_backup = EVIDENCE_ROOT.parent / ".phase57-evidence-replay-debug-backup"
        if debug_backup.exists():
            shutil.rmtree(debug_backup)
        if debug_root.exists():
            shutil.copytree(debug_root, debug_backup)
        shutil.rmtree(EVIDENCE_ROOT)
        if debug_backup.exists():
            shutil.copytree(debug_backup, debug_root)
            shutil.rmtree(debug_backup)

    baseline = _phase56_snapshot()
    historical_cases = {}
    source_rows = []
    for phase in PHASE57_PHASES:
        path = HISTORICAL_ROOTS[phase] / "evidence-evaluator-holdout/holdout_labeled.json"
        payload = _read_json(path)
        historical_cases[phase] = [dict(row) for row in payload.get("cases") or []]
        source_rows.append(
            {
                "phase": phase,
                "path": str(path.relative_to(REPO_ROOT)),
                "sha256": _sha256(path),
                "case_count": len(historical_cases[phase]),
                "label_counts": payload.get("label_counts"),
            }
        )
    blind = build_phase57_blind_replay(historical_cases, seed=5701)
    integrity = build_phase57_replay_integrity(
        historical_cases=historical_cases,
        public_items=blind["public_items"],
        hidden_key=blind["hidden_key"],
    )
    schema = phase56_ollama_json_schema()
    protocol = {
        "kind": "phase57_frozen_phase56_historical_replay_protocol",
        "historical_phases": list(PHASE57_PHASES),
        "replay_item_count": len(blind["public_items"]),
        "semantic_judge_aliases": list(JUDGE_ALIASES),
        "semantic_judge_models_private": dict(JUDGE_MODELS),
        "judges_receive_historical_phase": False,
        "judges_receive_gold_label": False,
        "judges_receive_other_judge_identity": False,
        "judges_return_direct_label": False,
        "phase56_prompt_schema_grounding_and_composer_unchanged": True,
        "ollama_json_schema": schema,
        "overall_accuracy_gate": PHASE57_OVERALL_ACCURACY_GATE,
        "per_phase_accuracy_gate": PHASE57_PER_PHASE_ACCURACY_GATE,
        "per_category_accuracy_gate": PHASE57_PER_CATEGORY_ACCURACY_GATE,
        "grounding_validity_gate": PHASE57_GROUNDING_VALIDITY_GATE,
        "invalid_dangerous_atom_gate": 0,
        "false_accept_gate": 0,
        "hard_reject_vs_two_safe_accept_conflict_gate": 0,
        "composer_received_ungrounded_atom_gate": 0,
        "temperature": 0,
        "think": False,
        "num_ctx": 4096,
        "num_predict": 384,
        "parallel_worker_count": 4,
        "one_independent_call_per_item_per_judge": True,
        "parallel_dispatch_changes_evaluator_semantics": False,
        "historical_replay_may_be_called_once": True,
        "post_replay_tuning_allowed": False,
        "runtime_replay_allowed": False,
        "runtime_prompt_or_router_change_allowed": False,
        "training_allowed": False,
        "hermes_attachment_allowed": False,
        "product_default_change_allowed": False,
    }
    protocol["protocol_sha256"] = stable_hash(protocol)
    freeze = {
        "kind": "phase57_pre_model_call_freeze",
        "phase56_canonical_snapshot_passed": baseline["passed"],
        "phase53_hard_detector_source_sha256": _sha256(PHASE53_SOURCE),
        "phase56_evaluator_source_sha256": _sha256(PHASE56_SOURCE),
        "phase57_replay_source_sha256": _sha256(PHASE57_SOURCE),
        "phase56_extraction_rubric_sha256": stable_hash(PHASE56_EXTRACTION_RUBRIC),
        "phase56_json_schema_sha256": stable_hash(schema),
        "public_items_sha256": stable_hash(blind["public_items"]),
        "hidden_key_sha256": stable_hash(blind["hidden_key"]),
        "historical_sources_sha256": stable_hash(source_rows),
        "protocol_sha256": protocol["protocol_sha256"],
        "frozen_before_replay_calls": True,
        "created_at": _utcnow(),
    }
    source_manifest = {
        "kind": "phase57_historical_source_manifest",
        "sources": source_rows,
        "phase_counts": blind["phase_counts"],
        "label_counts": blind["label_counts"],
        "replay_item_count": len(blind["public_items"]),
        "simulated_evaluator_fixture": True,
        "actual_user_feedback_count": 0,
        "not_for_training": True,
        "private_user_material_used": False,
    }
    preparation = {
        "kind": "phase57_preparation_decision",
        "status": "ready_for_historical_replay" if baseline["passed"] and integrity["passed"] else "blocked",
        "phase56_canonical_passed": baseline["passed"],
        "historical_replay_integrity_passed": integrity["passed"],
        "replay_item_count": len(blind["public_items"]),
        "phase56_evaluator_unchanged": True,
        "post_replay_tuning_allowed": False,
    }
    no_runtime = {
        "kind": "phase57_runtime_status",
        "runtime_replay_status": "not_requested_in_phase57",
        "runtime_replay_model_call_count": 0,
        "runtime_prompt_changed": False,
        "router_changed": False,
    }
    training = {
        "kind": "phase57_training_attempt",
        "status": "not_requested",
        "training_executed": False,
        "adapter_created": False,
        "auto_training_allowed": False,
    }

    replay_dir = EVIDENCE_ROOT / "evidence-historical-replay"
    _write_json(EVIDENCE_ROOT / "evidence-baseline/phase56_canonical_snapshot.json", baseline)
    _write_json(replay_dir / "historical_source_manifest.json", source_manifest)
    _write_jsonl(replay_dir / "blind_items_public.jsonl", blind["public_items"])
    _write_json(replay_dir / "blind_hidden_key.json", {"items": blind["hidden_key"]})
    _write_json(replay_dir / "replay_integrity.json", integrity)
    _write_json(EVIDENCE_ROOT / "evidence-no-runtime/runtime_status.json", no_runtime)
    _write_json(EVIDENCE_ROOT / "evidence-no-training/training_attempt.json", training)
    _write_json(EVIDENCE_ROOT / "evaluator_protocol.json", protocol)
    _write_json(EVIDENCE_ROOT / "pre_model_call_freeze.json", freeze)
    _write_json(EVIDENCE_ROOT / "source_manifest.json", source_manifest)
    _write_json(EVIDENCE_ROOT / "preparation_decision.json", preparation)
    print(json.dumps(preparation, ensure_ascii=False, indent=2))
    return 0 if preparation["status"] == "ready_for_historical_replay" else 1


if __name__ == "__main__":
    raise SystemExit(main())
