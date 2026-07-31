#!/usr/bin/env python3
"""Freeze Phase70 tasks, sparse JSON transport checks, and A/B gates."""

from __future__ import annotations

import argparse
from collections import Counter
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

from pfe_core.phase59_proposition_addressed_grounding import (
    build_phase59_proposition_candidates,
    phase59_ollama_json_schema,
)
from pfe_core.phase63_field_typed_candidate_wire import (
    PHASE63_FIELD_PREFIXES,
    PHASE63_WIRE_PATTERN,
    PHASE63_WIRE_VERSION,
)
from pfe_core.phase68_aligned_candidate_scope_recovery import PHASE68_CATEGORIES
from pfe_core.phase70_structured_boundary_contract import (
    PHASE70_ACCEPT_RATE_DELTA_GATE,
    PHASE70_ACCEPT_RATE_GATE,
    PHASE70_EXACT_STRUCTURE_GATE,
    PHASE70_VARIANTS,
    build_phase70_holdout,
    build_phase70_sparse_preflight_cases,
    stable_hash,
)


EVIDENCE_ROOT = REPO_ROOT / "docs/demo/phase70-structured-boundary-contract"
PHASE69_ROOT = REPO_ROOT / "docs/demo/phase69-minimal-runtime-ab"
PHASE68_ROOT = REPO_ROOT / "docs/demo/phase68-aligned-candidate-scope-recovery"
MODEL_PATH = REPO_ROOT / "models/Qwen3-4B"
SOURCE_FILES = {
    "phase46_generator_helpers": REPO_ROOT / "tools/phase46_qwen3_4b_generate.py",
    "phase53_hard_detector": CORE_ROOT / "pfe_core/phase53_evaluator_scope_recovery.py",
    "phase56_grounder_composer": CORE_ROOT / "pfe_core/phase56_evidence_span_grounded_atomic.py",
    "phase59_candidates": CORE_ROOT / "pfe_core/phase59_proposition_addressed_grounding.py",
    "phase63_typed_wire": CORE_ROOT / "pfe_core/phase63_field_typed_candidate_wire.py",
    "phase62_consensus": CORE_ROOT / "pfe_core/phase62_risk_asymmetric_candidate_consensus.py",
    "phase70_core": CORE_ROOT / "pfe_core/phase70_structured_boundary_contract.py",
    "phase70_prepare": Path(__file__).resolve(),
    "phase70_generate": REPO_ROOT / "tools/phase70_generate.py",
    "phase70_prepare_product_eval": REPO_ROOT / "tools/phase70_prepare_product_eval.py",
    "phase70_execute_eval": REPO_ROOT / "tools/phase70_execute_eval.py",
    "phase70_finalize": REPO_ROOT / "tools/phase70_finalize_evidence.py",
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


def _verify_manifest(root: Path) -> tuple[bool, str]:
    manifest = _read_json(root / "evidence_manifest.json")
    files = list(manifest.get("files") or [])
    passed = bool(files) and all(
        (REPO_ROOT / str(row.get("path") or "")).is_file()
        and _sha256(REPO_ROOT / str(row.get("path") or "")) == row.get("sha256")
        for row in files
    )
    return passed, str(manifest.get("manifest_sha256") or "")


def _phase69_snapshot() -> dict[str, Any]:
    decision = _read_json(PHASE69_ROOT / "phase69-final-decision.json")
    integrity = _read_json(PHASE69_ROOT / "evidence_integrity.json")
    comparison = _read_json(PHASE69_ROOT / "comparison_summary.json")
    manifest_ok, manifest_sha = _verify_manifest(PHASE69_ROOT)
    checks = {
        "phase69_held": decision.get("recommendation") == "hold_phase69_minimal_runtime_ab",
        "phase69_integrity_passed": integrity.get("passed") is True,
        "phase69_manifest_verified": manifest_ok,
        "phase69_zero_accept_preserved": decision.get("boundary_accept_rate_candidate") == 0.0,
        "phase69_no_training_or_default_change": comparison.get("training_executed") is False
        and comparison.get("product_default_changed") is False,
    }
    return {
        "kind": "phase70_phase69_hold_snapshot",
        "passed": all(checks.values()),
        "checks": checks,
        "manifest_sha256": manifest_sha,
        "recommendation": decision.get("recommendation"),
    }


def _blind_cases(cases: list[dict[str, Any]], prefix: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    public = []
    hidden = []
    for index, row in enumerate(cases, start=1):
        item_id = f"{prefix}-{index:03d}"
        response = str(row["assistant_response"])
        public.append(
            {
                "item_id": item_id,
                "assistant_response": response,
                "proposition_candidates": build_phase59_proposition_candidates(response),
                "simulated_evaluator_fixture": True,
                "actual_user_feedback": False,
                "not_for_training": True,
            }
        )
        hidden.append(
            {
                "item_id": item_id,
                "case_id": row["case_id"],
                "category": row["category"],
                "expected_label": row["expected_label"],
                "expected_typed": row["expected_typed"],
                "expected_candidate_ids": row["expected_candidate_ids"],
            }
        )
    return public, hidden


def _phase68_regression_cases() -> list[dict[str, Any]]:
    public = {
        str(row.get("item_id")): row
        for row in _read_jsonl(
            PHASE68_ROOT / "evidence-evaluator-holdout/blind_items_public.jsonl"
        )
    }
    hidden = list(
        _read_json(
            PHASE68_ROOT / "evidence-evaluator-holdout/blind_hidden_key.json"
        ).get("items")
        or []
    )
    counts: Counter[tuple[str, str]] = Counter()
    rows = []
    for key in hidden:
        group = (str(key.get("category") or ""), str(key.get("expected_label") or ""))
        if counts[group] >= 2:
            continue
        source = public[str(key["item_id"])]
        rows.append(
            {
                "case_id": f"phase70-reg-{len(rows) + 1:03d}",
                "category": key["category"],
                "assistant_response": source["assistant_response"],
                "expected_label": key["expected_label"],
                "expected_typed": key["expected_typed"],
                "expected_candidate_ids": key["expected_candidate_ids"],
            }
        )
        counts[group] += 1
    expected_groups = {(category, label) for category in PHASE68_CATEGORIES for label in ("accept", "edit", "reject")}
    if len(rows) != 30 or set(counts) != expected_groups or any(value != 2 for value in counts.values()):
        raise AssertionError("Phase70 Phase68 regression subset is not balanced")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-evidence", action="store_true")
    args = parser.parse_args()
    if args.clean_evidence and EVIDENCE_ROOT.exists():
        shutil.rmtree(EVIDENCE_ROOT)
    missing = [name for name, path in SOURCE_FILES.items() if not path.is_file()]
    if missing:
        raise SystemExit(f"Phase70 source files missing: {missing}")
    phase69 = _phase69_snapshot()
    holdout = build_phase70_holdout()
    sparse = build_phase70_sparse_preflight_cases()
    regression = _phase68_regression_cases()
    sparse_public, sparse_hidden = _blind_cases(sparse["cases"], "phase70-sparse")
    regression_public, regression_hidden = _blind_cases(regression, "phase70-regression")
    previous_holdout = _read_json(PHASE69_ROOT / "evidence-holdout/holdout.json")
    current_texts = {
        stable_hash(
            [row.get(key) for key in ("user_goal", "user_correction", "continuation_request", "acceptance_request")]
        )
        for row in holdout["sessions"]
    }
    previous_texts = {
        stable_hash(
            [row.get(key) for key in ("user_goal", "user_correction", "continuation_request", "acceptance_request")]
        )
        for row in previous_holdout.get("sessions") or []
    }
    overlap = {
        "kind": "phase70_holdout_overlap_audit",
        "passed": len(current_texts) == 48 and not (current_texts & previous_texts),
        "phase70_unique_task_count": len(current_texts),
        "phase69_exact_overlap_count": len(current_texts & previous_texts),
    }
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
    schema_template = phase59_ollama_json_schema(
        build_phase59_proposition_candidates("\n".join(
            [
                "当前模拟材料不得登记为 actual_user_feedback。",
                "真实用户结果尚未确认。",
                "当前测试证据不能证明真实用户结果。",
            ]
        ))
    )
    typed_wire_spec = {
        "version": PHASE63_WIRE_VERSION,
        "pattern": PHASE63_WIRE_PATTERN.pattern,
        "field_order": [
            "source_registration_candidate_id",
            "user_outcome_status_candidate_id",
            "test_to_user_outcome_relation_candidate_id",
        ],
        "field_prefixes": dict(PHASE63_FIELD_PREFIXES),
        "field_local_numbering": True,
        "typed_to_internal_mapping_required": True,
        "example": "PFE2|s001|u001|none",
        "json_schema_used": False,
        "free_text_allowed": False,
    }
    protocol = {
        "kind": "phase70_frozen_structured_boundary_protocol",
        "variants": list(PHASE70_VARIANTS),
        "only_ab_variable": "freeform_vs_fixed_three_line_boundary_contract",
        "generation": generation,
        "judge_transport": "alias_capability_routed_candidate_transport",
        "judge_transport_by_alias": {
            "semantic_judge_alpha": "phase59_nested_json_schema",
            "semantic_judge_beta": "phase63_typed_wire",
        },
        "semantic_judge_aliases": list(JUDGE_ALIASES),
        "semantic_judge_models_private": dict(JUDGE_MODELS),
        "num_ctx": 4096,
        "num_predict_by_alias": {
            "semantic_judge_alpha": 192,
            "semantic_judge_beta": 64,
        },
        "temperature": 0,
        "think": False,
        "parallel_worker_count": 4,
        "retry_limit": 2,
        "nested_json_schema_template_sha256": stable_hash(schema_template),
        "typed_wire_spec": typed_wire_spec,
        "typed_wire_spec_sha256": stable_hash(typed_wire_spec),
        "decision_gates": {
            "transport_preflight_status": "qualified",
            "phase68_regression_status": "qualified",
            "candidate_accept_rate_min": PHASE70_ACCEPT_RATE_GATE,
            "candidate_accept_rate_delta_min": PHASE70_ACCEPT_RATE_DELTA_GATE,
            "candidate_exact_structure_rate_min": PHASE70_EXACT_STRUCTURE_GATE,
            "dangerous_schema_conflict_and_ordinary_leak_max": 0,
        },
        "training_allowed": False,
        "product_default_change_allowed": False,
        "auto_promote_allowed": False,
    }
    protocol["protocol_sha256"] = stable_hash(protocol)
    source_hashes = {name: _sha256(path) for name, path in SOURCE_FILES.items()}
    freeze = {
        "kind": "phase70_pre_model_call_freeze",
        "phase69_snapshot_sha256": stable_hash(phase69),
        "holdout_sha256": stable_hash(holdout),
        "sparse_public_sha256": stable_hash(sparse_public),
        "sparse_hidden_sha256": stable_hash(sparse_hidden),
        "regression_public_sha256": stable_hash(regression_public),
        "regression_hidden_sha256": stable_hash(regression_hidden),
        "protocol_sha256": protocol["protocol_sha256"],
        "source_sha256": source_hashes,
        "frozen_before_any_phase70_model_call": True,
        "created_at": _utcnow(),
    }
    checks = {
        "phase69_snapshot_passed": phase69["passed"],
        "holdout_overlap_audit_passed": overlap["passed"],
        "holdout_counts_exact": holdout["session_count"] == 48
        and holdout["boundary_session_count"] == 36
        and holdout["ordinary_session_count"] == 12,
        "sparse_preflight_count_exact": len(sparse_public) == 12,
        "phase68_regression_balanced_count_exact": len(regression_public) == 30,
        "local_model_present": MODEL_PATH.is_dir(),
        "source_files_complete": not missing,
    }
    decision = {
        "kind": "phase70_preparation_decision",
        "status": "ready_for_sparse_transport_preflight" if all(checks.values()) else "blocked",
        "checks": checks,
        "failed_checks": [key for key, value in checks.items() if not value],
        "created_at": _utcnow(),
    }
    _write_json(EVIDENCE_ROOT / "evidence-baseline/phase69_hold_snapshot.json", phase69)
    _write_json(EVIDENCE_ROOT / "evidence-holdout/holdout.json", holdout)
    _write_json(EVIDENCE_ROOT / "evidence-holdout/overlap_audit.json", overlap)
    for directory, public, hidden in (
        ("evidence-sparse-preflight", sparse_public, sparse_hidden),
        ("evidence-phase68-regression", regression_public, regression_hidden),
    ):
        _write_jsonl(EVIDENCE_ROOT / directory / "blind_items_public.jsonl", public)
        _write_json(
            EVIDENCE_ROOT / directory / "blind_hidden_key.json",
            {"item_count": len(hidden), "items": hidden, "hidden_from_judges": True},
        )
    _write_json(EVIDENCE_ROOT / "runtime_ab_protocol.json", protocol)
    _write_json(EVIDENCE_ROOT / "pre_model_call_freeze.json", freeze)
    _write_json(EVIDENCE_ROOT / "preparation_decision.json", decision)
    _write_json(
        EVIDENCE_ROOT / "source_manifest.json",
        {
            "kind": "phase70_source_manifest",
            "source_sha256": source_hashes,
            "actual_user_feedback_count": 0,
            "not_for_training": True,
        },
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return 0 if decision["status"] == "ready_for_sparse_transport_preflight" else 1


if __name__ == "__main__":
    raise SystemExit(main())
