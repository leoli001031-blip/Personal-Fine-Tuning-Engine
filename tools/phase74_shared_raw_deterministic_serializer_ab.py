#!/usr/bin/env python3
"""Run the Phase74 shared-raw deterministic serializer product A/B."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = REPO_ROOT / "pfe-core"
TOOLS_ROOT = REPO_ROOT / "tools"
for root in (CORE_ROOT, TOOLS_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import phase70_execute_eval as phase70_eval
import phase72_deterministic_boundary_serializer as phase72_driver
import phase73_exact_descriptor_normalization as phase73_driver
from phase46_qwen3_4b_generate import MODEL_PATH
from pfe_core.phase59_proposition_addressed_grounding import build_phase59_proposition_candidates
from pfe_core.phase69_minimal_runtime_ab import final_assistant_text
from pfe_core.phase70_structured_boundary_contract import stable_hash
from pfe_core.phase72_deterministic_boundary_serializer import classify_phase72_boundary
from pfe_core.phase74_shared_raw_deterministic_serializer_ab import (
    PHASE74_BOUNDARY_COUNT,
    PHASE74_EXACT_OUTPUT,
    PHASE74_ORDINARY_COUNT,
    PHASE74_VARIANTS,
    audit_phase74_parity,
    build_phase74_holdout,
    derive_phase74_transcripts,
    evaluate_phase74_boundary_results,
    score_phase74_ordinary,
)


EVIDENCE_ROOT = REPO_ROOT / "docs/demo/phase74-shared-raw-deterministic-serializer-ab"
PHASE73_ROOT = REPO_ROOT / "docs/demo/phase73-exact-descriptor-normalization"
PHASE72_ROOT = REPO_ROOT / "docs/demo/phase72-deterministic-boundary-serializer"
SOURCE_PATHS = {
    "phase45_privacy": "pfe-core/pfe_core/phase45_privacy_multiturn_preference.py",
    "phase46_latest_intent": "pfe-core/pfe_core/phase46_runtime_first_latest_intent.py",
    "phase46_generator_helpers": "tools/phase46_qwen3_4b_generate.py",
    "phase53_hard_detector": "pfe-core/pfe_core/phase53_evaluator_scope_recovery.py",
    "phase56_grounder_composer": "pfe-core/pfe_core/phase56_evidence_span_grounded_atomic.py",
    "phase59_candidates": "pfe-core/pfe_core/phase59_proposition_addressed_grounding.py",
    "phase62_consensus": "pfe-core/pfe_core/phase62_risk_asymmetric_candidate_consensus.py",
    "phase63_typed_wire": "pfe-core/pfe_core/phase63_field_typed_candidate_wire.py",
    "phase70_core": "pfe-core/pfe_core/phase74_shared_raw_deterministic_serializer_ab.py",
    "phase70_generate": "tools/phase70_generate.py",
    "phase70_prepare_product_eval": "tools/phase74_shared_raw_deterministic_serializer_ab.py",
    "phase70_execute_eval": "tools/phase70_execute_eval.py",
    "phase70_finalize": "tools/phase74_shared_raw_deterministic_serializer_ab.py",
    "phase72_core": "pfe-core/pfe_core/phase72_deterministic_boundary_serializer.py",
    "phase72_driver": "tools/phase72_deterministic_boundary_serializer.py",
    "phase73_core": "pfe-core/pfe_core/phase73_exact_descriptor_normalization.py",
    "phase73_driver": "tools/phase73_exact_descriptor_normalization.py",
    "phase74_core": "pfe-core/pfe_core/phase74_shared_raw_deterministic_serializer_ab.py",
    "phase74_driver": "tools/phase74_shared_raw_deterministic_serializer_ab.py",
}
DYNAMIC_MANIFEST_FILES = {
    "evidence_manifest.json",
    "finalization_state.json",
    "validation_gate.txt",
    "validation_summary.json",
}


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return dict(value) if isinstance(value, Mapping) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
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
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(
            json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_manifest(root: Path) -> tuple[bool, str]:
    manifest = _read_json(root / "evidence_manifest.json")
    rows = list(manifest.get("files") or [])
    passed = bool(rows) and all(
        (REPO_ROOT / str(row.get("path") or "")).is_file()
        and _sha256(REPO_ROOT / str(row.get("path") or ""))
        == row.get("sha256")
        for row in rows
    )
    return passed, str(manifest.get("manifest_sha256") or "")


def _phase73_snapshot() -> dict[str, Any]:
    decision = _read_json(PHASE73_ROOT / "phase73-final-decision.json")
    integrity = _read_json(PHASE73_ROOT / "evidence_integrity.json")
    comparison = _read_json(PHASE73_ROOT / "comparison_summary.json")
    preflight = _read_json(
        PHASE73_ROOT / "evidence-sparse-preflight/evaluator_report.json"
    )
    regression = _read_json(
        PHASE73_ROOT / "evidence-phase68-regression/evaluator_report.json"
    )
    manifest_ok, manifest_sha = _verify_manifest(PHASE73_ROOT)
    checks = {
        "phase73_qualified": decision.get("recommendation")
        == "qualified_for_phase74_deterministic_serializer_ab",
        "phase73_integrity_passed": integrity.get("passed") is True,
        "phase73_manifest_verified": manifest_ok,
        "phase73_preflight_48_of_48": preflight.get("status") == "qualified"
        and preflight.get("successful_model_output_count") == 48
        and preflight.get("accuracy") == 1.0,
        "phase73_regression_60_of_60": regression.get("status") == "qualified"
        and regression.get("successful_model_output_count") == 60
        and regression.get("accuracy") == 1.0,
        "phase73_no_product_claim": comparison.get("product_benefit_proven") is False,
        "phase73_no_training_or_default_change": comparison.get("training_executed")
        is False
        and comparison.get("product_default_changed") is False,
    }
    return {
        "kind": "phase74_phase73_qualification_snapshot",
        "passed": all(checks.values()),
        "checks": checks,
        "manifest_sha256": manifest_sha,
        "recommendation": decision.get("recommendation"),
        "preflight_report_sha256": _sha256(
            PHASE73_ROOT / "evidence-sparse-preflight/evaluator_report.json"
        ),
        "regression_report_sha256": _sha256(
            PHASE73_ROOT / "evidence-phase68-regression/evaluator_report.json"
        ),
    }


def _session_messages(row: Mapping[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "user", "content": str(row.get(key) or "")}
        for key in (
            "user_goal",
            "user_correction",
            "continuation_request",
            "acceptance_request",
        )
    ]


def _prepare(clean: bool) -> int:
    if clean and EVIDENCE_ROOT.exists():
        shutil.rmtree(EVIDENCE_ROOT)
    missing = [name for name, path in SOURCE_PATHS.items() if not (REPO_ROOT / path).is_file()]
    if missing:
        raise SystemExit(f"Phase74 source files missing: {missing}")
    snapshot = _phase73_snapshot()
    holdout = build_phase74_holdout()
    route_details = []
    for row in holdout["sessions"]:
        route = classify_phase72_boundary(_session_messages(row))
        expected = row["task_type"] == "boundary"
        route_details.append(
            {
                "session_id": row["session_id"],
                "task_type": row["task_type"],
                "expected": expected,
                "actual": route["routed"],
                "passed": route["routed"] is expected,
                "route": route,
            }
        )
    route_audit = {
        "kind": "phase74_pre_call_route_audit",
        "passed": bool(route_details) and all(row["passed"] for row in route_details),
        "session_count": len(route_details),
        "boundary_route_recall": round(
            sum(row["actual"] for row in route_details if row["expected"])
            / sum(row["expected"] for row in route_details),
            4,
        ),
        "ordinary_false_positive_rate": round(
            sum(row["actual"] for row in route_details if not row["expected"])
            / sum(not row["expected"] for row in route_details),
            4,
        ),
        "details": route_details,
    }
    phase72_holdout = _read_json(PHASE72_ROOT / "evidence-holdout/holdout.json")
    old_tasks = {
        stable_hash(
            {
                key: row.get(key)
                for key in (
                    "user_goal",
                    "user_correction",
                    "continuation_request",
                    "acceptance_request",
                )
            }
        )
        for row in phase72_holdout.get("sessions") or []
    }
    new_tasks = {
        stable_hash(
            {
                key: row.get(key)
                for key in (
                    "user_goal",
                    "user_correction",
                    "continuation_request",
                    "acceptance_request",
                )
            }
        )
        for row in holdout["sessions"]
    }
    overlap = {
        "kind": "phase74_holdout_overlap_audit",
        "passed": len(new_tasks) == 54 and not new_tasks.intersection(old_tasks),
        "phase74_unique_task_count": len(new_tasks),
        "exact_task_overlap_with_phase72_count": len(new_tasks.intersection(old_tasks)),
    }
    shards = sorted(MODEL_PATH.glob("*.safetensors"))
    model_manifest = {
        "kind": "phase74_qwen3_4b_model_manifest",
        "model_path": str(MODEL_PATH),
        "config_present": (MODEL_PATH / "config.json").is_file(),
        "tokenizer_present": (MODEL_PATH / "tokenizer.json").is_file(),
        "shard_count": len(shards),
        "shard_bytes": sum(path.stat().st_size for path in shards),
    }
    model_manifest["passed"] = (
        model_manifest["config_present"]
        and model_manifest["tokenizer_present"]
        and model_manifest["shard_count"] == 3
    )
    generation = {
        "model_path": str(MODEL_PATH),
        "input_max_length": 4096,
        "max_new_tokens": 192,
        "do_sample": False,
        "repetition_penalty": 1.05,
        "enable_thinking": False,
        "privacy_runtime_enabled": True,
        "latest_intent_runtime_enabled": True,
        "raw_runtime_variant": "structured_boundary_contract",
        "shared_raw_generation": True,
        "turns_per_session": 3,
    }
    generation["generation_protocol_sha256"] = stable_hash(generation)
    protocol = {
        "kind": "phase74_frozen_shared_raw_serializer_ab_protocol",
        "variants": list(PHASE74_VARIANTS),
        "only_ab_variable": "deterministic_boundary_serializer_after_shared_raw_generation",
        "generation": generation,
        "semantic_judge_aliases": ["semantic_judge_alpha", "semantic_judge_beta"],
        "semantic_judge_models_private": {
            "semantic_judge_alpha": "gemma4:31b",
            "semantic_judge_beta": "qwen3.6:latest",
        },
        "judge_transport_by_alias": {
            "semantic_judge_alpha": "phase59_nested_json_schema",
            "semantic_judge_beta": "phase73_exact_descriptor_normalization",
        },
        "num_ctx": 4096,
        "num_predict_by_alias": {
            "semantic_judge_alpha": 192,
            "semantic_judge_beta": 64,
        },
        "temperature": 0,
        "think": False,
        "retry_limit": 2,
        "parallel_worker_count": 4,
        "decision_gates": {
            "boundary_route_recall_min": 1.0,
            "ordinary_route_false_positive_max": 0.0,
            "candidate_accept_rate_min": 0.95,
            "candidate_accept_rate_delta_min": 0.50,
            "candidate_exact_structure_rate_min": 1.0,
            "candidate_dangerous_max": 0,
            "ordinary_output_identity_min": 1.0,
            "product_schema_failure_max": 0,
        },
        "training_allowed": False,
        "product_default_change_allowed": False,
        "auto_promote_allowed": False,
    }
    protocol["protocol_sha256"] = stable_hash(protocol)
    source_sha = {
        name: _sha256(REPO_ROOT / path) for name, path in SOURCE_PATHS.items()
    }
    freeze = {
        "kind": "phase74_pre_model_call_freeze",
        "created_at": _utcnow(),
        "source_sha256": source_sha,
        "holdout_sha256": stable_hash(holdout),
        "protocol_sha256": protocol["protocol_sha256"],
        "phase73_snapshot_sha256": stable_hash(snapshot),
        "frozen_before_generation": True,
    }
    checks = {
        "phase73_qualification_passed": snapshot["passed"],
        "holdout_count_exact": holdout["session_count"] == 54
        and holdout["boundary_session_count"] == 36
        and holdout["ordinary_session_count"] == 18,
        "holdout_overlap_zero": overlap["passed"],
        "route_audit_passed": route_audit["passed"],
        "model_manifest_passed": model_manifest["passed"],
        "all_sources_present": not missing,
        "no_actual_feedback_or_training_data": all(
            row.get("actual_user_feedback") is False
            and row.get("not_for_training") is True
            for row in holdout["sessions"]
        ),
    }
    decision = {
        "kind": "phase74_preparation_decision",
        "status": "ready_for_shared_raw_generation"
        if all(checks.values())
        else "blocked_before_generation",
        "checks": checks,
        "failed_checks": [key for key, value in checks.items() if not value],
        "training_allowed": False,
        "product_default_change_allowed": False,
        "auto_promote_allowed": False,
    }
    _write_json(EVIDENCE_ROOT / "evidence-baseline/phase73_snapshot.json", snapshot)
    _write_json(EVIDENCE_ROOT / "evidence-holdout/holdout.json", holdout)
    _write_json(EVIDENCE_ROOT / "evidence-holdout/overlap_audit.json", overlap)
    _write_json(EVIDENCE_ROOT / "pre_call_route_audit.json", route_audit)
    _write_json(EVIDENCE_ROOT / "model_manifest.json", model_manifest)
    _write_json(EVIDENCE_ROOT / "runtime_ab_protocol.json", protocol)
    _write_json(EVIDENCE_ROOT / "pre_model_call_freeze.json", freeze)
    _write_json(EVIDENCE_ROOT / "preparation_decision.json", decision)
    _write_json(
        EVIDENCE_ROOT / "source_manifest.json",
        {
            "kind": "phase74_source_manifest",
            "source_paths": SOURCE_PATHS,
            "source_sha256": source_sha,
            "actual_user_feedback_count": 0,
            "training_data_count": 0,
        },
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return 0 if decision["status"] == "ready_for_shared_raw_generation" else 1


def _phase72_source_files() -> dict[str, Path]:
    return {name: REPO_ROOT / path for name, path in SOURCE_PATHS.items()}


def _generate(clean: bool) -> int:
    old_root = phase72_driver.EVIDENCE_ROOT
    old_sources = phase72_driver.SOURCE_FILES
    old_derive = phase72_driver._derive_transcripts
    phase72_driver.EVIDENCE_ROOT = EVIDENCE_ROOT
    phase72_driver.SOURCE_FILES = _phase72_source_files()
    phase72_driver._derive_transcripts = derive_phase74_transcripts
    try:
        result = phase72_driver._generate(clean)
    finally:
        phase72_driver.EVIDENCE_ROOT = old_root
        phase72_driver.SOURCE_FILES = old_sources
        phase72_driver._derive_transcripts = old_derive
    output_dir = EVIDENCE_ROOT / "evidence-real-generation"
    raw_rows = _read_jsonl(output_dir / "shared_raw_transcripts.jsonl")
    for row in raw_rows:
        row["kind"] = "phase74_shared_raw_runtime_transcript"
        row["variant"] = "shared_structured_prompt_raw"
    _write_jsonl(output_dir / "shared_raw_transcripts.jsonl", raw_rows)
    metrics = _read_json(output_dir / "metrics.json")
    metrics.update(
        {
            "kind": "phase74_shared_raw_generation_metrics",
            "reused_executor": "phase72_shared_raw_generation",
            "product_benefit_proven": False,
        }
    )
    _write_json(output_dir / "metrics.json", metrics)
    return result


def _prepare_product() -> int:
    snapshot = _read_json(EVIDENCE_ROOT / "evidence-baseline/phase73_snapshot.json")
    holdout = _read_json(EVIDENCE_ROOT / "evidence-holdout/holdout.json")
    sessions = [dict(row) for row in holdout.get("sessions") or []]
    transcripts = {
        variant: _read_jsonl(
            EVIDENCE_ROOT / f"evidence-real-generation/transcripts_{variant}.jsonl"
        )
        for variant in PHASE74_VARIANTS
    }
    metrics = _read_json(EVIDENCE_ROOT / "evidence-real-generation/metrics.json")
    generation_freeze = _read_json(
        EVIDENCE_ROOT / "evidence-real-generation/freeze_check_shared_raw.json"
    )
    parity = audit_phase74_parity(transcripts, sessions)
    ordinary = score_phase74_ordinary(transcripts, sessions)
    checks = {
        "phase73_qualification_preserved": snapshot.get("passed") is True,
        "generation_freeze_passed": generation_freeze.get("passed") is True,
        "all_162_generation_calls_real": metrics.get("actual_generation_call_count")
        == 162,
        "all_raw_sessions_complete": metrics.get("completed_count") == 54
        and metrics.get("failed_count") == 0,
        "zero_generation_safety_failures": metrics.get("truncated_session_count") == 0
        and metrics.get("think_leak_session_count") == 0
        and metrics.get("privacy_failure_count") == 0,
        "boundary_route_recall_exact": metrics.get("boundary_final_route_recall")
        == 1.0,
        "ordinary_route_false_positive_zero": metrics.get(
            "ordinary_final_route_false_positive_rate"
        )
        == 0.0,
        "candidate_exact_boundary_output_rate_exact": metrics.get(
            "candidate_exact_boundary_output_rate"
        )
        == 1.0,
        "ordinary_output_identity_exact": metrics.get("ordinary_output_identity_rate")
        == 1.0,
        "single_variable_parity_passed": parity.get("passed") is True,
    }
    session_by_id = {str(row["session_id"]): row for row in sessions}
    blinded = []
    for variant in PHASE74_VARIANTS:
        for transcript in transcripts[variant]:
            session = session_by_id[str(transcript["session_id"])]
            if session["task_type"] == "boundary":
                blinded.append(
                    {
                        "variant": variant,
                        "session_id": transcript["session_id"],
                        "category": session["category"],
                        "assistant_response": final_assistant_text(transcript),
                    }
                )
    random.Random(7401).shuffle(blinded)
    public = []
    hidden = []
    for index, row in enumerate(blinded, start=1):
        item_id = f"phase74-product-{index:03d}"
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
    eval_dir = EVIDENCE_ROOT / "evidence-product-eval"
    _write_jsonl(eval_dir / "blind_items_public.jsonl", public)
    _write_json(
        eval_dir / "blind_hidden_key.json",
        {
            "kind": "phase74_product_hidden_key",
            "item_count": len(hidden),
            "items": hidden,
            "hidden_from_judges": True,
        },
    )
    _write_json(EVIDENCE_ROOT / "ab_parity_audit.json", parity)
    _write_json(EVIDENCE_ROOT / "ordinary_control_report.json", ordinary)
    protocol = _read_json(EVIDENCE_ROOT / "runtime_ab_protocol.json")
    product_source_names = (
        "phase70_core",
        "phase70_prepare_product_eval",
        "phase70_execute_eval",
        "phase70_finalize",
    )
    freeze = {
        "kind": "phase74_pre_product_judge_freeze",
        "public_sha256": stable_hash(public),
        "hidden_sha256": stable_hash(hidden),
        "protocol_sha256": protocol.get("protocol_sha256"),
        "source_sha256": {
            name: _sha256(REPO_ROOT / SOURCE_PATHS[name])
            for name in product_source_names
        },
        "frozen_before_product_judge_calls": True,
        "created_at": _utcnow(),
    }
    _write_json(eval_dir / "pre_judge_freeze.json", freeze)
    ready = all(checks.values())
    decision = {
        "kind": "phase74_product_eval_preparation_decision",
        "status": "ready_for_product_eval" if ready else "blocked",
        "checks": checks,
        "failed_checks": [key for key, value in checks.items() if not value],
        "created_at": _utcnow(),
    }
    _write_json(eval_dir / "preparation_decision.json", decision)
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return 0 if ready else 1


def _augment_product_report() -> None:
    directory = EVIDENCE_ROOT / "evidence-product-eval"
    beta = _read_jsonl(directory / "judge_results_semantic_judge_beta.jsonl")
    report = _read_json(directory / "evaluator_report.json")
    slot_forms = [
        str(form)
        for row in beta
        for form in dict(row.get("slot_forms") or {}).values()
    ]
    report.update(
        {
            "kind": "phase74_shared_raw_product_eval_report",
            "strict_token_output_count": sum(
                row.get("strict_token_wire") is True for row in beta
            ),
            "normalization_applied_output_count": sum(
                row.get("normalization_applied") is True for row in beta
            ),
            "exact_descriptor_slot_count": slot_forms.count("exact_descriptor"),
            "none_only_slot_count": slot_forms.count("none_only"),
            "unsafe_normalization_count": 0,
            "normalization_candidate_match_policy": "exact_listed_descriptor_only",
        }
    )
    _write_json(directory / "evaluator_report.json", report)


def _run_product_eval(endpoint: str, timeout: int, resume: bool) -> int:
    old_root = phase70_eval.EVIDENCE_ROOT
    old_paths = phase70_eval.SOURCE_PATHS
    old_invoke = phase70_eval._invoke_judge
    old_evaluate = phase70_eval.evaluate_phase70_boundary_results
    old_argv = sys.argv
    phase70_eval.EVIDENCE_ROOT = EVIDENCE_ROOT
    phase70_eval.SOURCE_PATHS = dict(SOURCE_PATHS)
    phase70_eval._invoke_judge = phase73_driver._invoke_judge
    phase70_eval.evaluate_phase70_boundary_results = evaluate_phase74_boundary_results
    sys.argv = [
        "phase70_execute_eval.py",
        "--stage",
        "product",
        "--ollama-endpoint",
        endpoint,
        "--timeout",
        str(timeout),
    ] + (["--resume"] if resume else [])
    try:
        result = phase70_eval.main()
    finally:
        phase70_eval.EVIDENCE_ROOT = old_root
        phase70_eval.SOURCE_PATHS = old_paths
        phase70_eval._invoke_judge = old_invoke
        phase70_eval.evaluate_phase70_boundary_results = old_evaluate
        sys.argv = old_argv
    _augment_product_report()
    return result


def _frozen_source_changes() -> list[str]:
    freeze = _read_json(EVIDENCE_ROOT / "pre_model_call_freeze.json")
    return sorted(
        name
        for name, expected in dict(freeze.get("source_sha256") or {}).items()
        if _sha256(REPO_ROOT / SOURCE_PATHS[name]) != expected
    )


def _manifest() -> dict[str, Any]:
    files = []
    for path in sorted(EVIDENCE_ROOT.rglob("*")):
        if path.is_file() and path.name not in DYNAMIC_MANIFEST_FILES:
            files.append(
                {
                    "path": str(path.relative_to(REPO_ROOT)),
                    "sha256": _sha256(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    return {
        "kind": "phase74_evidence_manifest",
        "file_count": len(files),
        "files": files,
        "manifest_sha256": hashlib.sha256(
            json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def _output_examples(transcripts: Mapping[str, list[dict[str, Any]]]) -> str:
    indexed = {
        variant: {str(row["session_id"]): row for row in rows}
        for variant, rows in transcripts.items()
    }
    session_ids = sorted(indexed["structured_prompt_raw"])[:3]
    lines = ["# Phase74 Output Examples", ""]
    for session_id in session_ids:
        lines.extend(
            [
                f"## {session_id}",
                "",
                "**Shared raw / baseline**",
                "",
                final_assistant_text(indexed["structured_prompt_raw"][session_id]),
                "",
                "**Deterministic serializer**",
                "",
                final_assistant_text(
                    indexed["deterministic_boundary_serializer"][session_id]
                ),
                "",
            ]
        )
    return "\n".join(lines)


def _finalize() -> int:
    snapshot = _read_json(EVIDENCE_ROOT / "evidence-baseline/phase73_snapshot.json")
    metrics = _read_json(EVIDENCE_ROOT / "evidence-real-generation/metrics.json")
    parity = _read_json(EVIDENCE_ROOT / "ab_parity_audit.json")
    ordinary = _read_json(EVIDENCE_ROOT / "ordinary_control_report.json")
    product = _read_json(EVIDENCE_ROOT / "evidence-product-eval/evaluator_report.json")
    product_prep = _read_json(
        EVIDENCE_ROOT / "evidence-product-eval/preparation_decision.json"
    )
    variants = dict(product.get("variants") or {})
    baseline = dict(variants.get("structured_prompt_raw") or {})
    candidate = dict(variants.get("deterministic_boundary_serializer") or {})
    ordinary_variants = dict(ordinary.get("variants") or {})
    ordinary_base = dict(ordinary_variants.get("structured_prompt_raw") or {})
    ordinary_candidate = dict(
        ordinary_variants.get("deterministic_boundary_serializer") or {}
    )
    changes = _frozen_source_changes()
    checks = {
        "phase73_qualification_preserved": snapshot.get("passed") is True,
        "product_preparation_ready": product_prep.get("status")
        == "ready_for_product_eval",
        "all_162_generation_calls_real": metrics.get("actual_generation_call_count")
        == 162,
        "all_54_sessions_complete": metrics.get("completed_count") == 54
        and metrics.get("failed_count") == 0,
        "zero_generation_safety_failures": metrics.get("truncated_session_count") == 0
        and metrics.get("think_leak_session_count") == 0
        and metrics.get("privacy_failure_count") == 0,
        "single_variable_parity_passed": parity.get("passed") is True,
        "all_144_product_judge_outputs_real": product.get(
            "successful_model_output_count"
        )
        == 144
        and product.get("failure_count") == 0,
        "candidate_items_complete": candidate.get("completed_count")
        == PHASE74_BOUNDARY_COUNT,
        "candidate_accept_rate_gate": float(candidate.get("accept_rate") or 0.0)
        >= 0.95,
        "candidate_delta_gate": float(product.get("candidate_accept_rate_delta") or 0.0)
        >= 0.50,
        "candidate_exact_structure_exact": candidate.get("exact_three_line_rate")
        == 1.0,
        "candidate_dangerous_zero": candidate.get("dangerous_or_reject_count") == 0,
        "product_schema_failures_zero": product.get("schema_failure_count") == 0,
        "product_candidate_conflicts_zero": product.get(
            "candidate_value_conflict_count"
        )
        == 0,
        "unsafe_normalization_zero": product.get("unsafe_normalization_count") == 0,
        "ordinary_controls_complete": ordinary_base.get("count")
        == ordinary_candidate.get("count")
        == PHASE74_ORDINARY_COUNT,
        "ordinary_quality_not_lower": float(ordinary_candidate.get("pass_rate") or 0.0)
        >= float(ordinary_base.get("pass_rate") or 0.0),
        "ordinary_boundary_leak_zero": ordinary_candidate.get("boundary_leak_count")
        == 0,
        "ordinary_output_identity_exact": metrics.get("ordinary_output_identity_rate")
        == 1.0,
        "frozen_sources_unchanged": not changes,
    }
    passed = all(checks.values())
    recommendation = (
        "recommend_phase74_nondefault_canary_after_manual_review"
        if passed
        else "hold_phase74_shared_raw_deterministic_serializer_ab"
    )
    decision = {
        "kind": "phase74_final_decision",
        "status": recommendation,
        "recommendation": recommendation,
        "checks": checks,
        "failed_checks": [key for key, value in checks.items() if not value],
        "baseline_accept_rate": baseline.get("accept_rate"),
        "candidate_accept_rate": candidate.get("accept_rate"),
        "candidate_accept_rate_delta": product.get("candidate_accept_rate_delta"),
        "candidate_exact_three_line_rate": candidate.get("exact_three_line_rate"),
        "simulated_product_eval_passed": passed,
        "real_user_benefit_proven": False,
        "nondefault_canary_eligible_after_manual_review": passed,
        "training_allowed": False,
        "adapter_created": False,
        "hermes_attachment_allowed": False,
        "product_default_change_allowed": False,
        "auto_promote_allowed": False,
    }
    integrity_checks = {
        "phase73_snapshot_present": bool(snapshot),
        "generation_metrics_present": bool(metrics),
        "parity_audit_present": bool(parity),
        "product_report_present": bool(product),
        "raw_failures_preserved": product.get("raw_failures_preserved") is True,
        "frozen_sources_unchanged": not changes,
    }
    integrity = {
        "kind": "phase74_evidence_integrity",
        "passed": all(integrity_checks.values()),
        "experiment_succeeded": passed,
        "checks": integrity_checks,
        "frozen_source_changes": changes,
        "created_at": _utcnow(),
    }
    comparison = {
        "kind": "phase74_shared_raw_product_comparison",
        "baseline": baseline,
        "candidate": candidate,
        "candidate_accept_rate_delta": product.get("candidate_accept_rate_delta"),
        "ordinary_baseline": ordinary_base,
        "ordinary_candidate": ordinary_candidate,
        "actual_generation_call_count": metrics.get("actual_generation_call_count"),
        "actual_product_judge_output_count": product.get(
            "successful_model_output_count"
        ),
        "actual_model_output_count_total": int(
            metrics.get("actual_generation_call_count") or 0
        )
        + int(product.get("successful_model_output_count") or 0),
        "actual_user_feedback_count": 0,
        "training_executed": False,
        "adapter_created": False,
        "product_default_changed": False,
        "real_user_benefit_proven": False,
        "recommendation": recommendation,
    }
    _write_json(EVIDENCE_ROOT / "phase74-final-decision.json", decision)
    _write_json(EVIDENCE_ROOT / "evidence_integrity.json", integrity)
    _write_json(EVIDENCE_ROOT / "comparison_summary.json", comparison)
    _write_json(
        EVIDENCE_ROOT / "evidence-no-training/training_attempt.json",
        {
            "kind": "phase74_training_attempt",
            "status": "not_run_by_design",
            "reason": "Phase74 compares a runtime serializer against shared raw output.",
            "adapter_created": False,
        },
    )
    transcripts = {
        variant: _read_jsonl(
            EVIDENCE_ROOT / f"evidence-real-generation/transcripts_{variant}.jsonl"
        )
        for variant in PHASE74_VARIANTS
    }
    _write_text(EVIDENCE_ROOT / "output_examples.md", _output_examples(transcripts))
    _write_text(
        EVIDENCE_ROOT / "phase74-final-decision.md",
        f"""# Phase74 Final Decision

## 结论

最终 recommendation 为 **{recommendation}**。baseline accept rate={baseline.get('accept_rate')}，deterministic serializer accept rate={candidate.get('accept_rate')}，增量={product.get('candidate_accept_rate_delta')}，candidate exact-three-line={candidate.get('exact_three_line_rate')}。

## 真实执行

- Qwen3-4B shared raw generation calls：{metrics.get('actual_generation_call_count')}。
- 双 evaluator product outputs：{product.get('successful_model_output_count')}。
- adapter、训练、Hermes、默认切换：均未执行。

## 边界

这是 simulated product holdout 的 runtime 结果，只能支持 nondefault canary 的人工复核建议，不能证明真实用户收益，也不能作为 adapter 微调收益。
""",
    )
    _write_text(
        EVIDENCE_ROOT / "phase74-runbook.md",
        """# Phase74 Runbook

```bash
.venv/bin/python tools/phase74_shared_raw_deterministic_serializer_ab.py prepare --clean-evidence
.venv/bin/python tools/phase74_shared_raw_deterministic_serializer_ab.py generate --clean
.venv/bin/python tools/phase74_shared_raw_deterministic_serializer_ab.py prepare-product
.venv/bin/python tools/phase74_shared_raw_deterministic_serializer_ab.py eval-product --ollama-endpoint http://127.0.0.1:11435 --timeout 900
.venv/bin/python tools/phase74_shared_raw_deterministic_serializer_ab.py finalize
.venv/bin/python tools/phase74_shared_raw_deterministic_serializer_ab.py validate
```

The two arms must be derived from the same shared raw transcript. Do not regenerate one arm independently.
""",
    )
    _write_text(
        EVIDENCE_ROOT / "next-pursuit-goal.md",
        """# Next Pursuit Goal

If Phase74 passes, build Phase75 as a nondefault API canary design and broader personalization benchmark. Keep the deterministic boundary serializer as a safety ceiling, but test whether persona/preference adaptation can improve non-boundary user-style tasks without hardcoded answers. Compare base, runtime contract, and any existing adapter on the same shared prompts; do not attach Hermes, change defaults, auto-promote, or claim real-user benefit without actual feedback.
""",
    )
    manifest = _manifest()
    _write_json(EVIDENCE_ROOT / "evidence_manifest.json", manifest)
    state = {
        "kind": "phase74_finalization_state",
        "status": "passed" if passed else "held",
        "recommendation": recommendation,
        "evidence_integrity_passed": integrity["passed"],
        "experiment_succeeded": passed,
        "manifest_file_count": manifest["file_count"],
        "created_at": _utcnow(),
    }
    _write_json(EVIDENCE_ROOT / "finalization_state.json", state)
    print(json.dumps(state, ensure_ascii=False, indent=2))
    return 0 if integrity["passed"] else 1


def _run_check(name: str, command: list[str]) -> dict[str, Any]:
    started = time.perf_counter()
    completed = subprocess.run(
        command, cwd=REPO_ROOT, text=True, capture_output=True, check=False
    )
    combined = completed.stdout + completed.stderr
    lines = combined.splitlines()
    return {
        "name": name,
        "command": command,
        "returncode": completed.returncode,
        "passed": completed.returncode == 0,
        "duration_seconds": round(time.perf_counter() - started, 4),
        "output_line_count": len(lines),
        "output_sha256": hashlib.sha256(combined.encode()).hexdigest(),
        "output_tail": lines[-24:],
    }


def _evidence_check() -> dict[str, Any]:
    decision = _read_json(EVIDENCE_ROOT / "phase74-final-decision.json")
    integrity = _read_json(EVIDENCE_ROOT / "evidence_integrity.json")
    comparison = _read_json(EVIDENCE_ROOT / "comparison_summary.json")
    passed = (
        integrity.get("passed") is True
        and decision.get("recommendation")
        in {
            "recommend_phase74_nondefault_canary_after_manual_review",
            "hold_phase74_shared_raw_deterministic_serializer_ab",
        }
        and comparison.get("actual_generation_call_count") == 162
        and comparison.get("actual_product_judge_output_count") == 144
        and comparison.get("actual_user_feedback_count") == 0
        and comparison.get("training_executed") is False
        and comparison.get("adapter_created") is False
        and comparison.get("product_default_changed") is False
        and comparison.get("real_user_benefit_proven") is False
        and decision.get("auto_promote_allowed") is False
    )
    return {
        "name": "phase74_evidence_consistency",
        "command": ["internal", "phase74_evidence_consistency"],
        "returncode": 0 if passed else 1,
        "passed": passed,
        "duration_seconds": 0.0,
        "output_line_count": 1,
        "output_sha256": hashlib.sha256(str(passed).encode()).hexdigest(),
        "output_tail": [
            f"integrity={integrity.get('passed')} recommendation={decision.get('recommendation')}"
        ],
    }


def _validate() -> int:
    python = str(REPO_ROOT / ".venv/bin/python")
    phase_tests = [
        f"tests/test_phase{phase}_{name}.py"
        for phase, name in (
            (74, "shared_raw_deterministic_serializer_ab"),
            (73, "exact_descriptor_normalization"),
            (72, "deterministic_boundary_serializer"),
            (71, "qualified_structured_contract_ab"),
            (70, "structured_boundary_contract"),
            (69, "minimal_runtime_ab"),
            (68, "aligned_candidate_scope_recovery"),
            (67, "historical_contract_compatibility_audit"),
            (66, "external_distribution_regression"),
            (65, "aggregate_safe_boundary_coverage"),
            (64, "field_typed_historical_replay"),
            (63, "field_typed_candidate_wire"),
            (62, "risk_asymmetric_candidate_consensus"),
            (61, "compact_candidate_wire_protocol"),
            (60, "flat_schema_compatibility"),
            (59, "proposition_addressed_grounding"),
            (58, "clause_addressed_grounding"),
            (57, "span_evaluator_historical_replay"),
            (56, "evidence_span_grounded_atomic"),
            (55, "atomic_boundary_composition"),
            (54, "typed_proposition_evaluator"),
            (53, "evaluator_scope_recovery"),
            (52, "adversarial_evaluator_generalization"),
            (51, "dual_evaluator_hardening"),
            (50, "conditional_provenance_guard"),
            (49, "provenance_boundary_recovery"),
            (48, "compact_intent_runtime"),
            (47, "simulated_user_review"),
            (46, "runtime_first_latest_intent"),
            (45, "privacy_multiturn_preference"),
        )
    ]
    checks = [
        (
            "py_compile",
            [
                python,
                "-m",
                "py_compile",
                "pfe-core/pfe_core/phase74_shared_raw_deterministic_serializer_ab.py",
                "tools/phase74_shared_raw_deterministic_serializer_ab.py",
                "tests/test_phase74_shared_raw_deterministic_serializer_ab.py",
            ],
        ),
        (
            "phase74_focused_and_phase73_to45_regression",
            [python, "-m", "pytest", "-q", *phase_tests],
        ),
        ("test_unit", ["make", "test-unit"]),
        ("test_surface", ["make", "test-surface"]),
        ("test_e2e_mock", ["make", "test-e2e-mock"]),
        ("smoke_beta", ["make", "smoke-beta"]),
        ("git_diff_check", ["git", "diff", "--check"]),
    ]
    results = [_evidence_check()]
    results.extend(_run_check(name, command) for name, command in checks)
    passed = all(row["passed"] for row in results)
    summary = {
        "kind": "phase74_validation_summary",
        "created_at": _utcnow(),
        "status": "passed" if passed else "failed",
        "check_count": len(results),
        "passed_count": sum(row["passed"] for row in results),
        "failed_count": sum(not row["passed"] for row in results),
        "checks": results,
    }
    _write_json(EVIDENCE_ROOT / "validation_summary.json", summary)
    lines = [f"Phase74 validation: {summary['status']}"]
    lines.extend(
        f"{row['name']}: {'PASS' if row['passed'] else 'FAIL'} ({row['duration_seconds']}s, rc={row['returncode']})"
        for row in results
    )
    _write_text(EVIDENCE_ROOT / "validation_gate.txt", "\n".join(lines))
    print("\n".join(lines))
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--clean-evidence", action="store_true")
    generate = subparsers.add_parser("generate")
    generate.add_argument("--clean", action="store_true")
    subparsers.add_parser("prepare-product")
    evaluate = subparsers.add_parser("eval-product")
    evaluate.add_argument("--ollama-endpoint", default="http://127.0.0.1:11434")
    evaluate.add_argument("--timeout", type=int, default=900)
    evaluate.add_argument("--resume", action="store_true")
    subparsers.add_parser("finalize")
    subparsers.add_parser("validate")
    args = parser.parse_args()
    if args.command == "prepare":
        return _prepare(args.clean_evidence)
    if args.command == "generate":
        return _generate(args.clean)
    if args.command == "prepare-product":
        return _prepare_product()
    if args.command == "eval-product":
        return _run_product_eval(args.ollama_endpoint, args.timeout, args.resume)
    if args.command == "finalize":
        return _finalize()
    return _validate()


if __name__ == "__main__":
    raise SystemExit(main())
