#!/usr/bin/env python3
"""Run the frozen Phase71 qualification and structured-contract A/B loop."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
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
import phase70_generate as phase70_generate
import phase70_prepare_product_eval as phase70_product
from phase70_finalize_evidence import _examples as phase70_examples
from phase70_prepare import _blind_cases, _phase68_regression_cases
from pfe_core.phase59_proposition_addressed_grounding import (
    build_phase59_proposition_candidates,
    phase59_ollama_json_schema,
)
from pfe_core.phase63_field_typed_candidate_wire import (
    PHASE63_FIELD_PREFIXES,
    PHASE63_WIRE_PATTERN,
    PHASE63_WIRE_VERSION,
)
from pfe_core.phase70_structured_boundary_contract import (
    PHASE70_ACCEPT_RATE_DELTA_GATE,
    PHASE70_ACCEPT_RATE_GATE,
    PHASE70_EXACT_STRUCTURE_GATE,
    stable_hash,
)
from pfe_core.phase71_qualified_structured_contract_ab import (
    PHASE71_VARIANTS,
    audit_phase71_fixture_contract,
    build_phase71_decision,
    build_phase71_holdout,
    build_phase71_sparse_preflight_cases,
)


EVIDENCE_ROOT = REPO_ROOT / "docs/demo/phase71-qualified-structured-contract-ab"
PHASE70_ROOT = REPO_ROOT / "docs/demo/phase70-structured-boundary-contract"
MODEL_PATH = REPO_ROOT / "models/Qwen3-4B"
JUDGE_ALIASES = ("semantic_judge_alpha", "semantic_judge_beta")
JUDGE_MODELS = {
    "semantic_judge_alpha": "gemma4:31b",
    "semantic_judge_beta": "qwen3.6:latest",
}
SOURCE_FILES = {
    "phase46_generator_helpers": REPO_ROOT / "tools/phase46_qwen3_4b_generate.py",
    "phase53_hard_detector": CORE_ROOT / "pfe_core/phase53_evaluator_scope_recovery.py",
    "phase56_grounder_composer": CORE_ROOT / "pfe_core/phase56_evidence_span_grounded_atomic.py",
    "phase59_candidates": CORE_ROOT / "pfe_core/phase59_proposition_addressed_grounding.py",
    "phase63_typed_wire": CORE_ROOT / "pfe_core/phase63_field_typed_candidate_wire.py",
    "phase62_consensus": CORE_ROOT / "pfe_core/phase62_risk_asymmetric_candidate_consensus.py",
    "phase70_core": CORE_ROOT / "pfe_core/phase70_structured_boundary_contract.py",
    "phase70_generate": REPO_ROOT / "tools/phase70_generate.py",
    "phase70_prepare_product_eval": REPO_ROOT / "tools/phase70_prepare_product_eval.py",
    "phase70_execute_eval": REPO_ROOT / "tools/phase70_execute_eval.py",
    "phase70_finalize": REPO_ROOT / "tools/phase70_finalize_evidence.py",
    "phase71_core": CORE_ROOT / "pfe_core/phase71_qualified_structured_contract_ab.py",
    "phase71_driver": Path(__file__).resolve(),
}
SOURCE_PATHS = {
    name: str(path.relative_to(REPO_ROOT)) for name, path in SOURCE_FILES.items()
}
DYNAMIC = {
    "evidence_manifest.json",
    "finalization_state.json",
    "validation_gate.txt",
    "validation_summary.json",
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
    files = list(manifest.get("files") or [])
    passed = bool(files) and all(
        (REPO_ROOT / str(row.get("path") or "")).is_file()
        and _sha256(REPO_ROOT / str(row.get("path") or "")) == row.get("sha256")
        for row in files
    )
    return passed, str(manifest.get("manifest_sha256") or "")


def _phase70_snapshot() -> dict[str, Any]:
    decision = _read_json(PHASE70_ROOT / "phase70-final-decision.json")
    integrity = _read_json(PHASE70_ROOT / "evidence_integrity.json")
    finalization = _read_json(PHASE70_ROOT / "finalization_state.json")
    validation = _read_json(PHASE70_ROOT / "validation_summary.json")
    manifest_ok, manifest_sha = _verify_manifest(PHASE70_ROOT)
    checks = {
        "phase70_held": decision.get("recommendation")
        == "hold_phase70_structured_boundary_contract",
        "phase70_blocked_before_product_ab": decision.get("experiment_status")
        == "blocked_before_product_ab",
        "phase70_transport_envelope_qualified": decision.get(
            "transport_envelope_qualified"
        )
        is True,
        "phase70_evidence_integrity_passed": integrity.get("passed") is True,
        "phase70_finalization_blocked": finalization.get("status") == "blocked",
        "phase70_validation_passed": validation.get("status") == "passed",
        "phase70_manifest_verified": manifest_ok,
        "phase70_no_training_or_default_change": decision.get("training_allowed") is False
        and decision.get("product_default_change_allowed") is False,
    }
    return {
        "kind": "phase71_phase70_hold_snapshot",
        "passed": all(checks.values()),
        "checks": checks,
        "manifest_sha256": manifest_sha,
        "recommendation": decision.get("recommendation"),
    }


def _task_hash(row: Mapping[str, Any]) -> str:
    return stable_hash(
        [
            row.get(key)
            for key in (
                "user_goal",
                "user_correction",
                "continuation_request",
                "acceptance_request",
            )
        ]
    )


def _prepare(clean: bool) -> int:
    if clean and EVIDENCE_ROOT.exists():
        shutil.rmtree(EVIDENCE_ROOT)
    missing = [name for name, path in SOURCE_FILES.items() if not path.is_file()]
    snapshot = _phase70_snapshot()
    holdout = build_phase71_holdout()
    sparse = build_phase71_sparse_preflight_cases()
    fixture_audit = audit_phase71_fixture_contract(sparse["cases"])
    regression = _phase68_regression_cases()
    sparse_public, sparse_hidden = _blind_cases(sparse["cases"], "phase71-sparse")
    regression_public, regression_hidden = _blind_cases(
        regression, "phase71-regression"
    )
    previous = _read_json(PHASE70_ROOT / "evidence-holdout/holdout.json")
    current_hashes = {_task_hash(row) for row in holdout["sessions"]}
    previous_hashes = {_task_hash(row) for row in previous.get("sessions") or []}
    overlap = {
        "kind": "phase71_holdout_overlap_audit",
        "passed": len(current_hashes) == 48 and not (current_hashes & previous_hashes),
        "phase71_unique_task_count": len(current_hashes),
        "phase70_exact_overlap_count": len(current_hashes & previous_hashes),
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
        build_phase59_proposition_candidates(
            "\n".join(
                [
                    "当前模拟材料不得登记为 actual_user_feedback。",
                    "真实用户结果尚未确认。",
                    "当前测试证据不能证明真实用户结果。",
                ]
            )
        )
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
        "kind": "phase71_frozen_qualified_structured_contract_protocol",
        "variants": list(PHASE71_VARIANTS),
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
        "fixture_contract_audit_required_before_calls": True,
        "transport_and_full_composer_qualification_are_distinct": True,
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
        "kind": "phase71_pre_model_call_freeze",
        "phase70_snapshot_sha256": stable_hash(snapshot),
        "holdout_sha256": stable_hash(holdout),
        "fixture_audit_sha256": stable_hash(fixture_audit),
        "sparse_public_sha256": stable_hash(sparse_public),
        "sparse_hidden_sha256": stable_hash(sparse_hidden),
        "regression_public_sha256": stable_hash(regression_public),
        "regression_hidden_sha256": stable_hash(regression_hidden),
        "protocol_sha256": protocol["protocol_sha256"],
        "source_sha256": source_hashes,
        "frozen_before_any_phase71_model_call": True,
        "created_at": _utcnow(),
    }
    checks = {
        "phase70_snapshot_passed": snapshot["passed"],
        "holdout_overlap_audit_passed": overlap["passed"],
        "holdout_counts_exact": holdout["session_count"] == 48
        and holdout["boundary_session_count"] == 36
        and holdout["ordinary_session_count"] == 12,
        "fixture_contract_audit_passed_before_calls": fixture_audit["passed"],
        "sparse_preflight_count_exact": len(sparse_public) == 12,
        "phase68_regression_balanced_count_exact": len(regression_public) == 30,
        "local_model_present": MODEL_PATH.is_dir(),
        "source_files_complete": not missing,
    }
    decision = {
        "kind": "phase71_preparation_decision",
        "status": "ready_for_sparse_transport_preflight"
        if all(checks.values())
        else "blocked",
        "checks": checks,
        "failed_checks": [key for key, value in checks.items() if not value],
        "created_at": _utcnow(),
    }
    _write_json(EVIDENCE_ROOT / "evidence-baseline/phase70_hold_snapshot.json", snapshot)
    _write_json(EVIDENCE_ROOT / "evidence-holdout/holdout.json", holdout)
    _write_json(EVIDENCE_ROOT / "evidence-holdout/overlap_audit.json", overlap)
    _write_json(EVIDENCE_ROOT / "fixture_contract_audit.json", fixture_audit)
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
            "kind": "phase71_source_manifest",
            "source_sha256": source_hashes,
            "actual_user_feedback_count": 0,
            "not_for_training": True,
        },
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return 0 if decision["status"] == "ready_for_sparse_transport_preflight" else 1


def _run_phase70_eval(stage: str, endpoint: str, timeout: int, resume: bool) -> int:
    previous_root = phase70_eval.EVIDENCE_ROOT
    previous_paths = phase70_eval.SOURCE_PATHS
    previous_argv = sys.argv
    phase70_eval.EVIDENCE_ROOT = EVIDENCE_ROOT
    phase70_eval.SOURCE_PATHS = dict(SOURCE_PATHS)
    sys.argv = [
        "phase70_execute_eval.py",
        "--stage",
        stage,
        "--ollama-endpoint",
        endpoint,
        "--timeout",
        str(timeout),
    ] + (["--resume"] if resume else [])
    try:
        return phase70_eval.main()
    finally:
        phase70_eval.EVIDENCE_ROOT = previous_root
        phase70_eval.SOURCE_PATHS = previous_paths
        sys.argv = previous_argv


def _run_phase70_generate(variant: str, clean: bool) -> int:
    previous_root = phase70_generate.EVIDENCE_ROOT
    previous_paths = phase70_generate.SOURCE_PATHS
    previous_argv = sys.argv
    phase70_generate.EVIDENCE_ROOT = EVIDENCE_ROOT
    phase70_generate.SOURCE_PATHS = dict(SOURCE_PATHS)
    sys.argv = ["phase70_generate.py", "--variant", variant] + (
        ["--clean"] if clean else []
    )
    try:
        return phase70_generate.main()
    finally:
        phase70_generate.EVIDENCE_ROOT = previous_root
        phase70_generate.SOURCE_PATHS = previous_paths
        sys.argv = previous_argv


def _prepare_product() -> int:
    previous_root = phase70_product.EVIDENCE_ROOT
    phase70_product.EVIDENCE_ROOT = EVIDENCE_ROOT
    try:
        return phase70_product.main()
    finally:
        phase70_product.EVIDENCE_ROOT = previous_root


def _manifest() -> dict[str, Any]:
    files = []
    for path in sorted(EVIDENCE_ROOT.rglob("*")):
        if path.is_file() and path.name not in DYNAMIC:
            files.append(
                {
                    "path": str(path.relative_to(REPO_ROOT)),
                    "sha256": _sha256(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    return {
        "kind": "phase71_evidence_manifest",
        "file_count": len(files),
        "files": files,
        "manifest_sha256": hashlib.sha256(
            json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def _post_call_packaging_audit() -> dict[str, Any]:
    frozen = dict(
        _read_json(EVIDENCE_ROOT / "pre_model_call_freeze.json").get("source_sha256")
        or {}
    )
    current = {name: _sha256(path) for name, path in SOURCE_FILES.items()}
    changed = sorted(name for name, value in current.items() if frozen.get(name) != value)
    audit = {
        "kind": "phase71_post_call_packaging_change_audit",
        "passed": changed == ["phase71_driver"],
        "changed_source_names": changed,
        "allowed_changed_source_names": ["phase71_driver"],
        "change_scope": "blocked product-eval evidence packaging only",
        "external_evaluator_core_contract_generator_or_transport_changed_after_calls": any(
            name != "phase71_driver" for name in changed
        ),
        "frozen_source_sha256": frozen,
        "current_source_sha256": current,
        "created_at": _utcnow(),
    }
    _write_json(EVIDENCE_ROOT / "post_call_packaging_change_audit.json", audit)
    return audit


def _qualification_blocked(
    snapshot: Mapping[str, Any], sparse: Mapping[str, Any], regression: Mapping[str, Any]
) -> int:
    failed_stage = (
        "sparse_preflight"
        if sparse.get("status") != "qualified"
        else "phase68_regression"
    )
    reports = {
        "sparse_preflight": sparse,
        "phase68_regression": regression,
    }
    expected = {"sparse_preflight": 24, "phase68_regression": 60}
    counts = {
        stage: int(report.get("successful_model_output_count") or 0)
        for stage, report in reports.items()
    }
    prior_ok = counts["sparse_preflight"] == 24 and sparse.get("failure_count") == 0
    failed_complete = (
        counts[failed_stage] == expected[failed_stage]
        and reports[failed_stage].get("failure_count") == 0
    )
    downstream_absent = not (
        EVIDENCE_ROOT / "evidence-real-generation"
    ).exists() and not (EVIDENCE_ROOT / "evidence-product-eval/evaluator_report.json").exists()
    integrity = {
        "kind": "phase71_evidence_integrity",
        "passed": snapshot.get("passed") is True
        and prior_ok
        and failed_complete
        and downstream_absent,
        "experiment_succeeded": False,
        "blocked_evidence_complete": True,
        "failed_stage": failed_stage,
    }
    decision = {
        "kind": "phase71_final_decision",
        "status": "hold_phase71_qualified_structured_contract_ab",
        "recommendation": "hold_phase71_qualified_structured_contract_ab",
        "experiment_status": f"blocked_at_{failed_stage}",
        "failed_checks": [f"{failed_stage}_qualified"],
        "phase72_nondefault_api_canary_design_eligible": False,
        "product_default_change_allowed": False,
        "training_allowed": False,
        "adapter_created": False,
        "hermes_attachment_allowed": False,
        "auto_promote_allowed": False,
    }
    comparison = {
        "kind": "phase71_structured_boundary_comparison",
        "experiment_status": decision["experiment_status"],
        "qualification": {
            stage: {
                "status": report.get("status"),
                "accuracy": report.get("accuracy"),
                "successful_model_output_count": counts[stage],
            }
            for stage, report in reports.items()
        },
        "actual_judge_output_counts": {
            "sparse_preflight": counts["sparse_preflight"],
            "phase68_regression": counts["phase68_regression"],
            "product": 0,
        },
        "actual_generation_call_count": 0,
        "actual_model_output_count_total": sum(counts.values()),
        "actual_user_feedback_count": 0,
        "training_executed": False,
        "adapter_created": False,
        "product_default_changed": False,
        "recommendation": decision["recommendation"],
    }
    _write_json(EVIDENCE_ROOT / "comparison_summary.json", comparison)
    _write_json(EVIDENCE_ROOT / "phase71-final-decision.json", decision)
    _write_json(EVIDENCE_ROOT / "evidence_integrity.json", integrity)
    _write_json(
        EVIDENCE_ROOT / "evidence-no-training/training_attempt.json",
        {
            "kind": "phase71_training_attempt",
            "status": "not_run_by_design",
            "adapter_created": False,
        },
    )
    _write_text(
        EVIDENCE_ROOT / "phase71-final-decision.md",
        f"# Phase71 Final Decision\n\nQualification stopped at `{failed_stage}`. No product generation, training, adapter, Hermes attachment, default change, or promotion was run.",
    )
    _write_text(
        EVIDENCE_ROOT / "output_examples.md",
        "# Phase71 Output Examples\n\nProduct A/B was not run because evaluator qualification did not pass.",
    )
    _write_text(
        EVIDENCE_ROOT / "next-pursuit-goal.md",
        "# Next Pursuit Goal\n\nReview the preserved qualification failure and design one newly frozen evaluator repair. Do not integrate or train while held.",
    )
    return _finish_manifest(integrity, decision)


def _finish_manifest(
    integrity: Mapping[str, Any], decision: Mapping[str, Any]
) -> int:
    manifest = _manifest()
    _write_json(EVIDENCE_ROOT / "evidence_manifest.json", manifest)
    state = {
        "kind": "phase71_finalization_state",
        "status": "completed" if integrity.get("experiment_succeeded") else "blocked",
        "recommendation": decision.get("recommendation"),
        "evidence_integrity_passed": integrity.get("passed") is True,
        "experiment_succeeded": integrity.get("experiment_succeeded") is True,
        "manifest_file_count": manifest["file_count"],
        "created_at": _utcnow(),
    }
    _write_json(EVIDENCE_ROOT / "finalization_state.json", state)
    print(json.dumps(state, ensure_ascii=False, indent=2))
    return 0 if integrity.get("passed") is True else 1


def _finalize() -> int:
    snapshot = _read_json(EVIDENCE_ROOT / "evidence-baseline/phase70_hold_snapshot.json")
    sparse = _read_json(EVIDENCE_ROOT / "evidence-sparse-preflight/evaluator_report.json")
    regression = _read_json(EVIDENCE_ROOT / "evidence-phase68-regression/evaluator_report.json")
    if sparse.get("status") != "qualified" or regression.get("status") != "qualified":
        return _qualification_blocked(snapshot, sparse, regression)
    boundary = _read_json(EVIDENCE_ROOT / "evidence-product-eval/evaluator_report.json")
    parity = _read_json(EVIDENCE_ROOT / "ab_parity_audit.json")
    ordinary = _read_json(EVIDENCE_ROOT / "ordinary_control_report.json")
    holdout = _read_json(EVIDENCE_ROOT / "evidence-holdout/holdout.json")
    sessions = [dict(row) for row in holdout.get("sessions") or []]
    metrics = {
        variant: _read_json(
            EVIDENCE_ROOT / f"evidence-real-generation/metrics_{variant}.json"
        )
        for variant in PHASE71_VARIANTS
    }
    transcripts = {
        variant: _read_jsonl(
            EVIDENCE_ROOT / f"evidence-real-generation/transcripts_{variant}.jsonl"
        )
        for variant in PHASE71_VARIANTS
    }
    freezes_passed = all(
        _read_json(
            EVIDENCE_ROOT / f"evidence-real-generation/freeze_check_{variant}.json"
        ).get("passed")
        is True
        for variant in PHASE71_VARIANTS
    ) and all(
        _read_json(EVIDENCE_ROOT / directory / "freeze_check.json").get("passed")
        is True
        for directory in (
            "evidence-sparse-preflight",
            "evidence-phase68-regression",
            "evidence-product-eval",
        )
    )
    decision = build_phase71_decision(
        phase69_snapshot=snapshot,
        transport_preflight=sparse,
        phase68_regression=regression,
        parity=parity,
        boundary=boundary,
        ordinary=ordinary,
        freezes_passed=freezes_passed,
    )
    product_complete = (
        boundary.get("successful_model_output_count") == 144
        and boundary.get("failure_count") == 0
    )
    if not product_complete:
        decision = {
            **decision,
            "experiment_status": "blocked_at_product_eval",
            "product_eval_failure": {
                "successful_model_output_count": boundary.get(
                    "successful_model_output_count"
                ),
                "expected_model_output_count": boundary.get(
                    "expected_model_output_count"
                ),
                "failure_count": boundary.get("failure_count"),
                "failures": boundary.get("failures"),
                "raw_failure_attempt_count": boundary.get("raw_failure_attempt_count"),
            },
        }
    actual_generations = sum(
        int(row.get("actual_generation_call_count") or 0) for row in metrics.values()
    )
    judge_counts = {
        "sparse_preflight": sparse.get("successful_model_output_count"),
        "phase68_regression": regression.get("successful_model_output_count"),
        "product": boundary.get("successful_model_output_count"),
    }
    comparison = {
        "kind": "phase71_structured_boundary_comparison",
        "experiment_status": (
            "completed" if product_complete else "blocked_at_product_eval"
        ),
        "model": metrics["natural_boundary_contract"].get("model_id"),
        "device": metrics["natural_boundary_contract"].get("device"),
        "only_ab_variable": "freeform_vs_fixed_three_line_boundary_contract",
        "qualification": {
            "sparse_preflight": {
                "status": sparse.get("status"),
                "accuracy": sparse.get("accuracy"),
                "candidate_selection_exact_match_rate": sparse.get(
                    "candidate_selection_exact_match_rate"
                ),
            },
            "phase68_regression": {
                "status": regression.get("status"),
                "accuracy": regression.get("accuracy"),
                "typed_exact_match_rate": regression.get("typed_exact_match_rate"),
            },
        },
        "boundary": boundary.get("variants"),
        "candidate_accept_rate_delta": boundary.get("candidate_accept_rate_delta"),
        "ordinary_controls": ordinary.get("variants"),
        "generation_metrics": metrics,
        "actual_generation_call_count": actual_generations,
        "actual_judge_output_counts": judge_counts,
        "actual_model_output_count_total": actual_generations
        + sum(int(value or 0) for value in judge_counts.values()),
        "recommendation": decision["recommendation"],
        "actual_user_feedback_count": 0,
        "training_executed": False,
        "adapter_created": False,
        "hermes_attached": False,
        "product_default_changed": False,
    }
    integrity_checks = {
        "phase70_hold_snapshot_passed": snapshot.get("passed") is True,
        "all_freezes_passed": freezes_passed,
        "sparse_preflight_qualified": sparse.get("status") == "qualified"
        and sparse.get("successful_model_output_count") == 24,
        "phase68_regression_qualified": regression.get("status") == "qualified"
        and regression.get("successful_model_output_count") == 60,
        "product_eval_complete": product_complete,
        "both_generation_arms_complete": all(
            row.get("completed_count") == 48 and row.get("failed_count") == 0
            for row in metrics.values()
        ),
        "all_288_generations_real": actual_generations == 288,
        "no_training_adapter_hermes_or_default_change": decision.get(
            "training_allowed"
        )
        is False
        and decision.get("adapter_created") is False
        and decision.get("hermes_attachment_allowed") is False
        and decision.get("product_default_change_allowed") is False,
        "actual_user_feedback_zero": True,
    }
    product_failure_preserved = (
        not product_complete
        and boundary.get("successful_model_output_count") == 143
        and boundary.get("failure_count") == 1
        and boundary.get("raw_failure_attempt_count") == 2
        and boundary.get("raw_failures_preserved") is True
    )
    packaging_audit = _post_call_packaging_audit() if not product_complete else {}
    if not product_complete:
        integrity_checks["product_eval_failure_preserved"] = product_failure_preserved
        integrity_checks["post_call_change_limited_to_evidence_packaging"] = (
            packaging_audit.get("passed") is True
        )
    experiment_succeeded = all(integrity_checks.values()) and product_complete
    blocked_evidence_complete = (
        not product_complete
        and all(
            value
            for key, value in integrity_checks.items()
            if key != "product_eval_complete"
        )
    )
    integrity = {
        "kind": "phase71_evidence_integrity",
        "passed": experiment_succeeded or blocked_evidence_complete,
        "experiment_succeeded": experiment_succeeded,
        "blocked_evidence_complete": blocked_evidence_complete,
        "failed_stage": None if product_complete else "product_eval",
        "checks": integrity_checks,
        "created_at": _utcnow(),
    }
    _write_json(EVIDENCE_ROOT / "comparison_summary.json", comparison)
    _write_json(EVIDENCE_ROOT / "phase71-final-decision.json", decision)
    _write_json(EVIDENCE_ROOT / "evidence_integrity.json", integrity)
    _write_json(
        EVIDENCE_ROOT / "evidence-no-training/training_attempt.json",
        {
            "kind": "phase71_training_attempt",
            "status": "not_run_by_design",
            "reason": "Phase71 isolates runtime response structure after evaluator requalification.",
            "adapter_created": False,
        },
    )
    examples = "\n".join(
        line.rstrip()
        for line in phase70_examples(sessions, transcripts)
        .replace("Phase70", "Phase71", 1)
        .splitlines()
    )
    _write_text(EVIDENCE_ROOT / "output_examples.md", examples)
    _write_text(
        EVIDENCE_ROOT / "phase71-final-decision.md",
        f"""# Phase71 Final Decision

## 结论

最终 recommendation 为 **{decision['recommendation']}**。自然语言契约 accept rate `{decision.get('baseline_accept_rate')}`，固定三行契约 `{decision.get('candidate_accept_rate')}`，增量 `{decision.get('candidate_accept_rate_delta')}`。

## 证据

- 调用前 fixture/hard-detector contract audit 通过。
- sparse preflight：{sparse.get('status')}，accuracy `{sparse.get('accuracy')}`，24 个真实 judge 输出。
- Phase68 对齐回归：{regression.get('status')}，accuracy `{regression.get('accuracy')}`，60 个真实 judge 输出。
- 产品盲评：{boundary.get('successful_model_output_count')}/144 个真实 judge 输出；失败尝试已原样保存。
- Qwen3-4B A/B 生成：288 次，未加载 adapter。

## 边界

这是 simulated_usage runtime A/B，不是实际用户反馈或训练收益。没有训练、没有 adapter、没有接 Hermes、没有更改产品默认、没有自动 promote。
""",
    )
    _write_text(
        EVIDENCE_ROOT / "phase71-runbook.md",
        """# Phase71 Runbook

```bash
.venv/bin/python tools/phase71_qualified_structured_contract_ab.py prepare --clean-evidence
.venv/bin/python tools/phase71_qualified_structured_contract_ab.py eval --stage sparse_preflight --ollama-endpoint http://127.0.0.1:11435 --timeout 900
.venv/bin/python tools/phase71_qualified_structured_contract_ab.py eval --stage phase68_regression --ollama-endpoint http://127.0.0.1:11435 --timeout 900
.venv/bin/python tools/phase71_qualified_structured_contract_ab.py generate --variant natural_boundary_contract --clean
.venv/bin/python tools/phase71_qualified_structured_contract_ab.py generate --variant structured_boundary_contract --clean
.venv/bin/python tools/phase71_qualified_structured_contract_ab.py prepare-product
.venv/bin/python tools/phase71_qualified_structured_contract_ab.py eval --stage product --ollama-endpoint http://127.0.0.1:11435 --timeout 900
.venv/bin/python tools/phase71_qualified_structured_contract_ab.py finalize
.venv/bin/python tools/phase71_qualified_structured_contract_ab.py validate
```

Do not edit fixtures, contracts, transports, decoding, or gates after prepare.
""",
    )
    next_goal = (
        "Build Phase72 as a non-default API canary for the structured boundary contract. Add an explicit opt-in runtime mode, verify stream/non-stream parity, ordinary-task routing, fallback and rollback, and run a fresh canary holdout. Do not change the default or train in the same phase."
        if decision.get("phase72_nondefault_api_canary_design_eligible") is True
        else
        "Build Phase72 as a deterministic non-default boundary serializer experiment. Canonicalize only boundary-routed outputs to the exact three lines, strip Markdown line-break whitespace and extra text, preserve ordinary-task isolation, and use a fresh holdout plus complete dual-judge evidence. Do not change the default or train in the same phase."
    )
    _write_text(EVIDENCE_ROOT / "next-pursuit-goal.md", f"# Next Pursuit Goal\n\n{next_goal}")
    return _finish_manifest(integrity, decision)


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
    integrity = _read_json(EVIDENCE_ROOT / "evidence_integrity.json")
    decision = _read_json(EVIDENCE_ROOT / "phase71-final-decision.json")
    comparison = _read_json(EVIDENCE_ROOT / "comparison_summary.json")
    full = (
        integrity.get("passed") is True
        and integrity.get("experiment_succeeded") is True
        and decision.get("recommendation")
        in {
            "recommend_phase71_structured_contract_for_nondefault_canary_manual_review_only",
            "hold_phase71_qualified_structured_contract_ab",
        }
        and comparison.get("actual_generation_call_count") == 288
        and comparison.get("actual_judge_output_counts")
        == {"sparse_preflight": 24, "phase68_regression": 60, "product": 144}
        and comparison.get("actual_model_output_count_total") == 516
    )
    blocked = (
        integrity.get("passed") is True
        and integrity.get("experiment_succeeded") is False
        and integrity.get("blocked_evidence_complete") is True
        and decision.get("recommendation") == "hold_phase71_qualified_structured_contract_ab"
        and (
            comparison.get("actual_generation_call_count") == 0
            or (
                comparison.get("experiment_status") == "blocked_at_product_eval"
                and comparison.get("actual_generation_call_count") == 288
                and comparison.get("actual_judge_output_counts")
                == {
                    "sparse_preflight": 24,
                    "phase68_regression": 60,
                    "product": 143,
                }
                and comparison.get("actual_model_output_count_total") == 515
            )
        )
    )
    passed = (full or blocked) and comparison.get("actual_user_feedback_count") == 0
    passed = passed and comparison.get("training_executed") is False
    passed = passed and comparison.get("adapter_created") is False
    passed = passed and comparison.get("product_default_changed") is False
    passed = passed and decision.get("auto_promote_allowed") is False
    return {
        "name": "phase71_evidence_consistency",
        "command": ["internal", "phase71_evidence_consistency"],
        "returncode": 0 if passed else 1,
        "passed": passed,
        "duration_seconds": 0.0,
        "output_line_count": 1,
        "output_sha256": hashlib.sha256(str(passed).encode()).hexdigest(),
        "output_tail": [
            f"integrity={integrity.get('passed')} recommendation={decision.get('recommendation')} outputs={comparison.get('actual_model_output_count_total')}"
        ],
    }


def _validate() -> int:
    python = str(REPO_ROOT / ".venv/bin/python")
    phase_tests = [
        f"tests/test_phase{phase}_{name}.py"
        for phase, name in (
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
                "pfe-core/pfe_core/phase71_qualified_structured_contract_ab.py",
                "tools/phase71_qualified_structured_contract_ab.py",
                "tests/test_phase71_qualified_structured_contract_ab.py",
            ],
        ),
        (
            "phase71_focused_and_phase70_to45_regression",
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
        "kind": "phase71_validation_summary",
        "created_at": _utcnow(),
        "status": "passed" if passed else "failed",
        "check_count": len(results),
        "passed_count": sum(row["passed"] for row in results),
        "failed_count": sum(not row["passed"] for row in results),
        "checks": results,
    }
    _write_json(EVIDENCE_ROOT / "validation_summary.json", summary)
    lines = [f"Phase71 validation: {summary['status']}"]
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
    evaluate = subparsers.add_parser("eval")
    evaluate.add_argument(
        "--stage", choices=("sparse_preflight", "phase68_regression", "product"), required=True
    )
    evaluate.add_argument("--ollama-endpoint", default="http://127.0.0.1:11434")
    evaluate.add_argument("--timeout", type=int, default=900)
    evaluate.add_argument("--resume", action="store_true")
    generate = subparsers.add_parser("generate")
    generate.add_argument("--variant", choices=PHASE71_VARIANTS, required=True)
    generate.add_argument("--clean", action="store_true")
    subparsers.add_parser("prepare-product")
    subparsers.add_parser("finalize")
    subparsers.add_parser("validate")
    args = parser.parse_args()
    if args.command == "prepare":
        return _prepare(args.clean_evidence)
    if args.command == "eval":
        return _run_phase70_eval(
            args.stage, args.ollama_endpoint, args.timeout, args.resume
        )
    if args.command == "generate":
        return _run_phase70_generate(args.variant, args.clean)
    if args.command == "prepare-product":
        return _prepare_product()
    if args.command == "finalize":
        return _finalize()
    return _validate()


if __name__ == "__main__":
    raise SystemExit(main())
