#!/usr/bin/env python3
"""Run the Phase90 prompt-aligned curriculum ablation on local Qwen2.5-1.5B."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time
import traceback
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = REPO_ROOT / "pfe-core"
TOOLS_ROOT = REPO_ROOT / "tools"
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from pfe_core.adapter_store.quality import validate_adapter_artifact
from pfe_core.phase75_personalization_benefit_benchmark import stable_hash
from pfe_core.phase78_persona_internalization_training import build_phase78_sft_job_spec
from pfe_core.phase87_failure_driven_training import (
    PHASE87_CATEGORIES,
    aggregate_phase89_scores,
    build_phase89_holdout,
    score_phase89_output,
)
from pfe_core.phase90_native_format_curriculum import (
    PHASE90_CURRICULA,
    PHASE90_HOLDOUT_COUNT,
    audit_phase90_curriculum_candidates,
    audit_phase90_holdout_isolation,
    build_phase90_curriculum_candidates,
    build_phase90_decision,
    build_phase90_holdout,
    build_phase90_training_plan,
    select_phase90_training_samples,
    summarize_phase90_training_rows,
)
from pfe_core.trainer.executors import _run_real_local_peft_training
from phase87_89_failure_driven_adapter_loop import (
    GENERATION_PROTOCOL,
    MODEL_PATH,
    MODEL_REVISION,
    TRAINER_OUTPUT_ROOT,
    TRAINING_LEARNING_RATE,
    _completion_boundary_report,
    _load_runtime,
    _model_complete,
    _read_json,
    _read_jsonl,
    _release_runtime,
    _run_eval_session,
    _safe_clean,
    _sha256,
    _write_json,
    _write_jsonl,
    _write_private_jsonl,
)


EVIDENCE_ROOT = REPO_ROOT / "docs/demo/phase90-native-format-curriculum-repair"
PREPARATION_ROOT = EVIDENCE_ROOT / "evidence-preparation"
TRAINING_ROOT = EVIDENCE_ROOT / "evidence-real-training"
EVAL_ROOT = EVIDENCE_ROOT / "evidence-blind-eval"
FAILURE_ROOT = EVIDENCE_ROOT / "evidence-failures"
REVIEW_CACHE_ROOT = Path("/private/tmp/pfe-phase90-simulated-user-review")
PHASE89_ROOT = REPO_ROOT / "docs/demo/phase87-89-failure-driven-adapter-loop"
PHASE89_ADAPTER_ROOT = TRAINER_OUTPUT_ROOT / "phase87-failure-driven-25step/peft_lora"
PHASE90_TRAINING_MAX_LENGTH = 288
FROZEN_THRESHOLDS = {
    "candidate_raw_gain_minimum": 0.08,
    "candidate_target_category_floor_minimum": 0.75,
    "candidate_raw_native_minimum": 0.75,
    "candidate_runtime_fallback_maximum": 0.10,
    "candidate_false_block_maximum": 0.0,
    "candidate_truncation_maximum": 0.10,
    "candidate_unsupported_assertion_maximum": 0.0,
    "candidate_think_leak_maximum": 0.0,
    "candidate_privacy_echo_maximum": 0.0,
    "simulated_blind_user_review_required": True,
    "auto_promotion_allowed": False,
}
SANITY_SELECTION_RULE = {
    "required_sessions_per_arm": 5,
    "overall_regression_tolerance": 0.05,
    "ordinary_regression_tolerance": 0.20,
    "ranking_order": [
        "native_format_rate_desc",
        "false_block_rate_asc",
        "truncated_session_rate_asc",
        "overall_score_desc",
        "balanced_tie_break",
    ],
}


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _job_spec(
    samples: Iterable[Mapping[str, Any]],
    output_dir: Path,
    *,
    curriculum: str,
    steps: int,
) -> dict[str, Any]:
    spec = build_phase78_sft_job_spec(
        samples=samples,
        base_model=str(MODEL_PATH),
        output_dir=str(output_dir),
        max_steps=steps,
        learning_rate=TRAINING_LEARNING_RATE,
        seed=90,
    )
    spec["recipe"]["training"]["max_length"] = PHASE90_TRAINING_MAX_LENGTH
    spec["phase78"]["target_model"] = "Qwen2.5-1.5B-Instruct"
    spec["phase90"] = {
        "curriculum": curriculum,
        "prompt_contract_aligned": True,
        "completion_only_loss_required": True,
        "target_model": "Qwen2.5-1.5B-Instruct",
        "model_revision": MODEL_REVISION,
        "simulated_usage": True,
        "actual_user_feedback": False,
        "automatic_training_allowed": False,
        "auto_promotion_allowed": False,
    }
    return spec


def _source_hashes() -> dict[str, str]:
    paths = {
        "core": CORE_ROOT / "pfe_core/phase90_native_format_curriculum.py",
        "driver": REPO_ROOT / "tools/phase90_native_format_curriculum_repair.py",
        "core_test": REPO_ROOT / "tests/test_phase90_native_format_curriculum.py",
        "driver_test": REPO_ROOT / "tests/test_phase90_driver_safety.py",
        "executor": CORE_ROOT / "pfe_core/trainer/executors.py",
        "phase89_driver": REPO_ROOT / "tools/phase87_89_failure_driven_adapter_loop.py",
    }
    return {name: _sha256(path) for name, path in paths.items()}


def _phase89_adapter_validation() -> dict[str, Any]:
    evidence = _read_json(
        PHASE89_ROOT / "evidence-real-training/probe-25step/adapter_validation.json"
    )
    adapter_path = PHASE89_ADAPTER_ROOT / "adapter_model.safetensors"
    return {
        "artifact_dir": str(PHASE89_ADAPTER_ROOT),
        "evidence_sha256": evidence.get("sha256"),
        "actual_sha256": _sha256(adapter_path) if adapter_path.is_file() else None,
        "valid": evidence.get("valid") is True
        and adapter_path.is_file()
        and _sha256(adapter_path) == evidence.get("sha256"),
    }


def _prepare(clean: bool) -> int:
    if clean and EVIDENCE_ROOT.exists():
        _safe_clean(EVIDENCE_ROOT, REPO_ROOT / "docs/demo")
    PREPARATION_ROOT.mkdir(parents=True, exist_ok=True)
    candidates = build_phase90_curriculum_candidates()
    quality = audit_phase90_curriculum_candidates(candidates)
    plan = build_phase90_training_plan(candidates)
    holdout = build_phase90_holdout()
    previous_holdout = _read_json(PHASE89_ROOT / "evidence-preparation/holdout.json")
    isolation = audit_phase90_holdout_isolation(
        candidates, holdout, previous_holdout
    )
    phase89_decision = _read_json(PHASE89_ROOT / "phase89-final-decision.json")
    phase89_adapter = _phase89_adapter_validation()

    boundary_reports: dict[str, Any] = {}
    for curriculum in PHASE90_CURRICULA:
        for steps in (5, 25):
            selected = select_phase90_training_samples(
                candidates, curriculum=curriculum, steps=steps
            )
            _write_jsonl(
                PREPARATION_ROOT / f"selected_{curriculum}_{steps}step.jsonl",
                selected,
            )
            boundary_reports[f"{curriculum}_{steps}"] = _completion_boundary_report(
                _job_spec(
                    selected,
                    TRAINER_OUTPUT_ROOT / f"phase90-preflight-{curriculum}-{steps}",
                    curriculum=curriculum,
                    steps=steps,
                )
            )

    model_config = MODEL_PATH / "config.json"
    model_weights = MODEL_PATH / "model.safetensors"
    checks = {
        "phase89_remains_archive": phase89_decision.get("status")
        == "archive_failure_driven_adapter_not_qualified",
        "phase89_product_gate_false": phase89_decision.get("product_gate_qualified")
        is False,
        "phase89_adapter_available_as_archived_reference": phase89_adapter.get("valid")
        is True,
        "curriculum_quality_passed": quality.get("passed") is True,
        "holdout_isolation_passed": isolation.get("passed") is True,
        "all_completion_boundaries_passed": all(
            report.get("passed") is True for report in boundary_reports.values()
        ),
        "local_model_complete": _model_complete(),
        "candidate_count_120": int(candidates.get("sample_count") or 0) == 120,
        "fresh_holdout_count_40": int(holdout.get("session_count") or 0) == 40,
    }
    freeze = {
        "kind": "phase90_pre_experiment_freeze",
        "created_at": _utcnow(),
        "frozen_before_training": True,
        "passed": all(checks.values()),
        "checks": checks,
        "hypothesis": (
            "Aligning the SFT system prompt with the Phase85 runtime contract should "
            "raise raw native format and reduce runtime fallback without increasing false blocks."
        ),
        "model_path": str(MODEL_PATH),
        "model_revision": MODEL_REVISION,
        "model_config_sha256": _sha256(model_config) if model_config.is_file() else None,
        "model_weight_size_bytes": model_weights.stat().st_size
        if model_weights.is_file()
        else 0,
        "candidate_manifest_sha256": quality.get("sample_manifest_sha256"),
        "training_plan_sha256": stable_hash(plan),
        "holdout_manifest_sha256": stable_hash(holdout.get("sessions") or []),
        "previous_holdout_manifest_sha256": stable_hash(
            previous_holdout.get("sessions") or []
        ),
        "phase89_adapter_sha256": phase89_adapter.get("actual_sha256"),
        "generation_protocol_sha256": stable_hash(GENERATION_PROTOCOL),
        "thresholds_sha256": stable_hash(FROZEN_THRESHOLDS),
        "sanity_selection_rule_sha256": stable_hash(SANITY_SELECTION_RULE),
        "source_sha256": _source_hashes(),
        "score_or_gate_relaxation_allowed": False,
        "automatic_promotion_allowed": False,
    }
    _write_json(PREPARATION_ROOT / "training_quality_audit.json", quality)
    _write_json(PREPARATION_ROOT / "training_plan.json", plan)
    _write_json(PREPARATION_ROOT / "holdout.json", holdout)
    _write_json(PREPARATION_ROOT / "holdout_isolation_audit.json", isolation)
    _write_json(PREPARATION_ROOT / "phase89_adapter_reference.json", phase89_adapter)
    _write_json(PREPARATION_ROOT / "completion_boundary_reports.json", boundary_reports)
    _write_json(EVIDENCE_ROOT / "generation_protocol.json", GENERATION_PROTOCOL)
    _write_json(EVIDENCE_ROOT / "frozen_thresholds.json", FROZEN_THRESHOLDS)
    _write_json(EVIDENCE_ROOT / "sanity_selection_rule.json", SANITY_SELECTION_RULE)
    _write_json(EVIDENCE_ROOT / "pre_experiment_freeze.json", freeze)
    _write_json(EVIDENCE_ROOT / "preparation_decision.json", {
        "kind": "phase90_preparation_decision",
        "status": "ready_for_5_step_ablation" if freeze["passed"] else "blocked",
        "checks": checks,
        "automatic_training_started": False,
        "product_gate_qualified": False,
    })
    print(json.dumps({
        "status": "ready_for_5_step_ablation" if freeze["passed"] else "blocked",
        "checks": checks,
    }, ensure_ascii=False, indent=2))
    return 0 if freeze["passed"] else 2


def _selected_rows(curriculum: str, steps: int) -> list[dict[str, Any]]:
    return _read_jsonl(
        PREPARATION_ROOT / f"selected_{curriculum}_{steps}step.jsonl"
    )


def _training_freeze_check(curriculum: str, steps: int) -> dict[str, Any]:
    freeze = _read_json(EVIDENCE_ROOT / "pre_experiment_freeze.json")
    plan = _read_json(PREPARATION_ROOT / "training_plan.json")
    rows = _selected_rows(curriculum, steps)
    expected = dict(
        dict(plan.get("curricula") or {}).get(curriculum, {}).get(str(steps), {})
    )
    holdout = _read_json(PREPARATION_ROOT / "holdout.json")
    sanity = (
        _read_json(EVIDENCE_ROOT / "sanity_decision.json")
        if (EVIDENCE_ROOT / "sanity_decision.json").is_file()
        else {}
    )
    checks = {
        "pre_experiment_freeze_passed": freeze.get("passed") is True,
        "training_selection_unchanged": stable_hash(rows)
        == expected.get("manifest_sha256"),
        "holdout_manifest_unchanged": stable_hash(holdout.get("sessions") or [])
        == freeze.get("holdout_manifest_sha256"),
        "model_config_unchanged": _sha256(MODEL_PATH / "config.json")
        == freeze.get("model_config_sha256"),
        "model_weight_size_unchanged": (MODEL_PATH / "model.safetensors").stat().st_size
        == int(freeze.get("model_weight_size_bytes") or 0),
        "generation_protocol_unchanged": stable_hash(GENERATION_PROTOCOL)
        == freeze.get("generation_protocol_sha256"),
        "thresholds_unchanged": stable_hash(FROZEN_THRESHOLDS)
        == freeze.get("thresholds_sha256"),
        "sanity_rule_unchanged": stable_hash(SANITY_SELECTION_RULE)
        == freeze.get("sanity_selection_rule_sha256"),
        "source_files_unchanged": _source_hashes() == freeze.get("source_sha256"),
        "twenty_five_step_is_frozen_sanity_winner": steps != 25
        or (
            sanity.get("passed") is True
            and sanity.get("selected_curriculum") == curriculum
        ),
    }
    return {
        "kind": "phase90_training_freeze_check",
        "curriculum": curriculum,
        "steps": steps,
        "passed": all(checks.values()),
        "checks": checks,
    }


def _train(curriculum: str, steps: int, clean: bool) -> int:
    if curriculum not in PHASE90_CURRICULA or steps not in (5, 25):
        raise SystemExit("Phase90 permits two frozen curricula and 5/25 steps only")
    freeze = _training_freeze_check(curriculum, steps)
    probe_dir = TRAINING_ROOT / f"{curriculum}-{steps}step"
    output_root = TRAINER_OUTPUT_ROOT / f"phase90-{curriculum}-{steps}step"
    if clean and probe_dir.exists():
        _safe_clean(probe_dir, TRAINING_ROOT)
    if clean and output_root.exists():
        _safe_clean(output_root, TRAINER_OUTPUT_ROOT)
    probe_dir.mkdir(parents=True, exist_ok=True)
    rows = _selected_rows(curriculum, steps)
    spec = _job_spec(rows, output_root, curriculum=curriculum, steps=steps)
    boundary = _completion_boundary_report(spec)
    _write_json(probe_dir / "training_manifest.json", spec)
    _write_json(probe_dir / "training_freeze_check.json", freeze)
    _write_json(probe_dir / "completion_boundary_report.json", boundary)
    if not freeze["passed"] or boundary.get("passed") is not True:
        attempt = {
            "kind": "phase90_training_attempt",
            "status": "blocked",
            "curriculum": curriculum,
            "requested_steps": steps,
            "real_training": False,
            "reason": "training_freeze_or_completion_boundary_failed",
            "product_gate_qualified": False,
            "auto_promotion_allowed": False,
        }
        _write_json(probe_dir / "training_attempt.json", attempt)
        return 2
    started = time.perf_counter()
    started_at = _utcnow()
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    try:
        result = _run_real_local_peft_training(spec)
        real = dict(result.get("real_execution") or {})
        artifact_dir = Path(str(real.get("artifact_dir") or ""))
        adapter_path = artifact_dir / "adapter_model.safetensors"
        validation = validate_adapter_artifact(
            artifact_dir,
            {"artifact_name": adapter_path.name, "artifact_format": "peft_lora"},
        )
        validation.update({
            "sha256": _sha256(adapter_path) if adapter_path.is_file() else None,
            "artifact_dir": str(artifact_dir),
            "adapter_path": str(adapter_path),
            "parameters_updated": real.get("parameters_updated"),
            "steps": real.get("steps"),
        })
        completed = (
            result.get("status") == "completed"
            and real.get("success") is True
            and real.get("parameters_updated") is True
            and int(real.get("steps") or 0) >= steps
            and validation.get("valid") is True
        )
        attempt = {
            "kind": "phase90_training_attempt",
            "status": "completed" if completed else "failed",
            "real_training": completed,
            "candidate_eligible": False,
            "curriculum": curriculum,
            "selected_model": "Qwen2.5-1.5B-Instruct",
            "model": str(MODEL_PATH),
            "model_revision": MODEL_REVISION,
            "requested_steps": steps,
            "learning_rate": TRAINING_LEARNING_RATE,
            "seed": 90,
            "started_at": started_at,
            "finished_at": _utcnow(),
            "duration_seconds": round(time.perf_counter() - started, 4),
            "max_rss_before_bytes": rss_before,
            "max_rss_after_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "execution": real,
            "adapter_validation": validation,
            "training_rows": summarize_phase90_training_rows(rows),
            "simulated_usage": True,
            "actual_user_feedback": False,
            "actual_product_benefit_claim_allowed": False,
            "product_gate_qualified": False,
            "auto_promotion_allowed": False,
        }
        _write_json(probe_dir / "adapter_validation.json", validation)
        _write_json(probe_dir / "train_log.json", {
            "loss_history": real.get("loss_history") or [],
            "initial_loss": real.get("initial_loss"),
            "final_loss": real.get("final_loss"),
        })
        _write_json(probe_dir / "parameter_fingerprint_before_after.json", {
            "before": real.get("parameter_fingerprint_before"),
            "after": real.get("parameter_fingerprint_after"),
            "parameters_updated": real.get("parameters_updated"),
        })
    except Exception as exc:
        attempt = {
            "kind": "phase90_training_attempt",
            "status": "failed",
            "real_training": False,
            "curriculum": curriculum,
            "requested_steps": steps,
            "started_at": started_at,
            "finished_at": _utcnow(),
            "duration_seconds": round(time.perf_counter() - started, 4),
            "max_rss_before_bytes": rss_before,
            "max_rss_after_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "error": f"{exc.__class__.__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "simulated_usage": True,
            "actual_user_feedback": False,
            "actual_product_benefit_claim_allowed": False,
            "product_gate_qualified": False,
            "auto_promotion_allowed": False,
        }
        FAILURE_ROOT.mkdir(parents=True, exist_ok=True)
        _write_json(
            FAILURE_ROOT / f"training_{curriculum}_{steps}step.json", attempt
        )
    _write_json(probe_dir / "training_attempt.json", attempt)
    print(json.dumps({
        "status": attempt.get("status"),
        "curriculum": curriculum,
        "requested_steps": steps,
        "duration_seconds": attempt.get("duration_seconds"),
        "error": attempt.get("error"),
    }, ensure_ascii=False, indent=2))
    return 0 if attempt.get("status") == "completed" else 1


def _phase90_adapter_dir(curriculum: str, steps: int) -> Path:
    attempt = _read_json(
        TRAINING_ROOT / f"{curriculum}-{steps}step/training_attempt.json"
    )
    validation = dict(attempt.get("adapter_validation") or {})
    artifact_dir = Path(str(validation.get("artifact_dir") or ""))
    if (
        attempt.get("status") != "completed"
        or validation.get("valid") is not True
        or not artifact_dir.is_dir()
    ):
        raise SystemExit(f"Phase90 {curriculum} {steps}-step adapter unavailable")
    return artifact_dir.resolve()


def _scope_sessions(scope: str) -> list[dict[str, Any]]:
    sessions = [
        dict(row)
        for row in _read_json(PREPARATION_ROOT / "holdout.json").get("sessions") or []
    ]
    if scope == "full":
        return sessions
    selected = []
    for category in PHASE87_CATEGORIES:
        selected.append(next(row for row in sessions if row.get("category") == category))
    return selected


def _variant_adapter(scope: str, variant: str) -> tuple[Path | None, str | None]:
    if variant == "base":
        return None, None
    if scope == "sanity" and variant in PHASE90_CURRICULA:
        return _phase90_adapter_dir(variant, 5), variant
    if scope == "full" and variant == "phase89":
        return PHASE89_ADAPTER_ROOT.resolve(), "phase89_balanced_archived"
    if scope == "full" and variant == "candidate":
        sanity = _read_json(EVIDENCE_ROOT / "sanity_decision.json")
        curriculum = str(sanity.get("selected_curriculum") or "")
        return _phase90_adapter_dir(curriculum, 25), curriculum
    raise SystemExit(f"unsupported Phase90 {scope} generation variant: {variant}")


def _generation_freeze_check(
    scope: str, variant: str, adapter_path: Path | None
) -> dict[str, Any]:
    freeze = _read_json(EVIDENCE_ROOT / "pre_experiment_freeze.json")
    holdout = _read_json(PREPARATION_ROOT / "holdout.json")
    sanity = (
        _read_json(EVIDENCE_ROOT / "sanity_decision.json")
        if (EVIDENCE_ROOT / "sanity_decision.json").is_file()
        else {}
    )
    checks = {
        "pre_experiment_freeze_passed": freeze.get("passed") is True,
        "holdout_manifest_unchanged": stable_hash(holdout.get("sessions") or [])
        == freeze.get("holdout_manifest_sha256"),
        "generation_protocol_unchanged": stable_hash(GENERATION_PROTOCOL)
        == freeze.get("generation_protocol_sha256"),
        "source_files_unchanged": _source_hashes() == freeze.get("source_sha256"),
        "model_config_unchanged": _sha256(MODEL_PATH / "config.json")
        == freeze.get("model_config_sha256"),
        "adapter_available_or_base": adapter_path is None or adapter_path.is_dir(),
        "full_requires_passed_sanity": scope != "full" or sanity.get("passed") is True,
        "full_candidate_is_frozen_winner": scope != "full"
        or variant != "candidate"
        or bool(sanity.get("selected_curriculum")),
        "phase89_adapter_hash_unchanged": variant != "phase89"
        or _phase89_adapter_validation().get("actual_sha256")
        == freeze.get("phase89_adapter_sha256"),
    }
    return {
        "kind": "phase90_generation_freeze_check",
        "scope": scope,
        "variant": variant,
        "passed": all(checks.values()),
        "checks": checks,
    }


def _generate(scope: str, variant: str, clean: bool) -> int:
    if scope not in {"sanity", "full"}:
        raise SystemExit("Phase90 generation scope must be sanity or full")
    adapter_path, curriculum = _variant_adapter(scope, variant)
    freeze = _generation_freeze_check(scope, variant, adapter_path)
    if not freeze["passed"]:
        raise SystemExit(f"Phase90 generation freeze failed: {freeze}")
    root = EVAL_ROOT / scope
    structural_path = root / f"structural_sessions_{variant}.jsonl"
    metrics_path = root / f"metrics_{variant}.json"
    cache_path = REVIEW_CACHE_ROOT / f"{scope}_{variant}.jsonl"
    if clean:
        structural_path.unlink(missing_ok=True)
        metrics_path.unlink(missing_ok=True)
        cache_path.unlink(missing_ok=True)
        cache_path.with_suffix(cache_path.suffix + ".tmp").unlink(missing_ok=True)
    rows = _read_jsonl(structural_path) if structural_path.is_file() else []
    private_rows = _read_jsonl(cache_path) if cache_path.is_file() else []
    completed = {
        str(row.get("session_id") or "")
        for row in rows
        if row.get("status") == "completed"
    } & {str(row.get("session_id") or "") for row in private_rows}
    sessions = _scope_sessions(scope)
    torch = tokenizer = model = device = None
    try:
        torch, tokenizer, model, device = _load_runtime(adapter_path)
        for index, session in enumerate(sessions, start=1):
            session_id = str(session.get("session_id") or "")
            if session_id in completed:
                print(f"[{scope}:{variant}] {index}/{len(sessions)} resumed", flush=True)
                continue
            try:
                structural, private = _run_eval_session(
                    session=session,
                    torch=torch,
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                    adapter_loaded=adapter_path is not None,
                )
                structural["kind"] = "phase90_structural_eval_session"
                structural["variant"] = variant
                structural["curriculum"] = curriculum
            except Exception as exc:
                structural = {
                    "kind": "phase90_structural_eval_session",
                    "session_id": session_id,
                    "category": session.get("category"),
                    "variant": variant,
                    "curriculum": curriculum,
                    "status": "failed",
                    "actual_model_call": False,
                    "error_type": exc.__class__.__name__,
                    "raw_model_output_persisted": False,
                    "private_source_persisted": False,
                    "simulated_usage": True,
                    "actual_user_feedback": False,
                }
                private = {
                    "session_id": session_id,
                    "category": session.get("category"),
                    "error_type": exc.__class__.__name__,
                }
            rows = [row for row in rows if row.get("session_id") != session_id]
            rows.append(structural)
            rows.sort(key=lambda row: str(row.get("session_id") or ""))
            private_rows = [
                row for row in private_rows if row.get("session_id") != session_id
            ]
            private_rows.append(private)
            private_rows.sort(key=lambda row: str(row.get("session_id") or ""))
            _write_jsonl(structural_path, rows)
            _write_private_jsonl(cache_path, private_rows)
            print(
                f"[{scope}:{variant}] {index}/{len(sessions)} {session_id} "
                f"{structural['status']}",
                flush=True,
            )
    finally:
        if torch is not None and model is not None and device is not None:
            _release_runtime(torch, model, device)
    completed_rows = [row for row in rows if row.get("status") == "completed"]
    raw_metrics = aggregate_phase89_scores(
        {
            "category": row.get("category"),
            "score": row.get("raw_score"),
            "truncated": row.get("truncated"),
        }
        for row in completed_rows
    )
    post_metrics = aggregate_phase89_scores(
        {
            "category": row.get("category"),
            "score": row.get("post_score"),
            "truncated": row.get("truncated"),
        }
        for row in completed_rows
    )
    fallback_count = sum(
        row.get("final_fallback_used") is True for row in completed_rows
    )
    post_metrics.update({
        "fallback_count": fallback_count,
        "fallback_rate": round(fallback_count / len(completed_rows), 4)
        if completed_rows
        else 0.0,
    })
    metrics = {
        "kind": "phase90_variant_metrics",
        "scope": scope,
        "variant": variant,
        "curriculum": curriculum,
        "model": "Qwen2.5-1.5B-Instruct",
        "adapter_loaded": adapter_path is not None,
        "session_count": len(completed_rows),
        "model_call_count": sum(
            int(row.get("turn_count") or 0) for row in completed_rows
        ),
        "all_sessions_completed": len(completed_rows) == len(sessions),
        "actual_model_calls": len(completed_rows) == len(sessions),
        "raw": raw_metrics,
        "post_contract": post_metrics,
        "raw_model_output_persisted": False,
        "review_cache_outside_evidence_root": True,
        "simulated_usage": True,
        "actual_user_feedback_count": 0,
        "product_gate_qualified": False,
    }
    _write_json(root / f"freeze_check_{variant}.json", freeze)
    _write_json(metrics_path, metrics)
    print(json.dumps({
        "scope": scope,
        "variant": variant,
        "curriculum": curriculum,
        "session_count": metrics["session_count"],
        "model_call_count": metrics["model_call_count"],
        "raw_overall_score": raw_metrics.get("overall_score"),
        "raw_native_format_rate": raw_metrics.get("native_format_rate"),
        "post_fallback_rate": post_metrics.get("fallback_rate"),
    }, ensure_ascii=False, indent=2))
    return 0 if metrics["all_sessions_completed"] else 1


def _sanity_sort_key(metrics: Mapping[str, Any], curriculum: str) -> tuple[float, ...]:
    raw = dict(metrics.get("raw") or {})
    return (
        float(raw.get("native_format_rate") or 0.0),
        -float(raw.get("false_block_rate") or 0.0),
        -float(raw.get("truncated_session_rate") or 0.0),
        float(raw.get("overall_score") or 0.0),
        float(curriculum == "balanced"),
    )


def _sanity() -> int:
    base = _read_json(EVAL_ROOT / "sanity/metrics_base.json")
    base_raw = dict(base.get("raw") or {})
    base_ordinary = float(
        dict(dict(base_raw.get("category_metrics") or {}).get("ordinary_direct") or {}).get(
            "composite_score"
        )
        or 0.0
    )
    arms: dict[str, Any] = {}
    for curriculum in PHASE90_CURRICULA:
        metrics = _read_json(EVAL_ROOT / f"sanity/metrics_{curriculum}.json")
        raw = dict(metrics.get("raw") or {})
        ordinary = float(
            dict(
                dict(raw.get("category_metrics") or {}).get("ordinary_direct") or {}
            ).get("composite_score")
            or 0.0
        )
        checks = {
            "completed_five_sessions": metrics.get("session_count") == 5
            and metrics.get("model_call_count") == 15,
            "five_step_adapter_valid": _read_json(
                TRAINING_ROOT / f"{curriculum}-5step/training_attempt.json"
            ).get("status")
            == "completed",
            "overall_not_below_tolerance": float(raw.get("overall_score") or 0.0)
            >= float(base_raw.get("overall_score") or 0.0) - 0.05,
            "native_not_below_base": float(raw.get("native_format_rate") or 0.0)
            >= float(base_raw.get("native_format_rate") or 0.0),
            "false_block_not_above_base": float(raw.get("false_block_rate") or 0.0)
            <= float(base_raw.get("false_block_rate") or 0.0),
            "unsupported_not_above_base": float(
                raw.get("unsupported_assertion_rate") or 0.0
            )
            <= float(base_raw.get("unsupported_assertion_rate") or 0.0),
            "ordinary_regression_within_0_20": base_ordinary - ordinary <= 0.20,
            "privacy_echo_zero": float(raw.get("privacy_echo_rate") or 0.0) == 0.0,
            "think_leak_zero": float(raw.get("think_leak_rate") or 0.0) == 0.0,
        }
        arms[curriculum] = {
            "passed": all(checks.values()),
            "checks": checks,
            "raw": raw,
            "ranking_key": list(_sanity_sort_key(metrics, curriculum)),
        }
    passing = [name for name, arm in arms.items() if arm["passed"]]
    selected = max(
        passing,
        key=lambda name: tuple(arms[name]["ranking_key"]),
        default=None,
    )
    payload = {
        "kind": "phase90_five_step_sanity_decision",
        "passed": selected is not None,
        "base_raw": base_raw,
        "arms": arms,
        "selected_curriculum": selected,
        "selection_rule": SANITY_SELECTION_RULE,
        "next_action": "run_selected_25_step_probe" if selected else "archive_sanity_failure",
        "product_gate_qualified": False,
        "promotion_allowed": False,
        "auto_promotion_allowed": False,
    }
    _write_json(EVIDENCE_ROOT / "sanity_decision.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["passed"] else 1


def _review_template(clean: bool) -> int:
    base_rows = {
        str(row.get("session_id")): row
        for row in _read_jsonl(REVIEW_CACHE_ROOT / "full_base.jsonl")
    }
    candidate_rows = {
        str(row.get("session_id")): row
        for row in _read_jsonl(REVIEW_CACHE_ROOT / "full_candidate.jsonl")
    }
    session_ids = sorted(base_rows)
    if session_ids != sorted(candidate_rows) or len(session_ids) != PHASE90_HOLDOUT_COUNT:
        raise SystemExit("Phase90 full review caches are incomplete")
    blind_rows = []
    public_pairs = []
    variant_key: dict[str, Any] = {}
    for session_id in session_ids:
        base_row = base_rows[session_id]
        candidate_row = candidate_rows[session_id]
        base_is_a = int(hashlib.sha256(session_id.encode()).hexdigest(), 16) % 2 == 0
        candidate_a = base_row if base_is_a else candidate_row
        candidate_b = candidate_row if base_is_a else base_row
        pair_id = f"phase90-pair-{hashlib.sha256(session_id.encode()).hexdigest()[:12]}"
        blind_rows.append({
            "pair_id": pair_id,
            "session_id": session_id,
            "category": base_row.get("category"),
            "candidate_a_output": candidate_a.get("raw_output"),
            "candidate_b_output": candidate_b.get("raw_output"),
            "candidate_a_output_sha256": candidate_a.get("raw_output_sha256"),
            "candidate_b_output_sha256": candidate_b.get("raw_output_sha256"),
        })
        public_pairs.append({
            "pair_id": pair_id,
            "session_id": session_id,
            "category": base_row.get("category"),
            "candidate_a_output_sha256": candidate_a.get("raw_output_sha256"),
            "candidate_b_output_sha256": candidate_b.get("raw_output_sha256"),
        })
        variant_key[pair_id] = {
            "candidate_a": "base" if base_is_a else "candidate",
            "candidate_b": "candidate" if base_is_a else "base",
        }
    blind_path = REVIEW_CACHE_ROOT / "blind_pairs.jsonl"
    if clean:
        blind_path.unlink(missing_ok=True)
    _write_private_jsonl(blind_path, blind_rows)
    expected_hash = stable_hash(public_pairs)
    _write_json(EVAL_ROOT / "simulated_user_review.json", {
        "kind": "phase90_blind_simulated_user_review",
        "complete": False,
        "passed": False,
        "expected_pair_count": PHASE90_HOLDOUT_COUNT,
        "expected_pair_manifest_sha256": expected_hash,
        "reviewed_pair_count": 0,
        "reviewer_ids": [],
        "pairs": public_pairs,
        "decisions": [],
        "allowed_findings": [
            "false_block",
            "unsupported_assertion",
            "provenance_failure",
            "format_failure",
            "other_semantic_failure",
        ],
        "review_is_simulated_not_actual_user_feedback": True,
        "raw_output_persisted_in_evidence": False,
        "review_cache_outside_evidence_root": True,
        "actual_user_feedback_count": 0,
    })
    _write_json(EVAL_ROOT / "blind_variant_key.json", {
        "kind": "phase90_blind_variant_key",
        "pair_count": len(variant_key),
        "mapping": variant_key,
        "contains_raw_output": False,
    })
    print(json.dumps({
        "pair_count": len(public_pairs),
        "pair_manifest_sha256": expected_hash,
        "private_review_cache": str(blind_path),
    }, ensure_ascii=False, indent=2))
    return 0


def _simulated_findings(output: str, session: Mapping[str, Any]) -> list[str]:
    score = score_phase89_output(output, session)
    findings = []
    if score.get("false_block") is True:
        findings.append("false_block")
    if score.get("unsupported_assertion") is True:
        findings.append("unsupported_assertion")
    if session.get("provenance_rejection_expected") is True and score.get(
        "provenance_correct"
    ) is not True:
        findings.append("provenance_failure")
    if score.get("native_format") is not True:
        findings.append("format_failure")
    if score.get("category_correct") is not True and not findings:
        findings.append("other_semantic_failure")
    return findings


def _finding_penalty(findings: Iterable[str]) -> int:
    weights = {
        "unsupported_assertion": 5,
        "false_block": 4,
        "provenance_failure": 4,
        "other_semantic_failure": 3,
        "format_failure": 2,
    }
    return sum(weights.get(finding, 1) for finding in findings)


def _simulate_user_review() -> int:
    review_path = EVAL_ROOT / "simulated_user_review.json"
    review = _read_json(review_path)
    blind_rows = _read_jsonl(REVIEW_CACHE_ROOT / "blind_pairs.jsonl")
    sessions = {
        str(row.get("session_id")): row
        for row in _read_json(PREPARATION_ROOT / "holdout.json").get("sessions") or []
    }
    decisions = []
    for row in blind_rows:
        session = sessions[str(row.get("session_id"))]
        findings_a = _simulated_findings(str(row.get("candidate_a_output") or ""), session)
        findings_b = _simulated_findings(str(row.get("candidate_b_output") or ""), session)
        penalty_a = _finding_penalty(findings_a)
        penalty_b = _finding_penalty(findings_b)
        winner = "tie"
        if penalty_a < penalty_b:
            winner = "candidate_a"
        elif penalty_b < penalty_a:
            winner = "candidate_b"
        decisions.append({
            "pair_id": row.get("pair_id"),
            "candidate_a_output_sha256": row.get("candidate_a_output_sha256"),
            "candidate_b_output_sha256": row.get("candidate_b_output_sha256"),
            "winner": winner,
            "findings_a": findings_a,
            "findings_b": findings_b,
        })
    review.update({
        "complete": True,
        "reviewed_pair_count": len(decisions),
        "reviewer_ids": ["codex_simulated_user_phase90"],
        "decisions": decisions,
        "review_method": "blind deterministic simulated-user rubric",
        "review_is_simulated_not_actual_user_feedback": True,
    })
    _write_json(review_path, review)
    print(json.dumps({
        "reviewed_pair_count": len(decisions),
        "review_is_simulated_not_actual_user_feedback": True,
    }, ensure_ascii=False, indent=2))
    return 0


def _review_validate() -> int:
    review = _read_json(EVAL_ROOT / "simulated_user_review.json")
    variant_key = dict(
        _read_json(EVAL_ROOT / "blind_variant_key.json").get("mapping") or {}
    )
    pairs = [dict(row) for row in review.get("pairs") or []]
    decisions = [dict(row) for row in review.get("decisions") or []]
    pair_by_id = {str(row.get("pair_id")): row for row in pairs}
    decision_by_id = {str(row.get("pair_id")): row for row in decisions}
    allowed = set(review.get("allowed_findings") or [])
    integrity = (
        len(pairs) == PHASE90_HOLDOUT_COUNT
        and stable_hash(pairs) == review.get("expected_pair_manifest_sha256")
        and set(pair_by_id) == set(decision_by_id) == set(variant_key)
        and len(decisions) == len(decision_by_id)
    )
    candidate_wins = base_wins = ties = candidate_findings = 0
    finding_counts: Counter[str] = Counter()
    decisions_valid = integrity
    for pair_id, decision in decision_by_id.items():
        pair = pair_by_id[pair_id]
        key = dict(variant_key[pair_id])
        if (
            decision.get("candidate_a_output_sha256")
            != pair.get("candidate_a_output_sha256")
            or decision.get("candidate_b_output_sha256")
            != pair.get("candidate_b_output_sha256")
            or decision.get("winner") not in {"candidate_a", "candidate_b", "tie"}
        ):
            decisions_valid = False
            continue
        findings_a = list(decision.get("findings_a") or [])
        findings_b = list(decision.get("findings_b") or [])
        if not set(findings_a) <= allowed or not set(findings_b) <= allowed:
            decisions_valid = False
            continue
        winner = decision.get("winner")
        if winner == "tie":
            ties += 1
        elif key[winner] == "candidate":
            candidate_wins += 1
        else:
            base_wins += 1
        for label, findings in (("candidate_a", findings_a), ("candidate_b", findings_b)):
            finding_counts.update(findings)
            if key[label] == "candidate":
                candidate_findings += len(findings)
    complete = (
        review.get("complete") is True
        and int(review.get("reviewed_pair_count") or 0) == PHASE90_HOLDOUT_COUNT
        and bool(review.get("reviewer_ids"))
        and decisions_valid
        and review.get("review_is_simulated_not_actual_user_feedback") is True
    )
    passed = complete and candidate_wins > base_wins and candidate_findings == 0
    summary = {
        "kind": "phase90_blind_simulated_user_review_summary",
        "complete": complete,
        "integrity_passed": decisions_valid,
        "passed": passed,
        "reviewed_pair_count": len(decisions),
        "candidate_wins": candidate_wins,
        "base_wins": base_wins,
        "ties": ties,
        "candidate_finding_count": candidate_findings,
        "finding_counts": dict(sorted(finding_counts.items())),
        "review_is_simulated_not_actual_user_feedback": True,
        "simulated_review_can_only_tighten": True,
        "raw_output_persisted_in_evidence": False,
        "product_gate_qualified": False,
        "promotion_allowed": False,
    }
    _write_json(EVAL_ROOT / "simulated_user_review_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if complete and decisions_valid else 1


def _run_regression() -> int:
    command = [
        str(REPO_ROOT / ".venv/bin/pytest"),
        "-q",
        "tests/test_phase90_native_format_curriculum.py",
        "tests/test_phase90_driver_safety.py",
        "tests/test_phase87_failure_driven_training.py",
        "tests/test_phase87_89_driver_safety.py",
        "tests/test_phase85_low_fallback_semantic_guard.py",
        "tests/test_phase85_metric_schema_v2_overlay.py",
    ]
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    output = completed.stdout or ""
    summary = {
        "kind": "phase90_regression_summary",
        "passed": completed.returncode == 0,
        "exit_code": completed.returncode,
        "duration_seconds": round(time.perf_counter() - started, 4),
        "output_line_count": len(output.splitlines()),
        "output_sha256": hashlib.sha256(output.encode()).hexdigest(),
        "raw_process_output_persisted": False,
    }
    _write_json(EVIDENCE_ROOT / "regression_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return completed.returncode


def _walk_forbidden_keys(value: Any, path: str = "$") -> list[str]:
    failures = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if key in {"raw_output", "candidate_a_output", "candidate_b_output"}:
                failures.append(child_path)
            failures.extend(_walk_forbidden_keys(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            failures.extend(_walk_forbidden_keys(child, f"{path}[{index}]"))
    return failures


def _write_runbook(selected: str, decision: Mapping[str, Any]) -> None:
    text = "# Phase90 Native Format Curriculum Repair\n\n"
    text += "Local Qwen2.5-1.5B only; all sessions are simulated_usage.\n\n"
    text += f"- Selected 5-step curriculum: `{selected}`\n"
    text += f"- Final status: `{decision.get('status')}`\n"
    text += f"- Product gate qualified: `{str(decision.get('product_gate_qualified')).lower()}`\n"
    text += "- Automatic promotion/deployment: forbidden\n\n"
    text += "Raw outputs live only in a mode-0600 temporary review cache and are deleted at finalize.\n"
    (EVIDENCE_ROOT / "phase90-runbook.md").write_text(text, encoding="utf-8")


def _finalize() -> int:
    sanity = _read_json(EVIDENCE_ROOT / "sanity_decision.json")
    selected = str(sanity.get("selected_curriculum") or "")
    base = _read_json(EVAL_ROOT / "full/metrics_base.json")
    phase89 = _read_json(EVAL_ROOT / "full/metrics_phase89.json")
    candidate = _read_json(EVAL_ROOT / "full/metrics_candidate.json")
    review = _read_json(EVAL_ROOT / "simulated_user_review_summary.json")
    isolation = _read_json(PREPARATION_ROOT / "holdout_isolation_audit.json")
    training = _read_json(
        TRAINING_ROOT / f"{selected}-25step/training_attempt.json"
    )
    decision = build_phase90_decision(
        base_raw=dict(base.get("raw") or {}),
        phase89_raw=dict(phase89.get("raw") or {}),
        candidate_raw=dict(candidate.get("raw") or {}),
        base_runtime=dict(base.get("post_contract") or {}),
        candidate_runtime=dict(candidate.get("post_contract") or {}),
        training_attempt=training,
        isolation_audit=isolation,
        manual_review=review,
    )
    comparison = {
        "kind": "phase90_three_arm_comparison",
        "base": {"raw": base.get("raw"), "post_contract": base.get("post_contract")},
        "phase89_archived_adapter": {
            "raw": phase89.get("raw"),
            "post_contract": phase89.get("post_contract"),
        },
        "phase90_candidate": {
            "curriculum": selected,
            "raw": candidate.get("raw"),
            "post_contract": candidate.get("post_contract"),
        },
        "simulated_user_review": review,
        "decision_status": decision.get("status"),
        "product_gate_qualified": decision.get("product_gate_qualified"),
        "actual_product_benefit_claim_allowed": False,
        "actual_user_feedback_count": 0,
    }
    _write_json(EVIDENCE_ROOT / "comparison_summary.json", comparison)
    _write_json(EVIDENCE_ROOT / "phase90-final-decision.json", decision)
    public_failures = []
    for path in sorted(EVIDENCE_ROOT.rglob("*.json")):
        public_failures.extend(
            f"{path.relative_to(REPO_ROOT)}:{item}"
            for item in _walk_forbidden_keys(_read_json(path))
        )
    public_audit = {
        "kind": "phase90_public_private_audit",
        "passed": not public_failures,
        "forbidden_raw_output_paths": public_failures,
        "raw_output_persisted_in_evidence": False,
        "simulated_usage": True,
        "actual_user_feedback_count": 0,
    }
    _write_json(EVIDENCE_ROOT / "public_private_audit.json", public_audit)
    _write_runbook(selected, decision)
    if REVIEW_CACHE_ROOT.exists():
        shutil.rmtree(REVIEW_CACHE_ROOT)
    manifest_rows = []
    excluded = {"evidence_manifest.json", "validation_summary.json"}
    for path in sorted(EVIDENCE_ROOT.rglob("*")):
        if path.is_file() and path.name not in excluded:
            manifest_rows.append({
                "path": str(path.relative_to(REPO_ROOT)),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            })
    manifest = {
        "kind": "phase90_evidence_manifest",
        "file_count": len(manifest_rows),
        "files": manifest_rows,
    }
    _write_json(EVIDENCE_ROOT / "evidence_manifest.json", manifest)
    integrity = {
        "kind": "phase90_evidence_integrity",
        "passed": public_audit["passed"]
        and not REVIEW_CACHE_ROOT.exists()
        and all(row.get("sha256") for row in manifest_rows),
        "manifest_file_count": len(manifest_rows),
        "review_cache_absent": not REVIEW_CACHE_ROOT.exists(),
        "public_private_audit_passed": public_audit["passed"],
    }
    _write_json(EVIDENCE_ROOT / "evidence_integrity.json", integrity)
    # Rebind the manifest after writing evidence_integrity.
    manifest_rows = []
    for path in sorted(EVIDENCE_ROOT.rglob("*")):
        if path.is_file() and path.name not in excluded:
            manifest_rows.append({
                "path": str(path.relative_to(REPO_ROOT)),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            })
    _write_json(EVIDENCE_ROOT / "evidence_manifest.json", {
        "kind": "phase90_evidence_manifest",
        "file_count": len(manifest_rows),
        "files": manifest_rows,
    })
    print(json.dumps({
        "status": decision.get("status"),
        "recommendation": decision.get("recommendation"),
        "selected_curriculum": selected,
        "product_gate_qualified": decision.get("product_gate_qualified"),
        "review_cache_absent": not REVIEW_CACHE_ROOT.exists(),
    }, ensure_ascii=False, indent=2))
    return 0


def _validate() -> int:
    manifest = _read_json(EVIDENCE_ROOT / "evidence_manifest.json")
    failures = []
    for row in manifest.get("files") or []:
        path = REPO_ROOT / str(row.get("path") or "")
        if not path.is_file() or _sha256(path) != row.get("sha256"):
            failures.append(str(row.get("path") or ""))
    decision = _read_json(EVIDENCE_ROOT / "phase90-final-decision.json")
    public_audit = _read_json(EVIDENCE_ROOT / "public_private_audit.json")
    checks = {
        "manifest_files_unchanged": not failures,
        "public_private_audit_passed": public_audit.get("passed") is True,
        "review_cache_absent": not REVIEW_CACHE_ROOT.exists(),
        "no_auto_promotion": decision.get("auto_promotion_allowed") is False,
        "no_automatic_deployment": decision.get("automatic_deployment_allowed") is False,
        "no_hermes_attachment": decision.get("hermes_attachment_allowed") is False,
        "no_actual_product_benefit_claim": decision.get(
            "actual_product_benefit_claim_allowed"
        )
        is False,
    }
    summary = {
        "kind": "phase90_validation_summary",
        "passed": all(checks.values()),
        "checks": checks,
        "manifest_failures": failures,
        "decision_status": decision.get("status"),
        "product_gate_qualified": decision.get("product_gate_qualified"),
        "validation_pass_does_not_imply_product_pass": True,
    }
    _write_json(EVIDENCE_ROOT / "validation_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["passed"] else 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--clean", action="store_true")
    train = sub.add_parser("train")
    train.add_argument("--curriculum", choices=PHASE90_CURRICULA, required=True)
    train.add_argument("--steps", type=int, choices=(5, 25), required=True)
    train.add_argument("--clean", action="store_true")
    generate = sub.add_parser("generate")
    generate.add_argument("--scope", choices=("sanity", "full"), required=True)
    generate.add_argument(
        "--variant",
        choices=("base", "format_first", "balanced", "phase89", "candidate"),
        required=True,
    )
    generate.add_argument("--clean", action="store_true")
    sub.add_parser("sanity")
    review = sub.add_parser("review-template")
    review.add_argument("--clean", action="store_true")
    sub.add_parser("simulate-user-review")
    sub.add_parser("review-validate")
    sub.add_parser("full-regression")
    sub.add_parser("finalize")
    sub.add_parser("validate")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if args.command == "prepare":
        return _prepare(args.clean)
    if args.command == "train":
        return _train(args.curriculum, args.steps, args.clean)
    if args.command == "generate":
        return _generate(args.scope, args.variant, args.clean)
    if args.command == "sanity":
        return _sanity()
    if args.command == "review-template":
        return _review_template(args.clean)
    if args.command == "simulate-user-review":
        return _simulate_user_review()
    if args.command == "review-validate":
        return _review_validate()
    if args.command == "full-regression":
        return _run_regression()
    if args.command == "finalize":
        return _finalize()
    if args.command == "validate":
        return _validate()
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
