#!/usr/bin/env python3
"""Run the Phase91 controlled progressive-DPO diagnostic locally."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = REPO_ROOT / "pfe-core"
TOOLS_ROOT = REPO_ROOT / "tools"
for path in (CORE_ROOT, TOOLS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pfe_core.adapter_store.quality import validate_adapter_artifact
from pfe_core.phase75_personalization_benefit_benchmark import stable_hash
from pfe_core.phase91_controlled_dpo_preference import (
    PHASE91_EVAL_CATEGORIES,
    PHASE91_HOLDOUT_COUNT,
    PHASE91_PREFERENCE_CATEGORIES,
    aggregate_phase91_scores,
    audit_phase91_holdout_isolation,
    audit_phase91_preference_pairs,
    build_phase91_decision,
    build_phase91_holdout,
    build_phase91_preference_pairs,
    score_phase91_output,
    select_phase91_pairs,
)
from pfe_core.trainer.executors import execute_dpo_training
from phase17_qwen_dpo_product_probe import dpo_preflight
from phase87_89_failure_driven_adapter_loop import (
    GENERATION_PROTOCOL,
    MODEL_PATH,
    MODEL_REVISION,
    TRAINER_OUTPUT_ROOT,
    _load_runtime,
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


EVIDENCE_ROOT = REPO_ROOT / "docs/demo/phase91-controlled-dpo-preference-diagnostic"
PREPARATION_ROOT = EVIDENCE_ROOT / "evidence-preparation"
TRAINING_ROOT = EVIDENCE_ROOT / "evidence-real-training"
EVAL_ROOT = EVIDENCE_ROOT / "evidence-blind-eval"
FAILURE_ROOT = EVIDENCE_ROOT / "evidence-failures"
REVIEW_CACHE_ROOT = Path("/private/tmp/pfe-phase91-simulated-user-review")
PHASE89_ROOT = REPO_ROOT / "docs/demo/phase87-89-failure-driven-adapter-loop"
PHASE90_ROOT = REPO_ROOT / "docs/demo/phase90-native-format-curriculum-repair"
PARENT_ADAPTER_ROOT = TRAINER_OUTPUT_ROOT / "phase87-failure-driven-25step/peft_lora"
DPO_BETA = 0.1
DPO_LEARNING_RATE = 5e-6
DPO_MAX_LENGTH = 384
DPO_MAX_PROMPT_LENGTH = 288
DPO_LORA = {"r": 16, "lora_alpha": 32, "lora_dropout": 0.05}
FROZEN_THRESHOLDS = {
    "candidate_gain_vs_base_minimum": 0.08,
    "candidate_native_minimum": 0.75,
    "candidate_false_block_maximum": 0.0,
    "candidate_provenance_minimum": 0.90,
    "candidate_truncation_maximum": 0.10,
    "candidate_runtime_fallback_maximum": 0.10,
    "strict_core_improvement_required": True,
    "simulated_blind_review_required": True,
    "auto_promotion_allowed": False,
}
SANITY_RULE = {
    "session_count_per_arm": 4,
    "candidate_overall_regression_tolerance_vs_phase89": 0.05,
    "ordinary_regression_tolerance": 0.10,
    "native_not_below_phase89": True,
    "false_block_not_above_phase89": True,
    "provenance_not_below_phase89": True,
    "at_least_one_strict_core_improvement": True,
    "safety_regression_allowed": False,
}


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _source_hashes() -> dict[str, str]:
    paths = {
        "core": CORE_ROOT / "pfe_core/phase91_controlled_dpo_preference.py",
        "driver": REPO_ROOT / "tools/phase91_controlled_dpo_preference_diagnostic.py",
        "core_test": REPO_ROOT / "tests/test_phase91_controlled_dpo_preference.py",
        "driver_test": REPO_ROOT / "tests/test_phase91_driver_safety.py",
        "executor": CORE_ROOT / "pfe_core/trainer/executors.py",
        "phase89_driver": REPO_ROOT / "tools/phase87_89_failure_driven_adapter_loop.py",
    }
    return {name: _sha256(path) for name, path in paths.items()}


def _parent_validation() -> dict[str, Any]:
    evidence = _read_json(
        PHASE89_ROOT / "evidence-real-training/probe-25step/adapter_validation.json"
    )
    adapter_path = PARENT_ADAPTER_ROOT / "adapter_model.safetensors"
    actual = _sha256(adapter_path) if adapter_path.is_file() else None
    return {
        "kind": "phase91_parent_adapter_validation",
        "artifact_dir": str(PARENT_ADAPTER_ROOT),
        "evidence_sha256": evidence.get("sha256"),
        "actual_sha256": actual,
        "valid": evidence.get("valid") is True
        and actual is not None
        and actual == evidence.get("sha256"),
        "lineage_contract": "base_merge_phase89_then_apply_phase91_dpo",
    }


def _render_prompt(tokenizer: Any, messages: Iterable[Mapping[str, Any]]) -> str:
    rows = [
        {"role": str(row.get("role") or ""), "content": str(row.get("content") or "")}
        for row in messages
    ]
    return str(
        tokenizer.apply_chat_template(
            rows,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    )


def _trainer_rows(pairs: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), local_files_only=True)
    rows = []
    for pair in pairs:
        rows.append({
            "sample_id": pair.get("pair_id"),
            "preference_category": pair.get("preference_category"),
            "instruction": _render_prompt(tokenizer, pair.get("prompt_messages") or []),
            "chosen": str(pair.get("chosen") or ""),
            "rejected": str(pair.get("rejected") or ""),
            "sample_type": "dpo",
            "simulated_usage": True,
            "actual_user_feedback": False,
            "derived_from_eval_output": False,
        })
    return rows


def _job_spec(rows: list[Mapping[str, Any]], output_dir: Path, steps: int) -> dict[str, Any]:
    return {
        "backend": "dpo",
        "execution_backend": "dpo",
        "execution_executor": "dpo",
        "executor_mode": "real_import",
        "dry_run": True,
        "output_dir": str(output_dir),
        "recipe": {
            "training": {
                "method": "lora",
                "epochs": 1,
                "max_steps": steps,
                "learning_rate": DPO_LEARNING_RATE,
                "train_type": "dpo",
                "base_model": str(MODEL_PATH),
                "num_train_samples": len(rows),
                "output_dir": str(output_dir),
                "incremental_context": {
                    "parent_adapter_path": str(PARENT_ADAPTER_ROOT),
                },
            },
            "peft": {
                "trainer_class": "trl.DPOTrainer",
                "dpo_config": {
                    "beta": DPO_BETA,
                    "label_smoothing": 0.0,
                    "max_length": DPO_MAX_LENGTH,
                    "max_prompt_length": DPO_MAX_PROMPT_LENGTH,
                },
                "lora_config": dict(DPO_LORA),
            },
        },
        "training_examples": [dict(row) for row in rows],
        "phase91": {
            "probe_scope": "controlled_preference_objective_diagnostic",
            "parent_adapter": "phase89_failure_driven_25step",
            "parent_adapter_sha256": _parent_validation().get("actual_sha256"),
            "parent_merge_required": True,
            "evaluation_lineage": "base_plus_merged_phase89_plus_phase91_dpo",
            "requested_optimizer_steps": steps,
            "simulated_usage": True,
            "actual_user_feedback": False,
            "automatic_promotion_allowed": False,
        },
    }


def _token_boundary_report(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), local_files_only=True)
    details = []
    for row in rows:
        prompt = str(row.get("instruction") or "")
        chosen = str(row.get("chosen") or "")
        rejected = str(row.get("rejected") or "")
        prompt_tokens = len(tokenizer(prompt, add_special_tokens=False).get("input_ids") or [])
        chosen_total = len(tokenizer(prompt + chosen, add_special_tokens=False).get("input_ids") or [])
        rejected_total = len(tokenizer(prompt + rejected, add_special_tokens=False).get("input_ids") or [])
        details.append({
            "sample_id": row.get("sample_id"),
            "preference_category": row.get("preference_category"),
            "prompt_tokens": prompt_tokens,
            "chosen_total_tokens": chosen_total,
            "rejected_total_tokens": rejected_total,
            "prompt_within_limit": prompt_tokens <= DPO_MAX_PROMPT_LENGTH,
            "chosen_within_limit": chosen_total <= DPO_MAX_LENGTH,
            "rejected_within_limit": rejected_total <= DPO_MAX_LENGTH,
        })
    checks = {
        "all_rows_present": bool(details),
        "all_prompts_within_limit": all(row["prompt_within_limit"] for row in details),
        "all_chosen_within_limit": all(row["chosen_within_limit"] for row in details),
        "all_rejected_within_limit": all(row["rejected_within_limit"] for row in details),
    }
    return {
        "kind": "phase91_dpo_token_boundary_report",
        "passed": all(checks.values()),
        "checks": checks,
        "max_prompt_length": DPO_MAX_PROMPT_LENGTH,
        "max_length": DPO_MAX_LENGTH,
        "maximum_prompt_tokens": max((row["prompt_tokens"] for row in details), default=0),
        "maximum_total_tokens": max(
            (max(row["chosen_total_tokens"], row["rejected_total_tokens"]) for row in details),
            default=0,
        ),
        "details": details,
    }


def _prepare(clean: bool) -> int:
    if clean and EVIDENCE_ROOT.exists():
        _safe_clean(EVIDENCE_ROOT, REPO_ROOT / "docs/demo")
    PREPARATION_ROOT.mkdir(parents=True, exist_ok=True)
    pairs = build_phase91_preference_pairs()
    quality = audit_phase91_preference_pairs(pairs)
    holdout = build_phase91_holdout()
    previous_holdouts = [
        _read_json(PHASE89_ROOT / "evidence-preparation/holdout.json"),
        _read_json(PHASE90_ROOT / "evidence-preparation/holdout.json"),
    ]
    isolation = audit_phase91_holdout_isolation(pairs, holdout, previous_holdouts)
    parent = _parent_validation()
    preflight = dpo_preflight()
    selected: dict[str, Any] = {}
    boundaries: dict[str, Any] = {}
    dry_runs: dict[str, Any] = {}
    for steps in (12, 30):
        selected_pairs = select_phase91_pairs(pairs, steps=steps)
        trainer_rows = _trainer_rows(selected_pairs)
        selected[str(steps)] = {
            "pair_count": len(selected_pairs),
            "category_counts": dict(sorted(Counter(
                str(row.get("preference_category") or "") for row in selected_pairs
            ).items())),
            "pair_manifest_sha256": stable_hash(selected_pairs),
            "trainer_manifest_sha256": stable_hash(trainer_rows),
        }
        _write_jsonl(PREPARATION_ROOT / f"selected_pairs_{steps}step.jsonl", selected_pairs)
        _write_jsonl(PREPARATION_ROOT / f"trainer_rows_{steps}step.jsonl", trainer_rows)
        spec = _job_spec(
            trainer_rows,
            TRAINER_OUTPUT_ROOT / f"phase91-controlled-dpo-{steps}step",
            steps,
        )
        boundary = _token_boundary_report(trainer_rows)
        dry_run = execute_dpo_training(job_spec=spec, dry_run=True)
        boundaries[str(steps)] = boundary
        dry_runs[str(steps)] = dry_run
        _write_json(PREPARATION_ROOT / f"dpo_job_spec_{steps}step.json", spec)
        _write_json(PREPARATION_ROOT / f"dpo_dry_run_{steps}step.json", dry_run)
    phase90_decision = _read_json(PHASE90_ROOT / "phase90-final-decision.json")
    checks = {
        "phase90_commit_result_remains_archive": phase90_decision.get("status")
        == "archive_phase90_native_format_not_qualified",
        "phase90_product_gate_false": phase90_decision.get("product_gate_qualified") is False,
        "preference_quality_passed": quality.get("passed") is True,
        "holdout_isolation_passed": isolation.get("passed") is True,
        "parent_adapter_valid": parent.get("valid") is True,
        "dpo_preflight_ready": preflight.get("ready") is True,
        "all_token_boundaries_passed": all(report.get("passed") is True for report in boundaries.values()),
        "all_dry_runs_prepared": all(result.get("status") == "prepared" for result in dry_runs.values()),
        "all_dry_runs_use_parent": all(
            result.get("base_adapter_path") == str(PARENT_ADAPTER_ROOT)
            for result in dry_runs.values()
        ),
        "local_model_complete": MODEL_PATH.is_dir()
        and (MODEL_PATH / "model.safetensors").is_file(),
        "pair_pool_count_72": int(pairs.get("pair_count") or 0) == 72,
        "fresh_holdout_count_40": int(holdout.get("session_count") or 0) == 40,
    }
    freeze = {
        "kind": "phase91_pre_experiment_freeze",
        "created_at": _utcnow(),
        "frozen_before_training": True,
        "passed": all(checks.values()),
        "checks": checks,
        "model_path": str(MODEL_PATH),
        "model_revision": MODEL_REVISION,
        "model_config_sha256": _sha256(MODEL_PATH / "config.json"),
        "model_weight_size_bytes": (MODEL_PATH / "model.safetensors").stat().st_size,
        "parent_adapter_sha256": parent.get("actual_sha256"),
        "pair_pool_sha256": quality.get("pair_manifest_sha256"),
        "selected_manifests": selected,
        "holdout_manifest_sha256": stable_hash(holdout.get("sessions") or []),
        "generation_protocol_sha256": stable_hash(GENERATION_PROTOCOL),
        "thresholds_sha256": stable_hash(FROZEN_THRESHOLDS),
        "sanity_rule_sha256": stable_hash(SANITY_RULE),
        "source_sha256": _source_hashes(),
        "score_or_gate_relaxation_allowed": False,
        "thirty_step_requires_passed_twelve_step_sanity": True,
        "automatic_promotion_allowed": False,
    }
    _write_json(PREPARATION_ROOT / "preference_quality_audit.json", quality)
    _write_json(PREPARATION_ROOT / "holdout.json", holdout)
    _write_json(PREPARATION_ROOT / "holdout_isolation_audit.json", isolation)
    _write_json(PREPARATION_ROOT / "parent_adapter_validation.json", parent)
    _write_json(PREPARATION_ROOT / "dpo_preflight.json", preflight)
    _write_json(PREPARATION_ROOT / "token_boundary_reports.json", boundaries)
    _write_json(PREPARATION_ROOT / "selection_manifest.json", selected)
    _write_json(EVIDENCE_ROOT / "generation_protocol.json", GENERATION_PROTOCOL)
    _write_json(EVIDENCE_ROOT / "frozen_thresholds.json", FROZEN_THRESHOLDS)
    _write_json(EVIDENCE_ROOT / "sanity_rule.json", SANITY_RULE)
    _write_json(EVIDENCE_ROOT / "pre_experiment_freeze.json", freeze)
    _write_json(EVIDENCE_ROOT / "preparation_decision.json", {
        "kind": "phase91_preparation_decision",
        "status": "ready_for_12_step_dpo" if freeze["passed"] else "blocked",
        "checks": checks,
        "automatic_training_started": False,
        "product_gate_qualified": False,
    })
    print(json.dumps({
        "status": "ready_for_12_step_dpo" if freeze["passed"] else "blocked",
        "checks": checks,
    }, ensure_ascii=False, indent=2))
    return 0 if freeze["passed"] else 2


def _training_freeze_check(steps: int) -> dict[str, Any]:
    freeze = _read_json(EVIDENCE_ROOT / "pre_experiment_freeze.json")
    rows = _read_jsonl(PREPARATION_ROOT / f"trainer_rows_{steps}step.jsonl")
    selection = dict(freeze.get("selected_manifests") or {}).get(str(steps), {})
    holdout = _read_json(PREPARATION_ROOT / "holdout.json")
    sanity = _read_json(EVIDENCE_ROOT / "sanity_decision.json") if (
        EVIDENCE_ROOT / "sanity_decision.json"
    ).is_file() else {}
    checks = {
        "pre_experiment_freeze_passed": freeze.get("passed") is True,
        "trainer_rows_unchanged": stable_hash(rows) == selection.get("trainer_manifest_sha256"),
        "holdout_unchanged": stable_hash(holdout.get("sessions") or [])
        == freeze.get("holdout_manifest_sha256"),
        "model_config_unchanged": _sha256(MODEL_PATH / "config.json")
        == freeze.get("model_config_sha256"),
        "model_weight_size_unchanged": (MODEL_PATH / "model.safetensors").stat().st_size
        == int(freeze.get("model_weight_size_bytes") or 0),
        "parent_adapter_unchanged": _parent_validation().get("actual_sha256")
        == freeze.get("parent_adapter_sha256"),
        "source_files_unchanged": _source_hashes() == freeze.get("source_sha256"),
        "thresholds_unchanged": stable_hash(FROZEN_THRESHOLDS)
        == freeze.get("thresholds_sha256"),
        "sanity_rule_unchanged": stable_hash(SANITY_RULE) == freeze.get("sanity_rule_sha256"),
        "thirty_step_requires_passed_sanity": steps != 30 or sanity.get("passed") is True,
    }
    return {
        "kind": "phase91_training_freeze_check",
        "steps": steps,
        "passed": all(checks.values()),
        "checks": checks,
    }


def _train(steps: int, clean: bool) -> int:
    if steps not in (12, 30):
        raise SystemExit("Phase91 permits only 12-step and 30-step DPO probes")
    freeze = _training_freeze_check(steps)
    evidence_dir = TRAINING_ROOT / f"probe-{steps}step"
    output_dir = TRAINER_OUTPUT_ROOT / f"phase91-controlled-dpo-{steps}step"
    if clean and evidence_dir.exists():
        _safe_clean(evidence_dir, TRAINING_ROOT)
    if clean and output_dir.exists():
        _safe_clean(output_dir, TRAINER_OUTPUT_ROOT)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_jsonl(PREPARATION_ROOT / f"trainer_rows_{steps}step.jsonl")
    spec = _job_spec(rows, output_dir, steps)
    _write_json(evidence_dir / "training_freeze_check.json", freeze)
    _write_json(evidence_dir / "dpo_job_spec.json", spec)
    if not freeze["passed"]:
        attempt = {
            "kind": "phase91_dpo_training_attempt",
            "status": "blocked",
            "real_training": False,
            "requested_steps": steps,
            "reason": "training_freeze_failed",
            "product_gate_qualified": False,
        }
        _write_json(evidence_dir / "training_attempt.json", attempt)
        return 2
    started = time.perf_counter()
    result = execute_dpo_training(
        job_spec={**spec, "dry_run": False}, dry_run=False
    )
    real = dict(result.get("real_execution") or {})
    artifact_dir = Path(str(real.get("artifact_dir") or ""))
    adapter_path = artifact_dir / "adapter_model.safetensors"
    validation = validate_adapter_artifact(
        artifact_dir,
        {"artifact_name": "adapter_model.safetensors", "artifact_format": "peft_lora"},
    ) if artifact_dir.is_dir() else {"valid": False, "reason": "artifact_dir_missing"}
    validation.update({
        "artifact_dir": str(artifact_dir),
        "adapter_path": str(adapter_path),
        "sha256": _sha256(adapter_path) if adapter_path.is_file() else None,
        "requested_steps": steps,
        "completed_steps": real.get("steps"),
        "parent_adapter_sha256": _parent_validation().get("actual_sha256"),
        "lineage_contract": "base_merge_phase89_then_apply_phase91_dpo",
    })
    completed = (
        result.get("status") == "completed"
        and real.get("success") is True
        and real.get("parameters_updated") is True
        and int(real.get("steps") or 0) == steps
        and validation.get("valid") is True
    )
    attempt = {
        "kind": "phase91_dpo_training_attempt",
        "status": "completed" if completed else "failed",
        "real_training": completed,
        "requested_steps": steps,
        "completed_steps": int(real.get("steps") or 0),
        "duration_seconds": round(time.perf_counter() - started, 4),
        "selected_model": "Qwen2.5-1.5B-Instruct",
        "parent_adapter": str(PARENT_ADAPTER_ROOT),
        "parent_adapter_sha256": _parent_validation().get("actual_sha256"),
        "learning_rate": DPO_LEARNING_RATE,
        "beta": DPO_BETA,
        "result": result,
        "adapter_validation": validation,
        "simulated_usage": True,
        "actual_user_feedback": False,
        "actual_product_benefit_claim_allowed": False,
        "product_gate_qualified": False,
        "auto_promotion_allowed": False,
    }
    if not completed:
        FAILURE_ROOT.mkdir(parents=True, exist_ok=True)
        _write_json(FAILURE_ROOT / f"training_{steps}step.json", attempt)
    _write_json(evidence_dir / "training_attempt.json", attempt)
    _write_json(evidence_dir / "adapter_validation.json", validation)
    _write_json(evidence_dir / "train_log.json", {
        "status": attempt["status"],
        "requested_steps": steps,
        "completed_steps": attempt["completed_steps"],
        "train_loss": result.get("train_loss"),
        "loss_history": real.get("loss_history") or [],
        "parameters_updated": real.get("parameters_updated"),
        "parameter_fingerprint_before": real.get("parameter_fingerprint_before"),
        "parameter_fingerprint_after": real.get("parameter_fingerprint_after"),
    })
    print(json.dumps({
        "status": attempt["status"],
        "requested_steps": steps,
        "completed_steps": attempt["completed_steps"],
        "duration_seconds": attempt["duration_seconds"],
        "error": result.get("error"),
    }, ensure_ascii=False, indent=2))
    return 0 if completed else 1


def _dpo_adapter_dir(steps: int) -> Path:
    attempt = _read_json(TRAINING_ROOT / f"probe-{steps}step/training_attempt.json")
    validation = dict(attempt.get("adapter_validation") or {})
    artifact_dir = Path(str(validation.get("artifact_dir") or ""))
    if attempt.get("status") != "completed" or validation.get("valid") is not True:
        raise SystemExit(f"Phase91 {steps}-step DPO adapter unavailable")
    return artifact_dir.resolve()


def _load_candidate_runtime(adapter_path: Path) -> tuple[Any, Any, Any, str]:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device == "mps" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH), local_files_only=True, low_cpu_mem_usage=True, dtype=dtype
    )
    model = PeftModel.from_pretrained(model, str(PARENT_ADAPTER_ROOT), local_files_only=True)
    model = model.merge_and_unload()
    model = PeftModel.from_pretrained(model, str(adapter_path), local_files_only=True)
    model.to(device)
    model.eval()
    return torch, tokenizer, model, device


def _scope_sessions(scope: str) -> list[dict[str, Any]]:
    sessions = [
        dict(row)
        for row in _read_json(PREPARATION_ROOT / "holdout.json").get("sessions") or []
    ]
    if scope == "full":
        return sessions
    return [next(row for row in sessions if row.get("category") == category) for category in PHASE91_EVAL_CATEGORIES]


def _generation_freeze_check(scope: str, variant: str, adapter_path: Path | None) -> dict[str, Any]:
    freeze = _read_json(EVIDENCE_ROOT / "pre_experiment_freeze.json")
    holdout = _read_json(PREPARATION_ROOT / "holdout.json")
    sanity = _read_json(EVIDENCE_ROOT / "sanity_decision.json") if (
        EVIDENCE_ROOT / "sanity_decision.json"
    ).is_file() else {}
    checks = {
        "pre_experiment_freeze_passed": freeze.get("passed") is True,
        "holdout_unchanged": stable_hash(holdout.get("sessions") or [])
        == freeze.get("holdout_manifest_sha256"),
        "source_files_unchanged": _source_hashes() == freeze.get("source_sha256"),
        "parent_adapter_unchanged": _parent_validation().get("actual_sha256")
        == freeze.get("parent_adapter_sha256"),
        "adapter_available_or_base": adapter_path is None or adapter_path.is_dir(),
        "full_requires_passed_sanity": scope != "full" or sanity.get("passed") is True,
    }
    return {
        "kind": "phase91_generation_freeze_check",
        "scope": scope,
        "variant": variant,
        "passed": all(checks.values()),
        "checks": checks,
    }


def _generate(scope: str, variant: str, clean: bool) -> int:
    if scope not in {"sanity", "full"} or variant not in {"base", "phase89", "candidate"}:
        raise SystemExit("unsupported Phase91 generation arm")
    steps = 12 if scope == "sanity" else 30
    adapter_path = None
    if variant == "phase89":
        adapter_path = PARENT_ADAPTER_ROOT
    elif variant == "candidate":
        adapter_path = _dpo_adapter_dir(steps)
    freeze = _generation_freeze_check(scope, variant, adapter_path)
    if not freeze["passed"]:
        raise SystemExit(f"Phase91 generation freeze failed: {freeze}")
    root = EVAL_ROOT / scope
    structural_path = root / f"structural_sessions_{variant}.jsonl"
    metrics_path = root / f"metrics_{variant}.json"
    cache_path = REVIEW_CACHE_ROOT / f"{scope}_{variant}.jsonl"
    if clean:
        structural_path.unlink(missing_ok=True)
        metrics_path.unlink(missing_ok=True)
        cache_path.unlink(missing_ok=True)
    rows = []
    private_rows = []
    sessions = _scope_sessions(scope)
    torch = tokenizer = model = device = None
    try:
        if variant == "candidate":
            torch, tokenizer, model, device = _load_candidate_runtime(adapter_path)
        else:
            torch, tokenizer, model, device = _load_runtime(adapter_path)
        for index, session in enumerate(sessions, start=1):
            try:
                structural, private = _run_eval_session(
                    session=session,
                    torch=torch,
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                    adapter_loaded=variant != "base",
                )
                structural.update({
                    "kind": "phase91_structural_eval_session",
                    "variant": variant,
                    "lineage": (
                        "base_plus_merged_phase89_plus_phase91_dpo"
                        if variant == "candidate"
                        else ("base_plus_phase89" if variant == "phase89" else "base")
                    ),
                    "raw_score": score_phase91_output(private["raw_output"], session),
                    "post_score": score_phase91_output(private["post_output"], session),
                })
            except Exception as exc:
                structural = {
                    "kind": "phase91_structural_eval_session",
                    "session_id": session.get("session_id"),
                    "category": session.get("category"),
                    "variant": variant,
                    "status": "failed",
                    "actual_model_call": False,
                    "error_type": exc.__class__.__name__,
                    "raw_model_output_persisted": False,
                    "simulated_usage": True,
                    "actual_user_feedback": False,
                }
                private = {
                    "session_id": session.get("session_id"),
                    "category": session.get("category"),
                    "error_type": exc.__class__.__name__,
                }
            rows.append(structural)
            private_rows.append(private)
            _write_jsonl(structural_path, rows)
            _write_private_jsonl(cache_path, private_rows)
            print(
                f"[{scope}:{variant}] {index}/{len(sessions)} {session.get('session_id')} "
                f"{structural['status']}",
                flush=True,
            )
    finally:
        if torch is not None and model is not None and device is not None:
            _release_runtime(torch, model, device)
    completed = [row for row in rows if row.get("status") == "completed"]
    raw = aggregate_phase91_scores({
        "category": row.get("category"),
        "score": row.get("raw_score"),
        "truncated": row.get("truncated"),
    } for row in completed)
    post = aggregate_phase91_scores({
        "category": row.get("category"),
        "score": row.get("post_score"),
        "truncated": row.get("truncated"),
    } for row in completed)
    fallback_count = sum(row.get("final_fallback_used") is True for row in completed)
    post.update({
        "fallback_count": fallback_count,
        "fallback_rate": round(fallback_count / len(completed), 4) if completed else 0.0,
    })
    metrics = {
        "kind": "phase91_variant_metrics",
        "scope": scope,
        "variant": variant,
        "session_count": len(completed),
        "model_call_count": sum(int(row.get("turn_count") or 0) for row in completed),
        "all_sessions_completed": len(completed) == len(sessions),
        "actual_model_calls": len(completed) == len(sessions),
        "raw": raw,
        "post_contract": post,
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
        "session_count": metrics["session_count"],
        "raw_overall": raw.get("overall_score"),
        "raw_native": raw.get("native_format_rate"),
        "raw_false_block": raw.get("false_block_rate"),
        "raw_provenance": raw.get("provenance_correct_rate"),
        "fallback": post.get("fallback_rate"),
    }, ensure_ascii=False, indent=2))
    return 0 if metrics["all_sessions_completed"] else 1


def _strict_core_improvement(phase89: Mapping[str, Any], candidate: Mapping[str, Any]) -> bool:
    return (
        float(candidate.get("native_format_rate") or 0.0)
        > float(phase89.get("native_format_rate") or 0.0)
        or float(candidate.get("false_block_rate") or 0.0)
        < float(phase89.get("false_block_rate") or 0.0)
        or float(candidate.get("provenance_correct_rate") or 0.0)
        > float(phase89.get("provenance_correct_rate") or 0.0)
    )


def _sanity() -> int:
    metrics = {
        variant: _read_json(EVAL_ROOT / f"sanity/metrics_{variant}.json")
        for variant in ("base", "phase89", "candidate")
    }
    phase89 = dict(metrics["phase89"].get("raw") or {})
    candidate = dict(metrics["candidate"].get("raw") or {})
    phase89_ordinary = float(dict(dict(phase89.get("category_metrics") or {}).get("ordinary_control") or {}).get("composite_score") or 0.0)
    candidate_ordinary = float(dict(dict(candidate.get("category_metrics") or {}).get("ordinary_control") or {}).get("composite_score") or 0.0)
    checks = {
        "all_three_arms_completed_four_sessions": all(
            payload.get("session_count") == 4 and payload.get("model_call_count") == 12
            for payload in metrics.values()
        ),
        "real_12_step_dpo_completed": _read_json(
            TRAINING_ROOT / "probe-12step/training_attempt.json"
        ).get("status")
        == "completed",
        "candidate_overall_within_tolerance": float(candidate.get("overall_score") or 0.0)
        >= float(phase89.get("overall_score") or 0.0) - 0.05,
        "candidate_native_not_below_phase89": float(candidate.get("native_format_rate") or 0.0)
        >= float(phase89.get("native_format_rate") or 0.0),
        "candidate_false_block_not_above_phase89": float(candidate.get("false_block_rate") or 0.0)
        <= float(phase89.get("false_block_rate") or 0.0),
        "candidate_provenance_not_below_phase89": float(candidate.get("provenance_correct_rate") or 0.0)
        >= float(phase89.get("provenance_correct_rate") or 0.0),
        "candidate_has_strict_core_improvement": _strict_core_improvement(phase89, candidate),
        "ordinary_regression_within_0_10": phase89_ordinary - candidate_ordinary <= 0.10,
        "candidate_unsupported_zero": float(candidate.get("unsupported_assertion_rate") or 0.0) == 0.0,
        "candidate_think_leak_zero": float(candidate.get("think_leak_rate") or 0.0) == 0.0,
        "candidate_privacy_echo_zero": float(candidate.get("privacy_echo_rate") or 0.0) == 0.0,
    }
    payload = {
        "kind": "phase91_12step_sanity_decision",
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": {name: value.get("raw") for name, value in metrics.items()},
        "next_action": "run_30_step_dpo" if all(checks.values()) else "archive_12step_sanity_failure",
        "product_gate_qualified": False,
        "promotion_allowed": False,
        "auto_promotion_allowed": False,
    }
    _write_json(EVIDENCE_ROOT / "sanity_decision.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["passed"] else 1


def _findings(output: str, session: Mapping[str, Any]) -> list[str]:
    score = score_phase91_output(output, session)
    findings = []
    if score.get("native_format") is not True:
        findings.append("format_failure")
    if score.get("false_block") is True:
        findings.append("false_block")
    if session.get("provenance_rejection_expected") is True and score.get("provenance_correct") is not True:
        findings.append("provenance_failure")
    if score.get("unsupported_assertion") is True:
        findings.append("unsupported_assertion")
    if score.get("category_correct") is not True and not findings:
        findings.append("other_semantic_failure")
    return findings


def _review(clean: bool) -> int:
    base_rows = {str(row.get("session_id")): row for row in _read_jsonl(REVIEW_CACHE_ROOT / "full_phase89.jsonl")}
    candidate_rows = {str(row.get("session_id")): row for row in _read_jsonl(REVIEW_CACHE_ROOT / "full_candidate.jsonl")}
    sessions = {str(row.get("session_id")): row for row in _read_json(PREPARATION_ROOT / "holdout.json").get("sessions") or []}
    ids = sorted(base_rows)
    if ids != sorted(candidate_rows) or len(ids) != PHASE91_HOLDOUT_COUNT:
        raise SystemExit("Phase91 full review caches incomplete")
    private_pairs = []
    public_pairs = []
    variant_key = {}
    decisions = []
    for session_id in ids:
        base_row = base_rows[session_id]
        candidate_row = candidate_rows[session_id]
        phase89_is_a = int(hashlib.sha256(session_id.encode()).hexdigest(), 16) % 2 == 0
        a = base_row if phase89_is_a else candidate_row
        b = candidate_row if phase89_is_a else base_row
        pair_id = f"phase91-pair-{hashlib.sha256(session_id.encode()).hexdigest()[:12]}"
        private_pairs.append({
            "pair_id": pair_id,
            "session_id": session_id,
            "candidate_a_output": a.get("raw_output"),
            "candidate_b_output": b.get("raw_output"),
        })
        public_pairs.append({
            "pair_id": pair_id,
            "session_id": session_id,
            "category": sessions[session_id].get("category"),
            "candidate_a_output_sha256": a.get("raw_output_sha256"),
            "candidate_b_output_sha256": b.get("raw_output_sha256"),
        })
        variant_key[pair_id] = {
            "candidate_a": "phase89" if phase89_is_a else "candidate",
            "candidate_b": "candidate" if phase89_is_a else "phase89",
        }
        findings_a = _findings(str(a.get("raw_output") or ""), sessions[session_id])
        findings_b = _findings(str(b.get("raw_output") or ""), sessions[session_id])
        penalty_a = len(findings_a)
        penalty_b = len(findings_b)
        winner = "tie" if penalty_a == penalty_b else ("candidate_a" if penalty_a < penalty_b else "candidate_b")
        decisions.append({
            "pair_id": pair_id,
            "candidate_a_output_sha256": a.get("raw_output_sha256"),
            "candidate_b_output_sha256": b.get("raw_output_sha256"),
            "winner": winner,
            "findings_a": findings_a,
            "findings_b": findings_b,
        })
    if clean:
        (REVIEW_CACHE_ROOT / "blind_pairs.jsonl").unlink(missing_ok=True)
    _write_private_jsonl(REVIEW_CACHE_ROOT / "blind_pairs.jsonl", private_pairs)
    candidate_wins = phase89_wins = ties = candidate_findings = 0
    finding_counts: Counter[str] = Counter()
    for decision in decisions:
        key = variant_key[decision["pair_id"]]
        winner = decision["winner"]
        if winner == "tie":
            ties += 1
        elif key[winner] == "candidate":
            candidate_wins += 1
        else:
            phase89_wins += 1
        for label, values in (("candidate_a", decision["findings_a"]), ("candidate_b", decision["findings_b"])):
            finding_counts.update(values)
            if key[label] == "candidate":
                candidate_findings += len(values)
    complete = len(decisions) == PHASE91_HOLDOUT_COUNT
    passed = complete and candidate_wins > phase89_wins and candidate_findings == 0
    review = {
        "kind": "phase91_blind_simulated_user_review",
        "complete": complete,
        "passed": passed,
        "reviewed_pair_count": len(decisions),
        "reviewer_ids": ["codex_simulated_user_phase91"],
        "pairs": public_pairs,
        "decisions": decisions,
        "candidate_wins": candidate_wins,
        "phase89_wins": phase89_wins,
        "ties": ties,
        "candidate_finding_count": candidate_findings,
        "finding_counts": dict(sorted(finding_counts.items())),
        "review_is_simulated_not_actual_user_feedback": True,
        "raw_output_persisted_in_evidence": False,
        "actual_user_feedback_count": 0,
        "product_gate_qualified": False,
        "promotion_allowed": False,
    }
    _write_json(EVAL_ROOT / "simulated_user_review.json", review)
    _write_json(EVAL_ROOT / "blind_variant_key.json", {
        "kind": "phase91_blind_variant_key",
        "mapping": variant_key,
        "contains_raw_output": False,
    })
    print(json.dumps({key: review[key] for key in (
        "complete", "passed", "candidate_wins", "phase89_wins", "ties", "candidate_finding_count"
    )}, ensure_ascii=False, indent=2))
    return 0


def _walk_forbidden(value: Any, path: str = "$") -> list[str]:
    failures = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if key in {"raw_output", "candidate_a_output", "candidate_b_output"}:
                failures.append(child_path)
            failures.extend(_walk_forbidden(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            failures.extend(_walk_forbidden(child, f"{path}[{index}]"))
    return failures


def _write_runbook(decision: Mapping[str, Any]) -> None:
    sanity = _read_json(EVIDENCE_ROOT / "sanity_decision.json")
    text = "# Phase91 Controlled DPO Preference Diagnostic\n\n"
    text += "- Model: local Qwen2.5-1.5B-Instruct\n"
    text += "- Parent: Phase89 25-step SFT adapter\n"
    text += "- Data: simulated_usage only; actual user feedback count 0\n"
    text += f"- 12-step sanity passed: `{str(sanity.get('passed')).lower()}`\n"
    text += f"- Final status: `{decision.get('status')}`\n"
    text += "- Automatic promotion, deployment, Hermes, external Provider: forbidden\n"
    (EVIDENCE_ROOT / "phase91-runbook.md").write_text(text, encoding="utf-8")


def _manifest_and_cleanup() -> None:
    if REVIEW_CACHE_ROOT.exists():
        shutil.rmtree(REVIEW_CACHE_ROOT)
    excluded = {"evidence_manifest.json", "validation_summary.json"}
    rows = []
    for path in sorted(EVIDENCE_ROOT.rglob("*")):
        if path.is_file() and path.name not in excluded:
            rows.append({
                "path": str(path.relative_to(REPO_ROOT)),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            })
    _write_json(EVIDENCE_ROOT / "evidence_manifest.json", {
        "kind": "phase91_evidence_manifest",
        "file_count": len(rows),
        "files": rows,
    })


def _finalize() -> int:
    sanity = _read_json(EVIDENCE_ROOT / "sanity_decision.json")
    if sanity.get("passed") is not True:
        decision = {
            "kind": "phase91_controlled_dpo_decision",
            "status": "archive_phase91_12step_sanity_failed",
            "recommendation": "archive_and_move_to_larger_model",
            "sanity_checks": sanity.get("checks"),
            "thirty_step_training_run": False,
            "full_eval_run": False,
            "product_gate_qualified": False,
            "promotion_allowed": False,
            "auto_promotion_allowed": False,
            "automatic_deployment_allowed": False,
            "hermes_attachment_allowed": False,
            "actual_product_benefit_claim_allowed": False,
            "simulated_usage": True,
            "actual_user_feedback_count": 0,
        }
        comparison = {
            "kind": "phase91_12step_sanity_comparison",
            "scope": "sanity_only",
            "metrics": sanity.get("metrics"),
            "decision_status": decision["status"],
            "product_gate_qualified": False,
        }
    else:
        metrics = {
            variant: _read_json(EVAL_ROOT / f"full/metrics_{variant}.json")
            for variant in ("base", "phase89", "candidate")
        }
        review = _read_json(EVAL_ROOT / "simulated_user_review.json")
        training = _read_json(TRAINING_ROOT / "probe-30step/training_attempt.json")
        isolation = _read_json(PREPARATION_ROOT / "holdout_isolation_audit.json")
        decision = build_phase91_decision(
            base=dict(metrics["base"].get("raw") or {}),
            phase89=dict(metrics["phase89"].get("raw") or {}),
            candidate=dict(metrics["candidate"].get("raw") or {}),
            training_attempt=training,
            isolation_audit=isolation,
            review=review,
        )
        comparison = {
            "kind": "phase91_three_arm_comparison",
            "base": metrics["base"],
            "phase89": metrics["phase89"],
            "candidate": metrics["candidate"],
            "simulated_user_review": review,
            "decision_status": decision["status"],
            "product_gate_qualified": decision["product_gate_qualified"],
        }
    _write_json(EVIDENCE_ROOT / "comparison_summary.json", comparison)
    _write_json(EVIDENCE_ROOT / "phase91-final-decision.json", decision)
    public_failures = []
    for path in sorted(EVIDENCE_ROOT.rglob("*.json")):
        public_failures.extend(
            f"{path.relative_to(REPO_ROOT)}:{item}"
            for item in _walk_forbidden(_read_json(path))
        )
    _write_json(EVIDENCE_ROOT / "public_private_audit.json", {
        "kind": "phase91_public_private_audit",
        "passed": not public_failures,
        "forbidden_raw_output_paths": public_failures,
        "raw_output_persisted_in_evidence": False,
        "simulated_usage": True,
        "actual_user_feedback_count": 0,
    })
    _write_runbook(decision)
    _manifest_and_cleanup()
    print(json.dumps({
        "status": decision["status"],
        "recommendation": decision["recommendation"],
        "product_gate_qualified": decision["product_gate_qualified"],
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
    decision = _read_json(EVIDENCE_ROOT / "phase91-final-decision.json")
    public = _read_json(EVIDENCE_ROOT / "public_private_audit.json")
    checks = {
        "manifest_files_unchanged": not failures,
        "public_private_audit_passed": public.get("passed") is True,
        "review_cache_absent": not REVIEW_CACHE_ROOT.exists(),
        "no_auto_promotion": decision.get("auto_promotion_allowed") is False,
        "no_automatic_deployment": decision.get("automatic_deployment_allowed") is False,
        "no_hermes_attachment": decision.get("hermes_attachment_allowed") is False,
        "no_actual_product_benefit_claim": decision.get("actual_product_benefit_claim_allowed") is False,
    }
    summary = {
        "kind": "phase91_validation_summary",
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


def _regression() -> int:
    command = [
        str(REPO_ROOT / ".venv/bin/pytest"), "-q",
        "tests/test_phase91_controlled_dpo_preference.py",
        "tests/test_phase91_driver_safety.py",
        "tests/test_phase90_native_format_curriculum.py",
        "tests/test_phase90_driver_safety.py",
        "tests/test_phase87_failure_driven_training.py",
        "tests/test_phase87_89_driver_safety.py",
    ]
    completed = subprocess.run(command, cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    output = completed.stdout or ""
    _write_json(EVIDENCE_ROOT / "regression_summary.json", {
        "kind": "phase91_regression_summary",
        "passed": completed.returncode == 0,
        "exit_code": completed.returncode,
        "output_line_count": len(output.splitlines()),
        "output_sha256": hashlib.sha256(output.encode()).hexdigest(),
        "raw_process_output_persisted": False,
    })
    return completed.returncode


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--clean", action="store_true")
    train = sub.add_parser("train")
    train.add_argument("--steps", type=int, choices=(12, 30), required=True)
    train.add_argument("--clean", action="store_true")
    generate = sub.add_parser("generate")
    generate.add_argument("--scope", choices=("sanity", "full"), required=True)
    generate.add_argument("--variant", choices=("base", "phase89", "candidate"), required=True)
    generate.add_argument("--clean", action="store_true")
    sub.add_parser("sanity")
    review = sub.add_parser("review")
    review.add_argument("--clean", action="store_true")
    sub.add_parser("full-regression")
    sub.add_parser("finalize")
    sub.add_parser("validate")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "prepare":
        return _prepare(args.clean)
    if args.command == "train":
        return _train(args.steps, args.clean)
    if args.command == "generate":
        return _generate(args.scope, args.variant, args.clean)
    if args.command == "sanity":
        return _sanity()
    if args.command == "review":
        return _review(args.clean)
    if args.command == "full-regression":
        return _regression()
    if args.command == "finalize":
        return _finalize()
    if args.command == "validate":
        return _validate()
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
