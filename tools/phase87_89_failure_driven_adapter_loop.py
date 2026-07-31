#!/usr/bin/env python3
"""Run the Phase87-89 failure-driven local adapter loop."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import shutil
import subprocess
import sys
import time
import traceback
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = REPO_ROOT / "pfe-core"
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

from pfe_core.adapter_store.quality import validate_adapter_artifact
from pfe_core.phase75_personalization_benefit_benchmark import stable_hash
from pfe_core.phase77_private_value_guarded_runtime import (
    guard_phase77_messages,
    guard_phase77_output,
)
from pfe_core.phase78_persona_internalization_training import build_phase78_sft_job_spec
from pfe_core.phase87_failure_driven_training import (
    PHASE87_CATEGORIES,
    PHASE89_HOLDOUT_COUNT,
    aggregate_phase89_scores,
    audit_phase87_holdout_isolation,
    audit_phase87_training_candidates,
    build_phase87_failure_taxonomy,
    build_phase87_training_candidates,
    build_phase89_holdout,
    build_phase89_decision,
    score_phase89_output,
)
from pfe_core.phase85_low_fallback_semantic_guard import (
    contract_for_phase85_messages,
    enforce_phase85_persona_output,
)
from pfe_core.trainer.executors import (
    _build_sft_prompt_and_text,
    _encode_sft_examples,
    _encode_training_text,
    _run_real_local_peft_training,
)


EVIDENCE_ROOT = REPO_ROOT / "docs/demo/phase87-89-failure-driven-adapter-loop"
PREPARATION_ROOT = EVIDENCE_ROOT / "evidence-preparation"
TRAINING_ROOT = EVIDENCE_ROOT / "evidence-real-training"
EVAL_ROOT = EVIDENCE_ROOT / "evidence-blind-eval"
FAILURE_ROOT = EVIDENCE_ROOT / "evidence-failures"
REVIEW_CACHE_ROOT = Path("/private/tmp/pfe-phase89-manual-review")
PHASE85_ROOT = REPO_ROOT / "docs/demo/phase85-low-fallback-semantic-guard"
PHASE85_OVERLAY_ROOT = REPO_ROOT / "docs/demo/phase85-metric-schema-v2-overlay"
MODEL_PATH = REPO_ROOT / "models/Qwen2.5-1.5B-Instruct"
TRAINER_OUTPUT_ROOT = REPO_ROOT / "trainer_job_outputs"
TRAINING_LEARNING_RATE = 2e-5
TRAINING_MAX_LENGTH = 224
MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
GENERATION_PROTOCOL = {
    "kind": "phase89_same_prompt_base_adapter_protocol",
    "input_max_length": 3072,
    "max_new_tokens": 160,
    "do_sample": False,
    "repetition_penalty": 1.15,
    "no_repeat_ngram_size": 4,
    "enable_thinking": False,
    "three_user_turns_per_session": True,
    "same_prompt_and_decoding_for_base_adapter": True,
    "raw_and_post_contract_scores_separate": True,
    "score_or_gate_relaxation_allowed": False,
}
FROZEN_THRESHOLDS = {
    "adapter_raw_gain_minimum": 0.05,
    "adapter_runtime_fallback_maximum": 0.10,
    "adapter_false_block_maximum": 0.0,
    "adapter_unsupported_assertion_maximum": 0.0,
    "adapter_think_leak_maximum": 0.0,
    "manual_review_required": True,
    "auto_promotion_allowed": False,
}


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"expected JSONL objects: {path}")
        rows.append(payload)
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_clean(path: Path, allowed_root: Path) -> None:
    resolved = path.resolve(strict=False)
    root = allowed_root.resolve(strict=False)
    if resolved == root or root not in resolved.parents:
        raise ValueError(f"refusing unsafe clean: {path}")
    if path.is_symlink():
        raise ValueError(f"refusing symlink clean: {path}")
    shutil.rmtree(path, ignore_errors=False)


def _model_complete() -> bool:
    required = (
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
    )
    return MODEL_PATH.is_dir() and all((MODEL_PATH / name).is_file() for name in required)


def _job_spec(samples: Iterable[Mapping[str, Any]], output_dir: Path, steps: int) -> dict[str, Any]:
    spec = build_phase78_sft_job_spec(
        samples=samples,
        base_model=str(MODEL_PATH),
        output_dir=str(output_dir),
        max_steps=steps,
        learning_rate=TRAINING_LEARNING_RATE,
        seed=87,
    )
    spec["recipe"]["training"]["max_length"] = TRAINING_MAX_LENGTH
    spec["phase78"]["target_model"] = "Qwen2.5-1.5B-Instruct"
    spec["phase87"] = {
        "target_model": "Qwen2.5-1.5B-Instruct",
        "model_revision": MODEL_REVISION,
        "failure_driven": True,
        "completion_only_loss_required": True,
        "simulated_usage": True,
        "actual_user_feedback": False,
        "automatic_training_allowed": False,
        "auto_promotion_allowed": False,
    }
    return spec


def _select_probe_samples(samples: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = [dict(row) for row in samples]
    selected = []
    for category in PHASE87_CATEGORIES:
        category_rows = sorted(
            (row for row in rows if row.get("taxonomy_dimension") == category),
            key=lambda row: str(row.get("sample_id") or ""),
        )
        selected.extend(category_rows[:5])
    return selected


def _completion_boundary_report(spec: Mapping[str, Any]) -> dict[str, Any]:
    tokenizer = None
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), local_files_only=True)
    except ModuleNotFoundError:
        tokenizer = None
    training = dict(dict(spec.get("recipe") or {}).get("training") or {})
    maximum = int(training.get("max_length") or TRAINING_MAX_LENGTH)
    examples = [dict(row) for row in spec.get("training_examples") or []]
    vocab_size = int(getattr(tokenizer, "vocab_size", 0) or 151936)
    encoded = _encode_sft_examples(
        tokenizer=tokenizer,
        training_examples=examples,
        max_length=maximum,
        vocab_size=vocab_size,
    )
    details = []
    for source, row in zip(examples, encoded):
        prompt, full_text = _build_sft_prompt_and_text(
            tokenizer,
            str(source.get("instruction") or ""),
            str(source.get("chosen") or ""),
            messages=source.get("messages"),
        )
        if tokenizer is None:
            full_token_count = len(
                _encode_training_text(
                    full_text,
                    max_length=max(maximum, len(full_text) + 1),
                    vocab_size=vocab_size,
                )
            )
            prompt_tokens = _encode_training_text(
                prompt,
                max_length=maximum,
                vocab_size=vocab_size,
            )
            prompt_boundary = min(
                max(0, len(prompt_tokens) - 1),
                len(row.get("labels") or []),
            )
        else:
            full_token_count = len(
                tokenizer(full_text, add_special_tokens=False).get("input_ids") or []
            )
            prompt_tokens = tokenizer(
                prompt,
                truncation=True,
                max_length=maximum,
                add_special_tokens=False,
            ).get("input_ids") or []
            prompt_boundary = min(len(prompt_tokens), len(row.get("labels") or []))
        labels = list(row.get("labels") or [])
        completion = [index for index, value in enumerate(labels) if int(value) != -100]
        details.append({
            "sample_id": source.get("sample_id"),
            "taxonomy_dimension": source.get("taxonomy_dimension"),
            "full_token_count": full_token_count,
            "truncated": full_token_count > maximum,
            "prompt_token_count": prompt_boundary,
            "completion_label_token_count": len(completion),
            "prompt_labels_all_masked": all(
                int(labels[index]) == -100 for index in range(prompt_boundary)
            ),
            "completion_begins_at_or_after_prompt": bool(completion)
            and min(completion) >= prompt_boundary,
        })
    checks = {
        "all_samples_encoded": bool(examples) and len(encoded) == len(examples),
        "no_training_sample_truncated": not any(row["truncated"] for row in details),
        "all_prompt_labels_masked": all(row["prompt_labels_all_masked"] for row in details),
        "all_completions_after_prompt": all(
            row["completion_begins_at_or_after_prompt"] for row in details
        ),
        "minimum_completion_tokens_at_least_4": min(
            (row["completion_label_token_count"] for row in details),
            default=0,
        )
        >= 4,
    }
    return {
        "kind": "phase87_completion_only_boundary_report",
        "passed": all(checks.values()),
        "checks": checks,
        "source_sample_count": len(examples),
        "encoded_sample_count": len(encoded),
        "max_length": maximum,
        "maximum_full_token_count": max(
            (row["full_token_count"] for row in details), default=0
        ),
        "minimum_completion_label_token_count": min(
            (row["completion_label_token_count"] for row in details), default=0
        ),
        "prompt_turns_use_loss": False,
        "final_assistant_completion_uses_loss": True,
        "details": details,
    }


def _source_hashes() -> dict[str, str]:
    paths = {
        "core": CORE_ROOT / "pfe_core/phase87_failure_driven_training.py",
        "driver": REPO_ROOT / "tools/phase87_89_failure_driven_adapter_loop.py",
        "core_test": REPO_ROOT / "tests/test_phase87_failure_driven_training.py",
        "driver_test": REPO_ROOT / "tests/test_phase87_89_driver_safety.py",
        "executor": CORE_ROOT / "pfe_core/trainer/executors.py",
    }
    return {name: _sha256(path) for name, path in paths.items()}


def _prepare(clean: bool) -> int:
    if clean and EVIDENCE_ROOT.exists():
        _safe_clean(EVIDENCE_ROOT, REPO_ROOT / "docs/demo")
    PREPARATION_ROOT.mkdir(parents=True, exist_ok=True)
    manual_review = _read_json(PHASE85_ROOT / "manual-semantic-review.json")
    phase85_decision = _read_json(PHASE85_ROOT / "phase85-final-decision.json")
    taxonomy = build_phase87_failure_taxonomy(manual_review)
    candidates = build_phase87_training_candidates()
    quality = audit_phase87_training_candidates(candidates)
    holdout = build_phase89_holdout()
    isolation = audit_phase87_holdout_isolation(candidates, holdout)
    samples = [dict(row) for row in candidates["samples"]]
    pairs = [dict(row) for row in candidates["dpo_pairs"]]
    probe_samples = _select_probe_samples(samples)
    probe_counts = {
        category: sum(row.get("taxonomy_dimension") == category for row in probe_samples)
        for category in PHASE87_CATEGORIES
    }
    probe_selection = {
        "kind": "phase87_probe_sample_selection",
        "passed": len(probe_samples) == 25 and set(probe_counts.values()) == {5},
        "sample_count": len(probe_samples),
        "category_counts": probe_counts,
        "full_candidate_count": len(samples),
        "selection_rule": "first_five_stable_ids_per_category",
        "manifest_sha256": stable_hash(probe_samples),
    }
    boundary = _completion_boundary_report(
        _job_spec(probe_samples, TRAINER_OUTPUT_ROOT / "phase87-preflight", 5)
    )
    overlay_decision_path = PHASE85_OVERLAY_ROOT / "phase85-metric-schema-v2-overlay.json"
    overlay_payload = _read_json(overlay_decision_path) if overlay_decision_path.is_file() else {}
    overlay_decision = dict(overlay_payload.get("overlay_decision") or {})
    model_config = MODEL_PATH / "config.json"
    model_weights = MODEL_PATH / "model.safetensors"
    checks = {
        "phase85_manual_review_bound": taxonomy.get("passed") is True,
        "phase85_canonical_still_archive": phase85_decision.get("status")
        == "archive_incomplete_phase85_evidence",
        "phase85_overlay_archive_complete": overlay_decision.get("status")
        == "archive_low_fallback_runtime_not_qualified",
        "phase85_overlay_product_gate_false": overlay_decision.get("product_gate_qualified")
        is False,
        "training_quality_passed": quality.get("passed") is True,
        "probe_selection_passed": probe_selection.get("passed") is True,
        "holdout_isolation_passed": isolation.get("passed") is True,
        "completion_boundary_passed": boundary.get("passed") is True,
        "local_model_complete": _model_complete(),
        "sample_count_120": len(samples) == 120,
        "holdout_count_30": int(holdout.get("session_count") or 0) == 30,
    }
    freeze = {
        "kind": "phase87_89_pre_experiment_freeze",
        "created_at": _utcnow(),
        "frozen_before_training": True,
        "passed": all(checks.values()),
        "checks": checks,
        "model_path": str(MODEL_PATH),
        "model_revision": MODEL_REVISION,
        "model_config_sha256": _sha256(model_config) if model_config.is_file() else None,
        "model_weight_size_bytes": model_weights.stat().st_size if model_weights.is_file() else 0,
        "training_manifest_sha256": stable_hash(probe_samples),
        "full_candidate_manifest_sha256": stable_hash(samples),
        "dpo_manifest_sha256": stable_hash(pairs),
        "holdout_manifest_sha256": stable_hash(holdout.get("sessions") or []),
        "phase85_manual_review_sha256": _sha256(
            PHASE85_ROOT / "manual-semantic-review.json"
        ),
        "phase85_decision_sha256": _sha256(PHASE85_ROOT / "phase85-final-decision.json"),
        "phase85_overlay_decision_sha256": _sha256(overlay_decision_path)
        if overlay_decision_path.is_file()
        else None,
        "generation_protocol_sha256": stable_hash(GENERATION_PROTOCOL),
        "thresholds_sha256": stable_hash(FROZEN_THRESHOLDS),
        "source_sha256": _source_hashes(),
        "score_or_gate_relaxation_allowed": False,
        "automatic_training_allowed": False,
        "automatic_promotion_allowed": False,
    }
    _write_json(PREPARATION_ROOT / "failure_taxonomy.json", taxonomy)
    _write_jsonl(PREPARATION_ROOT / "selected_sft_samples.jsonl", samples)
    _write_jsonl(PREPARATION_ROOT / "selected_probe_sft_samples.jsonl", probe_samples)
    _write_jsonl(PREPARATION_ROOT / "selected_dpo_pairs.jsonl", pairs)
    _write_json(PREPARATION_ROOT / "training_quality_audit.json", quality)
    _write_json(PREPARATION_ROOT / "probe_sample_selection.json", probe_selection)
    _write_json(PREPARATION_ROOT / "holdout.json", holdout)
    _write_json(PREPARATION_ROOT / "holdout_isolation_audit.json", isolation)
    _write_json(PREPARATION_ROOT / "completion_boundary_report.json", boundary)
    _write_json(PREPARATION_ROOT / "model_selection.json", {
        "kind": "phase87_local_model_selection",
        "status": "selected" if _model_complete() else "blocked",
        "selected_model": "Qwen2.5-1.5B-Instruct" if _model_complete() else None,
        "local_path": str(MODEL_PATH),
        "download_required": False,
        "external_provider_required": False,
        "selection_reason": "reuse the completed Phase81 local PEFT path and isolate the training objective",
    })
    _write_json(EVIDENCE_ROOT / "generation_protocol.json", GENERATION_PROTOCOL)
    _write_json(EVIDENCE_ROOT / "frozen_thresholds.json", FROZEN_THRESHOLDS)
    _write_json(EVIDENCE_ROOT / "pre_experiment_freeze.json", freeze)
    _write_json(EVIDENCE_ROOT / "preparation_decision.json", {
        "kind": "phase87_89_preparation_decision",
        "status": "ready_for_5_step_probe" if freeze["passed"] else "blocked",
        "checks": checks,
        "automatic_training_started": False,
        "product_gate_qualified": False,
    })
    print(json.dumps({
        "status": "ready_for_5_step_probe" if freeze["passed"] else "blocked",
        "checks": checks,
    }, ensure_ascii=False, indent=2))
    return 0 if freeze["passed"] else 2


def _training_freeze_check(steps: int) -> dict[str, Any]:
    freeze = _read_json(EVIDENCE_ROOT / "pre_experiment_freeze.json")
    samples = _read_jsonl(PREPARATION_ROOT / "selected_probe_sft_samples.jsonl")
    holdout = _read_json(PREPARATION_ROOT / "holdout.json")
    checks = {
        "pre_experiment_freeze_passed": freeze.get("passed") is True,
        "training_manifest_unchanged": stable_hash(samples)
        == freeze.get("training_manifest_sha256"),
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
        "source_files_unchanged": _source_hashes() == freeze.get("source_sha256"),
        "twenty_five_step_requires_passed_five_step_sanity": steps != 25
        or _read_json(EVIDENCE_ROOT / "sanity_decision.json").get("passed") is True,
    }
    return {
        "kind": "phase87_training_freeze_check",
        "passed": all(checks.values()),
        "checks": checks,
    }


def _train(steps: int, clean: bool) -> int:
    if steps not in (5, 25):
        raise SystemExit("Phase87 permits only 5 or 25 step probes")
    freeze = _training_freeze_check(steps)
    probe_dir = TRAINING_ROOT / f"probe-{steps}step"
    output_root = TRAINER_OUTPUT_ROOT / f"phase87-failure-driven-{steps}step"
    if clean and probe_dir.exists():
        _safe_clean(probe_dir, TRAINING_ROOT)
    if clean and output_root.exists():
        _safe_clean(output_root, TRAINER_OUTPUT_ROOT)
    probe_dir.mkdir(parents=True, exist_ok=True)
    samples = _read_jsonl(PREPARATION_ROOT / "selected_probe_sft_samples.jsonl")
    spec = _job_spec(samples, output_root, steps)
    boundary = _completion_boundary_report(spec)
    _write_json(probe_dir / "training_manifest.json", spec)
    _write_json(probe_dir / "training_freeze_check.json", freeze)
    _write_json(probe_dir / "completion_boundary_report.json", boundary)
    if not freeze["passed"] or boundary.get("passed") is not True:
        attempt = {
            "kind": "phase87_training_attempt",
            "status": "blocked",
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
            "kind": "phase87_training_attempt",
            "status": "completed" if completed else "failed",
            "real_training": completed,
            "candidate_eligible": False,
            "selected_model": "Qwen2.5-1.5B-Instruct",
            "model": str(MODEL_PATH),
            "model_revision": MODEL_REVISION,
            "requested_steps": steps,
            "learning_rate": TRAINING_LEARNING_RATE,
            "seed": 87,
            "started_at": started_at,
            "finished_at": _utcnow(),
            "duration_seconds": round(time.perf_counter() - started, 4),
            "max_rss_before_bytes": rss_before,
            "max_rss_after_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "execution": real,
            "adapter_validation": validation,
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
            "kind": "phase87_training_attempt",
            "status": "failed",
            "real_training": False,
            "candidate_eligible": False,
            "selected_model": "Qwen2.5-1.5B-Instruct",
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
        _write_json(FAILURE_ROOT / f"training_probe_{steps}step.json", attempt)
    _write_json(probe_dir / "training_attempt.json", attempt)
    _write_json(TRAINING_ROOT / "latest_training_attempt.json", attempt)
    print(json.dumps({
        "status": attempt.get("status"),
        "requested_steps": steps,
        "duration_seconds": attempt.get("duration_seconds"),
        "error": attempt.get("error"),
    }, ensure_ascii=False, indent=2))
    return 0 if attempt.get("status") == "completed" else 1


def _adapter_dir(steps: int) -> Path:
    attempt = _read_json(TRAINING_ROOT / f"probe-{steps}step/training_attempt.json")
    validation = dict(attempt.get("adapter_validation") or {})
    artifact_dir = Path(str(validation.get("artifact_dir") or ""))
    if (
        attempt.get("status") != "completed"
        or validation.get("valid") is not True
        or not artifact_dir.is_dir()
    ):
        raise SystemExit(f"Phase87 {steps}-step adapter is unavailable")
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


def _generation_freeze_check(
    scope: str, variant: str, adapter_path: Path | None
) -> dict[str, Any]:
    freeze = _read_json(EVIDENCE_ROOT / "pre_experiment_freeze.json")
    holdout = _read_json(PREPARATION_ROOT / "holdout.json")
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
        "sanity_adapter_is_5step": scope != "sanity"
        or variant != "adapter"
        or adapter_path == _adapter_dir(5),
        "full_adapter_is_25step": scope != "full"
        or variant != "adapter"
        or adapter_path == _adapter_dir(25),
        "full_requires_passed_sanity": scope != "full"
        or _read_json(EVIDENCE_ROOT / "sanity_decision.json").get("passed") is True,
    }
    return {
        "kind": "phase89_generation_freeze_check",
        "scope": scope,
        "variant": variant,
        "passed": all(checks.values()),
        "checks": checks,
    }


def _render_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
    return "\n".join(
        f"{row.get('role', 'user')}: {row.get('content', '')}" for row in messages
    ) + "\nassistant:"


def _load_runtime(adapter_path: Path | None) -> tuple[Any, Any, Any, str]:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = (
        "mps"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        else "cpu"
    )
    dtype = torch.float16 if device == "mps" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        local_files_only=True,
        low_cpu_mem_usage=True,
        dtype=dtype,
    )
    if adapter_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_path), local_files_only=True)
    model.to(device)
    model.eval()
    return torch, tokenizer, model, device


def _release_runtime(torch: Any, model: Any, device: str) -> None:
    try:
        del model
        if device == "mps":
            torch.mps.empty_cache()
    except Exception:
        pass


def _generate_one(
    torch: Any,
    tokenizer: Any,
    model: Any,
    device: str,
    messages: list[dict[str, str]],
) -> tuple[str, dict[str, Any]]:
    prompt = _render_prompt(tokenizer, messages)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=int(GENERATION_PROTOCOL["input_max_length"]),
    )
    inputs = {name: value.to(device) for name, value in inputs.items()}
    input_length = int(inputs["input_ids"].shape[-1])
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=int(GENERATION_PROTOCOL["max_new_tokens"]),
            do_sample=False,
            repetition_penalty=float(GENERATION_PROTOCOL["repetition_penalty"]),
            no_repeat_ngram_size=int(GENERATION_PROTOCOL["no_repeat_ngram_size"]),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = output[0][input_length:]
    raw = tokenizer.decode(generated, skip_special_tokens=True).strip()
    if not raw:
        raise RuntimeError("Phase89 local generation returned empty output")
    return raw, {
        "input_tokens": input_length,
        "completion_tokens": int(generated.shape[-1]),
        "latency_seconds": round(time.perf_counter() - started, 4),
        "think_leak_detected": bool(re.search(r"</?think>", raw, flags=re.IGNORECASE)),
        "truncated": int(generated.shape[-1])
        >= int(GENERATION_PROTOCOL["max_new_tokens"]),
    }


def _run_eval_session(
    *,
    session: Mapping[str, Any],
    torch: Any,
    tokenizer: Any,
    model: Any,
    device: str,
    adapter_loaded: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    history: list[dict[str, str]] = []
    private_values = [str(value) for value in session.get("declared_private_values") or []]
    turn_rows = []
    final_raw = ""
    final_post = ""
    final_contract_info: dict[str, Any] = {}
    final_raw_score: dict[str, Any] = {}
    final_post_score: dict[str, Any] = {}
    for turn, user_text in enumerate(session.get("user_turns") or [], start=1):
        guarded_user, input_guard = guard_phase77_messages(
            [{"role": "user", "content": str(user_text)}], private_values
        )
        history.append(dict(guarded_user[0]))
        system_prompt, route = contract_for_phase85_messages(history)
        prompt_messages = (
            ([{"role": "system", "content": system_prompt}] if system_prompt else [])
            + history
        )
        raw, generation = _generate_one(
            torch, tokenizer, model, device, prompt_messages
        )
        raw_private_echo = any(value and value in raw for value in private_values)
        guarded_raw, output_guard = guard_phase77_output(raw, private_values)
        post, contract_info = enforce_phase85_persona_output(
            guarded_raw,
            messages=history,
            declared_private_values=private_values,
        )
        assistant = {"role": "assistant", "content": post}
        history.append(assistant)
        raw_score = score_phase89_output(guarded_raw, session)
        post_score = score_phase89_output(post, session)
        if raw_private_echo:
            raw_score["privacy_echo"] = True
            post_score["privacy_echo"] = True
        turn_rows.append({
            "turn": turn,
            "routed": route.get("routed") is True,
            "route_reason": route.get("reason"),
            "system_prompt_applied": bool(system_prompt),
            "input_guard_passed": input_guard.get("blocked") is not True,
            "raw_output_sha256": hashlib.sha256(guarded_raw.encode()).hexdigest(),
            "post_output_sha256": hashlib.sha256(post.encode()).hexdigest(),
            "fallback_used": contract_info.get("fallback_used") is True,
            "native_format": contract_info.get("native_format") is True,
            "semantic_repair_used": contract_info.get("semantic_repair_used") is True,
            "privacy_echo_detected": raw_private_echo,
            **generation,
        })
        final_raw = guarded_raw
        final_post = post
        final_contract_info = dict(contract_info)
        final_raw_score = raw_score
        final_post_score = post_score
    structural = {
        "kind": "phase89_structural_eval_session",
        "session_id": session.get("session_id"),
        "category": session.get("category"),
        "status": "completed",
        "actual_model_call": True,
        "adapter_loaded": adapter_loaded,
        "turn_count": len(turn_rows),
        "turns": turn_rows,
        "final_raw_output_sha256": hashlib.sha256(final_raw.encode()).hexdigest(),
        "final_post_output_sha256": hashlib.sha256(final_post.encode()).hexdigest(),
        "raw_score": final_raw_score,
        "post_score": final_post_score,
        "final_fallback_used": final_contract_info.get("fallback_used") is True,
        "truncated": any(row.get("truncated") is True for row in turn_rows),
        "raw_model_output_persisted": False,
        "private_source_persisted": False,
        "simulated_usage": True,
        "actual_user_feedback": False,
    }
    private = {
        "session_id": session.get("session_id"),
        "category": session.get("category"),
        "raw_output": final_raw,
        "post_output": final_post,
        "raw_output_sha256": structural["final_raw_output_sha256"],
        "post_output_sha256": structural["final_post_output_sha256"],
    }
    return structural, private


def _write_private_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    fd = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)


def _generate(scope: str, variant: str, clean: bool) -> int:
    if scope not in {"sanity", "full"} or variant not in {"base", "adapter"}:
        raise SystemExit("unsupported Phase89 generation scope or variant")
    adapter_path = None if variant == "base" else _adapter_dir(5 if scope == "sanity" else 25)
    freeze = _generation_freeze_check(scope, variant, adapter_path)
    if not freeze["passed"]:
        raise SystemExit(f"Phase89 generation freeze failed: {freeze}")
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
                print(f"[{scope}:{variant}] {index}/{len(sessions)} {session_id} resumed", flush=True)
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
            except Exception as exc:
                structural = {
                    "kind": "phase89_structural_eval_session",
                    "session_id": session_id,
                    "category": session.get("category"),
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
            print(f"[{scope}:{variant}] {index}/{len(sessions)} {session_id} {structural['status']}", flush=True)
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
    fallback_count = sum(row.get("final_fallback_used") is True for row in completed_rows)
    post_metrics.update({
        "fallback_count": fallback_count,
        "fallback_rate": round(fallback_count / len(completed_rows), 4)
        if completed_rows
        else 0.0,
    })
    metrics = {
        "kind": "phase89_variant_metrics",
        "scope": scope,
        "variant": variant,
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
        "session_count": metrics["session_count"],
        "model_call_count": metrics["model_call_count"],
        "raw_overall_score": raw_metrics.get("overall_score"),
        "post_fallback_rate": post_metrics.get("fallback_rate"),
    }, ensure_ascii=False, indent=2))
    return 0 if metrics["all_sessions_completed"] else 1


def _sanity() -> int:
    base = _read_json(EVAL_ROOT / "sanity/metrics_base.json")
    adapter = _read_json(EVAL_ROOT / "sanity/metrics_adapter.json")
    base_raw = dict(base.get("raw") or {})
    adapter_raw = dict(adapter.get("raw") or {})
    checks = {
        "base_completed_five_sessions": base.get("session_count") == 5
        and base.get("model_call_count") == 15,
        "adapter_completed_five_sessions": adapter.get("session_count") == 5
        and adapter.get("model_call_count") == 15,
        "five_step_adapter_valid": _read_json(
            TRAINING_ROOT / "probe-5step/training_attempt.json"
        ).get("status")
        == "completed",
        "adapter_score_not_below_base_by_more_than_0_10": float(
            adapter_raw.get("overall_score") or 0.0
        )
        >= float(base_raw.get("overall_score") or 0.0) - 0.10,
        "adapter_false_block_not_above_base": float(
            adapter_raw.get("false_block_rate") or 0.0
        )
        <= float(base_raw.get("false_block_rate") or 0.0),
        "adapter_unsupported_not_above_base": float(
            adapter_raw.get("unsupported_assertion_rate") or 0.0
        )
        <= float(base_raw.get("unsupported_assertion_rate") or 0.0),
        "adapter_privacy_echo_zero": float(adapter_raw.get("privacy_echo_rate") or 0.0)
        == 0.0,
        "adapter_think_leak_zero": float(adapter_raw.get("think_leak_rate") or 0.0)
        == 0.0,
    }
    payload = {
        "kind": "phase87_five_step_sanity_decision",
        "passed": all(checks.values()),
        "checks": checks,
        "base_raw_score": base_raw.get("overall_score"),
        "adapter_raw_score": adapter_raw.get("overall_score"),
        "next_action": "run_25_step_probe" if all(checks.values()) else "archive_sanity_failure",
        "product_gate_qualified": False,
        "promotion_allowed": False,
        "auto_promotion_allowed": False,
    }
    _write_json(EVIDENCE_ROOT / "sanity_decision.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["passed"] else 1


def _review_template(clean: bool) -> int:
    base_path = REVIEW_CACHE_ROOT / "full_base.jsonl"
    adapter_path = REVIEW_CACHE_ROOT / "full_adapter.jsonl"
    base_rows = {str(row.get("session_id")): row for row in _read_jsonl(base_path)}
    adapter_rows = {str(row.get("session_id")): row for row in _read_jsonl(adapter_path)}
    session_ids = sorted(base_rows)
    if session_ids != sorted(adapter_rows) or len(session_ids) != PHASE89_HOLDOUT_COUNT:
        raise SystemExit("Phase89 full review caches are incomplete")
    blind_rows = []
    public_pairs = []
    variant_key = {}
    for session_id in session_ids:
        base_row = base_rows[session_id]
        adapter_row = adapter_rows[session_id]
        base_is_a = int(hashlib.sha256(session_id.encode()).hexdigest(), 16) % 2 == 0
        candidate_a = base_row if base_is_a else adapter_row
        candidate_b = adapter_row if base_is_a else base_row
        pair_id = f"phase89-pair-{hashlib.sha256(session_id.encode()).hexdigest()[:12]}"
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
            "candidate_a": "base" if base_is_a else "adapter",
            "candidate_b": "adapter" if base_is_a else "base",
        }
    blind_cache = REVIEW_CACHE_ROOT / "blind_pairs.jsonl"
    if clean:
        blind_cache.unlink(missing_ok=True)
    _write_private_jsonl(blind_cache, blind_rows)
    expected_hash = stable_hash(public_pairs)
    template = {
        "kind": "phase89_blind_manual_review",
        "complete": False,
        "passed": False,
        "expected_pair_count": PHASE89_HOLDOUT_COUNT,
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
        "raw_output_persisted_in_evidence": False,
        "review_cache_outside_evidence_root": True,
        "actual_user_feedback_count": 0,
    }
    _write_json(EVAL_ROOT / "manual_review.json", template)
    _write_json(EVAL_ROOT / "blind_variant_key.json", {
        "kind": "phase89_blind_variant_key",
        "pair_count": len(variant_key),
        "mapping": variant_key,
        "contains_raw_output": False,
    })
    print(json.dumps({
        "pair_count": len(public_pairs),
        "pair_manifest_sha256": expected_hash,
        "private_review_cache": str(blind_cache),
    }, ensure_ascii=False, indent=2))
    return 0


def _review_validate() -> int:
    review = _read_json(EVAL_ROOT / "manual_review.json")
    variant_key = dict(
        _read_json(EVAL_ROOT / "blind_variant_key.json").get("mapping") or {}
    )
    pairs = [dict(row) for row in review.get("pairs") or []]
    decisions = [dict(row) for row in review.get("decisions") or []]
    pair_by_id = {str(row.get("pair_id")): row for row in pairs}
    decision_by_id = {str(row.get("pair_id")): row for row in decisions}
    allowed = set(review.get("allowed_findings") or [])
    structural_integrity = (
        len(pairs) == PHASE89_HOLDOUT_COUNT
        and stable_hash(pairs) == review.get("expected_pair_manifest_sha256")
        and set(pair_by_id) == set(decision_by_id) == set(variant_key)
        and len(decisions) == len(decision_by_id)
    )
    adapter_wins = base_wins = ties = adapter_findings = 0
    finding_counts: dict[str, int] = {}
    decisions_valid = structural_integrity
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
        elif key[winner] == "adapter":
            adapter_wins += 1
        else:
            base_wins += 1
        for candidate, findings in (("candidate_a", findings_a), ("candidate_b", findings_b)):
            for finding in findings:
                finding_counts[finding] = finding_counts.get(finding, 0) + 1
            if key[candidate] == "adapter":
                adapter_findings += len(findings)
    complete = (
        review.get("complete") is True
        and int(review.get("reviewed_pair_count") or 0) == PHASE89_HOLDOUT_COUNT
        and bool(review.get("reviewer_ids"))
        and decisions_valid
    )
    passed = complete and adapter_wins > base_wins and adapter_findings == 0
    summary = {
        "kind": "phase89_blind_manual_review_summary",
        "complete": complete,
        "integrity_passed": decisions_valid,
        "passed": passed,
        "reviewed_pair_count": len(decisions),
        "adapter_wins": adapter_wins,
        "base_wins": base_wins,
        "ties": ties,
        "adapter_finding_count": adapter_findings,
        "finding_counts": dict(sorted(finding_counts.items())),
        "manual_review_can_only_tighten": True,
        "raw_output_persisted_in_evidence": False,
        "product_gate_qualified": False,
        "promotion_allowed": False,
    }
    _write_json(EVAL_ROOT / "manual_review_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if complete and decisions_valid else 1


def _run_regression() -> int:
    command = [
        str(REPO_ROOT / ".venv/bin/pytest"),
        "-q",
        "tests/test_phase85_metric_schema_v2_overlay.py",
        "tests/test_phase87_failure_driven_training.py",
        "tests/test_phase87_89_driver_safety.py",
        "tests/test_phase85_low_fallback_semantic_guard.py",
        "tests/test_phase85_driver_safety.py",
        "tests/test_phase85_engine_status_privacy.py",
        "tests/test_phase85_semantic_guard_hardening.py",
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
        "kind": "phase87_89_full_regression_summary",
        "passed": completed.returncode == 0,
        "exit_code": completed.returncode,
        "duration_seconds": round(time.perf_counter() - started, 4),
        "output_line_count": len(output.splitlines()),
        "output_sha256": hashlib.sha256(output.encode()).hexdigest(),
        "raw_process_output_persisted": False,
    }
    _write_json(EVIDENCE_ROOT / "full_regression_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["passed"] else 1


def _walk_forbidden_keys(value: Any, path: str = "$") -> list[str]:
    forbidden = {"raw_output", "post_output", "candidate_a_output", "candidate_b_output"}
    hits = []
    if isinstance(value, Mapping):
        for key, nested in value.items():
            child = f"{path}.{key}"
            if key in forbidden:
                hits.append(child)
            hits.extend(_walk_forbidden_keys(nested, child))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            hits.extend(_walk_forbidden_keys(nested, f"{path}[{index}]"))
    return hits


def _write_runbook() -> None:
    text = "# Phase87-89 Failure-driven Adapter Loop\n\n"
    text += "This loop uses only local Qwen2.5-1.5B and simulated_usage evidence.\n\n"
    text += "```bash\n"
    text += ".venv/bin/python tools/phase85_metric_schema_v2_overlay.py finalize\n"
    text += ".venv/bin/python tools/phase85_metric_schema_v2_overlay.py validate\n"
    text += ".venv/bin/python tools/phase87_89_failure_driven_adapter_loop.py prepare --clean\n"
    text += "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python tools/phase87_89_failure_driven_adapter_loop.py train --steps 5 --clean\n"
    text += ".venv/bin/python tools/phase87_89_failure_driven_adapter_loop.py generate --scope sanity --variant base --clean\n"
    text += ".venv/bin/python tools/phase87_89_failure_driven_adapter_loop.py generate --scope sanity --variant adapter --clean\n"
    text += ".venv/bin/python tools/phase87_89_failure_driven_adapter_loop.py sanity\n"
    text += "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python tools/phase87_89_failure_driven_adapter_loop.py train --steps 25 --clean\n"
    text += ".venv/bin/python tools/phase87_89_failure_driven_adapter_loop.py generate --scope full --variant base --clean\n"
    text += ".venv/bin/python tools/phase87_89_failure_driven_adapter_loop.py generate --scope full --variant adapter --clean\n"
    text += ".venv/bin/python tools/phase87_89_failure_driven_adapter_loop.py review-template --clean\n"
    text += ".venv/bin/python tools/phase87_89_failure_driven_adapter_loop.py review-validate\n"
    text += ".venv/bin/python tools/phase87_89_failure_driven_adapter_loop.py full-regression\n"
    text += ".venv/bin/python tools/phase87_89_failure_driven_adapter_loop.py finalize\n"
    text += ".venv/bin/python tools/phase87_89_failure_driven_adapter_loop.py validate\n"
    text += "```\n\nNo automatic promotion, deployment, Hermes attachment, or actual-user-benefit claim is permitted.\n"
    (EVIDENCE_ROOT / "phase87-89-runbook.md").write_text(text, encoding="utf-8")


def _finalize() -> int:
    base = _read_json(EVAL_ROOT / "full/metrics_base.json")
    adapter = _read_json(EVAL_ROOT / "full/metrics_adapter.json")
    attempt = _read_json(TRAINING_ROOT / "probe-25step/training_attempt.json")
    isolation = _read_json(PREPARATION_ROOT / "holdout_isolation_audit.json")
    manual = _read_json(EVAL_ROOT / "manual_review_summary.json")
    decision = build_phase89_decision(
        base_raw=dict(base.get("raw") or {}),
        adapter_raw=dict(adapter.get("raw") or {}),
        base_runtime=dict(base.get("post_contract") or {}),
        adapter_runtime=dict(adapter.get("post_contract") or {}),
        training_attempt=attempt,
        isolation_audit=isolation,
        manual_review=manual,
    )
    comparison = {
        "kind": "phase87_89_base_adapter_comparison",
        "base": {"raw": base.get("raw"), "post_contract": base.get("post_contract")},
        "adapter": {"raw": adapter.get("raw"), "post_contract": adapter.get("post_contract")},
        "decision_status": decision.get("status"),
        "product_gate_qualified": decision.get("product_gate_qualified") is True,
        "actual_user_feedback_count": 0,
        "actual_product_benefit_claim_allowed": False,
    }
    forbidden_locations = []
    for path in sorted(EVIDENCE_ROOT.rglob("*.json")):
        forbidden_locations.extend(
            f"{path.relative_to(EVIDENCE_ROOT)}:{location}"
            for location in _walk_forbidden_keys(_read_json(path))
        )
    public_private = {
        "kind": "phase87_89_public_private_audit",
        "passed": not forbidden_locations,
        "forbidden_raw_output_key_locations": forbidden_locations,
        "raw_model_output_persisted": False,
        "review_cache_outside_evidence_root": True,
    }
    regression = _read_json(EVIDENCE_ROOT / "full_regression_summary.json")
    integrity_checks = {
        "pre_experiment_freeze_still_passes": _training_freeze_check(25).get("passed")
        is True,
        "training_completed": attempt.get("status") == "completed",
        "base_and_adapter_completed_30_sessions": base.get("session_count") == 30
        and adapter.get("session_count") == 30,
        "holdout_isolation_passed": isolation.get("passed") is True,
        "manual_review_complete_and_integral": manual.get("complete") is True
        and manual.get("integrity_passed") is True,
        "public_private_audit_passed": public_private["passed"],
        "full_regression_passed": regression.get("passed") is True,
    }
    integrity = {
        "kind": "phase87_89_evidence_integrity",
        "passed": all(integrity_checks.values()),
        "checks": integrity_checks,
        "product_gate_qualified": decision.get("product_gate_qualified") is True,
        "validation_pass_does_not_imply_product_pass": True,
    }
    _write_json(EVIDENCE_ROOT / "comparison_summary.json", comparison)
    _write_json(EVIDENCE_ROOT / "phase89-final-decision.json", decision)
    _write_json(EVIDENCE_ROOT / "public_private_audit.json", public_private)
    _write_json(EVIDENCE_ROOT / "evidence_integrity.json", integrity)
    _write_runbook()
    for path in REVIEW_CACHE_ROOT.glob("*.jsonl"):
        path.unlink(missing_ok=True)
    files = {}
    for path in sorted(EVIDENCE_ROOT.rglob("*")):
        if path.is_file() and path.name not in {"evidence_manifest.json", "validation_summary.json"}:
            files[path.relative_to(EVIDENCE_ROOT).as_posix()] = _sha256(path)
    manifest = {
        "kind": "phase87_89_evidence_manifest",
        "files": files,
        "file_count": len(files),
        "product_gate_qualified": decision.get("product_gate_qualified") is True,
    }
    _write_json(EVIDENCE_ROOT / "evidence_manifest.json", manifest)
    print(json.dumps({
        "status": decision.get("status"),
        "recommendation": decision.get("recommendation"),
        "product_gate_qualified": decision.get("product_gate_qualified"),
        "integrity_passed": integrity.get("passed"),
    }, ensure_ascii=False, indent=2))
    return 0 if integrity["passed"] else 1


def _validate() -> int:
    manifest = _read_json(EVIDENCE_ROOT / "evidence_manifest.json")
    failures = []
    for relative, expected in dict(manifest.get("files") or {}).items():
        path = EVIDENCE_ROOT / relative
        if not path.is_file() or _sha256(path) != expected:
            failures.append(relative)
    decision = _read_json(EVIDENCE_ROOT / "phase89-final-decision.json")
    integrity = _read_json(EVIDENCE_ROOT / "evidence_integrity.json")
    checks = {
        "manifest_files_unchanged": not failures,
        "evidence_integrity_passed": integrity.get("passed") is True,
        "no_auto_promotion": decision.get("auto_promotion_allowed") is False
        and decision.get("promotion_allowed") is False,
        "no_auto_deployment": decision.get("automatic_deployment_allowed") is False,
        "no_hermes_attachment": decision.get("hermes_attachment_allowed") is False,
        "no_actual_product_benefit_claim": decision.get(
            "actual_product_benefit_claim_allowed"
        )
        is False,
        "review_cache_absent": not any(REVIEW_CACHE_ROOT.glob("*.jsonl")),
    }
    summary = {
        "kind": "phase87_89_validation_summary",
        "passed": all(checks.values()),
        "checks": checks,
        "manifest_failures": failures,
        "decision_status": decision.get("status"),
        "product_gate_qualified": decision.get("product_gate_qualified") is True,
        "validation_pass_does_not_imply_product_pass": True,
    }
    _write_json(EVIDENCE_ROOT / "validation_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["passed"] else 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--clean", action="store_true")
    train = subparsers.add_parser("train")
    train.add_argument("--steps", type=int, choices=(5, 25), required=True)
    train.add_argument("--clean", action="store_true")
    generate = subparsers.add_parser("generate")
    generate.add_argument("--scope", choices=("sanity", "full"), required=True)
    generate.add_argument("--variant", choices=("base", "adapter"), required=True)
    generate.add_argument("--clean", action="store_true")
    subparsers.add_parser("sanity")
    review_template = subparsers.add_parser("review-template")
    review_template.add_argument("--clean", action="store_true")
    subparsers.add_parser("review-validate")
    subparsers.add_parser("full-regression")
    subparsers.add_parser("finalize")
    subparsers.add_parser("validate")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if args.command == "prepare":
        return _prepare(args.clean)
    if args.command == "train":
        return _train(args.steps, args.clean)
    if args.command == "generate":
        return _generate(args.scope, args.variant, args.clean)
    if args.command == "sanity":
        return _sanity()
    if args.command == "review-template":
        return _review_template(args.clean)
    if args.command == "review-validate":
        return _review_validate()
    if args.command == "full-regression":
        return _run_regression()
    if args.command == "finalize":
        return _finalize()
    if args.command == "validate":
        return _validate()
    raise SystemExit(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
