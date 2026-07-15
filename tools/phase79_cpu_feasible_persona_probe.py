#!/usr/bin/env python3
"""Run the Phase79 CPU-feasible persona adapter benefit probe."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
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

import phase75_personalization_benefit_benchmark as phase75_driver
import phase78_persona_internalization_training as phase78_driver
from pfe_core.adapter_store.quality import validate_adapter_artifact
from pfe_core.phase75_personalization_benefit_benchmark import (
    PHASE75_MINIMAL_CONTRACT,
    aggregate_phase75_variant,
    stable_hash,
)
from pfe_core.phase77_private_value_guarded_runtime import (
    build_phase77_holdout,
    contract_for_phase77_messages,
    guard_phase77_messages,
    guard_phase77_output,
)
from pfe_core.phase78_persona_internalization_training import (
    PHASE78_COMPARISONS,
    PHASE78_PERSONA_CATEGORIES,
    PHASE78_TRAINING_SAMPLE_COUNT,
    audit_phase78_public_private_values,
    audit_phase78_training_samples,
    build_phase78_blind_pairs,
    build_phase78_holdout,
    build_phase78_sft_job_spec,
    build_phase78_training_samples,
    score_phase78_blind_pairs_deterministic,
    summarize_phase78_blind_results,
)
from pfe_core.phase79_cpu_feasible_persona_probe import (
    PHASE79_MODEL_NAME,
    PHASE79_SESSION_COUNT,
    PHASE79_VARIANTS,
    audit_phase79_isolation,
    build_phase79_decision,
    build_phase79_holdout,
    build_phase79_sanity_blocked_decision,
)
from pfe_core.trainer.executors import (
    _build_sft_prompt_and_text,
    _encode_sft_examples,
    _run_real_local_peft_training,
)


EVIDENCE_ROOT = REPO_ROOT / "docs/demo/phase79-cpu-feasible-persona-adapter-probe"
PREPARATION_ROOT = EVIDENCE_ROOT / "evidence-preparation"
TRAINING_ROOT = EVIDENCE_ROOT / "evidence-real-training"
GENERATION_ROOT = EVIDENCE_ROOT / "evidence-real-generation"
JUDGE_ROOT = EVIDENCE_ROOT / "evidence-blind-eval"
FAILURE_ROOT = EVIDENCE_ROOT / "evidence-failures"
MODEL_PATH = REPO_ROOT / "models/Qwen2.5-0.5B-Instruct"
TRAINER_OUTPUT_ROOT = REPO_ROOT / "trainer_job_outputs"
CORE_SOURCE = CORE_ROOT / "pfe_core/phase79_cpu_feasible_persona_probe.py"
DRIVER_SOURCE = REPO_ROOT / "tools/phase79_cpu_feasible_persona_probe.py"
TEST_SOURCE = REPO_ROOT / "tests/test_phase79_cpu_feasible_persona_probe.py"
EXECUTOR_SOURCE = CORE_ROOT / "pfe_core/trainer/executors.py"
PHASE78_ROOT = REPO_ROOT / "docs/demo/phase78-persona-internalization-training-probe"
PHASE32_ROOT = REPO_ROOT / "docs/demo/phase32-personal-agent-preference-training-loop"
HISTORICAL_PHASE32_ADAPTER = (
    REPO_ROOT
    / "trainer_job_outputs/phase32-personal-agent-preference-qwen25-0_5b/dpo_adapter"
)
JUDGE_MODELS = ("gemma4:31b", "qwen3.6")
GENERATION_PROTOCOL = {
    **phase75_driver.GENERATION_PROTOCOL,
    "kind": "phase79_frozen_generation_protocol",
    "variants": list(PHASE79_VARIANTS),
    "selected_model": PHASE79_MODEL_NAME,
    "declared_private_values_redacted_before_every_model_call": True,
    "raw_model_output_checked_before_persistence": True,
    "same_decoding_all_arms": True,
    "runtime_reference_uses_phase77_conditional_contract": True,
    "historical_phase32_adapter_reused": False,
}
DYNAMIC_FILES = {
    "evidence_manifest.json",
    "evidence_integrity.json",
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


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
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


def _phase79_job_spec(samples: Iterable[Mapping[str, Any]], output_dir: Path, steps: int) -> dict[str, Any]:
    spec = build_phase78_sft_job_spec(
        samples=samples,
        base_model=str(MODEL_PATH),
        output_dir=str(output_dir),
        max_steps=steps,
        learning_rate=5e-5,
        seed=79,
    )
    spec["recipe"]["training"]["max_length"] = 176
    spec["phase79"] = {
        "target_model": PHASE79_MODEL_NAME,
        "cpu_feasible_diagnostic": True,
        "source_curriculum": "phase78_privacy_safe_persona_curriculum",
        "completion_only_loss_required": True,
        "full_coverage_required_for_final_candidate": True,
        "historical_phase32_adapter_reused": False,
        "simulated_usage": True,
        "actual_user_feedback": False,
        "auto_promotion_allowed": False,
    }
    return spec


def _model_selection() -> dict[str, Any]:
    config = MODEL_PATH / "config.json"
    shards = sorted(MODEL_PATH.glob("*.safetensors"))
    try:
        import torch

        torch_version = str(torch.__version__)
        mps_available = bool(torch.backends.mps.is_available())
        runtime_available = True
    except Exception:
        torch_version = None
        mps_available = False
        runtime_available = False
    phase32_attempt = PHASE32_ROOT / "evidence-training/training_attempt.json"
    checks = {
        "local_config_exists": config.exists(),
        "weight_shards_exist": bool(shards),
        "torch_runtime_available": runtime_available,
        "phase32_cpu_training_history_exists": phase32_attempt.exists(),
    }
    return {
        "kind": "phase79_model_selection",
        "status": "selected" if all(checks.values()) else "blocked",
        "selected_model": PHASE79_MODEL_NAME,
        "model_path": str(MODEL_PATH),
        "reason": (
            "The local unquantized Qwen2.5-0.5B-Instruct is small enough for complete CPU LoRA "
            "training and evaluation in the current sandbox. It is a diagnostic candidate for "
            "persona internalization, not a replacement for the Phase77 runtime quality ceiling."
        ),
        "checks": checks,
        "config_sha256": _sha256(config) if config.exists() else None,
        "weight_shard_count": len(shards),
        "weight_bytes": sum(path.stat().st_size for path in shards),
        "torch_version": torch_version,
        "mps_available_in_current_process": mps_available,
        "selected_execution_device": "mps" if mps_available else "cpu",
        "historical_phase32_adapter_path": str(HISTORICAL_PHASE32_ADAPTER),
        "historical_phase32_adapter_exists": HISTORICAL_PHASE32_ADAPTER.is_dir(),
        "historical_phase32_adapter_reused": False,
        "not_selected": {
            "Qwen3-4B": "Phase78 produced no adapter after two CPU attempts in this sandbox.",
            "Qwen3.6-27B": "Phase12 training hit Metal OOM and is not a CPU training candidate.",
        },
    }


def _phase32_audit() -> dict[str, Any]:
    path = PHASE32_ROOT / "evidence-eval/decision.json"
    decision = _read_json(path)
    base = float(dict(decision.get("base_scores") or {}).get("overall_personalization_score") or 0.0)
    adapter = float(dict(decision.get("adapter_scores") or {}).get("overall_personalization_score") or 0.0)
    gain = round(adapter - base, 4)
    checks = {
        "historical_decision_exists": bool(decision),
        "actual_user_feedback_was_absent": decision.get("actual_user_feedback_collected") is False,
        "historical_product_claim_was_allowed": decision.get("product_benefit_claim_allowed") is True,
        "historical_gain_below_phase79_gate": gain < 0.08,
        "historical_adapter_not_reused": True,
    }
    return {
        "kind": "phase79_phase32_overclaim_audit",
        "passed": all(checks.values()),
        "checks": checks,
        "historical_base_score": base,
        "historical_adapter_score": adapter,
        "historical_adapter_gain": gain,
        "historical_recommendation": decision.get("recommendation"),
        "historical_product_benefit_claim_allowed": decision.get("product_benefit_claim_allowed"),
        "phase79_interpretation": (
            "Phase32 proves this model family can train on CPU. Its 0.041 simulated score gain "
            "does not satisfy the Phase79 benefit gate and its product-benefit claim is not inherited."
        ),
        "historical_adapter_reused": False,
    }


def _completion_boundary_report(job_spec: Mapping[str, Any]) -> dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), local_files_only=True)
    maximum = int(dict(dict(job_spec.get("recipe") or {}).get("training") or {}).get("max_length") or 160)
    examples = [dict(row) for row in job_spec.get("training_examples") or []]
    encoded = _encode_sft_examples(
        tokenizer=tokenizer,
        training_examples=examples,
        max_length=maximum,
        vocab_size=int(getattr(tokenizer, "vocab_size", 0) or 151936),
    )
    details = []
    for source, row in zip(examples, encoded):
        prompt, full_text = _build_sft_prompt_and_text(
            tokenizer,
            str(source.get("instruction") or ""),
            str(source.get("chosen") or ""),
            messages=source.get("messages"),
        )
        full_token_count = len(tokenizer(full_text, add_special_tokens=False).get("input_ids") or [])
        prompt_tokens = tokenizer(
            prompt,
            truncation=True,
            max_length=maximum,
            add_special_tokens=False,
        ).get("input_ids") or []
        labels = list(row.get("labels") or [])
        completion = [index for index, value in enumerate(labels) if int(value) != -100]
        prompt_boundary = min(len(prompt_tokens), len(labels))
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
        "all_samples_encoded": len(encoded) == len(examples) == PHASE78_TRAINING_SAMPLE_COUNT,
        "no_training_sample_truncated": not any(row["truncated"] for row in details),
        "all_prompt_labels_masked": all(row["prompt_labels_all_masked"] for row in details),
        "all_completions_after_prompt": all(
            row["completion_begins_at_or_after_prompt"] for row in details
        ),
        "minimum_completion_tokens_at_least_4": min(
            (row["completion_label_token_count"] for row in details),
            default=0,
        ) >= 4,
    }
    return {
        "kind": "phase79_completion_only_boundary_report",
        "passed": all(checks.values()),
        "checks": checks,
        "source_sample_count": len(examples),
        "encoded_sample_count": len(encoded),
        "max_length": maximum,
        "maximum_full_token_count": max((row["full_token_count"] for row in details), default=0),
        "minimum_completion_label_token_count": min(
            (row["completion_label_token_count"] for row in details), default=0
        ),
        "prompt_turns_use_loss": False,
        "final_assistant_completion_uses_loss": True,
        "details": details,
    }


def _prepare(clean: bool) -> int:
    if clean and EVIDENCE_ROOT.exists():
        shutil.rmtree(EVIDENCE_ROOT)
    required = (
        CORE_SOURCE,
        DRIVER_SOURCE,
        TEST_SOURCE,
        EXECUTOR_SOURCE,
        MODEL_PATH / "config.json",
        PHASE78_ROOT / "phase78-final-decision.json",
        PHASE32_ROOT / "evidence-eval/decision.json",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"Phase79 required sources missing: {missing}")

    samples = build_phase78_training_samples()
    holdout = build_phase79_holdout()
    quality = audit_phase78_training_samples(samples)
    isolation = audit_phase79_isolation(
        holdout["sessions"],
        build_phase78_holdout()["sessions"],
        build_phase77_holdout()["sessions"],
    )
    model_selection = _model_selection()
    phase32 = _phase32_audit()
    phase78 = _read_json(PHASE78_ROOT / "phase78-final-decision.json")
    preview_spec = _phase79_job_spec(
        samples,
        TRAINER_OUTPUT_ROOT / "phase79-preview",
        PHASE78_TRAINING_SAMPLE_COUNT,
    )
    boundary = _completion_boundary_report(preview_spec)
    baseline_checks = {
        "phase78_archived_for_execution_environment": phase78.get("status")
        == "archive_execution_environment_blocked",
        "phase78_adapter_benefit_not_evaluated": phase78.get("adapter_benefit")
        == "not_evaluated_without_artifact",
        "phase78_no_auto_promotion": phase78.get("auto_promotion_allowed") is False,
    }
    checks = {
        "training_quality_passed": quality.get("passed") is True,
        "fresh_holdout_isolation_passed": isolation.get("passed") is True,
        "completion_boundary_passed": boundary.get("passed") is True,
        "cpu_feasible_model_selected": model_selection.get("status") == "selected",
        "phase78_archive_acknowledged": all(baseline_checks.values()),
        "phase32_overclaim_audited": phase32.get("passed") is True,
    }
    freeze = {
        "kind": "phase79_pre_training_freeze",
        "frozen_at": _utcnow(),
        "frozen_before_training": True,
        "passed": all(checks.values()),
        "checks": checks,
        "training_manifest_sha256": stable_hash(samples),
        "holdout_manifest_sha256": stable_hash(holdout["sessions"]),
        "quality_audit_sha256": stable_hash(quality),
        "isolation_audit_sha256": stable_hash(isolation),
        "completion_boundary_sha256": stable_hash(boundary),
        "model_selection_sha256": stable_hash(model_selection),
        "phase32_audit_sha256": stable_hash(phase32),
        "phase78_archive_sha256": stable_hash(phase78),
        "core_source_sha256": _sha256(CORE_SOURCE),
        "driver_source_sha256": _sha256(DRIVER_SOURCE),
        "test_source_sha256": _sha256(TEST_SOURCE),
        "executor_source_sha256": _sha256(EXECUTOR_SOURCE),
        "generation_protocol_sha256": stable_hash(GENERATION_PROTOCOL),
        "frozen_learning_rate": 5e-5,
        "frozen_seed": 79,
        "frozen_final_steps": PHASE78_TRAINING_SAMPLE_COUNT,
        "historical_adapter_reuse_allowed": False,
        "score_or_gate_relaxation_allowed": False,
    }
    _write_jsonl(PREPARATION_ROOT / "selected_training_samples.jsonl", samples)
    _write_json(PREPARATION_ROOT / "training_quality_audit.json", quality)
    _write_json(PREPARATION_ROOT / "holdout.json", holdout)
    _write_json(PREPARATION_ROOT / "isolation_audit.json", isolation)
    _write_json(PREPARATION_ROOT / "model_selection.json", model_selection)
    _write_json(PREPARATION_ROOT / "completion_boundary_report.json", boundary)
    _write_json(PREPARATION_ROOT / "phase32_overclaim_audit.json", phase32)
    _write_json(PREPARATION_ROOT / "phase78_archive_snapshot.json", {
        "kind": "phase79_phase78_archive_snapshot",
        "checks": baseline_checks,
        "passed": all(baseline_checks.values()),
        "decision": phase78,
    })
    _write_json(EVIDENCE_ROOT / "generation_protocol.json", GENERATION_PROTOCOL)
    _write_json(EVIDENCE_ROOT / "pre_training_freeze.json", freeze)
    _write_json(EVIDENCE_ROOT / "preparation_decision.json", {
        "kind": "phase79_preparation_decision",
        "status": "ready_for_12_step_cpu_probe" if freeze["passed"] else "blocked",
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "automatic_training_allowed": False,
        "explicit_phase79_command_required": True,
    })
    existing_probe_path = TRAINING_ROOT / "probe-12step/training_attempt.json"
    existing_manifest_path = TRAINING_ROOT / "probe-12step/training_manifest.json"
    if existing_probe_path.exists() and existing_manifest_path.exists():
        existing_probe = _read_json(existing_probe_path)
        existing_manifest = _read_json(existing_manifest_path)
        current_manifest = _phase79_job_spec(
            samples,
            TRAINER_OUTPUT_ROOT / "phase79-probe-12step",
            12,
        )
        validation = dict(existing_probe.get("adapter_validation") or {})
        adapter_path = Path(str(validation.get("adapter_path") or ""))
        amendment_checks = {
            "existing_probe_completed": existing_probe.get("status") == "completed",
            "training_manifest_unchanged": stable_hash(existing_manifest)
            == stable_hash(current_manifest),
            "adapter_hash_unchanged": adapter_path.is_file()
            and _sha256(adapter_path) == validation.get("sha256"),
            "historical_adapter_not_reused": existing_probe.get("historical_adapter_reused") is False,
        }
        _write_json(TRAINING_ROOT / "probe-12step/post_training_freeze_amendment.json", {
            "kind": "phase79_post_training_driver_amendment",
            "created_at": _utcnow(),
            "reason": (
                "The first sanity command reused the Phase75 runtime loader, which is hardcoded to "
                "Qwen3-4B. Phase79 now loads its selected 0.5B base explicitly. No training recipe, "
                "sample, or adapter artifact changed."
            ),
            "checks": amendment_checks,
            "passed": all(amendment_checks.values()),
            "probe_retraining_required": not all(amendment_checks.values()),
        })
    print(json.dumps({
        "status": "ready" if freeze["passed"] else "blocked",
        "selected_model": PHASE79_MODEL_NAME,
        "device": model_selection.get("selected_execution_device"),
        "maximum_full_token_count": boundary.get("maximum_full_token_count"),
        "checks": checks,
    }, ensure_ascii=False, indent=2))
    return 0 if freeze["passed"] else 1


def _training_freeze_check() -> dict[str, Any]:
    freeze = _read_json(EVIDENCE_ROOT / "pre_training_freeze.json")
    samples = _read_jsonl(PREPARATION_ROOT / "selected_training_samples.jsonl")
    holdout = _read_json(PREPARATION_ROOT / "holdout.json")
    quality = _read_json(PREPARATION_ROOT / "training_quality_audit.json")
    isolation = _read_json(PREPARATION_ROOT / "isolation_audit.json")
    boundary = _read_json(PREPARATION_ROOT / "completion_boundary_report.json")
    model_selection = _read_json(PREPARATION_ROOT / "model_selection.json")
    phase32 = _read_json(PREPARATION_ROOT / "phase32_overclaim_audit.json")
    phase78 = _read_json(PHASE78_ROOT / "phase78-final-decision.json")
    checks = {
        "preparation_passed": freeze.get("passed") is True,
        "training_unchanged": stable_hash(samples) == freeze.get("training_manifest_sha256"),
        "holdout_unchanged": stable_hash(holdout.get("sessions") or [])
        == freeze.get("holdout_manifest_sha256"),
        "quality_unchanged": stable_hash(quality) == freeze.get("quality_audit_sha256"),
        "isolation_unchanged": stable_hash(isolation) == freeze.get("isolation_audit_sha256"),
        "boundary_unchanged": stable_hash(boundary) == freeze.get("completion_boundary_sha256"),
        "model_selection_unchanged": stable_hash(model_selection)
        == freeze.get("model_selection_sha256"),
        "phase32_audit_unchanged": stable_hash(phase32) == freeze.get("phase32_audit_sha256"),
        "phase78_archive_unchanged": stable_hash(phase78) == freeze.get("phase78_archive_sha256"),
        "core_unchanged": _sha256(CORE_SOURCE) == freeze.get("core_source_sha256"),
        "driver_unchanged": _sha256(DRIVER_SOURCE) == freeze.get("driver_source_sha256"),
        "test_unchanged": _sha256(TEST_SOURCE) == freeze.get("test_source_sha256"),
        "executor_unchanged": _sha256(EXECUTOR_SOURCE) == freeze.get("executor_source_sha256"),
    }
    return {"kind": "phase79_training_freeze_check", "passed": all(checks.values()), "checks": checks}


def _probe_name(steps: int) -> str:
    return "candidate-full-120step" if steps >= PHASE78_TRAINING_SAMPLE_COUNT else f"probe-{steps}step"


def _coverage_report(real: Mapping[str, Any], requested_steps: int) -> dict[str, Any]:
    samples = {str(key): int(value) for key, value in dict(real.get("sample_exposure_counts") or {}).items()}
    categories = {str(key): int(value) for key, value in dict(real.get("category_exposure_counts") or {}).items()}
    full = len(samples) == PHASE78_TRAINING_SAMPLE_COUNT and all(value >= 1 for value in samples.values())
    return {
        "kind": "phase79_actual_exposure_report",
        "requested_steps": requested_steps,
        "actual_step_count": real.get("steps"),
        "sampling_strategy": real.get("sampling_strategy"),
        "sample_exposure_counts": dict(sorted(samples.items())),
        "category_exposure_counts": dict(sorted(categories.items())),
        "unique_samples_exposed": len(samples),
        "unique_categories_exposed": len(categories),
        "minimum_sample_exposure": min(samples.values(), default=0),
        "maximum_sample_exposure": max(samples.values(), default=0),
        "full_coverage": full,
        "eligible_as_final_candidate": requested_steps >= PHASE78_TRAINING_SAMPLE_COUNT
        and full
        and len(categories) == len(PHASE78_PERSONA_CATEGORIES) + 1,
    }


def _train(steps: int, clean: bool) -> int:
    steps = max(1, int(steps))
    freeze = _training_freeze_check()
    if not freeze["passed"]:
        raise SystemExit(f"Phase79 training freeze failed: {freeze}")
    name = _probe_name(steps)
    probe_dir = TRAINING_ROOT / name
    output_dir = TRAINER_OUTPUT_ROOT / f"phase79-{name}"
    if clean:
        shutil.rmtree(probe_dir, ignore_errors=True)
        shutil.rmtree(output_dir, ignore_errors=True)
    probe_dir.mkdir(parents=True, exist_ok=True)
    samples = _read_jsonl(PREPARATION_ROOT / "selected_training_samples.jsonl")
    job_spec = _phase79_job_spec(samples, output_dir, steps)
    boundary = _completion_boundary_report(job_spec)
    _write_json(probe_dir / "training_manifest.json", job_spec)
    _write_json(probe_dir / "completion_boundary_report.json", boundary)
    _write_json(probe_dir / "training_freeze_check.json", freeze)
    sanity_path = TRAINING_ROOT / "probe-12step/sanity_report.json"
    sanity = _read_json(sanity_path) if sanity_path.exists() else {}
    full_probe_blocked = steps >= PHASE78_TRAINING_SAMPLE_COUNT and sanity.get("passed") is not True
    if not boundary["passed"] or len(samples) != PHASE78_TRAINING_SAMPLE_COUNT or full_probe_blocked:
        attempt = {
            "kind": "phase79_qwen25_0_5b_sft_training_attempt",
            "status": "blocked",
            "requested_steps": steps,
            "reason": "probe_sanity_not_passed" if full_probe_blocked else "training_preflight_failed",
            "real_training": False,
            "selected_model": PHASE79_MODEL_NAME,
            "historical_adapter_reused": False,
            "candidate_eligible": False,
            "actual_product_benefit_claim_allowed": False,
            "auto_promotion_allowed": False,
        }
        _write_json(probe_dir / "training_attempt.json", attempt)
        return 2

    started = time.perf_counter()
    started_at = _utcnow()
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    try:
        result = _run_real_local_peft_training(job_spec)
        real = dict(result.get("real_execution") or {})
        artifact_dir = Path(str(real.get("artifact_dir") or ""))
        adapter_path = artifact_dir / "adapter_model.safetensors"
        validation = validate_adapter_artifact(
            artifact_dir,
            {"artifact_name": adapter_path.name, "artifact_format": "peft_lora"},
        )
        validation.update({
            "sha256": _sha256(adapter_path) if adapter_path.exists() else None,
            "artifact_dir": str(artifact_dir),
            "adapter_path": str(adapter_path),
            "parameters_updated": real.get("parameters_updated"),
            "steps": real.get("steps"),
        })
        exposure = _coverage_report(real, steps)
        completed = (
            result.get("status") == "completed"
            and real.get("success") is True
            and real.get("parameters_updated") is True
            and int(real.get("steps") or 0) >= steps
            and validation.get("valid") is True
        )
        if steps >= PHASE78_TRAINING_SAMPLE_COUNT:
            completed = completed and exposure["eligible_as_final_candidate"]
        attempt = {
            "kind": "phase79_qwen25_0_5b_sft_training_attempt",
            "status": "completed" if completed else "failed",
            "real_training": completed,
            "candidate_eligible": completed and exposure["eligible_as_final_candidate"],
            "selected_model": PHASE79_MODEL_NAME,
            "model": str(MODEL_PATH),
            "requested_steps": steps,
            "learning_rate": 5e-5,
            "seed": 79,
            "started_at": started_at,
            "finished_at": _utcnow(),
            "duration_seconds": round(time.perf_counter() - started, 4),
            "max_rss_before_bytes": rss_before,
            "max_rss_after_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "execution": real,
            "adapter_validation": validation,
            "exposure": exposure,
            "historical_adapter_reused": False,
            "simulated_usage": True,
            "actual_user_feedback": False,
            "actual_product_benefit_claim_allowed": False,
            "auto_promotion_allowed": False,
        }
        _write_json(probe_dir / "adapter_validation.json", validation)
        _write_json(probe_dir / "actual_exposure_report.json", exposure)
        _write_json(probe_dir / "train_log.json", {"loss_history": real.get("loss_history") or []})
        _write_json(probe_dir / "parameter_fingerprint_before_after.json", {
            "before": real.get("parameter_fingerprint_before"),
            "after": real.get("parameter_fingerprint_after"),
            "parameters_updated": real.get("parameters_updated"),
        })
    except Exception as exc:
        attempt = {
            "kind": "phase79_qwen25_0_5b_sft_training_attempt",
            "status": "failed",
            "real_training": False,
            "candidate_eligible": False,
            "selected_model": PHASE79_MODEL_NAME,
            "model": str(MODEL_PATH),
            "requested_steps": steps,
            "started_at": started_at,
            "finished_at": _utcnow(),
            "duration_seconds": round(time.perf_counter() - started, 4),
            "max_rss_before_bytes": rss_before,
            "max_rss_after_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "error": f"{exc.__class__.__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "historical_adapter_reused": False,
            "simulated_usage": True,
            "actual_user_feedback": False,
            "actual_product_benefit_claim_allowed": False,
            "auto_promotion_allowed": False,
        }
        _write_json(FAILURE_ROOT / f"training_{name}.json", attempt)
    _write_json(probe_dir / "training_attempt.json", attempt)
    _write_json(TRAINING_ROOT / "latest_training_attempt.json", attempt)
    print(json.dumps({key: attempt.get(key) for key in (
        "status", "requested_steps", "candidate_eligible", "duration_seconds", "error",
    )}, ensure_ascii=False, indent=2))
    return 0 if attempt.get("status") == "completed" else 1


def _adapter_path(steps: int) -> Path:
    attempt_path = TRAINING_ROOT / _probe_name(steps) / "training_attempt.json"
    if not attempt_path.exists():
        raise SystemExit(f"Phase79 training attempt missing: {attempt_path}")
    attempt = _read_json(attempt_path)
    validation = dict(attempt.get("adapter_validation") or {})
    if attempt.get("status") != "completed" or validation.get("valid") is not True:
        raise SystemExit(f"Phase79 adapter is not valid: {attempt_path}")
    artifact_dir = validation.get("artifact_dir")
    if not artifact_dir:
        raise SystemExit(f"Phase79 adapter path missing: {attempt_path}")
    return Path(str(artifact_dir)).expanduser().resolve()


def _load_runtime(adapter_path: Path | None) -> tuple[Any, Any, Any, str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
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


def _run_session(
    *,
    session: Mapping[str, Any],
    variant: str,
    torch: Any,
    tokenizer: Any,
    model: Any,
    device: str,
    adapter_loaded: bool,
) -> dict[str, Any]:
    model_history: list[dict[str, str]] = []
    routing_history: list[dict[str, str]] = []
    persisted_turns: list[dict[str, str]] = []
    generations = []
    routes = []
    input_guards = []
    output_guards = []
    contract_hashes = []
    private_values = [str(value) for value in session.get("declared_private_values") or []]
    raw_model_private_echo = False
    for turn, user_text in enumerate(
        (
            str(session.get("user_goal") or ""),
            str(session.get("user_correction") or ""),
            str(session.get("continuation_request") or ""),
        ),
        start=1,
    ):
        guarded_user, input_guard = guard_phase77_messages(
            [{"role": "user", "content": user_text}],
            private_values,
        )
        user_message = guarded_user[0]
        model_history.append(user_message)
        routing_history.append(user_message)
        persisted_turns.append(user_message)
        input_guard["applied_before_model_call"] = True
        input_guard["model_input_contains_declared_private_value"] = any(
            value in str(row.get("content") or "")
            for value in private_values
            for row in model_history
        )
        input_guard["passed"] = not input_guard["model_input_contains_declared_private_value"]
        if variant == "runtime_reference":
            contract, route = contract_for_phase77_messages(routing_history)
        else:
            contract = PHASE75_MINIMAL_CONTRACT
            route = {
                "kind": "phase79_persona_route",
                "routed": False,
                "reason": "minimal_contract_training_comparison_arm",
                "latest_user_sha256": stable_hash(user_message["content"].strip()),
                "raw_user_text_persisted": False,
            }
        messages = [{"role": "system", "content": contract}, *model_history]
        answer, info = phase75_driver._generate_one(
            torch,
            tokenizer,
            model,
            device,
            messages,
        )
        returned_answer, output_guard = guard_phase77_output(answer, private_values)
        raw_model_private_echo = raw_model_private_echo or bool(
            output_guard["raw_model_private_echo_detected"]
        )
        assistant = {"role": "assistant", "content": returned_answer}
        model_history.append(assistant)
        routing_history.append(assistant)
        persisted_turns.append(assistant)
        routes.append({"turn": turn, **route})
        input_guards.append({"turn": turn, **input_guard})
        output_guards.append({"turn": turn, **output_guard})
        contract_hashes.append(hashlib.sha256(contract.encode()).hexdigest())
        generations.append({"turn": turn, **info})
    return {
        "kind": "phase79_real_multiturn_transcript",
        "session_id": session.get("session_id"),
        "category": session.get("category"),
        "task_type": session.get("task_type"),
        "variant": variant,
        "model_id": str(MODEL_PATH),
        "adapter_loaded": adapter_loaded,
        "runtime_reference": variant == "runtime_reference",
        "device": device,
        "actual_model_call": True,
        "hardcoded_response": False,
        "status": "completed",
        "turns": persisted_turns,
        "route_manifests": routes,
        "private_input_guards": input_guards,
        "private_output_guards": output_guards,
        "system_contract_sha256_by_turn": contract_hashes,
        "generation": generations,
        "truncated_response": any(row["truncated"] for row in generations),
        "think_leak_detected": any(row["think_leak_detected"] for row in generations),
        "privacy_canary_echo_detected": raw_model_private_echo,
        "returned_private_value_detected": any(
            row["returned_output_contains_declared_private_value"] for row in output_guards
        ),
        "simulated_usage": True,
        "actual_user_feedback": False,
        "actual_product_benefit_claim_allowed": False,
        "created_at": _utcnow(),
    }


def _release_runtime(torch: Any, model: Any, device: str) -> None:
    try:
        del model
        if device == "mps":
            torch.mps.empty_cache()
    except Exception:
        pass


def _sanity(clean: bool) -> int:
    probe_dir = TRAINING_ROOT / "probe-12step"
    output_path = probe_dir / "sanity_transcripts.jsonl"
    report_path = probe_dir / "sanity_report.json"
    if clean:
        output_path.unlink(missing_ok=True)
        report_path.unlink(missing_ok=True)
    adapter_path = _adapter_path(12)
    sessions = [dict(row) for row in _read_json(PREPARATION_ROOT / "holdout.json").get("sessions") or []]
    selected = []
    for category in (*PHASE78_PERSONA_CATEGORIES, "ordinary_direct"):
        selected.append(next(row for row in sessions if row.get("category") == category))
    torch, tokenizer, model, device = _load_runtime(adapter_path)
    rows = []
    try:
        for index, session in enumerate(selected, start=1):
            try:
                row = _run_session(
                    session=session,
                    variant="adapter_minimal_guarded",
                    torch=torch,
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                    adapter_loaded=True,
                )
            except Exception as exc:
                row = {
                    "kind": "phase79_sanity_transcript",
                    "session_id": session.get("session_id"),
                    "category": session.get("category"),
                    "variant": "adapter_minimal_guarded",
                    "actual_model_call": False,
                    "status": "failed",
                    "error": f"{exc.__class__.__name__}: {exc}",
                    "turns": [],
                }
            rows.append(row)
            _write_jsonl(output_path, rows)
            print(f"[sanity] {index}/{len(selected)} {row['session_id']} {row['status']}", flush=True)
    finally:
        _release_runtime(torch, model, device)
    metrics = aggregate_phase75_variant(rows, selected)
    checks = {
        "seven_sessions_completed": len(rows) == 7
        and all(row.get("status") == "completed" for row in rows),
        "all_real_model_calls": all(row.get("actual_model_call") is True for row in rows),
        "no_empty_transcripts": all(len(row.get("turns") or []) == 6 for row in rows),
        "no_truncation": not any(row.get("truncated_response") for row in rows),
        "no_private_echo": not any(row.get("privacy_canary_echo_detected") for row in rows),
        "no_returned_private_value": not any(row.get("returned_private_value_detected") for row in rows),
    }
    report = {
        "kind": "phase79_12_step_adapter_sanity_report",
        "passed": all(checks.values()),
        "checks": checks,
        "session_count": len(rows),
        "metrics": metrics,
        "adapter_path": str(adapter_path),
        "simulated_usage": True,
        "actual_product_benefit_claim_allowed": False,
    }
    _write_json(report_path, report)
    print(json.dumps({"passed": report["passed"], "checks": checks}, indent=2))
    return 0 if report["passed"] else 1


def _sanity_diagnostic(clean: bool) -> int:
    probe_dir = TRAINING_ROOT / "probe-12step"
    adapter_rows = _read_jsonl(probe_dir / "sanity_transcripts.jsonl")
    adapter_report = _read_json(probe_dir / "sanity_report.json")
    if adapter_report.get("passed") is not False or len(adapter_rows) != 7:
        raise SystemExit("Phase79 sanity diagnostic requires the saved failed 7-session adapter sanity")
    sessions = [dict(row) for row in _read_json(PREPARATION_ROOT / "holdout.json").get("sessions") or []]
    selected = []
    for category in (*PHASE78_PERSONA_CATEGORIES, "ordinary_direct"):
        selected.append(next(row for row in sessions if row.get("category") == category))
    outputs: dict[str, list[dict[str, Any]]] = {
        "adapter_minimal_guarded": adapter_rows,
    }
    torch, tokenizer, model, device = _load_runtime(None)
    try:
        for variant in ("base_minimal_guarded", "runtime_reference"):
            output_path = probe_dir / f"sanity_transcripts_{variant}.jsonl"
            if clean:
                output_path.unlink(missing_ok=True)
            rows = []
            for index, session in enumerate(selected, start=1):
                row = _run_session(
                    session=session,
                    variant=variant,
                    torch=torch,
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                    adapter_loaded=False,
                )
                rows.append(row)
                _write_jsonl(output_path, rows)
                print(f"[sanity-diagnostic:{variant}] {index}/{len(selected)} {row['session_id']}", flush=True)
            outputs[variant] = rows
    finally:
        _release_runtime(torch, model, device)
    metrics = {
        variant: aggregate_phase75_variant(rows, selected)
        for variant, rows in outputs.items()
    }
    target_scores = {
        variant: round(
            sum(
                float(row.get("composite_personalization_score") or 0.0)
                for name, row in dict(summary.get("category_metrics") or {}).items()
                if name != "ordinary_direct"
            ) / len(PHASE78_PERSONA_CATEGORIES),
            4,
        )
        for variant, summary in metrics.items()
    }
    truncation_rates = {
        variant: round(sum(bool(row.get("truncated_response")) for row in rows) / len(rows), 4)
        for variant, rows in outputs.items()
    }
    checks = {
        "all_21_sessions_completed": sum(len(rows) for rows in outputs.values()) == 21
        and all(row.get("status") == "completed" for rows in outputs.values() for row in rows),
        "all_real_model_calls": all(
            row.get("actual_model_call") is True for rows in outputs.values() for row in rows
        ),
        "adapter_sanity_failure_preserved": adapter_report.get("passed") is False,
        "adapter_truncation_reproduced": truncation_rates["adapter_minimal_guarded"] > 0.0,
        "no_private_echo_all_arms": not any(
            row.get("privacy_canary_echo_detected") for rows in outputs.values() for row in rows
        ),
    }
    report = {
        "kind": "phase79_same_scene_sanity_diagnostic",
        "passed": all(checks.values()),
        "checks": checks,
        "session_count_per_arm": 7,
        "real_model_call_count": 63,
        "target_scores": target_scores,
        "truncation_rates": truncation_rates,
        "adapter_target_gain_vs_base": round(
            target_scores["adapter_minimal_guarded"] - target_scores["base_minimal_guarded"], 4
        ),
        "adapter_gap_to_runtime": round(
            target_scores["adapter_minimal_guarded"] - target_scores["runtime_reference"], 4
        ),
        "metrics": metrics,
        "full_training_started": False,
        "full_training_blocked_by_sanity": True,
        "simulated_usage": True,
        "actual_user_feedback": False,
        "actual_product_benefit_claim_allowed": False,
    }
    _write_json(probe_dir / "sanity_diagnostic.json", report)
    print(json.dumps({
        "passed": report["passed"],
        "target_scores": target_scores,
        "truncation_rates": truncation_rates,
        "adapter_target_gain_vs_base": report["adapter_target_gain_vs_base"],
    }, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


def _generation_freeze_check() -> dict[str, Any]:
    training = _read_json(TRAINING_ROOT / "candidate-full-120step/training_attempt.json")
    validation = dict(training.get("adapter_validation") or {})
    adapter_path = Path(str(validation.get("adapter_path") or ""))
    freeze = _training_freeze_check()
    holdout = _read_json(PREPARATION_ROOT / "holdout.json")
    source_freeze = _read_json(EVIDENCE_ROOT / "pre_training_freeze.json")
    checks = {
        "training_freeze_passed": freeze["passed"],
        "full_training_completed": training.get("status") == "completed"
        and training.get("real_training") is True,
        "candidate_eligible": training.get("candidate_eligible") is True,
        "adapter_valid": validation.get("valid") is True,
        "adapter_hash_unchanged": adapter_path.is_file()
        and _sha256(adapter_path) == validation.get("sha256"),
        "full_exposure": dict(training.get("exposure") or {}).get("full_coverage") is True,
        "holdout_unchanged": stable_hash(holdout.get("sessions") or [])
        == source_freeze.get("holdout_manifest_sha256"),
        "protocol_unchanged": stable_hash(GENERATION_PROTOCOL)
        == source_freeze.get("generation_protocol_sha256"),
        "historical_adapter_not_reused": training.get("historical_adapter_reused") is False,
    }
    return {"kind": "phase79_generation_freeze_check", "passed": all(checks.values()), "checks": checks}


def _generate(variant: str, clean: bool) -> int:
    if variant not in PHASE79_VARIANTS:
        raise SystemExit(f"unsupported Phase79 variant: {variant}")
    freeze = _generation_freeze_check()
    if not freeze["passed"]:
        raise SystemExit(f"Phase79 generation freeze failed: {freeze}")
    sessions = [dict(row) for row in _read_json(PREPARATION_ROOT / "holdout.json").get("sessions") or []]
    output_path = GENERATION_ROOT / f"transcripts_{variant}.jsonl"
    metrics_path = GENERATION_ROOT / f"metrics_{variant}.json"
    if clean:
        output_path.unlink(missing_ok=True)
        metrics_path.unlink(missing_ok=True)
    existing = [] if clean else _read_jsonl(output_path)
    completed = {str(row.get("session_id")) for row in existing if row.get("status") == "completed"}
    valid_ids = {str(item["session_id"]) for item in sessions}
    transcripts = [row for row in existing if str(row.get("session_id")) in valid_ids]
    adapter_path = _adapter_path(PHASE78_TRAINING_SAMPLE_COUNT) if variant == "adapter_minimal_guarded" else None
    torch, tokenizer, model, device = _load_runtime(adapter_path)
    try:
        for index, session in enumerate(sessions, start=1):
            session_id = str(session["session_id"])
            if session_id in completed:
                print(f"[{variant}] {index}/{len(sessions)} {session_id} resumed", flush=True)
                continue
            try:
                transcript = _run_session(
                    session=session,
                    variant=variant,
                    torch=torch,
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                    adapter_loaded=adapter_path is not None,
                )
            except Exception as exc:
                transcript = {
                    "kind": "phase79_real_multiturn_transcript",
                    "session_id": session_id,
                    "category": session.get("category"),
                    "task_type": session.get("task_type"),
                    "variant": variant,
                    "model_id": str(MODEL_PATH),
                    "adapter_loaded": adapter_path is not None,
                    "device": device,
                    "actual_model_call": False,
                    "status": "failed",
                    "error": f"{exc.__class__.__name__}: {exc}",
                    "turns": [],
                    "simulated_usage": True,
                    "actual_user_feedback": False,
                    "created_at": _utcnow(),
                }
            transcripts = [row for row in transcripts if row.get("session_id") != session_id]
            transcripts.append(transcript)
            transcripts.sort(key=lambda row: str(row.get("session_id")))
            _write_jsonl(output_path, transcripts)
            print(f"[{variant}] {index}/{len(sessions)} {session_id} {transcript['status']}", flush=True)
    finally:
        _release_runtime(torch, model, device)
    metrics = aggregate_phase75_variant(transcripts, sessions)
    metrics.update({
        "kind": "phase79_variant_metrics",
        "variant": variant,
        "model_id": str(MODEL_PATH),
        "adapter_loaded": adapter_path is not None,
        "runtime_reference": variant == "runtime_reference",
        "model_call_count": sum(
            len(row.get("generation") or []) for row in transcripts if row.get("actual_model_call") is True
        ),
        "all_sessions_completed": len(transcripts) == len(sessions)
        and all(row.get("status") == "completed" for row in transcripts),
        "truncated_session_rate": round(
            sum(bool(row.get("truncated_response")) for row in transcripts) / len(transcripts), 4
        ) if transcripts else 0.0,
        "think_leak_rate": round(
            sum(bool(row.get("think_leak_detected")) for row in transcripts) / len(transcripts), 4
        ) if transcripts else 0.0,
        "model_input_private_value_session_rate": round(
            sum(
                any(bool(item.get("model_input_contains_declared_private_value"))
                    for item in row.get("private_input_guards") or [])
                for row in transcripts
            ) / len(transcripts), 4
        ) if transcripts else 0.0,
        "returned_private_value_session_rate": round(
            sum(bool(row.get("returned_private_value_detected")) for row in transcripts)
            / len(transcripts), 4
        ) if transcripts else 0.0,
        "simulated_usage": True,
        "actual_user_feedback": False,
        "actual_product_benefit_claim_allowed": False,
    })
    _write_json(GENERATION_ROOT / f"freeze_check_{variant}.json", freeze)
    _write_json(metrics_path, metrics)
    print(json.dumps({key: metrics.get(key) for key in (
        "variant", "session_count", "model_call_count", "personalization_score",
        "hard_gate_pass_rate", "privacy_canary_echo_rate", "unsupported_claim_rate",
    )}, ensure_ascii=False, indent=2))
    return 0 if metrics["all_sessions_completed"] else 1


def _prepare_eval() -> int:
    sessions = [dict(row) for row in _read_json(PREPARATION_ROOT / "holdout.json").get("sessions") or []]
    transcripts = {
        variant: _read_jsonl(GENERATION_ROOT / f"transcripts_{variant}.jsonl")
        for variant in PHASE79_VARIANTS
    }
    generation_checks = {
        f"{variant}_48_completed": len(rows) == PHASE79_SESSION_COUNT
        and all(row.get("status") == "completed" and row.get("actual_model_call") is True for row in rows)
        for variant, rows in transcripts.items()
    }
    if not all(generation_checks.values()):
        raise SystemExit(f"Phase79 generation incomplete: {generation_checks}")
    blind = build_phase78_blind_pairs(transcripts, sessions, seed=79)
    public_audit = audit_phase78_public_private_values(blind["public_pairs"], sessions)
    if not public_audit["passed"]:
        (JUDGE_ROOT / "blind_items_public.jsonl").unlink(missing_ok=True)
        raise SystemExit("Phase79 public blind package contains a declared private value")
    deterministic_results = score_phase78_blind_pairs_deterministic(blind, sessions)
    deterministic = summarize_phase78_blind_results(
        deterministic_results,
        blind["hidden_key"],
        blind["public_pairs"],
    )
    deterministic.update({
        "status": "completed",
        "judge": "phase79_frozen_deterministic_rubric",
        "actual_model_calls": False,
        "completed_pair_count": len(deterministic_results),
        "failure_count": 0,
    })
    _write_jsonl(JUDGE_ROOT / "blind_items_public.jsonl", blind["public_pairs"])
    _write_json(JUDGE_ROOT / "blind_hidden_key.json", {"hidden_key": blind["hidden_key"]})
    _write_jsonl(JUDGE_ROOT / "deterministic_results.jsonl", deterministic_results)
    _write_json(JUDGE_ROOT / "deterministic_summary.json", deterministic)
    _write_json(JUDGE_ROOT / "public_blind_package_audit.json", public_audit)
    pair_count = PHASE79_SESSION_COUNT * len(PHASE78_COMPARISONS)
    freeze_checks = {
        "all_generation_complete": all(generation_checks.values()),
        "pair_count_96": blind["pair_count"] == pair_count,
        "public_private_audit_passed": public_audit["passed"],
        "deterministic_result_count_96": len(deterministic_results) == pair_count,
        "deterministic_invalid_zero": deterministic["invalid_result_count"] == 0,
    }
    freeze = {
        "kind": "phase79_pre_judge_freeze",
        "frozen_at": _utcnow(),
        "frozen_before_judge_calls": True,
        "passed": all(freeze_checks.values()),
        "checks": freeze_checks,
        "generation_checks": generation_checks,
        "blind_pair_count": blind["pair_count"],
        "public_items_sha256": stable_hash(blind["public_pairs"]),
        "hidden_key_sha256": stable_hash(blind["hidden_key"]),
        "judge_prompt_sha256": hashlib.sha256(phase78_driver._judge_prompt_template().encode()).hexdigest(),
        "judge_models": list(JUDGE_MODELS),
        "identity_hidden": True,
        "score_or_gate_relaxation_allowed": False,
    }
    _write_json(JUDGE_ROOT / "pre_judge_freeze.json", freeze)
    print(json.dumps({
        "status": "ready" if freeze["passed"] else "blocked",
        "pair_count": blind["pair_count"],
        "public_private_audit_passed": public_audit["passed"],
        "generation_checks": generation_checks,
    }, indent=2))
    return 0 if freeze["passed"] else 1


def _judge(model: str, endpoint: str, timeout: int, clean: bool) -> int:
    if model not in JUDGE_MODELS:
        raise SystemExit(f"Phase79 requires one of {JUDGE_MODELS}, got {model}")
    freeze = _read_json(JUDGE_ROOT / "pre_judge_freeze.json")
    pairs = _read_jsonl(JUDGE_ROOT / "blind_items_public.jsonl")
    hidden = _read_json(JUDGE_ROOT / "blind_hidden_key.json").get("hidden_key") or []
    pair_count = PHASE79_SESSION_COUNT * len(PHASE78_COMPARISONS)
    checks = {
        "pre_judge_freeze_passed": freeze.get("passed") is True,
        "public_items_unchanged": stable_hash(pairs) == freeze.get("public_items_sha256"),
        "hidden_key_unchanged": stable_hash(hidden) == freeze.get("hidden_key_sha256"),
        "judge_prompt_unchanged": hashlib.sha256(phase78_driver._judge_prompt_template().encode()).hexdigest()
        == freeze.get("judge_prompt_sha256"),
        "pair_count_96": len(pairs) == pair_count,
    }
    if not all(checks.values()):
        raise SystemExit(f"Phase79 judge freeze failed: {checks}")
    slug = re.sub(r"[^a-z0-9]+", "-", model.lower()).strip("-")
    result_path = JUDGE_ROOT / f"judge_results_{slug}.jsonl"
    summary_path = JUDGE_ROOT / f"judge_summary_{slug}.json"
    if clean:
        result_path.unlink(missing_ok=True)
        summary_path.unlink(missing_ok=True)
    results = [] if clean else _read_jsonl(result_path)
    done = {str(row.get("pair_id")) for row in results if row.get("actual_model_call") is True}
    failures = []
    for index, pair in enumerate(pairs, start=1):
        pair_id = str(pair["pair_id"])
        if pair_id in done:
            print(f"[{model}] {index}/{len(pairs)} {pair_id} resumed", flush=True)
            continue
        try:
            result = phase78_driver._ollama_judge(pair, model, endpoint, timeout)
            results.append(result)
            _write_jsonl(result_path, results)
            print(f"[{model}] {index}/{len(pairs)} {pair_id} {result['winner']}", flush=True)
        except Exception as exc:
            failure = {
                "pair_id": pair_id,
                "error": f"{exc.__class__.__name__}: {exc}",
                "created_at": _utcnow(),
            }
            failures.append(failure)
            print(f"[{model}] {index}/{len(pairs)} {pair_id} failed: {failure['error']}", flush=True)
    summary = summarize_phase78_blind_results(results, hidden, pairs)
    complete = len(results) == len(pairs) and not failures and summary["invalid_result_count"] == 0
    summary.update({
        "status": "completed" if complete else "blocked",
        "judge": "independent_ollama_blind_judge",
        "judge_model": model,
        "actual_model_calls": bool(results),
        "completed_pair_count": len(results),
        "expected_pair_count": len(pairs),
        "failure_count": len(failures),
        "failures": failures,
        "identity_hidden_from_judge": True,
        "fabricated_scores": False,
    })
    _write_json(summary_path, summary)
    if failures:
        _write_json(FAILURE_ROOT / f"judge_failures_{slug}.json", {
            "kind": "phase79_judge_failures",
            "judge_model": model,
            "failures": failures,
        })
    return 0 if complete else 1


def _collect_metrics() -> dict[str, dict[str, Any]]:
    return {
        variant: _read_json(GENERATION_ROOT / f"metrics_{variant}.json")
        for variant in PHASE79_VARIANTS
    }


def _judge_summaries() -> dict[str, dict[str, Any]]:
    return {
        model: _read_json(
            JUDGE_ROOT / f"judge_summary_{re.sub(r'[^a-z0-9]+', '-', model.lower()).strip('-')}.json"
        )
        for model in JUDGE_MODELS
    }


def _output_examples(transcripts: Mapping[str, list[dict[str, Any]]]) -> str:
    selected = (
        "phase79-evidence_truthfulness-01",
        "phase79-latest_action_switch-01",
        "phase79-provenance_labeling-01",
        "phase79-autonomous_execution-01",
        "phase79-concise_workstyle-01",
        "phase79-privacy_non_echo-01",
        "phase79-ordinary_direct-01",
    )
    by_variant = {
        variant: {str(row.get("session_id")): row for row in rows}
        for variant, rows in transcripts.items()
    }
    lines = [
        "# Phase79 Output Examples",
        "",
        (
            "All answers below are real local Qwen2.5-0.5B-Instruct outputs from simulated_usage "
            "sessions. Declared synthetic private values were replaced before model calls."
        ),
        "",
    ]
    for session_id in selected:
        lines.extend((f"## {session_id}", ""))
        for variant in PHASE79_VARIANTS:
            row = by_variant[variant][session_id]
            final = [
                str(turn.get("content") or "")
                for turn in row.get("turns") or []
                if turn.get("role") == "assistant"
            ][-1]
            lines.extend((f"### {variant}", "", final, ""))
    return "\n".join(lines)


def _evidence_manifest() -> dict[str, Any]:
    files = []
    for path in sorted(EVIDENCE_ROOT.rglob("*")):
        if not path.is_file() or path.name in DYNAMIC_FILES:
            continue
        files.append({
            "path": str(path.relative_to(REPO_ROOT)),
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        })
    return {
        "kind": "phase79_evidence_manifest",
        "files": files,
        "file_count": len(files),
        "manifest_sha256": stable_hash(files),
    }


def _finalize() -> int:
    metrics = _collect_metrics()
    judges = _judge_summaries()
    deterministic = _read_json(JUDGE_ROOT / "deterministic_summary.json")
    training = _read_json(TRAINING_ROOT / "candidate-full-120step/training_attempt.json")
    quality = _read_json(PREPARATION_ROOT / "training_quality_audit.json")
    isolation = _read_json(PREPARATION_ROOT / "isolation_audit.json")
    boundary = _read_json(PREPARATION_ROOT / "completion_boundary_report.json")
    public_audit = _read_json(JUDGE_ROOT / "public_blind_package_audit.json")
    phase78_archive = _read_json(PHASE78_ROOT / "phase78-final-decision.json")
    phase32 = _read_json(PREPARATION_ROOT / "phase32_overclaim_audit.json")
    regression = _read_json(EVIDENCE_ROOT / "full_regression_summary.json")
    prerequisites = {
        "all_three_generation_arms_complete": all(
            row.get("all_sessions_completed") is True for row in metrics.values()
        ),
        "real_full_training_complete": training.get("status") == "completed"
        and training.get("real_training") is True,
        "deterministic_complete": deterministic.get("status") == "completed",
        "gemma_judge_complete": judges["gemma4:31b"].get("status") == "completed",
        "qwen_judge_complete": judges["qwen3.6"].get("status") == "completed",
        "full_regression_passed": regression.get("passed") is True,
    }
    if not all(prerequisites.values()):
        raise SystemExit(f"Phase79 finalization prerequisites failed: {prerequisites}")
    decision = build_phase79_decision(
        metrics=metrics,
        training_attempt=training,
        quality_audit=quality,
        isolation_audit=isolation,
        completion_boundary=boundary,
        public_private_audit=public_audit,
        deterministic=deterministic,
        independent=judges,
        phase78_archive=phase78_archive,
        phase32_audit=phase32,
    )
    target_scores = {
        variant: round(
            sum(
                float(row.get("composite_personalization_score") or 0.0)
                for name, row in dict(metrics[variant].get("category_metrics") or {}).items()
                if name != "ordinary_direct"
            ) / len(PHASE78_PERSONA_CATEGORIES),
            4,
        )
        for variant in PHASE79_VARIANTS
    }
    comparison = {
        "kind": "phase79_cpu_feasible_persona_adapter_comparison",
        "created_at": _utcnow(),
        "model": PHASE79_MODEL_NAME,
        "training_steps": training.get("requested_steps"),
        "training_sample_count": PHASE78_TRAINING_SAMPLE_COUNT,
        "holdout_session_count_per_arm": PHASE79_SESSION_COUNT,
        "real_generation_model_call_count": sum(
            int(row.get("model_call_count") or 0) for row in metrics.values()
        ),
        "real_independent_judge_call_count": sum(
            int(row.get("completed_pair_count") or 0) for row in judges.values()
        ),
        "target_scores": target_scores,
        "adapter_target_gain_vs_base": round(
            target_scores["adapter_minimal_guarded"] - target_scores["base_minimal_guarded"], 4
        ),
        "adapter_gap_to_runtime_reference": round(
            target_scores["adapter_minimal_guarded"] - target_scores["runtime_reference"], 4
        ),
        "metrics": metrics,
        "deterministic_blind": deterministic,
        "independent_blind": judges,
        "training_attempt": training,
        "phase78_archive": phase78_archive,
        "phase32_overclaim_audit": phase32,
        "decision": decision,
        "simulated_usage": True,
        "actual_user_feedback_count": 0,
        "actual_product_benefit_claim_allowed": False,
    }
    transcripts = {
        variant: _read_jsonl(GENERATION_ROOT / f"transcripts_{variant}.jsonl")
        for variant in PHASE79_VARIANTS
    }
    _write_json(EVIDENCE_ROOT / "phase79-final-decision.json", decision)
    _write_json(EVIDENCE_ROOT / "comparison_summary.json", comparison)
    _write_text(EVIDENCE_ROOT / "output_examples.md", _output_examples(transcripts))
    _write_text(EVIDENCE_ROOT / "phase79-final-decision.md", f"""# Phase79 Final Decision

Recommendation: **{decision['recommendation']}**

- Lifecycle status: `{decision['status']}`
- Real {PHASE79_MODEL_NAME} training: `{training.get('status')}` at `{training.get('requested_steps')}` steps
- Base persona-target score: `{decision['base_target_score']}`
- Adapter persona-target score: `{decision['adapter_target_score']}`
- Runtime-contract reference score: `{decision['runtime_reference_target_score']}`
- Adapter gain over base: `{decision['adapter_target_gain']}`
- Adapter gap to same-model runtime reference: `{decision['adapter_gap_to_runtime']}`
- Real generation calls: `{comparison['real_generation_model_call_count']}`
- Real independent judge calls: `{comparison['real_independent_judge_call_count']}`
- Failed checks: `{decision['failed_checks']}`

This is a privacy-safe `simulated_usage` laboratory benchmark. Phase79 proves real CPU training separately from adapter benefit. It does not contain `actual_user_feedback`, does not reuse the historical Phase32 adapter, and does not inherit Phase32's product-benefit claim. It cannot auto-promote, attach Hermes, or change product defaults.
""")
    _write_text(EVIDENCE_ROOT / "phase79-runbook.md", """# Phase79 Runbook

```bash
.venv/bin/python tools/phase79_cpu_feasible_persona_probe.py prepare --clean
.venv/bin/python tools/phase79_cpu_feasible_persona_probe.py train --steps 12 --clean
.venv/bin/python tools/phase79_cpu_feasible_persona_probe.py sanity --clean
.venv/bin/python tools/phase79_cpu_feasible_persona_probe.py train --steps 120 --clean
.venv/bin/python tools/phase79_cpu_feasible_persona_probe.py generate --variant base_minimal_guarded --clean
.venv/bin/python tools/phase79_cpu_feasible_persona_probe.py generate --variant adapter_minimal_guarded --clean
.venv/bin/python tools/phase79_cpu_feasible_persona_probe.py generate --variant runtime_reference --clean
.venv/bin/python tools/phase79_cpu_feasible_persona_probe.py prepare-eval
.venv/bin/python tools/phase79_cpu_feasible_persona_probe.py judge --model gemma4:31b
.venv/bin/python tools/phase79_cpu_feasible_persona_probe.py judge --model qwen3.6
.venv/bin/python tools/phase79_cpu_feasible_persona_probe.py full-regression
.venv/bin/python tools/phase79_cpu_feasible_persona_probe.py finalize
.venv/bin/python tools/phase79_cpu_feasible_persona_probe.py validate
```

The 120-row Phase78 curriculum and fresh 48-session Phase79 holdout are frozen before training. The deterministic private-value guard runs before every model call in all three arms. The runtime arm uses the same selected model plus the Phase77 conditional contract, so adapter and prompt behavior are compared on the same base. Judge commands are resumable and never receive variant identity.
""")
    next_goal = (
        "Design Phase80 as a limited, consented actual-usage capture protocol with a kill switch and "
        "manual acceptance review. Keep Phase77 privacy guards and do not claim product benefit until "
        "real opted-in feedback confirms the simulated result."
        if decision["status"] == "qualified_simulated_cpu_persona_adapter"
        else
        "Develop Phase80 small-model failure taxonomy from the saved base, adapter, and runtime outputs. "
        "Separate capacity failures from curriculum and decoding failures, revise only the failed dimensions, "
        "and use a fresh holdout without changing Phase79 gates."
    )
    _write_text(EVIDENCE_ROOT / "next-pursuit-goal.md", f"# Phase80 Pursuit Goal\n\n{next_goal}")
    _write_json(EVIDENCE_ROOT / "phase79-result-taxonomy.json", {
        "kind": "phase79_result_taxonomy",
        "training_proof": "real_cpu_training_completed",
        "adapter_benefit": decision["status"],
        "runtime_reference": "same_model_phase77_guarded_conditional_runtime",
        "phase77_quality_ceiling": "historical_qwen3_4b_guarded_runtime_reference",
        "actual_user_feedback": "absent",
        "product_benefit": "not_established",
        "promotion": "forbidden",
        "next_gate": decision["next_gate"],
    })
    manifest = _evidence_manifest()
    _write_json(EVIDENCE_ROOT / "evidence_manifest.json", manifest)
    integrity_checks = {
        "preparation_quality_passed": quality.get("passed") is True,
        "holdout_isolation_passed": isolation.get("passed") is True,
        "completion_boundary_passed": boundary.get("passed") is True,
        "public_private_audit_passed": public_audit.get("passed") is True,
        "real_training_recorded": training.get("real_training") is True,
        "historical_adapter_not_reused": training.get("historical_adapter_reused") is False,
        "phase32_overclaim_audited": phase32.get("passed") is True,
        "decision_has_no_auto_promotion": decision.get("auto_promotion_allowed") is False,
        "decision_has_no_actual_product_claim": decision.get("actual_product_benefit_claim_allowed") is False,
        "full_regression_passed": regression.get("passed") is True,
    }
    _write_json(EVIDENCE_ROOT / "evidence_integrity.json", {
        "kind": "phase79_evidence_integrity",
        "passed": all(integrity_checks.values()),
        "checks": integrity_checks,
        "manifest_sha256": manifest["manifest_sha256"],
    })
    _write_json(EVIDENCE_ROOT / "finalization_state.json", {
        "kind": "phase79_finalization_state",
        "status": "completed",
        "decision": decision["recommendation"],
        "created_at": _utcnow(),
    })
    print(json.dumps({
        "recommendation": decision["recommendation"],
        "status": decision["status"],
        "adapter_target_gain": decision["adapter_target_gain"],
        "failed_checks": decision["failed_checks"],
    }, ensure_ascii=False, indent=2))
    return 0


def _finalize_sanity_blocked() -> int:
    probe_dir = TRAINING_ROOT / "probe-12step"
    training = _read_json(probe_dir / "training_attempt.json")
    sanity = _read_json(probe_dir / "sanity_report.json")
    diagnostic = _read_json(probe_dir / "sanity_diagnostic.json")
    regression = _read_json(EVIDENCE_ROOT / "full_regression_summary.json")
    full_attempt = TRAINING_ROOT / "candidate-full-120step/training_attempt.json"
    prerequisites = {
        "real_12_step_training_complete": training.get("status") == "completed"
        and training.get("real_training") is True,
        "adapter_artifact_valid": dict(training.get("adapter_validation") or {}).get("valid") is True,
        "sanity_gate_failed": sanity.get("passed") is False,
        "same_scene_diagnostic_complete": diagnostic.get("passed") is True,
        "full_120_step_not_started": not full_attempt.exists(),
        "full_regression_passed": regression.get("passed") is True,
    }
    if not all(prerequisites.values()):
        raise SystemExit(f"Phase79 sanity-blocked prerequisites failed: {prerequisites}")
    decision = build_phase79_sanity_blocked_decision(
        training_attempt=training,
        sanity_report=sanity,
        sanity_diagnostic=diagnostic,
    )
    comparison = {
        "kind": "phase79_sanity_blocked_comparison",
        "created_at": _utcnow(),
        "model": PHASE79_MODEL_NAME,
        "training_steps": 12,
        "training_sample_count": PHASE78_TRAINING_SAMPLE_COUNT,
        "sanity_session_count_per_arm": 7,
        "real_generation_model_call_count": diagnostic.get("real_model_call_count"),
        "full_holdout_generation_model_call_count": 0,
        "independent_judge_call_count": 0,
        "target_scores": diagnostic.get("target_scores"),
        "truncation_rates": diagnostic.get("truncation_rates"),
        "adapter_target_gain_vs_base": diagnostic.get("adapter_target_gain_vs_base"),
        "adapter_gap_to_runtime": diagnostic.get("adapter_gap_to_runtime"),
        "training_attempt": training,
        "sanity_report": sanity,
        "sanity_diagnostic": diagnostic,
        "decision": decision,
        "simulated_usage": True,
        "actual_user_feedback_count": 0,
        "actual_product_benefit_claim_allowed": False,
    }
    rows = {
        "adapter_minimal_guarded": _read_jsonl(probe_dir / "sanity_transcripts.jsonl"),
        "base_minimal_guarded": _read_jsonl(probe_dir / "sanity_transcripts_base_minimal_guarded.jsonl"),
        "runtime_reference": _read_jsonl(probe_dir / "sanity_transcripts_runtime_reference.jsonl"),
    }
    _write_json(EVIDENCE_ROOT / "phase79-final-decision.json", decision)
    _write_json(EVIDENCE_ROOT / "comparison_summary.json", comparison)
    _write_text(EVIDENCE_ROOT / "output_examples.md", _output_examples(rows))
    _write_text(EVIDENCE_ROOT / "phase79-final-decision.md", f"""# Phase79 Final Decision

Recommendation: **{decision['recommendation']}**

- Lifecycle status: `{decision['status']}`
- Real {PHASE79_MODEL_NAME} 12-step training: `completed`
- Adapter artifact: `valid`
- 12-step sanity: `failed`
- Adapter sanity truncation rate: `{dict(diagnostic.get('truncation_rates') or {}).get('adapter_minimal_guarded')}`
- Base sanity target score: `{dict(diagnostic.get('target_scores') or {}).get('base_minimal_guarded')}`
- Adapter sanity target score: `{dict(diagnostic.get('target_scores') or {}).get('adapter_minimal_guarded')}`
- Runtime-contract sanity target score: `{dict(diagnostic.get('target_scores') or {}).get('runtime_reference')}`
- 120-step training: `not started by frozen sanity gate`
- Full 48-session eval: `not reached`
- Independent judge calls: `0`

Phase79 proves that real completion-only LoRA training can complete on CPU for the local 0.5B Qwen model. It does not prove adapter benefit. The first independent sanity exposed truncated repetition and weak persona answers, so the frozen gate correctly stopped the expensive 120-step run. All sessions are `simulated_usage`; there is no `actual_user_feedback`, product-benefit claim, promotion, Hermes attachment, or default change.
""")
    _write_text(EVIDENCE_ROOT / "phase79-runbook.md", """# Phase79 Runbook

```bash
.venv/bin/python tools/phase79_cpu_feasible_persona_probe.py prepare --clean
.venv/bin/python tools/phase79_cpu_feasible_persona_probe.py train --steps 12 --clean
.venv/bin/python tools/phase79_cpu_feasible_persona_probe.py sanity --clean
.venv/bin/python tools/phase79_cpu_feasible_persona_probe.py sanity-diagnostic --clean
.venv/bin/python tools/phase79_cpu_feasible_persona_probe.py full-regression
.venv/bin/python tools/phase79_cpu_feasible_persona_probe.py finalize-sanity-blocked
.venv/bin/python tools/phase79_cpu_feasible_persona_probe.py validate
```

The 12-step probe creates a new Phase79 adapter from the frozen Phase78 privacy-safe curriculum. If sanity fails, the 120-step command is forbidden. The diagnostic compares base, adapter, and same-model Phase77 conditional runtime on the same seven fresh sessions. Training success and benefit remain separate.
""")
    _write_text(EVIDENCE_ROOT / "next-pursuit-goal.md", """# Phase80 Pursuit Goal

Build a small-model failure taxonomy from the saved Phase79 same-scene outputs. Determine whether the 0.5B failure is primarily capacity, curriculum balance, early-step instability, or generation stopping. Use short predeclared probes that can finish on CPU, keep a fresh holdout, and do not resume 120-step training unless an independent sanity gate passes. Preserve the Phase77 runtime contract as the product path and do not claim actual-user benefit.
""")
    _write_json(EVIDENCE_ROOT / "phase79-result-taxonomy.json", {
        "kind": "phase79_result_taxonomy",
        "training_proof": "real_cpu_12_step_training_completed",
        "adapter_benefit": "not_evaluated_on_full_holdout",
        "sanity_result": "failed_truncation_and_weak_persona_behavior",
        "full_training": "blocked_not_started",
        "runtime_reference": "phase77_guarded_conditional_runtime_retained",
        "actual_user_feedback": "absent",
        "product_benefit": "not_established",
        "promotion": "forbidden",
        "next_gate": decision["next_gate"],
    })
    manifest = _evidence_manifest()
    _write_json(EVIDENCE_ROOT / "evidence_manifest.json", manifest)
    integrity_checks = {
        **prerequisites,
        "decision_checks_passed": all(dict(decision.get("checks") or {}).values()),
        "training_and_benefit_separated": decision.get("training_success") is True
        and decision.get("adapter_benefit") == "not_evaluated_on_full_holdout",
        "no_actual_product_claim": decision.get("actual_product_benefit_claim_allowed") is False,
        "no_auto_promotion": decision.get("auto_promotion_allowed") is False,
    }
    _write_json(EVIDENCE_ROOT / "evidence_integrity.json", {
        "kind": "phase79_evidence_integrity",
        "passed": all(integrity_checks.values()),
        "checks": integrity_checks,
        "manifest_sha256": manifest["manifest_sha256"],
    })
    _write_json(EVIDENCE_ROOT / "finalization_state.json", {
        "kind": "phase79_finalization_state",
        "status": "completed",
        "decision": decision["recommendation"],
        "created_at": _utcnow(),
    })
    print(json.dumps({
        "status": decision["status"],
        "recommendation": decision["recommendation"],
        "training_success": decision["training_success"],
        "adapter_benefit": decision["adapter_benefit"],
        "target_scores": diagnostic.get("target_scores"),
    }, ensure_ascii=False, indent=2))
    return 0


def _run_logged(command: list[str]) -> dict[str, Any]:
    started = time.perf_counter()
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    lines = []
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    code = process.wait()
    return {
        "command": command,
        "exit_code": code,
        "duration_seconds": round(time.perf_counter() - started, 4),
        "output": "".join(lines),
    }


def _full_regression() -> int:
    commands = (
        [
            str(REPO_ROOT / ".venv/bin/pytest"),
            "-q",
            "tests/test_phase79_cpu_feasible_persona_probe.py",
            "tests/test_phase78_persona_internalization_training.py",
            "tests/test_phase77_private_value_guarded_runtime.py",
        ],
        ["make", "test-unit", "test-surface", "test-e2e-mock", "smoke-beta"],
    )
    results = []
    for command in commands:
        result = _run_logged(command)
        results.append(result)
        if result["exit_code"] != 0:
            break
    summary = {
        "kind": "phase79_full_regression_summary",
        "created_at": _utcnow(),
        "passed": len(results) == len(commands) and all(row["exit_code"] == 0 for row in results),
        "results": results,
    }
    _write_json(EVIDENCE_ROOT / "full_regression_summary.json", summary)
    return 0 if summary["passed"] else 1


def _validate() -> int:
    manifest = _read_json(EVIDENCE_ROOT / "evidence_manifest.json")
    integrity = _read_json(EVIDENCE_ROOT / "evidence_integrity.json")
    decision = _read_json(EVIDENCE_ROOT / "phase79-final-decision.json")
    blocked = decision.get("status") == "archive_12_step_sanity_failed"
    training = _read_json(TRAINING_ROOT / (
        "probe-12step/training_attempt.json"
        if blocked
        else "candidate-full-120step/training_attempt.json"
    ))
    public_audit_path = JUDGE_ROOT / "public_blind_package_audit.json"
    public_audit = _read_json(public_audit_path) if public_audit_path.exists() else {}
    manifest_failures = []
    for row in manifest.get("files") or []:
        path = REPO_ROOT / str(row.get("path") or "")
        if not path.is_file() or _sha256(path) != row.get("sha256"):
            manifest_failures.append(str(row.get("path") or ""))
    raw_private_locations = []
    for path in EVIDENCE_ROOT.rglob("*"):
        if not path.is_file() or path.name in DYNAMIC_FILES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if "SYNTHETIC_PHASE79_PRIVATE_" in text:
            raw_private_locations.append(str(path.relative_to(EVIDENCE_ROOT)))
    allowed_private_locations = ["evidence-preparation/holdout.json"]
    checks = {
        "manifest_files_unchanged": not manifest_failures,
        "integrity_passed": integrity.get("passed") is True,
        "real_training_completed": training.get("status") == "completed"
        and training.get("real_training") is True,
        "historical_adapter_not_reused": training.get("historical_adapter_reused") is False,
        "public_private_audit_passed_or_not_reached": public_audit.get("passed") is True or blocked,
        "private_canaries_only_in_frozen_holdout": raw_private_locations == allowed_private_locations,
        "no_actual_user_claim": decision.get("actual_product_benefit_claim_allowed") is False,
        "no_auto_promotion": decision.get("auto_promotion_allowed") is False,
        "no_hermes_attachment": decision.get("hermes_attachment_allowed") is False,
        "product_default_unchanged": decision.get("product_default_changed") is False,
    }
    summary = {
        "kind": "phase79_validation_summary",
        "created_at": _utcnow(),
        "passed": all(checks.values()),
        "checks": checks,
        "manifest_failures": manifest_failures,
        "raw_private_locations": raw_private_locations,
    }
    _write_json(EVIDENCE_ROOT / "validation_summary.json", summary)
    _write_text(EVIDENCE_ROOT / "validation_gate.txt", "PASS" if summary["passed"] else "FAIL")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["passed"] else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--clean", action="store_true")

    train = subparsers.add_parser("train")
    train.add_argument("--steps", type=int, required=True)
    train.add_argument("--clean", action="store_true")

    sanity = subparsers.add_parser("sanity")
    sanity.add_argument("--clean", action="store_true")

    sanity_diagnostic = subparsers.add_parser("sanity-diagnostic")
    sanity_diagnostic.add_argument("--clean", action="store_true")

    generate = subparsers.add_parser("generate")
    generate.add_argument("--variant", choices=PHASE79_VARIANTS, required=True)
    generate.add_argument("--clean", action="store_true")

    subparsers.add_parser("prepare-eval")

    judge = subparsers.add_parser("judge")
    judge.add_argument("--model", choices=JUDGE_MODELS, required=True)
    judge.add_argument("--endpoint", default="http://127.0.0.1:11434")
    judge.add_argument("--timeout", type=int, default=300)
    judge.add_argument("--clean", action="store_true")

    subparsers.add_parser("full-regression")
    subparsers.add_parser("finalize")
    subparsers.add_parser("finalize-sanity-blocked")
    subparsers.add_parser("validate")
    args = parser.parse_args()

    if args.command == "prepare":
        return _prepare(args.clean)
    if args.command == "train":
        return _train(args.steps, args.clean)
    if args.command == "sanity":
        return _sanity(args.clean)
    if args.command == "sanity-diagnostic":
        return _sanity_diagnostic(args.clean)
    if args.command == "generate":
        return _generate(args.variant, args.clean)
    if args.command == "prepare-eval":
        return _prepare_eval()
    if args.command == "judge":
        return _judge(args.model, args.endpoint, args.timeout, args.clean)
    if args.command == "full-regression":
        return _full_regression()
    if args.command == "finalize":
        return _finalize()
    if args.command == "finalize-sanity-blocked":
        return _finalize_sanity_blocked()
    if args.command == "validate":
        return _validate()
    raise SystemExit(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
