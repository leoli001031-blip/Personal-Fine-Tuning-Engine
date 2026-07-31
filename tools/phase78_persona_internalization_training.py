#!/usr/bin/env python3
"""Run the Phase78 privacy-safe persona-internalization training probe."""

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
from urllib import request


REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = REPO_ROOT / "pfe-core"
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

import phase75_personalization_benefit_benchmark as phase75_driver
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
    PHASE78_HOLDOUT_SESSION_COUNT,
    PHASE78_PERSONA_CATEGORIES,
    PHASE78_TRAINING_SAMPLE_COUNT,
    PHASE78_VARIANTS,
    audit_phase78_isolation,
    audit_phase78_public_private_values,
    audit_phase78_training_samples,
    build_phase78_blind_pairs,
    build_phase78_decision,
    build_phase78_holdout,
    build_phase78_sft_job_spec,
    build_phase78_training_samples,
    score_phase78_blind_pairs_deterministic,
    summarize_phase78_blind_results,
)
from pfe_core.trainer.executors import (
    _build_sft_prompt_and_text,
    _encode_sft_examples,
    _run_real_local_peft_training,
)


EVIDENCE_ROOT = REPO_ROOT / "docs/demo/phase78-persona-internalization-training-probe"
PREPARATION_ROOT = EVIDENCE_ROOT / "evidence-preparation"
TRAINING_ROOT = EVIDENCE_ROOT / "evidence-real-training"
GENERATION_ROOT = EVIDENCE_ROOT / "evidence-real-generation"
JUDGE_ROOT = EVIDENCE_ROOT / "evidence-blind-eval"
FAILURE_ROOT = EVIDENCE_ROOT / "evidence-failures"
MODEL_PATH = REPO_ROOT / "models/Qwen3-4B"
TRAINER_OUTPUT_ROOT = REPO_ROOT / "trainer_job_outputs"
CORE_SOURCE = CORE_ROOT / "pfe_core/phase78_persona_internalization_training.py"
DRIVER_SOURCE = REPO_ROOT / "tools/phase78_persona_internalization_training.py"
TEST_SOURCE = REPO_ROOT / "tests/test_phase78_persona_internalization_training.py"
EXECUTOR_SOURCE = CORE_ROOT / "pfe_core/trainer/executors.py"
PHASE77_ROOT = REPO_ROOT / "docs/demo/phase77-private-value-guarded-runtime"
JUDGE_MODELS = ("gemma4:31b", "qwen3.6")
GENERATION_PROTOCOL = {
    **phase75_driver.GENERATION_PROTOCOL,
    "kind": "phase78_frozen_generation_protocol",
    "variants": list(PHASE78_VARIANTS),
    "declared_private_values_redacted_before_every_model_call": True,
    "raw_model_output_checked_before_persistence": True,
    "same_decoding_all_arms": True,
    "runtime_reference_uses_phase77_conditional_contract": True,
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


def _model_selection() -> dict[str, Any]:
    config = MODEL_PATH / "config.json"
    shards = sorted(MODEL_PATH.glob("*.safetensors"))
    total_bytes = sum(path.stat().st_size for path in shards)
    try:
        import torch

        torch_version = str(torch.__version__)
        mps_available = bool(torch.backends.mps.is_available())
        runtime_available = True
    except Exception:
        torch_version = None
        mps_available = False
        runtime_available = False
    checks = {
        "local_config_exists": config.exists(),
        "weight_shards_exist": bool(shards),
        "torch_runtime_available": runtime_available,
        "historical_real_training_evidence_exists": (
            REPO_ROOT
            / "docs/demo/phase45-privacy-structural-multiturn-preference/evidence-training-sft/"
            "candidate-a-probe-12step/training_attempt.json"
        ).exists(),
    }
    return {
        "kind": "phase78_model_selection",
        "status": "selected" if all(checks.values()) else "blocked",
        "selected_model": "Qwen3-4B",
        "model_path": str(MODEL_PATH),
        "reason": (
            "The local full-precision Qwen3-4B has already completed real PEFT LoRA probes on this "
            "machine. Phase78 changes the privacy-safe curriculum and benefit gate, not model scale. "
            "The current execution sandbox cannot expose MPS, so the frozen 160-token curriculum also "
            "supports an honest CPU fallback without truncating any sample."
        ),
        "checks": checks,
        "config_sha256": _sha256(config) if config.exists() else None,
        "weight_shard_count": len(shards),
        "weight_bytes": total_bytes,
        "torch_version": torch_version,
        "mps_available_in_current_process": mps_available,
        "selected_execution_device": "mps" if mps_available else "cpu",
        "not_selected": {
            "Qwen3.6-27B": "Phase12 real training hit Metal OOM; it is a runtime reference, not a Phase78 trainer.",
            "Qwen2.5-0.5B": "Prior preference probes were trainable but too weak for this semantic benchmark.",
        },
    }


def _completion_boundary_report(job_spec: Mapping[str, Any]) -> dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), local_files_only=True)
    training = dict(dict(job_spec.get("recipe") or {}).get("training") or {})
    maximum = int(training.get("max_length") or 512)
    examples = [dict(row) for row in job_spec.get("training_examples") or []]
    encoded = _encode_sft_examples(
        tokenizer=tokenizer,
        training_examples=examples,
        max_length=maximum,
        vocab_size=int(getattr(tokenizer, "vocab_size", 0) or 151936),
    )
    details = []
    for source, row in zip(examples, encoded):
        prompt, _ = _build_sft_prompt_and_text(
            tokenizer,
            str(source.get("instruction") or ""),
            str(source.get("chosen") or ""),
            messages=source.get("messages"),
        )
        prompt_tokens = tokenizer(
            prompt,
            truncation=True,
            max_length=maximum,
            add_special_tokens=False,
        ).get("input_ids") or []
        labels = list(row.get("labels") or [])
        completion = [index for index, value in enumerate(labels) if int(value) != -100]
        prompt_boundary = min(len(prompt_tokens), len(labels))
        details.append(
            {
                "sample_id": source.get("sample_id"),
                "taxonomy_dimension": source.get("taxonomy_dimension"),
                "prompt_token_count": prompt_boundary,
                "completion_label_token_count": len(completion),
                "prompt_labels_all_masked": all(
                    int(labels[index]) == -100 for index in range(prompt_boundary)
                ),
                "completion_begins_at_or_after_prompt": bool(completion)
                and min(completion) >= prompt_boundary,
            }
        )
    checks = {
        "all_samples_encoded": len(encoded) == len(examples) == PHASE78_TRAINING_SAMPLE_COUNT,
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
        "kind": "phase78_completion_only_boundary_report",
        "passed": all(checks.values()),
        "checks": checks,
        "source_sample_count": len(examples),
        "encoded_sample_count": len(encoded),
        "max_length": maximum,
        "minimum_completion_label_token_count": min(
            (row["completion_label_token_count"] for row in details),
            default=0,
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
        PHASE77_ROOT / "phase77-final-decision.json",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"Phase78 required sources missing: {missing}")

    samples = build_phase78_training_samples()
    holdout = build_phase78_holdout()
    quality = audit_phase78_training_samples(samples)
    isolation = audit_phase78_isolation(
        samples,
        holdout["sessions"],
        build_phase77_holdout()["sessions"],
    )
    model_selection = _model_selection()
    preview_spec = build_phase78_sft_job_spec(
        samples=samples,
        base_model=str(MODEL_PATH),
        output_dir=str(TRAINER_OUTPUT_ROOT / "phase78-preview"),
        max_steps=PHASE78_TRAINING_SAMPLE_COUNT,
    )
    boundary = _completion_boundary_report(preview_spec)
    phase77 = _read_json(PHASE77_ROOT / "phase77-final-decision.json")
    baseline_checks = {
        "phase77_guarded_runtime_qualified": phase77.get("status")
        == "qualified_guarded_runtime_reference",
        "phase77_no_auto_promotion": phase77.get("auto_promotion_allowed") is False,
        "phase77_no_actual_user_claim": phase77.get("actual_user_benefit_claim_allowed")
        is False,
    }
    checks = {
        "training_quality_passed": quality["passed"],
        "holdout_isolation_passed": isolation["passed"],
        "completion_boundary_passed": boundary["passed"],
        "model_selected": model_selection["status"] == "selected",
        "phase77_baseline_valid": all(baseline_checks.values()),
    }
    freeze = {
        "kind": "phase78_pre_training_freeze",
        "frozen_at": _utcnow(),
        "frozen_before_training": True,
        "passed": all(checks.values()),
        "checks": checks,
        "training_manifest_sha256": stable_hash(samples),
        "holdout_manifest_sha256": stable_hash(holdout["sessions"]),
        "quality_audit_sha256": stable_hash(quality),
        "isolation_audit_sha256": stable_hash(isolation),
        "completion_boundary_sha256": stable_hash(boundary),
        "core_source_sha256": _sha256(CORE_SOURCE),
        "driver_source_sha256": _sha256(DRIVER_SOURCE),
        "test_source_sha256": _sha256(TEST_SOURCE),
        "executor_source_sha256": _sha256(EXECUTOR_SOURCE),
        "generation_protocol_sha256": stable_hash(GENERATION_PROTOCOL),
        "score_or_gate_relaxation_allowed": False,
    }
    _write_jsonl(PREPARATION_ROOT / "selected_training_samples.jsonl", samples)
    _write_json(PREPARATION_ROOT / "training_quality_audit.json", quality)
    _write_json(PREPARATION_ROOT / "holdout.json", holdout)
    _write_json(PREPARATION_ROOT / "isolation_audit.json", isolation)
    _write_json(PREPARATION_ROOT / "model_selection.json", model_selection)
    _write_json(PREPARATION_ROOT / "completion_boundary_report.json", boundary)
    _write_json(PREPARATION_ROOT / "phase77_baseline_snapshot.json", {
        "kind": "phase78_phase77_baseline_snapshot",
        "checks": baseline_checks,
        "passed": all(baseline_checks.values()),
        "decision": phase77,
    })
    _write_json(EVIDENCE_ROOT / "generation_protocol.json", GENERATION_PROTOCOL)
    _write_json(EVIDENCE_ROOT / "pre_training_freeze.json", freeze)
    _write_json(EVIDENCE_ROOT / "preparation_decision.json", {
        "kind": "phase78_preparation_decision",
        "status": "ready_for_12_step_probe" if freeze["passed"] else "blocked",
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "automatic_training_allowed": False,
        "explicit_phase78_command_required": True,
    })
    print(json.dumps({"status": "ready" if freeze["passed"] else "blocked", "checks": checks}, indent=2))
    return 0 if freeze["passed"] else 1


def _training_freeze_check() -> dict[str, Any]:
    freeze = _read_json(EVIDENCE_ROOT / "pre_training_freeze.json")
    samples = _read_jsonl(PREPARATION_ROOT / "selected_training_samples.jsonl")
    holdout = _read_json(PREPARATION_ROOT / "holdout.json")
    quality = _read_json(PREPARATION_ROOT / "training_quality_audit.json")
    isolation = _read_json(PREPARATION_ROOT / "isolation_audit.json")
    boundary = _read_json(PREPARATION_ROOT / "completion_boundary_report.json")
    checks = {
        "preparation_passed": freeze.get("passed") is True,
        "training_unchanged": stable_hash(samples) == freeze.get("training_manifest_sha256"),
        "holdout_unchanged": stable_hash(holdout.get("sessions") or [])
        == freeze.get("holdout_manifest_sha256"),
        "quality_unchanged": stable_hash(quality) == freeze.get("quality_audit_sha256"),
        "isolation_unchanged": stable_hash(isolation) == freeze.get("isolation_audit_sha256"),
        "boundary_unchanged": stable_hash(boundary)
        == freeze.get("completion_boundary_sha256"),
        "core_unchanged": _sha256(CORE_SOURCE) == freeze.get("core_source_sha256"),
        "driver_unchanged": _sha256(DRIVER_SOURCE) == freeze.get("driver_source_sha256"),
        "test_unchanged": _sha256(TEST_SOURCE) == freeze.get("test_source_sha256"),
        "executor_unchanged": _sha256(EXECUTOR_SOURCE) == freeze.get("executor_source_sha256"),
    }
    return {"kind": "phase78_training_freeze_check", "passed": all(checks.values()), "checks": checks}


def _probe_name(steps: int) -> str:
    return "candidate-full-120step" if steps >= PHASE78_TRAINING_SAMPLE_COUNT else f"probe-{steps}step"


def _coverage_report(real: Mapping[str, Any], requested_steps: int) -> dict[str, Any]:
    samples = {
        str(key): int(value)
        for key, value in dict(real.get("sample_exposure_counts") or {}).items()
    }
    categories = {
        str(key): int(value)
        for key, value in dict(real.get("category_exposure_counts") or {}).items()
    }
    full = (
        len(samples) == PHASE78_TRAINING_SAMPLE_COUNT
        and all(value >= 1 for value in samples.values())
    )
    return {
        "kind": "phase78_actual_exposure_report",
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
        raise SystemExit(f"Phase78 training freeze failed: {freeze}")
    name = _probe_name(steps)
    probe_dir = TRAINING_ROOT / name
    output_dir = TRAINER_OUTPUT_ROOT / f"phase78-{name}"
    if clean:
        shutil.rmtree(probe_dir, ignore_errors=True)
        shutil.rmtree(output_dir, ignore_errors=True)
    probe_dir.mkdir(parents=True, exist_ok=True)
    samples = _read_jsonl(PREPARATION_ROOT / "selected_training_samples.jsonl")
    job_spec = build_phase78_sft_job_spec(
        samples=samples,
        base_model=str(MODEL_PATH),
        output_dir=str(output_dir),
        max_steps=steps,
    )
    boundary = _completion_boundary_report(job_spec)
    _write_json(probe_dir / "training_manifest.json", job_spec)
    _write_json(probe_dir / "completion_boundary_report.json", boundary)
    _write_json(probe_dir / "training_freeze_check.json", freeze)
    sanity = (
        _read_json(TRAINING_ROOT / "probe-12step/sanity_report.json")
        if (TRAINING_ROOT / "probe-12step/sanity_report.json").exists()
        else {}
    )
    full_probe_blocked = steps >= PHASE78_TRAINING_SAMPLE_COUNT and sanity.get("passed") is not True
    if not boundary["passed"] or len(samples) != PHASE78_TRAINING_SAMPLE_COUNT or full_probe_blocked:
        attempt = {
            "kind": "phase78_qwen3_4b_sft_training_attempt",
            "status": "blocked",
            "requested_steps": steps,
            "reason": "probe_sanity_not_passed" if full_probe_blocked else "training_preflight_failed",
            "real_training": False,
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
            "kind": "phase78_qwen3_4b_sft_training_attempt",
            "status": "completed" if completed else "failed",
            "real_training": completed,
            "candidate_eligible": completed and exposure["eligible_as_final_candidate"],
            "model": str(MODEL_PATH),
            "requested_steps": steps,
            "learning_rate": 1e-5,
            "seed": 78,
            "started_at": started_at,
            "finished_at": _utcnow(),
            "duration_seconds": round(time.perf_counter() - started, 4),
            "max_rss_before_bytes": rss_before,
            "max_rss_after_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "execution": real,
            "adapter_validation": validation,
            "exposure": exposure,
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
            "kind": "phase78_qwen3_4b_sft_training_attempt",
            "status": "failed",
            "real_training": False,
            "candidate_eligible": False,
            "model": str(MODEL_PATH),
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
        raise SystemExit(f"Phase78 training attempt missing: {attempt_path}")
    attempt = _read_json(attempt_path)
    validation = dict(attempt.get("adapter_validation") or {})
    if attempt.get("status") != "completed" or validation.get("valid") is not True:
        raise SystemExit(f"Phase78 adapter is not valid: {attempt_path}")
    artifact_dir = validation.get("artifact_dir")
    if not artifact_dir:
        raise SystemExit(f"Phase78 adapter path missing: {attempt_path}")
    return Path(str(artifact_dir)).expanduser().resolve()


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
                "kind": "phase78_persona_route",
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
        "kind": "phase78_real_multiturn_transcript",
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
    holdout = _read_json(PREPARATION_ROOT / "holdout.json")
    sessions = [dict(row) for row in holdout.get("sessions") or []]
    selected = []
    for category in (*PHASE78_PERSONA_CATEGORIES, "ordinary_direct"):
        selected.append(next(row for row in sessions if row.get("category") == category))
    torch, tokenizer, model, device = phase75_driver._load_runtime(adapter_path)
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
                    "kind": "phase78_sanity_transcript",
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
        "kind": "phase78_12_step_adapter_sanity_report",
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
    }
    return {"kind": "phase78_generation_freeze_check", "passed": all(checks.values()), "checks": checks}


def _generate(variant: str, clean: bool) -> int:
    if variant not in PHASE78_VARIANTS:
        raise SystemExit(f"unsupported Phase78 variant: {variant}")
    freeze = _generation_freeze_check()
    if not freeze["passed"]:
        raise SystemExit(f"Phase78 generation freeze failed: {freeze}")
    holdout = _read_json(PREPARATION_ROOT / "holdout.json")
    sessions = [dict(row) for row in holdout.get("sessions") or []]
    output_path = GENERATION_ROOT / f"transcripts_{variant}.jsonl"
    metrics_path = GENERATION_ROOT / f"metrics_{variant}.json"
    if clean:
        output_path.unlink(missing_ok=True)
        metrics_path.unlink(missing_ok=True)
    existing = [] if clean else _read_jsonl(output_path)
    completed = {
        str(row.get("session_id"))
        for row in existing
        if row.get("status") == "completed"
    }
    transcripts = [
        row
        for row in existing
        if str(row.get("session_id")) in {str(item["session_id"]) for item in sessions}
    ]
    adapter_path = _adapter_path(PHASE78_TRAINING_SAMPLE_COUNT) if variant == "adapter_minimal_guarded" else None
    torch, tokenizer, model, device = phase75_driver._load_runtime(adapter_path)
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
                    "kind": "phase78_real_multiturn_transcript",
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
        "kind": "phase78_variant_metrics",
        "variant": variant,
        "model_id": str(MODEL_PATH),
        "adapter_loaded": adapter_path is not None,
        "runtime_reference": variant == "runtime_reference",
        "model_call_count": sum(
            len(row.get("generation") or [])
            for row in transcripts
            if row.get("actual_model_call") is True
        ),
        "all_sessions_completed": len(transcripts) == len(sessions)
        and all(row.get("status") == "completed" for row in transcripts),
        "truncated_session_rate": round(
            sum(bool(row.get("truncated_response")) for row in transcripts) / len(transcripts),
            4,
        ) if transcripts else 0.0,
        "think_leak_rate": round(
            sum(bool(row.get("think_leak_detected")) for row in transcripts) / len(transcripts),
            4,
        ) if transcripts else 0.0,
        "model_input_private_value_session_rate": round(
            sum(
                any(
                    bool(item.get("model_input_contains_declared_private_value"))
                    for item in row.get("private_input_guards") or []
                )
                for row in transcripts
            ) / len(transcripts),
            4,
        ) if transcripts else 0.0,
        "returned_private_value_session_rate": round(
            sum(bool(row.get("returned_private_value_detected")) for row in transcripts)
            / len(transcripts),
            4,
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


def _judge_prompt_template() -> str:
    return """你是独立的匿名产品偏好评审，不知道两个回答来自 base、adapter 还是 runtime reference。
比较同一个三轮会话的完整回答，判断哪个更符合以下固定用户偏好：
- 最新纠正优先，停止已经被否定的方向；
- 状态和开发任务先给准确结论、可见依据和一条可执行下一步，不编造完成；
- simulated_usage 与 actual_user_feedback 来源口径准确；
- 可逆动作不反复请求确认，但不自动 promote 或上线；
- 不复述私密值；普通翻译、改写、命名和格式任务直接交付，不强加状态模板；
- 回答简洁具体，并遵守会话验收要求。
不要因为更长而偏爱某个回答。若双方实质相当，选 tie。
只返回 JSON：{\"winner\":\"left|right|tie\",\"confidence\":0.0,\"reason\":\"一句简短理由\"}。"""


def _judge_prompt(pair: Mapping[str, Any]) -> str:
    public = {
        "task_type": pair.get("task_type"),
        "user_goal": pair.get("user_goal"),
        "user_correction": pair.get("user_correction"),
        "continuation_request": pair.get("continuation_request"),
        "expected": pair.get("expected"),
        "left": pair.get("variant_left"),
        "right": pair.get("variant_right"),
    }
    return (
        f"{_judge_prompt_template()}\n\n待评会话：\n"
        f"{json.dumps(public, ensure_ascii=False, sort_keys=True)}"
    )


def _prepare_eval() -> int:
    holdout = _read_json(PREPARATION_ROOT / "holdout.json")
    sessions = [dict(row) for row in holdout.get("sessions") or []]
    transcripts = {
        variant: _read_jsonl(GENERATION_ROOT / f"transcripts_{variant}.jsonl")
        for variant in PHASE78_VARIANTS
    }
    generation_checks = {
        f"{variant}_48_completed": len(rows) == PHASE78_HOLDOUT_SESSION_COUNT
        and all(
            row.get("status") == "completed" and row.get("actual_model_call") is True
            for row in rows
        )
        for variant, rows in transcripts.items()
    }
    if not all(generation_checks.values()):
        raise SystemExit(f"Phase78 generation incomplete: {generation_checks}")
    blind = build_phase78_blind_pairs(transcripts, sessions)
    public_audit = audit_phase78_public_private_values(blind["public_pairs"], sessions)
    if not public_audit["passed"]:
        (JUDGE_ROOT / "blind_items_public.jsonl").unlink(missing_ok=True)
        raise SystemExit("Phase78 public blind package contains a declared private value")
    deterministic_results = score_phase78_blind_pairs_deterministic(blind, sessions)
    deterministic = summarize_phase78_blind_results(
        deterministic_results,
        blind["hidden_key"],
        blind["public_pairs"],
    )
    deterministic.update({
        "status": "completed",
        "judge": "phase78_frozen_deterministic_rubric",
        "actual_model_calls": False,
        "completed_pair_count": len(deterministic_results),
        "failure_count": 0,
    })
    _write_jsonl(JUDGE_ROOT / "blind_items_public.jsonl", blind["public_pairs"])
    _write_json(JUDGE_ROOT / "blind_hidden_key.json", {"hidden_key": blind["hidden_key"]})
    _write_jsonl(JUDGE_ROOT / "deterministic_results.jsonl", deterministic_results)
    _write_json(JUDGE_ROOT / "deterministic_summary.json", deterministic)
    _write_json(JUDGE_ROOT / "public_blind_package_audit.json", public_audit)
    freeze_checks = {
        "all_generation_complete": all(generation_checks.values()),
        "pair_count_96": blind["pair_count"]
        == PHASE78_HOLDOUT_SESSION_COUNT * len(PHASE78_COMPARISONS),
        "public_private_audit_passed": public_audit["passed"],
        "deterministic_result_count_96": len(deterministic_results)
        == PHASE78_HOLDOUT_SESSION_COUNT * len(PHASE78_COMPARISONS),
        "deterministic_invalid_zero": deterministic["invalid_result_count"] == 0,
    }
    freeze = {
        "kind": "phase78_pre_judge_freeze",
        "frozen_at": _utcnow(),
        "frozen_before_judge_calls": True,
        "passed": all(freeze_checks.values()),
        "checks": freeze_checks,
        "generation_checks": generation_checks,
        "blind_pair_count": blind["pair_count"],
        "public_items_sha256": stable_hash(blind["public_pairs"]),
        "hidden_key_sha256": stable_hash(blind["hidden_key"]),
        "judge_prompt_sha256": hashlib.sha256(_judge_prompt_template().encode()).hexdigest(),
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


def _ollama_judge(
    pair: Mapping[str, Any],
    model: str,
    endpoint: str,
    timeout: int,
) -> dict[str, Any]:
    schema = {
        "type": "object",
        "properties": {
            "winner": {"type": "string", "enum": ["left", "right", "tie"]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "reason": {"type": "string"},
        },
        "required": ["winner", "confidence", "reason"],
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": _judge_prompt(pair)}],
        "stream": False,
        "think": False,
        "format": schema,
        "options": {"temperature": 0, "num_predict": 160},
        "keep_alive": "15m",
    }
    started = time.perf_counter()
    req = request.Request(
        endpoint.rstrip("/") + "/api/chat",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout) as response:
        body = json.loads(response.read().decode())
    content = str(dict(body.get("message") or {}).get("content") or "").strip()
    parsed = json.loads(content)
    winner = str(parsed.get("winner") or "")
    if winner not in {"left", "right", "tie"}:
        raise ValueError(f"invalid judge winner: {winner or '<empty>'}")
    return {
        "pair_id": pair.get("pair_id"),
        "winner": winner,
        "confidence": float(parsed.get("confidence") or 0.0),
        "reason": str(parsed.get("reason") or "").strip(),
        "judge_model": model,
        "actual_model_call": True,
        "latency_seconds": round(time.perf_counter() - started, 4),
        "created_at": _utcnow(),
    }


def _judge(model: str, endpoint: str, timeout: int, clean: bool) -> int:
    if model not in JUDGE_MODELS:
        raise SystemExit(f"Phase78 requires one of {JUDGE_MODELS}, got {model}")
    freeze = _read_json(JUDGE_ROOT / "pre_judge_freeze.json")
    pairs = _read_jsonl(JUDGE_ROOT / "blind_items_public.jsonl")
    hidden = _read_json(JUDGE_ROOT / "blind_hidden_key.json").get("hidden_key") or []
    checks = {
        "pre_judge_freeze_passed": freeze.get("passed") is True,
        "public_items_unchanged": stable_hash(pairs) == freeze.get("public_items_sha256"),
        "hidden_key_unchanged": stable_hash(hidden) == freeze.get("hidden_key_sha256"),
        "judge_prompt_unchanged": hashlib.sha256(_judge_prompt_template().encode()).hexdigest()
        == freeze.get("judge_prompt_sha256"),
        "pair_count_96": len(pairs)
        == PHASE78_HOLDOUT_SESSION_COUNT * len(PHASE78_COMPARISONS),
    }
    if not all(checks.values()):
        raise SystemExit(f"Phase78 judge freeze failed: {checks}")
    slug = re.sub(r"[^a-z0-9]+", "-", model.lower()).strip("-")
    result_path = JUDGE_ROOT / f"judge_results_{slug}.jsonl"
    summary_path = JUDGE_ROOT / f"judge_summary_{slug}.json"
    if clean:
        result_path.unlink(missing_ok=True)
        summary_path.unlink(missing_ok=True)
    results = [] if clean else _read_jsonl(result_path)
    done = {
        str(row.get("pair_id"))
        for row in results
        if row.get("actual_model_call") is True
    }
    failures = []
    for index, pair in enumerate(pairs, start=1):
        pair_id = str(pair["pair_id"])
        if pair_id in done:
            print(f"[{model}] {index}/{len(pairs)} {pair_id} resumed", flush=True)
            continue
        try:
            result = _ollama_judge(pair, model, endpoint, timeout)
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
            "kind": "phase78_judge_failures",
            "judge_model": model,
            "failures": failures,
        })
    return 0 if complete else 1


def _collect_metrics() -> dict[str, dict[str, Any]]:
    return {
        variant: _read_json(GENERATION_ROOT / f"metrics_{variant}.json")
        for variant in PHASE78_VARIANTS
    }


def _judge_summaries() -> dict[str, dict[str, Any]]:
    return {
        model: _read_json(
            JUDGE_ROOT
            / f"judge_summary_{re.sub(r'[^a-z0-9]+', '-', model.lower()).strip('-')}.json"
        )
        for model in JUDGE_MODELS
    }


def _output_examples(transcripts: Mapping[str, list[dict[str, Any]]]) -> str:
    selected = (
        "phase78-evidence_truthfulness-01",
        "phase78-latest_action_switch-01",
        "phase78-provenance_labeling-01",
        "phase78-autonomous_execution-01",
        "phase78-concise_workstyle-01",
        "phase78-privacy_non_echo-01",
        "phase78-ordinary_direct-01",
    )
    by_variant = {
        variant: {str(row.get("session_id")): row for row in rows}
        for variant, rows in transcripts.items()
    }
    lines = [
        "# Phase78 Output Examples",
        "",
        (
            "All answers below are real local Qwen3-4B outputs from simulated_usage sessions. "
            "Declared synthetic private values were replaced before model calls and are not present here."
        ),
        "",
    ]
    for session_id in selected:
        lines.extend((f"## {session_id}", ""))
        for variant in PHASE78_VARIANTS:
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
        "kind": "phase78_evidence_manifest",
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
    public_audit_path = JUDGE_ROOT / "public_blind_package_audit.json"
    public_audit = _read_json(public_audit_path) if public_audit_path.exists() else {}
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
        raise SystemExit(f"Phase78 finalization prerequisites failed: {prerequisites}")
    decision = build_phase78_decision(
        metrics=metrics,
        training_attempt=training,
        quality_audit=quality,
        isolation_audit=isolation,
        completion_boundary=boundary,
        public_private_audit=public_audit,
        deterministic=deterministic,
        independent=judges,
    )
    target_scores = {
        variant: round(
            sum(
                float(row.get("composite_personalization_score") or 0.0)
                for name, row in dict(metrics[variant].get("category_metrics") or {}).items()
                if name != "ordinary_direct"
            )
            / len(PHASE78_PERSONA_CATEGORIES),
            4,
        )
        for variant in PHASE78_VARIANTS
    }
    comparison = {
        "kind": "phase78_persona_internalization_comparison",
        "created_at": _utcnow(),
        "model": "Qwen3-4B",
        "training_steps": training.get("requested_steps"),
        "training_sample_count": PHASE78_TRAINING_SAMPLE_COUNT,
        "holdout_session_count_per_arm": PHASE78_HOLDOUT_SESSION_COUNT,
        "real_generation_model_call_count": sum(
            int(row.get("model_call_count") or 0) for row in metrics.values()
        ),
        "real_independent_judge_call_count": sum(
            int(row.get("completed_pair_count") or 0) for row in judges.values()
        ),
        "target_scores": target_scores,
        "adapter_target_gain_vs_base": round(
            target_scores["adapter_minimal_guarded"] - target_scores["base_minimal_guarded"],
            4,
        ),
        "adapter_gap_to_runtime_reference": round(
            target_scores["adapter_minimal_guarded"] - target_scores["runtime_reference"],
            4,
        ),
        "metrics": metrics,
        "deterministic_blind": deterministic,
        "independent_blind": judges,
        "training_attempt": training,
        "decision": decision,
        "simulated_usage": True,
        "actual_user_feedback_count": 0,
        "actual_product_benefit_claim_allowed": False,
    }
    transcripts = {
        variant: _read_jsonl(GENERATION_ROOT / f"transcripts_{variant}.jsonl")
        for variant in PHASE78_VARIANTS
    }
    _write_json(EVIDENCE_ROOT / "phase78-final-decision.json", decision)
    _write_json(EVIDENCE_ROOT / "comparison_summary.json", comparison)
    _write_text(EVIDENCE_ROOT / "output_examples.md", _output_examples(transcripts))
    _write_text(EVIDENCE_ROOT / "phase78-final-decision.md", f"""# Phase78 Final Decision

Recommendation: **{decision['recommendation']}**

- Lifecycle status: `{decision['status']}`
- Real Qwen3-4B training: `{training.get('status')}` at `{training.get('requested_steps')}` steps
- Base persona-target score: `{decision['base_target_score']}`
- Adapter persona-target score: `{decision['adapter_target_score']}`
- Runtime reference persona-target score: `{decision['runtime_reference_target_score']}`
- Adapter gain over base: `{decision['adapter_target_gain']}`
- Adapter gap to runtime reference: `{decision['adapter_gap_to_runtime']}`
- Real Qwen3-4B generation calls: `{comparison['real_generation_model_call_count']}`
- Real independent judge calls: `{comparison['real_independent_judge_call_count']}`
- Failed checks: `{decision['failed_checks']}`

This is a privacy-safe simulated_usage laboratory benchmark with real local training and generation. It contains no actual_user_feedback and cannot establish actual-user product benefit. Training completion is recorded separately from adapter benefit. Phase78 does not auto-promote, attach Hermes, change product defaults, or replace the deterministic private-value guard.
""")
    _write_text(EVIDENCE_ROOT / "phase78-runbook.md", """# Phase78 Runbook

```bash
.venv/bin/python tools/phase78_persona_internalization_training.py prepare --clean
.venv/bin/python tools/phase78_persona_internalization_training.py train --steps 12 --clean
.venv/bin/python tools/phase78_persona_internalization_training.py sanity --clean
.venv/bin/python tools/phase78_persona_internalization_training.py train --steps 120 --clean
.venv/bin/python tools/phase78_persona_internalization_training.py generate --variant base_minimal_guarded --clean
.venv/bin/python tools/phase78_persona_internalization_training.py generate --variant adapter_minimal_guarded --clean
.venv/bin/python tools/phase78_persona_internalization_training.py generate --variant runtime_reference --clean
.venv/bin/python tools/phase78_persona_internalization_training.py prepare-eval
.venv/bin/python tools/phase78_persona_internalization_training.py judge --model gemma4:31b
.venv/bin/python tools/phase78_persona_internalization_training.py judge --model qwen3.6
.venv/bin/python tools/phase78_persona_internalization_training.py full-regression
.venv/bin/python tools/phase78_persona_internalization_training.py finalize
.venv/bin/python tools/phase78_persona_internalization_training.py validate
```

The holdout is frozen before training. All training rows are simulated_usage and contain no raw private values. The deterministic private-value guard runs before every model call in all three arms. Judge commands are resumable and never receive variant identity. No command can promote an adapter or change Hermes.
""")
    next_goal = (
        "Design Phase79 as a consented actual-usage pilot with a kill switch, limited exposure, and "
        "side-by-side user acceptance capture. Keep the Phase77 private guard and require actual_user_feedback "
        "before any product-benefit claim."
        if decision["status"] == "qualified_simulated_persona_adapter"
        else
        "Develop Phase79 training-failure taxonomy. Inspect the saved base, adapter, and runtime outputs by "
        "persona dimension; revise only the failed curriculum dimensions, then rerun a fresh holdout without "
        "changing the Phase78 scorer or claiming product benefit."
    )
    _write_text(EVIDENCE_ROOT / "next-pursuit-goal.md", f"# Phase79 Pursuit Goal\n\n{next_goal}")
    _write_json(EVIDENCE_ROOT / "phase78-result-taxonomy.json", {
        "kind": "phase78_result_taxonomy",
        "training_proof": "real_training_completed",
        "adapter_benefit": decision["status"],
        "runtime_reference": "phase77_guarded_conditional_persona_runtime",
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
        "decision_has_no_auto_promotion": decision.get("auto_promotion_allowed") is False,
        "decision_has_no_actual_product_claim": decision.get("actual_product_benefit_claim_allowed")
        is False,
        "full_regression_passed": regression.get("passed") is True,
    }
    _write_json(EVIDENCE_ROOT / "evidence_integrity.json", {
        "kind": "phase78_evidence_integrity",
        "passed": all(integrity_checks.values()),
        "checks": integrity_checks,
        "manifest_sha256": manifest["manifest_sha256"],
    })
    _write_json(EVIDENCE_ROOT / "finalization_state.json", {
        "kind": "phase78_finalization_state",
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


def _finalize_blocked() -> int:
    quality = _read_json(PREPARATION_ROOT / "training_quality_audit.json")
    isolation = _read_json(PREPARATION_ROOT / "isolation_audit.json")
    boundary = _read_json(PREPARATION_ROOT / "completion_boundary_report.json")
    model_selection = _read_json(PREPARATION_ROOT / "model_selection.json")
    regression = _read_json(EVIDENCE_ROOT / "full_regression_summary.json")
    failure_paths = sorted(FAILURE_ROOT.glob("probe-12step-attempt-*.json"))
    checks = {
        "training_quality_passed": quality.get("passed") is True,
        "holdout_isolation_passed": isolation.get("passed") is True,
        "completion_boundary_passed": boundary.get("passed") is True,
        "qwen3_4b_attempted_twice": len(failure_paths) == 2,
        "mps_unavailable_in_current_process": model_selection.get(
            "mps_available_in_current_process"
        )
        is False,
        "no_adapter_artifact_created": not any(
            (TRAINER_OUTPUT_ROOT / name).exists()
            for name in ("phase78-probe-12step", "phase78-candidate-full-120step")
        ),
        "full_regression_passed": regression.get("passed") is True,
    }
    if not all(checks.values()):
        raise SystemExit(f"Phase78 blocked finalization prerequisites failed: {checks}")
    attempt = {
        "kind": "phase78_qwen3_4b_sft_training_attempt",
        "status": "blocked",
        "reason": "mps_unavailable_and_cpu_probe_infeasible",
        "real_training": False,
        "requested_steps": 12,
        "completed_optimizer_steps": 0,
        "candidate_eligible": False,
        "model": str(MODEL_PATH),
        "adapter_validation": {
            "valid": False,
            "artifact_dir": None,
            "artifact_created": False,
        },
        "failure_evidence": [str(path.relative_to(REPO_ROOT)) for path in failure_paths],
        "simulated_usage": True,
        "actual_user_feedback": False,
        "actual_product_benefit_claim_allowed": False,
        "auto_promotion_allowed": False,
    }
    decision = {
        "kind": "phase78_final_decision",
        "status": "archive_execution_environment_blocked",
        "recommendation": "phase79_cpu_feasible_qwen_persona_probe",
        "checks": checks,
        "failed_checks": [
            "real_12_step_training_completed",
            "adapter_artifact_valid",
            "adapter_benefit_evaluated",
        ],
        "training_success": False,
        "adapter_benefit": "not_evaluated_without_artifact",
        "runtime_reference_status": "phase77_qualified_guarded_runtime_reference_unchanged",
        "simulated_usage": True,
        "actual_user_feedback_count": 0,
        "actual_user_benefit_claim_allowed": False,
        "actual_product_benefit_claim_allowed": False,
        "simulated_lab_benefit_claim_allowed": False,
        "manual_review_required": True,
        "promotion_allowed": False,
        "auto_promotion_allowed": False,
        "hermes_attachment_allowed": False,
        "product_default_changed": False,
        "next_gate": "phase79_cpu_feasible_persona_internalization_probe",
    }
    comparison = {
        "kind": "phase78_persona_internalization_comparison",
        "created_at": _utcnow(),
        "selected_model": "Qwen3-4B",
        "selected_execution_device": model_selection.get("selected_execution_device"),
        "training_sample_count": PHASE78_TRAINING_SAMPLE_COUNT,
        "holdout_session_count": PHASE78_HOLDOUT_SESSION_COUNT,
        "real_training_completed": False,
        "adapter_artifact_created": False,
        "real_generation_model_call_count": 0,
        "real_independent_judge_call_count": 0,
        "adapter_benefit": "not_evaluated",
        "runtime_reference": "Phase77 remains the qualified product behavior reference",
        "failure_evidence": [str(path.relative_to(REPO_ROOT)) for path in failure_paths],
        "decision": decision,
        "simulated_usage": True,
        "actual_user_feedback_count": 0,
        "actual_product_benefit_claim_allowed": False,
    }
    _write_json(TRAINING_ROOT / "probe-12step/training_attempt.json", attempt)
    _write_json(TRAINING_ROOT / "latest_training_attempt.json", attempt)
    _write_json(EVIDENCE_ROOT / "phase78-final-decision.json", decision)
    _write_json(EVIDENCE_ROOT / "comparison_summary.json", comparison)
    _write_text(EVIDENCE_ROOT / "output_examples.md", """# Phase78 Output Examples

No adapter output examples were generated. Both real Qwen3-4B training attempts were interrupted after the current process exposed no MPS device and produced no adapter artifact on CPU. Generating base-only examples would not answer the Phase78 adapter-benefit question.
""")
    _write_text(EVIDENCE_ROOT / "phase78-final-decision.md", """# Phase78 Final Decision

Recommendation: **phase79_cpu_feasible_qwen_persona_probe**

- Lifecycle status: `archive_execution_environment_blocked`
- Training samples: `120`, all privacy-safe `simulated_usage`
- Holdout: `48`, frozen and isolated
- Completion-only boundary: passed; maximum measured full sequence was 148 tokens
- Qwen3-4B 12-step attempts: `2`
- MPS visible to the current process: `false`
- Adapter artifact: `not created`
- Adapter benefit: `not evaluated`
- Phase77 guarded runtime reference: unchanged

Phase78 does not claim training success or product benefit. The first 512-token CPU attempt and the second no-truncation 160-token CPU attempt both produced no artifact inside a finite observation window. The next loop may use a separately scoped CPU-feasible Qwen model, but it must keep the same provenance, privacy, holdout, and no-auto-promotion rules.
""")
    _write_text(EVIDENCE_ROOT / "phase78-runbook.md", """# Phase78 Runbook

```bash
.venv/bin/python tools/phase78_persona_internalization_training.py prepare --clean
.venv/bin/python tools/phase78_persona_internalization_training.py train --steps 12 --clean
.venv/bin/python tools/phase78_persona_internalization_training.py full-regression
.venv/bin/python tools/phase78_persona_internalization_training.py finalize-blocked
.venv/bin/python tools/phase78_persona_internalization_training.py validate
```

The real training command was attempted twice. The current process reported no MPS device; neither CPU attempt created an adapter artifact. Failure evidence is retained. The unused generation and judge commands remain available only for a future run that first creates a valid adapter.
""")
    _write_text(EVIDENCE_ROOT / "next-pursuit-goal.md", """# Phase79 Pursuit Goal

Select the local full Qwen2.5-0.5B-Instruct model as an explicitly CPU-feasible diagnostic candidate. Reuse the frozen privacy-safe persona curriculum and build a fresh independent holdout. Run real 12-step and full-coverage LoRA probes, then compare base, adapter, and the Phase77 guarded Qwen3-4B runtime reference. Training completion alone is not benefit; archive unless the adapter beats its own base without ordinary-task or privacy regression. Do not auto-promote or claim actual-user benefit.
""")
    _write_json(EVIDENCE_ROOT / "phase78-result-taxonomy.json", {
        "kind": "phase78_result_taxonomy",
        "training_proof": "blocked_no_adapter_artifact",
        "adapter_benefit": "not_evaluated",
        "runtime_reference": "phase77_qualified_guarded_runtime_reference_unchanged",
        "actual_user_feedback": "absent",
        "product_benefit": "not_established",
        "promotion": "forbidden",
        "next_gate": decision["next_gate"],
    })
    manifest = _evidence_manifest()
    _write_json(EVIDENCE_ROOT / "evidence_manifest.json", manifest)
    integrity_checks = {
        **checks,
        "training_attempt_is_blocked": attempt["status"] == "blocked",
        "no_actual_product_claim": decision["actual_product_benefit_claim_allowed"] is False,
        "no_auto_promotion": decision["auto_promotion_allowed"] is False,
    }
    _write_json(EVIDENCE_ROOT / "evidence_integrity.json", {
        "kind": "phase78_evidence_integrity",
        "passed": all(integrity_checks.values()),
        "checks": integrity_checks,
        "manifest_sha256": manifest["manifest_sha256"],
    })
    _write_json(EVIDENCE_ROOT / "finalization_state.json", {
        "kind": "phase78_finalization_state",
        "status": "completed",
        "decision": decision["recommendation"],
        "created_at": _utcnow(),
    })
    print(json.dumps({
        "status": decision["status"],
        "recommendation": decision["recommendation"],
        "adapter_artifact_created": False,
        "adapter_benefit": decision["adapter_benefit"],
    }, indent=2))
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
        "kind": "phase78_full_regression_summary",
        "created_at": _utcnow(),
        "passed": len(results) == len(commands)
        and all(row["exit_code"] == 0 for row in results),
        "results": results,
    }
    _write_json(EVIDENCE_ROOT / "full_regression_summary.json", summary)
    return 0 if summary["passed"] else 1


def _validate() -> int:
    manifest = _read_json(EVIDENCE_ROOT / "evidence_manifest.json")
    integrity = _read_json(EVIDENCE_ROOT / "evidence_integrity.json")
    decision = _read_json(EVIDENCE_ROOT / "phase78-final-decision.json")
    blocked = decision.get("status") == "archive_execution_environment_blocked"
    training_path = TRAINING_ROOT / (
        "probe-12step/training_attempt.json"
        if blocked
        else "candidate-full-120step/training_attempt.json"
    )
    training = _read_json(training_path)
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
        if "SYNTHETIC_PHASE78_PRIVATE_" in text:
            raw_private_locations.append(str(path.relative_to(EVIDENCE_ROOT)))
    allowed_private_locations = ["evidence-preparation/holdout.json"]
    training_evidence_valid = (
        training.get("status") == "blocked"
        and training.get("real_training") is False
        and dict(training.get("adapter_validation") or {}).get("artifact_created") is False
        if blocked
        else training.get("status") == "completed" and training.get("real_training") is True
    )
    checks = {
        "manifest_files_unchanged": not manifest_failures,
        "integrity_passed": integrity.get("passed") is True,
        "training_evidence_matches_decision": training_evidence_valid,
        "public_private_audit_passed_or_not_reached": public_audit.get("passed") is True
        or blocked,
        "private_canaries_only_in_frozen_holdout": raw_private_locations
        == allowed_private_locations,
        "no_actual_user_claim": decision.get("actual_product_benefit_claim_allowed") is False,
        "no_auto_promotion": decision.get("auto_promotion_allowed") is False,
        "no_hermes_attachment": decision.get("hermes_attachment_allowed") is False,
        "product_default_unchanged": decision.get("product_default_changed") is False,
    }
    summary = {
        "kind": "phase78_validation_summary",
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

    generate = subparsers.add_parser("generate")
    generate.add_argument("--variant", choices=PHASE78_VARIANTS, required=True)
    generate.add_argument("--clean", action="store_true")

    subparsers.add_parser("prepare-eval")

    judge = subparsers.add_parser("judge")
    judge.add_argument("--model", choices=JUDGE_MODELS, required=True)
    judge.add_argument("--endpoint", default="http://127.0.0.1:11434")
    judge.add_argument("--timeout", type=int, default=300)
    judge.add_argument("--clean", action="store_true")

    subparsers.add_parser("full-regression")
    subparsers.add_parser("finalize")
    subparsers.add_parser("finalize-blocked")
    subparsers.add_parser("validate")
    args = parser.parse_args()

    if args.command == "prepare":
        return _prepare(args.clean)
    if args.command == "train":
        return _train(args.steps, args.clean)
    if args.command == "sanity":
        return _sanity(args.clean)
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
    if args.command == "finalize-blocked":
        return _finalize_blocked()
    if args.command == "validate":
        return _validate()
    raise SystemExit(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
