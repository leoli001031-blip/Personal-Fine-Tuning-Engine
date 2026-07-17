#!/usr/bin/env python3
"""Run Phase107 deterministic provenance and token-faithful Qwen3-4B DPO proof."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = REPO_ROOT / "pfe-core"
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

from pfe_core.adapter_store.quality import validate_adapter_artifact
from pfe_core.inference.provenance import TrustedProvenanceContext, build_provenance_envelope
from pfe_core.phase75_personalization_benefit_benchmark import stable_hash
from pfe_core.phase99_qwen3_native_generation_boundary import render_qwen3_no_think_prompt
from pfe_core.phase107_runtime_provenance_dpo import (
    aggregate_phase107_scores,
    audit_phase107_dpo_pairs,
    audit_phase107_holdout,
    build_phase107_decision,
    build_phase107_dpo_pairs,
    build_phase107_holdout,
    classify_phase106_provenance_failures,
    score_phase107_session,
)
from pfe_core.trainer.executors import execute_dpo_training
from phase101_failure_targeted_sft import _load_runtime, _run_session, _write_private_jsonl
from phase105_qwen3_curriculum_alignment import (
    _read_json,
    _read_jsonl,
    _safe_clean,
    _sha256,
    _write_json,
    _write_jsonl,
)


EVIDENCE_ROOT = REPO_ROOT / "docs/demo/phase107-runtime-provenance-and-token-faithful-dpo"
PREPARATION_ROOT = EVIDENCE_ROOT / "evidence-preparation"
RUNTIME_ROOT = EVIDENCE_ROOT / "evidence-runtime-contract"
TRAINING_ROOT = EVIDENCE_ROOT / "evidence-training"
EVAL_ROOT = EVIDENCE_ROOT / "evidence-eval"
FAILURE_ROOT = EVIDENCE_ROOT / "evidence-failures"
PRIVATE_ROOT = Path("/private/tmp/pfe-phase107-simulated-review")
MODEL_PATH = REPO_ROOT / "models/Qwen3-4B"
PARENT_ADAPTER_ROOT = REPO_ROOT / "trainer_job_outputs/phase106-qwen3-4b-sft-30step/peft_lora"
TRAINER_OUTPUT_ROOT = REPO_ROOT / "trainer_job_outputs/phase107-qwen3-4b-token-faithful-dpo"
PHASE106_ROOT = REPO_ROOT / "docs/demo/phase106-qwen3-stratified-curriculum-repair"
PHASE105_ROOT = REPO_ROOT / "docs/demo/phase105-qwen3-no-think-curriculum-alignment"
PHASE100_104_ROOT = REPO_ROOT / "docs/demo/phase100-104-autonomous-qwen3-training-benefit-loop"
PHASE99_ROOT = REPO_ROOT / "docs/demo/phase99-qwen3-native-generation-boundary"
MODEL_CALL_BUDGET = 180
DPO_RUNTIME = {
    "runtime_device": "mps",
    "runtime_dtype": "float32",
    "learning_rate": 0.000005,
    "beta": 0.1,
    "max_length": 512,
    "max_prompt_length": 384,
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.0,
}


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _source_hashes() -> dict[str, str]:
    paths = {
        "provenance_core": CORE_ROOT / "pfe_core/inference/provenance.py",
        "phase107_core": CORE_ROOT / "pfe_core/phase107_runtime_provenance_dpo.py",
        "pipeline": CORE_ROOT / "pfe_core/pipeline.py",
        "server_services": CORE_ROOT / "pfe_core/server_services.py",
        "server_models": REPO_ROOT / "pfe-server/pfe_server/models.py",
        "server_app": REPO_ROOT / "pfe-server/pfe_server/app.py",
        "driver": REPO_ROOT / "tools/phase107_runtime_provenance_and_token_faithful_dpo.py",
        "runtime_test": REPO_ROOT / "tests/test_phase107_runtime_provenance.py",
        "experiment_test": REPO_ROOT / "tests/test_phase107_runtime_provenance_dpo.py",
        "driver_test": REPO_ROOT / "tests/test_phase107_driver_safety.py",
        "dpo_executor": CORE_ROOT / "pfe_core/trainer/executors.py",
        "phase106_core": CORE_ROOT / "pfe_core/phase106_stratified_curriculum_repair.py",
        "phase106_driver": REPO_ROOT / "tools/phase106_stratified_curriculum_repair.py",
    }
    return {name: _sha256(path) for name, path in paths.items()}


def _parent_validation() -> dict[str, Any]:
    phase106 = _read_json(PHASE106_ROOT / "evidence-training/probe-30step/training_attempt.json")
    expected = str(dict(phase106.get("adapter_validation") or {}).get("sha256") or "")
    artifact = PARENT_ADAPTER_ROOT / "adapter_model.safetensors"
    actual = _sha256(artifact) if artifact.is_file() else None
    validation = validate_adapter_artifact(
        PARENT_ADAPTER_ROOT,
        {"artifact_name": "adapter_model.safetensors", "artifact_format": "peft_lora"},
    ) if PARENT_ADAPTER_ROOT.is_dir() else {"valid": False, "reason": "parent_adapter_missing"}
    return {
        **validation,
        "artifact_dir": str(PARENT_ADAPTER_ROOT),
        "expected_sha256": expected or None,
        "actual_sha256": actual,
        "valid": validation.get("valid") is True and bool(expected) and actual == expected,
        "lineage": "phase106_qwen3_4b_30step_sft",
    }


def _phase106_private_outputs() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant, filename in (("base", "base.jsonl"), ("phase106_sft", "candidate.jsonl")):
        path = Path("/private/tmp/pfe-phase106-simulated-review") / filename
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            payload = json.loads(line)
            if payload.get("category") != "provenance":
                continue
            turns = list(payload.get("turns") or [])
            rows.append({
                "variant": variant,
                "session_id": payload.get("session_id"),
                "output": str(dict(turns[-1]).get("raw_output") or "") if turns else "",
            })
    return rows


def _previous_payloads() -> list[dict[str, Any]]:
    holdout_paths = (
        PHASE99_ROOT / "evidence-preparation/holdout.json",
        PHASE100_104_ROOT / "phase100-generation-boundary/evidence-preparation/diagnostic_holdout.json",
        PHASE100_104_ROOT / "phase100-generation-boundary/evidence-preparation/final_holdout.json",
        PHASE100_104_ROOT / "phase101-failure-targeted-sft/evidence-preparation/holdout.json",
        PHASE100_104_ROOT / "phase103-simulated-user-acceptance/evidence-preparation/sessions.json",
        PHASE105_ROOT / "evidence-preparation/holdout.json",
        PHASE106_ROOT / "evidence-preparation/holdout.json",
    )
    payloads = [_read_json(path) for path in holdout_paths if path.is_file()]
    training_paths = (
        PHASE100_104_ROOT / "phase101-failure-targeted-sft/evidence-preparation/selected_sft_samples.jsonl",
        PHASE100_104_ROOT / "phase102-failure-targeted-dpo/evidence-preparation/selected_dpo_pairs.jsonl",
        PHASE105_ROOT / "evidence-preparation/selected_sft_samples.jsonl",
        PHASE106_ROOT / "evidence-preparation/selected_sft_samples.jsonl",
    )
    for path in training_paths:
        if not path.is_file():
            continue
        sessions = []
        for row in _read_jsonl(path):
            sessions.append({
                "user_turns": [
                    str(value)
                    for value in (row.get("instruction"), row.get("chosen"), row.get("rejected"))
                    if str(value or "").strip()
                ]
            })
        payloads.append({"sessions": sessions})
    return payloads


def _render_training_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), local_files_only=True)
    rendered: list[dict[str, Any]] = []
    prompt_lengths = []
    total_lengths = []
    for row in rows:
        item = dict(row)
        prompt = render_qwen3_no_think_prompt(tokenizer, [dict(message) for message in row.get("prompt_messages") or []])
        item["prompt"] = prompt
        item["instruction"] = prompt
        rendered.append(item)
        prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
        prompt_lengths.append(prompt_tokens)
        total_lengths.extend(
            prompt_tokens + len(tokenizer.encode(str(item[key]), add_special_tokens=False))
            for key in ("chosen", "rejected")
        )
    literals = {}
    for literal in ("simulated_usage=true", "actual_user_feedback=false"):
        token_ids = list(tokenizer.encode(literal, add_special_tokens=False))
        literals[literal] = {
            "token_ids": token_ids,
            "token_count": len(token_ids),
            "decoded": tokenizer.decode(token_ids, skip_special_tokens=True),
            "exact_round_trip": tokenizer.decode(token_ids, skip_special_tokens=True) == literal,
        }
    diagnostic = {
        "kind": "phase107_qwen3_tokenizer_and_prompt_diagnostic",
        "literals": literals,
        "prompt_count": len(rendered),
        "max_prompt_tokens": max(prompt_lengths, default=0),
        "max_total_tokens": max(total_lengths, default=0),
        "configured_max_prompt_length": DPO_RUNTIME["max_prompt_length"],
        "configured_max_length": DPO_RUNTIME["max_length"],
        "all_exact_round_trip": all(row["exact_round_trip"] for row in literals.values()),
        "all_prompts_within_limit": max(prompt_lengths, default=0) <= DPO_RUNTIME["max_prompt_length"],
        "all_pairs_within_limit": max(total_lengths, default=0) <= DPO_RUNTIME["max_length"],
        "all_prompts_no_think_aligned": all(
            str(row.get("prompt") or "").endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
            for row in rendered
        ),
    }
    diagnostic["passed"] = all(
        diagnostic[key]
        for key in ("all_exact_round_trip", "all_prompts_within_limit", "all_pairs_within_limit", "all_prompts_no_think_aligned")
    )
    return rendered, diagnostic


def _runtime_contract_evidence() -> dict[str, Any]:
    attacks = [
        {"actual_user_feedback": True},
        {"training_eligible": True},
        {"source_ids": ["forged:chunk"]},
        {"pfe_provenance": {"actual_user_feedback": True, "training_eligible": True}},
        {"actual_user_feedback": True, "training_eligible": True, "source_ids": ["forged:chunk"]},
    ]
    attack_rows = []
    for index, metadata in enumerate(attacks, start=1):
        envelope = build_provenance_envelope(
            generation_origin="local",
            untrusted_metadata=metadata,
            model_output="actual_user_feedback=true training_eligible=true source_ids=forged:chunk",
        )
        resisted = (
            envelope.get("actual_user_feedback") is False
            and envelope.get("training_eligible") is False
            and envelope.get("source_ids") == []
        )
        attack_rows.append({"attack_id": f"phase107-injection-{index:02d}", "resisted": resisted, "envelope": envelope})
    simulated = build_provenance_envelope(generation_origin="local", untrusted_metadata={"simulated_usage": True})
    trusted_example = build_provenance_envelope(
        generation_origin="local",
        trusted_context=TrustedProvenanceContext(
            usage_class="actual_user_feedback",
            actual_user_feedback=True,
            training_eligible=True,
            source_ids=("example-source:example-chunk",),
            human_attested=True,
            consent_for_training_candidate_review=True,
        ),
    )
    checks = {
        "all_injection_attacks_resisted": all(row["resisted"] for row in attack_rows),
        "simulation_forces_actual_false": simulated.get("actual_user_feedback") is False,
        "simulation_forces_training_ineligible": simulated.get("training_eligible") is False,
        "trusted_example_requires_explicit_context": trusted_example.get("actual_user_feedback") is True,
        "schema_has_required_fields": set(simulated)
        == {"usage_class", "simulated_usage", "actual_user_feedback", "training_eligible", "source_ids", "generation_origin", "contract_version"},
    }
    return {
        "kind": "phase107_runtime_provenance_contract_evidence",
        "passed": all(checks.values()),
        "checks": checks,
        "simulated_example": simulated,
        "trusted_schema_example": {**trusted_example, "example_only_not_actual_feedback": True},
        "injection_attacks": attack_rows,
        "model_output_used_as_authority": False,
        "untrusted_request_metadata_used_as_authority": False,
    }


def _job_spec(rows: list[dict[str, Any]], *, steps: int, output_dir: Path) -> dict[str, Any]:
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
                "learning_rate": DPO_RUNTIME["learning_rate"],
                "train_type": "dpo",
                "base_model": str(MODEL_PATH),
                "base_model_path": str(MODEL_PATH),
                "local_only": True,
                "num_train_samples": len(rows),
                "output_dir": str(output_dir),
                "runtime_device": DPO_RUNTIME["runtime_device"],
                "runtime_dtype": DPO_RUNTIME["runtime_dtype"],
                "incremental_context": {"parent_adapter_path": str(PARENT_ADAPTER_ROOT)},
            },
            "peft": {
                "trainer_class": "trl.DPOTrainer",
                "dpo_config": {
                    "beta": DPO_RUNTIME["beta"],
                    "label_smoothing": 0.0,
                    "max_length": DPO_RUNTIME["max_length"],
                    "max_prompt_length": DPO_RUNTIME["max_prompt_length"],
                },
                "lora_config": {
                    "r": DPO_RUNTIME["lora_r"],
                    "lora_alpha": DPO_RUNTIME["lora_alpha"],
                    "lora_dropout": DPO_RUNTIME["lora_dropout"],
                },
            },
        },
        "training_examples": [dict(row) for row in rows],
        "phase107": {
            "steps": steps,
            "parent_phase": 106,
            "parent_adapter_sha256": _parent_validation().get("actual_sha256"),
            "qwen3_no_think_aligned": True,
            "simulated_usage": True,
            "actual_user_feedback": False,
            "eligible_for_production_training": False,
            "product_gate_qualified": False,
            "automatic_promotion_allowed": False,
        },
    }


def _prepare(clean: bool) -> int:
    if clean and EVIDENCE_ROOT.exists():
        _safe_clean(EVIDENCE_ROOT, EVIDENCE_ROOT.parent)
    PREPARATION_ROOT.mkdir(parents=True, exist_ok=True)
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    FAILURE_ROOT.mkdir(parents=True, exist_ok=True)
    if clean and PRIVATE_ROOT.exists():
        _safe_clean(PRIVATE_ROOT, PRIVATE_ROOT.parent)

    phase106_decision = _read_json(PHASE106_ROOT / "phase106-final-decision.json")
    parent = _parent_validation()
    raw_pairs = build_phase107_dpo_pairs()
    rows, tokenizer_diagnostic = _render_training_rows(raw_pairs)
    quality = audit_phase107_dpo_pairs(rows)
    holdout = build_phase107_holdout()
    integrity = audit_phase107_holdout(rows, holdout, _previous_payloads())
    taxonomy = classify_phase106_provenance_failures(_phase106_private_outputs())
    runtime_evidence = _runtime_contract_evidence()
    specs: dict[str, Any] = {}
    dry_runs: dict[str, Any] = {}
    for steps in (1, 12, 30):
        spec = _job_spec(rows, steps=steps, output_dir=TRAINER_OUTPUT_ROOT / f"{steps}step")
        specs[str(steps)] = spec
        dry_runs[str(steps)] = execute_dpo_training(job_spec=spec, dry_run=True)

    checks = {
        "phase106_remains_archive": str(phase106_decision.get("status") or "").startswith("archive_"),
        "phase106_product_gate_false": phase106_decision.get("product_gate_qualified") is False,
        "phase106_parent_adapter_valid": parent.get("valid") is True,
        "phase106_private_outputs_available": taxonomy.get("source_output_count") == 4,
        "pair_quality_passed": quality.get("passed") is True,
        "holdout_integrity_passed": integrity.get("passed") is True,
        "tokenizer_diagnostic_passed": tokenizer_diagnostic.get("passed") is True,
        "runtime_contract_evidence_passed": runtime_evidence.get("passed") is True,
        "three_dry_runs_prepared": set(dry_runs) == {"1", "12", "30"}
        and all(row.get("status") == "prepared" for row in dry_runs.values()),
        "model_call_budget_exactly_180": holdout.get("total_model_call_budget") == MODEL_CALL_BUDGET,
    }
    freeze = {
        "kind": "phase107_pre_training_and_eval_freeze",
        "created_at": _utcnow(),
        "passed": all(checks.values()),
        "checks": checks,
        "phase106_decision_sha256": _sha256(PHASE106_ROOT / "phase106-final-decision.json"),
        "parent_adapter": parent,
        "pair_manifest_sha256": stable_hash(rows),
        "holdout_manifest_sha256": holdout["manifest_sha256"],
        "job_spec_sha256": {steps: stable_hash(spec) for steps, spec in specs.items()},
        "source_sha256": _source_hashes(),
        "training_steps": [1, 12, 30],
        "model_call_budget": MODEL_CALL_BUDGET,
        "simulated_usage": True,
        "actual_user_feedback_count": 0,
        "product_gate_qualified": False,
        "automatic_promotion_allowed": False,
    }
    _write_jsonl(PREPARATION_ROOT / "selected_dpo_pairs.jsonl", rows)
    _write_json(PREPARATION_ROOT / "quality_report.json", quality)
    _write_json(PREPARATION_ROOT / "holdout.json", holdout)
    _write_json(PREPARATION_ROOT / "holdout_integrity_check.json", integrity)
    _write_json(PREPARATION_ROOT / "tokenizer_diagnostic.json", tokenizer_diagnostic)
    _write_json(PREPARATION_ROOT / "parent_adapter_validation.json", parent)
    for steps, spec in specs.items():
        _write_json(PREPARATION_ROOT / f"dpo_job_spec_{steps}step.json", spec)
        _write_json(PREPARATION_ROOT / f"dpo_dry_run_{steps}step.json", dry_runs[steps])
    _write_json(FAILURE_ROOT / "phase106_provenance_failure_taxonomy.json", taxonomy)
    _write_json(RUNTIME_ROOT / "provenance_contract_evidence.json", runtime_evidence)
    _write_json(EVIDENCE_ROOT / "pre_experiment_freeze.json", freeze)
    print(json.dumps({"status": "ready" if freeze["passed"] else "blocked", "checks": checks}, ensure_ascii=False, indent=2))
    return 0 if freeze["passed"] else 2


def _finite_training_log(attempt: Mapping[str, Any]) -> bool:
    real = dict(dict(attempt.get("result") or {}).get("real_execution") or {})
    for row in real.get("loss_history") or []:
        if not isinstance(row, Mapping):
            continue
        for key, value in row.items():
            if isinstance(value, (int, float)) and key not in {"epoch", "step"} and not math.isfinite(float(value)):
                return False
    loss = dict(attempt.get("result") or {}).get("train_loss")
    return loss is not None and math.isfinite(float(loss))


def _training_freeze_check(steps: int) -> dict[str, Any]:
    freeze = _read_json(EVIDENCE_ROOT / "pre_experiment_freeze.json")
    rows = _read_jsonl(PREPARATION_ROOT / "selected_dpo_pairs.jsonl")
    spec = _job_spec(rows, steps=steps, output_dir=TRAINER_OUTPUT_ROOT / f"{steps}step")
    checks = {
        "pre_experiment_freeze_passed": freeze.get("passed") is True,
        "source_files_unchanged": _source_hashes() == freeze.get("source_sha256"),
        "pairs_unchanged": stable_hash(rows) == freeze.get("pair_manifest_sha256"),
        "job_spec_unchanged": stable_hash(spec) == dict(freeze.get("job_spec_sha256") or {}).get(str(steps)),
        "parent_adapter_unchanged": _parent_validation() == freeze.get("parent_adapter"),
        "phase106_decision_unchanged": _sha256(PHASE106_ROOT / "phase106-final-decision.json") == freeze.get("phase106_decision_sha256"),
        "step_is_frozen": steps in (1, 12, 30),
        "no_existing_attempt": not (TRAINING_ROOT / f"{steps}step/training_attempt.json").exists(),
    }
    if steps == 12:
        prior = _read_json(TRAINING_ROOT / "1step/training_attempt.json") if (TRAINING_ROOT / "1step/training_attempt.json").is_file() else {}
        checks["one_step_completed"] = prior.get("status") == "completed"
        checks["one_step_adapter_valid"] = dict(prior.get("adapter_validation") or {}).get("valid") is True
        checks["one_step_metrics_finite"] = _finite_training_log(prior)
    if steps == 30:
        prior = _read_json(TRAINING_ROOT / "12step/training_attempt.json") if (TRAINING_ROOT / "12step/training_attempt.json").is_file() else {}
        checks["twelve_step_completed"] = prior.get("status") == "completed"
        checks["twelve_step_adapter_valid"] = dict(prior.get("adapter_validation") or {}).get("valid") is True
        checks["twelve_step_metrics_finite"] = _finite_training_log(prior)
    return {"kind": "phase107_training_freeze_check", "steps": steps, "passed": all(checks.values()), "checks": checks}


def _train(steps: int, clean: bool) -> int:
    if steps not in (1, 12, 30):
        raise SystemExit("Phase107 permits 1, 12, or 30 steps only")
    evidence_dir = TRAINING_ROOT / f"{steps}step"
    output_dir = TRAINER_OUTPUT_ROOT / f"{steps}step"
    if clean and evidence_dir.exists():
        _safe_clean(evidence_dir, TRAINING_ROOT)
    if clean and output_dir.exists():
        _safe_clean(output_dir, TRAINER_OUTPUT_ROOT)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_jsonl(PREPARATION_ROOT / "selected_dpo_pairs.jsonl")
    freeze = _training_freeze_check(steps)
    spec = _job_spec(rows, steps=steps, output_dir=output_dir)
    _write_json(evidence_dir / "freeze_check.json", freeze)
    _write_json(evidence_dir / "dpo_job_spec.json", spec)
    if not freeze["passed"]:
        attempt = {
            "kind": "phase107_dpo_training_attempt",
            "status": "blocked",
            "requested_steps": steps,
            "reason": "freeze_check_failed",
            "freeze_check": freeze,
            "product_gate_qualified": False,
        }
        _write_json(evidence_dir / "training_attempt.json", attempt)
        return 2
    started = time.perf_counter()
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = execute_dpo_training(job_spec={**spec, "dry_run": False}, dry_run=False)
    duration = round(time.perf_counter() - started, 4)
    real = dict(result.get("real_execution") or {})
    artifact_dir = Path(str(real.get("artifact_dir") or ""))
    validation = validate_adapter_artifact(
        artifact_dir,
        {"artifact_name": "adapter_model.safetensors", "artifact_format": "peft_lora"},
    ) if artifact_dir.is_dir() else {"valid": False, "reason": "artifact_dir_missing"}
    adapter = artifact_dir / "adapter_model.safetensors"
    validation.update({
        "artifact_dir": str(artifact_dir),
        "sha256": _sha256(adapter) if adapter.is_file() else None,
        "requested_steps": steps,
        "completed_steps": int(real.get("steps") or 0),
        "parent_adapter_sha256": _parent_validation().get("actual_sha256"),
        "lineage": "qwen3_4b_phase106_sft_merge_then_phase107_dpo",
    })
    completed = (
        result.get("status") == "completed"
        and real.get("success") is True
        and int(real.get("steps") or 0) == steps
        and real.get("parameters_updated") is True
        and validation.get("valid") is True
    )
    attempt = {
        "kind": "phase107_dpo_training_attempt",
        "status": "completed" if completed else "failed",
        "real_training": completed,
        "requested_steps": steps,
        "completed_steps": int(real.get("steps") or 0),
        "duration_seconds": duration,
        "resource_usage": {
            "ru_maxrss_before": rss_before,
            "ru_maxrss_after": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "ru_maxrss_unit": "bytes_on_macos",
        },
        "parent_adapter": _parent_validation(),
        "result": result,
        "adapter_validation": validation,
        "simulated_usage": True,
        "actual_user_feedback": False,
        "product_gate_qualified": False,
        "automatic_promotion_allowed": False,
    }
    _write_json(evidence_dir / "training_attempt.json", attempt)
    _write_json(evidence_dir / "train_log.json", {
        "status": attempt["status"],
        "requested_steps": steps,
        "completed_steps": attempt["completed_steps"],
        "train_loss": result.get("train_loss"),
        "loss_history": real.get("loss_history") or [],
        "non_finite_metrics": real.get("non_finite_metrics") or [],
        "parameters_updated": real.get("parameters_updated"),
        "runtime_audit": real.get("runtime_audit"),
        "error": result.get("error"),
    })
    _write_json(evidence_dir / "adapter_validation.json", validation)
    print(json.dumps({
        "status": attempt["status"],
        "requested_steps": steps,
        "completed_steps": attempt["completed_steps"],
        "duration_seconds": duration,
        "train_loss": result.get("train_loss"),
        "error": result.get("error"),
        "adapter_valid": validation.get("valid"),
    }, ensure_ascii=False, indent=2))
    return 0 if completed else 1


def _phase107_adapter_dir() -> Path:
    attempt = _read_json(TRAINING_ROOT / "30step/training_attempt.json")
    validation = dict(attempt.get("adapter_validation") or {})
    path = Path(str(validation.get("artifact_dir") or ""))
    if attempt.get("status") != "completed" or validation.get("valid") is not True or not path.is_dir():
        raise SystemExit("Phase107 30-step DPO adapter is unavailable")
    return path.resolve()


def _load_phase107_runtime(adapter_path: Path) -> tuple[Any, Any, Any, str]:
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
    model = AutoModelForCausalLM.from_pretrained(str(MODEL_PATH), local_files_only=True, low_cpu_mem_usage=True, dtype=dtype)
    model = PeftModel.from_pretrained(model, str(PARENT_ADAPTER_ROOT), local_files_only=True)
    model = model.merge_and_unload()
    model = PeftModel.from_pretrained(model, str(adapter_path), local_files_only=True)
    model.to(device)
    model.eval()
    return torch, tokenizer, model, device


def _existing_model_calls() -> int:
    return sum(
        int(_read_json(path).get("model_call_count") or 0)
        for path in EVAL_ROOT.glob("*/metrics.json")
        if path.is_file()
    ) if EVAL_ROOT.exists() else 0


def _eval_freeze_check(variant: str, adapter_path: Path | None) -> dict[str, Any]:
    freeze = _read_json(EVIDENCE_ROOT / "pre_experiment_freeze.json")
    holdout = _read_json(PREPARATION_ROOT / "holdout.json")
    training = _read_json(TRAINING_ROOT / "30step/training_attempt.json") if (TRAINING_ROOT / "30step/training_attempt.json").is_file() else {}
    checks = {
        "pre_experiment_freeze_passed": freeze.get("passed") is True,
        "source_files_unchanged": _source_hashes() == freeze.get("source_sha256"),
        "holdout_unchanged": stable_hash(holdout.get("sessions") or []) == freeze.get("holdout_manifest_sha256"),
        "parent_adapter_unchanged": _parent_validation() == freeze.get("parent_adapter"),
        "phase107_training_completed_or_not_required": variant != "phase107_dpo" or training.get("status") == "completed",
        "adapter_available_or_base": adapter_path is None or adapter_path.is_dir(),
        "variant_frozen": variant in {"base", "phase106_sft", "phase107_dpo"},
        "call_budget_not_exceeded": _existing_model_calls() + 60 <= MODEL_CALL_BUDGET,
        "no_completed_eval_exists": not (EVAL_ROOT / variant / "metrics.json").exists(),
    }
    return {"kind": "phase107_eval_freeze_check", "variant": variant, "passed": all(checks.values()), "checks": checks}


def _evaluate(variant: str, clean: bool) -> int:
    if variant not in {"base", "phase106_sft", "phase107_dpo"}:
        raise SystemExit("unsupported Phase107 eval variant")
    adapter_path = None
    if variant == "phase106_sft":
        adapter_path = PARENT_ADAPTER_ROOT
    elif variant == "phase107_dpo":
        adapter_path = _phase107_adapter_dir()
    output_root = EVAL_ROOT / variant
    if clean and output_root.exists():
        _safe_clean(output_root, EVAL_ROOT)
    cache_path = PRIVATE_ROOT / f"{variant}.jsonl"
    if clean:
        cache_path.unlink(missing_ok=True)
    freeze = _eval_freeze_check(variant, adapter_path)
    _write_json(output_root / "freeze_check.json", freeze)
    if not freeze["passed"]:
        return 2
    sessions = [dict(row) for row in _read_json(PREPARATION_ROOT / "holdout.json").get("sessions") or []]
    structural_rows: list[dict[str, Any]] = []
    scores: list[dict[str, Any]] = []
    private_rows: list[dict[str, Any]] = []
    torch = tokenizer = model = device = None
    try:
        if variant == "phase107_dpo":
            torch, tokenizer, model, device = _load_phase107_runtime(adapter_path)
        else:
            torch, tokenizer, model, device = _load_runtime(adapter_path)
        for index, session in enumerate(sessions, start=1):
            structural, private = _run_session(session=session, torch=torch, tokenizer=tokenizer, model=model, device=device)
            outputs = [str(turn.get("raw_output") or "") for turn in private.get("turns") or []]
            score = score_phase107_session(
                session=session,
                outputs=outputs,
                structural_turns=structural.get("turns") or [],
            )
            structural.update({
                "kind": "phase107_structural_session",
                "scenario_type": session.get("scenario_type"),
                "phase107_score": score,
            })
            structural_rows.append(structural)
            scores.append(score)
            private_rows.append({
                "session_id": session.get("session_id"),
                "scenario_type": session.get("scenario_type"),
                "turns": private.get("turns") or [],
                "accepted": score.get("accepted"),
            })
            _write_jsonl(output_root / "structural_sessions.jsonl", structural_rows)
            _write_jsonl(output_root / "simulated_user_scores.jsonl", scores)
            _write_private_jsonl(cache_path, private_rows)
            print(f"[phase107:{variant}] {index}/{len(sessions)} {session.get('session_id')} accepted={score['accepted']}", flush=True)
    finally:
        if model is not None:
            del model
            if device == "mps" and torch is not None:
                torch.mps.empty_cache()
    metrics = aggregate_phase107_scores(scores)
    payload = {
        "kind": "phase107_variant_metrics",
        "variant": variant,
        "model_call_count": sum(int(row.get("turn_count") or 0) for row in structural_rows),
        "metrics": metrics,
        "adapter_loaded": adapter_path is not None,
        "guided_generation_used": False,
        "private_cache": str(cache_path),
        "private_cache_outside_repo": True,
        "simulated_usage": True,
        "actual_user_feedback_count": 0,
    }
    _write_json(output_root / "metrics.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _build_manifest() -> dict[str, Any]:
    excluded = {EVIDENCE_ROOT / "evidence_manifest.json", EVIDENCE_ROOT / "validation_summary.json"}
    files = [path for path in sorted(EVIDENCE_ROOT.rglob("*")) if path.is_file() and path not in excluded]
    return {
        "kind": "phase107_evidence_manifest",
        "files": [
            {"path": str(path.relative_to(REPO_ROOT)), "sha256": _sha256(path), "size_bytes": path.stat().st_size}
            for path in files
        ],
        "file_count": len(files),
        "private_transcripts_committed": False,
        "actual_user_feedback_count": 0,
    }


def _decide() -> int:
    metrics = {
        variant: dict(_read_json(EVAL_ROOT / f"{variant}/metrics.json").get("metrics") or {})
        for variant in ("base", "phase106_sft", "phase107_dpo")
    }
    training = _read_json(TRAINING_ROOT / "30step/training_attempt.json")
    parent = _parent_validation()
    decision = build_phase107_decision(
        base_metrics=metrics["base"],
        phase106_metrics=metrics["phase106_sft"],
        candidate_metrics=metrics["phase107_dpo"],
        training_completed=training.get("status") == "completed" and training.get("real_training") is True,
        parent_lineage_valid=parent.get("valid") is True
        and dict(training.get("adapter_validation") or {}).get("parent_adapter_sha256") == parent.get("actual_sha256"),
    )
    runtime_guaranteed = all(
        row.get("provenance_envelope_integrity_rate") == 1.0
        and row.get("metadata_injection_resistance_rate") == 1.0
        for row in metrics.values()
    )
    decision.update({
        "base_metrics": metrics["base"],
        "phase106_sft_metrics": metrics["phase106_sft"],
        "phase107_dpo_metrics": metrics["phase107_dpo"],
        "runtime_provenance_deterministically_guaranteed": runtime_guaranteed,
        "selected_training_steps": 30,
        "model_call_count": sum(int(_read_json(EVAL_ROOT / f"{variant}/metrics.json").get("model_call_count") or 0) for variant in metrics),
        "private_transcripts_committed": False,
    })
    _write_json(EVIDENCE_ROOT / "phase107-final-decision.json", decision)
    lines = [
        "# Phase107 Final Decision",
        "",
        f"- Status: `{decision['status']}`",
        f"- Recommendation: `{decision['recommendation']}`",
        f"- Runtime provenance deterministically guaranteed: `{str(runtime_guaranteed).lower()}`",
        f"- Phase106 semantic/literal: `{metrics['phase106_sft'].get('semantic_provenance_rate')}` / `{metrics['phase106_sft'].get('literal_provenance_rate')}`",
        f"- Phase107 semantic/literal: `{metrics['phase107_dpo'].get('semantic_provenance_rate')}` / `{metrics['phase107_dpo'].get('literal_provenance_rate')}`",
        "- Product gate qualified: `false`",
        "- Automatic promotion allowed: `false`",
    ]
    (EVIDENCE_ROOT / "phase107-final-decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    runbook = [
        "# Phase107 Runbook",
        "",
        "```bash",
        ".venv/bin/python tools/phase107_runtime_provenance_and_token_faithful_dpo.py prepare --clean",
        ".venv/bin/python tools/phase107_runtime_provenance_and_token_faithful_dpo.py train --steps 1 --clean",
        ".venv/bin/python tools/phase107_runtime_provenance_and_token_faithful_dpo.py train --steps 12 --clean",
        ".venv/bin/python tools/phase107_runtime_provenance_and_token_faithful_dpo.py train --steps 30 --clean",
        ".venv/bin/python tools/phase107_runtime_provenance_and_token_faithful_dpo.py eval --variant base --clean",
        ".venv/bin/python tools/phase107_runtime_provenance_and_token_faithful_dpo.py eval --variant phase106_sft --clean",
        ".venv/bin/python tools/phase107_runtime_provenance_and_token_faithful_dpo.py eval --variant phase107_dpo --clean",
        ".venv/bin/python tools/phase107_runtime_provenance_and_token_faithful_dpo.py decide",
        ".venv/bin/python tools/phase107_runtime_provenance_and_token_faithful_dpo.py validate",
        "```",
        "",
        "All model calls are local and simulated. Never auto-promote this candidate.",
    ]
    (EVIDENCE_ROOT / "phase107-runbook.md").write_text("\n".join(runbook) + "\n", encoding="utf-8")
    _write_json(EVIDENCE_ROOT / "evidence_manifest.json", _build_manifest())
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return 0 if decision["passed"] else 1


def _validate() -> int:
    decision = _read_json(EVIDENCE_ROOT / "phase107-final-decision.json")
    manifest = _read_json(EVIDENCE_ROOT / "evidence_manifest.json")
    expected = {str(row["path"]): str(row["sha256"]) for row in manifest.get("files") or []}
    excluded = {EVIDENCE_ROOT / "evidence_manifest.json", EVIDENCE_ROOT / "validation_summary.json"}
    current = {
        str(path.relative_to(REPO_ROOT)): _sha256(path)
        for path in sorted(EVIDENCE_ROOT.rglob("*"))
        if path.is_file() and path not in excluded
    }
    phase106 = _read_json(PHASE106_ROOT / "phase106-final-decision.json")
    text = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in EVIDENCE_ROOT.rglob("*")
        if path.is_file()
    )
    checks = {
        "manifest_unchanged": expected == current,
        "phase106_remains_archive": str(phase106.get("status") or "").startswith("archive_"),
        "runtime_provenance_guaranteed": decision.get("runtime_provenance_deterministically_guaranteed") is True,
        "real_training_completed": dict(decision.get("checks") or {}).get("real_dpo_training_completed") is True,
        "recommendation_allowed": decision.get("recommendation") in {"runtime_contract_remains_primary", "promote_after_manual_review"},
        "product_gate_false": decision.get("product_gate_qualified") is False,
        "automatic_promotion_false": decision.get("automatic_promotion_allowed") is False,
        "model_call_count_180": decision.get("model_call_count") == MODEL_CALL_BUDGET,
        "actual_feedback_zero": decision.get("actual_user_feedback_count") == 0,
        "private_transcripts_not_committed": decision.get("private_transcripts_committed") is False,
        "raw_output_field_absent": '"raw_output":' not in text,
        "required_directories_present": all(
            path.is_dir()
            for path in (PREPARATION_ROOT, RUNTIME_ROOT, TRAINING_ROOT, EVAL_ROOT, FAILURE_ROOT)
        ),
    }
    payload = {"kind": "phase107_validation_summary", "validated_at": _utcnow(), "passed": all(checks.values()), "checks": checks}
    _write_json(EVIDENCE_ROOT / "validation_summary.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["passed"] else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--clean", action="store_true")
    train = sub.add_parser("train")
    train.add_argument("--steps", type=int, required=True)
    train.add_argument("--clean", action="store_true")
    evaluate = sub.add_parser("eval")
    evaluate.add_argument("--variant", required=True)
    evaluate.add_argument("--clean", action="store_true")
    sub.add_parser("decide")
    sub.add_parser("validate")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "prepare":
        return _prepare(args.clean)
    if args.command == "train":
        return _train(args.steps, args.clean)
    if args.command == "eval":
        return _evaluate(args.variant, args.clean)
    if args.command == "decide":
        return _decide()
    if args.command == "validate":
        return _validate()
    raise SystemExit("unsupported command")


if __name__ == "__main__":
    raise SystemExit(main())
