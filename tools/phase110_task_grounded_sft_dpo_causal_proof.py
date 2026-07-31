#!/usr/bin/env python3
"""Run the bounded local Phase110 task-grounded SFT/DPO causal proof."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import statistics
import sys
import time
import traceback
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = REPO_ROOT / "pfe-core"
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

from pfe_core.adapter_store.quality import validate_adapter_artifact
from pfe_core.phase99_qwen3_native_generation_boundary import render_qwen3_no_think_prompt
from pfe_core.phase110_task_grounded_sft_dpo import (
    PHASE110_BASELINE_VARIANTS,
    PHASE110_DIAGNOSTIC_COUNT,
    PHASE110_FINAL_VARIANTS,
    PHASE110_HOLDOUT_COUNT,
    PHASE110_RUNTIME_CONTRACT,
    aggregate_phase110_scores,
    audit_phase110_data,
    build_phase110_diagnostic_prompts,
    build_phase110_dpo_pairs,
    build_phase110_final_decision,
    build_phase110_holdout,
    build_phase110_sft_gate,
    build_phase110_sft_samples,
    compare_phase110_variants,
    score_phase110_output,
    stable_hash,
)
from pfe_core.trainer.executors import (
    _encode_sft_examples,
    _run_real_local_peft_training,
    execute_dpo_training,
)


EVIDENCE_ROOT = REPO_ROOT / "docs/demo/phase110-task-grounded-sft-dpo-causal-proof"
PREPARATION_ROOT = EVIDENCE_ROOT / "evidence-preparation"
DIAGNOSTIC_ROOT = EVIDENCE_ROOT / "evidence-diagnostic"
TRAINING_ROOT = EVIDENCE_ROOT / "evidence-training"
EVAL_ROOT = EVIDENCE_ROOT / "evidence-eval"
FAILURE_ROOT = EVIDENCE_ROOT / "evidence-failures"
PRIVATE_ROOT = Path("/private/tmp/pfe-phase110-simulated-review")
MODEL_PATH = REPO_ROOT / "models/Qwen3-4B"
PHASE109_ROOT = REPO_ROOT / "docs/demo/phase109-personal-engineering-copilot-benefit-proof"
PHASE109_ADAPTER = REPO_ROOT / "trainer_job_outputs/phase109-personal-engineering-dpo/30step/dpo_adapter"
SFT_OUTPUT_ROOT = REPO_ROOT / "trainer_job_outputs/phase110-task-grounded-sft"
DPO_OUTPUT_ROOT = REPO_ROOT / "trainer_job_outputs/phase110-task-grounded-dpo"
CALL_LEDGER = EVAL_ROOT / "call_ledger.jsonl"
MAX_MODEL_CALL_BUDGET = PHASE110_HOLDOUT_COUNT * len(PHASE110_FINAL_VARIANTS)

GENERATION_PROTOCOL = {
    "input_max_length": 3072,
    "max_new_tokens": 160,
    "do_sample": False,
    "repetition_penalty": 1.12,
    "no_repeat_ngram_size": 4,
    "enable_thinking": False,
    "automatic_retry_count": 0,
    "post_hoc_truncation_allowed": False,
}
SFT_RUNTIME = {
    "steps": [1, 12, 30],
    "max_length": 384,
    "learning_rate": 0.0001,
    "seed": 110,
    "sampling_strategy": "seeded_stratified",
}
DPO_RUNTIME = {
    "steps": [1, 12, 30],
    "runtime_device": "mps",
    "runtime_dtype": "float32",
    "learning_rate": 0.000003,
    "beta": 0.1,
    "max_length": 512,
    "max_prompt_length": 384,
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.0,
}


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), ensure_ascii=False, sort_keys=True) + "\n")


def _write_private_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_clean(path: Path, parent: Path) -> None:
    resolved = path.resolve()
    if resolved == parent.resolve() or parent.resolve() not in resolved.parents:
        raise RuntimeError(f"refusing unsafe clean: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def _source_hashes() -> dict[str, str]:
    paths = {
        "core": CORE_ROOT / "pfe_core/phase110_task_grounded_sft_dpo.py",
        "driver": REPO_ROOT / "tools/phase110_task_grounded_sft_dpo_causal_proof.py",
        "core_test": REPO_ROOT / "tests/test_phase110_task_grounded_sft_dpo.py",
        "driver_test": REPO_ROOT / "tests/test_phase110_driver_safety.py",
        "executor": CORE_ROOT / "pfe_core/trainer/executors.py",
        "phase109_decision": PHASE109_ROOT / "phase109-final-decision.json",
    }
    return {name: _sha256(path) for name, path in paths.items()}


def _artifact_manifest(path: Path) -> dict[str, Any]:
    artifact = path / "adapter_model.safetensors"
    validation = validate_adapter_artifact(path, {"artifact_name": artifact.name, "artifact_format": "peft_lora"}) if path.is_dir() else {"valid": False, "reason": "adapter_directory_missing"}
    return {**validation, "artifact_dir": str(path), "artifact_sha256": _sha256(artifact) if artifact.is_file() else None}


def _model_manifest() -> dict[str, Any]:
    config = MODEL_PATH / "config.json"
    files = list(MODEL_PATH.rglob("*")) if MODEL_PATH.is_dir() else []
    return {
        "path": str(MODEL_PATH), "exists": MODEL_PATH.is_dir(),
        "file_count": sum(path.is_file() for path in files),
        "total_bytes": sum(path.stat().st_size for path in files if path.is_file()),
        "config_sha256": _sha256(config) if config.is_file() else None,
        "required_files_present": config.is_file() and (MODEL_PATH / "tokenizer_config.json").is_file(),
    }


def _sft_job_spec(rows: list[dict[str, Any]], *, steps: int, output_dir: Path) -> dict[str, Any]:
    examples = [
        {
            "sample_id": row["sample_id"],
            "category": row["category"],
            "instruction": str(row["prompt_messages"][-1]["content"]),
            "messages": [{"role": "system", "content": PHASE110_RUNTIME_CONTRACT}, *[{"role": message["role"], "content": message["content"]} for message in row["prompt_messages"]]],
            "chosen": row["chosen"],
            "rejected": None,
            "sample_type": "sft",
            "feedback_source": "simulated_usage",
            "actual_user_feedback": False,
        }
        for row in rows
    ]
    return {
        "backend": "peft", "execution_backend": "peft", "execution_executor": "peft",
        "executor_mode": "real_local", "ready": bool(examples), "dry_run": False,
        "recipe": {"training": {
            "method": "lora", "train_type": "sft_completion_only", "base_model_path": str(MODEL_PATH),
            "base_model": str(MODEL_PATH), "local_only": True, "epochs": 1, "max_steps": steps,
            "max_length": SFT_RUNTIME["max_length"], "learning_rate": SFT_RUNTIME["learning_rate"],
            "seed": SFT_RUNTIME["seed"], "sampling_strategy": SFT_RUNTIME["sampling_strategy"],
            "output_dir": str(output_dir),
        }},
        "audit": {"import_probe": {"ready": True, "missing_modules": []}},
        "training_examples": examples,
        "phase110": {"stage": "task_grounded_sft", "steps": steps, "completion_only_loss_required": True, "simulated_usage": True, "actual_user_feedback": False, "automatic_promotion_allowed": False},
    }


def _render_dpo_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), local_files_only=True)
    rendered: list[dict[str, Any]] = []
    prompt_lengths: list[int] = []
    total_lengths: list[int] = []
    for row in rows:
        messages = [{"role": "system", "content": PHASE110_RUNTIME_CONTRACT}, *[{"role": message["role"], "content": message["content"]} for message in row["prompt_messages"]]]
        prompt = render_qwen3_no_think_prompt(tokenizer, messages)
        item = {**row, "prompt": prompt, "instruction": prompt}
        rendered.append(item)
        prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
        prompt_lengths.append(prompt_tokens)
        total_lengths.extend(prompt_tokens + len(tokenizer.encode(str(item[key]), add_special_tokens=False)) for key in ("chosen", "rejected"))
    diagnostic = {
        "kind": "phase110_dpo_tokenizer_diagnostic", "prompt_count": len(rendered),
        "max_prompt_tokens": max(prompt_lengths, default=0), "max_total_tokens": max(total_lengths, default=0),
        "all_prompts_within_limit": max(prompt_lengths, default=0) <= DPO_RUNTIME["max_prompt_length"],
        "all_pairs_within_limit": max(total_lengths, default=0) <= DPO_RUNTIME["max_length"],
        "all_prompts_no_think_aligned": all(str(row["prompt"]).endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n") for row in rendered),
    }
    diagnostic["passed"] = all(value is True for key, value in diagnostic.items() if key.startswith("all_"))
    return rendered, diagnostic


def _dpo_job_spec(rows: list[dict[str, Any]], *, steps: int, output_dir: Path, parent_adapter: Path) -> dict[str, Any]:
    return {
        "backend": "dpo", "execution_backend": "dpo", "execution_executor": "dpo",
        "executor_mode": "real_import", "dry_run": True, "output_dir": str(output_dir),
        "recipe": {
            "training": {
                "method": "lora", "epochs": 1, "max_steps": steps, "learning_rate": DPO_RUNTIME["learning_rate"],
                "train_type": "dpo", "base_model": str(MODEL_PATH), "base_model_path": str(MODEL_PATH),
                "local_only": True, "num_train_samples": len(rows), "output_dir": str(output_dir),
                "runtime_device": DPO_RUNTIME["runtime_device"], "runtime_dtype": DPO_RUNTIME["runtime_dtype"],
                "incremental_context": {"parent_adapter_path": str(parent_adapter)},
            },
            "peft": {
                "trainer_class": "trl.DPOTrainer",
                "dpo_config": {"beta": DPO_RUNTIME["beta"], "label_smoothing": 0.0, "max_length": DPO_RUNTIME["max_length"], "max_prompt_length": DPO_RUNTIME["max_prompt_length"]},
                "lora_config": {"r": DPO_RUNTIME["lora_r"], "lora_alpha": DPO_RUNTIME["lora_alpha"], "lora_dropout": DPO_RUNTIME["lora_dropout"]},
            },
        },
        "training_examples": rows,
        "phase110": {"stage": "task_grounded_dpo", "steps": steps, "parent_adapter": str(parent_adapter), "simulated_usage": True, "actual_user_feedback": False, "automatic_promotion_allowed": False},
    }


def _completion_boundary_report(spec: Mapping[str, Any]) -> dict[str, Any]:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), local_files_only=True)
    encoded = _encode_sft_examples(tokenizer=tokenizer, training_examples=list(spec["training_examples"]), max_length=int(spec["recipe"]["training"]["max_length"]), vocab_size=int(getattr(tokenizer, "vocab_size", 151936)))
    counts = [sum(int(value) != -100 for value in row["labels"]) for row in encoded]
    return {
        "kind": "phase110_completion_boundary_report", "passed": len(counts) == 84 and min(counts, default=0) >= 12,
        "sample_count": len(counts), "minimum_completion_label_token_count": min(counts, default=0),
        "prompt_tokens_use_loss": False, "completion_tokens_use_loss": True,
    }


def _attempted_call_count() -> int:
    return sum(row.get("event") == "attempted" for row in _read_jsonl(CALL_LEDGER))


def _reserve_call(variant: str, session_id: str) -> str:
    call_id = f"phase110-{variant}-{session_id}"
    rows = _read_jsonl(CALL_LEDGER)
    if any(row.get("event") == "attempted" and row.get("call_id") == call_id for row in rows):
        raise RuntimeError(f"duplicate Phase110 call id: {call_id}")
    if _attempted_call_count() >= MAX_MODEL_CALL_BUDGET:
        raise RuntimeError(f"Phase110 model call budget exhausted: {MAX_MODEL_CALL_BUDGET}")
    _append_jsonl(CALL_LEDGER, {"event": "attempted", "call_id": call_id, "variant": variant, "session_id": session_id, "provider": "local_qwen3_4b", "created_at": _utcnow()})
    return call_id


def _prepare(clean: bool) -> int:
    if clean:
        if _attempted_call_count():
            raise RuntimeError("Phase110 clean refused after model-call ledger contains attempts")
        if EVIDENCE_ROOT.exists():
            _safe_clean(EVIDENCE_ROOT, EVIDENCE_ROOT.parent)
        if PRIVATE_ROOT.exists():
            _safe_clean(PRIVATE_ROOT, PRIVATE_ROOT.parent)
    for root in (PREPARATION_ROOT, DIAGNOSTIC_ROOT, TRAINING_ROOT, EVAL_ROOT, FAILURE_ROOT, PRIVATE_ROOT):
        root.mkdir(parents=True, exist_ok=True)
    sft = build_phase110_sft_samples()
    dpo = build_phase110_dpo_pairs()
    holdout = build_phase110_holdout()
    previous_holdout = _read_json(PHASE109_ROOT / "evidence-preparation/holdout.json")
    integrity = audit_phase110_data(sft, dpo, holdout, previous_holdout)
    rendered_dpo, dpo_diagnostic = _render_dpo_rows(dpo)
    sft_specs = {str(steps): _sft_job_spec(sft, steps=steps, output_dir=SFT_OUTPUT_ROOT / f"{steps}step") for steps in SFT_RUNTIME["steps"]}
    completion = _completion_boundary_report(sft_specs["30"])
    phase109 = _read_json(PHASE109_ROOT / "phase109-final-decision.json")
    checks = {
        "phase109_archived": str(phase109.get("status") or "").startswith("archive_"),
        "phase109_product_gate_false": phase109.get("product_gate_qualified") is False,
        "data_integrity_passed": integrity.get("passed") is True,
        "completion_boundary_passed": completion.get("passed") is True,
        "dpo_tokenizer_diagnostic_passed": dpo_diagnostic.get("passed") is True,
        "model_available": _model_manifest().get("required_files_present") is True,
        "phase109_adapter_available": _artifact_manifest(PHASE109_ADAPTER).get("valid") is True,
        "private_root_outside_repo": REPO_ROOT.resolve() not in PRIVATE_ROOT.resolve().parents,
        "call_ledger_empty": _attempted_call_count() == 0,
    }
    freeze = {
        "kind": "phase110_pre_experiment_freeze", "passed": all(checks.values()), "checks": checks,
        "source_sha256": _source_hashes(), "sft_manifest_sha256": stable_hash(sft),
        "dpo_manifest_sha256": stable_hash(rendered_dpo), "holdout_manifest_sha256": stable_hash(holdout),
        "runtime_contract_sha256": stable_hash(PHASE110_RUNTIME_CONTRACT), "model_manifest": _model_manifest(),
        "phase109_adapter_manifest": _artifact_manifest(PHASE109_ADAPTER), "generation_protocol": GENERATION_PROTOCOL,
        "max_model_call_budget": MAX_MODEL_CALL_BUDGET, "diagnostic_forward_passes": PHASE110_DIAGNOSTIC_COUNT * 2,
        "dpo_requires_sft_gate": True, "external_provider_allowed": False, "paid_api_allowed": False,
        "automatic_promotion_allowed": False, "product_gate_qualified": False, "frozen_at": _utcnow(),
    }
    _write_jsonl(PREPARATION_ROOT / "selected_sft_samples.jsonl", sft)
    _write_jsonl(PREPARATION_ROOT / "selected_dpo_pairs.jsonl", rendered_dpo)
    _write_json(PREPARATION_ROOT / "holdout.json", holdout)
    _write_json(PREPARATION_ROOT / "holdout_integrity_check.json", integrity)
    _write_json(PREPARATION_ROOT / "completion_boundary_report.json", completion)
    _write_json(PREPARATION_ROOT / "dpo_tokenizer_diagnostic.json", dpo_diagnostic)
    _write_json(PREPARATION_ROOT / "diagnostic_prompts.json", {"count": PHASE110_DIAGNOSTIC_COUNT, "manifest_sha256": stable_hash(build_phase110_diagnostic_prompts()), "raw_prompts_committed": False})
    for steps, spec in sft_specs.items():
        _write_json(PREPARATION_ROOT / f"sft_job_spec_{steps}step.json", spec)
    _write_json(EVIDENCE_ROOT / "pre_experiment_freeze.json", freeze)
    print(json.dumps({"status": "prepared" if freeze["passed"] else "blocked", "checks": checks}, ensure_ascii=False, indent=2))
    return 0 if freeze["passed"] else 2


def _load_model(adapter_path: Path | None = None, parent_adapter: Path | None = None) -> tuple[Any, Any, Any, str, list[str]]:
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
    load_plan = ["qwen3_4b_base"]
    if parent_adapter is not None:
        model = PeftModel.from_pretrained(model, str(parent_adapter), local_files_only=True)
        model = model.merge_and_unload()
        load_plan.append(f"merge:{parent_adapter.name}")
    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, str(adapter_path), local_files_only=True)
        load_plan.append(f"adapter:{adapter_path.name}")
    model.to(device)
    model.eval()
    return torch, tokenizer, model, device, load_plan


def _free_model(torch: Any, model: Any, device: str) -> None:
    del model
    gc.collect()
    if device == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()


def _diagnostic_logits(adapter_path: Path | None) -> tuple[list[Any], list[dict[str, Any]], dict[str, Any]]:
    torch, tokenizer, model, device, load_plan = _load_model(adapter_path)
    logits_rows: list[Any] = []
    token_rows: list[dict[str, Any]] = []
    for row in build_phase110_diagnostic_prompts():
        messages = [{"role": "system", "content": PHASE110_RUNTIME_CONTRACT}, *row["messages"]]
        prompt = render_qwen3_no_think_prompt(tokenizer, messages)
        encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3072, add_special_tokens=False)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            logits = model(**encoded).logits[0, -1].float().cpu()
        chosen_ids = tokenizer.encode(str(row["chosen"]), add_special_tokens=False)
        rejected_ids = tokenizer.encode(str(row["rejected"]), add_special_tokens=False)
        logits_rows.append(logits)
        token_rows.append({"diagnostic_id": row["diagnostic_id"], "chosen_first_token_id": chosen_ids[0], "rejected_first_token_id": rejected_ids[0]})
    runtime = {"device": device, "load_plan": load_plan}
    _free_model(torch, model, device)
    return logits_rows, token_rows, runtime


def _diagnose_adapter() -> int:
    freeze = _read_json(EVIDENCE_ROOT / "pre_experiment_freeze.json")
    if freeze.get("passed") is not True or _source_hashes() != freeze.get("source_sha256"):
        raise RuntimeError("Phase110 diagnostic freeze check failed")
    base_logits, tokens, base_runtime = _diagnostic_logits(None)
    adapter_logits, adapter_tokens, adapter_runtime = _diagnostic_logits(PHASE109_ADAPTER)
    if tokens != adapter_tokens:
        raise RuntimeError("diagnostic tokenization changed between model loads")
    import torch

    rows: list[dict[str, Any]] = []
    for base, candidate, token_info in zip(base_logits, adapter_logits, tokens):
        base_logp = torch.log_softmax(base, dim=-1)
        candidate_logp = torch.log_softmax(candidate, dim=-1)
        base_prob = torch.softmax(base, dim=-1)
        kl = float(torch.sum(base_prob * (base_logp - candidate_logp)).item())
        chosen = int(token_info["chosen_first_token_id"])
        rejected = int(token_info["rejected_first_token_id"])
        base_margin = float((base_logp[chosen] - base_logp[rejected]).item())
        candidate_margin = float((candidate_logp[chosen] - candidate_logp[rejected]).item())
        rows.append({
            **token_info, "max_abs_logit_delta": float(torch.max(torch.abs(candidate - base)).item()),
            "l2_logit_delta": float(torch.linalg.vector_norm(candidate - base).item()), "kl_base_to_adapter": kl,
            "base_greedy_token_id": int(torch.argmax(base).item()), "adapter_greedy_token_id": int(torch.argmax(candidate).item()),
            "greedy_token_changed": int(torch.argmax(base).item()) != int(torch.argmax(candidate).item()),
            "base_chosen_rejected_first_token_margin": base_margin,
            "adapter_chosen_rejected_first_token_margin": candidate_margin,
            "margin_delta": candidate_margin - base_margin,
        })
    max_deltas = [row["max_abs_logit_delta"] for row in rows]
    kls = [row["kl_base_to_adapter"] for row in rows]
    checks = {
        "phase109_adapter_valid": _artifact_manifest(PHASE109_ADAPTER).get("valid") is True,
        "twenty_prompts_measured": len(rows) == PHASE110_DIAGNOSTIC_COUNT,
        "all_values_finite": all(math.isfinite(float(value)) for row in rows for key, value in row.items() if key not in {"diagnostic_id", "greedy_token_changed"} and isinstance(value, (int, float))),
        "all_prompts_have_logit_change": all(value > 0.00001 for value in max_deltas),
        "median_max_abs_logit_delta_above_1e_4": statistics.median(max_deltas) > 0.0001,
        "mean_kl_above_1e_9": statistics.fmean(kls) > 0.000000001,
    }
    report = {
        "kind": "phase110_phase109_adapter_activation_diagnostic", "passed": all(checks.values()), "checks": checks,
        "prompt_count": len(rows), "base_runtime": base_runtime, "adapter_runtime": adapter_runtime,
        "adapter_manifest": _artifact_manifest(PHASE109_ADAPTER),
        "summary": {
            "median_max_abs_logit_delta": statistics.median(max_deltas), "mean_kl_base_to_adapter": statistics.fmean(kls),
            "greedy_next_token_change_count": sum(row["greedy_token_changed"] for row in rows),
            "mean_chosen_rejected_margin_delta": statistics.fmean(row["margin_delta"] for row in rows),
        },
        "details": rows, "generation_calls_used": 0, "raw_prompts_committed": False,
    }
    _write_json(DIAGNOSTIC_ROOT / "phase109_adapter_activation.json", report)
    print(json.dumps({"passed": report["passed"], "checks": checks, "summary": report["summary"]}, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 2


def _finite_losses(values: Iterable[Any]) -> bool:
    rows = list(values)
    return bool(rows) and all(math.isfinite(float(dict(row).get("loss"))) for row in rows)


def _sft_training_freeze(steps: int) -> dict[str, Any]:
    freeze = _read_json(EVIDENCE_ROOT / "pre_experiment_freeze.json")
    rows = _read_jsonl(PREPARATION_ROOT / "selected_sft_samples.jsonl")
    activation = _read_json(DIAGNOSTIC_ROOT / "phase109_adapter_activation.json") if (DIAGNOSTIC_ROOT / "phase109_adapter_activation.json").is_file() else {}
    checks = {
        "pre_experiment_freeze_passed": freeze.get("passed") is True,
        "source_files_unchanged": _source_hashes() == freeze.get("source_sha256"),
        "sft_samples_unchanged": stable_hash(rows) == freeze.get("sft_manifest_sha256"),
        "adapter_activation_passed": activation.get("passed") is True,
        "step_frozen": steps in SFT_RUNTIME["steps"],
        "no_existing_attempt": not (TRAINING_ROOT / f"sft-{steps}step/training_attempt.json").exists(),
    }
    if steps == 12:
        prior = _read_json(TRAINING_ROOT / "sft-1step/training_attempt.json") if (TRAINING_ROOT / "sft-1step/training_attempt.json").is_file() else {}
        checks["one_step_completed"] = prior.get("status") == "completed"
        checks["one_step_finite"] = _finite_losses(dict(prior.get("execution") or {}).get("loss_history") or [])
    if steps == 30:
        prior = _read_json(TRAINING_ROOT / "sft-12step/training_attempt.json") if (TRAINING_ROOT / "sft-12step/training_attempt.json").is_file() else {}
        checks["twelve_step_completed"] = prior.get("status") == "completed"
        checks["twelve_step_finite"] = _finite_losses(dict(prior.get("execution") or {}).get("loss_history") or [])
    return {"kind": "phase110_sft_training_freeze", "steps": steps, "passed": all(checks.values()), "checks": checks}


def _train_sft(steps: int, clean: bool) -> int:
    evidence_dir = TRAINING_ROOT / f"sft-{steps}step"
    output_dir = SFT_OUTPUT_ROOT / f"{steps}step"
    if clean:
        if evidence_dir.exists():
            _safe_clean(evidence_dir, TRAINING_ROOT)
        if output_dir.exists():
            _safe_clean(output_dir, SFT_OUTPUT_ROOT)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_jsonl(PREPARATION_ROOT / "selected_sft_samples.jsonl")
    spec = _sft_job_spec(rows, steps=steps, output_dir=output_dir)
    freeze = _sft_training_freeze(steps)
    boundary = _completion_boundary_report(spec)
    _write_json(evidence_dir / "training_manifest.json", spec)
    _write_json(evidence_dir / "freeze_check.json", freeze)
    _write_json(evidence_dir / "completion_boundary_report.json", boundary)
    if not freeze["passed"] or not boundary["passed"]:
        attempt = {"kind": "phase110_sft_training_attempt", "status": "blocked", "requested_steps": steps, "real_training": False, "reason": "freeze_or_completion_boundary_failed", "product_gate_qualified": False, "automatic_promotion_allowed": False}
        _write_json(evidence_dir / "training_attempt.json", attempt)
        return 2
    started = time.perf_counter()
    try:
        result = _run_real_local_peft_training(spec)
        real = dict(result.get("real_execution") or {})
        artifact_dir = Path(str(real.get("artifact_dir") or ""))
        validation = _artifact_manifest(artifact_dir)
        completed = result.get("status") == "completed" and real.get("success") is True and real.get("parameters_updated") is True and int(real.get("steps") or 0) == steps and validation.get("valid") is True
        attempt = {
            "kind": "phase110_sft_training_attempt", "status": "completed" if completed else "failed", "real_training": completed,
            "requested_steps": steps, "completed_steps": int(real.get("steps") or 0), "duration_seconds": round(time.perf_counter() - started, 4),
            "execution": real, "adapter_validation": validation, "simulated_usage": True, "actual_user_feedback": False,
            "product_gate_qualified": False, "automatic_promotion_allowed": False,
        }
    except Exception as exc:
        attempt = {"kind": "phase110_sft_training_attempt", "status": "failed", "real_training": False, "requested_steps": steps, "duration_seconds": round(time.perf_counter() - started, 4), "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(), "product_gate_qualified": False, "automatic_promotion_allowed": False}
    _write_json(evidence_dir / "training_attempt.json", attempt)
    _write_json(evidence_dir / "train_log.json", {"status": attempt["status"], "requested_steps": steps, "completed_steps": attempt.get("completed_steps", 0), "loss_history": dict(attempt.get("execution") or {}).get("loss_history") or [], "parameters_updated": dict(attempt.get("execution") or {}).get("parameters_updated"), "error": attempt.get("error")})
    if attempt.get("adapter_validation"):
        _write_json(evidence_dir / "adapter_validation.json", attempt["adapter_validation"])
    print(json.dumps({key: attempt.get(key) for key in ("status", "requested_steps", "completed_steps", "duration_seconds", "error")}, ensure_ascii=False, indent=2))
    return 0 if attempt["status"] == "completed" and _finite_losses(dict(attempt.get("execution") or {}).get("loss_history") or []) else 1


def _adapter_from_attempt(stage: str, steps: int = 30) -> Path:
    attempt = _read_json(TRAINING_ROOT / f"{stage}-{steps}step/training_attempt.json")
    path = Path(str(dict(attempt.get("adapter_validation") or {}).get("artifact_dir") or ""))
    if attempt.get("status") != "completed" or not path.is_dir():
        raise RuntimeError(f"Phase110 {stage} {steps}-step adapter unavailable")
    return path


def _variant_adapters(variant: str) -> tuple[Path | None, Path | None]:
    if variant == "base":
        return None, None
    if variant == "phase109_dpo":
        return PHASE109_ADAPTER, None
    if variant == "phase110_sft":
        return _adapter_from_attempt("sft"), None
    if variant == "phase110_sft_dpo":
        return _adapter_from_attempt("dpo"), _adapter_from_attempt("sft")
    raise RuntimeError(f"unsupported variant: {variant}")


def _generate_once(torch: Any, tokenizer: Any, model: Any, device: str, messages: list[dict[str, str]]) -> str:
    prompt = render_qwen3_no_think_prompt(tokenizer, messages)
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=GENERATION_PROTOCOL["input_max_length"], add_special_tokens=False)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    eos_ids = [tokenizer.eos_token_id]
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end, int) and im_end >= 0 and im_end not in eos_ids:
        eos_ids.append(im_end)
    with torch.inference_mode():
        generated = model.generate(
            **encoded, max_new_tokens=GENERATION_PROTOCOL["max_new_tokens"], do_sample=False,
            repetition_penalty=GENERATION_PROTOCOL["repetition_penalty"], no_repeat_ngram_size=GENERATION_PROTOCOL["no_repeat_ngram_size"],
            eos_token_id=eos_ids, pad_token_id=tokenizer.pad_token_id, use_cache=True,
        )
    return tokenizer.decode(generated[0, encoded["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def _eval_freeze(variant: str) -> dict[str, Any]:
    freeze = _read_json(EVIDENCE_ROOT / "pre_experiment_freeze.json")
    allowed = PHASE110_FINAL_VARIANTS if (TRAINING_ROOT / "dpo-30step/training_attempt.json").is_file() else PHASE110_BASELINE_VARIANTS
    checks = {
        "variant_allowed": variant in allowed,
        "pre_experiment_freeze_passed": freeze.get("passed") is True,
        "source_files_unchanged": _source_hashes() == freeze.get("source_sha256"),
        "holdout_unchanged": stable_hash(_read_json(PREPARATION_ROOT / "holdout.json")) == freeze.get("holdout_manifest_sha256"),
        "within_max_call_budget": _attempted_call_count() + PHASE110_HOLDOUT_COUNT <= MAX_MODEL_CALL_BUDGET,
        "variant_not_previously_run": not (EVAL_ROOT / variant / "metrics.json").exists(),
    }
    if variant in {"phase110_sft", "phase110_sft_dpo"}:
        checks["sft_adapter_valid"] = _artifact_manifest(_adapter_from_attempt("sft")).get("valid") is True
    if variant == "phase110_sft_dpo":
        checks["dpo_adapter_valid"] = _artifact_manifest(_adapter_from_attempt("dpo")).get("valid") is True
        checks["sft_gate_passed"] = _read_json(EVAL_ROOT / "sft_gate.json").get("passed") is True
    return {"kind": "phase110_eval_freeze", "variant": variant, "passed": all(checks.values()), "checks": checks}


def _evaluate(variant: str) -> int:
    evidence_dir = EVAL_ROOT / variant
    evidence_dir.mkdir(parents=True, exist_ok=True)
    freeze = _eval_freeze(variant)
    _write_json(evidence_dir / "freeze_check.json", freeze)
    if not freeze["passed"]:
        return 2
    adapter, parent = _variant_adapters(variant)
    torch, tokenizer, model, device, load_plan = _load_model(adapter, parent)
    sessions = list(_read_json(PREPARATION_ROOT / "holdout.json")["sessions"])
    scores: list[dict[str, Any]] = []
    private_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, session in enumerate(sessions, start=1):
        call_id = _reserve_call(variant, session["session_id"])
        messages = [{"role": "system", "content": PHASE110_RUNTIME_CONTRACT}, *[{"role": row["role"], "content": row["content"]} for row in session["messages"]]]
        try:
            output = _generate_once(torch, tokenizer, model, device, messages)
            if not output:
                raise RuntimeError("empty local generation")
        except Exception as exc:
            _append_jsonl(CALL_LEDGER, {"event": "failed", "call_id": call_id, "variant": variant, "session_id": session["session_id"], "error_type": type(exc).__name__, "created_at": _utcnow()})
            raise
        score = score_phase110_output(output, session)
        scores.append(score)
        private_rows.append({"variant": variant, "session_id": session["session_id"], "messages": messages, "output": output, "score": score, "usage_class": "simulated_usage", "actual_user_feedback": False})
        _append_jsonl(CALL_LEDGER, {"event": "completed", "call_id": call_id, "variant": variant, "session_id": session["session_id"], "output_sha256": score["output_sha256"], "created_at": _utcnow()})
        print(f"[{variant}] {index:02d}/{len(sessions)} {session['session_id']} score={score['overall_score']:.3f}", flush=True)
    _write_private_jsonl(PRIVATE_ROOT / f"{variant}.jsonl", private_rows)
    metrics = aggregate_phase110_scores(scores)
    metrics.update({"variant": variant, "model_call_count": len(sessions), "duration_seconds": round(time.perf_counter() - started, 4), "device": device, "load_plan": load_plan, "generation_protocol": GENERATION_PROTOCOL, "private_transcripts_committed": False, "external_provider_used": False})
    _write_json(evidence_dir / "metrics.json", metrics)
    _write_jsonl(evidence_dir / "structural_sessions.jsonl", scores)
    print(json.dumps({key: metrics[key] for key in ("variant", "session_count", "accepted_rate", "overall_score", "exact_three_line_rate", "duration_seconds")}, ensure_ascii=False, indent=2))
    _free_model(torch, model, device)
    return 0


def _analyze_sft() -> int:
    metrics = {variant: _read_json(EVAL_ROOT / variant / "metrics.json") for variant in PHASE110_BASELINE_VARIANTS}
    comparison = compare_phase110_variants(metrics["phase110_sft"], metrics["base"], seed=110)
    activation = _read_json(DIAGNOSTIC_ROOT / "phase109_adapter_activation.json")
    gate = build_phase110_sft_gate(activation_passed=activation.get("passed") is True, metrics=metrics, comparison=comparison)
    _write_json(EVAL_ROOT / "sft_comparison_summary.json", {"kind": "phase110_sft_comparison_summary", "metrics": metrics, "phase110_sft_vs_base": comparison, "raw_outputs_committed": False})
    _write_json(EVAL_ROOT / "sft_gate.json", gate)
    if not gate["passed"]:
        _write_json(TRAINING_ROOT / "dpo-not-run.json", {"kind": "phase110_conditional_dpo_status", "status": "not_run", "reason": "phase110_sft_gate_failed", "real_training": False, "product_gate_qualified": False, "automatic_promotion_allowed": False})
    print(json.dumps(gate, ensure_ascii=False, indent=2))
    return 0 if gate["passed"] else 3


def _dpo_training_freeze(steps: int) -> dict[str, Any]:
    freeze = _read_json(EVIDENCE_ROOT / "pre_experiment_freeze.json")
    gate = _read_json(EVAL_ROOT / "sft_gate.json") if (EVAL_ROOT / "sft_gate.json").is_file() else {}
    sft_adapter = _adapter_from_attempt("sft")
    rows = _read_jsonl(PREPARATION_ROOT / "selected_dpo_pairs.jsonl")
    checks = {
        "pre_experiment_freeze_passed": freeze.get("passed") is True,
        "source_files_unchanged": _source_hashes() == freeze.get("source_sha256"),
        "dpo_samples_unchanged": stable_hash(rows) == freeze.get("dpo_manifest_sha256"),
        "sft_gate_passed": gate.get("passed") is True,
        "sft_parent_adapter_valid": _artifact_manifest(sft_adapter).get("valid") is True,
        "step_frozen": steps in DPO_RUNTIME["steps"],
        "no_existing_attempt": not (TRAINING_ROOT / f"dpo-{steps}step/training_attempt.json").exists(),
    }
    if steps == 12:
        prior = _read_json(TRAINING_ROOT / "dpo-1step/training_attempt.json") if (TRAINING_ROOT / "dpo-1step/training_attempt.json").is_file() else {}
        checks["one_step_completed"] = prior.get("status") == "completed"
    if steps == 30:
        prior = _read_json(TRAINING_ROOT / "dpo-12step/training_attempt.json") if (TRAINING_ROOT / "dpo-12step/training_attempt.json").is_file() else {}
        checks["twelve_step_completed"] = prior.get("status") == "completed"
    return {"kind": "phase110_dpo_training_freeze", "steps": steps, "passed": all(checks.values()), "checks": checks}


def _train_dpo(steps: int, clean: bool) -> int:
    evidence_dir = TRAINING_ROOT / f"dpo-{steps}step"
    output_dir = DPO_OUTPUT_ROOT / f"{steps}step"
    if clean:
        if evidence_dir.exists():
            _safe_clean(evidence_dir, TRAINING_ROOT)
        if output_dir.exists():
            _safe_clean(output_dir, DPO_OUTPUT_ROOT)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_jsonl(PREPARATION_ROOT / "selected_dpo_pairs.jsonl")
    parent = _adapter_from_attempt("sft")
    spec = _dpo_job_spec(rows, steps=steps, output_dir=output_dir, parent_adapter=parent)
    freeze = _dpo_training_freeze(steps)
    _write_json(evidence_dir / "dpo_job_spec.json", spec)
    _write_json(evidence_dir / "freeze_check.json", freeze)
    if not freeze["passed"]:
        attempt = {"kind": "phase110_dpo_training_attempt", "status": "blocked", "requested_steps": steps, "real_training": False, "reason": "conditional_dpo_freeze_failed", "product_gate_qualified": False, "automatic_promotion_allowed": False}
        _write_json(evidence_dir / "training_attempt.json", attempt)
        return 2
    started = time.perf_counter()
    result = execute_dpo_training(job_spec={**spec, "dry_run": False}, dry_run=False)
    real = dict(result.get("real_execution") or {})
    artifact_dir = Path(str(real.get("artifact_dir") or ""))
    validation = _artifact_manifest(artifact_dir) if artifact_dir.is_dir() else {"valid": False, "reason": "artifact_dir_missing"}
    completed = result.get("status") == "completed" and real.get("success") is True and real.get("parameters_updated") is True and int(real.get("steps") or 0) == steps and validation.get("valid") is True
    attempt = {
        "kind": "phase110_dpo_training_attempt", "status": "completed" if completed else "failed", "real_training": completed,
        "requested_steps": steps, "completed_steps": int(real.get("steps") or 0), "duration_seconds": round(time.perf_counter() - started, 4),
        "result": result, "adapter_validation": validation, "parent_adapter": _artifact_manifest(parent),
        "simulated_usage": True, "actual_user_feedback": False, "product_gate_qualified": False, "automatic_promotion_allowed": False,
    }
    _write_json(evidence_dir / "training_attempt.json", attempt)
    _write_json(evidence_dir / "train_log.json", {"status": attempt["status"], "requested_steps": steps, "completed_steps": attempt["completed_steps"], "train_loss": result.get("train_loss"), "loss_history": real.get("loss_history") or [], "parameters_updated": real.get("parameters_updated"), "error": result.get("error")})
    _write_json(evidence_dir / "adapter_validation.json", validation)
    print(json.dumps({"status": attempt["status"], "steps": steps, "train_loss": result.get("train_loss"), "adapter_valid": validation.get("valid")}, ensure_ascii=False, indent=2))
    return 0 if completed else 1


def _build_final_analysis() -> tuple[dict[str, Any], dict[str, Any], bool]:
    sft_summary = _read_json(EVAL_ROOT / "sft_comparison_summary.json")
    metrics = dict(sft_summary["metrics"])
    dpo_completed = (TRAINING_ROOT / "dpo-30step/training_attempt.json").is_file() and _read_json(TRAINING_ROOT / "dpo-30step/training_attempt.json").get("status") == "completed"
    if dpo_completed:
        metrics["phase110_sft_dpo"] = _read_json(EVAL_ROOT / "phase110_sft_dpo/metrics.json")
    candidate = "phase110_sft_dpo" if dpo_completed else "phase110_sft"
    comparisons = {
        f"{candidate}_vs_base": compare_phase110_variants(metrics[candidate], metrics["base"], seed=111),
        f"{candidate}_vs_phase109": compare_phase110_variants(metrics[candidate], metrics["phase109_dpo"], seed=112),
        "phase110_sft_vs_base": sft_summary["phase110_sft_vs_base"],
    }
    summary = {"kind": "phase110_final_comparison_summary", "evaluated_candidate": candidate, "metrics": metrics, "comparisons": comparisons, "raw_outputs_committed": False, "actual_user_feedback": False}
    _write_json(EVAL_ROOT / "comparison_summary.json", summary)
    return metrics, comparisons, dpo_completed


def _build_manifest() -> dict[str, Any]:
    excluded = {EVIDENCE_ROOT / "evidence_manifest.json", EVIDENCE_ROOT / "validation_summary.json"}
    files = [{"path": str(path.relative_to(REPO_ROOT)), "sha256": _sha256(path), "bytes": path.stat().st_size} for path in sorted(EVIDENCE_ROOT.rglob("*")) if path.is_file() and path not in excluded]
    return {"kind": "phase110_evidence_manifest", "file_count": len(files), "files": files}


def _decide() -> int:
    metrics, comparisons, dpo_completed = _build_final_analysis()
    activation = _read_json(DIAGNOSTIC_ROOT / "phase109_adapter_activation.json")
    integrity = _read_json(PREPARATION_ROOT / "holdout_integrity_check.json")
    sft_attempt = _read_json(TRAINING_ROOT / "sft-30step/training_attempt.json")
    sft_gate = _read_json(EVAL_ROOT / "sft_gate.json")
    candidate = "phase110_sft_dpo" if dpo_completed else "phase110_sft"
    decision = build_phase110_final_decision(
        data_integrity_passed=integrity.get("passed") is True,
        activation_passed=activation.get("passed") is True,
        sft_training_completed=sft_attempt.get("status") == "completed",
        dpo_training_completed=dpo_completed,
        sft_gate=sft_gate,
        metrics=metrics,
        comparison_vs_base=comparisons[f"{candidate}_vs_base"],
    )
    decision.update({
        "model_call_count": _attempted_call_count(), "max_model_call_budget": MAX_MODEL_CALL_BUDGET,
        "expected_model_call_count": PHASE110_HOLDOUT_COUNT * (4 if dpo_completed else 3),
        "adapter_activation_summary": activation.get("summary"), "sft_gate": sft_gate,
        "sft_training_attempt": {"status": sft_attempt.get("status"), "requested_steps": sft_attempt.get("requested_steps"), "completed_steps": sft_attempt.get("completed_steps"), "adapter_validation": sft_attempt.get("adapter_validation")},
        "external_provider_used": False, "paid_api_used": False, "push_performed": False,
        "deployment_performed": False, "promotion_performed": False, "raw_private_text_committed": False,
    })
    _write_json(EVIDENCE_ROOT / "phase110-final-decision.json", decision)
    runbook = """# Phase110 Runbook

```bash
.venv/bin/python tools/phase110_task_grounded_sft_dpo_causal_proof.py prepare --clean
.venv/bin/python tools/phase110_task_grounded_sft_dpo_causal_proof.py diagnose-adapter
.venv/bin/python tools/phase110_task_grounded_sft_dpo_causal_proof.py train-sft --steps 1 --clean
.venv/bin/python tools/phase110_task_grounded_sft_dpo_causal_proof.py train-sft --steps 12 --clean
.venv/bin/python tools/phase110_task_grounded_sft_dpo_causal_proof.py train-sft --steps 30 --clean
.venv/bin/python tools/phase110_task_grounded_sft_dpo_causal_proof.py eval --variant base
.venv/bin/python tools/phase110_task_grounded_sft_dpo_causal_proof.py eval --variant phase109_dpo
.venv/bin/python tools/phase110_task_grounded_sft_dpo_causal_proof.py eval --variant phase110_sft
.venv/bin/python tools/phase110_task_grounded_sft_dpo_causal_proof.py analyze-sft
# Run DPO 1/12/30 and its eval only when analyze-sft exits 0.
.venv/bin/python tools/phase110_task_grounded_sft_dpo_causal_proof.py decide
.venv/bin/python tools/phase110_task_grounded_sft_dpo_causal_proof.py validate
```

All data is `simulated_usage` derived only from Phase31/32 aggregate signals. The 42-session holdout is fresh and excluded from training. Raw generations stay under `/private/tmp`. No external provider, paid API, push, deployment, automatic retraining, or automatic promotion is permitted.
"""
    (EVIDENCE_ROOT / "phase110-runbook.md").write_text(runbook, encoding="utf-8")
    (EVIDENCE_ROOT / "phase110-final-decision.md").write_text(
        "\n".join(["# Phase110 Final Decision", "", f"- Status: `{decision['status']}`", f"- Recommendation: `{decision['recommendation']}`", f"- Evaluated candidate: `{decision['evaluated_candidate']}`", f"- Experiment gate passed: `{str(decision['experiment_gate_passed']).lower()}`", "- Product gate qualified: `false`", "- Automatic promotion allowed: `false`", f"- Local model calls: `{decision['model_call_count']}/{MAX_MODEL_CALL_BUDGET}`", "- Evidence class: `simulated_usage` and `historical_signal_derived`, not actual user feedback."]) + "\n",
        encoding="utf-8",
    )
    _write_json(EVIDENCE_ROOT / "evidence_manifest.json", _build_manifest())
    print(json.dumps({key: decision[key] for key in ("status", "recommendation", "evaluated_candidate", "experiment_gate_passed", "model_call_count", "failed_checks")}, ensure_ascii=False, indent=2))
    return 0


def _validate() -> int:
    decision = _read_json(EVIDENCE_ROOT / "phase110-final-decision.json")
    manifest = _read_json(EVIDENCE_ROOT / "evidence_manifest.json")
    expected = {row["path"]: row["sha256"] for row in manifest["files"]}
    excluded = {EVIDENCE_ROOT / "evidence_manifest.json", EVIDENCE_ROOT / "validation_summary.json"}
    current = {str(path.relative_to(REPO_ROOT)): _sha256(path) for path in sorted(EVIDENCE_ROOT.rglob("*")) if path.is_file() and path not in excluded}
    ledger = _read_jsonl(CALL_LEDGER)
    attempted = [row for row in ledger if row.get("event") == "attempted"]
    completed = [row for row in ledger if row.get("event") == "completed"]
    expected_calls = int(decision["expected_model_call_count"])
    evidence_text = "\n".join(path.read_text(encoding="utf-8", errors="replace") for path in EVIDENCE_ROOT.rglob("*") if path.is_file())
    checks = {
        "manifest_unchanged": expected == current,
        "source_freeze_unchanged": _source_hashes() == _read_json(EVIDENCE_ROOT / "pre_experiment_freeze.json").get("source_sha256"),
        "exact_expected_attempted_calls": len(attempted) == expected_calls,
        "exact_expected_completed_calls": len(completed) == expected_calls,
        "no_failed_calls": not any(row.get("event") == "failed" for row in ledger),
        "no_duplicate_call_ids": len({row["call_id"] for row in attempted}) == len(attempted),
        "product_gate_false": decision.get("product_gate_qualified") is False,
        "automatic_promotion_false": decision.get("automatic_promotion_allowed") is False,
        "actual_feedback_zero": decision.get("actual_user_feedback_count") == 0,
        "no_external_provider": decision.get("external_provider_used") is False,
        "no_paid_api": decision.get("paid_api_used") is False,
        "no_push_deploy_promote": not any(decision.get(key) for key in ("push_performed", "deployment_performed", "promotion_performed")),
        "private_transcripts_not_committed": decision.get("raw_private_text_committed") is False,
        "no_raw_generated_output_in_evidence": '"output":' not in evidence_text,
    }
    validation = {"kind": "phase110_validation_summary", "passed": all(checks.values()), "checks": checks, "validated_at": _utcnow()}
    _write_json(EVIDENCE_ROOT / "validation_summary.json", validation)
    print(json.dumps(validation, ensure_ascii=False, indent=2))
    return 0 if validation["passed"] else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--clean", action="store_true")
    sub.add_parser("diagnose-adapter")
    sft = sub.add_parser("train-sft")
    sft.add_argument("--steps", type=int, choices=tuple(SFT_RUNTIME["steps"]), required=True)
    sft.add_argument("--clean", action="store_true")
    evaluate = sub.add_parser("eval")
    evaluate.add_argument("--variant", choices=PHASE110_FINAL_VARIANTS, required=True)
    sub.add_parser("analyze-sft")
    dpo = sub.add_parser("train-dpo")
    dpo.add_argument("--steps", type=int, choices=tuple(DPO_RUNTIME["steps"]), required=True)
    dpo.add_argument("--clean", action="store_true")
    sub.add_parser("decide")
    sub.add_parser("validate")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "prepare":
        return _prepare(args.clean)
    if args.command == "diagnose-adapter":
        return _diagnose_adapter()
    if args.command == "train-sft":
        return _train_sft(args.steps, args.clean)
    if args.command == "eval":
        return _evaluate(args.variant)
    if args.command == "analyze-sft":
        return _analyze_sft()
    if args.command == "train-dpo":
        return _train_dpo(args.steps, args.clean)
    if args.command == "decide":
        return _decide()
    if args.command == "validate":
        return _validate()
    raise SystemExit(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
