#!/usr/bin/env python3
"""Run the bounded local Phase109 personal engineering-copilot proof."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata as importlib_metadata
import json
import math
import os
from pathlib import Path
import resource
import shutil
import sys
import time
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = REPO_ROOT / "pfe-core"
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

from pfe_core.adapter_store.quality import validate_adapter_artifact
from pfe_core.phase99_qwen3_native_generation_boundary import render_qwen3_no_think_prompt
from pfe_core.phase109_personal_engineering_copilot import (
    PHASE109_MODEL_CALL_BUDGET,
    PHASE109_PERSONAL_CONTRACT,
    PHASE109_SESSION_COUNT,
    PHASE109_VARIANTS,
    aggregate_phase109_scores,
    audit_phase109_data,
    build_phase109_decision,
    build_phase109_holdout,
    build_phase109_training_pairs,
    compare_phase109_variants,
    score_phase109_output,
    stable_hash,
)
from pfe_core.trainer.executors import execute_dpo_training


EVIDENCE_ROOT = REPO_ROOT / "docs/demo/phase109-personal-engineering-copilot-benefit-proof"
PREPARATION_ROOT = EVIDENCE_ROOT / "evidence-preparation"
TRAINING_ROOT = EVIDENCE_ROOT / "evidence-training"
EVAL_ROOT = EVIDENCE_ROOT / "evidence-eval"
FAILURE_ROOT = EVIDENCE_ROOT / "evidence-failures"
PRIVATE_ROOT = Path("/private/tmp/pfe-phase109-simulated-review")
MODEL_PATH = REPO_ROOT / "models/Qwen3-4B"
PHASE106_ADAPTER = REPO_ROOT / "trainer_job_outputs/phase106-qwen3-4b-sft-30step/peft_lora"
PHASE107_ADAPTER = REPO_ROOT / "trainer_job_outputs/phase107-qwen3-4b-token-faithful-dpo/30step/dpo_adapter"
PHASE109_OUTPUT_ROOT = REPO_ROOT / "trainer_job_outputs/phase109-personal-engineering-dpo"
PHASE108_ROOT = REPO_ROOT / "docs/demo/phase108-runtime-adapter-causal-value-proof"
PHASE31_ROOT = REPO_ROOT / "docs/demo/phase31-obsidian-agent-conversation-signal-mining"
PHASE32_ROOT = REPO_ROOT / "docs/demo/phase32-personal-agent-preference-training-loop"
CALL_LEDGER = EVAL_ROOT / "call_ledger.jsonl"
MODEL_CALL_BUDGET = PHASE109_MODEL_CALL_BUDGET
EVAL_VARIANTS = PHASE109_VARIANTS

DPO_RUNTIME = {
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
GENERATION_PROTOCOL = {
    "input_max_length": 3072,
    "max_new_tokens": 160,
    "do_sample": False,
    "repetition_penalty": 1.12,
    "no_repeat_ngram_size": 4,
    "enable_thinking": False,
    "post_hoc_truncation_allowed": False,
    "automatic_retry_count": 0,
}


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), ensure_ascii=False, sort_keys=True) + "\n")


def _safe_clean(path: Path, parent: Path) -> None:
    resolved = path.resolve()
    if resolved == parent.resolve() or parent.resolve() not in resolved.parents:
        raise RuntimeError(f"refusing unsafe clean: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def _source_hashes() -> dict[str, str]:
    paths = {
        "phase109_core": CORE_ROOT / "pfe_core/phase109_personal_engineering_copilot.py",
        "phase109_driver": REPO_ROOT / "tools/phase109_personal_engineering_copilot.py",
        "phase109_test": REPO_ROOT / "tests/test_phase109_personal_engineering_copilot.py",
        "phase109_driver_test": REPO_ROOT / "tests/test_phase109_driver_safety.py",
        "phase32_taxonomy": PHASE32_ROOT / "evidence-review/taxonomy.json",
        "phase32_review_summary": PHASE32_ROOT / "evidence-review/review_summary.json",
        "phase108_decision": PHASE108_ROOT / "phase108-final-decision.json",
    }
    return {name: _sha256(path) for name, path in paths.items()}


def _artifact_manifest(path: Path) -> dict[str, Any]:
    artifact = path / "adapter_model.safetensors"
    validation = validate_adapter_artifact(
        path,
        {"artifact_name": "adapter_model.safetensors", "artifact_format": "peft_lora"},
    ) if path.is_dir() else {"valid": False, "reason": "adapter_directory_missing"}
    return {
        **validation,
        "artifact_dir": str(path),
        "artifact_sha256": _sha256(artifact) if artifact.is_file() else None,
    }


def _model_manifest() -> dict[str, Any]:
    files = [path for path in MODEL_PATH.rglob("*") if path.is_file()] if MODEL_PATH.is_dir() else []
    required = [MODEL_PATH / "config.json", MODEL_PATH / "tokenizer_config.json"]
    return {
        "path": str(MODEL_PATH),
        "exists": MODEL_PATH.is_dir(),
        "file_count": len(files),
        "total_bytes": sum(path.stat().st_size for path in files),
        "required_files_present": all(path.is_file() for path in required),
        "config_sha256": _sha256(MODEL_PATH / "config.json") if (MODEL_PATH / "config.json").is_file() else None,
    }


def _signal_summary() -> dict[str, Any]:
    routing = _read_json(PHASE31_ROOT / "evidence-signals/signal_routing_report.json")
    review = _read_json(PHASE32_ROOT / "evidence-review/review_summary.json")
    return {
        "kind": "phase109_aggregate_personal_signal_basis",
        "phase31_signal_count": routing.get("signal_count"),
        "phase31_signal_type_counts": routing.get("signal_type_counts"),
        "phase32_approved_for_training_count": review.get("approved_for_training_count"),
        "phase32_quarantined_count": review.get("quarantined_count"),
        "raw_private_text_read_by_phase109": False,
        "aggregate_evidence_only": True,
        "actual_user_feedback": False,
    }


def _render_training_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), local_files_only=True)
    rendered: list[dict[str, Any]] = []
    prompt_lengths: list[int] = []
    total_lengths: list[int] = []
    for row in rows:
        messages = [{"role": "system", "content": PHASE109_PERSONAL_CONTRACT}, *[dict(message) for message in row.get("prompt_messages") or []]]
        prompt = render_qwen3_no_think_prompt(tokenizer, messages)
        item = {**row, "prompt": prompt, "instruction": prompt}
        rendered.append(item)
        prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
        prompt_lengths.append(prompt_tokens)
        total_lengths.extend(
            prompt_tokens + len(tokenizer.encode(str(item[key]), add_special_tokens=False))
            for key in ("chosen", "rejected")
        )
    diagnostic = {
        "kind": "phase109_tokenizer_diagnostic",
        "prompt_count": len(rendered),
        "max_prompt_tokens": max(prompt_lengths, default=0),
        "max_total_tokens": max(total_lengths, default=0),
        "configured_max_prompt_length": DPO_RUNTIME["max_prompt_length"],
        "configured_max_length": DPO_RUNTIME["max_length"],
        "all_prompts_within_limit": max(prompt_lengths, default=0) <= DPO_RUNTIME["max_prompt_length"],
        "all_pairs_within_limit": max(total_lengths, default=0) <= DPO_RUNTIME["max_length"],
        "all_prompts_no_think_aligned": all(str(row.get("prompt") or "").endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n") for row in rendered),
    }
    diagnostic["passed"] = all(value is True for key, value in diagnostic.items() if key.startswith("all_"))
    return rendered, diagnostic


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
        "phase109": {
            "steps": steps,
            "lineage": "qwen3_4b_base_to_focused_personal_dpo",
            "simulated_usage": True,
            "actual_user_feedback": False,
            "historical_signal_derived": True,
            "eligible_for_automatic_promotion": False,
            "product_gate_qualified": False,
        },
    }


def _attempted_call_count() -> int:
    return sum(row.get("event") == "attempted" for row in _read_jsonl(CALL_LEDGER))


def _reserve_call(variant: str, session_id: str) -> str:
    call_id = f"phase109-{variant}-{session_id}"
    rows = _read_jsonl(CALL_LEDGER)
    if any(row.get("event") == "attempted" and row.get("call_id") == call_id for row in rows):
        raise RuntimeError(f"duplicate Phase109 call id: {call_id}")
    if sum(row.get("event") == "attempted" for row in rows) >= MODEL_CALL_BUDGET:
        raise RuntimeError(f"Phase109 model call budget exhausted: {MODEL_CALL_BUDGET}")
    _append_jsonl(CALL_LEDGER, {
        "event": "attempted", "call_id": call_id, "variant": variant,
        "session_id": session_id, "provider": "local_qwen3_4b", "created_at": _utcnow(),
    })
    return call_id


def _clean_prepare() -> None:
    if _attempted_call_count():
        raise RuntimeError("Phase109 clean refused because the append-only call ledger already contains attempts")
    if EVIDENCE_ROOT.exists():
        _safe_clean(EVIDENCE_ROOT, EVIDENCE_ROOT.parent)
    if PRIVATE_ROOT.exists():
        _safe_clean(PRIVATE_ROOT, PRIVATE_ROOT.parent)


def _prepare(clean: bool) -> int:
    if clean:
        _clean_prepare()
    for root in (PREPARATION_ROOT, TRAINING_ROOT, EVAL_ROOT, FAILURE_ROOT, PRIVATE_ROOT):
        root.mkdir(parents=True, exist_ok=True)
    raw_pairs = build_phase109_training_pairs()
    holdout = build_phase109_holdout()
    integrity = audit_phase109_data(raw_pairs, holdout)
    rendered, tokenizer_diagnostic = _render_training_rows(raw_pairs)
    phase108 = _read_json(PHASE108_ROOT / "phase108-final-decision.json")
    source_hashes = _source_hashes()
    specs = {
        str(steps): _job_spec(rendered, steps=steps, output_dir=PHASE109_OUTPUT_ROOT / f"{steps}step")
        for steps in (1, 12, 30)
    }
    dry_runs = {steps: execute_dpo_training(job_spec=spec, dry_run=True) for steps, spec in specs.items()}
    checks = {
        "phase108_remains_archive": str(phase108.get("status") or "").startswith("archive_"),
        "phase108_product_gate_false": phase108.get("product_gate_qualified") is False,
        "data_integrity_passed": integrity.get("passed") is True,
        "tokenizer_diagnostic_passed": tokenizer_diagnostic.get("passed") is True,
        "model_available": _model_manifest().get("required_files_present") is True,
        "phase106_adapter_available": _artifact_manifest(PHASE106_ADAPTER).get("valid") is True,
        "phase107_adapter_available": _artifact_manifest(PHASE107_ADAPTER).get("valid") is True,
        "all_dpo_dry_runs_ready": all(dict(row).get("status") in {"prepared", "ready"} for row in dry_runs.values()),
        "exact_call_budget": PHASE109_SESSION_COUNT * len(PHASE109_VARIANTS) == MODEL_CALL_BUDGET,
        "call_ledger_empty": _attempted_call_count() == 0,
        "private_root_outside_repo": REPO_ROOT.resolve() not in PRIVATE_ROOT.resolve().parents,
    }
    freeze = {
        "kind": "phase109_pre_experiment_freeze",
        "passed": all(checks.values()),
        "checks": checks,
        "source_sha256": source_hashes,
        "training_pair_manifest_sha256": stable_hash(rendered),
        "holdout_manifest_sha256": stable_hash(holdout),
        "personal_contract_sha256": stable_hash(PHASE109_PERSONAL_CONTRACT),
        "job_spec_sha256": {steps: stable_hash(spec) for steps, spec in specs.items()},
        "model_manifest": _model_manifest(),
        "historical_adapter_manifests": {
            "phase106_sft": _artifact_manifest(PHASE106_ADAPTER),
            "phase107_dpo": _artifact_manifest(PHASE107_ADAPTER),
        },
        "generation_protocol": GENERATION_PROTOCOL,
        "model_call_budget": MODEL_CALL_BUDGET,
        "eval_variants": list(PHASE109_VARIANTS),
        "external_provider_allowed": False,
        "automatic_training_allowed": False,
        "automatic_promotion_allowed": False,
        "actual_user_feedback": False,
        "private_transcripts_committed": False,
        "frozen_at": _utcnow(),
    }
    _write_json(PREPARATION_ROOT / "aggregate_signal_basis.json", _signal_summary())
    _write_json(PREPARATION_ROOT / "holdout.json", holdout)
    _write_json(PREPARATION_ROOT / "holdout_integrity_check.json", integrity)
    _write_json(PREPARATION_ROOT / "tokenizer_diagnostic.json", tokenizer_diagnostic)
    _write_jsonl(PREPARATION_ROOT / "selected_dpo_pairs.jsonl", rendered)
    for steps, spec in specs.items():
        _write_json(PREPARATION_ROOT / f"dpo_job_spec_{steps}step.json", spec)
        _write_json(PREPARATION_ROOT / f"dpo_dry_run_{steps}step.json", dry_runs[steps])
    _write_json(EVIDENCE_ROOT / "pre_experiment_freeze.json", freeze)
    print(json.dumps({"status": "prepared" if freeze["passed"] else "blocked", "checks": checks}, ensure_ascii=False, indent=2))
    return 0 if freeze["passed"] else 2


def _finite_training_attempt(attempt: Mapping[str, Any]) -> bool:
    result = dict(attempt.get("result") or {})
    real = dict(result.get("real_execution") or {})
    loss = result.get("train_loss")
    if loss is None or not math.isfinite(float(loss)):
        return False
    for row in real.get("loss_history") or []:
        for key, value in dict(row).items():
            if isinstance(value, (int, float)) and key not in {"epoch", "step"} and not math.isfinite(float(value)):
                return False
    return True


def _training_freeze_check(steps: int) -> dict[str, Any]:
    freeze = _read_json(EVIDENCE_ROOT / "pre_experiment_freeze.json")
    rows = _read_jsonl(PREPARATION_ROOT / "selected_dpo_pairs.jsonl")
    spec = _job_spec(rows, steps=steps, output_dir=PHASE109_OUTPUT_ROOT / f"{steps}step")
    checks = {
        "pre_experiment_freeze_passed": freeze.get("passed") is True,
        "source_files_unchanged": _source_hashes() == freeze.get("source_sha256"),
        "training_pairs_unchanged": stable_hash(rows) == freeze.get("training_pair_manifest_sha256"),
        "job_spec_unchanged": stable_hash(spec) == dict(freeze.get("job_spec_sha256") or {}).get(str(steps)),
        "step_frozen": steps in (1, 12, 30),
        "no_existing_attempt": not (TRAINING_ROOT / f"{steps}step/training_attempt.json").exists(),
    }
    if steps == 12:
        prior = _read_json(TRAINING_ROOT / "1step/training_attempt.json") if (TRAINING_ROOT / "1step/training_attempt.json").is_file() else {}
        checks["one_step_completed"] = prior.get("status") == "completed"
        checks["one_step_finite"] = _finite_training_attempt(prior)
        checks["one_step_adapter_valid"] = dict(prior.get("adapter_validation") or {}).get("valid") is True
    if steps == 30:
        prior = _read_json(TRAINING_ROOT / "12step/training_attempt.json") if (TRAINING_ROOT / "12step/training_attempt.json").is_file() else {}
        checks["twelve_step_completed"] = prior.get("status") == "completed"
        checks["twelve_step_finite"] = _finite_training_attempt(prior)
        checks["twelve_step_adapter_valid"] = dict(prior.get("adapter_validation") or {}).get("valid") is True
    return {"kind": "phase109_training_freeze_check", "steps": steps, "passed": all(checks.values()), "checks": checks}


def _train(steps: int, clean: bool) -> int:
    if steps not in (1, 12, 30):
        raise SystemExit("Phase109 permits only 1, 12, or 30 training steps")
    evidence_dir = TRAINING_ROOT / f"{steps}step"
    output_dir = PHASE109_OUTPUT_ROOT / f"{steps}step"
    if clean:
        if evidence_dir.exists():
            _safe_clean(evidence_dir, TRAINING_ROOT)
        if output_dir.exists():
            _safe_clean(output_dir, PHASE109_OUTPUT_ROOT)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_jsonl(PREPARATION_ROOT / "selected_dpo_pairs.jsonl")
    freeze = _training_freeze_check(steps)
    spec = _job_spec(rows, steps=steps, output_dir=output_dir)
    _write_json(evidence_dir / "freeze_check.json", freeze)
    _write_json(evidence_dir / "dpo_job_spec.json", spec)
    if not freeze["passed"]:
        attempt = {
            "kind": "phase109_training_attempt", "status": "blocked", "requested_steps": steps,
            "reason": "freeze_check_failed", "freeze_check": freeze,
            "product_gate_qualified": False, "automatic_promotion_allowed": False,
        }
        _write_json(evidence_dir / "training_attempt.json", attempt)
        return 2
    started = time.perf_counter()
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = execute_dpo_training(job_spec={**spec, "dry_run": False}, dry_run=False)
    real = dict(result.get("real_execution") or {})
    artifact_dir_value = str(real.get("artifact_dir") or "").strip()
    artifact_dir = Path(artifact_dir_value) if artifact_dir_value else None
    validation = _artifact_manifest(artifact_dir) if artifact_dir and artifact_dir.is_dir() else {"valid": False, "reason": "artifact_dir_missing"}
    validation.update({
        "requested_steps": steps,
        "completed_steps": int(real.get("steps") or 0),
        "lineage": "qwen3_4b_base_to_focused_personal_dpo",
    })
    completed = (
        result.get("status") == "completed"
        and real.get("success") is True
        and int(real.get("steps") or 0) == steps
        and real.get("parameters_updated") is True
        and validation.get("valid") is True
    )
    attempt = {
        "kind": "phase109_training_attempt",
        "status": "completed" if completed else "failed",
        "real_training": completed,
        "requested_steps": steps,
        "completed_steps": int(real.get("steps") or 0),
        "duration_seconds": round(time.perf_counter() - started, 4),
        "resource_usage": {
            "ru_maxrss_before": rss_before,
            "ru_maxrss_after": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "ru_maxrss_unit": "bytes_on_macos",
        },
        "result": result,
        "adapter_validation": validation,
        "simulated_usage": True,
        "actual_user_feedback": False,
        "historical_signal_derived": True,
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
        "error": result.get("error"),
    })
    _write_json(evidence_dir / "adapter_validation.json", validation)
    print(json.dumps({
        "status": attempt["status"], "steps": steps, "duration_seconds": attempt["duration_seconds"],
        "train_loss": result.get("train_loss"), "adapter_valid": validation.get("valid"),
    }, ensure_ascii=False, indent=2))
    return 0 if completed and _finite_training_attempt(attempt) else 1


def _phase109_adapter() -> Path:
    attempt = _read_json(TRAINING_ROOT / "30step/training_attempt.json")
    path = Path(str(dict(attempt.get("adapter_validation") or {}).get("artifact_dir") or ""))
    if attempt.get("status") != "completed" or not path.is_dir():
        raise RuntimeError("Phase109 30-step adapter is unavailable")
    return path


def _load_model(variant: str) -> tuple[Any, Any, Any, str, str]:
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
        str(MODEL_PATH), local_files_only=True, low_cpu_mem_usage=True, dtype=dtype,
    )
    load_plan = ["qwen3_4b_base"]
    if variant == "phase107_dpo":
        model = PeftModel.from_pretrained(model, str(PHASE106_ADAPTER), local_files_only=True)
        model = model.merge_and_unload()
        model = PeftModel.from_pretrained(model, str(PHASE107_ADAPTER), local_files_only=True)
        load_plan.extend(["phase106_sft_merge", "phase107_dpo"])
    elif variant == "phase109_personal_dpo":
        model = PeftModel.from_pretrained(model, str(_phase109_adapter()), local_files_only=True)
        load_plan.append("phase109_personal_dpo")
    elif variant != "base":
        raise RuntimeError(f"unsupported Phase109 variant: {variant}")
    model.to(device)
    model.eval()
    return torch, tokenizer, model, device, "+".join(load_plan)


def _generate_once(torch: Any, tokenizer: Any, model: Any, device: str, messages: list[dict[str, str]]) -> str:
    prompt = render_qwen3_no_think_prompt(tokenizer, messages)
    encoded = tokenizer(
        prompt, return_tensors="pt", truncation=True,
        max_length=GENERATION_PROTOCOL["input_max_length"], add_special_tokens=False,
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    eos_ids = [tokenizer.eos_token_id]
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end, int) and im_end >= 0 and im_end not in eos_ids:
        eos_ids.append(im_end)
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            max_new_tokens=GENERATION_PROTOCOL["max_new_tokens"],
            do_sample=GENERATION_PROTOCOL["do_sample"],
            repetition_penalty=GENERATION_PROTOCOL["repetition_penalty"],
            no_repeat_ngram_size=GENERATION_PROTOCOL["no_repeat_ngram_size"],
            eos_token_id=eos_ids,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )
    new_tokens = generated[0, encoded["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def _eval_freeze_check(variant: str) -> dict[str, Any]:
    freeze = _read_json(EVIDENCE_ROOT / "pre_experiment_freeze.json")
    checks = {
        "variant_allowed": variant in PHASE109_VARIANTS,
        "pre_experiment_freeze_passed": freeze.get("passed") is True,
        "source_files_unchanged": _source_hashes() == freeze.get("source_sha256"),
        "holdout_unchanged": stable_hash(_read_json(PREPARATION_ROOT / "holdout.json")) == freeze.get("holdout_manifest_sha256"),
        "within_call_budget": _attempted_call_count() + PHASE109_SESSION_COUNT <= MODEL_CALL_BUDGET,
        "variant_not_previously_run": not (EVAL_ROOT / variant / "metrics.json").exists(),
    }
    if variant == "phase109_personal_dpo":
        checks["phase109_adapter_valid"] = _artifact_manifest(_phase109_adapter()).get("valid") is True
    return {"kind": "phase109_eval_freeze_check", "variant": variant, "passed": all(checks.values()), "checks": checks}


def _evaluate(variant: str) -> int:
    if variant not in PHASE109_VARIANTS:
        raise SystemExit(f"unsupported Phase109 variant: {variant}")
    evidence_dir = EVAL_ROOT / variant
    evidence_dir.mkdir(parents=True, exist_ok=True)
    freeze = _eval_freeze_check(variant)
    _write_json(evidence_dir / "freeze_check.json", freeze)
    if not freeze["passed"]:
        return 2
    torch, tokenizer, model, device, load_plan = _load_model(variant)
    sessions = list(_read_json(PREPARATION_ROOT / "holdout.json").get("sessions") or [])
    scores: list[dict[str, Any]] = []
    private_path = PRIVATE_ROOT / f"{variant}.jsonl"
    started = time.perf_counter()
    for index, session in enumerate(sessions, start=1):
        call_id = _reserve_call(variant, str(session["session_id"]))
        messages = [{"role": "system", "content": PHASE109_PERSONAL_CONTRACT}]
        messages.extend({"role": str(row["role"]), "content": str(row["content"])} for row in session.get("messages") or [])
        try:
            output = _generate_once(torch, tokenizer, model, device, messages)
        except Exception as exc:
            _append_jsonl(CALL_LEDGER, {
                "event": "failed", "call_id": call_id, "variant": variant,
                "session_id": session["session_id"], "error_type": type(exc).__name__, "created_at": _utcnow(),
            })
            raise
        score = score_phase109_output(output, session)
        scores.append(score)
        _append_jsonl(private_path, {
            "variant": variant, "session_id": session["session_id"], "messages": messages,
            "output": output, "score": score, "usage_class": "simulated_usage",
            "actual_user_feedback": False, "raw_private_source_text": False,
        })
        _append_jsonl(CALL_LEDGER, {
            "event": "completed", "call_id": call_id, "variant": variant,
            "session_id": session["session_id"], "output_sha256": score["output_sha256"], "created_at": _utcnow(),
        })
        print(f"[{variant}] {index:02d}/{len(sessions)} {session['session_id']} score={score['overall_score']:.3f}", flush=True)
    metrics = aggregate_phase109_scores(scores)
    metrics.update({
        "variant": variant,
        "model_call_count": len(sessions),
        "duration_seconds": round(time.perf_counter() - started, 4),
        "device": device,
        "load_plan": load_plan,
        "generation_protocol": GENERATION_PROTOCOL,
        "private_cache_outside_repo": True,
        "private_transcripts_committed": False,
        "post_hoc_truncation_used": False,
        "external_provider_used": False,
    })
    _write_json(evidence_dir / "metrics.json", metrics)
    _write_jsonl(evidence_dir / "structural_sessions.jsonl", scores)
    print(json.dumps({key: metrics[key] for key in ("variant", "session_count", "accepted_rate", "overall_score", "duration_seconds")}, ensure_ascii=False, indent=2))
    del model
    gc.collect()
    if device == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()
    return 0


def _analyze() -> int:
    metrics = {variant: _read_json(EVAL_ROOT / variant / "metrics.json") for variant in PHASE109_VARIANTS}
    comparisons = {
        "phase109_vs_base": compare_phase109_variants(metrics["phase109_personal_dpo"], metrics["base"], seed=109),
        "phase109_vs_phase107": compare_phase109_variants(metrics["phase109_personal_dpo"], metrics["phase107_dpo"], seed=110),
        "phase107_vs_base": compare_phase109_variants(metrics["phase107_dpo"], metrics["base"], seed=111),
    }
    summary = {
        "kind": "phase109_comparison_summary",
        "metrics": metrics,
        "comparisons": comparisons,
        "raw_outputs_committed": False,
        "actual_user_feedback": False,
    }
    _write_json(EVAL_ROOT / "comparison_summary.json", summary)
    failure_counts = Counter(
        failed
        for row in metrics["phase109_personal_dpo"].get("details") or []
        for failed in (
            "latest_intent" if not row.get("latest_intent_followed") else None,
            "correction" if not row.get("correction_responsive") else None,
            "concise" if not row.get("concise_status") else None,
            "false_completion" if row.get("false_completion") else None,
            "private_canary_leak" if row.get("private_canary_leak") else None,
        )
        if failed
    )
    _write_json(FAILURE_ROOT / "failure_diagnosis.json", {
        "kind": "phase109_failure_diagnosis",
        "candidate_failure_counts": dict(sorted(failure_counts.items())),
        "dominant_failure": failure_counts.most_common(1)[0][0] if failure_counts else None,
        "targeted_retraining_allowed": False,
        "reason": "Phase109 is a frozen one-candidate proof; failures require a new phase.",
    })
    print(json.dumps({name: {key: value for key, value in row.items() if key in {"pair_count", "candidate_wins", "benchmark_wins", "mean_delta", "ci_low", "ci_high"}} for name, row in comparisons.items()}, ensure_ascii=False, indent=2))
    return 0


def _build_manifest() -> dict[str, Any]:
    excluded = {EVIDENCE_ROOT / "evidence_manifest.json", EVIDENCE_ROOT / "validation_summary.json"}
    files = [
        {"path": str(path.relative_to(REPO_ROOT)), "sha256": _sha256(path), "bytes": path.stat().st_size}
        for path in sorted(EVIDENCE_ROOT.rglob("*")) if path.is_file() and path not in excluded
    ]
    return {"kind": "phase109_evidence_manifest", "file_count": len(files), "files": files}


def _decide() -> int:
    phase108 = _read_json(PHASE108_ROOT / "phase108-final-decision.json")
    training = _read_json(TRAINING_ROOT / "30step/training_attempt.json")
    integrity = _read_json(PREPARATION_ROOT / "holdout_integrity_check.json")
    summary = _read_json(EVAL_ROOT / "comparison_summary.json")
    metrics = dict(summary.get("metrics") or {})
    comparisons = dict(summary.get("comparisons") or {})
    decision = build_phase109_decision(
        training_completed=training.get("status") == "completed" and _finite_training_attempt(training),
        data_integrity_passed=integrity.get("passed") is True,
        phase108_remains_archive=str(phase108.get("status") or "").startswith("archive_"),
        metrics=metrics,
        comparison_vs_base=comparisons["phase109_vs_base"],
        comparison_vs_phase107=comparisons["phase109_vs_phase107"],
    )
    decision.update({
        "model_call_count": _attempted_call_count(),
        "model_call_budget": MODEL_CALL_BUDGET,
        "training_attempt": {
            "status": training.get("status"),
            "requested_steps": training.get("requested_steps"),
            "completed_steps": training.get("completed_steps"),
            "train_loss": dict(training.get("result") or {}).get("train_loss"),
            "adapter_validation": training.get("adapter_validation"),
        },
        "phase108_lifecycle": phase108.get("status"),
        "external_provider_used": False,
        "paid_api_used": False,
        "push_performed": False,
        "deployment_performed": False,
    })
    _write_json(EVIDENCE_ROOT / "phase109-final-decision.json", decision)
    runbook = """# Phase109 Runbook

```bash
.venv/bin/python tools/phase109_personal_engineering_copilot.py prepare --clean
.venv/bin/python tools/phase109_personal_engineering_copilot.py train --steps 1 --clean
.venv/bin/python tools/phase109_personal_engineering_copilot.py train --steps 12 --clean
.venv/bin/python tools/phase109_personal_engineering_copilot.py train --steps 30 --clean
.venv/bin/python tools/phase109_personal_engineering_copilot.py eval --variant base
.venv/bin/python tools/phase109_personal_engineering_copilot.py eval --variant phase107_dpo
.venv/bin/python tools/phase109_personal_engineering_copilot.py eval --variant phase109_personal_dpo
.venv/bin/python tools/phase109_personal_engineering_copilot.py analyze
.venv/bin/python tools/phase109_personal_engineering_copilot.py decide
.venv/bin/python tools/phase109_personal_engineering_copilot.py validate
```

The experiment uses 42 historical-signal-derived simulated preference pairs and 35 fresh simulated multi-turn holdout sessions. All 105 model calls are local Qwen3-4B. Raw generated transcripts stay under `/private/tmp`; no external provider, push, deployment, automatic retraining, or automatic promotion is permitted.
"""
    (EVIDENCE_ROOT / "phase109-runbook.md").write_text(runbook, encoding="utf-8")
    lines = [
        "# Phase109 Final Decision", "",
        f"- Status: `{decision['status']}`",
        f"- Recommendation: `{decision['recommendation']}`",
        f"- Experiment gate passed: `{str(decision['experiment_gate_passed']).lower()}`",
        "- Product gate qualified: `false`",
        "- Automatic promotion allowed: `false`",
        f"- Local model calls: `{decision['model_call_count']}/{MODEL_CALL_BUDGET}`",
        "- Evidence class: `simulated_usage` and `historical_signal_derived`, not actual user feedback.",
    ]
    (EVIDENCE_ROOT / "phase109-final-decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_json(EVIDENCE_ROOT / "evidence_manifest.json", _build_manifest())
    print(json.dumps({key: decision[key] for key in ("status", "recommendation", "experiment_gate_passed", "model_call_count", "failed_checks")}, ensure_ascii=False, indent=2))
    return 0


def _validate() -> int:
    decision = _read_json(EVIDENCE_ROOT / "phase109-final-decision.json")
    manifest = _read_json(EVIDENCE_ROOT / "evidence_manifest.json")
    expected = {str(row["path"]): str(row["sha256"]) for row in manifest.get("files") or []}
    excluded = {EVIDENCE_ROOT / "evidence_manifest.json", EVIDENCE_ROOT / "validation_summary.json"}
    current = {
        str(path.relative_to(REPO_ROOT)): _sha256(path)
        for path in sorted(EVIDENCE_ROOT.rglob("*")) if path.is_file() and path not in excluded
    }
    ledger = _read_jsonl(CALL_LEDGER)
    attempted = [row for row in ledger if row.get("event") == "attempted"]
    completed = [row for row in ledger if row.get("event") == "completed"]
    phase108 = _read_json(PHASE108_ROOT / "phase108-final-decision.json")
    evidence_text = "\n".join(path.read_text(encoding="utf-8", errors="replace") for path in EVIDENCE_ROOT.rglob("*") if path.is_file())
    checks = {
        "manifest_unchanged": expected == current,
        "source_freeze_unchanged": _source_hashes() == _read_json(EVIDENCE_ROOT / "pre_experiment_freeze.json").get("source_sha256"),
        "phase108_remains_archive": str(phase108.get("status") or "").startswith("archive_"),
        "exactly_105_attempted_calls": len(attempted) == MODEL_CALL_BUDGET,
        "exactly_105_completed_calls": len(completed) == MODEL_CALL_BUDGET,
        "no_failed_calls": not any(row.get("event") == "failed" for row in ledger),
        "no_duplicate_attempted_call_ids": len({row.get("call_id") for row in attempted}) == len(attempted),
        "product_gate_false": decision.get("product_gate_qualified") is False,
        "automatic_promotion_false": decision.get("automatic_promotion_allowed") is False,
        "actual_feedback_zero": decision.get("actual_user_feedback_count") == 0,
        "private_transcripts_not_committed": decision.get("raw_private_text_committed") is False,
        "no_external_provider": decision.get("external_provider_used") is False,
        "no_raw_output_field_in_evidence": '"output":' not in evidence_text,
        "all_variants_present": all((EVAL_ROOT / variant / "metrics.json").is_file() for variant in PHASE109_VARIANTS),
    }
    validation = {"kind": "phase109_validation_summary", "passed": all(checks.values()), "checks": checks, "validated_at": _utcnow()}
    _write_json(EVIDENCE_ROOT / "validation_summary.json", validation)
    print(json.dumps(validation, ensure_ascii=False, indent=2))
    return 0 if validation["passed"] else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--clean", action="store_true")
    train = sub.add_parser("train")
    train.add_argument("--steps", type=int, choices=(1, 12, 30), required=True)
    train.add_argument("--clean", action="store_true")
    evaluate = sub.add_parser("eval")
    evaluate.add_argument("--variant", choices=PHASE109_VARIANTS, required=True)
    sub.add_parser("analyze")
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
        return _evaluate(args.variant)
    if args.command == "analyze":
        return _analyze()
    if args.command == "decide":
        return _decide()
    if args.command == "validate":
        return _validate()
    raise SystemExit(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
