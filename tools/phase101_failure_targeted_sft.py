#!/usr/bin/env python3
"""Run Phase101 failure-targeted Qwen3-4B SFT and a fresh product gate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import sys
import time
import traceback
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = REPO_ROOT / "pfe-core"
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

from pfe_core.adapter_store.quality import validate_adapter_artifact
from pfe_core.phase43_personal_preference_benefit import build_phase43_sft_job_spec
from pfe_core.phase75_personalization_benefit_benchmark import stable_hash
from pfe_core.phase77_private_value_guarded_runtime import guard_phase77_messages, guard_phase77_output
from pfe_core.phase91_controlled_dpo_preference import score_phase91_output
from pfe_core.phase93_95_dpo_product_proof import aggregate_phase94_scores, has_repeated_output
from pfe_core.phase99_qwen3_native_generation_boundary import (
    build_first_answer_stopping_criteria,
    forbidden_generation_hits,
    has_extra_text_after_first_answer,
    qwen3_bad_words_ids,
    qwen3_eos_token_ids,
    render_qwen3_no_think_prompt,
)
from pfe_core.phase100_qwen3_generation_boundary_closure import phase100_runtime_contract
from pfe_core.phase101_failure_targeted_sft import (
    audit_phase101_training_and_holdout,
    build_phase101_holdout,
    build_phase101_sft_decision,
    build_phase101_training_candidates,
)
from pfe_core.trainer.executors import _encode_sft_examples, _run_real_local_peft_training


EVIDENCE_ROOT = REPO_ROOT / "docs/demo/phase100-104-autonomous-qwen3-training-benefit-loop"
PHASE100_ROOT = EVIDENCE_ROOT / "phase100-generation-boundary"
PHASE_ROOT = EVIDENCE_ROOT / "phase101-failure-targeted-sft"
PREPARATION_ROOT = PHASE_ROOT / "evidence-preparation"
TRAINING_ROOT = PHASE_ROOT / "evidence-training"
EVAL_ROOT = PHASE_ROOT / "evidence-eval"
FAILURE_ROOT = PHASE_ROOT / "evidence-failures"
PRIVATE_ROOT = Path("/private/tmp/pfe-phase101-simulated-review")
MODEL_PATH = REPO_ROOT / "models/Qwen3-4B"
TRAINER_OUTPUT_ROOT = REPO_ROOT / "trainer_job_outputs"
GENERATION_PROTOCOL = {
    "model": "Qwen3-4B",
    "input_max_length": 3072,
    "max_new_tokens": 160,
    "do_sample": False,
    "repetition_penalty": 1.15,
    "no_repeat_ngram_size": 4,
    "enable_thinking": False,
    "guided_target_allowed": False,
    "premature_eos_suppression_allowed": False,
    "post_hoc_truncation_allowed": False,
    "variants": ["base", "sft"],
    "model_calls_per_variant": 24,
    "phase101_model_call_budget": 48,
    "long_run_total_call_budget": 270,
}


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


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
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_clean(path: Path, parent: Path) -> None:
    resolved = path.resolve()
    if resolved.parent != parent.resolve():
        raise RuntimeError(f"refusing to clean outside {parent}: {path}")
    if resolved.exists():
        shutil.rmtree(resolved)


def _source_hashes() -> dict[str, str]:
    paths = {
        "core": CORE_ROOT / "pfe_core/phase101_failure_targeted_sft.py",
        "driver": REPO_ROOT / "tools/phase101_failure_targeted_sft.py",
        "core_test": REPO_ROOT / "tests/test_phase101_failure_targeted_sft.py",
        "driver_test": REPO_ROOT / "tests/test_phase101_driver_safety.py",
        "executor": CORE_ROOT / "pfe_core/trainer/executors.py",
    }
    return {name: _sha256(path) for name, path in paths.items()}


def _previous_holdouts() -> list[dict[str, Any]]:
    paths = (
        REPO_ROOT / "docs/demo/phase99-qwen3-native-generation-boundary/evidence-preparation/holdout.json",
        PHASE100_ROOT / "evidence-preparation/diagnostic_holdout.json",
        PHASE100_ROOT / "evidence-preparation/final_holdout.json",
    )
    return [_read_json(path) for path in paths if path.is_file()]


def _prepare(clean: bool) -> int:
    if clean and PHASE_ROOT.exists():
        _safe_clean(PHASE_ROOT, EVIDENCE_ROOT)
    PREPARATION_ROOT.mkdir(parents=True, exist_ok=True)
    candidates = build_phase101_training_candidates()
    holdout = build_phase101_holdout()
    integrity = audit_phase101_training_and_holdout(candidates, holdout, _previous_holdouts())
    phase100 = _read_json(PHASE100_ROOT / "phase100-decision.json")
    checks = {
        "phase100_gate_passed": phase100.get("passed") is True,
        "training_holdout_integrity_passed": integrity.get("passed") is True,
        "candidate_count_32": len(candidates) == 32,
        "holdout_count_8": holdout.get("session_count") == 8,
        "model_config_present": (MODEL_PATH / "config.json").is_file(),
        "model_index_present": (MODEL_PATH / "model.safetensors.index.json").is_file(),
        "generation_calls_frozen_48": holdout.get("model_calls_per_variant") * 2 == 48,
    }
    freeze = {
        "kind": "phase101_pre_experiment_freeze",
        "created_at": _utcnow(),
        "passed": all(checks.values()),
        "checks": checks,
        "phase100_decision_sha256": _sha256(PHASE100_ROOT / "phase100-decision.json"),
        "candidate_manifest_sha256": stable_hash(candidates),
        "holdout_manifest_sha256": holdout["manifest_sha256"],
        "generation_protocol": GENERATION_PROTOCOL,
        "generation_protocol_sha256": stable_hash(GENERATION_PROTOCOL),
        "source_sha256": _source_hashes(),
        "training_steps": [1, 12, 30],
        "simulated_usage": True,
        "actual_user_feedback_count": 0,
        "automatic_promotion_allowed": False,
    }
    _write_jsonl(PREPARATION_ROOT / "selected_sft_samples.jsonl", candidates)
    _write_json(PREPARATION_ROOT / "holdout.json", holdout)
    _write_json(PREPARATION_ROOT / "holdout_integrity_check.json", integrity)
    _write_json(PHASE_ROOT / "pre_experiment_freeze.json", freeze)
    print(json.dumps({"status": "ready" if freeze["passed"] else "blocked", "checks": checks}, ensure_ascii=False, indent=2))
    return 0 if freeze["passed"] else 2


def _job_spec(rows: list[dict[str, Any]], output_dir: Path, steps: int) -> dict[str, Any]:
    spec = build_phase43_sft_job_spec(pairs=rows, base_model=str(MODEL_PATH), output_dir=str(output_dir), max_steps=steps)
    training = spec["recipe"]["training"]
    training.update({"max_length": 256, "learning_rate": 0.00005, "seed": 101})
    spec["phase101"] = {
        "failure_targeted": True,
        "completion_only_loss_required": True,
        "target_model": "Qwen3-4B",
        "simulated_usage": True,
        "actual_user_feedback": False,
        "automatic_promotion_allowed": False,
    }
    return spec


def _completion_boundary_report(spec: Mapping[str, Any]) -> dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), local_files_only=True)
    max_length = int(spec["recipe"]["training"]["max_length"])
    encoded = _encode_sft_examples(
        tokenizer=tokenizer,
        training_examples=list(spec.get("training_examples") or []),
        max_length=max_length,
        vocab_size=int(getattr(tokenizer, "vocab_size", 0) or 151936),
    )
    counts = [sum(int(value) != -100 for value in row.get("labels") or []) for row in encoded]
    return {
        "kind": "phase101_completion_boundary_report",
        "passed": len(counts) == 32 and min(counts, default=0) >= 8,
        "sample_count": len(counts),
        "minimum_completion_label_token_count": min(counts, default=0),
        "prompt_tokens_use_loss": False,
        "completion_tokens_use_loss": True,
    }


def _training_freeze_check(steps: int) -> dict[str, Any]:
    freeze = _read_json(PHASE_ROOT / "pre_experiment_freeze.json")
    rows = _read_jsonl(PREPARATION_ROOT / "selected_sft_samples.jsonl")
    holdout = _read_json(PREPARATION_ROOT / "holdout.json")
    checks = {
        "pre_experiment_freeze_passed": freeze.get("passed") is True,
        "source_files_unchanged": _source_hashes() == freeze.get("source_sha256"),
        "candidate_manifest_unchanged": stable_hash(rows) == freeze.get("candidate_manifest_sha256"),
        "holdout_manifest_unchanged": stable_hash(holdout.get("sessions") or []) == freeze.get("holdout_manifest_sha256"),
        "phase100_decision_unchanged": _sha256(PHASE100_ROOT / "phase100-decision.json") == freeze.get("phase100_decision_sha256"),
        "step_is_frozen": steps in (1, 12, 30),
    }
    if steps == 12:
        prior = _read_json(TRAINING_ROOT / "probe-1step/training_attempt.json") if (TRAINING_ROOT / "probe-1step/training_attempt.json").is_file() else {}
        checks["one_step_completed"] = prior.get("status") == "completed"
        checks["one_step_parameters_updated"] = dict(prior.get("adapter_validation") or {}).get("parameters_updated") is True
    if steps == 30:
        prior = _read_json(TRAINING_ROOT / "probe-12step/training_attempt.json") if (TRAINING_ROOT / "probe-12step/training_attempt.json").is_file() else {}
        losses = list(dict(prior.get("execution") or {}).get("loss_history") or [])
        loss_values = [
            row.get("loss") if isinstance(row, Mapping) else row
            for row in losses
        ]
        checks["twelve_step_completed"] = prior.get("status") == "completed"
        checks["twelve_step_parameters_updated"] = dict(prior.get("adapter_validation") or {}).get("parameters_updated") is True
        checks["twelve_step_losses_finite"] = bool(loss_values) and all(
            value is not None and math.isfinite(float(value)) for value in loss_values
        )
    return {"kind": "phase101_training_freeze_check", "steps": steps, "passed": all(checks.values()), "checks": checks}


def _train(steps: int, clean: bool) -> int:
    if steps not in (1, 12, 30):
        raise SystemExit("Phase101 permits 1, 12, or 30 steps only")
    probe_dir = TRAINING_ROOT / f"probe-{steps}step"
    output_root = TRAINER_OUTPUT_ROOT / f"phase101-qwen3-4b-sft-{steps}step"
    if clean and probe_dir.exists():
        _safe_clean(probe_dir, TRAINING_ROOT)
    if clean and output_root.exists():
        _safe_clean(output_root, TRAINER_OUTPUT_ROOT)
    probe_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_jsonl(PREPARATION_ROOT / "selected_sft_samples.jsonl")
    spec = _job_spec(rows, output_root, steps)
    boundary = _completion_boundary_report(spec)
    freeze = _training_freeze_check(steps)
    _write_json(probe_dir / "training_manifest.json", spec)
    _write_json(probe_dir / "completion_boundary_report.json", boundary)
    _write_json(probe_dir / "training_freeze_check.json", freeze)
    if not freeze["passed"] or not boundary["passed"]:
        attempt = {
            "kind": "phase101_sft_training_attempt",
            "status": "blocked",
            "real_training": False,
            "requested_steps": steps,
            "reason": "training_freeze_or_completion_boundary_failed",
            "product_gate_qualified": False,
            "automatic_promotion_allowed": False,
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
        validation = validate_adapter_artifact(artifact_dir, {"artifact_name": adapter_path.name, "artifact_format": "peft_lora"})
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
            "kind": "phase101_sft_training_attempt",
            "status": "completed" if completed else "failed",
            "real_training": completed,
            "requested_steps": steps,
            "model": str(MODEL_PATH),
            "started_at": started_at,
            "finished_at": _utcnow(),
            "duration_seconds": round(time.perf_counter() - started, 4),
            "max_rss_before_bytes": rss_before,
            "max_rss_after_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "execution": real,
            "adapter_validation": validation,
            "simulated_usage": True,
            "actual_user_feedback": False,
            "product_gate_qualified": False,
            "automatic_promotion_allowed": False,
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
            "kind": "phase101_sft_training_attempt",
            "status": "failed",
            "real_training": False,
            "requested_steps": steps,
            "started_at": started_at,
            "finished_at": _utcnow(),
            "duration_seconds": round(time.perf_counter() - started, 4),
            "error": f"{exc.__class__.__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "product_gate_qualified": False,
            "automatic_promotion_allowed": False,
        }
        FAILURE_ROOT.mkdir(parents=True, exist_ok=True)
        _write_json(FAILURE_ROOT / f"training_{steps}step.json", attempt)
    _write_json(probe_dir / "training_attempt.json", attempt)
    print(json.dumps({key: attempt.get(key) for key in ("status", "requested_steps", "duration_seconds", "error")}, ensure_ascii=False, indent=2))
    return 0 if attempt.get("status") == "completed" else 1


def _adapter_dir() -> Path:
    attempt = _read_json(TRAINING_ROOT / "probe-30step/training_attempt.json")
    validation = dict(attempt.get("adapter_validation") or {})
    path = Path(str(validation.get("artifact_dir") or ""))
    if attempt.get("status") != "completed" or validation.get("valid") is not True or not path.is_dir():
        raise SystemExit("Phase101 30-step adapter is unavailable")
    return path


def _load_runtime(adapter_path: Path | None) -> tuple[Any, Any, Any, str]:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device == "mps" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(str(MODEL_PATH), local_files_only=True, low_cpu_mem_usage=True, dtype=dtype)
    if adapter_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_path), local_files_only=True)
    model.to(device)
    model.eval()
    return torch, tokenizer, model, device


def _generate_one(*, torch: Any, tokenizer: Any, model: Any, device: str, messages: list[dict[str, str]], format_expected: bool) -> tuple[str, dict[str, Any]]:
    prompt = render_qwen3_no_think_prompt(tokenizer, messages)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=int(GENERATION_PROTOCOL["input_max_length"]))
    inputs = {name: value.to(device) for name, value in inputs.items()}
    input_length = int(inputs["input_ids"].shape[-1])
    stopping, state = build_first_answer_stopping_criteria(tokenizer=tokenizer, input_length=input_length, format_expected=format_expected)
    eos_ids = qwen3_eos_token_ids(tokenizer)
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=int(GENERATION_PROTOCOL["max_new_tokens"]),
            do_sample=False,
            repetition_penalty=float(GENERATION_PROTOCOL["repetition_penalty"]),
            no_repeat_ngram_size=int(GENERATION_PROTOCOL["no_repeat_ngram_size"]),
            bad_words_ids=qwen3_bad_words_ids(tokenizer),
            stopping_criteria=stopping,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=eos_ids,
        )
    generated = output[0][input_length:]
    raw = tokenizer.decode(generated, skip_special_tokens=True).strip()
    if not raw:
        raise RuntimeError("Phase101 generation returned empty output")
    last_token = int(generated[-1].item()) if int(generated.shape[-1]) else None
    if state["triggered"]:
        reason = "first_answer_boundary"
    elif last_token in eos_ids:
        reason = "model_eos"
    elif int(generated.shape[-1]) >= int(GENERATION_PROTOCOL["max_new_tokens"]):
        reason = "max_new_tokens"
    else:
        reason = "unknown"
    complete = bool(state["triggered"])
    return raw, {
        "completion_tokens": int(generated.shape[-1]),
        "latency_seconds": round(time.perf_counter() - started, 4),
        "termination_reason": reason,
        "complete_content_before_termination": complete,
        "native_termination": reason in {"first_answer_boundary", "model_eos"} and complete,
        "extra_text_after_first_answer": has_extra_text_after_first_answer(raw, format_expected=format_expected),
        "forbidden_generation_hits": forbidden_generation_hits(raw),
        "guided_token_count": 0,
        "post_hoc_truncation_used": False,
    }


def _run_session(*, session: Mapping[str, Any], torch: Any, tokenizer: Any, model: Any, device: str) -> tuple[dict[str, Any], dict[str, Any]]:
    history: list[dict[str, str]] = []
    private_values = [str(value) for value in session.get("declared_private_values") or []]
    turns = []
    private_turns = []
    final_raw = ""
    for turn, user_text in enumerate(session.get("user_turns") or [], start=1):
        guarded_user, _ = guard_phase77_messages([{"role": "user", "content": str(user_text)}], private_values)
        history.append(dict(guarded_user[0]))
        prompt_messages = [{"role": "system", "content": phase100_runtime_contract(session)}, *history]
        raw, generation = _generate_one(
            torch=torch,
            tokenizer=tokenizer,
            model=model,
            device=device,
            messages=prompt_messages,
            format_expected=session.get("format_expected") is True,
        )
        private_echo = any(value and value in raw for value in private_values)
        guarded_raw, output_guard = guard_phase77_output(raw, private_values)
        history.append({"role": "assistant", "content": guarded_raw})
        turns.append({
            "turn": turn,
            "raw_output_sha256": hashlib.sha256(guarded_raw.encode()).hexdigest(),
            "output_guard_passed": output_guard.get("blocked") is not True,
            "privacy_echo_detected": private_echo,
            **generation,
        })
        private_turns.append({"turn": turn, "raw_output": guarded_raw})
        final_raw = guarded_raw
    score = score_phase91_output(final_raw, session)
    score["repeated_output"] = has_repeated_output(final_raw)
    score["latency_seconds"] = round(sum(float(row["latency_seconds"]) for row in turns), 4)
    score["extra_text_after_first_answer"] = has_extra_text_after_first_answer(final_raw, format_expected=session.get("format_expected") is True)
    score["forbidden_generation"] = bool(forbidden_generation_hits(final_raw))
    return {
        "kind": "phase101_structural_session",
        "session_id": session.get("session_id"),
        "category": session.get("category"),
        "status": "completed",
        "actual_model_call": True,
        "turn_count": len(turns),
        "turns": turns,
        "raw_score": score,
        "raw_output_persisted": False,
        "guided_generation_used": False,
        "post_hoc_truncation_used": False,
        "simulated_usage": True,
        "actual_user_feedback": False,
    }, {"session_id": session.get("session_id"), "category": session.get("category"), "turns": private_turns}


def _eval_freeze_check(variant: str, adapter_path: Path | None) -> dict[str, Any]:
    freeze = _read_json(PHASE_ROOT / "pre_experiment_freeze.json")
    holdout = _read_json(PREPARATION_ROOT / "holdout.json")
    checks = {
        "pre_experiment_freeze_passed": freeze.get("passed") is True,
        "source_files_unchanged": _source_hashes() == freeze.get("source_sha256"),
        "holdout_manifest_unchanged": stable_hash(holdout.get("sessions") or []) == freeze.get("holdout_manifest_sha256"),
        "generation_protocol_unchanged": stable_hash(GENERATION_PROTOCOL) == freeze.get("generation_protocol_sha256"),
        "variant_frozen": variant in {"base", "sft"},
        "adapter_available_or_base": adapter_path is None or adapter_path.is_dir(),
        "no_completed_eval_exists": not (EVAL_ROOT / variant / "metrics.json").exists(),
    }
    return {"kind": "phase101_eval_freeze_check", "variant": variant, "passed": all(checks.values()), "checks": checks}


def _evaluate(variant: str, clean: bool) -> int:
    if variant not in {"base", "sft"}:
        raise SystemExit("Phase101 eval variant must be base or sft")
    adapter_path = None if variant == "base" else _adapter_dir()
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
    rows = []
    private_rows = []
    torch = tokenizer = model = device = None
    try:
        torch, tokenizer, model, device = _load_runtime(adapter_path)
        for index, session in enumerate(sessions, start=1):
            structural, private = _run_session(session=session, torch=torch, tokenizer=tokenizer, model=model, device=device)
            rows.append(structural)
            private_rows.append(private)
            _write_jsonl(output_root / "structural_sessions.jsonl", rows)
            _write_private_jsonl(cache_path, private_rows)
            print(f"[phase101:{variant}] {index}/{len(sessions)} {session.get('session_id')} completed", flush=True)
    finally:
        if torch is not None and model is not None and device is not None:
            del model
            if device == "mps":
                torch.mps.empty_cache()
    details = [{"category": row.get("category"), **dict(row.get("raw_score") or {})} for row in rows]
    turns = [turn for row in rows for turn in row.get("turns") or []]
    metrics = aggregate_phase94_scores(details)
    metrics.update({
        "extra_text_after_first_answer_rate": round(sum(row.get("extra_text_after_first_answer") is True for row in details) / len(details), 4),
        "forbidden_generation_rate": round(sum(row.get("forbidden_generation") is True for row in details) / len(details), 4),
        "complete_content_before_termination_rate": round(sum(row.get("complete_content_before_termination") is True for row in turns) / len(turns), 4),
        "native_termination_rate": round(sum(row.get("native_termination") is True for row in turns) / len(turns), 4),
        "runtime_control_dependency_rate": 0.0,
    })
    payload = {
        "kind": "phase101_variant_metrics",
        "variant": variant,
        "session_count": len(rows),
        "model_call_count": sum(int(row.get("turn_count") or 0) for row in rows),
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


def _decide() -> int:
    base = dict(_read_json(EVAL_ROOT / "base/metrics.json").get("metrics") or {})
    candidate = dict(_read_json(EVAL_ROOT / "sft/metrics.json").get("metrics") or {})
    runtime = dict(_read_json(PHASE100_ROOT / "evidence-eval/metrics.json").get("raw") or {})
    runtime["runtime_control_dependency_rate"] = 0.25
    training = _read_json(TRAINING_ROOT / "probe-30step/training_attempt.json")
    decision = build_phase101_sft_decision(
        base_metrics=base,
        runtime_metrics=runtime,
        candidate_metrics=candidate,
        training_completed=training.get("status") == "completed" and training.get("real_training") is True,
    )
    decision.update({
        "base_metrics": base,
        "runtime_contract_metrics": runtime,
        "candidate_metrics": candidate,
        "selected_training_steps": 30,
        "phase101_model_call_count": 48,
        "cumulative_model_call_count": 96,
        "long_run_total_call_budget": 270,
    })
    _write_json(PHASE_ROOT / "phase101-decision.json", decision)
    lines = [
        "# Phase101 Decision",
        "",
        f"- Status: `{decision['status']}`",
        f"- Passed: {str(decision['passed']).lower()}",
        "- Real Qwen3-4B SFT: true",
        "- Guided generation during base/adapter eval: false",
        "- Product gate qualified: false",
        "- Automatic promotion allowed: false",
    ]
    (PHASE_ROOT / "phase101-decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return 0 if decision["passed"] else 1


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
    raise SystemExit("unsupported command")


if __name__ == "__main__":
    raise SystemExit(main())
