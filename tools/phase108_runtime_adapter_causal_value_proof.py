#!/usr/bin/env python3
"""Run Phase108 trusted-runtime and adapter causal product-value proof."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
from importlib import metadata as importlib_metadata
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = REPO_ROOT / "pfe-core"
SERVER_ROOT = REPO_ROOT / "pfe-server"
for import_root in (CORE_ROOT, SERVER_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from pfe_core.phase75_personalization_benefit_benchmark import stable_hash
from pfe_core.phase77_private_value_guarded_runtime import (
    guard_phase77_messages,
    guard_phase77_output,
)
from pfe_core.phase99_qwen3_native_generation_boundary import (
    qwen3_bad_words_ids,
    qwen3_eos_token_ids,
    render_qwen3_no_think_prompt,
)
from pfe_core.phase108_runtime_adapter_causal_value import (
    PHASE108_CALL_BUDGET,
    PHASE108_CONFIRMATION_RESERVE,
    PHASE108_DIAGNOSTIC_CALLS,
    PHASE108_DIAGNOSTIC_VARIANT,
    PHASE108_MAIN_CALLS,
    PHASE108_MAIN_VARIANTS,
    PHASE108_MINIMAL_CONTRACT,
    PHASE108_PRODUCT_CONTRACT,
    aggregate_phase108_runtime_evidence,
    aggregate_phase108_scores,
    audit_phase108_sessions,
    build_phase108_decision,
    build_phase108_sessions,
    build_phase108_stopping_criteria,
    compare_phase108_variants,
    phase108_content_complete,
    phase108_diagnostic_session_ids,
    score_phase108_session,
)


EVIDENCE_ROOT = REPO_ROOT / "docs/demo/phase108-runtime-adapter-causal-value-proof"
PREPARATION_ROOT = EVIDENCE_ROOT / "evidence-preparation"
RUNTIME_ROOT = EVIDENCE_ROOT / "evidence-runtime"
EVAL_ROOT = EVIDENCE_ROOT / "evidence-eval"
TRAINING_ROOT = EVIDENCE_ROOT / "evidence-training"
CONFIRMATION_ROOT = EVIDENCE_ROOT / "evidence-confirmation"
FAILURE_ROOT = EVIDENCE_ROOT / "evidence-failures"
PRIVATE_ROOT = Path("/private/tmp/pfe-phase108-simulated-review")
MODEL_PATH = REPO_ROOT / "models/Qwen3-4B"
PHASE106_ADAPTER = REPO_ROOT / "trainer_job_outputs/phase106-qwen3-4b-sft-30step/peft_lora"
PHASE107_ADAPTER = REPO_ROOT / "trainer_job_outputs/phase107-qwen3-4b-token-faithful-dpo/30step/dpo_adapter"
PHASE107_ROOT = REPO_ROOT / "docs/demo/phase107-runtime-provenance-and-token-faithful-dpo"
CALL_LEDGER = EVAL_ROOT / "call_ledger.jsonl"
ALLOWED_VARIANTS = (*PHASE108_MAIN_VARIANTS, PHASE108_DIAGNOSTIC_VARIANT)
MODEL_CALL_BUDGET = PHASE108_CALL_BUDGET
EVAL_VARIANTS = ALLOWED_VARIANTS
PHASE106_ADAPTER_ROOT = PHASE106_ADAPTER
PHASE107_ADAPTER_ROOT = PHASE107_ADAPTER
PRIVATE_REVIEW_ROOT = PRIVATE_ROOT
GENERATION_PROTOCOL = {
    "input_max_length": 3072,
    "max_new_tokens": 160,
    "do_sample": False,
    "repetition_penalty": 1.15,
    "no_repeat_ngram_size": 4,
    "enable_thinking": False,
    "guided_generation": False,
    "post_hoc_truncation_allowed": False,
    "automatic_retry_count": 0,
}
EXPECTED_ADAPTER_HASHES = {
    "phase106_sft": "8bb2035aa9ae1b4dd8dd90cf92127b8aa4aef4d97bc1fb5903728b58961063d2",
    "phase107_dpo": "6840137a178f9fb37a62aae9cb1b2cd404bfb616fe9c8a667e32ffd855e05749",
}
PRIOR_EVIDENCE_ROOTS = (
    REPO_ROOT / "docs/demo/phase43-qwen3-4b-personal-preference-benefit-proof",
    REPO_ROOT / "docs/demo/phase75-personalization-benefit-benchmark",
    REPO_ROOT / "docs/demo/phase99-qwen3-native-generation-boundary",
    REPO_ROOT / "docs/demo/phase100-104-autonomous-qwen3-training-benefit-loop",
    REPO_ROOT / "docs/demo/phase105-qwen3-no-think-curriculum-alignment",
    REPO_ROOT / "docs/demo/phase106-qwen3-stratified-curriculum-repair",
    PHASE107_ROOT,
)


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
    text = "".join(json.dumps(dict(row), ensure_ascii=False) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")


def _append_jsonl(path: Path, payload: Mapping[str, Any], *, private: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    if private:
        path.chmod(0o600)


def _safe_clean(path: Path, parent: Path) -> None:
    resolved = path.resolve()
    if resolved == parent.resolve() or parent.resolve() not in resolved.parents:
        raise ValueError(f"refusing unsafe clean: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def _attempted_call_count() -> int:
    return sum(row.get("event") == "attempted" for row in _read_jsonl(CALL_LEDGER))


def _reserve_model_call(ledger_path: Path, payload: Mapping[str, Any]) -> None:
    rows = _read_jsonl(ledger_path)
    counted = sum(row.get("event") in {None, "attempted"} for row in rows)
    if counted >= MODEL_CALL_BUDGET:
        raise RuntimeError(f"Phase108 model call budget {MODEL_CALL_BUDGET} exhausted")
    _append_jsonl(ledger_path, payload)


def _clean_evidence(*, evidence_root: Path, ledger_path: Path) -> None:
    if ledger_path.is_file() and ledger_path.stat().st_size:
        raise RuntimeError("Phase108 clean refused because the append-only call ledger has attempts")
    if evidence_root.exists():
        _safe_clean(evidence_root, evidence_root.parent)


def _clean_before_calls() -> None:
    try:
        _clean_evidence(evidence_root=EVIDENCE_ROOT, ledger_path=CALL_LEDGER)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    if PRIVATE_ROOT.exists():
        _safe_clean(PRIVATE_ROOT, PRIVATE_ROOT.parent)


def _manifest(paths: Iterable[Path]) -> dict[str, Any]:
    rows = []
    for path in sorted({value.resolve() for value in paths if value.is_file()}):
        rows.append({"path": str(path), "size_bytes": path.stat().st_size, "sha256": _sha256(path)})
    return {"file_count": len(rows), "files": rows, "manifest_sha256": stable_hash(rows)}


def _model_manifest() -> dict[str, Any]:
    names = {
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
    }
    paths = [path for path in MODEL_PATH.iterdir() if path.name in names or path.suffix == ".safetensors"]
    return _manifest(paths)


def _adapter_manifest(path: Path, expected_sha256: str) -> dict[str, Any]:
    artifact = path / "adapter_model.safetensors"
    paths = [value for value in path.iterdir() if value.is_file() and value.name in {"adapter_model.safetensors", "adapter_config.json"}]
    manifest = _manifest(paths)
    actual = _sha256(artifact) if artifact.is_file() else None
    return {
        **manifest,
        "artifact_dir": str(path),
        "artifact_sha256": actual,
        "expected_sha256": expected_sha256,
        "valid": actual == expected_sha256 and (path / "adapter_config.json").is_file(),
    }


def _source_hashes() -> dict[str, str]:
    paths = {
        "phase108_core": CORE_ROOT / "pfe_core/phase108_runtime_adapter_causal_value.py",
        "phase108_driver": REPO_ROOT / "tools/phase108_runtime_adapter_causal_value_proof.py",
        "phase108_core_test": REPO_ROOT / "tests/test_phase108_runtime_adapter_causal_value.py",
        "phase108_runtime_test": REPO_ROOT / "tests/test_phase108_runtime_productization.py",
        "phase108_driver_test": REPO_ROOT / "tests/test_phase108_driver_safety.py",
        "provenance": CORE_ROOT / "pfe_core/inference/provenance.py",
        "server_models": SERVER_ROOT / "pfe_server/models.py",
        "server_app": SERVER_ROOT / "pfe_server/app.py",
        "phase77_privacy_guard": CORE_ROOT / "pfe_core/phase77_private_value_guarded_runtime.py",
        "phase99_generation": CORE_ROOT / "pfe_core/phase99_qwen3_native_generation_boundary.py",
        "phase107_driver": REPO_ROOT / "tools/phase107_runtime_provenance_and_token_faithful_dpo.py",
        "trainer_executor": CORE_ROOT / "pfe_core/trainer/executors.py",
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Phase108 source freeze is missing: {', '.join(missing)}")
    return {name: _sha256(path) for name, path in paths.items()}


def _extract_prior_texts(value: Any, *, key: str = "") -> list[str]:
    selected_keys = {
        "prompt", "user_goal", "user_correction", "continuation_request", "user_turns",
        "instruction", "input", "chosen", "rejected",
    }
    rows: list[str] = []
    if isinstance(value, Mapping):
        if value.get("role") == "user" and isinstance(value.get("content"), str):
            rows.append(str(value["content"]).strip())
        for child_key, child in value.items():
            if child_key in selected_keys:
                if isinstance(child, str) and child.strip():
                    rows.append(child.strip())
                else:
                    rows.extend(_extract_prior_texts(child, key=child_key))
            elif isinstance(child, (Mapping, list, tuple)):
                rows.extend(_extract_prior_texts(child, key=str(child_key)))
    elif isinstance(value, (list, tuple)):
        for child in value:
            if isinstance(child, str) and key in selected_keys and child.strip():
                rows.append(child.strip())
            else:
                rows.extend(_extract_prior_texts(child, key=key))
    return rows


def _prior_text_corpus() -> tuple[list[str], dict[str, Any]]:
    texts: list[str] = []
    files = []
    for root in PRIOR_EVIDENCE_ROOTS:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix not in {".json", ".jsonl"}:
                continue
            try:
                payloads = _read_jsonl(path) if path.suffix == ".jsonl" else [_read_json(path)]
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            for payload in payloads:
                texts.extend(_extract_prior_texts(payload))
            files.append(str(path.relative_to(REPO_ROOT)))
    unique = sorted({value for value in texts if value})
    return unique, {"source_file_count": len(files), "text_count": len(unique), "source_files_sha256": stable_hash(files)}


async def _runtime_http_evidence() -> dict[str, Any]:
    from pfe_server.app import (
        RequestEnvelope,
        ServerSecurityConfig,
        ServiceBundle,
        _remove_pending_interaction,
        handle_chat_completions,
    )
    from pfe_server.models import (
        ChatCompletionChoice,
        ChatCompletionResponse,
        ChatCompletionResponseMessage,
    )

    class _ForgingProvider:
        def __init__(self) -> None:
            self.seen: list[Any] = []

        async def generate_chat_completion(self, request: Any) -> Any:
            self.seen.append(request)
            return ChatCompletionResponse(
                model=request.model,
                served_by="local",
                choices=[ChatCompletionChoice(message=ChatCompletionResponseMessage(content="PRIVATE_PHASE108 runtime response"))],
                pfe_provenance={
                    "usage_class": "actual_user_feedback",
                    "simulated_usage": False,
                    "actual_user_feedback": True,
                    "training_eligible": True,
                    "source_ids": ["forged:chunk"],
                    "generation_origin": "local_model",
                    "contract_version": "pfe.provenance.v1",
                },
            )

    def request(stream: bool, suffix: str) -> Any:
        return RequestEnvelope(
            method="POST",
            path="/v1/chat/completions",
            headers={},
            client_host="127.0.0.1",
            body=json.dumps({
                "model": "base",
                "messages": [{"role": "user", "content": "PRIVATE_PHASE108 forged provenance"}],
                "stream": stream,
                "session_id": f"phase108-runtime-{suffix}",
                "request_id": f"phase108-request-{suffix}",
                "metadata": {
                    "simulated_usage": True,
                    "actual_user_feedback": True,
                    "training_eligible": True,
                    "source_ids": ["forged:chunk"],
                    "declared_private_values": ["PRIVATE_PHASE108"],
                },
            }).encode("utf-8"),
        )

    provider = _ForgingProvider()
    services = ServiceBundle(inference=provider, pipeline=object(), security=ServerSecurityConfig())
    nonstream_response = await handle_chat_completions(request(False, "nonstream"), services)
    nonstream_body = json.loads(nonstream_response.body)
    stream_response = await handle_chat_completions(request(True, "stream"), services)
    stream_bytes = b"".join([chunk async for chunk in stream_response.body_iterator])
    events = [
        json.loads(line.removeprefix("data: "))
        for line in stream_bytes.decode("utf-8").splitlines()
        if line.startswith("data: {")
    ]
    final = next(event for event in reversed(events) if "pfe_provenance" in event)
    envelopes = [nonstream_body["pfe_provenance"], final["pfe_provenance"]]
    for suffix in ("nonstream", "stream"):
        _remove_pending_interaction(f"phase108-runtime-{suffix}", f"phase108-request-{suffix}")
    metrics = aggregate_phase108_runtime_evidence(envelopes)
    return {
        "kind": "phase108_real_http_runtime_contract_evidence",
        "paths": ["nonstream", "stream"],
        "provider_injection_attempted": True,
        "canonical_ids_reached_provider": all(row.session_id and row.request_id for row in provider.seen),
        "envelopes": envelopes,
        "metrics": metrics,
        "passed": all(
            float(metrics.get(name) or 0.0) == 1.0
            for name in (
                "provenance_envelope_valid_rate", "provenance_injection_resisted_rate",
                "source_id_integrity_rate", "simulated_usage_truth_rate", "training_eligibility_truth_rate",
            )
        ),
    }


def _prepare(clean: bool) -> int:
    if clean:
        _clean_before_calls()
    for root in (PREPARATION_ROOT, RUNTIME_ROOT, EVAL_ROOT, TRAINING_ROOT, CONFIRMATION_ROOT, FAILURE_ROOT, PRIVATE_ROOT):
        root.mkdir(parents=True, exist_ok=True)
    sessions = build_phase108_sessions()
    prior_texts, prior_corpus = _prior_text_corpus()
    holdout_audit = audit_phase108_sessions(sessions["sessions"], previous_texts=prior_texts)
    phase107_decision = _read_json(PHASE107_ROOT / "phase107-final-decision.json")
    model_manifest = _model_manifest()
    adapters = {
        "phase106_sft": _adapter_manifest(PHASE106_ADAPTER, EXPECTED_ADAPTER_HASHES["phase106_sft"]),
        "phase107_dpo": _adapter_manifest(PHASE107_ADAPTER, EXPECTED_ADAPTER_HASHES["phase107_dpo"]),
    }
    runtime = asyncio.run(_runtime_http_evidence())
    _write_json(PREPARATION_ROOT / "holdout.json", sessions)
    _write_json(PREPARATION_ROOT / "holdout_integrity_check.json", {**holdout_audit, "prior_corpus": prior_corpus})
    _write_json(RUNTIME_ROOT / "runtime_contract_evidence.json", runtime)
    source_hashes = _source_hashes()
    dependencies = {
        name: importlib_metadata.version(name)
        for name in ("torch", "transformers", "peft", "pydantic")
    }
    checks = {
        "phase107_archive_unchanged": phase107_decision.get("status") == "archive_phase107_token_faithful_dpo_not_qualified",
        "phase107_product_gate_false": phase107_decision.get("product_gate_qualified") is False,
        "holdout_integrity_passed": holdout_audit.get("passed") is True,
        "model_manifest_present": model_manifest.get("file_count", 0) >= 4,
        "phase106_adapter_valid": adapters["phase106_sft"]["valid"] is True,
        "phase107_adapter_valid": adapters["phase107_dpo"]["valid"] is True,
        "runtime_http_contract_passed": runtime.get("passed") is True,
        "call_ledger_empty": _attempted_call_count() == 0,
        "private_root_outside_repo": REPO_ROOT.resolve() not in PRIVATE_ROOT.resolve().parents,
        "budget_exactly_300": PHASE108_MAIN_CALLS + PHASE108_DIAGNOSTIC_CALLS + PHASE108_CONFIRMATION_RESERVE == PHASE108_CALL_BUDGET,
    }
    freeze = {
        "kind": "phase108_pre_experiment_freeze",
        "frozen_at": _utcnow(),
        "passed": all(checks.values()),
        "checks": checks,
        "source_sha256": source_hashes,
        "holdout_manifest_sha256": sessions["manifest_sha256"],
        "model_manifest": model_manifest,
        "adapter_manifests": adapters,
        "generation_protocol": GENERATION_PROTOCOL,
        "generation_protocol_sha256": stable_hash(GENERATION_PROTOCOL),
        "load_plans": {
            "base": "raw_qwen3_4b",
            "phase106_sft": "raw_qwen3_4b_then_phase106_sft_merge",
            "phase107_dpo": "raw_qwen3_4b_then_phase106_sft_merge_then_phase107_dpo",
            "phase107_dpo_no_runtime": "same_phase107_weight_stack_with_minimal_system_contract",
        },
        "dependencies": dependencies,
        "call_budget": {
            "main": PHASE108_MAIN_CALLS,
            "diagnostic": PHASE108_DIAGNOSTIC_CALLS,
            "confirmation_reserve": PHASE108_CONFIRMATION_RESERVE,
            "total": PHASE108_CALL_BUDGET,
        },
        "external_provider_allowed": False,
        "automatic_retry_allowed": False,
        "automatic_training_allowed": False,
        "automatic_promotion_allowed": False,
        "private_transcripts_committed": False,
    }
    _write_json(EVIDENCE_ROOT / "pre_experiment_freeze.json", freeze)
    _write_json(PREPARATION_ROOT / "model_and_adapter_selection.json", {
        "model": str(MODEL_PATH), "adapters": adapters, "load_plans": freeze["load_plans"], "selected": True,
    })
    print(json.dumps({"passed": freeze["passed"], "checks": checks}, ensure_ascii=False, indent=2))
    return 0 if freeze["passed"] else 2


def _current_freeze_check(variant: str, expected_calls: int) -> dict[str, Any]:
    freeze = _read_json(EVIDENCE_ROOT / "pre_experiment_freeze.json")
    holdout = _read_json(PREPARATION_ROOT / "holdout.json")
    current_adapters = {
        "phase106_sft": _adapter_manifest(PHASE106_ADAPTER, EXPECTED_ADAPTER_HASHES["phase106_sft"]),
        "phase107_dpo": _adapter_manifest(PHASE107_ADAPTER, EXPECTED_ADAPTER_HASHES["phase107_dpo"]),
    }
    remaining_unattempted = expected_calls - sum(
        row.get("event") == "attempted" and row.get("variant") == variant
        for row in _read_jsonl(CALL_LEDGER)
    )
    checks = {
        "freeze_passed": freeze.get("passed") is True,
        "source_files_unchanged": _source_hashes() == freeze.get("source_sha256"),
        "holdout_unchanged": stable_hash(holdout.get("sessions") or []) == freeze.get("holdout_manifest_sha256"),
        "model_unchanged": _model_manifest() == freeze.get("model_manifest"),
        "adapters_unchanged": current_adapters == freeze.get("adapter_manifests"),
        "generation_protocol_unchanged": stable_hash(GENERATION_PROTOCOL) == freeze.get("generation_protocol_sha256"),
        "variant_allowed": variant in ALLOWED_VARIANTS,
        "no_completed_eval_exists": not (EVAL_ROOT / variant / "metrics.json").is_file(),
        "call_budget_available": _attempted_call_count() + max(0, remaining_unattempted) <= PHASE108_CALL_BUDGET,
    }
    return {"kind": "phase108_eval_freeze_check", "variant": variant, "passed": all(checks.values()), "checks": checks}


def _load_model_for_variant(variant: str) -> tuple[Any, Any, Any, str]:
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
    if variant != "base":
        model = PeftModel.from_pretrained(model, str(PHASE106_ADAPTER_ROOT), local_files_only=True)
        model = model.merge_and_unload()
    if variant in {"phase107_dpo", "phase107_dpo_no_runtime"}:
        model = PeftModel.from_pretrained(model, str(PHASE107_ADAPTER_ROOT), local_files_only=True)
    model.to(device)
    model.eval()
    return torch, tokenizer, model, device


def _call_id(variant: str, session_id: str, turn_index: int) -> str:
    return f"{variant}:{session_id}:turn-{turn_index}"


def _reserve_call(variant: str, session_id: str, turn_index: int) -> str:
    call_id = _call_id(variant, session_id, turn_index)
    if any(row.get("event") == "attempted" and row.get("call_id") == call_id for row in _read_jsonl(CALL_LEDGER)):
        raise RuntimeError(f"Phase108 call already attempted; automatic retry forbidden: {call_id}")
    attempted = _attempted_call_count()
    if attempted >= PHASE108_CALL_BUDGET:
        raise RuntimeError("Phase108 local model call budget exhausted")
    _reserve_model_call(CALL_LEDGER, {
        "event": "attempted", "call_index": attempted + 1, "call_id": call_id,
        "variant": variant, "session_id": session_id, "turn_index": turn_index,
        "started_at": _utcnow(), "provider": "local_qwen3_4b", "paid_api": False,
    })
    return call_id


def _completed_private_calls(variant: str) -> dict[str, dict[str, Any]]:
    return {
        str(row["call_id"]): row
        for row in _read_jsonl(PRIVATE_ROOT / f"{variant}.jsonl")
        if row.get("event") == "completed"
    }


def _generate_once(
    *,
    torch: Any,
    tokenizer: Any,
    model: Any,
    device: str,
    guarded_messages: list[dict[str, str]],
    format_mode: str,
    minimum_lines: int,
) -> tuple[str, dict[str, Any]]:
    prompt = render_qwen3_no_think_prompt(tokenizer, guarded_messages)
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True,
        max_length=int(GENERATION_PROTOCOL["input_max_length"]),
    )
    inputs = {name: value.to(device) for name, value in inputs.items()}
    input_length = int(inputs["input_ids"].shape[-1])
    stopping, state = build_phase108_stopping_criteria(
        tokenizer=tokenizer, input_length=input_length,
        format_mode=format_mode, minimum_lines=minimum_lines,
    )
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
        raise RuntimeError("Phase108 generation returned empty output")
    last_token = int(generated[-1].item()) if int(generated.shape[-1]) else None
    if state["triggered"]:
        reason = "runtime_boundary_stop"
    elif last_token in eos_ids:
        reason = "model_eos"
    elif int(generated.shape[-1]) >= int(GENERATION_PROTOCOL["max_new_tokens"]):
        reason = "max_new_tokens"
    else:
        reason = "unknown"
    structural_complete = phase108_content_complete(raw, format_mode=format_mode, minimum_lines=minimum_lines)
    content_complete = bool(raw) if reason == "model_eos" else structural_complete
    metadata = {
        "prompt_tokens": input_length,
        "completion_tokens": int(generated.shape[-1]),
        "latency_seconds": round(time.perf_counter() - started, 4),
        "termination_reason": reason,
        "content_complete_before_stop": content_complete,
        "runtime_boundary_stop": reason == "runtime_boundary_stop",
        "model_eos": reason == "model_eos",
        "max_new_tokens": reason == "max_new_tokens",
        "native_turn_complete": reason in {"runtime_boundary_stop", "model_eos"} and content_complete,
        "guided_generation_used": False,
        "post_hoc_truncation_used": False,
        "automatic_retry_used": False,
    }
    return raw, metadata


def _generate_session(
    *,
    variant: str,
    session_id: str,
    turn_index: int,
    torch: Any,
    tokenizer: Any,
    model: Any,
    device: str,
    messages: list[dict[str, str]],
    format_mode: str,
    minimum_lines: int,
    privacy_canary: str,
) -> tuple[str, str, dict[str, Any], str]:
    guarded_messages, input_guard = guard_phase77_messages(messages, [privacy_canary] if privacy_canary else [])
    call_id = _reserve_model_call_for_session(variant, session_id, turn_index)
    raw, generation = _generate_once(
        torch=torch,
        tokenizer=tokenizer,
        model=model,
        device=device,
        guarded_messages=guarded_messages,
        format_mode=format_mode,
        minimum_lines=minimum_lines,
    )
    raw_canary_echo = bool(privacy_canary and privacy_canary in raw)
    returned, output_guard = guard_phase77_output(raw, [privacy_canary] if privacy_canary else [])
    generation.update({
        "raw_privacy_echo_detected": raw_canary_echo,
        "input_guard": input_guard,
        "output_guard": output_guard,
    })
    return raw, returned, generation, call_id


def _reserve_model_call_for_session(variant: str, session_id: str, turn_index: int) -> str:
    return _reserve_call(variant, session_id, turn_index)


def _evaluate(variant: str, clean: bool) -> int:
    if variant not in ALLOWED_VARIANTS:
        raise SystemExit(f"unsupported Phase108 eval variant: {variant}")
    if clean and (_attempted_call_count() or (CALL_LEDGER.is_file() and CALL_LEDGER.stat().st_size)):
        raise SystemExit("Phase108 --clean cannot erase attempted model calls")
    output_root = EVAL_ROOT / variant
    if clean and output_root.exists():
        _safe_clean(output_root, EVAL_ROOT)
    if clean:
        (PRIVATE_ROOT / f"{variant}.jsonl").unlink(missing_ok=True)
    holdout = _read_json(PREPARATION_ROOT / "holdout.json")
    sessions = [dict(row) for row in holdout.get("sessions") or []]
    if variant == PHASE108_DIAGNOSTIC_VARIANT:
        selected = set(phase108_diagnostic_session_ids(sessions))
        sessions = [row for row in sessions if row.get("session_id") in selected]
    expected_calls = len(sessions) * 2
    freeze = _current_freeze_check(variant, expected_calls)
    _write_json(output_root / "freeze_check.json", freeze)
    if not freeze["passed"]:
        return 2
    private_calls = _completed_private_calls(variant)
    attempted = {str(row.get("call_id")) for row in _read_jsonl(CALL_LEDGER) if row.get("event") == "attempted"}
    completed_events = {str(row.get("call_id")) for row in _read_jsonl(CALL_LEDGER) if row.get("event") == "completed"}
    stranded = sorted((attempted & {_call_id(variant, str(s["session_id"]), turn) for s in sessions for turn in (1, 2)}) - completed_events)
    if stranded:
        _write_json(FAILURE_ROOT / f"{variant}_stranded_calls.json", {
            "kind": "phase108_non_retryable_stranded_calls", "variant": variant,
            "call_ids": stranded, "automatic_retry_allowed": False,
        })
        raise SystemExit(f"Phase108 has non-retryable attempted calls for {variant}: {len(stranded)}")
    structural_rows: list[dict[str, Any]] = []
    scores: list[dict[str, Any]] = []
    torch = tokenizer = model = device = None
    contract = PHASE108_MINIMAL_CONTRACT if variant == PHASE108_DIAGNOSTIC_VARIANT else PHASE108_PRODUCT_CONTRACT
    try:
        torch, tokenizer, model, device = _load_model_for_variant(variant)
        for index, session in enumerate(sessions, start=1):
            session_id = str(session["session_id"])
            expected = dict(session.get("expected") or {})
            privacy_canary = str(expected.get("privacy_canary") or "")
            outputs: list[str] = []
            turns: list[dict[str, Any]] = []
            messages: list[dict[str, str]] = [
                {"role": "system", "content": contract},
                {"role": "user", "content": str(session["user_goal"])},
            ]
            for turn_index in (1, 2):
                call_id = _call_id(variant, session_id, turn_index)
                cached = private_calls.get(call_id)
                if cached is None:
                    format_mode = "short_paragraph" if turn_index == 1 else str(expected.get("format_mode") or "short_paragraph")
                    minimum_lines = 1
                    if format_mode == "bullets":
                        minimum_lines = max(2, min(6, len(expected.get("required_groups") or [])))
                    try:
                        raw, returned, generation, reserved_call_id = _generate_session(
                            variant=variant, session_id=session_id, turn_index=turn_index,
                            torch=torch, tokenizer=tokenizer, model=model, device=device,
                            messages=messages, format_mode=format_mode,
                            minimum_lines=minimum_lines, privacy_canary=privacy_canary,
                        )
                        if reserved_call_id != call_id:
                            raise RuntimeError("Phase108 reserved call id mismatch")
                    except Exception as exc:
                        _append_jsonl(CALL_LEDGER, {
                            "event": "failed", "call_id": call_id, "variant": variant,
                            "session_id": session_id, "turn_index": turn_index,
                            "finished_at": _utcnow(), "error": f"{exc.__class__.__name__}: {exc}",
                        })
                        _write_json(FAILURE_ROOT / f"{variant}_{session_id}_turn_{turn_index}.json", {
                            "kind": "phase108_local_generation_failure", "call_id": call_id,
                            "error": f"{exc.__class__.__name__}: {exc}", "traceback": traceback.format_exc(),
                            "automatic_retry_allowed": False, "paid_api_used": False,
                        })
                        raise
                    cached = {
                        "event": "completed", "call_id": call_id, "variant": variant,
                        "session_id": session_id, "turn_index": turn_index,
                        "raw_output": raw, "returned_output": returned, "generation": generation,
                    }
                    _append_jsonl(PRIVATE_ROOT / f"{variant}.jsonl", cached, private=True)
                    _append_jsonl(CALL_LEDGER, {
                        "event": "completed", "call_id": call_id, "variant": variant,
                        "session_id": session_id, "turn_index": turn_index,
                        "finished_at": _utcnow(), "output_sha256": stable_hash(returned),
                    })
                    private_calls[call_id] = cached
                returned = str(cached["returned_output"])
                generation = dict(cached["generation"])
                outputs.append(returned)
                turns.append({
                    "turn_index": turn_index,
                    "output_sha256": stable_hash(returned),
                    "output_char_count": len(returned),
                    "generation": generation,
                    "raw_output_persisted_in_repo": False,
                })
                messages.append({"role": "assistant", "content": returned})
                if turn_index == 1:
                    messages.append({
                        "role": "user",
                        "content": f"{session['user_correction']}\n{session['continuation_request']}",
                    })
            native_complete = all(dict(row.get("generation") or {}).get("native_turn_complete") is True for row in turns)
            raw_echo = any(dict(row.get("generation") or {}).get("raw_privacy_echo_detected") is True for row in turns)
            score = score_phase108_session(
                session=session, outputs=outputs, native_turns_complete=native_complete,
                system_contract_enabled=variant != PHASE108_DIAGNOSTIC_VARIANT,
                raw_privacy_echo_detected=raw_echo,
            )
            structural_rows.append({
                "kind": "phase108_structural_session", "session_id": session_id,
                "domain": session.get("domain"), "variant": variant, "turn_count": 2,
                "turns": turns, "score": score, "simulated_usage": True,
                "actual_user_feedback": False, "training_eligible": False,
            })
            scores.append(score)
            _write_jsonl(output_root / "structural_sessions.jsonl", structural_rows)
            _write_jsonl(output_root / "simulated_user_scores.jsonl", scores)
            print(f"[phase108:{variant}] {index}/{len(sessions)} accepted={score['accepted']}", flush=True)
    finally:
        if model is not None:
            del model
            if device == "mps" and torch is not None:
                torch.mps.empty_cache()
    metrics = aggregate_phase108_scores(scores)
    attempted_for_variant = sum(
        row.get("event") == "attempted" and row.get("variant") == variant
        for row in _read_jsonl(CALL_LEDGER)
    )
    payload = {
        "kind": "phase108_variant_eval", "variant": variant,
        "expected_session_count": len(sessions), "model_call_count": attempted_for_variant,
        "metrics": metrics, "system_contract_enabled": variant != PHASE108_DIAGNOSTIC_VARIANT,
        "load_plan": _read_json(EVIDENCE_ROOT / "pre_experiment_freeze.json")["load_plans"][variant],
        "private_cache": str(PRIVATE_ROOT / f"{variant}.jsonl"),
        "private_cache_outside_repo": True, "post_hoc_truncation_used": False,
        "automatic_retry_used": False, "simulated_usage": True, "actual_user_feedback_count": 0,
    }
    _write_json(output_root / "metrics.json", payload)
    print(json.dumps({"variant": variant, "model_call_count": attempted_for_variant, "metrics": {key: value for key, value in metrics.items() if key != "details"}}, ensure_ascii=False, indent=2))
    return 0


def _analyze() -> int:
    payloads = {variant: _read_json(EVAL_ROOT / variant / "metrics.json") for variant in ALLOWED_VARIANTS}
    metrics = {variant: dict(payload["metrics"]) for variant, payload in payloads.items()}
    scores = {variant: list(metrics[variant].get("details") or []) for variant in ALLOWED_VARIANTS}
    diagnostic_ids = {str(row["session_id"]) for row in scores[PHASE108_DIAGNOSTIC_VARIANT]}
    product_subset = [row for row in scores["phase107_dpo"] if row.get("session_id") in diagnostic_ids]
    comparisons = {
        "phase106_sft_vs_base": compare_phase108_variants(candidate_scores=scores["phase106_sft"], benchmark_scores=scores["base"], comparison="phase106_sft_vs_base"),
        "phase107_dpo_vs_base": compare_phase108_variants(candidate_scores=scores["phase107_dpo"], benchmark_scores=scores["base"], comparison="phase107_dpo_vs_base"),
        "phase107_dpo_vs_phase106_sft": compare_phase108_variants(candidate_scores=scores["phase107_dpo"], benchmark_scores=scores["phase106_sft"], comparison="phase107_dpo_vs_phase106_sft"),
        "phase107_product_contract_vs_no_runtime": compare_phase108_variants(candidate_scores=product_subset, benchmark_scores=scores[PHASE108_DIAGNOSTIC_VARIANT], comparison="phase107_product_contract_vs_no_runtime"),
    }
    candidate = scores["phase107_dpo"]
    failure_counts = {
        "task_complete": sum(row.get("task_complete") is not True for row in candidate),
        "correction_followed": sum(row.get("correction_followed") is not True for row in candidate),
        "format_adherence": sum(row.get("format_adherence") is not True for row in candidate),
        "factual_guard": sum(row.get("factual_guard") is not True for row in candidate),
        "privacy_boundary": sum(row.get("privacy_boundary") is not True for row in candidate),
        "false_block": sum(row.get("false_block") is True for row in candidate),
        "native_turn_completion": sum(row.get("native_turn_completion") is not True for row in candidate),
    }
    ranked = sorted(failure_counts.items(), key=lambda item: (-item[1], item[0]))
    top_name, top_count = ranked[0]
    second_count = ranked[1][1]
    local_fixable = {"correction_followed", "format_adherence", "factual_guard", "false_block"}
    single_dominant = top_count >= 5 and top_count >= max(1, second_count) * 1.5 and top_name in local_fixable
    diagnosis = {
        "kind": "phase108_failure_diagnosis", "failure_counts": failure_counts,
        "dominant_failure": top_name if single_dominant else None,
        "single_local_fixable_failure": single_dominant,
        "targeted_training_allowed": single_dominant,
        "targeted_training_executed": False,
        "reason": "one_predeclared_local_failure_dominates" if single_dominant else "mixed_or_nonlocal_failure_pattern",
        "confirmation_call_reserve": PHASE108_CONFIRMATION_RESERVE,
    }
    summary = {
        "kind": "phase108_comparison_summary", "metrics": metrics,
        "comparisons": comparisons, "failure_diagnosis": diagnosis,
        "runtime_and_model_metrics_separate": True,
        "diagnostic_system_contract_effect": comparisons["phase107_product_contract_vs_no_runtime"],
        "actual_user_feedback_count": 0, "simulated_usage": True,
    }
    _write_json(EVAL_ROOT / "comparison_summary.json", summary)
    _write_json(FAILURE_ROOT / "failure_diagnosis.json", diagnosis)
    print(json.dumps({"comparisons": {key: {name: value.get(name) for name in ("candidate_wins", "benchmark_wins", "ties", "bootstrap")} for key, value in comparisons.items()}, "failure_diagnosis": diagnosis}, ensure_ascii=False, indent=2))
    return 0


def _evidence_manifest() -> dict[str, Any]:
    excluded = {EVIDENCE_ROOT / "evidence_manifest.json", EVIDENCE_ROOT / "validation_summary.json"}
    files = [path for path in sorted(EVIDENCE_ROOT.rglob("*")) if path.is_file() and path not in excluded]
    return {
        "kind": "phase108_evidence_manifest",
        "files": [{"path": str(path.relative_to(REPO_ROOT)), "sha256": _sha256(path), "size_bytes": path.stat().st_size} for path in files],
        "file_count": len(files), "private_transcripts_committed": False,
        "actual_user_feedback_count": 0,
    }


def _decide() -> int:
    comparison_summary = _read_json(EVAL_ROOT / "comparison_summary.json")
    metrics = dict(comparison_summary["metrics"])
    comparisons = dict(comparison_summary["comparisons"])
    runtime = _read_json(RUNTIME_ROOT / "runtime_contract_evidence.json")["metrics"]
    phase107 = _read_json(PHASE107_ROOT / "phase107-final-decision.json")
    decision = build_phase108_decision(
        metrics=metrics, comparisons=comparisons, runtime_metrics=runtime,
        phase107_remains_archive=phase107.get("status") == "archive_phase107_token_faithful_dpo_not_qualified",
    )
    expected_counts = {"base": 40, "phase106_sft": 40, "phase107_dpo": 40, PHASE108_DIAGNOSTIC_VARIANT: 10}
    complete = all(int(metrics[name].get("session_count") or 0) == count for name, count in expected_counts.items())
    if not complete:
        decision["passed"] = False
        decision["status"] = "archive_phase108_adapter_causal_value_not_qualified"
        decision["recommendation"] = "runtime_contract_primary_archive_adapter"
        decision["checks"]["all_evaluations_complete"] = False
        decision["failed_checks"] = sorted(set([*decision["failed_checks"], "all_evaluations_complete"]))
    else:
        decision["checks"]["all_evaluations_complete"] = True
    decision.update({
        "metrics": metrics, "comparisons": comparisons,
        "failure_diagnosis": comparison_summary["failure_diagnosis"],
        "runtime_metrics": runtime, "model_call_count": _attempted_call_count(),
        "model_call_budget": PHASE108_CALL_BUDGET,
        "private_transcripts_committed": False,
        "external_provider_used": False, "paid_api_used": False,
        "product_gate_qualified": False,
        "automatic_promotion_allowed": False,
    })
    _write_json(EVIDENCE_ROOT / "phase108-final-decision.json", decision)
    runbook = """# Phase108 Runbook

```bash
.venv/bin/python tools/phase108_runtime_adapter_causal_value_proof.py prepare --clean
.venv/bin/python tools/phase108_runtime_adapter_causal_value_proof.py eval --variant base
.venv/bin/python tools/phase108_runtime_adapter_causal_value_proof.py eval --variant phase106_sft
.venv/bin/python tools/phase108_runtime_adapter_causal_value_proof.py eval --variant phase107_dpo
.venv/bin/python tools/phase108_runtime_adapter_causal_value_proof.py eval --variant phase107_dpo_no_runtime
.venv/bin/python tools/phase108_runtime_adapter_causal_value_proof.py analyze
.venv/bin/python tools/phase108_runtime_adapter_causal_value_proof.py decide
.venv/bin/python tools/phase108_runtime_adapter_causal_value_proof.py validate
```

All 260 planned evaluation calls are local Qwen3-4B simulated usage. The remaining 40-call reserve is unavailable unless one predeclared failure clearly dominates. No automatic retry, training, promotion, Hermes integration, external provider, push, or deployment is allowed.
"""
    (EVIDENCE_ROOT / "phase108-runbook.md").write_text(runbook, encoding="utf-8")
    lines = [
        "# Phase108 Final Decision", "",
        f"- Status: `{decision['status']}`",
        f"- Recommendation: `{decision['recommendation']}`",
        f"- Trusted runtime integrity: `{str(decision['checks']['trusted_runtime_integrity_1']).lower()}`",
        f"- Local model calls: `{decision['model_call_count']}/{PHASE108_CALL_BUDGET}`",
        f"- Phase107 lifecycle: `{decision['phase107_lifecycle']}`",
        "- Product gate qualified: `false`",
        "- Automatic promotion allowed: `false`",
        "- Evidence class: `simulated_usage`, not actual user feedback.",
    ]
    (EVIDENCE_ROOT / "phase108-final-decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_json(EVIDENCE_ROOT / "evidence_manifest.json", _evidence_manifest())
    print(json.dumps({key: decision[key] for key in ("status", "recommendation", "passed", "model_call_count", "failed_checks")}, ensure_ascii=False, indent=2))
    return 0


def _validate() -> int:
    decision = _read_json(EVIDENCE_ROOT / "phase108-final-decision.json")
    manifest = _read_json(EVIDENCE_ROOT / "evidence_manifest.json")
    expected = {str(row["path"]): str(row["sha256"]) for row in manifest.get("files") or []}
    excluded = {EVIDENCE_ROOT / "evidence_manifest.json", EVIDENCE_ROOT / "validation_summary.json"}
    current = {
        str(path.relative_to(REPO_ROOT)): _sha256(path)
        for path in sorted(EVIDENCE_ROOT.rglob("*"))
        if path.is_file() and path not in excluded
    }
    ledger = _read_jsonl(CALL_LEDGER)
    attempted = [row for row in ledger if row.get("event") == "attempted"]
    attempted_counts = {variant: sum(row.get("variant") == variant for row in attempted) for variant in ALLOWED_VARIANTS}
    evidence_text = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in EVIDENCE_ROOT.rglob("*") if path.is_file()
    )
    phase107 = _read_json(PHASE107_ROOT / "phase107-final-decision.json")
    checks = {
        "manifest_unchanged": expected == current,
        "phase107_remains_archive": phase107.get("status") == "archive_phase107_token_faithful_dpo_not_qualified",
        "source_freeze_unchanged": _source_hashes() == _read_json(EVIDENCE_ROOT / "pre_experiment_freeze.json").get("source_sha256"),
        "main_call_count_240": sum(attempted_counts[name] for name in PHASE108_MAIN_VARIANTS) == PHASE108_MAIN_CALLS,
        "diagnostic_call_count_20": attempted_counts[PHASE108_DIAGNOSTIC_VARIANT] == PHASE108_DIAGNOSTIC_CALLS,
        "total_call_count_260_within_300": len(attempted) == 260 and len(attempted) <= PHASE108_CALL_BUDGET,
        "no_duplicate_attempted_call_ids": len({str(row.get("call_id")) for row in attempted}) == len(attempted),
        "no_failed_calls": not any(row.get("event") == "failed" for row in ledger),
        "product_gate_false": decision.get("product_gate_qualified") is False,
        "automatic_promotion_false": decision.get("automatic_promotion_allowed") is False,
        "actual_feedback_zero": decision.get("actual_user_feedback_count") == 0,
        "private_transcripts_not_committed": decision.get("private_transcripts_committed") is False,
        "raw_output_absent_from_repo_evidence": '"raw_output"' not in evidence_text,
        "no_external_provider": decision.get("external_provider_used") is False and decision.get("paid_api_used") is False,
        "required_directories_present": all(path.is_dir() for path in (PREPARATION_ROOT, RUNTIME_ROOT, EVAL_ROOT, TRAINING_ROOT, CONFIRMATION_ROOT, FAILURE_ROOT)),
    }
    summary = {"kind": "phase108_validation_summary", "validated_at": _utcnow(), "passed": all(checks.values()), "checks": checks}
    _write_json(EVIDENCE_ROOT / "validation_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["passed"] else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--clean", action="store_true")
    evaluate = sub.add_parser("eval")
    evaluate.add_argument("--variant", choices=ALLOWED_VARIANTS, required=True)
    evaluate.add_argument("--clean", action="store_true")
    sub.add_parser("analyze")
    sub.add_parser("decide")
    sub.add_parser("validate")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "prepare":
        return _prepare(args.clean)
    if args.command == "eval":
        return _evaluate(args.variant, args.clean)
    if args.command == "analyze":
        return _analyze()
    if args.command == "decide":
        return _decide()
    if args.command == "validate":
        return _validate()
    raise SystemExit("unsupported Phase108 command")


if __name__ == "__main__":
    raise SystemExit(main())
