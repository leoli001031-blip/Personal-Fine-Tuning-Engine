#!/usr/bin/env python3
"""Generate real Qwen3-4B Phase46 runtime-ablation transcripts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import time
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = REPO_ROOT / "pfe-core"
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

from pfe_core.phase45_privacy_multiturn_preference import sanitize_privacy_output, transform_privacy_messages
from pfe_core.phase46_runtime_first_latest_intent import (
    aggregate_phase46_variant,
    build_latest_intent_envelope,
    stable_hash,
)


EVIDENCE_ROOT = REPO_ROOT / "docs" / "demo" / "phase46-runtime-first-latest-intent-ablation"
PHASE45_ROOT = REPO_ROOT / "docs" / "demo" / "phase45-privacy-structural-multiturn-preference"
MODEL_PATH = REPO_ROOT / "models" / "Qwen3-4B"
HOLDOUT_PATH = EVIDENCE_ROOT / "evidence-holdout" / "holdout.json"
HOLDOUT_FREEZE_PATH = EVIDENCE_ROOT / "evidence-holdout" / "holdout_freeze.json"
PROTOCOL_PATH = EVIDENCE_ROOT / "evidence-holdout" / "runtime_protocol.json"
SCORER_FREEZE_PATH = EVIDENCE_ROOT / "evidence-scorer-calibration" / "scorer_freeze.json"
SCORER_SOURCE = CORE_ROOT / "pfe_core" / "phase46_runtime_first_latest_intent.py"
CANDIDATE_PATH = EVIDENCE_ROOT / "evidence-curated-candidates" / "simulated_review_candidates.jsonl"
SELECTION_PATH = PHASE45_ROOT / "evidence-diagnostic" / "candidate_selection.json"
VARIANTS = ("base_privacy", "base_privacy_intent", "adapter_privacy_intent")


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return dict(value) if isinstance(value, Mapping) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _write_jsonl_atomic(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _intent_enabled(variant: str) -> bool:
    return variant.endswith("_intent")


def _adapter_path(variant: str) -> Path | None:
    if not variant.startswith("adapter_"):
        return None
    selection = _read_json(SELECTION_PATH)
    path = Path(str(selection.get("selected_adapter_path") or "")).expanduser().resolve()
    if not path.is_dir() or not (path / "adapter_model.safetensors").exists():
        raise SystemExit(f"Phase45 archived adapter is unavailable for eval-only use: {path}")
    return path


def _freeze_check(sessions: list[dict[str, Any]]) -> dict[str, Any]:
    scorer = _read_json(SCORER_FREEZE_PATH)
    scorer_current = _sha256(SCORER_SOURCE)
    scorer_ok = scorer.get("source_sha256") == scorer_current and scorer.get("calibration_status") == "passed"
    holdout = _read_json(HOLDOUT_FREEZE_PATH)
    holdout_current = stable_hash(sessions)
    holdout_ok = holdout.get("holdout_manifest_sha256") == holdout_current and holdout.get("frozen_before_model_calls") is True
    protocol = _read_json(PROTOCOL_PATH)
    protocol_hash = str(protocol.get("protocol_sha256") or "")
    protocol_copy = {key: value for key, value in protocol.items() if key != "protocol_sha256"}
    protocol_ok = bool(protocol_hash) and stable_hash(protocol_copy) == protocol_hash and int(protocol.get("max_new_tokens") or 0) == 384
    return {
        "kind": "phase46_generation_freeze_check",
        "passed": scorer_ok and holdout_ok and protocol_ok,
        "scorer_expected_sha256": scorer.get("source_sha256"),
        "scorer_current_sha256": scorer_current,
        "scorer_passed": scorer_ok,
        "holdout_expected_sha256": holdout.get("holdout_manifest_sha256"),
        "holdout_current_sha256": holdout_current,
        "holdout_passed": holdout_ok,
        "protocol_sha256": protocol_hash,
        "protocol_passed": protocol_ok,
    }


def _load_runtime(adapter_path: Path | None) -> tuple[Any, Any, Any, str, dict[str, Any]]:
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
    return torch, tokenizer, model, device, {
        "adapter_loaded": adapter_path is not None,
        "adapter_path": str(adapter_path) if adapter_path else None,
    }


def _render_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    try:
        return str(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )
    except TypeError:
        return str(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))


def _strip_thinking(value: str) -> tuple[str, bool]:
    raw = str(value or "").strip()
    leaked = bool(re.search(r"<think>|</think>", raw, flags=re.IGNORECASE))
    cleaned = re.sub(r"<think>.*?</think>\s*", "", raw, flags=re.IGNORECASE | re.DOTALL).strip()
    return (cleaned or raw), leaked


def _generate_raw(
    *,
    torch: Any,
    tokenizer: Any,
    model: Any,
    device: str,
    messages: list[dict[str, str]],
    protocol: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    prompt = _render_prompt(tokenizer, messages)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=int(protocol.get("input_max_length") or 4096),
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    input_length = int(inputs["input_ids"].shape[-1])
    maximum = int(protocol.get("max_new_tokens") or 384)
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=maximum,
            do_sample=False,
            repetition_penalty=float(protocol.get("repetition_penalty") or 1.05),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = output[0][input_length:]
    raw = tokenizer.decode(generated, skip_special_tokens=True).strip()
    if not raw:
        raise RuntimeError("real Qwen3-4B generation returned empty text")
    count = int(generated.shape[-1])
    eos_value = tokenizer.eos_token_id
    eos_ids = {int(value) for value in eos_value} if isinstance(eos_value, (list, tuple, set)) else {int(eos_value)}
    final_token = int(generated[-1].item()) if count else None
    return raw, {
        "input_tokens": input_length,
        "completion_tokens": count,
        "max_new_tokens": maximum,
        "eos_token_ids": sorted(eos_ids),
        "final_token_id": final_token,
        "truncated": count >= maximum and final_token not in eos_ids,
        "latency_seconds": round(time.perf_counter() - started, 4),
    }


def _aggregate_privacy_manifests(manifests: list[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [dict(redaction) for manifest in manifests for redaction in manifest.get("redactions") or []]
    return {
        "kind": "phase46_session_privacy_manifest",
        "transform_call_count": len(manifests),
        "redaction_count": len(rows),
        "redactions": rows,
        "raw_values_persisted": False,
        "manifest_sha256": stable_hash(rows),
    }


def _run_session(
    *,
    session: Mapping[str, Any],
    variant: str,
    torch: Any,
    tokenizer: Any,
    model: Any,
    device: str,
    protocol: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    intent = _intent_enabled(variant)
    system = str(protocol.get("length_contract") or "")
    if intent:
        system = f"{system}\n{protocol.get('latest_intent_contract') or ''}".strip()
    raw_history: list[dict[str, str]] = [{"role": "system", "content": system}]
    persisted_turns: list[dict[str, str]] = []
    generations: list[dict[str, Any]] = []
    privacy_manifests: list[dict[str, Any]] = []
    intent_manifests: list[dict[str, Any]] = []
    output_audits: list[dict[str, Any]] = []
    user_turns = [
        str(session.get("user_goal") or ""),
        str(session.get("user_correction") or ""),
        f"{session.get('continuation_request') or ''}\n{session.get('acceptance_request') or ''}".strip(),
    ]
    for turn_index, user_text in enumerate(user_turns, start=1):
        raw_history.append({"role": "user", "content": user_text})
        privacy = transform_privacy_messages(raw_history)
        privacy_manifests.append(privacy.manifest)
        if intent:
            model_messages, intent_manifest = build_latest_intent_envelope(privacy.messages)
            intent_manifests.append(intent_manifest)
        else:
            model_messages = privacy.messages
            intent_manifest = {
                "kind": "phase46_latest_intent_manifest",
                "latest_intent_wrapped": False,
                "latest_user_message_index": max(index for index, row in enumerate(privacy.messages) if row["role"] == "user"),
            }
            intent_manifests.append(intent_manifest)
        persisted_turns.append(dict(model_messages[-1]))
        raw_output, info = _generate_raw(
            torch=torch,
            tokenizer=tokenizer,
            model=model,
            device=device,
            messages=model_messages,
            protocol=protocol,
        )
        persisted_raw, audit = sanitize_privacy_output(raw_output, privacy)
        output_audits.append({"turn": turn_index, **audit})
        cleaned, think_leak = _strip_thinking(persisted_raw)
        if not cleaned:
            raise RuntimeError("real Qwen3-4B output became empty after privacy handling")
        info.update(
            {
                "turn": turn_index,
                "raw_content": persisted_raw,
                "raw_content_sanitized_before_persistence": True,
                "raw_output_sha256_before_sanitization": audit["raw_output_sha256_before_sanitization"],
                "output_redaction_count": audit["output_redaction_count"],
                "think_leak_detected": think_leak,
            }
        )
        assistant = {"role": "assistant", "content": cleaned}
        persisted_turns.append(assistant)
        raw_history.append(assistant)
        generations.append(info)
    transcript: dict[str, Any] = {
        "kind": "phase46_real_runtime_ablation_transcript",
        "session_id": session.get("session_id"),
        "category": session.get("category"),
        "variant": variant,
        "model_id": str(MODEL_PATH),
        "adapter_path": runtime.get("adapter_path"),
        "adapter_loaded": runtime.get("adapter_loaded"),
        "privacy_runtime_enabled": True,
        "latest_intent_runtime_enabled": intent,
        "system_contract_sha256": hashlib.sha256(system.encode("utf-8")).hexdigest(),
        "device": device,
        "actual_model_call": True,
        "hardcoded_response": False,
        "status": "completed",
        "turns": persisted_turns,
        "generation": generations,
        "latency_seconds": [row["latency_seconds"] for row in generations],
        "truncated_response": any(row["truncated"] for row in generations),
        "think_leak_detected": any(row["think_leak_detected"] for row in generations),
        "privacy_runtime": {
            "input_manifest": _aggregate_privacy_manifests(privacy_manifests),
            "input_manifests": privacy_manifests,
            "output_audits": output_audits,
            "raw_private_values_entered_model": False,
            "raw_private_values_persisted": False,
        },
        "phase46_runtime": {
            "intent_manifest": {
                "latest_intent_wrapped": intent,
                "turn_count": len(intent_manifests),
                "manifests": intent_manifests,
            },
            "old_messages_removed": False,
        },
        "simulated_usage": True,
        "actual_user_feedback": False,
        "actual_product_benefit_claim_allowed": False,
        "created_at": _utcnow(),
    }
    canary = str(dict(session.get("expected") or {}).get("privacy_canary") or "")
    serialized = json.dumps(transcript, ensure_ascii=False)
    transcript["privacy_persistence_check"] = {
        "passed": not canary or canary not in serialized,
        "synthetic_canary_absent": not canary or canary not in serialized,
    }
    return transcript


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    holdout = _read_json(HOLDOUT_PATH)
    sessions = [dict(row) for row in holdout.get("sessions") or []]
    freeze = _freeze_check(sessions)
    if freeze["passed"] is not True:
        raise SystemExit(f"Phase46 scorer/holdout/protocol freeze failed: {freeze}")
    protocol = _read_json(PROTOCOL_PATH)
    output_dir = EVIDENCE_ROOT / "evidence-real-runtime-ablation"
    output_path = output_dir / f"transcripts_{args.variant}.jsonl"
    metrics_path = output_dir / f"metrics_{args.variant}.json"
    freeze_path = output_dir / f"freeze_check_{args.variant}.json"
    if args.clean:
        output_path.unlink(missing_ok=True)
        metrics_path.unlink(missing_ok=True)
    _write_json(freeze_path, freeze)
    existing = [] if args.clean else _read_jsonl(output_path)
    wanted = {str(row.get("session_id")) for row in sessions}
    transcripts = [row for row in existing if str(row.get("session_id")) in wanted]
    completed = {str(row.get("session_id")) for row in transcripts if row.get("status") == "completed"}

    adapter_path = _adapter_path(args.variant)
    torch, tokenizer, model, device, runtime = _load_runtime(adapter_path)
    try:
        for index, session in enumerate(sessions, start=1):
            session_id = str(session.get("session_id") or "")
            if session_id in completed:
                print(f"[{args.variant}] {index}/{len(sessions)} {session_id} resumed", flush=True)
                continue
            try:
                transcript = _run_session(
                    session=session,
                    variant=args.variant,
                    torch=torch,
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                    protocol=protocol,
                    runtime=runtime,
                )
            except Exception as exc:
                transcript = {
                    "kind": "phase46_real_runtime_ablation_transcript",
                    "session_id": session.get("session_id"),
                    "category": session.get("category"),
                    "variant": args.variant,
                    "model_id": str(MODEL_PATH),
                    "adapter_path": runtime.get("adapter_path"),
                    "adapter_loaded": runtime.get("adapter_loaded"),
                    "actual_model_call": False,
                    "hardcoded_response": False,
                    "status": "failed",
                    "error": f"{exc.__class__.__name__}: {exc}",
                    "turns": [],
                    "generation": [],
                    "latency_seconds": [],
                    "actual_user_feedback": False,
                    "created_at": _utcnow(),
                }
            transcripts = [row for row in transcripts if row.get("session_id") != transcript.get("session_id")]
            transcripts.append(transcript)
            transcripts.sort(key=lambda row: str(row.get("session_id")))
            _write_jsonl_atomic(output_path, transcripts)
            print(f"[{args.variant}] {index}/{len(sessions)} {session_id} {transcript['status']}", flush=True)
    finally:
        try:
            del model
            if device == "mps":
                torch.mps.empty_cache()
        except Exception:
            pass

    training_targets = [str(row.get("chosen") or "") for row in _read_jsonl(CANDIDATE_PATH)]
    report = aggregate_phase46_variant(transcripts, sessions, training_targets=training_targets)
    persistence = [
        dict(row.get("privacy_persistence_check") or {}).get("passed") is True
        for row in transcripts
        if row.get("status") == "completed"
    ]
    report.update(
        {
            "variant": args.variant,
            "model_id": str(MODEL_PATH),
            "adapter_path": str(adapter_path) if adapter_path else None,
            "adapter_loaded": adapter_path is not None,
            "privacy_runtime_enabled": True,
            "latest_intent_runtime_enabled": _intent_enabled(args.variant),
            "all_transcripts_completed": len(transcripts) == len(sessions) and all(row.get("status") == "completed" for row in transcripts),
            "privacy_persistence_checks_passed": all(persistence),
            "transcript_path": str(output_path),
            "freeze_check": freeze,
            "runtime_protocol": protocol,
            "model_call_count": sum(len(row.get("generation") or []) for row in transcripts if row.get("actual_model_call") is True),
            "think_leak_rate": round(sum(bool(row.get("think_leak_detected")) for row in transcripts) / len(transcripts), 4) if transcripts else 0.0,
            "actual_user_feedback": False,
            "simulated_usage": True,
            "actual_product_benefit_claim_allowed": False,
            "created_at": _utcnow(),
        }
    )
    _write_json(metrics_path, report)
    print(
        json.dumps(
            {key: report.get(key) for key in (
                "variant",
                "session_count",
                "model_call_count",
                "user_preference_score",
                "latest_intent_satisfaction_rate",
                "old_goal_residue_rate",
                "privacy_violation_rate",
                "response_diversity",
                "repetition_rate",
                "truncated_response_rate",
            )},
            ensure_ascii=False,
            indent=2,
        )
    )
    passed = (
        report.get("actual_model_calls") is True
        and report.get("all_transcripts_completed") is True
        and report.get("privacy_persistence_checks_passed") is True
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
