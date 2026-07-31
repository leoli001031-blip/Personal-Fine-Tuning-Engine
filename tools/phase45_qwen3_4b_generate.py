#!/usr/bin/env python3
"""Generate real Phase45 diagnostic or frozen-holdout transcripts for one arm."""

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

from pfe_core.phase45_privacy_multiturn_preference import (
    aggregate_phase45_variant,
    sanitize_privacy_output,
    stable_hash,
    transform_privacy_messages,
)


EVIDENCE_ROOT = REPO_ROOT / "docs" / "demo" / "phase45-privacy-structural-multiturn-preference"
MODEL_PATH = REPO_ROOT / "models" / "Qwen3-4B"
HOLDOUT_PATH = EVIDENCE_ROOT / "evidence-holdout" / "holdout.json"
DIAGNOSTIC_PATH = EVIDENCE_ROOT / "evidence-diagnostic" / "diagnostic_sessions.json"
HOLDOUT_FREEZE_PATH = EVIDENCE_ROOT / "evidence-holdout" / "holdout_freeze.json"
GENERATION_PROTOCOL_PATH = EVIDENCE_ROOT / "evidence-holdout" / "generation_protocol.json"
PREFLIGHT_PATH = EVIDENCE_ROOT / "evidence-diagnostic" / "generation_preflight.json"
SELECTION_PATH = EVIDENCE_ROOT / "evidence-diagnostic" / "candidate_selection.json"
CANDIDATE_PATH = EVIDENCE_ROOT / "evidence-curriculum" / "selected_preference_pairs.jsonl"
SCORER_FREEZE_PATH = EVIDENCE_ROOT / "evidence-scorer-calibration" / "scorer_freeze.json"
SCORER_SOURCE = CORE_ROOT / "pfe_core" / "phase45_privacy_multiturn_preference.py"

VARIANTS = (
    "base_raw",
    "base_privacy",
    "candidate_a_raw",
    "candidate_a_privacy",
    "candidate_b_raw",
    "candidate_b_privacy",
    "adapter_raw",
    "adapter_privacy",
)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return dict(value) if isinstance(value, Mapping) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


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


def _privacy_enabled(variant: str) -> bool:
    return variant.endswith("_privacy")


def _candidate_id(variant: str) -> str | None:
    if variant.startswith("candidate_a"):
        return "candidate_a"
    if variant.startswith("candidate_b"):
        return "candidate_b"
    if variant.startswith("adapter_"):
        selection = _read_json(SELECTION_PATH)
        selected = str(selection.get("selected_candidate_id") or "")
        if selected not in {"candidate_a", "candidate_b"}:
            raise SystemExit(f"Phase45 selected adapter is missing: {SELECTION_PATH}")
        return selected
    return None


def _candidate_attempt(candidate_id: str) -> dict[str, Any]:
    letter = "a" if candidate_id == "candidate_a" else "b"
    path = EVIDENCE_ROOT / "evidence-training-sft" / f"candidate-{letter}-full-160step" / "training_attempt.json"
    attempt = _read_json(path)
    if attempt.get("status") != "completed" or attempt.get("candidate_eligible") is not True:
        raise SystemExit(f"eligible Phase45 adapter is missing: {path}")
    return attempt


def _resolve_adapter(variant: str) -> tuple[Path | None, str | None]:
    candidate_id = _candidate_id(variant)
    if candidate_id is None:
        return None, None
    attempt = _candidate_attempt(candidate_id)
    path = dict(attempt.get("adapter_validation") or {}).get("artifact_dir")
    if not path:
        path = dict(attempt.get("execution") or {}).get("artifact_dir")
    if not path:
        raise SystemExit(f"adapter path missing for {candidate_id}")
    return Path(str(path)).expanduser().resolve(), candidate_id


def _freeze_check(*, mode: str, sessions: list[dict[str, Any]]) -> dict[str, Any]:
    scorer = _read_json(SCORER_FREEZE_PATH)
    current_scorer = _sha256(SCORER_SOURCE)
    scorer_ok = scorer.get("source_sha256") == current_scorer and scorer.get("calibration_status") == "passed"
    protocol = _read_json(GENERATION_PROTOCOL_PATH)
    protocol_hash = str(protocol.get("protocol_sha256") or "")
    protocol_copy = {key: value for key, value in protocol.items() if key != "protocol_sha256"}
    protocol_ok = bool(protocol_hash) and stable_hash(protocol_copy) == protocol_hash and int(protocol.get("max_new_tokens") or 0) == 384
    holdout_ok = True
    expected_holdout_hash = None
    current_holdout_hash = None
    preflight_ok = True
    if mode == "holdout":
        holdout = _read_json(HOLDOUT_FREEZE_PATH)
        expected_holdout_hash = holdout.get("holdout_manifest_sha256")
        current_holdout_hash = stable_hash(sessions)
        holdout_ok = expected_holdout_hash == current_holdout_hash and holdout.get("frozen_before_training") is True
        preflight = _read_json(PREFLIGHT_PATH)
        preflight_ok = preflight.get("status") == "passed" and preflight.get("all_arms_truncation_at_most_0_05") is True
    return {
        "kind": "phase45_generation_freeze_check",
        "passed": scorer_ok and protocol_ok and holdout_ok and preflight_ok,
        "mode": mode,
        "scorer_expected_sha256": scorer.get("source_sha256"),
        "scorer_current_sha256": current_scorer,
        "scorer_passed": scorer_ok,
        "generation_protocol_sha256": protocol_hash,
        "generation_protocol_passed": protocol_ok,
        "holdout_expected_sha256": expected_holdout_hash,
        "holdout_current_sha256": current_holdout_hash,
        "holdout_passed": holdout_ok,
        "diagnostic_preflight_passed": preflight_ok,
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
        str(MODEL_PATH), local_files_only=True, low_cpu_mem_usage=True, dtype=dtype,
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
        return str(tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        ))
    except TypeError:
        return str(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))


def _strip_thinking(text: str) -> tuple[str, bool]:
    raw = str(text or "").strip()
    leaked = bool(re.search(r"<think>|</think>", raw, flags=re.IGNORECASE))
    cleaned = re.sub(r"<think>.*?</think>\s*", "", raw, flags=re.IGNORECASE | re.DOTALL).strip()
    return (cleaned or raw), leaked


def _generate_raw(
    *, torch: Any, tokenizer: Any, model: Any, device: str,
    messages: list[dict[str, str]], protocol: Mapping[str, Any],
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
    generated_count = int(generated.shape[-1])
    eos_values = tokenizer.eos_token_id
    eos_ids = {int(value) for value in eos_values} if isinstance(eos_values, (list, tuple, set)) else {int(eos_values)}
    final_token = int(generated[-1].item()) if generated_count else None
    return raw, {
        "input_tokens": input_length,
        "completion_tokens": generated_count,
        "max_new_tokens": maximum,
        "eos_token_ids": sorted(eos_ids),
        "final_token_id": final_token,
        "truncated": generated_count >= maximum and final_token not in eos_ids,
        "latency_seconds": round(time.perf_counter() - started, 4),
    }


def _aggregate_manifest(manifests: list[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [dict(redaction) for manifest in manifests for redaction in manifest.get("redactions") or []]
    counts: dict[str, int] = {}
    for row in rows:
        name = str(row.get("type") or "")
        counts[name] = counts.get(name, 0) + 1
    return {
        "kind": "phase45_session_privacy_redaction_manifest",
        "transform_call_count": len(manifests),
        "redaction_count": len(rows),
        "redaction_type_counts": dict(sorted(counts.items())),
        "redactions": rows,
        "raw_values_persisted": False,
        "manifest_sha256": stable_hash(rows),
    }


def _run_session(
    *, session: Mapping[str, Any], variant: str, torch: Any, tokenizer: Any,
    model: Any, device: str, protocol: Mapping[str, Any], runtime: Mapping[str, Any],
) -> dict[str, Any]:
    privacy = _privacy_enabled(variant)
    system_contract = str(protocol.get("uniform_system_contract") or "").strip()
    raw_history: list[dict[str, str]] = (
        [{"role": "system", "content": system_contract}] if system_contract else []
    )
    persisted_turns: list[dict[str, str]] = []
    generations: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    output_audits: list[dict[str, Any]] = []
    user_turns = [
        str(session.get("user_goal") or ""),
        str(session.get("user_correction") or ""),
        f"{session.get('continuation_request') or ''}\n{session.get('acceptance_request') or ''}".strip(),
    ]
    for turn_index, user_text in enumerate(user_turns, start=1):
        raw_history.append({"role": "user", "content": user_text})
        transformed = transform_privacy_messages(raw_history) if privacy else None
        model_messages = transformed.messages if transformed else list(raw_history)
        if transformed:
            manifests.append(transformed.manifest)
            persisted_user = dict(model_messages[-1])
        else:
            persisted_user = {"role": "user", "content": user_text}
        persisted_turns.append(persisted_user)
        raw_output, info = _generate_raw(
            torch=torch,
            tokenizer=tokenizer,
            model=model,
            device=device,
            messages=model_messages,
            protocol=protocol,
        )
        if transformed:
            persisted_raw, audit = sanitize_privacy_output(raw_output, transformed)
            output_audits.append({"turn": turn_index, **audit})
            cleaned, think_leak = _strip_thinking(persisted_raw)
            info.update({
                "raw_content": persisted_raw,
                "raw_content_sanitized_before_persistence": True,
                "raw_output_sha256_before_sanitization": audit["raw_output_sha256_before_sanitization"],
                "output_redaction_count": audit["output_redaction_count"],
            })
        else:
            cleaned, think_leak = _strip_thinking(raw_output)
            info.update({"raw_content": raw_output, "raw_content_sanitized_before_persistence": False})
        if not cleaned:
            raise RuntimeError("real Qwen3-4B generation became empty after output handling")
        info["think_leak_detected"] = think_leak
        assistant = {"role": "assistant", "content": cleaned}
        persisted_turns.append(assistant)
        raw_history.append(assistant)
        generations.append({"turn": turn_index, **info})

    transcript: dict[str, Any] = {
        "kind": "phase45_real_multiturn_transcript",
        "session_id": session.get("session_id"),
        "category": session.get("category"),
        "variant": variant,
        "model_id": str(MODEL_PATH),
        "adapter_path": runtime.get("adapter_path"),
        "adapter_loaded": runtime.get("adapter_loaded"),
        "privacy_runtime_enabled": privacy,
        "uniform_system_contract": system_contract or None,
        "uniform_system_contract_sha256": hashlib.sha256(system_contract.encode("utf-8")).hexdigest() if system_contract else None,
        "device": device,
        "actual_model_call": True,
        "hardcoded_response": False,
        "status": "completed",
        "turns": persisted_turns,
        "generation": generations,
        "latency_seconds": [row["latency_seconds"] for row in generations],
        "truncated_response": any(row["truncated"] for row in generations),
        "think_leak_detected": any(row["think_leak_detected"] for row in generations),
        "synthetic_privacy_values_only": True,
        "actual_user_feedback": False,
        "created_at": _utcnow(),
    }
    if privacy:
        transcript["privacy_runtime"] = {
            "input_manifest": _aggregate_manifest(manifests),
            "input_manifests": manifests,
            "output_audits": output_audits,
            "raw_private_values_entered_model": False,
            "raw_private_values_persisted": False,
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
    parser.add_argument("--mode", choices=("diagnostic", "holdout"), default="holdout")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    if args.mode == "diagnostic" and args.variant.startswith("adapter_"):
        raise SystemExit("diagnostic runs must name candidate_a or candidate_b explicitly")
    if args.mode == "holdout" and args.variant.startswith("candidate_"):
        raise SystemExit("formal holdout runs must use selected adapter_raw or adapter_privacy")

    source = _read_json(DIAGNOSTIC_PATH if args.mode == "diagnostic" else HOLDOUT_PATH)
    sessions = [dict(row) for row in source.get("sessions") or []]
    if args.limit is not None:
        sessions = sessions[:max(0, int(args.limit))]
    freeze = _freeze_check(mode=args.mode, sessions=sessions)
    if freeze["passed"] is not True:
        raise SystemExit(f"Phase45 frozen scorer/protocol/holdout check failed: {freeze}")
    protocol = _read_json(GENERATION_PROTOCOL_PATH)
    output_dir = EVIDENCE_ROOT / ("evidence-diagnostic/runs" if args.mode == "diagnostic" else "evidence-holdout/real-80-session")
    output_path = output_dir / f"transcripts_{args.variant}.jsonl"
    metrics_path = output_dir / f"metrics_{args.variant}.json"
    freeze_path = output_dir / f"freeze_check_{args.variant}.json"
    if args.clean:
        output_path.unlink(missing_ok=True)
        metrics_path.unlink(missing_ok=True)
    _write_json(freeze_path, freeze)
    existing = [] if args.clean else _read_jsonl(output_path)
    wanted_ids = {str(row.get("session_id")) for row in sessions}
    transcripts = [row for row in existing if str(row.get("session_id")) in wanted_ids]
    completed_ids = {str(row.get("session_id")) for row in transcripts if row.get("status") == "completed"}

    adapter_path, candidate_id = _resolve_adapter(args.variant)
    torch, tokenizer, model, device, runtime = _load_runtime(adapter_path)
    try:
        for index, session in enumerate(sessions, start=1):
            session_id = str(session.get("session_id") or "")
            if session_id in completed_ids:
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
                    "kind": "phase45_real_multiturn_transcript",
                    "session_id": session.get("session_id"),
                    "category": session.get("category"),
                    "variant": args.variant,
                    "model_id": str(MODEL_PATH),
                    "adapter_path": runtime.get("adapter_path"),
                    "adapter_loaded": runtime.get("adapter_loaded"),
                    "privacy_runtime_enabled": _privacy_enabled(args.variant),
                    "device": device,
                    "actual_model_call": False,
                    "hardcoded_response": False,
                    "status": "failed",
                    "error": f"{exc.__class__.__name__}: {exc}",
                    "turns": [],
                    "generation": [],
                    "latency_seconds": [],
                    "synthetic_privacy_values_only": True,
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

    targets = [str(row.get("chosen") or "") for row in _read_jsonl(CANDIDATE_PATH)]
    report = aggregate_phase45_variant(transcripts, sessions, training_targets=targets)
    privacy_checks = [
        dict(row.get("privacy_persistence_check") or {}).get("passed") is True
        for row in transcripts
        if row.get("privacy_runtime_enabled") is True and row.get("status") == "completed"
    ]
    report.update({
        "variant": args.variant,
        "mode": args.mode,
        "candidate_id": candidate_id,
        "model_id": str(MODEL_PATH),
        "adapter_path": str(adapter_path) if adapter_path else None,
        "adapter_loaded": adapter_path is not None,
        "privacy_runtime_enabled": _privacy_enabled(args.variant),
        "all_transcripts_completed": len(transcripts) == len(sessions) and all(row.get("status") == "completed" for row in transcripts),
        "privacy_persistence_checks_passed": all(privacy_checks),
        "transcript_path": str(output_path),
        "freeze_check": freeze,
        "generation_protocol": protocol,
        "model_call_count": sum(len(row.get("generation") or []) for row in transcripts if row.get("actual_model_call") is True),
        "think_leak_rate": round(sum(bool(row.get("think_leak_detected")) for row in transcripts) / len(transcripts), 4) if transcripts else 0.0,
        "actual_user_feedback": False,
        "simulated_usage": True,
        "actual_product_benefit_claim_allowed": False,
        "created_at": _utcnow(),
    })
    _write_json(metrics_path, report)
    print(json.dumps({key: report.get(key) for key in (
        "variant", "session_count", "model_call_count", "actual_model_calls", "user_preference_score",
        "privacy_violation_rate", "secret_echo_rate", "placeholder_leak_rate", "over_redaction_rate",
        "follows_latest_user_intent_rate", "correction_responsiveness_rate", "response_diversity",
        "repetition_rate", "truncated_response_rate",
    )}, ensure_ascii=False, indent=2))
    passed = (
        report.get("actual_model_calls") is True
        and report.get("all_transcripts_completed") is True
        and report.get("privacy_persistence_checks_passed") is True
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
