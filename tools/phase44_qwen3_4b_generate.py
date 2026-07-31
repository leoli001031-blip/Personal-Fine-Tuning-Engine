#!/usr/bin/env python3
"""Generate real Phase44 Qwen3-4B multi-turn transcripts for one arm."""

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

from pfe_core.phase43_personal_preference_benefit import PHASE43_RUNTIME_CONTRACT
from pfe_core.phase44_preference_curriculum import PHASE44_SOFT_RUNTIME_CONTRACT, aggregate_phase44_variant, stable_hash


EVIDENCE_ROOT = REPO_ROOT / "docs" / "demo" / "phase44-preference-curriculum-privacy-safe-retraining"
MODEL_PATH = REPO_ROOT / "models" / "Qwen3-4B"
HOLDOUT_PATH = EVIDENCE_ROOT / "evidence-holdout" / "holdout.json"
DIAGNOSTIC_PATH = EVIDENCE_ROOT / "evidence-holdout" / "diagnostic_sessions.json"
HOLDOUT_FREEZE_PATH = EVIDENCE_ROOT / "evidence-holdout" / "holdout_freeze.json"
CANDIDATE_PATH = EVIDENCE_ROOT / "evidence-curriculum" / "selected_preference_pairs.jsonl"
SCORER_FREEZE_PATH = EVIDENCE_ROOT / "evidence-scorer-calibration" / "scorer_freeze.json"
SCORER_SOURCE = CORE_ROOT / "pfe_core" / "phase44_preference_curriculum.py"


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


def _strip_thinking(text: str) -> tuple[str, bool]:
    raw = str(text or "").strip()
    leaked = bool(re.search(r"<think>|</think>", raw, flags=re.IGNORECASE))
    cleaned = re.sub(r"<think>.*?</think>\s*", "", raw, flags=re.IGNORECASE | re.DOTALL).strip()
    return (cleaned or raw), leaked


def _resolve_adapter(variant: str, steps: int, explicit: Path | None) -> Path | None:
    if variant in {"base", "runtime_v1", "soft_runtime"}:
        return None
    if explicit is not None:
        return explicit.expanduser().resolve()
    attempt_path = EVIDENCE_ROOT / "evidence-training-sft" / f"probe-{steps}step" / "training_attempt.json"
    attempt = _read_json(attempt_path)
    if attempt.get("status") != "completed" or attempt.get("candidate_eligible") is not True:
        raise SystemExit(f"eligible Phase44 adapter is missing: {attempt_path}")
    adapter_dir = dict(attempt.get("adapter_validation") or {}).get("artifact_dir")
    if not adapter_dir:
        adapter_dir = dict(attempt.get("execution") or {}).get("artifact_dir")
    if not adapter_dir:
        raise SystemExit(f"adapter path is missing in {attempt_path}")
    return Path(str(adapter_dir)).expanduser().resolve()


def _freeze_check(*, mode: str, sessions: list[dict[str, Any]]) -> dict[str, Any]:
    scorer = _read_json(SCORER_FREEZE_PATH)
    scorer_current = _sha256(SCORER_SOURCE)
    scorer_ok = scorer.get("source_sha256") == scorer_current and scorer.get("calibration_status") == "passed"
    holdout_ok = True
    expected_holdout_hash = None
    current_holdout_hash = None
    if mode == "holdout":
        holdout_freeze = _read_json(HOLDOUT_FREEZE_PATH)
        expected_holdout_hash = holdout_freeze.get("holdout_manifest_sha256")
        current_holdout_hash = stable_hash(sessions)
        holdout_ok = expected_holdout_hash == current_holdout_hash and holdout_freeze.get("frozen_before_training") is True
    return {
        "kind": "phase44_generation_freeze_check",
        "passed": scorer_ok and holdout_ok,
        "mode": mode,
        "scorer_expected_sha256": scorer.get("source_sha256"),
        "scorer_current_sha256": scorer_current,
        "scorer_passed": scorer_ok,
        "holdout_expected_sha256": expected_holdout_hash,
        "holdout_current_sha256": current_holdout_hash,
        "holdout_passed": holdout_ok,
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
    adapter_loaded = False
    if adapter_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_path), local_files_only=True)
        adapter_loaded = True
    model.to(device)
    model.eval()
    return torch, tokenizer, model, device, {
        "adapter_loaded": adapter_loaded,
        "adapter_path": str(adapter_path) if adapter_path else None,
    }


def _render_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    try:
        return str(tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        ))
    except TypeError:
        return str(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))


def _generate(
    *, torch: Any, tokenizer: Any, model: Any, device: str,
    messages: list[dict[str, str]], max_new_tokens: int,
) -> tuple[str, dict[str, Any]]:
    prompt = _render_prompt(tokenizer, messages)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    input_length = int(inputs["input_ids"].shape[-1])
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.05,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
        )
    generated = output[0][input_length:]
    raw = tokenizer.decode(generated, skip_special_tokens=True).strip()
    cleaned, think_leak = _strip_thinking(raw)
    if not cleaned:
        raise RuntimeError("real Qwen3-4B generation returned empty text")
    generated_count = int(generated.shape[-1])
    final_token = int(generated[-1].item()) if generated_count else None
    truncated = generated_count >= max_new_tokens and final_token != tokenizer.eos_token_id
    return cleaned, {
        "raw_content": raw,
        "think_leak_detected": think_leak,
        "input_tokens": input_length,
        "completion_tokens": generated_count,
        "max_new_tokens": max_new_tokens,
        "truncated": truncated,
        "latency_seconds": round(time.perf_counter() - started, 4),
    }


def _system_contract(variant: str) -> str | None:
    if variant == "runtime_v1":
        return PHASE43_RUNTIME_CONTRACT
    if variant in {"soft_runtime", "hybrid"}:
        return PHASE44_SOFT_RUNTIME_CONTRACT
    return None


def _run_session(
    *, session: Mapping[str, Any], variant: str, torch: Any, tokenizer: Any,
    model: Any, device: str, max_new_tokens: int, runtime: Mapping[str, Any],
) -> dict[str, Any]:
    model_messages: list[dict[str, str]] = []
    contract = _system_contract(variant)
    if contract:
        model_messages.append({"role": "system", "content": contract})
    turns: list[dict[str, Any]] = []
    generation: list[dict[str, Any]] = []
    user_turns = [
        str(session.get("user_goal") or ""),
        str(session.get("user_correction") or ""),
        f"{session.get('continuation_request') or ''}\n{session.get('acceptance_request') or ''}".strip(),
    ]
    for turn_index, user_text in enumerate(user_turns, start=1):
        user = {"role": "user", "content": user_text}
        model_messages.append(user)
        turns.append(user)
        answer, info = _generate(
            torch=torch, tokenizer=tokenizer, model=model, device=device,
            messages=model_messages, max_new_tokens=max_new_tokens,
        )
        assistant = {"role": "assistant", "content": answer}
        model_messages.append(assistant)
        turns.append(assistant)
        generation.append({"turn": turn_index, **info})
    return {
        "kind": "phase44_real_multiturn_transcript",
        "session_id": session.get("session_id"), "category": session.get("category"), "variant": variant,
        "model_id": str(MODEL_PATH), "adapter_path": runtime.get("adapter_path"),
        "adapter_loaded": runtime.get("adapter_loaded"), "runtime_contract": contract,
        "device": device, "actual_model_call": True, "hardcoded_response": False, "status": "completed",
        "turns": turns, "generation": generation,
        "latency_seconds": [row["latency_seconds"] for row in generation],
        "truncated_response": any(row["truncated"] for row in generation),
        "think_leak_detected": any(row["think_leak_detected"] for row in generation),
        "created_at": _utcnow(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("base", "runtime_v1", "soft_runtime", "sft", "hybrid"), required=True)
    parser.add_argument("--mode", choices=("diagnostic", "holdout"), default="holdout")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    source = _read_json(DIAGNOSTIC_PATH if args.mode == "diagnostic" else HOLDOUT_PATH)
    sessions = [dict(row) for row in source.get("sessions") or []]
    if args.limit is not None:
        sessions = sessions[:max(0, int(args.limit))]
    freeze_check = _freeze_check(mode=args.mode, sessions=sessions)
    if freeze_check["passed"] is not True:
        raise SystemExit(f"Phase44 frozen scorer/holdout check failed: {freeze_check}")

    output_dir = EVIDENCE_ROOT / "evidence-holdout" / ("diagnostic" if args.mode == "diagnostic" else "real-60-session")
    output_path = output_dir / f"transcripts_{args.variant}.jsonl"
    summary_path = output_dir / f"metrics_{args.variant}.json"
    freeze_path = output_dir / f"freeze_check_{args.variant}.json"
    if args.clean:
        output_path.unlink(missing_ok=True)
        summary_path.unlink(missing_ok=True)
    _write_json(freeze_path, freeze_check)
    existing = [] if args.clean else _read_jsonl(output_path)
    completed_ids = {str(row.get("session_id")) for row in existing if row.get("status") == "completed"}
    transcripts = [row for row in existing if str(row.get("session_id")) in {str(item.get("session_id")) for item in sessions}]

    adapter_path = _resolve_adapter(args.variant, args.steps, args.adapter_path)
    torch, tokenizer, model, device, runtime = _load_runtime(adapter_path)
    try:
        for index, session in enumerate(sessions, start=1):
            if str(session.get("session_id")) in completed_ids:
                print(f"[{args.variant}] {index}/{len(sessions)} {session['session_id']} resumed", flush=True)
                continue
            try:
                transcript = _run_session(
                    session=session, variant=args.variant, torch=torch, tokenizer=tokenizer,
                    model=model, device=device, max_new_tokens=max(24, int(args.max_new_tokens)), runtime=runtime,
                )
            except Exception as exc:
                transcript = {
                    "kind": "phase44_real_multiturn_transcript", "session_id": session.get("session_id"),
                    "category": session.get("category"), "variant": args.variant, "model_id": str(MODEL_PATH),
                    "adapter_path": runtime.get("adapter_path"), "adapter_loaded": runtime.get("adapter_loaded"),
                    "device": device, "actual_model_call": False, "hardcoded_response": False, "status": "failed",
                    "error": f"{exc.__class__.__name__}: {exc}", "turns": [], "generation": [],
                    "latency_seconds": [], "created_at": _utcnow(),
                }
            transcripts = [row for row in transcripts if row.get("session_id") != transcript.get("session_id")]
            transcripts.append(transcript)
            transcripts.sort(key=lambda row: str(row.get("session_id")))
            _write_jsonl_atomic(output_path, transcripts)
            print(f"[{args.variant}] {index}/{len(sessions)} {transcript['session_id']} {transcript['status']}", flush=True)
    finally:
        try:
            del model
            if device == "mps":
                torch.mps.empty_cache()
        except Exception:
            pass

    training_targets = [str(row.get("chosen") or "") for row in _read_jsonl(CANDIDATE_PATH)]
    report = aggregate_phase44_variant(transcripts, sessions, training_targets=training_targets)
    report.update({
        "variant": args.variant, "mode": args.mode, "model_id": str(MODEL_PATH),
        "adapter_path": str(adapter_path) if adapter_path else None, "adapter_loaded": bool(adapter_path),
        "runtime_contract": _system_contract(args.variant),
        "all_transcripts_completed": len(transcripts) == len(sessions) and all(row.get("status") == "completed" for row in transcripts),
        "transcript_path": str(output_path), "freeze_check": freeze_check,
        "model_call_count": sum(len(row.get("generation") or []) for row in transcripts if row.get("actual_model_call") is True),
        "think_leak_rate": round(sum(bool(row.get("think_leak_detected")) for row in transcripts) / len(transcripts), 4) if transcripts else 0.0,
        "actual_user_feedback": False, "simulated_usage": True,
        "actual_product_benefit_claim_allowed": False, "created_at": _utcnow(),
    })
    _write_json(summary_path, report)
    print(json.dumps({key: report.get(key) for key in (
        "variant", "session_count", "model_call_count", "actual_model_calls", "user_preference_score",
        "privacy_violation_rate", "ordinary_task_overcontract_rate", "response_diversity", "repetition_rate",
    )}, ensure_ascii=False, indent=2))
    return 0 if report.get("actual_model_calls") is True and report.get("all_transcripts_completed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
