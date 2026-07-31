#!/usr/bin/env python3
"""Close the Qwen3 native generation boundary before Phase101 training."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = REPO_ROOT / "pfe-core"
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

from pfe_core.phase75_personalization_benefit_benchmark import stable_hash
from pfe_core.phase77_private_value_guarded_runtime import guard_phase77_messages, guard_phase77_output
from pfe_core.phase85_low_fallback_semantic_guard import contract_for_phase85_messages, enforce_phase85_persona_output
from pfe_core.phase91_controlled_dpo_preference import score_phase91_output
from pfe_core.phase93_95_dpo_product_proof import aggregate_phase94_scores, has_repeated_output
from pfe_core.phase99_qwen3_native_generation_boundary import (
    forbidden_generation_hits,
    has_extra_text_after_first_answer,
    qwen3_bad_words_ids,
    qwen3_eos_token_ids,
    render_qwen3_no_think_prompt,
)
from pfe_core.phase100_qwen3_generation_boundary_closure import (
    audit_phase100_holdout,
    build_phase100_gate,
    build_phase100_generation_controls,
    build_phase100_holdout,
    phase100_answer_complete,
    phase100_runtime_contract,
)


EVIDENCE_ROOT = REPO_ROOT / "docs/demo/phase100-104-autonomous-qwen3-training-benefit-loop"
PHASE_ROOT = EVIDENCE_ROOT / "phase100-generation-boundary"
PREPARATION_ROOT = PHASE_ROOT / "evidence-preparation"
DIAGNOSTIC_ROOT = PHASE_ROOT / "evidence-diagnostic"
EVAL_ROOT = PHASE_ROOT / "evidence-eval"
PRIVATE_ROOT = Path("/private/tmp/pfe-phase100-simulated-review")
MODEL_PATH = REPO_ROOT / "models/Qwen3-4B"
GENERATION_PROTOCOL = {
    "model": "Qwen3-4B",
    "input_max_length": 3072,
    "max_new_tokens": 160,
    "do_sample": False,
    "repetition_penalty": 1.15,
    "no_repeat_ngram_size": 4,
    "enable_thinking": False,
    "bad_words_required": True,
    "premature_eos_suppression_required": True,
    "semantic_first_answer_stopping_required": True,
    "provenance_final_guided_target_required": True,
    "post_hoc_truncation_allowed": False,
    "diagnostic_call_budget": 24,
    "final_gate_call_budget": 24,
    "long_run_total_call_budget": 270,
}


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


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
    if resolved.parent != parent.resolve():
        raise RuntimeError(f"refusing to clean outside {parent}: {path}")
    if resolved.exists():
        shutil.rmtree(resolved)


def _source_hashes() -> dict[str, str]:
    paths = {
        "core": CORE_ROOT / "pfe_core/phase100_qwen3_generation_boundary_closure.py",
        "driver": REPO_ROOT / "tools/phase100_qwen3_generation_boundary_closure.py",
        "phase99_core": CORE_ROOT / "pfe_core/phase99_qwen3_native_generation_boundary.py",
        "core_test": REPO_ROOT / "tests/test_phase100_qwen3_generation_boundary_closure.py",
        "driver_test": REPO_ROOT / "tests/test_phase100_driver_safety.py",
    }
    return {name: _sha256(path) for name, path in paths.items()}


def _previous_holdouts() -> list[dict[str, Any]]:
    paths = (
        REPO_ROOT / "docs/demo/phase43-qwen3-4b-personal-preference-benefit-proof/evidence-holdout/holdout.json",
        REPO_ROOT / "docs/demo/phase87-89-failure-driven-adapter-loop/evidence-preparation/holdout.json",
        REPO_ROOT / "docs/demo/phase90-native-format-curriculum-repair/evidence-preparation/holdout.json",
        REPO_ROOT / "docs/demo/phase91-controlled-dpo-preference-diagnostic/evidence-preparation/holdout.json",
        REPO_ROOT / "docs/demo/phase96-98-qwen3-4b-capacity-ladder/phase96-capacity-diagnostic/evidence-preparation/capacity_holdout.json",
        REPO_ROOT / "docs/demo/phase99-qwen3-native-generation-boundary/evidence-preparation/holdout.json",
    )
    payloads = [_read_json(path) for path in paths if path.is_file()]
    phase93_path = REPO_ROOT / "docs/demo/phase92-95-autonomous-dpo-stability-product-proof/phase93-stable-dpo-training/evidence-preparation/fresh_holdouts.json"
    if phase93_path.is_file():
        phase93 = _read_json(phase93_path)
        payloads.append({"sessions": list(phase93.get("sanity_sessions") or []) + list(phase93.get("product_sessions") or [])})
    return payloads


def _model_integrity() -> dict[str, Any]:
    index = _read_json(MODEL_PATH / "model.safetensors.index.json")
    shards = sorted(set(dict(index.get("weight_map") or {}).values()))
    return {
        "model_path": str(MODEL_PATH),
        "shards": shards,
        "all_shards_present": all((MODEL_PATH / shard).is_file() for shard in shards),
        "total_weight_bytes": sum((MODEL_PATH / shard).stat().st_size for shard in shards),
        "config_sha256": _sha256(MODEL_PATH / "config.json"),
        "tokenizer_config_sha256": _sha256(MODEL_PATH / "tokenizer_config.json"),
    }


def _prepare(clean: bool) -> int:
    if clean and PHASE_ROOT.exists():
        _safe_clean(PHASE_ROOT, EVIDENCE_ROOT)
    PREPARATION_ROOT.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), local_files_only=True)
    diagnostic = build_phase100_holdout(scope="diagnostic")
    final = build_phase100_holdout(scope="final")
    previous = _previous_holdouts()
    diagnostic_isolation = audit_phase100_holdout(diagnostic, previous)
    final_isolation = audit_phase100_holdout(final, [*previous, diagnostic])
    probe_prompt = render_qwen3_no_think_prompt(tokenizer, [{"role": "user", "content": "boundary probe"}])
    controls = {
        "prompt_suffix": probe_prompt[-64:],
        "prompt_has_empty_think_block": probe_prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"),
        "bad_words_ids": qwen3_bad_words_ids(tokenizer),
        "eos_token_ids": qwen3_eos_token_ids(tokenizer),
        "premature_eos_suppression": True,
        "semantic_first_answer_stopping": True,
        "provenance_final_guided_target": True,
    }
    model = _model_integrity()
    checks = {
        "diagnostic_isolation_passed": diagnostic_isolation.get("passed") is True,
        "final_isolation_passed": final_isolation.get("passed") is True,
        "model_complete": model["all_shards_present"] is True,
        "diagnostic_calls_12": diagnostic.get("model_call_count") == 12,
        "final_calls_24": final.get("model_call_count") == 24,
        "no_think_prompt_verified": controls["prompt_has_empty_think_block"] is True,
        "bad_words_include_seven_sequences": len(controls["bad_words_ids"]) == 7,
        "end_tokens_available": bool(controls["eos_token_ids"]),
        "post_hoc_truncation_forbidden": GENERATION_PROTOCOL["post_hoc_truncation_allowed"] is False,
    }
    freeze = {
        "kind": "phase100_pre_generation_freeze",
        "created_at": _utcnow(),
        "passed": all(checks.values()),
        "checks": checks,
        "model": model,
        "diagnostic_manifest_sha256": diagnostic["manifest_sha256"],
        "final_manifest_sha256": final["manifest_sha256"],
        "generation_protocol": GENERATION_PROTOCOL,
        "generation_protocol_sha256": stable_hash(GENERATION_PROTOCOL),
        "generation_controls": controls,
        "source_sha256": _source_hashes(),
        "simulated_usage": True,
        "actual_user_feedback_count": 0,
        "automatic_promotion_allowed": False,
    }
    _write_json(PREPARATION_ROOT / "diagnostic_holdout.json", diagnostic)
    _write_json(PREPARATION_ROOT / "final_holdout.json", final)
    _write_json(PREPARATION_ROOT / "diagnostic_isolation_audit.json", diagnostic_isolation)
    _write_json(PREPARATION_ROOT / "final_isolation_audit.json", final_isolation)
    _write_json(PREPARATION_ROOT / "generation_controls.json", controls)
    _write_json(PREPARATION_ROOT / "model_integrity.json", model)
    _write_json(PHASE_ROOT / "pre_generation_freeze.json", freeze)
    print(json.dumps({"status": "ready" if freeze["passed"] else "blocked", "checks": checks}, ensure_ascii=False, indent=2))
    return 0 if freeze["passed"] else 2


def _load_runtime() -> tuple[Any, Any, Any, str]:
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
    model.to(device)
    model.eval()
    return torch, tokenizer, model, device


def _generate_one(
    *,
    torch: Any,
    tokenizer: Any,
    model: Any,
    device: str,
    messages: list[dict[str, str]],
    session: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    prompt = render_qwen3_no_think_prompt(tokenizer, messages)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=int(GENERATION_PROTOCOL["input_max_length"]))
    inputs = {name: value.to(device) for name, value in inputs.items()}
    input_length = int(inputs["input_ids"].shape[-1])
    eos_ids = qwen3_eos_token_ids(tokenizer)
    logits_processor, stopping_criteria, state = build_phase100_generation_controls(
        tokenizer=tokenizer,
        input_length=input_length,
        session=session,
        eos_token_ids=eos_ids,
    )
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=int(GENERATION_PROTOCOL["max_new_tokens"]),
            do_sample=False,
            repetition_penalty=float(GENERATION_PROTOCOL["repetition_penalty"]),
            no_repeat_ngram_size=int(GENERATION_PROTOCOL["no_repeat_ngram_size"]),
            bad_words_ids=qwen3_bad_words_ids(tokenizer),
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=eos_ids,
        )
    generated = output[0][input_length:]
    raw = tokenizer.decode(generated, skip_special_tokens=True).strip()
    if not raw:
        raise RuntimeError("Phase100 Qwen3 generation returned empty output")
    last_token = int(generated[-1].item()) if int(generated.shape[-1]) else None
    if state["stopping_triggered"]:
        termination_reason = "semantic_first_answer_boundary"
    elif last_token in eos_ids:
        termination_reason = "model_eos"
    elif int(generated.shape[-1]) >= int(GENERATION_PROTOCOL["max_new_tokens"]):
        termination_reason = "max_new_tokens"
    else:
        termination_reason = "unknown"
    complete = phase100_answer_complete(raw, session)
    return raw, {
        "input_tokens": input_length,
        "completion_tokens": int(generated.shape[-1]),
        "latency_seconds": round(time.perf_counter() - started, 4),
        "termination_reason": termination_reason,
        "stopping_criteria_triggered": state["stopping_triggered"],
        "premature_eos_suppression_count": state["eos_suppression_count"],
        "guided_token_count": state["guided_token_count"],
        "complete_content_before_termination": complete,
        "native_termination": termination_reason in {"semantic_first_answer_boundary", "model_eos"} and complete,
        "extra_text_after_first_answer": has_extra_text_after_first_answer(raw, format_expected=session.get("format_expected") is True),
        "forbidden_generation_hits": forbidden_generation_hits(raw),
        "prompt_no_think_verified": prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"),
        "post_hoc_truncation_used": False,
    }


def _run_session(
    *,
    session: Mapping[str, Any],
    torch: Any,
    tokenizer: Any,
    model: Any,
    device: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    history: list[dict[str, str]] = []
    private_values = [str(value) for value in session.get("declared_private_values") or []]
    turns = []
    private_turns = []
    final_raw = ""
    final_post = ""
    final_contract: dict[str, Any] = {}
    for turn, user_text in enumerate(session.get("user_turns") or [], start=1):
        guarded_user, input_guard = guard_phase77_messages([{"role": "user", "content": str(user_text)}], private_values)
        history.append(dict(guarded_user[0]))
        _, route = contract_for_phase85_messages(history)
        runtime_contract = phase100_runtime_contract(session)
        prompt_messages = [{"role": "system", "content": runtime_contract}, *history]
        generation_session = dict(session)
        generation_session["semantic_boundary_required"] = turn == len(session.get("user_turns") or [])
        raw, generation = _generate_one(
            torch=torch,
            tokenizer=tokenizer,
            model=model,
            device=device,
            messages=prompt_messages,
            session=generation_session,
        )
        raw_private_echo = any(value and value in raw for value in private_values)
        guarded_raw, output_guard = guard_phase77_output(raw, private_values)
        post, contract = enforce_phase85_persona_output(guarded_raw, messages=history, declared_private_values=private_values)
        history.append({"role": "assistant", "content": post})
        turns.append({
            "turn": turn,
            "routed": route.get("routed") is True,
            "input_guard_passed": input_guard.get("blocked") is not True,
            "output_guard_passed": output_guard.get("blocked") is not True,
            "raw_output_sha256": hashlib.sha256(guarded_raw.encode()).hexdigest(),
            "post_output_sha256": hashlib.sha256(post.encode()).hexdigest(),
            "privacy_echo_detected": raw_private_echo,
            **generation,
        })
        private_turns.append({"turn": turn, "raw_output": guarded_raw, "post_output": post})
        final_raw = guarded_raw
        final_post = post
        final_contract = dict(contract)
    raw_score = score_phase91_output(final_raw, session)
    post_score = score_phase91_output(final_post, session)
    raw_score["repeated_output"] = has_repeated_output(final_raw)
    post_score["repeated_output"] = has_repeated_output(final_post)
    raw_score["latency_seconds"] = round(sum(float(row["latency_seconds"]) for row in turns), 4)
    post_score["latency_seconds"] = raw_score["latency_seconds"]
    raw_score["extra_text_after_first_answer"] = has_extra_text_after_first_answer(
        final_raw, format_expected=session.get("format_expected") is True
    )
    raw_score["forbidden_generation"] = bool(forbidden_generation_hits(final_raw))
    structural = {
        "kind": "phase100_structural_session",
        "session_id": session.get("session_id"),
        "category": session.get("category"),
        "status": "completed",
        "actual_model_call": True,
        "turn_count": len(turns),
        "turns": turns,
        "raw_score": raw_score,
        "post_score": post_score,
        "final_fallback_used": final_contract.get("fallback_used") is True,
        "raw_output_persisted": False,
        "post_hoc_truncation_used": False,
        "simulated_usage": True,
        "actual_user_feedback": False,
    }
    private = {
        "session_id": session.get("session_id"),
        "category": session.get("category"),
        "turns": private_turns,
    }
    return structural, private


def _freeze_check(scope: str, output_root: Path) -> dict[str, Any]:
    freeze = _read_json(PHASE_ROOT / "pre_generation_freeze.json")
    holdout = _read_json(PREPARATION_ROOT / f"{scope}_holdout.json")
    expected_hash = freeze.get(f"{scope}_manifest_sha256")
    checks = {
        "pre_generation_freeze_passed": freeze.get("passed") is True,
        "source_files_unchanged": _source_hashes() == freeze.get("source_sha256"),
        "holdout_unchanged": stable_hash(holdout.get("sessions") or []) == expected_hash,
        "generation_protocol_unchanged": stable_hash(GENERATION_PROTOCOL) == freeze.get("generation_protocol_sha256"),
        "no_completed_eval_exists": not (output_root / "metrics.json").exists(),
    }
    return {"kind": "phase100_generation_freeze_check", "scope": scope, "passed": all(checks.values()), "checks": checks}


def _run_scope(scope: str, *, output_root: Path, cache_path: Path, clean: bool) -> int:
    if clean and output_root.exists():
        _safe_clean(output_root, output_root.parent)
    if clean:
        cache_path.unlink(missing_ok=True)
    freeze = _freeze_check(scope, output_root)
    _write_json(output_root / "freeze_check.json", freeze)
    if not freeze["passed"]:
        return 2
    sessions = [dict(row) for row in _read_json(PREPARATION_ROOT / f"{scope}_holdout.json").get("sessions") or []]
    rows: list[dict[str, Any]] = []
    private_rows: list[dict[str, Any]] = []
    torch = tokenizer = model = device = None
    try:
        torch, tokenizer, model, device = _load_runtime()
        for index, session in enumerate(sessions, start=1):
            try:
                structural, private = _run_session(session=session, torch=torch, tokenizer=tokenizer, model=model, device=device)
            except Exception as exc:
                structural = {
                    "kind": "phase100_structural_session",
                    "session_id": session.get("session_id"),
                    "category": session.get("category"),
                    "status": "failed",
                    "actual_model_call": False,
                    "error_type": exc.__class__.__name__,
                    "raw_output_persisted": False,
                    "simulated_usage": True,
                    "actual_user_feedback": False,
                }
                private = {"session_id": session.get("session_id"), "category": session.get("category"), "error": f"{exc.__class__.__name__}: {exc}"}
            rows.append(structural)
            private_rows.append(private)
            _write_jsonl(output_root / "structural_sessions.jsonl", rows)
            _write_private_jsonl(cache_path, private_rows)
            print(f"[phase100:{scope}] {index}/{len(sessions)} {session.get('session_id')} {structural['status']}", flush=True)
    finally:
        if torch is not None and model is not None and device is not None:
            del model
            if device == "mps":
                torch.mps.empty_cache()
    completed = [row for row in rows if row.get("status") == "completed"]
    details = [{"category": row.get("category"), **dict(row.get("raw_score") or {})} for row in completed]
    turns = [turn for row in completed for turn in row.get("turns") or []]
    metrics = aggregate_phase94_scores(details)
    metrics.update({
        "extra_text_after_first_answer_rate": round(sum(row.get("extra_text_after_first_answer") is True for row in details) / len(details), 4) if details else 0.0,
        "forbidden_generation_rate": round(sum(row.get("forbidden_generation") is True for row in details) / len(details), 4) if details else 0.0,
        "complete_content_before_termination_rate": round(sum(row.get("complete_content_before_termination") is True for row in turns) / len(turns), 4) if turns else 0.0,
        "native_termination_rate": round(sum(row.get("native_termination") is True for row in turns) / len(turns), 4) if turns else 0.0,
        "premature_eos_suppression_count": sum(int(row.get("premature_eos_suppression_count") or 0) for row in turns),
        "max_new_tokens_termination_count": sum(row.get("termination_reason") == "max_new_tokens" for row in turns),
    })
    gate = build_phase100_gate(metrics, expected_sessions=len(sessions))
    payload = {
        "kind": "phase100_variant_metrics",
        "scope": scope,
        "session_count": len(completed),
        "model_call_count": sum(int(row.get("turn_count") or 0) for row in completed),
        "all_sessions_completed": len(completed) == len(sessions),
        "raw": metrics,
        "gate": gate,
        "private_cache": str(cache_path),
        "private_cache_outside_repo": True,
        "post_hoc_truncation_used": False,
        "simulated_usage": True,
        "actual_user_feedback_count": 0,
    }
    _write_json(output_root / "metrics.json", payload)
    _write_json(output_root / "gate.json", gate)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["all_sessions_completed"] and gate["passed"] else 1


def _diagnose(attempt: int, clean: bool) -> int:
    if attempt not in (1, 2):
        raise SystemExit("Phase100 permits diagnostic attempts 1 or 2 only")
    output_root = DIAGNOSTIC_ROOT / f"attempt-{attempt}"
    cache_path = PRIVATE_ROOT / f"diagnostic-attempt-{attempt}.jsonl"
    return _run_scope("diagnostic", output_root=output_root, cache_path=cache_path, clean=clean)


def _generate(clean: bool) -> int:
    diagnostic_paths = sorted(DIAGNOSTIC_ROOT.glob("attempt-*/gate.json"))
    diagnostic_passed = any(_read_json(path).get("passed") is True for path in diagnostic_paths)
    diagnostics_exhausted = len(diagnostic_paths) == 2
    if not diagnostic_passed and not diagnostics_exhausted:
        print("Phase100 final gate requires a passing diagnostic or both frozen diagnostic attempts", file=sys.stderr)
        return 2
    return _run_scope("final", output_root=EVAL_ROOT, cache_path=PRIVATE_ROOT / "final-gate.jsonl", clean=clean)


def _decide() -> int:
    payload = _read_json(EVAL_ROOT / "metrics.json")
    metrics = dict(payload.get("raw") or {})
    decision = build_phase100_gate(metrics, expected_sessions=8)
    decision.update({
        "metrics": metrics,
        "model_call_count": payload.get("model_call_count"),
        "phase100_model_call_count": sum(
            int(_read_json(path).get("model_call_count") or 0)
            for path in [*DIAGNOSTIC_ROOT.glob("attempt-*/metrics.json"), EVAL_ROOT / "metrics.json"]
            if path.is_file()
        ),
        "phase100_model_call_budget_maximum": 48,
        "post_hoc_truncation_used": False,
        "diagnostic_attempts_completed": len(list(DIAGNOSTIC_ROOT.glob("attempt-*/metrics.json"))),
    })
    _write_json(PHASE_ROOT / "phase100-decision.json", decision)
    lines = [
        "# Phase100 Decision",
        "",
        f"- Status: `{decision['status']}`",
        f"- Passed: {str(decision['passed']).lower()}",
        f"- Phase100 local model calls: {decision['phase100_model_call_count']}/48",
        "- Post-hoc truncation used: false",
        "- Product gate qualified: false",
        "- Evidence: simulated usage only",
    ]
    (PHASE_ROOT / "phase100-decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return 0 if decision["passed"] else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--clean", action="store_true")
    diagnose = sub.add_parser("diagnose")
    diagnose.add_argument("--attempt", type=int, required=True)
    diagnose.add_argument("--clean", action="store_true")
    generate = sub.add_parser("generate")
    generate.add_argument("--clean", action="store_true")
    sub.add_parser("decide")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "prepare":
        return _prepare(args.clean)
    if args.command == "diagnose":
        return _diagnose(args.attempt, args.clean)
    if args.command == "generate":
        return _generate(args.clean)
    if args.command == "decide":
        return _decide()
    raise SystemExit("unsupported command")


if __name__ == "__main__":
    raise SystemExit(main())
