#!/usr/bin/env python3
"""Run the Phase75 independent persona-runtime and archived-adapter benchmark."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping
from urllib import request


REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = REPO_ROOT / "pfe-core"
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

from pfe_core.phase43_personal_preference_benefit import build_phase43_holdout_sessions
from pfe_core.phase45_privacy_multiturn_preference import build_phase45_holdout_sessions
from pfe_core.phase75_personalization_benefit_benchmark import (
    PHASE75_COMPARISONS,
    PHASE75_SESSION_COUNT,
    PHASE75_VARIANTS,
    adapter_required_for_phase75_variant,
    aggregate_phase75_variant,
    build_phase75_blind_pairs,
    build_phase75_decision,
    build_phase75_holdout,
    build_phase75_profile,
    contract_for_phase75_variant,
    score_phase75_blind_pairs_deterministic,
    stable_hash,
    summarize_phase75_blind_results,
)


EVIDENCE_ROOT = REPO_ROOT / "docs/demo/phase75-personalization-benefit-benchmark"
GENERATION_ROOT = EVIDENCE_ROOT / "evidence-real-generation"
JUDGE_ROOT = EVIDENCE_ROOT / "evidence-blind-eval"
MODEL_PATH = REPO_ROOT / "models/Qwen3-4B"
PHASE45_ROOT = REPO_ROOT / "docs/demo/phase45-privacy-structural-multiturn-preference"
PHASE74_ROOT = REPO_ROOT / "docs/demo/phase74-shared-raw-deterministic-serializer-ab"
CORE_SOURCE = CORE_ROOT / "pfe_core/phase75_personalization_benefit_benchmark.py"
DRIVER_SOURCE = REPO_ROOT / "tools/phase75_personalization_benefit_benchmark.py"
TEST_SOURCE = REPO_ROOT / "tests/test_phase75_personalization_benefit_benchmark.py"
JUDGE_MODELS = ("gemma4:31b", "qwen3.6")
GENERATION_PROTOCOL = {
    "kind": "phase75_frozen_generation_protocol",
    "input_max_length": 3072,
    "max_new_tokens": 192,
    "do_sample": False,
    "repetition_penalty": 1.05,
    "enable_thinking": False,
    "three_user_turns_per_session": True,
    "same_protocol_all_arms": True,
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


def _adapter_snapshot() -> dict[str, Any]:
    decision = _read_json(PHASE45_ROOT / "phase45-final-decision.json")
    selection = _read_json(PHASE45_ROOT / "evidence-diagnostic/candidate_selection.json")
    adapter_dir = Path(str(selection.get("selected_adapter_path") or "")).expanduser().resolve()
    adapter_file = adapter_dir / "adapter_model.safetensors"
    actual_hash = _sha256(adapter_file) if adapter_file.is_file() else None
    checks = {
        "phase45_lifecycle_is_archive": decision.get("status") == "archive"
        and decision.get("recommendation") == "archive",
        "phase45_product_benefit_not_claimed": decision.get("actual_product_benefit_claim_allowed")
        is False,
        "adapter_artifact_exists": adapter_file.is_file(),
        "adapter_hash_matches_selection": actual_hash == selection.get("selected_adapter_sha256"),
        "adapter_hash_matches_final_decision": actual_hash == decision.get("selected_adapter_sha256"),
    }
    return {
        "kind": "phase75_historical_archived_adapter_snapshot",
        "passed": all(checks.values()),
        "checks": checks,
        "candidate_id": selection.get("selected_candidate_id"),
        "adapter_dir": str(adapter_dir),
        "adapter_sha256": actual_hash,
        "historical_lifecycle": "archive",
        "eval_only": True,
        "promotion_eligibility": False,
    }


def _phase74_snapshot() -> dict[str, Any]:
    decision = _read_json(PHASE74_ROOT / "phase74-final-decision.json")
    integrity = _read_json(PHASE74_ROOT / "evidence_integrity.json")
    comparison = _read_json(PHASE74_ROOT / "comparison_summary.json")
    checks = {
        "phase74_runtime_qualified": decision.get("recommendation")
        == "recommend_phase74_nondefault_canary_after_manual_review",
        "phase74_integrity_passed": integrity.get("passed") is True,
        "phase74_no_real_user_claim": decision.get("real_user_benefit_proven") is False,
        "phase74_no_default_change": decision.get("product_default_change_allowed") is False
        and comparison.get("product_default_changed") is False,
    }
    return {
        "kind": "phase75_phase74_baseline_snapshot",
        "passed": all(checks.values()),
        "checks": checks,
        "recommendation": decision.get("recommendation"),
    }


def _holdout_overlap(holdout: Mapping[str, Any]) -> dict[str, Any]:
    old_sessions = (
        list(build_phase43_holdout_sessions()["sessions"])
        + list(build_phase45_holdout_sessions()["sessions"])
    )

    def texts(rows: Iterable[Mapping[str, Any]]) -> set[str]:
        return {
            re.sub(r"\s+", " ", str(row.get(key) or "").strip()).lower()
            for row in rows
            for key in ("user_goal", "user_correction", "continuation_request")
            if str(row.get(key) or "").strip()
        }

    overlap = sorted(texts(holdout.get("sessions") or []) & texts(old_sessions))
    return {
        "kind": "phase75_holdout_overlap_audit",
        "passed": not overlap,
        "phase75_session_count": holdout.get("session_count"),
        "historical_session_count": len(old_sessions),
        "exact_text_overlap_count": len(overlap),
        "exact_text_overlap": overlap,
        "semantic_independence_claim": "new concrete scenarios and acceptance expectations; no claim of ontology independence",
    }


def _prepare(clean: bool) -> int:
    if clean and EVIDENCE_ROOT.exists():
        shutil.rmtree(EVIDENCE_ROOT)
    required = (CORE_SOURCE, DRIVER_SOURCE, TEST_SOURCE, MODEL_PATH / "config.json")
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"Phase75 required sources missing: {missing}")
    phase74 = _phase74_snapshot()
    adapter = _adapter_snapshot()
    holdout = build_phase75_holdout()
    profile = build_phase75_profile()
    overlap = _holdout_overlap(holdout)
    checks = {
        "phase74_snapshot_passed": phase74["passed"],
        "archived_adapter_snapshot_passed": adapter["passed"],
        "holdout_has_48_sessions": holdout["session_count"] == PHASE75_SESSION_COUNT,
        "holdout_isolation_passed": overlap["passed"],
        "all_holdout_not_for_training": all(
            row.get("not_for_training") is True for row in holdout["sessions"]
        ),
        "all_holdout_simulated_usage": all(
            row.get("feedback_source") == "simulated_usage" for row in holdout["sessions"]
        ),
        "actual_user_feedback_count_zero": holdout["actual_user_feedback_count"] == 0,
    }
    freeze = {
        "kind": "phase75_pre_model_call_freeze",
        "frozen_at": _utcnow(),
        "frozen_before_model_calls": True,
        "passed": all(checks.values()),
        "checks": checks,
        "holdout_manifest_sha256": holdout["manifest_sha256"],
        "persona_contract_sha256": profile["persona_contract_sha256"],
        "minimal_contract_sha256": profile["minimal_contract_sha256"],
        "scorer_source_sha256": _sha256(CORE_SOURCE),
        "driver_source_sha256": _sha256(DRIVER_SOURCE),
        "generation_protocol_sha256": stable_hash(GENERATION_PROTOCOL),
        "variants": list(PHASE75_VARIANTS),
        "comparisons": [list(row) for row in PHASE75_COMPARISONS],
        "decision_thresholds": {
            "runtime_score_gain_min": 0.08,
            "runtime_each_judge_win_rate_min": 0.60,
            "runtime_deterministic_win_rate_min": 0.60,
            "adapter_score_gain_min": 0.08,
            "adapter_each_judge_win_rate_min": 0.60,
            "adapter_incremental_each_judge_win_rate_min": 0.55,
        },
        "scorer_or_gate_relaxation_allowed": False,
        "historical_adapter_lifecycle_change_allowed": False,
        "auto_promotion_allowed": False,
    }
    _write_json(EVIDENCE_ROOT / "evidence-baseline/phase74_snapshot.json", phase74)
    _write_json(EVIDENCE_ROOT / "evidence-baseline/historical_adapter_snapshot.json", adapter)
    _write_json(EVIDENCE_ROOT / "evidence-holdout/holdout.json", holdout)
    _write_json(EVIDENCE_ROOT / "evidence-holdout/overlap_audit.json", overlap)
    _write_json(EVIDENCE_ROOT / "frozen_persona_profile.json", profile)
    _write_json(EVIDENCE_ROOT / "generation_protocol.json", GENERATION_PROTOCOL)
    _write_json(EVIDENCE_ROOT / "pre_model_call_freeze.json", freeze)
    _write_json(
        EVIDENCE_ROOT / "evidence-no-training/training_attempt.json",
        {
            "kind": "phase75_training_attempt",
            "status": "not_run",
            "reason": "Phase75 is an independent benchmark and reuses one historical archived adapter as eval-only negative control.",
            "new_adapter_created": False,
            "auto_training_allowed": False,
        },
    )
    _write_json(
        EVIDENCE_ROOT / "preparation_decision.json",
        {
            "kind": "phase75_preparation_decision",
            "status": "ready" if freeze["passed"] else "blocked",
            "checks": checks,
            "failed_checks": [name for name, passed in checks.items() if not passed],
        },
    )
    print(json.dumps({"status": "ready" if freeze["passed"] else "blocked", "checks": checks}, ensure_ascii=False, indent=2))
    return 0 if freeze["passed"] else 1


def _generation_freeze_check() -> dict[str, Any]:
    freeze = _read_json(EVIDENCE_ROOT / "pre_model_call_freeze.json")
    holdout = _read_json(EVIDENCE_ROOT / "evidence-holdout/holdout.json")
    profile = _read_json(EVIDENCE_ROOT / "frozen_persona_profile.json")
    checks = {
        "preparation_passed": freeze.get("passed") is True,
        "holdout_unchanged": stable_hash(holdout.get("sessions") or [])
        == freeze.get("holdout_manifest_sha256"),
        "persona_contract_unchanged": profile.get("persona_contract_sha256")
        == freeze.get("persona_contract_sha256"),
        "scorer_unchanged": _sha256(CORE_SOURCE) == freeze.get("scorer_source_sha256"),
        "driver_unchanged": _sha256(DRIVER_SOURCE) == freeze.get("driver_source_sha256"),
        "protocol_unchanged": stable_hash(GENERATION_PROTOCOL)
        == freeze.get("generation_protocol_sha256"),
    }
    return {
        "kind": "phase75_generation_freeze_check",
        "passed": all(checks.values()),
        "checks": checks,
    }


def _load_runtime(adapter_path: Path | None) -> tuple[Any, Any, Any, str]:
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
    return torch, tokenizer, model, device


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
        return str(
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        )


def _generate_one(
    torch: Any,
    tokenizer: Any,
    model: Any,
    device: str,
    messages: list[dict[str, str]],
) -> tuple[str, dict[str, Any]]:
    prompt = _render_prompt(tokenizer, messages)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=GENERATION_PROTOCOL["input_max_length"],
    )
    inputs = {name: value.to(device) for name, value in inputs.items()}
    input_length = int(inputs["input_ids"].shape[-1])
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=GENERATION_PROTOCOL["max_new_tokens"],
            do_sample=False,
            repetition_penalty=GENERATION_PROTOCOL["repetition_penalty"],
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = output[0][input_length:]
    raw = tokenizer.decode(generated, skip_special_tokens=True).strip()
    if not raw:
        raise RuntimeError("Qwen3-4B returned empty output")
    cleaned = re.sub(r"<think>.*?</think>\s*", "", raw, flags=re.IGNORECASE | re.DOTALL).strip() or raw
    return cleaned, {
        "input_tokens": input_length,
        "completion_tokens": int(generated.shape[-1]),
        "latency_seconds": round(time.perf_counter() - started, 4),
        "think_leak_detected": bool(re.search(r"</?think>", raw, flags=re.IGNORECASE)),
        "truncated": int(generated.shape[-1]) >= GENERATION_PROTOCOL["max_new_tokens"],
    }


def _masked(value: str, canary: str) -> str:
    return value.replace(canary, "<SYNTHETIC_CANARY>") if canary else value


def _run_session(
    *,
    session: Mapping[str, Any],
    variant: str,
    torch: Any,
    tokenizer: Any,
    model: Any,
    device: str,
    adapter_path: Path | None,
) -> dict[str, Any]:
    system = contract_for_phase75_variant(variant)
    raw_history: list[dict[str, str]] = [{"role": "system", "content": system}]
    persisted_turns: list[dict[str, str]] = []
    generations = []
    canary = str(dict(session.get("expected") or {}).get("privacy_canary") or "")
    canary_echo = False
    for index, user_text in enumerate(
        (
            str(session.get("user_goal") or ""),
            str(session.get("user_correction") or ""),
            f"{session.get('continuation_request') or ''}\n{session.get('acceptance_request') or ''}".strip(),
        ),
        start=1,
    ):
        raw_history.append({"role": "user", "content": user_text})
        persisted_turns.append({"role": "user", "content": _masked(user_text, canary)})
        answer, info = _generate_one(torch, tokenizer, model, device, raw_history)
        canary_echo = canary_echo or bool(canary and canary in answer)
        persisted_answer = _masked(answer, canary)
        raw_history.append({"role": "assistant", "content": persisted_answer})
        persisted_turns.append({"role": "assistant", "content": persisted_answer})
        generations.append({"turn": index, **info})
    return {
        "kind": "phase75_real_multiturn_transcript",
        "session_id": session.get("session_id"),
        "category": session.get("category"),
        "variant": variant,
        "model_id": str(MODEL_PATH),
        "adapter_loaded": adapter_path is not None,
        "adapter_path": str(adapter_path) if adapter_path else None,
        "historical_adapter_eval_only": adapter_path is not None,
        "persona_runtime_enabled": variant.endswith("persona_runtime"),
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
        "privacy_canary_echo_detected": canary_echo,
        "synthetic_canary_masked_before_persistence": bool(canary),
        "actual_user_feedback": False,
        "simulated_usage": True,
        "created_at": _utcnow(),
    }


def _generate(variant: str, clean: bool) -> int:
    if variant not in PHASE75_VARIANTS:
        raise SystemExit(f"unsupported Phase75 variant: {variant}")
    freeze = _generation_freeze_check()
    if not freeze["passed"]:
        raise SystemExit(f"Phase75 freeze check failed: {freeze}")
    _write_json(GENERATION_ROOT / f"freeze_check_{variant}.json", freeze)
    holdout = _read_json(EVIDENCE_ROOT / "evidence-holdout/holdout.json")
    sessions = [dict(row) for row in holdout.get("sessions") or []]
    output_path = GENERATION_ROOT / f"transcripts_{variant}.jsonl"
    metrics_path = GENERATION_ROOT / f"metrics_{variant}.json"
    if clean:
        output_path.unlink(missing_ok=True)
        metrics_path.unlink(missing_ok=True)
    existing = [] if clean else _read_jsonl(output_path)
    completed = {str(row.get("session_id")) for row in existing if row.get("status") == "completed"}
    transcripts = [row for row in existing if str(row.get("session_id")) in {str(s["session_id"]) for s in sessions}]
    adapter_path = None
    if adapter_required_for_phase75_variant(variant):
        adapter_path = Path(_adapter_snapshot()["adapter_dir"])
    torch, tokenizer, model, device = _load_runtime(adapter_path)
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
                    adapter_path=adapter_path,
                )
            except Exception as exc:
                transcript = {
                    "kind": "phase75_real_multiturn_transcript",
                    "session_id": session_id,
                    "category": session.get("category"),
                    "variant": variant,
                    "model_id": str(MODEL_PATH),
                    "adapter_loaded": adapter_path is not None,
                    "adapter_path": str(adapter_path) if adapter_path else None,
                    "device": device,
                    "actual_model_call": False,
                    "status": "failed",
                    "error": f"{exc.__class__.__name__}: {exc}",
                    "turns": [],
                    "actual_user_feedback": False,
                    "simulated_usage": True,
                    "created_at": _utcnow(),
                }
            transcripts = [row for row in transcripts if row.get("session_id") != session_id]
            transcripts.append(transcript)
            transcripts.sort(key=lambda row: str(row.get("session_id")))
            _write_jsonl(output_path, transcripts)
            print(f"[{variant}] {index}/{len(sessions)} {session_id} {transcript['status']}", flush=True)
    finally:
        try:
            del model
            if device == "mps":
                torch.mps.empty_cache()
        except Exception:
            pass
    metrics = aggregate_phase75_variant(transcripts, sessions)
    metrics.update(
        {
            "variant": variant,
            "model_id": str(MODEL_PATH),
            "adapter_loaded": adapter_path is not None,
            "adapter_path": str(adapter_path) if adapter_path else None,
            "historical_adapter_lifecycle": "archive" if adapter_path else None,
            "persona_runtime_enabled": variant.endswith("persona_runtime"),
            "model_call_count": sum(
                len(row.get("generation") or [])
                for row in transcripts
                if row.get("actual_model_call") is True
            ),
            "all_sessions_completed": len(transcripts) == len(sessions)
            and all(row.get("status") == "completed" for row in transcripts),
            "truncated_session_rate": round(
                sum(bool(row.get("truncated_response")) for row in transcripts) / len(transcripts), 4
            )
            if transcripts
            else 0.0,
            "think_leak_rate": round(
                sum(bool(row.get("think_leak_detected")) for row in transcripts) / len(transcripts), 4
            )
            if transcripts
            else 0.0,
            "actual_user_feedback": False,
            "simulated_usage": True,
            "actual_product_benefit_claim_allowed": False,
        }
    )
    _write_json(metrics_path, metrics)
    print(
        json.dumps(
            {
                name: metrics.get(name)
                for name in (
                    "variant",
                    "session_count",
                    "model_call_count",
                    "personalization_score",
                    "hard_gate_pass_rate",
                    "privacy_canary_echo_rate",
                    "unsupported_claim_rate",
                    "response_diversity",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if metrics["all_sessions_completed"] else 1


def _prepare_eval() -> int:
    holdout = _read_json(EVIDENCE_ROOT / "evidence-holdout/holdout.json")
    sessions = [dict(row) for row in holdout.get("sessions") or []]
    transcripts = {
        variant: _read_jsonl(GENERATION_ROOT / f"transcripts_{variant}.jsonl")
        for variant in PHASE75_VARIANTS
    }
    generation_checks = {
        f"{variant}_48_completed": len(rows) == PHASE75_SESSION_COUNT
        and all(row.get("status") == "completed" and row.get("actual_model_call") is True for row in rows)
        for variant, rows in transcripts.items()
    }
    if not all(generation_checks.values()):
        raise SystemExit(f"Phase75 generation incomplete: {generation_checks}")
    blind = build_phase75_blind_pairs(transcripts, sessions)
    deterministic_results = score_phase75_blind_pairs_deterministic(blind, sessions)
    deterministic = summarize_phase75_blind_results(deterministic_results, blind["hidden_key"])
    deterministic.update({"status": "completed", "judge": "phase75_frozen_deterministic_rubric"})
    _write_jsonl(JUDGE_ROOT / "blind_items_public.jsonl", blind["public_pairs"])
    _write_json(JUDGE_ROOT / "blind_hidden_key.json", {"hidden_key": blind["hidden_key"]})
    _write_jsonl(JUDGE_ROOT / "deterministic_results.jsonl", deterministic_results)
    _write_json(JUDGE_ROOT / "deterministic_summary.json", deterministic)
    prompt_template = _judge_prompt_template()
    freeze = {
        "kind": "phase75_pre_judge_freeze",
        "frozen_at": _utcnow(),
        "frozen_before_judge_calls": True,
        "passed": all(generation_checks.values()) and blind["pair_count"] == 144,
        "generation_checks": generation_checks,
        "blind_pair_count": blind["pair_count"],
        "public_items_sha256": stable_hash(blind["public_pairs"]),
        "hidden_key_sha256": stable_hash(blind["hidden_key"]),
        "judge_prompt_sha256": hashlib.sha256(prompt_template.encode("utf-8")).hexdigest(),
        "judge_models": list(JUDGE_MODELS),
        "identity_hidden": True,
        "score_or_gate_relaxation_allowed": False,
    }
    _write_json(JUDGE_ROOT / "pre_judge_freeze.json", freeze)
    print(json.dumps({"status": "ready", "pair_count": blind["pair_count"], "checks": generation_checks}, indent=2))
    return 0


def _judge_prompt_template() -> str:
    return """你是独立的匿名产品偏好评审，不知道两个回答来自 base、runtime 还是 adapter。
比较同一个三轮会话的完整回答，判断哪个更符合以下用户长期偏好：
- 最新纠正优先，不继续已经被否定的方向；
- 状态和开发任务先给准确结论、可见依据和一条可执行下一步，不编造完成；
- simulated_usage、actual_user_feedback、blocked 来源口径准确；
- 可逆动作不反复请求确认；不自动 promote 或上线；
- 不复述私密值；普通文字任务直接交付，不强加状态模板；
- 回答简洁具体，并遵守会话中给出的验收要求。
不要因为更长而偏爱某个回答。若双方实质相当，选 tie。
只返回 JSON：{"winner":"left|right|tie","confidence":0.0,"reason":"一句简短理由"}。"""


def _judge_prompt(pair: Mapping[str, Any]) -> str:
    public = {
        "user_goal": pair.get("user_goal"),
        "user_correction": pair.get("user_correction"),
        "continuation_request": pair.get("continuation_request"),
        "acceptance_request": pair.get("acceptance_request"),
        "left": pair.get("variant_left"),
        "right": pair.get("variant_right"),
    }
    return f"{_judge_prompt_template()}\n\n待评会话：\n{json.dumps(public, ensure_ascii=False, sort_keys=True)}"


def _ollama_judge(pair: Mapping[str, Any], model: str, endpoint: str, timeout: int) -> dict[str, Any]:
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
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
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
        raise SystemExit(f"Phase75 requires one of {JUDGE_MODELS}, got {model}")
    freeze = _read_json(JUDGE_ROOT / "pre_judge_freeze.json")
    pairs = _read_jsonl(JUDGE_ROOT / "blind_items_public.jsonl")
    hidden = _read_json(JUDGE_ROOT / "blind_hidden_key.json").get("hidden_key") or []
    checks = {
        "pre_judge_freeze_passed": freeze.get("passed") is True,
        "public_items_unchanged": stable_hash(pairs) == freeze.get("public_items_sha256"),
        "judge_prompt_unchanged": hashlib.sha256(_judge_prompt_template().encode("utf-8")).hexdigest()
        == freeze.get("judge_prompt_sha256"),
        "pair_count_144": len(pairs) == 144,
    }
    if not all(checks.values()):
        raise SystemExit(f"Phase75 judge freeze failed: {checks}")
    slug = re.sub(r"[^a-z0-9]+", "-", model.lower()).strip("-")
    result_path = JUDGE_ROOT / f"judge_results_{slug}.jsonl"
    summary_path = JUDGE_ROOT / f"judge_summary_{slug}.json"
    if clean:
        result_path.unlink(missing_ok=True)
        summary_path.unlink(missing_ok=True)
    results = [] if clean else _read_jsonl(result_path)
    done = {str(row.get("pair_id")) for row in results if row.get("actual_model_call") is True}
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
    summary = summarize_phase75_blind_results(results, hidden)
    complete = len(results) == len(pairs) and not failures
    summary.update(
        {
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
        }
    )
    _write_json(summary_path, summary)
    return 0 if complete else 1


def _collect_metrics() -> dict[str, dict[str, Any]]:
    return {
        variant: _read_json(GENERATION_ROOT / f"metrics_{variant}.json")
        for variant in PHASE75_VARIANTS
    }


def _judge_summaries() -> dict[str, dict[str, Any]]:
    return {
        model: _read_json(
            JUDGE_ROOT / f"judge_summary_{re.sub(r'[^a-z0-9]+', '-', model.lower()).strip('-')}.json"
        )
        for model in JUDGE_MODELS
    }


def _examples(transcripts: Mapping[str, list[dict[str, Any]]]) -> str:
    selected_ids = (
        "phase75-evidence_truthfulness-01",
        "phase75-latest_correction-01",
        "phase75-ordinary_direct-01",
        "phase75-profile_recall-01",
    )
    lines = [
        "# Phase75 Output Examples",
        "",
        "All rows are real local Qwen3-4B outputs from simulated_usage sessions. Synthetic canaries are masked before persistence.",
        "",
    ]
    by_variant = {
        variant: {str(row.get("session_id")): row for row in rows}
        for variant, rows in transcripts.items()
    }
    for session_id in selected_ids:
        lines.extend((f"## {session_id}", ""))
        for variant in PHASE75_VARIANTS:
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
        files.append(
            {
                "path": str(path.relative_to(REPO_ROOT)),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return {
        "kind": "phase75_evidence_manifest",
        "files": files,
        "file_count": len(files),
        "manifest_sha256": stable_hash(files),
    }


def _finalize() -> int:
    metrics = _collect_metrics()
    deterministic = _read_json(JUDGE_ROOT / "deterministic_summary.json")
    judges = _judge_summaries()
    prerequisites = {
        "all_generation_complete": all(
            value.get("all_sessions_completed") is True for value in metrics.values()
        ),
        "deterministic_complete": deterministic.get("status") == "completed",
        "gemma_judge_complete": judges["gemma4:31b"].get("status") == "completed",
        "qwen36_judge_complete": judges["qwen3.6"].get("status") == "completed",
    }
    if not all(prerequisites.values()):
        raise SystemExit(f"Phase75 finalization prerequisites failed: {prerequisites}")
    decision = build_phase75_decision(
        metrics=metrics,
        deterministic=deterministic,
        independent_judges=judges,
    )
    comparison = {
        "kind": "phase75_personalization_benefit_comparison",
        "created_at": _utcnow(),
        "model": "Qwen3-4B",
        "holdout_session_count_per_arm": PHASE75_SESSION_COUNT,
        "real_generation_model_call_count": sum(
            int(value.get("model_call_count") or 0) for value in metrics.values()
        ),
        "real_judge_model_call_count": sum(
            int(value.get("completed_pair_count") or 0) for value in judges.values()
        ),
        "metrics": metrics,
        "score_deltas": {
            "runtime_vs_base": round(
                float(metrics["base_persona_runtime"]["personalization_score"])
                - float(metrics["base_minimal"]["personalization_score"]),
                4,
            ),
            "archived_adapter_vs_base": round(
                float(metrics["archived_adapter_minimal"]["personalization_score"])
                - float(metrics["base_minimal"]["personalization_score"]),
                4,
            ),
            "adapter_incremental_under_runtime": round(
                float(metrics["archived_adapter_persona_runtime"]["personalization_score"])
                - float(metrics["base_persona_runtime"]["personalization_score"]),
                4,
            ),
        },
        "deterministic_blind": deterministic,
        "independent_blind": judges,
        "decision": decision,
        "simulated_usage": True,
        "actual_user_feedback_count": 0,
        "actual_user_benefit_claim_allowed": False,
        "actual_product_benefit_claim_allowed": False,
    }
    transcripts = {
        variant: _read_jsonl(GENERATION_ROOT / f"transcripts_{variant}.jsonl")
        for variant in PHASE75_VARIANTS
    }
    _write_json(EVIDENCE_ROOT / "phase75-final-decision.json", decision)
    _write_json(EVIDENCE_ROOT / "comparison_summary.json", comparison)
    _write_text(EVIDENCE_ROOT / "output_examples.md", _examples(transcripts))
    _write_text(
        EVIDENCE_ROOT / "phase75-final-decision.md",
        f"""# Phase75 Final Decision

Recommendation: **{decision['recommendation']}**

- Runtime qualified: `{decision['runtime_qualified']}`
- Historical archived adapter requalified: `{decision['historical_archived_adapter_requalified']}`
- Runtime vs base deterministic score delta: `{comparison['score_deltas']['runtime_vs_base']}`
- Archived adapter vs base deterministic score delta: `{comparison['score_deltas']['archived_adapter_vs_base']}`
- Adapter incremental under runtime: `{comparison['score_deltas']['adapter_incremental_under_runtime']}`
- Real Qwen3-4B generation calls: `{comparison['real_generation_model_call_count']}`
- Real independent judge calls: `{comparison['real_judge_model_call_count']}`

This is a simulated_usage laboratory benchmark with real local model calls. It does not contain actual_user_feedback and cannot establish real-user product benefit. The Phase45 adapter remains archived regardless of this evaluation unless a separate manual reaudit explicitly changes its lifecycle; Phase75 itself cannot promote, attach Hermes, or change defaults.
""",
    )
    _write_text(
        EVIDENCE_ROOT / "phase75-runbook.md",
        """# Phase75 Runbook

```bash
.venv/bin/python tools/phase75_personalization_benefit_benchmark.py prepare --clean-evidence
.venv/bin/python tools/phase75_personalization_benefit_benchmark.py generate --variant base_minimal --clean
.venv/bin/python tools/phase75_personalization_benefit_benchmark.py generate --variant base_persona_runtime --clean
.venv/bin/python tools/phase75_personalization_benefit_benchmark.py generate --variant archived_adapter_minimal --clean
.venv/bin/python tools/phase75_personalization_benefit_benchmark.py generate --variant archived_adapter_persona_runtime --clean
.venv/bin/python tools/phase75_personalization_benefit_benchmark.py prepare-eval
.venv/bin/python tools/phase75_personalization_benefit_benchmark.py judge --model gemma4:31b --ollama-endpoint http://127.0.0.1:11435 --clean
.venv/bin/python tools/phase75_personalization_benefit_benchmark.py judge --model qwen3.6 --ollama-endpoint http://127.0.0.1:11435 --clean
.venv/bin/python tools/phase75_personalization_benefit_benchmark.py finalize
.venv/bin/python tools/phase75_personalization_benefit_benchmark.py validate
```

Phase75 uses 48 frozen simulated_usage sessions, four real Qwen3-4B arms, and two independent local Ollama judges. The Phase45 adapter is an archived eval-only negative control. No new training, Hermes attachment, default change, automatic promotion, or real-user claim is allowed.
""",
    )
    _write_text(
        EVIDENCE_ROOT / "next-pursuit-goal.md",
        """# Next Pursuit Goal

If persona runtime qualifies while the historical adapter remains archived, build Phase76 as a new privacy-safe persona-internalization training experiment. Freeze a fresh training curriculum and a separate holdout, train Qwen3-4B with completion-only loss, and compare base, persona runtime, new adapter, and adapter plus runtime. The adapter must improve independent blind preference quality without losing privacy, ordinary-task directness, diversity, or truthfulness. Keep all generated feedback labeled simulated_usage and do not auto-promote or attach Hermes.
""",
    )
    manifest = _evidence_manifest()
    _write_json(EVIDENCE_ROOT / "evidence_manifest.json", manifest)
    integrity_checks = {
        "all_prerequisites_complete": all(prerequisites.values()),
        "holdout_not_for_training": all(
            row.get("not_for_training") is True
            for row in _read_json(EVIDENCE_ROOT / "evidence-holdout/holdout.json").get("sessions") or []
        ),
        "historical_adapter_still_archive": decision["historical_adapter_lifecycle"] == "archive_unchanged",
        "no_actual_user_claim": decision["actual_user_benefit_claim_allowed"] is False,
        "no_auto_promotion": decision["auto_promotion_allowed"] is False,
        "no_new_training": decision["new_training_executed"] is False,
    }
    _write_json(
        EVIDENCE_ROOT / "evidence_integrity.json",
        {
            "kind": "phase75_evidence_integrity",
            "passed": all(integrity_checks.values()),
            "checks": integrity_checks,
            "manifest_sha256": manifest["manifest_sha256"],
        },
    )
    _write_json(
        EVIDENCE_ROOT / "finalization_state.json",
        {
            "kind": "phase75_finalization_state",
            "status": "completed",
            "decision": decision["recommendation"],
            "created_at": _utcnow(),
        },
    )
    print(json.dumps({"recommendation": decision["recommendation"], "score_deltas": comparison["score_deltas"]}, indent=2))
    return 0


def _command(args: list[str]) -> dict[str, Any]:
    started = time.perf_counter()
    completed = subprocess.run(
        args,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return {
        "command": args,
        "exit_code": completed.returncode,
        "duration_seconds": round(time.perf_counter() - started, 3),
        "output": completed.stdout[-12000:],
    }


def _validate() -> int:
    commands = [
        [sys.executable, "-m", "py_compile", str(CORE_SOURCE), str(DRIVER_SOURCE)],
        [
            str(REPO_ROOT / ".venv/bin/pytest"),
            "-q",
            "tests/test_phase75_personalization_benefit_benchmark.py",
            "tests/test_phase74_shared_raw_deterministic_serializer_ab.py",
            "tests/test_phase45_privacy_multiturn_preference.py",
        ],
        ["git", "diff", "--check"],
    ]
    results = []
    for args in commands:
        result = _command(args)
        results.append(result)
        print(f"[validate] {' '.join(args)} -> {result['exit_code']}", flush=True)
    evidence = _read_json(EVIDENCE_ROOT / "evidence_integrity.json")
    decision = _read_json(EVIDENCE_ROOT / "phase75-final-decision.json")
    checks = {
        "commands_passed": all(row["exit_code"] == 0 for row in results),
        "evidence_integrity_passed": evidence.get("passed") is True,
        "decision_is_truthful": decision.get("actual_user_benefit_claim_allowed") is False
        and decision.get("auto_promotion_allowed") is False
        and decision.get("historical_adapter_lifecycle") == "archive_unchanged",
    }
    summary = {
        "kind": "phase75_validation_summary",
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "commands": results,
        "full_make_gates_recorded_separately": True,
    }
    _write_json(EVIDENCE_ROOT / "validation_summary.json", summary)
    _write_text(
        EVIDENCE_ROOT / "validation_gate.txt",
        f"Phase75 validation: {summary['status']}\n"
        + "\n".join(f"{name}: {passed}" for name, passed in checks.items()),
    )
    return 0 if summary["status"] == "passed" else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--clean-evidence", action="store_true")
    generate = subparsers.add_parser("generate")
    generate.add_argument("--variant", choices=PHASE75_VARIANTS, required=True)
    generate.add_argument("--clean", action="store_true")
    subparsers.add_parser("prepare-eval")
    judge = subparsers.add_parser("judge")
    judge.add_argument("--model", choices=JUDGE_MODELS, required=True)
    judge.add_argument("--ollama-endpoint", default="http://127.0.0.1:11435")
    judge.add_argument("--timeout", type=int, default=900)
    judge.add_argument("--clean", action="store_true")
    subparsers.add_parser("finalize")
    subparsers.add_parser("validate")
    args = parser.parse_args()
    if args.command == "prepare":
        return _prepare(args.clean_evidence)
    if args.command == "generate":
        return _generate(args.variant, args.clean)
    if args.command == "prepare-eval":
        return _prepare_eval()
    if args.command == "judge":
        return _judge(args.model, args.ollama_endpoint, args.timeout, args.clean)
    if args.command == "finalize":
        return _finalize()
    if args.command == "validate":
        return _validate()
    raise SystemExit(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
