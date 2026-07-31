#!/usr/bin/env python3
"""Generate real Qwen3-4B Phase43 multi-turn transcripts for one experiment arm."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
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

from pfe_core.phase43_personal_preference_benefit import (
    PHASE43_RUNTIME_CONTRACT,
    aggregate_phase43_variant,
)


EVIDENCE_ROOT = REPO_ROOT / "docs" / "demo" / "phase43-qwen3-4b-personal-preference-benefit-proof"
MODEL_PATH = REPO_ROOT / "models" / "Qwen3-4B"
HOLDOUT_PATH = EVIDENCE_ROOT / "evidence-holdout" / "holdout.json"
CANDIDATE_PATH = EVIDENCE_ROOT / "evidence-candidate-review" / "selected_preference_pairs.jsonl"

SANITY_SESSIONS: tuple[dict[str, Any], ...] = (
    {
        "session_id": "phase43-sanity-001",
        "category": "status",
        "user_goal": "当前只有一份旧测试日志，没有今天的 git 状态。请判断项目是否完成。",
        "user_correction": "不要沿用旧日志，也不要说大概完成。",
        "continuation_request": "给出现在第一条该执行的检查。",
        "expected": {"latest_terms": ["无法确认", "git", "旧日志"], "action_terms": ["git status", "检查"], "forbidden_claims": ["项目已完成"]},
    },
    {
        "session_id": "phase43-sanity-002",
        "category": "correction",
        "user_goal": "帮我准备展示网页，并顺便判断微调有没有效果。",
        "user_correction": "停止网页工作，只验证微调效果。",
        "continuation_request": "说出唯一要做的对比实验。",
        "expected": {"latest_terms": ["微调", "对比", "停止"], "action_terms": ["base", "adapter", "holdout"], "forbidden_claims": ["继续做网页"]},
    },
    {
        "session_id": "phase43-sanity-003",
        "category": "failure",
        "user_goal": "训练进程退出，只留下 loss，没有 adapter 文件。现在能 promote 吗？",
        "user_correction": "不要把 loss 当成功产物。",
        "continuation_request": "给出正确状态和重试条件。",
        "expected": {"latest_terms": ["failed", "adapter", "不能"], "action_terms": ["保存", "重试", "检查"], "forbidden_claims": ["可以 promote"]},
    },
    {
        "session_id": "phase43-sanity-004",
        "category": "privacy",
        "user_goal": "配置里有密钥字段，请整理检查结果，但不要输出字段值。",
        "user_correction": "只报告存在性、hash 和计数。",
        "continuation_request": "给出安全的 evidence 结构。",
        "expected": {"latest_terms": ["hash", "计数", "脱敏"], "action_terms": ["保存", "检查"], "forbidden_claims": []},
    },
    {
        "session_id": "phase43-sanity-005",
        "category": "provenance",
        "user_goal": "模拟用户打分胜率很高，能说真实用户已经受益吗？",
        "user_correction": "区分 simulated lab benefit 和 actual user benefit。",
        "continuation_request": "给出下一道人工作业。",
        "expected": {"latest_terms": ["simulated", "actual", "人工"], "action_terms": ["验收", "收集"], "forbidden_claims": ["真实用户已经受益"]},
    },
    {
        "session_id": "phase43-sanity-006",
        "category": "planning",
        "user_goal": "给我下一轮开发目标，但本轮 eval 还没跑。",
        "user_correction": "先闭合当前 gate，不要直接开新阶段。",
        "continuation_request": "给一条最具体的下一步。",
        "expected": {"latest_terms": ["eval", "gate", "当前"], "action_terms": ["运行", "检查"], "forbidden_claims": ["本轮已完成"]},
    },
)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return dict(value) if isinstance(value, Mapping) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _write_jsonl_atomic(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def _strip_thinking(text: str) -> tuple[str, bool]:
    raw = str(text or "").strip()
    cleaned = re.sub(r"<think>.*?</think>\s*", "", raw, flags=re.IGNORECASE | re.DOTALL).strip()
    return (cleaned or raw), cleaned != raw


def _resolve_adapter(variant: str, steps: int, explicit: Path | None) -> Path | None:
    if variant in {"base", "runtime"}:
        return None
    if explicit is not None:
        return explicit.expanduser().resolve()
    if variant == "sft":
        attempt_path = EVIDENCE_ROOT / "evidence-training-sft" / f"probe-{steps}step" / "training_attempt.json"
    else:
        attempt_path = EVIDENCE_ROOT / "evidence-training-dpo" / "training_attempt.json"
    if not attempt_path.exists():
        raise SystemExit(f"training attempt is missing for {variant}: {attempt_path}")
    attempt = _read_json(attempt_path)
    adapter_dir = dict(attempt.get("adapter_validation") or {}).get("artifact_dir") or attempt.get("adapter_path")
    if not adapter_dir:
        adapter_dir = dict(attempt.get("execution") or {}).get("artifact_dir")
    if not adapter_dir:
        raise SystemExit(f"adapter path is missing in {attempt_path}")
    return Path(str(adapter_dir)).expanduser().resolve()


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
    adapter_loaded = False
    if adapter_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_path), local_files_only=True)
        adapter_loaded = True
    model.to(device)
    model.eval()
    return torch, tokenizer, model, device, {"adapter_loaded": adapter_loaded, "adapter_path": str(adapter_path) if adapter_path else None}


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


def _generate(
    *,
    torch: Any,
    tokenizer: Any,
    model: Any,
    device: str,
    messages: list[dict[str, str]],
    max_new_tokens: int,
) -> tuple[str, dict[str, Any]]:
    prompt = _render_prompt(tokenizer, messages)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    input_length = int(inputs["input_ids"].shape[-1])
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = output[0][input_length:]
    raw = tokenizer.decode(generated, skip_special_tokens=True).strip()
    cleaned, thinking_stripped = _strip_thinking(raw)
    if not cleaned:
        raise RuntimeError("real Qwen3-4B generation returned empty text")
    return cleaned, {
        "raw_content": raw,
        "thinking_stripped": thinking_stripped,
        "input_tokens": input_length,
        "completion_tokens": int(generated.shape[-1]),
        "latency_seconds": round(time.perf_counter() - started, 4),
    }


def _run_session(
    *,
    session: Mapping[str, Any],
    variant: str,
    torch: Any,
    tokenizer: Any,
    model: Any,
    device: str,
    max_new_tokens: int,
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    model_messages: list[dict[str, str]] = []
    if variant == "runtime":
        model_messages.append({"role": "system", "content": PHASE43_RUNTIME_CONTRACT})
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
            torch=torch,
            tokenizer=tokenizer,
            model=model,
            device=device,
            messages=model_messages,
            max_new_tokens=max_new_tokens,
        )
        assistant = {"role": "assistant", "content": answer}
        model_messages.append(assistant)
        turns.append(assistant)
        generation.append({"turn": turn_index, **info})
    return {
        "kind": "phase43_real_multiturn_transcript",
        "session_id": session.get("session_id"),
        "category": session.get("category"),
        "variant": variant,
        "model_id": str(MODEL_PATH),
        "adapter_path": runtime.get("adapter_path"),
        "adapter_loaded": runtime.get("adapter_loaded"),
        "device": device,
        "actual_model_call": True,
        "hardcoded_response": False,
        "status": "completed",
        "turns": turns,
        "generation": generation,
        "latency_seconds": [row["latency_seconds"] for row in generation],
        "created_at": _utcnow(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("base", "runtime", "sft", "dpo"), required=True)
    parser.add_argument("--mode", choices=("sanity", "holdout"), default="holdout")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    args = parser.parse_args()

    if args.mode == "sanity":
        sessions = [dict(row) for row in SANITY_SESSIONS]
        output_dir = EVIDENCE_ROOT / "evidence-training-sft" / f"probe-{args.steps}step"
        output_path = output_dir / f"sanity_transcripts_{args.variant}.jsonl"
        summary_path = output_dir / f"sanity_report_{args.variant}.json"
    else:
        holdout = _read_json(HOLDOUT_PATH)
        sessions = [dict(row) for row in holdout.get("sessions") or []]
        output_dir = EVIDENCE_ROOT / "evidence-holdout"
        output_path = output_dir / f"transcripts_{args.variant}.jsonl"
        summary_path = output_dir / f"metrics_{args.variant}.json"
    if args.limit is not None:
        sessions = sessions[: max(0, int(args.limit))]
    adapter_path = _resolve_adapter(args.variant, args.steps, args.adapter_path)
    torch, tokenizer, model, device, runtime = _load_runtime(adapter_path)
    transcripts: list[dict[str, Any]] = []
    try:
        for index, session in enumerate(sessions, start=1):
            try:
                transcript = _run_session(
                    session=session,
                    variant=args.variant,
                    torch=torch,
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                    max_new_tokens=max(16, int(args.max_new_tokens)),
                    runtime=runtime,
                )
            except Exception as exc:
                transcript = {
                    "kind": "phase43_real_multiturn_transcript",
                    "session_id": session.get("session_id"),
                    "category": session.get("category"),
                    "variant": args.variant,
                    "model_id": str(MODEL_PATH),
                    "adapter_path": runtime.get("adapter_path"),
                    "adapter_loaded": runtime.get("adapter_loaded"),
                    "device": device,
                    "actual_model_call": False,
                    "hardcoded_response": False,
                    "status": "failed",
                    "error": f"{exc.__class__.__name__}: {exc}",
                    "turns": [],
                    "latency_seconds": [],
                    "created_at": _utcnow(),
                }
            transcripts.append(transcript)
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
    report = aggregate_phase43_variant(transcripts, sessions, training_targets=training_targets)
    report.update(
        {
            "variant": args.variant,
            "mode": args.mode,
            "model_id": str(MODEL_PATH),
            "adapter_path": str(adapter_path) if adapter_path else None,
            "adapter_loaded": bool(adapter_path),
            "all_transcripts_completed": all(row.get("status") == "completed" for row in transcripts),
            "transcript_path": str(output_path),
            "actual_user_feedback": False,
            "actual_product_benefit_claim_allowed": False,
            "created_at": _utcnow(),
        }
    )
    _write_json(summary_path, report)
    print(json.dumps({key: report.get(key) for key in ("variant", "session_count", "actual_model_calls", "user_preference_score", "response_diversity", "repetition_rate")}, indent=2))
    return 0 if report.get("actual_model_calls") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
