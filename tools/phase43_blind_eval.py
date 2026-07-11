#!/usr/bin/env python3
"""Build Phase43 anonymous pairs and run deterministic plus Ollama blind judges."""

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
from urllib.error import URLError
from urllib.request import Request, urlopen


REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = REPO_ROOT / "pfe-core"
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

from pfe_core.phase43_personal_preference_benefit import (
    build_phase43_blind_pairs,
    score_phase43_blind_pairs_deterministic,
    summarize_phase43_blind_results,
)


EVIDENCE_ROOT = REPO_ROOT / "docs" / "demo" / "phase43-qwen3-4b-personal-preference-benefit-proof"
HOLDOUT_DIR = EVIDENCE_ROOT / "evidence-holdout"
BLIND_DIR = EVIDENCE_ROOT / "evidence-blind-eval"
CANDIDATE_PATH = EVIDENCE_ROOT / "evidence-candidate-review" / "selected_preference_pairs.jsonl"


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


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _assistant_answers(transcript: Mapping[str, Any]) -> list[str]:
    return [
        str(row.get("content") or "")
        for row in transcript.get("turns") or []
        if isinstance(row, Mapping) and row.get("role") == "assistant"
    ]


def _judge_prompt(pair: Mapping[str, Any]) -> str:
    left = _assistant_answers(dict(pair.get("variant_left") or {}))
    right = _assistant_answers(dict(pair.get("variant_right") or {}))
    return (
        "你是独立的盲测评审。你不知道两个回答来自什么模型或训练方式。\n"
        "请只根据用户目标和多轮纠正判断哪个 Agent 更符合以下偏好：先依据证据再下结论；"
        "遵循最新纠正；给出具体下一步；不假装完成；不泄露私密值；简洁且不套固定模板。\n"
        "如果双方各有明显优缺点且无法稳定区分，选 tie。\n\n"
        f"用户初始目标：{pair.get('user_goal')}\n"
        f"用户纠正：{pair.get('user_correction')}\n"
        f"继续要求：{pair.get('continuation_request')}\n\n"
        f"左侧 Agent 三轮回答：\n{json.dumps(left, ensure_ascii=False)}\n\n"
        f"右侧 Agent 三轮回答：\n{json.dumps(right, ensure_ascii=False)}\n\n"
        "只输出 JSON 对象，不要 Markdown："
        '{"winner":"left|right|tie","confidence":0.0,"reason":"一句简短理由"}'
    )


def _parse_json_object(text: str) -> dict[str, Any]:
    stripped = str(text or "").strip()
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            return {}
        try:
            value = json.loads(stripped[start : end + 1])
        except json.JSONDecodeError:
            return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _ollama_judge(*, pair: Mapping[str, Any], model: str, endpoint: str, timeout: int) -> dict[str, Any]:
    prompt = _judge_prompt(pair)
    payload = {
        "model": model,
        "stream": False,
        "think": False,
        "format": "json",
        "messages": [{"role": "user", "content": prompt}],
        "options": {"temperature": 0, "num_predict": 256},
    }
    request = Request(
        endpoint.rstrip("/") + "/api/chat",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urlopen(request, timeout=timeout) as response:
        raw_body = response.read().decode("utf-8")
    body = json.loads(raw_body)
    content = str(dict(body.get("message") or {}).get("content") or "")
    parsed = _parse_json_object(content)
    winner = str(parsed.get("winner") or "").lower()
    if winner not in {"left", "right", "tie"}:
        raise ValueError(f"judge returned invalid winner: {winner or '<empty>'}")
    try:
        confidence = max(0.0, min(1.0, float(parsed.get("confidence") or 0.0)))
    except (TypeError, ValueError):
        confidence = 0.0
    return {
        "pair_id": pair.get("pair_id"),
        "winner": winner,
        "confidence": confidence,
        "reason": str(parsed.get("reason") or "").strip(),
        "judge": "independent_ollama_blind_judge",
        "judge_model": model,
        "actual_model_call": True,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "raw_response": content,
        "latency_seconds": round(time.perf_counter() - started, 4),
        "created_at": _utcnow(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge-model", default="gemma4:31b")
    parser.add_argument("--ollama-endpoint", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--skip-independent", action="store_true")
    parser.add_argument("--judge-limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    BLIND_DIR.mkdir(parents=True, exist_ok=True)

    holdout = _read_json(HOLDOUT_DIR / "holdout.json")
    sessions = [dict(row) for row in holdout.get("sessions") or []]
    transcripts = {
        variant: _read_jsonl(HOLDOUT_DIR / f"transcripts_{variant}.jsonl")
        for variant in ("base", "runtime", "sft")
    }
    if not all(len(rows) == 40 for rows in transcripts.values()):
        raise SystemExit(f"Phase43 blind eval requires 40 transcripts per variant: { {key: len(value) for key, value in transcripts.items()} }")
    blind = build_phase43_blind_pairs(transcripts, sessions, seed=43)
    public_pairs = list(blind["public_pairs"])
    hidden_key = list(blind["hidden_key"])
    _write_jsonl(BLIND_DIR / "blind_eval_pairs_public.jsonl", public_pairs)
    _write_json(BLIND_DIR / "blind_variant_key.json", {"kind": blind["kind"], "seed": blind["seed"], "items": hidden_key})
    _write_json(
        BLIND_DIR / "blind_integrity_check.json",
        {
            "kind": "phase43_blind_integrity_check",
            "passed": blind.get("identity_hidden_from_judge") is True and len(public_pairs) == 80,
            "pair_count": len(public_pairs),
            "expected_pair_count": 80,
            "pair_ids_are_opaque": all(re.fullmatch(r"phase43-blind-\d{4}", str(row.get("pair_id") or "")) for row in public_pairs),
            "public_transcripts_only_contain_session_and_turns": all(
                set(dict(row.get(side) or {})) <= {"session_id", "turns"}
                for row in public_pairs
                for side in ("variant_left", "variant_right")
            ),
        },
    )

    training_targets = [str(row.get("chosen") or "") for row in _read_jsonl(CANDIDATE_PATH)]
    deterministic = score_phase43_blind_pairs_deterministic(blind, training_targets=training_targets)
    deterministic_summary = summarize_phase43_blind_results(deterministic, hidden_key)
    deterministic_summary["status"] = "completed"
    deterministic_summary["judge"] = "deterministic_phase43_rubric"
    _write_jsonl(BLIND_DIR / "deterministic_results.jsonl", deterministic)
    _write_json(BLIND_DIR / "deterministic_summary.json", deterministic_summary)

    independent_path = BLIND_DIR / "independent_judge_results.jsonl"
    existing = _read_jsonl(independent_path) if args.resume else []
    by_id = {str(row.get("pair_id")): dict(row) for row in existing if row.get("winner") in {"left", "right", "tie"}}
    failures: list[dict[str, Any]] = []
    selected_pairs = public_pairs[: max(0, args.judge_limit)] if args.judge_limit is not None else public_pairs
    if not args.skip_independent:
        for index, pair in enumerate(selected_pairs, start=1):
            pair_id = str(pair.get("pair_id") or "")
            if pair_id in by_id:
                continue
            try:
                result = _ollama_judge(
                    pair=pair,
                    model=args.judge_model,
                    endpoint=args.ollama_endpoint,
                    timeout=max(30, args.timeout),
                )
                by_id[pair_id] = result
                _write_jsonl(independent_path, [by_id[key] for key in sorted(by_id)])
                print(f"[judge] {index}/{len(selected_pairs)} {pair_id} {result['winner']}", flush=True)
            except (OSError, URLError, ValueError, json.JSONDecodeError) as exc:
                failure = {
                    "pair_id": pair_id,
                    "error": f"{exc.__class__.__name__}: {exc}",
                    "actual_model_call": False,
                    "created_at": _utcnow(),
                }
                failures.append(failure)
                print(f"[judge] {index}/{len(selected_pairs)} {pair_id} failed: {failure['error']}", flush=True)
    results = [by_id[key] for key in sorted(by_id)]
    independent_summary = summarize_phase43_blind_results(results, hidden_key)
    complete = not args.skip_independent and len(results) == len(public_pairs) and not failures
    independent_summary.update(
        {
            "status": "completed" if complete else "blocked",
            "judge": "independent_ollama_blind_judge",
            "judge_model": args.judge_model,
            "expected_pair_count": len(public_pairs),
            "completed_pair_count": len(results),
            "failure_count": len(failures),
            "failures": failures,
            "identity_hidden_from_judge": True,
            "actual_model_calls": bool(results),
            "fabricated_scores": False,
            "created_at": _utcnow(),
        }
    )
    _write_json(BLIND_DIR / "independent_judge_summary.json", independent_summary)
    print(json.dumps({"deterministic": deterministic_summary.get("variants"), "independent": independent_summary.get("variants"), "independent_status": independent_summary["status"]}, ensure_ascii=False, indent=2))
    return 0 if complete or args.skip_independent else 1


if __name__ == "__main__":
    raise SystemExit(main())
