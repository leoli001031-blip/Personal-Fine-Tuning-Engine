#!/usr/bin/env python3
"""Run the Phase76 conditional persona-runtime recovery experiment."""

from __future__ import annotations

import argparse
from collections import Counter
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

import phase75_personalization_benefit_benchmark as phase75_driver
from pfe_core.phase75_personalization_benefit_benchmark import (
    PHASE75_MINIMAL_CONTRACT,
    aggregate_phase75_variant,
    stable_hash,
)
from pfe_core.phase76_conditional_persona_runtime import (
    PHASE76_CONTROL_COUNT,
    PHASE76_TARGET_COUNT,
    PHASE76_VARIANTS,
    audit_phase76_ordinary_identity,
    audit_phase76_routes,
    build_phase76_blind_pairs,
    build_phase76_decision,
    build_phase76_holdout,
    build_phase76_router_calibration,
    contract_for_phase76_messages,
    score_phase76_blind_pairs_deterministic,
    summarize_phase76_blind_results,
)


EVIDENCE_ROOT = REPO_ROOT / "docs/demo/phase76-conditional-persona-runtime"
GENERATION_ROOT = EVIDENCE_ROOT / "evidence-real-generation"
JUDGE_ROOT = EVIDENCE_ROOT / "evidence-blind-eval"
PHASE75_ROOT = REPO_ROOT / "docs/demo/phase75-personalization-benefit-benchmark"
MODEL_PATH = REPO_ROOT / "models/Qwen3-4B"
CORE_SOURCE = CORE_ROOT / "pfe_core/phase76_conditional_persona_runtime.py"
DRIVER_SOURCE = REPO_ROOT / "tools/phase76_conditional_persona_runtime.py"
TEST_SOURCE = REPO_ROOT / "tests/test_phase76_conditional_persona_runtime.py"
JUDGE_MODELS = ("gemma4:31b", "qwen3.6")
GENERATION_PROTOCOL = dict(phase75_driver.GENERATION_PROTOCOL)
GENERATION_PROTOCOL.update(
    {
        "kind": "phase76_frozen_generation_protocol",
        "variants": list(PHASE76_VARIANTS),
        "conditional_contract_recomputed_before_each_turn": True,
        "ordinary_control_requires_byte_identity": True,
    }
)
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


def _phase75_failure_taxonomy() -> dict[str, Any]:
    decision = _read_json(PHASE75_ROOT / "phase75-final-decision.json")
    comparison = _read_json(PHASE75_ROOT / "comparison_summary.json")
    integrity = _read_json(PHASE75_ROOT / "evidence_integrity.json")
    public = {
        str(row.get("pair_id")): row
        for row in _read_jsonl(PHASE75_ROOT / "evidence-blind-eval/blind_items_public.jsonl")
    }
    hidden = {
        str(row.get("pair_id")): row
        for row in _read_json(PHASE75_ROOT / "evidence-blind-eval/blind_hidden_key.json").get("hidden_key") or []
    }
    results = _read_jsonl(PHASE75_ROOT / "evidence-blind-eval/deterministic_results.jsonl")
    target = Counter()
    ordinary = Counter()
    for row in results:
        pair_id = str(row.get("pair_id") or "")
        key = hidden.get(pair_id, {})
        if key.get("comparison") != "runtime_vs_base":
            continue
        winner = str(row.get("winner") or "")
        outcome = "tie"
        if winner in {"left", "right"}:
            outcome = (
                "candidate"
                if key.get(f"variant_{winner}") == key.get("candidate")
                else "benchmark"
            )
        bucket = ordinary if public.get(pair_id, {}).get("category") == "ordinary_direct" else target
        bucket["pair_count"] += 1
        bucket[outcome] += 1

    metrics = dict(comparison.get("metrics") or {})
    base_ordinary = float(
        dict(dict(metrics.get("base_minimal") or {}).get("category_metrics") or {})
        .get("ordinary_direct", {})
        .get("composite_personalization_score")
        or 0.0
    )
    runtime_ordinary = float(
        dict(dict(metrics.get("base_persona_runtime") or {}).get("category_metrics") or {})
        .get("ordinary_direct", {})
        .get("composite_personalization_score")
        or 0.0
    )

    def summary(counts: Counter[str]) -> dict[str, Any]:
        total = counts["pair_count"]
        return {
            "pair_count": total,
            "candidate_wins": counts["candidate"],
            "benchmark_wins": counts["benchmark"],
            "ties": counts["tie"],
            "candidate_win_rate": round(counts["candidate"] / total, 4) if total else 0.0,
        }

    checks = {
        "phase75_integrity_passed": integrity.get("passed") is True,
        "phase75_status_is_hold": decision.get("status") == "hold",
        "phase75_runtime_failed_only_deterministic_gate": decision.get("runtime_failed_checks")
        == ["runtime_deterministic_win_rate_at_least_0_60"],
        "phase75_adapter_remains_archived": decision.get("historical_adapter_lifecycle")
        == "archive_unchanged",
        "phase75_no_actual_user_claim": decision.get("actual_user_benefit_claim_allowed")
        is False,
    }
    return {
        "kind": "phase76_phase75_failure_taxonomy",
        "passed": all(checks.values()),
        "checks": checks,
        "runtime_vs_base_target_slice": summary(target),
        "runtime_vs_base_ordinary_slice": summary(ordinary),
        "ordinary_score_base": base_ordinary,
        "ordinary_score_unconditional_runtime": runtime_ordinary,
        "ordinary_score_delta": round(runtime_ordinary - base_ordinary, 4),
        "diagnosis": (
            "Phase75 showed target-slice semantic benefit, but the unconditional persona contract "
            "also changed ordinary direct tasks. Phase76 tests a conditional router without "
            "retroactively changing the Phase75 gate."
        ),
        "phase75_decision_unchanged": True,
    }


def _holdout_overlap(holdout: Mapping[str, Any]) -> dict[str, Any]:
    previous = _read_json(PHASE75_ROOT / "evidence-holdout/holdout.json").get("sessions") or []

    def texts(rows: Iterable[Mapping[str, Any]]) -> set[str]:
        return {
            re.sub(r"\s+", " ", str(row.get(key) or "").strip()).lower()
            for row in rows
            for key in ("user_goal", "user_correction", "continuation_request")
            if str(row.get(key) or "").strip()
        }

    overlap = sorted(texts(holdout.get("sessions") or []) & texts(previous))
    return {
        "kind": "phase76_holdout_overlap_audit",
        "passed": not overlap,
        "phase76_session_count": holdout.get("session_count"),
        "phase75_session_count": len(previous),
        "exact_text_overlap_count": len(overlap),
        "exact_text_overlap": overlap,
    }


def _prepare(clean: bool) -> int:
    if clean and EVIDENCE_ROOT.exists():
        shutil.rmtree(EVIDENCE_ROOT)
    required = (
        CORE_SOURCE,
        DRIVER_SOURCE,
        TEST_SOURCE,
        MODEL_PATH / "config.json",
        PHASE75_ROOT / "phase75-final-decision.json",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"Phase76 required sources missing: {missing}")

    taxonomy = _phase75_failure_taxonomy()
    holdout = build_phase76_holdout()
    overlap = _holdout_overlap(holdout)
    calibration = build_phase76_router_calibration()
    route_audit = audit_phase76_routes(holdout["sessions"])
    checks = {
        "phase75_taxonomy_passed": taxonomy["passed"],
        "fresh_holdout_48": holdout["session_count"] == 48,
        "target_count_36": holdout["target_count"] == PHASE76_TARGET_COUNT,
        "ordinary_control_count_12": holdout["ordinary_control_count"] == PHASE76_CONTROL_COUNT,
        "holdout_overlap_zero": overlap["passed"],
        "router_calibration_64_of_64": calibration["passed"] is True
        and calibration["accuracy"] == 1.0,
        "pre_call_route_audit_144_of_144": route_audit["passed"] is True
        and route_audit["accuracy"] == 1.0,
        "all_holdout_not_for_training": all(
            row.get("not_for_training") is True for row in holdout["sessions"]
        ),
        "actual_user_feedback_count_zero": holdout["actual_user_feedback_count"] == 0,
    }
    freeze = {
        "kind": "phase76_pre_model_call_freeze",
        "frozen_at": _utcnow(),
        "frozen_before_model_calls": True,
        "passed": all(checks.values()),
        "checks": checks,
        "holdout_manifest_sha256": holdout["manifest_sha256"],
        "router_calibration_sha256": stable_hash(calibration),
        "route_audit_sha256": stable_hash(route_audit),
        "minimal_contract_sha256": hashlib.sha256(PHASE75_MINIMAL_CONTRACT.encode()).hexdigest(),
        "scorer_source_sha256": _sha256(CORE_SOURCE),
        "driver_source_sha256": _sha256(DRIVER_SOURCE),
        "test_source_sha256": _sha256(TEST_SOURCE),
        "generation_helper_source_sha256": _sha256(phase75_driver.DRIVER_SOURCE),
        "generation_protocol_sha256": stable_hash(GENERATION_PROTOCOL),
        "variants": list(PHASE76_VARIANTS),
        "decision_thresholds": {
            "target_score_gain_min": 0.08,
            "target_each_judge_win_rate_min": 0.60,
            "target_deterministic_win_rate_min": 0.60,
            "ordinary_full_transcript_identity_rate": 1.0,
            "ordinary_route_off_rate": 1.0,
        },
        "score_or_gate_relaxation_allowed": False,
        "training_allowed": False,
        "auto_promotion_allowed": False,
    }
    _write_json(EVIDENCE_ROOT / "evidence-baseline/phase75_failure_taxonomy.json", taxonomy)
    _write_json(EVIDENCE_ROOT / "evidence-holdout/holdout.json", holdout)
    _write_json(EVIDENCE_ROOT / "evidence-holdout/overlap_audit.json", overlap)
    _write_json(EVIDENCE_ROOT / "evidence-router/router_calibration.json", calibration)
    _write_json(EVIDENCE_ROOT / "evidence-router/pre_call_route_audit.json", route_audit)
    _write_json(EVIDENCE_ROOT / "generation_protocol.json", GENERATION_PROTOCOL)
    _write_json(EVIDENCE_ROOT / "pre_model_call_freeze.json", freeze)
    _write_json(
        EVIDENCE_ROOT / "evidence-no-training/training_attempt.json",
        {
            "kind": "phase76_training_attempt",
            "status": "not_run_by_design",
            "reason": "Phase76 qualifies the conditional runtime reference before any new training.",
            "actual_training_executed": False,
            "adapter_created": False,
            "historical_adapter_lifecycle": "archive_unchanged",
        },
    )
    _write_json(
        EVIDENCE_ROOT / "preparation_decision.json",
        {
            "kind": "phase76_preparation_decision",
            "status": "ready_for_real_generation" if freeze["passed"] else "blocked",
            "checks": checks,
        },
    )
    print(json.dumps({"status": "ready" if freeze["passed"] else "blocked", "checks": checks}, indent=2))
    return 0 if freeze["passed"] else 1


def _generation_freeze_check() -> dict[str, Any]:
    freeze = _read_json(EVIDENCE_ROOT / "pre_model_call_freeze.json")
    holdout = _read_json(EVIDENCE_ROOT / "evidence-holdout/holdout.json")
    calibration = _read_json(EVIDENCE_ROOT / "evidence-router/router_calibration.json")
    route_audit = _read_json(EVIDENCE_ROOT / "evidence-router/pre_call_route_audit.json")
    checks = {
        "preparation_passed": freeze.get("passed") is True,
        "holdout_unchanged": stable_hash(holdout.get("sessions") or [])
        == freeze.get("holdout_manifest_sha256"),
        "router_calibration_unchanged": stable_hash(calibration)
        == freeze.get("router_calibration_sha256"),
        "route_audit_unchanged": stable_hash(route_audit) == freeze.get("route_audit_sha256"),
        "scorer_unchanged": _sha256(CORE_SOURCE) == freeze.get("scorer_source_sha256"),
        "driver_unchanged": _sha256(DRIVER_SOURCE) == freeze.get("driver_source_sha256"),
        "test_unchanged": _sha256(TEST_SOURCE) == freeze.get("test_source_sha256"),
        "generation_helper_unchanged": _sha256(phase75_driver.DRIVER_SOURCE)
        == freeze.get("generation_helper_source_sha256"),
        "protocol_unchanged": stable_hash(GENERATION_PROTOCOL)
        == freeze.get("generation_protocol_sha256"),
    }
    return {"kind": "phase76_generation_freeze_check", "passed": all(checks.values()), "checks": checks}


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
) -> dict[str, Any]:
    raw_history: list[dict[str, str]] = []
    persisted_turns: list[dict[str, str]] = []
    generations = []
    routes = []
    contract_hashes = []
    canary = str(dict(session.get("expected") or {}).get("privacy_canary") or "")
    canary_echo = False
    for index, user_text in enumerate(
        (
            str(session.get("user_goal") or ""),
            str(session.get("user_correction") or ""),
            str(session.get("continuation_request") or ""),
        ),
        start=1,
    ):
        raw_history.append({"role": "user", "content": user_text})
        persisted_turns.append({"role": "user", "content": _masked(user_text, canary)})
        if variant == "conditional_persona_runtime":
            contract, route = contract_for_phase76_messages(raw_history)
        else:
            contract = PHASE75_MINIMAL_CONTRACT
            route = {
                "kind": "phase76_persona_route",
                "routed": False,
                "reason": "base_arm_minimal_contract",
                "latest_user_sha256": stable_hash(user_text.strip()),
                "raw_user_text_persisted": False,
            }
        route_record = {"turn": index, **route}
        routes.append(route_record)
        contract_hashes.append(hashlib.sha256(contract.encode()).hexdigest())
        messages = [{"role": "system", "content": contract}, *raw_history]
        answer, info = phase75_driver._generate_one(torch, tokenizer, model, device, messages)
        canary_echo = canary_echo or bool(canary and canary in answer)
        persisted_answer = _masked(answer, canary)
        raw_history.append({"role": "assistant", "content": persisted_answer})
        persisted_turns.append({"role": "assistant", "content": persisted_answer})
        generations.append({"turn": index, **info})
    return {
        "kind": "phase76_real_multiturn_transcript",
        "session_id": session.get("session_id"),
        "category": session.get("category"),
        "task_type": session.get("task_type"),
        "variant": variant,
        "model_id": str(MODEL_PATH),
        "adapter_loaded": False,
        "conditional_persona_runtime": variant == "conditional_persona_runtime",
        "route_manifests": routes,
        "system_contract_sha256_by_turn": contract_hashes,
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
    if variant not in PHASE76_VARIANTS:
        raise SystemExit(f"unsupported Phase76 variant: {variant}")
    freeze = _generation_freeze_check()
    if not freeze["passed"]:
        raise SystemExit(f"Phase76 freeze check failed: {freeze}")
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
    session_ids = {str(row["session_id"]) for row in sessions}
    transcripts = [row for row in existing if str(row.get("session_id")) in session_ids]
    torch, tokenizer, model, device = phase75_driver._load_runtime(None)
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
                )
            except Exception as exc:
                transcript = {
                    "kind": "phase76_real_multiturn_transcript",
                    "session_id": session_id,
                    "category": session.get("category"),
                    "task_type": session.get("task_type"),
                    "variant": variant,
                    "model_id": str(MODEL_PATH),
                    "adapter_loaded": False,
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
            "kind": "phase76_variant_metrics",
            "variant": variant,
            "model_id": str(MODEL_PATH),
            "adapter_loaded": False,
            "conditional_persona_runtime": variant == "conditional_persona_runtime",
            "model_call_count": sum(
                len(row.get("generation") or [])
                for row in transcripts
                if row.get("actual_model_call") is True
            ),
            "route_decision_count": sum(len(row.get("route_manifests") or []) for row in transcripts),
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
        for variant in PHASE76_VARIANTS
    }
    generation_checks = {
        f"{variant}_48_completed": len(rows) == 48
        and all(row.get("status") == "completed" and row.get("actual_model_call") is True for row in rows)
        for variant, rows in transcripts.items()
    }
    if not all(generation_checks.values()):
        raise SystemExit(f"Phase76 generation incomplete: {generation_checks}")
    ordinary_identity = audit_phase76_ordinary_identity(transcripts, sessions)
    _write_json(EVIDENCE_ROOT / "evidence-router/ordinary_passthrough_identity.json", ordinary_identity)
    blind = build_phase76_blind_pairs(transcripts, sessions)
    deterministic_results = score_phase76_blind_pairs_deterministic(blind, sessions)
    deterministic = summarize_phase76_blind_results(
        deterministic_results,
        blind["hidden_key"],
        blind["public_pairs"],
    )
    deterministic.update({"status": "completed", "judge": "phase76_frozen_deterministic_rubric"})
    _write_jsonl(JUDGE_ROOT / "blind_items_public.jsonl", blind["public_pairs"])
    _write_json(JUDGE_ROOT / "blind_hidden_key.json", {"hidden_key": blind["hidden_key"]})
    _write_jsonl(JUDGE_ROOT / "deterministic_results.jsonl", deterministic_results)
    _write_json(JUDGE_ROOT / "deterministic_summary.json", deterministic)
    prompt_template = _judge_prompt_template()
    freeze = {
        "kind": "phase76_pre_judge_freeze",
        "frozen_at": _utcnow(),
        "frozen_before_judge_calls": True,
        "passed": all(generation_checks.values()) and blind["pair_count"] == 48,
        "generation_checks": generation_checks,
        "ordinary_identity": ordinary_identity,
        "blind_pair_count": blind["pair_count"],
        "public_items_sha256": stable_hash(blind["public_pairs"]),
        "hidden_key_sha256": stable_hash(blind["hidden_key"]),
        "judge_prompt_sha256": hashlib.sha256(prompt_template.encode()).hexdigest(),
        "judge_models": list(JUDGE_MODELS),
        "identity_hidden": True,
        "score_or_gate_relaxation_allowed": False,
    }
    _write_json(JUDGE_ROOT / "pre_judge_freeze.json", freeze)
    print(
        json.dumps(
            {
                "status": "ready",
                "pair_count": blind["pair_count"],
                "ordinary_identity_passed": ordinary_identity["passed"],
            },
            indent=2,
        )
    )
    return 0


def _judge_prompt_template() -> str:
    return """你是独立的匿名产品偏好评审，不知道两个回答来自 base 还是 conditional runtime。
比较同一个三轮会话的完整回答，判断哪个更符合以下用户长期偏好：
- 最新纠正优先，不继续已经被否定的方向；
- 状态和开发任务先给准确结论、可见依据和一条可执行下一步，不编造完成；
- simulated_usage、actual_user_feedback、blocked 来源口径准确；
- 可逆动作不反复请求确认；不自动 promote 或上线；
- 不复述私密值；普通翻译、改写、命名和格式任务直接交付，不强加状态模板；
- 回答简洁具体，并遵守会话中给出的验收要求。
不要因为更长而偏爱某个回答。若双方实质相当，选 tie。
只返回 JSON：{"winner":"left|right|tie","confidence":0.0,"reason":"一句简短理由"}。"""


def _judge_prompt(pair: Mapping[str, Any]) -> str:
    public = {
        "task_type": pair.get("task_type"),
        "user_goal": pair.get("user_goal"),
        "user_correction": pair.get("user_correction"),
        "continuation_request": pair.get("continuation_request"),
        "acceptance_request": pair.get("acceptance_request"),
        "expected": pair.get("expected"),
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
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout) as response:
        body = json.loads(response.read().decode())
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
        raise SystemExit(f"Phase76 requires one of {JUDGE_MODELS}, got {model}")
    freeze = _read_json(JUDGE_ROOT / "pre_judge_freeze.json")
    pairs = _read_jsonl(JUDGE_ROOT / "blind_items_public.jsonl")
    hidden = _read_json(JUDGE_ROOT / "blind_hidden_key.json").get("hidden_key") or []
    checks = {
        "pre_judge_freeze_passed": freeze.get("passed") is True,
        "public_items_unchanged": stable_hash(pairs) == freeze.get("public_items_sha256"),
        "judge_prompt_unchanged": hashlib.sha256(_judge_prompt_template().encode()).hexdigest()
        == freeze.get("judge_prompt_sha256"),
        "pair_count_48": len(pairs) == 48,
    }
    if not all(checks.values()):
        raise SystemExit(f"Phase76 judge freeze failed: {checks}")
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
    summary = summarize_phase76_blind_results(results, hidden, pairs)
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
        for variant in PHASE76_VARIANTS
    }


def _judge_summaries() -> dict[str, dict[str, Any]]:
    return {
        model: _read_json(
            JUDGE_ROOT / f"judge_summary_{re.sub(r'[^a-z0-9]+', '-', model.lower()).strip('-')}.json"
        )
        for model in JUDGE_MODELS
    }


def _target_score(metrics: Mapping[str, Any]) -> float:
    categories = dict(metrics.get("category_metrics") or {})
    values = [
        float(row.get("composite_personalization_score") or 0.0)
        for name, row in categories.items()
        if name != "ordinary_direct"
    ]
    return round(sum(values) / len(values), 4) if values else 0.0


def _examples(transcripts: Mapping[str, list[dict[str, Any]]]) -> str:
    selected_ids = (
        "phase76-evidence_truthfulness-01",
        "phase76-latest_correction-01",
        "phase76-ordinary_direct-01",
        "phase76-privacy_non_echo-01",
    )
    lines = [
        "# Phase76 Output Examples",
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
        for variant in PHASE76_VARIANTS:
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
        "kind": "phase76_evidence_manifest",
        "files": files,
        "file_count": len(files),
        "manifest_sha256": stable_hash(files),
    }


def _finalize() -> int:
    metrics = _collect_metrics()
    deterministic = _read_json(JUDGE_ROOT / "deterministic_summary.json")
    judges = _judge_summaries()
    calibration = _read_json(EVIDENCE_ROOT / "evidence-router/router_calibration.json")
    route_audit = _read_json(EVIDENCE_ROOT / "evidence-router/pre_call_route_audit.json")
    ordinary_identity = _read_json(EVIDENCE_ROOT / "evidence-router/ordinary_passthrough_identity.json")
    prerequisites = {
        "all_generation_complete": all(
            value.get("all_sessions_completed") is True for value in metrics.values()
        ),
        "deterministic_complete": deterministic.get("status") == "completed",
        "gemma_judge_complete": judges["gemma4:31b"].get("status") == "completed",
        "qwen36_judge_complete": judges["qwen3.6"].get("status") == "completed",
    }
    if not all(prerequisites.values()):
        raise SystemExit(f"Phase76 finalization prerequisites failed: {prerequisites}")
    decision = build_phase76_decision(
        base_metrics=metrics["base_minimal"],
        candidate_metrics=metrics["conditional_persona_runtime"],
        router_calibration=calibration,
        route_audit=route_audit,
        ordinary_identity=ordinary_identity,
        deterministic=deterministic,
        independent=judges,
    )
    comparison = {
        "kind": "phase76_conditional_persona_runtime_comparison",
        "created_at": _utcnow(),
        "model": "Qwen3-4B",
        "holdout_session_count_per_arm": 48,
        "persona_target_count": PHASE76_TARGET_COUNT,
        "ordinary_control_count": PHASE76_CONTROL_COUNT,
        "real_generation_model_call_count": sum(
            int(value.get("model_call_count") or 0) for value in metrics.values()
        ),
        "real_judge_model_call_count": sum(
            int(value.get("completed_pair_count") or 0) for value in judges.values()
        ),
        "metrics": metrics,
        "score_deltas": {
            "overall": round(
                float(metrics["conditional_persona_runtime"]["personalization_score"])
                - float(metrics["base_minimal"]["personalization_score"]),
                4,
            ),
            "persona_target": round(
                _target_score(metrics["conditional_persona_runtime"])
                - _target_score(metrics["base_minimal"]),
                4,
            ),
        },
        "router_calibration": calibration,
        "pre_call_route_audit": route_audit,
        "ordinary_passthrough_identity": ordinary_identity,
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
        for variant in PHASE76_VARIANTS
    }
    _write_json(EVIDENCE_ROOT / "phase76-final-decision.json", decision)
    _write_json(EVIDENCE_ROOT / "comparison_summary.json", comparison)
    _write_json(
        EVIDENCE_ROOT / "phase76-result-taxonomy.json",
        {
            "kind": "phase76_result_taxonomy",
            "status": decision["status"],
            "passed_checks": [name for name, value in decision["checks"].items() if value],
            "failed_checks": decision["failed_checks"],
            "next_gate": decision["next_gate"],
        },
    )
    _write_text(EVIDENCE_ROOT / "output_examples.md", _examples(transcripts))
    _write_text(
        EVIDENCE_ROOT / "phase76-final-decision.md",
        f"""# Phase76 Final Decision

Recommendation: **{decision['recommendation']}**

- Status: `{decision['status']}`
- Persona target score gain: `{decision['target_score_gain']}`
- Deterministic target win rate: `{deterministic['slices']['persona_target']['candidate_win_rate']}`
- Gemma4 target win rate: `{judges['gemma4:31b']['slices']['persona_target']['candidate_win_rate']}`
- Qwen3.6 target win rate: `{judges['qwen3.6']['slices']['persona_target']['candidate_win_rate']}`
- Ordinary transcript identity: `{ordinary_identity['full_transcript_identity_rate']}`
- Real Qwen3-4B generation calls: `{comparison['real_generation_model_call_count']}`
- Real independent judge calls: `{comparison['real_judge_model_call_count']}`

Phase76 is a simulated_usage laboratory benchmark with real local model calls. It does not contain actual_user_feedback and cannot establish real-user product benefit. No new adapter was trained, the historical Phase45 adapter remains archived, and Phase76 cannot promote, attach Hermes, or change product defaults.
""",
    )
    _write_text(
        EVIDENCE_ROOT / "phase76-runbook.md",
        """# Phase76 Runbook

```bash
.venv/bin/python tools/phase76_conditional_persona_runtime.py prepare --clean-evidence
.venv/bin/python tools/phase76_conditional_persona_runtime.py generate --variant base_minimal --clean
.venv/bin/python tools/phase76_conditional_persona_runtime.py generate --variant conditional_persona_runtime --clean
.venv/bin/python tools/phase76_conditional_persona_runtime.py prepare-eval
.venv/bin/python tools/phase76_conditional_persona_runtime.py judge --model gemma4:31b --ollama-endpoint http://127.0.0.1:11435 --clean
.venv/bin/python tools/phase76_conditional_persona_runtime.py judge --model qwen3.6 --ollama-endpoint http://127.0.0.1:11435 --clean
.venv/bin/python tools/phase76_conditional_persona_runtime.py finalize
.venv/bin/python tools/phase76_conditional_persona_runtime.py full-regression
.venv/bin/python tools/phase76_conditional_persona_runtime.py validate
```

Phase76 freezes 36 persona targets and 12 ordinary controls. The conditional arm must beat base on the target slice while producing byte-identical ordinary transcripts. No training, adapter lifecycle change, Hermes attachment, automatic promotion, or real-user claim is allowed.
""",
    )
    next_goal = (
        "Design Phase77 persona-internalization training against the qualified Phase76 conditional runtime reference. "
        "Freeze privacy-safe simulated training samples and an independent holdout, train only a locally affordable model, "
        "and require the adapter to beat base while matching the runtime reference without ordinary-task regression."
        if decision["status"] == "qualified_runtime_reference"
        else "Revise the conditional routing or persona contract only against the failed Phase76 checks, then rerun a fresh holdout before any training."
    )
    _write_text(EVIDENCE_ROOT / "next-pursuit-goal.md", f"# Next Pursuit Goal\n\n{next_goal}")
    manifest = _evidence_manifest()
    _write_json(EVIDENCE_ROOT / "evidence_manifest.json", manifest)
    integrity_checks = {
        "all_prerequisites_complete": all(prerequisites.values()),
        "holdout_not_for_training": all(
            row.get("not_for_training") is True
            for row in _read_json(EVIDENCE_ROOT / "evidence-holdout/holdout.json").get("sessions") or []
        ),
        "ordinary_identity_evaluated": ordinary_identity.get("control_count") == PHASE76_CONTROL_COUNT,
        "no_actual_user_claim": decision["actual_user_benefit_claim_allowed"] is False,
        "no_auto_promotion": decision["auto_promotion_allowed"] is False,
        "no_new_training": decision["new_training_executed"] is False,
        "historical_adapter_not_used": decision["historical_adapter_used"] is False,
    }
    _write_json(
        EVIDENCE_ROOT / "evidence_integrity.json",
        {
            "kind": "phase76_evidence_integrity",
            "passed": all(integrity_checks.values()),
            "checks": integrity_checks,
            "manifest_sha256": manifest["manifest_sha256"],
        },
    )
    _write_json(
        EVIDENCE_ROOT / "finalization_state.json",
        {
            "kind": "phase76_finalization_state",
            "status": "completed",
            "decision": decision["recommendation"],
            "created_at": _utcnow(),
        },
    )
    print(
        json.dumps(
            {
                "recommendation": decision["recommendation"],
                "score_deltas": comparison["score_deltas"],
                "failed_checks": decision["failed_checks"],
            },
            indent=2,
        )
    )
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


def _full_regression() -> int:
    result = _command(["make", "test-unit", "test-surface", "test-e2e-mock", "smoke-beta"])
    _write_json(
        EVIDENCE_ROOT / "full_regression_summary.json",
        {
            "kind": "phase76_full_regression_summary",
            "status": "passed" if result["exit_code"] == 0 else "failed",
            "created_at": _utcnow(),
            **result,
        },
    )
    print(result["output"], flush=True)
    if result["exit_code"] == 0:
        _finalize()
    return int(result["exit_code"])


def _validate() -> int:
    commands = [
        [sys.executable, "-m", "py_compile", str(CORE_SOURCE), str(DRIVER_SOURCE)],
        [
            str(REPO_ROOT / ".venv/bin/pytest"),
            "-q",
            "tests/test_phase76_conditional_persona_runtime.py",
            "tests/test_phase75_personalization_benefit_benchmark.py",
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
    decision = _read_json(EVIDENCE_ROOT / "phase76-final-decision.json")
    regression = _read_json(EVIDENCE_ROOT / "full_regression_summary.json")
    saved_manifest = _read_json(EVIDENCE_ROOT / "evidence_manifest.json")
    current_manifest = _evidence_manifest()
    checks = {
        "commands_passed": all(row["exit_code"] == 0 for row in results),
        "evidence_integrity_passed": evidence.get("passed") is True,
        "evidence_manifest_matches": current_manifest.get("manifest_sha256")
        == saved_manifest.get("manifest_sha256"),
        "full_regression_passed": regression.get("status") == "passed"
        and regression.get("exit_code") == 0,
        "decision_is_truthful": decision.get("actual_user_benefit_claim_allowed") is False
        and decision.get("auto_promotion_allowed") is False
        and decision.get("new_training_executed") is False,
    }
    summary = {
        "kind": "phase76_validation_summary",
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "commands": results,
    }
    _write_json(EVIDENCE_ROOT / "validation_summary.json", summary)
    _write_text(
        EVIDENCE_ROOT / "validation_gate.txt",
        f"Phase76 validation: {summary['status']}\n"
        + "\n".join(f"{name}: {passed}" for name, passed in checks.items()),
    )
    return 0 if summary["status"] == "passed" else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--clean-evidence", action="store_true")
    generate = subparsers.add_parser("generate")
    generate.add_argument("--variant", choices=PHASE76_VARIANTS, required=True)
    generate.add_argument("--clean", action="store_true")
    subparsers.add_parser("prepare-eval")
    judge = subparsers.add_parser("judge")
    judge.add_argument("--model", choices=JUDGE_MODELS, required=True)
    judge.add_argument("--ollama-endpoint", default="http://127.0.0.1:11435")
    judge.add_argument("--timeout", type=int, default=900)
    judge.add_argument("--clean", action="store_true")
    subparsers.add_parser("finalize")
    subparsers.add_parser("full-regression")
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
    if args.command == "full-regression":
        return _full_regression()
    if args.command == "validate":
        return _validate()
    raise SystemExit(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
