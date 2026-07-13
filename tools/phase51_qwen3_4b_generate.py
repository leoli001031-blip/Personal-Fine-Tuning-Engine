#!/usr/bin/env python3
"""Generate fresh Phase51 three-arm Qwen3-4B runtime transcripts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = REPO_ROOT / "pfe-core"
TOOLS_ROOT = REPO_ROOT / "tools"
for path in (CORE_ROOT, TOOLS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pfe_core.phase46_runtime_first_latest_intent import stable_hash
from pfe_core.phase50_conditional_provenance_guard import aggregate_phase50_variant
from phase46_qwen3_4b_generate import (
    _load_runtime,
    _read_json,
    _read_jsonl,
    _sha256,
    _write_json,
    _write_jsonl_atomic,
)
from phase50_qwen3_4b_generate import _run_session


EVIDENCE_ROOT = REPO_ROOT / "docs" / "demo" / "phase51-dual-evaluator-hardening"
HOLDOUT_PATH = EVIDENCE_ROOT / "evidence-runtime-holdout" / "holdout.json"
PROTOCOL_PATH = EVIDENCE_ROOT / "evaluator_runtime_protocol.json"
FREEZE_PATH = EVIDENCE_ROOT / "pre_model_call_freeze.json"
PHASE50_SOURCE = CORE_ROOT / "pfe_core" / "phase50_conditional_provenance_guard.py"
PHASE51_SOURCE = CORE_ROOT / "pfe_core" / "phase51_dual_evaluator_hardening.py"
REFERENCE_TARGETS_PATH = (
    REPO_ROOT
    / "docs"
    / "demo"
    / "phase47-simulated-user-review"
    / "evidence-candidates"
    / "reviewed_candidates.jsonl"
)
VARIANTS = ("base_compact_v1", "base_global_v2", "base_conditional_guard")


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _freeze_check(sessions: list[dict[str, Any]]) -> dict[str, Any]:
    freeze = _read_json(FREEZE_PATH)
    protocol = _read_json(PROTOCOL_PATH)
    protocol_hash = str(protocol.get("protocol_sha256") or "")
    protocol_copy = {key: value for key, value in protocol.items() if key != "protocol_sha256"}
    calibration = _read_json(
        EVIDENCE_ROOT / "evidence-evaluator-calibration" / "dual_evaluator_report.json"
    )
    holdout = _read_json(
        EVIDENCE_ROOT / "evidence-evaluator-holdout" / "dual_evaluator_report.json"
    )
    checks = {
        "phase50_runtime_source": freeze.get("phase50_runtime_source_sha256") == _sha256(PHASE50_SOURCE),
        "phase51_evaluator_source": freeze.get("phase51_evaluator_source_sha256") == _sha256(PHASE51_SOURCE),
        "runtime_holdout": freeze.get("runtime_holdout_sha256") == stable_hash(sessions),
        "protocol": bool(protocol_hash)
        and stable_hash(protocol_copy) == protocol_hash == freeze.get("protocol_sha256"),
        "calibration_qualified": calibration.get("status") == "qualified",
        "evaluator_holdout_qualified": holdout.get("status") == "qualified",
        "phase50_runtime_unchanged": dict(protocol.get("runtime") or {}).get(
            "phase50_router_and_prompts_unchanged"
        ) is True,
    }
    return {
        "kind": "phase51_runtime_generation_freeze_check",
        "passed": all(checks.values()),
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    sessions = [dict(row) for row in _read_json(HOLDOUT_PATH).get("sessions") or []]
    freeze = _freeze_check(sessions)
    if freeze["passed"] is not True:
        raise SystemExit(f"Phase51 runtime generation freeze failed: {freeze}")
    full_protocol = _read_json(PROTOCOL_PATH)
    runtime_protocol = dict(full_protocol.get("runtime") or {})
    runtime_protocol["protocol_sha256"] = str(full_protocol.get("protocol_sha256") or "")
    output_dir = EVIDENCE_ROOT / "evidence-real-runtime"
    output_path = output_dir / f"transcripts_{args.variant}.jsonl"
    metrics_path = output_dir / f"metrics_{args.variant}.json"
    freeze_path = output_dir / f"freeze_check_{args.variant}.json"
    if args.clean:
        output_path.unlink(missing_ok=True)
        metrics_path.unlink(missing_ok=True)
    _write_json(freeze_path, freeze)
    existing = [] if args.clean else _read_jsonl(output_path)
    wanted = {str(row.get("session_id") or "") for row in sessions}
    transcripts = [row for row in existing if str(row.get("session_id") or "") in wanted]
    completed = {
        str(row.get("session_id") or "")
        for row in transcripts
        if row.get("status") == "completed"
    }

    torch, tokenizer, model, device, runtime_info = _load_runtime(None)
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
                    protocol=runtime_protocol,
                )
                transcript.update(
                    {
                        "kind": "phase51_fresh_real_runtime_transcript",
                        "phase51_fresh_holdout": True,
                        "prior_phase_transcript_reused": False,
                    }
                )
            except Exception as exc:
                transcript = {
                    "kind": "phase51_fresh_real_runtime_transcript",
                    "session_id": session.get("session_id"),
                    "category": session.get("category"),
                    "variant": args.variant,
                    "model_id": str(REPO_ROOT / "models" / "Qwen3-4B"),
                    "adapter_loaded": False,
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
            transcripts = [
                row for row in transcripts if row.get("session_id") != transcript.get("session_id")
            ]
            transcripts.append(transcript)
            transcripts.sort(key=lambda row: str(row.get("session_id") or ""))
            _write_jsonl_atomic(output_path, transcripts)
            print(f"[{args.variant}] {index}/{len(sessions)} {session_id} {transcript['status']}", flush=True)
    finally:
        try:
            del model
            if device == "mps":
                torch.mps.empty_cache()
        except Exception:
            pass

    targets = [str(row.get("chosen") or "") for row in _read_jsonl(REFERENCE_TARGETS_PATH)]
    report = aggregate_phase50_variant(transcripts, sessions, reference_targets=targets)
    persistence = [
        dict(row.get("privacy_persistence_check") or {}).get("passed") is True
        for row in transcripts
        if row.get("status") == "completed"
    ]
    route_exact = [
        dict(row.get("phase50_runtime") or {}).get("route_sequence_exact") is True
        for row in transcripts
        if row.get("status") == "completed"
    ]
    report.update(
        {
            "kind": "phase51_fresh_runtime_variant_metrics",
            "variant": args.variant,
            "model_id": str(REPO_ROOT / "models" / "Qwen3-4B"),
            "adapter_loaded": False,
            "runtime_info": runtime_info,
            "all_transcripts_completed": len(transcripts) == len(sessions)
            and all(row.get("status") == "completed" for row in transcripts),
            "actual_model_calls": len(transcripts) == len(sessions)
            and all(row.get("actual_model_call") is True for row in transcripts),
            "privacy_persistence_checks_passed": all(persistence),
            "router_sequence_checks_passed": all(route_exact),
            "transcript_path": str(output_path),
            "freeze_check": freeze,
            "runtime_protocol": runtime_protocol,
            "model_call_count": sum(
                len(row.get("generation") or [])
                for row in transcripts
                if row.get("actual_model_call") is True
            ),
            "think_leak_rate": round(
                sum(bool(row.get("think_leak_detected")) for row in transcripts) / len(transcripts), 4
            ) if transcripts else 0.0,
            "actual_user_feedback": False,
            "simulated_usage": True,
            "actual_product_benefit_claim_allowed": False,
            "created_at": _utcnow(),
        }
    )
    _write_json(metrics_path, report)
    print(
        json.dumps(
            {
                key: report.get(key)
                for key in (
                    "variant",
                    "session_count",
                    "model_call_count",
                    "user_preference_score",
                    "latest_intent_satisfaction_rate",
                    "provenance_boundary_rate",
                    "unsupported_product_benefit_claim_rate",
                    "nontrigger_latest_intent_satisfaction_rate",
                    "privacy_violation_rate",
                    "secret_echo_rate",
                    "repetition_rate",
                    "all_transcripts_completed",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report["all_transcripts_completed"] and report["actual_model_calls"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
