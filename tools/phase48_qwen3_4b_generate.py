#!/usr/bin/env python3
"""Generate real Qwen3-4B Phase48 compact-runtime ablation transcripts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = REPO_ROOT / "pfe-core"
TOOLS_ROOT = REPO_ROOT / "tools"
for path in (CORE_ROOT, TOOLS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pfe_core.phase45_privacy_multiturn_preference import sanitize_privacy_output, transform_privacy_messages
from pfe_core.phase46_runtime_first_latest_intent import build_latest_intent_envelope, stable_hash
from pfe_core.phase48_compact_intent_runtime import aggregate_phase48_variant, build_phase48_compact_runtime_messages
from phase46_qwen3_4b_generate import (
    _aggregate_privacy_manifests,
    _generate_raw,
    _load_runtime,
    _read_json,
    _read_jsonl,
    _sha256,
    _strip_thinking,
    _write_json,
    _write_jsonl_atomic,
)


EVIDENCE_ROOT = REPO_ROOT / "docs" / "demo" / "phase48-compact-intent-runtime-ablation"
PHASE47_ROOT = REPO_ROOT / "docs" / "demo" / "phase47-simulated-user-review"
HOLDOUT_PATH = EVIDENCE_ROOT / "evidence-holdout" / "holdout.json"
HOLDOUT_FREEZE_PATH = EVIDENCE_ROOT / "evidence-holdout" / "holdout_freeze.json"
PROTOCOL_PATH = EVIDENCE_ROOT / "evidence-holdout" / "runtime_protocol.json"
SCORER_FREEZE_PATH = EVIDENCE_ROOT / "evidence-scorer-calibration" / "scorer_freeze.json"
PHASE45_SOURCE = CORE_ROOT / "pfe_core" / "phase45_privacy_multiturn_preference.py"
PHASE46_SOURCE = CORE_ROOT / "pfe_core" / "phase46_runtime_first_latest_intent.py"
PHASE48_SOURCE = CORE_ROOT / "pfe_core" / "phase48_compact_intent_runtime.py"
REFERENCE_TARGETS_PATH = PHASE47_ROOT / "evidence-candidates" / "reviewed_candidates.jsonl"
VARIANTS = ("base_privacy", "base_compact_intent", "base_full_intent")


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _mode(variant: str) -> str:
    return {
        "base_privacy": "privacy_base",
        "base_compact_intent": "compact_system_instruction",
        "base_full_intent": "phase46_full_envelope",
    }[variant]


def _freeze_check(sessions: list[dict[str, Any]]) -> dict[str, Any]:
    scorer = _read_json(SCORER_FREEZE_PATH)
    source_checks = {
        "phase45_privacy_source": scorer.get("phase45_privacy_source_sha256") == _sha256(PHASE45_SOURCE),
        "phase46_scorer_source": scorer.get("phase46_scorer_source_sha256") == _sha256(PHASE46_SOURCE),
        "phase48_runtime_source": scorer.get("phase48_runtime_source_sha256") == _sha256(PHASE48_SOURCE),
    }
    scorer_ok = all(source_checks.values()) and scorer.get("calibration_status") == "passed"
    holdout = _read_json(HOLDOUT_FREEZE_PATH)
    holdout_current = stable_hash(sessions)
    holdout_ok = holdout.get("holdout_manifest_sha256") == holdout_current and holdout.get("frozen_before_model_calls") is True
    protocol = _read_json(PROTOCOL_PATH)
    protocol_hash = str(protocol.get("protocol_sha256") or "")
    protocol_copy = {key: value for key, value in protocol.items() if key != "protocol_sha256"}
    protocol_ok = bool(protocol_hash) and stable_hash(protocol_copy) == protocol_hash and int(protocol.get("max_new_tokens") or 0) == 384
    return {
        "kind": "phase48_generation_freeze_check",
        "passed": scorer_ok and holdout_ok and protocol_ok,
        "source_checks": source_checks,
        "scorer_passed": scorer_ok,
        "holdout_expected_sha256": holdout.get("holdout_manifest_sha256"),
        "holdout_current_sha256": holdout_current,
        "holdout_passed": holdout_ok,
        "protocol_sha256": protocol_hash,
        "protocol_passed": protocol_ok,
    }


def _runtime_messages(
    raw_history: list[dict[str, str]],
    variant: str,
) -> tuple[list[dict[str, str]], Any, dict[str, Any]]:
    if variant == "base_compact_intent":
        result = build_phase48_compact_runtime_messages(raw_history)
        manifest = dict(result.manifest)
        return result.messages, result.privacy, manifest
    privacy = transform_privacy_messages(raw_history)
    if variant == "base_full_intent":
        messages, intent = build_latest_intent_envelope(privacy.messages)
        manifest = {
            "kind": "phase48_full_intent_manifest",
            "runtime_mode": "phase46_full_envelope",
            "latest_intent_wrapped": True,
            "phase46_intent_manifest": intent,
        }
        return messages, privacy, manifest
    return privacy.messages, privacy, {
        "kind": "phase48_privacy_base_manifest",
        "runtime_mode": "privacy_base",
        "latest_intent_wrapped": False,
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
) -> dict[str, Any]:
    system = str(protocol.get("length_contract") or "")
    if variant == "base_full_intent":
        system = f"{system}\n{protocol.get('full_intent_contract') or ''}".strip()
    raw_history: list[dict[str, str]] = [{"role": "system", "content": system}]
    persisted_turns: list[dict[str, str]] = []
    generations: list[dict[str, Any]] = []
    privacy_manifests: list[dict[str, Any]] = []
    runtime_manifests: list[dict[str, Any]] = []
    output_audits: list[dict[str, Any]] = []
    actual_system_sha256 = ""
    user_turns = [
        str(session.get("user_goal") or ""),
        str(session.get("user_correction") or ""),
        f"{session.get('continuation_request') or ''}\n{session.get('acceptance_request') or ''}".strip(),
    ]
    for turn_index, user_text in enumerate(user_turns, start=1):
        raw_history.append({"role": "user", "content": user_text})
        model_messages, privacy, runtime_manifest = _runtime_messages(raw_history, variant)
        if not actual_system_sha256:
            actual_system = next(
                (str(message.get("content") or "") for message in model_messages if message.get("role") == "system"),
                "",
            )
            actual_system_sha256 = hashlib.sha256(actual_system.encode("utf-8")).hexdigest()
        privacy_manifests.append(privacy.manifest)
        runtime_manifests.append(runtime_manifest)
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
    latest_wrapped = variant == "base_full_intent"
    transcript: dict[str, Any] = {
        "kind": "phase48_real_compact_runtime_ablation_transcript",
        "session_id": session.get("session_id"),
        "category": session.get("category"),
        "variant": variant,
        "model_id": str(REPO_ROOT / "models" / "Qwen3-4B"),
        "adapter_loaded": False,
        "privacy_runtime_enabled": True,
        "runtime_mode": _mode(variant),
        "system_contract_sha256": actual_system_sha256,
        "runtime_protocol_sha256": str(protocol.get("protocol_sha256") or ""),
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
            "intent_manifest": {"latest_intent_wrapped": latest_wrapped},
            "old_messages_removed": False,
        },
        "phase48_runtime": {
            "mode": _mode(variant),
            "history_preserved": True,
            "runtime_manifests": runtime_manifests,
            "compact_xml_or_tag_envelope_used": False if variant == "base_compact_intent" else None,
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
        raise SystemExit(f"Phase48 scorer/holdout/protocol freeze failed: {freeze}")
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

    torch, tokenizer, model, device, _runtime = _load_runtime(None)
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
                )
            except Exception as exc:
                transcript = {
                    "kind": "phase48_real_compact_runtime_ablation_transcript",
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

    reference_targets = [str(row.get("chosen") or "") for row in _read_jsonl(REFERENCE_TARGETS_PATH)]
    report = aggregate_phase48_variant(transcripts, sessions, reference_targets=reference_targets)
    persistence = [
        dict(row.get("privacy_persistence_check") or {}).get("passed") is True
        for row in transcripts
        if row.get("status") == "completed"
    ]
    report.update(
        {
            "variant": args.variant,
            "model_id": str(REPO_ROOT / "models" / "Qwen3-4B"),
            "adapter_loaded": False,
            "privacy_runtime_enabled": True,
            "runtime_mode": _mode(args.variant),
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
    print(json.dumps({key: report.get(key) for key in (
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
    )}, ensure_ascii=False, indent=2))
    passed = (
        report.get("actual_model_calls") is True
        and report.get("all_transcripts_completed") is True
        and report.get("privacy_persistence_checks_passed") is True
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
