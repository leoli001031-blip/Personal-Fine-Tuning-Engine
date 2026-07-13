#!/usr/bin/env python3
"""Generate real Qwen3-4B Phase50 conditional provenance transcripts."""

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

from pfe_core.phase45_privacy_multiturn_preference import sanitize_privacy_output
from pfe_core.phase48_compact_intent_runtime import build_phase48_compact_runtime_messages
from pfe_core.phase49_provenance_boundary_recovery import build_phase49_compact_v2_messages
from pfe_core.phase50_conditional_provenance_guard import (
    aggregate_phase50_variant,
    build_phase50_conditional_messages,
    route_phase50_provenance_guard,
)
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


EVIDENCE_ROOT = REPO_ROOT / "docs" / "demo" / "phase50-conditional-provenance-guard"
PHASE47_ROOT = REPO_ROOT / "docs" / "demo" / "phase47-simulated-user-review"
HOLDOUT_PATH = EVIDENCE_ROOT / "evidence-holdout" / "holdout.json"
HOLDOUT_FREEZE_PATH = EVIDENCE_ROOT / "evidence-holdout" / "holdout_freeze.json"
PROTOCOL_PATH = EVIDENCE_ROOT / "evidence-holdout" / "runtime_protocol.json"
ROUTER_FREEZE_PATH = EVIDENCE_ROOT / "evidence-router" / "router_freeze.json"
PHASE45_SOURCE = CORE_ROOT / "pfe_core" / "phase45_privacy_multiturn_preference.py"
PHASE46_SOURCE = CORE_ROOT / "pfe_core" / "phase46_runtime_first_latest_intent.py"
PHASE48_SOURCE = CORE_ROOT / "pfe_core" / "phase48_compact_intent_runtime.py"
PHASE49_SOURCE = CORE_ROOT / "pfe_core" / "phase49_provenance_boundary_recovery.py"
PHASE50_SOURCE = CORE_ROOT / "pfe_core" / "phase50_conditional_provenance_guard.py"
REFERENCE_TARGETS_PATH = PHASE47_ROOT / "evidence-candidates" / "reviewed_candidates.jsonl"
VARIANTS = ("base_compact_v1", "base_global_v2", "base_conditional_guard")


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _mode(variant: str) -> str:
    return {
        "base_compact_v1": "compact_v1_latest_intent",
        "base_global_v2": "global_v2_evidence_boundary",
        "base_conditional_guard": "conditional_provenance_router",
    }[variant]


def _freeze_check(sessions: list[dict[str, Any]]) -> dict[str, Any]:
    freeze = _read_json(ROUTER_FREEZE_PATH)
    source_checks = {
        "phase45_privacy_source": freeze.get("phase45_privacy_source_sha256") == _sha256(PHASE45_SOURCE),
        "phase46_generic_scorer_source": freeze.get("phase46_generic_scorer_source_sha256") == _sha256(PHASE46_SOURCE),
        "phase48_compact_v1_source": freeze.get("phase48_compact_v1_source_sha256") == _sha256(PHASE48_SOURCE),
        "phase49_provenance_scorer_source": freeze.get("phase49_provenance_scorer_source_sha256") == _sha256(PHASE49_SOURCE),
        "phase50_router_source": freeze.get("phase50_router_source_sha256") == _sha256(PHASE50_SOURCE),
    }
    router_ok = (
        all(source_checks.values())
        and freeze.get("calibration_status") == "passed"
        and float(freeze.get("calibration_exact_decision_accuracy") or 0.0) == 1.0
        and freeze.get("provenance_scorer_calibration_status") == "passed"
        and float(freeze.get("provenance_scorer_exact_label_accuracy") or 0.0) == 1.0
        and freeze.get("premodel_holdout_status") == "passed"
        and float(freeze.get("premodel_sequence_exact_rate") or 0.0) == 1.0
    )
    from pfe_core.phase46_runtime_first_latest_intent import stable_hash

    holdout = _read_json(HOLDOUT_FREEZE_PATH)
    holdout_current = stable_hash(sessions)
    holdout_ok = (
        holdout.get("holdout_manifest_sha256") == holdout_current
        and holdout.get("frozen_before_model_calls") is True
    )
    protocol = _read_json(PROTOCOL_PATH)
    protocol_hash = str(protocol.get("protocol_sha256") or "")
    protocol_copy = {key: value for key, value in protocol.items() if key != "protocol_sha256"}
    protocol_ok = (
        bool(protocol_hash)
        and stable_hash(protocol_copy) == protocol_hash
        and int(protocol.get("max_new_tokens") or 0) == 384
        and protocol.get("conditional_requires_source_and_outcome_axes") is True
    )
    return {
        "kind": "phase50_generation_freeze_check",
        "passed": router_ok and holdout_ok and protocol_ok,
        "source_checks": source_checks,
        "router_passed": router_ok,
        "holdout_expected_sha256": holdout.get("holdout_manifest_sha256"),
        "holdout_current_sha256": holdout_current,
        "holdout_passed": holdout_ok,
        "protocol_sha256": protocol_hash,
        "protocol_passed": protocol_ok,
    }


def _runtime_messages(
    raw_history: list[dict[str, str]],
    variant: str,
) -> tuple[list[dict[str, str]], Any, dict[str, Any], dict[str, Any]]:
    if variant == "base_conditional_guard":
        result = build_phase50_conditional_messages(raw_history)
        route = dict(result.manifest["route"])
        return result.messages, result.privacy, dict(result.manifest), route
    if variant == "base_global_v2":
        result = build_phase49_compact_v2_messages(raw_history)
        route = route_phase50_provenance_guard(result.messages)
        return result.messages, result.privacy, {
            "kind": "phase50_global_v2_manifest",
            "runtime_mode": "global_v2_evidence_boundary",
            "phase49_manifest": result.manifest,
        }, route
    result = build_phase48_compact_runtime_messages(raw_history)
    route = route_phase50_provenance_guard(result.messages)
    return result.messages, result.privacy, {
        "kind": "phase50_compact_v1_manifest",
        "runtime_mode": "compact_v1_latest_intent",
        "phase48_manifest": result.manifest,
    }, route


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
    raw_history: list[dict[str, str]] = [
        {"role": "system", "content": str(protocol.get("length_contract") or "")}
    ]
    persisted_turns: list[dict[str, str]] = []
    generations: list[dict[str, Any]] = []
    privacy_manifests: list[dict[str, Any]] = []
    runtime_manifests: list[dict[str, Any]] = []
    route_decisions: list[dict[str, Any]] = []
    output_audits: list[dict[str, Any]] = []
    system_hashes: list[str] = []
    user_turns = [
        str(session.get("user_goal") or ""),
        str(session.get("user_correction") or ""),
        f"{session.get('continuation_request') or ''}\n{session.get('acceptance_request') or ''}".strip(),
    ]
    for turn_index, user_text in enumerate(user_turns, start=1):
        raw_history.append({"role": "user", "content": user_text})
        model_messages, privacy, runtime_manifest, route = _runtime_messages(raw_history, variant)
        actual_system = next(
            (str(message.get("content") or "") for message in model_messages if message.get("role") == "system"),
            "",
        )
        system_hashes.append(hashlib.sha256(actual_system.encode("utf-8")).hexdigest())
        privacy_manifests.append(privacy.manifest)
        runtime_manifests.append(runtime_manifest)
        route_decisions.append({"turn": turn_index, **route})
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
                "system_contract_sha256": system_hashes[-1],
                "router_activate_guard": route["activate_guard"],
            }
        )
        assistant = {"role": "assistant", "content": cleaned}
        persisted_turns.append(assistant)
        raw_history.append(assistant)
        generations.append(info)
    activation_sequence = [bool(row["activate_guard"]) for row in route_decisions]
    transcript: dict[str, Any] = {
        "kind": "phase50_real_conditional_provenance_transcript",
        "session_id": session.get("session_id"),
        "category": session.get("category"),
        "variant": variant,
        "model_id": str(REPO_ROOT / "models" / "Qwen3-4B"),
        "adapter_loaded": False,
        "privacy_runtime_enabled": True,
        "runtime_mode": _mode(variant),
        "system_contract_sha256_by_turn": system_hashes,
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
            "intent_manifest": {"latest_intent_wrapped": False},
            "old_messages_removed": False,
        },
        "phase50_runtime": {
            "mode": _mode(variant),
            "history_preserved": True,
            "runtime_manifests": runtime_manifests,
            "route_decisions": route_decisions,
            "route_activation_sequence": activation_sequence,
            "expected_activation_sequence": list(session.get("expected_activation_sequence") or []),
            "route_sequence_exact": activation_sequence
            == list(session.get("expected_activation_sequence") or []),
            "router_action_applied": variant == "base_conditional_guard",
            "xml_or_tag_envelope_used": False,
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
    sessions = [dict(row) for row in _read_json(HOLDOUT_PATH).get("sessions") or []]
    freeze = _freeze_check(sessions)
    if freeze["passed"] is not True:
        raise SystemExit(f"Phase50 router/holdout/protocol freeze failed: {freeze}")
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
    wanted = {str(row.get("session_id") or "") for row in sessions}
    transcripts = [row for row in existing if str(row.get("session_id") or "") in wanted]
    completed = {str(row.get("session_id") or "") for row in transcripts if row.get("status") == "completed"}

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
                    "kind": "phase50_real_conditional_provenance_transcript",
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

    reference_targets = [str(row.get("chosen") or "") for row in _read_jsonl(REFERENCE_TARGETS_PATH)]
    report = aggregate_phase50_variant(transcripts, sessions, reference_targets=reference_targets)
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
            "variant": args.variant,
            "model_id": str(REPO_ROOT / "models" / "Qwen3-4B"),
            "adapter_loaded": False,
            "privacy_runtime_enabled": True,
            "runtime_mode": _mode(args.variant),
            "all_transcripts_completed": len(transcripts) == len(sessions)
            and all(row.get("status") == "completed" for row in transcripts),
            "privacy_persistence_checks_passed": all(persistence),
            "router_sequence_checks_passed": all(route_exact),
            "transcript_path": str(output_path),
            "freeze_check": freeze,
            "runtime_protocol": protocol,
            "model_call_count": sum(
                len(row.get("generation") or []) for row in transcripts if row.get("actual_model_call") is True
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
                    "response_diversity",
                    "repetition_rate",
                    "truncated_response_rate",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    passed = (
        report.get("actual_model_calls") is True
        and report.get("all_transcripts_completed") is True
        and report.get("privacy_persistence_checks_passed") is True
        and report.get("router_sequence_checks_passed") is True
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
