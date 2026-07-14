#!/usr/bin/env python3
"""Generate real Qwen3-4B transcripts for one frozen Phase69 arm."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = REPO_ROOT / "pfe-core"
TOOLS_ROOT = REPO_ROOT / "tools"
for root in (CORE_ROOT, TOOLS_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from pfe_core.phase45_privacy_multiturn_preference import sanitize_privacy_output
from pfe_core.phase46_runtime_first_latest_intent import (
    PHASE46_LATEST_INTENT_CONTRACT,
    PHASE46_LENGTH_CONTRACT,
)
from pfe_core.phase69_minimal_runtime_ab import (
    PHASE69_VARIANTS,
    build_phase69_runtime_messages,
    stable_hash,
)
from phase46_qwen3_4b_generate import (
    MODEL_PATH,
    _aggregate_privacy_manifests,
    _generate_raw,
    _load_runtime,
    _strip_thinking,
)


EVIDENCE_ROOT = REPO_ROOT / "docs/demo/phase69-minimal-runtime-ab"


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


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl_atomic(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _freeze_check(holdout: Mapping[str, Any], protocol: Mapping[str, Any]) -> dict[str, Any]:
    freeze = _read_json(EVIDENCE_ROOT / "pre_model_call_freeze.json")
    source_ok = all(
        (REPO_ROOT / path).is_file() and _sha256(REPO_ROOT / path) == expected
        for name, expected in dict(freeze.get("source_sha256") or {}).items()
        for path in [
            {
                "phase46_generator_helpers": "tools/phase46_qwen3_4b_generate.py",
                "phase53_hard_detector": "pfe-core/pfe_core/phase53_evaluator_scope_recovery.py",
                "phase56_grounder_composer": "pfe-core/pfe_core/phase56_evidence_span_grounded_atomic.py",
                "phase59_candidates": "pfe-core/pfe_core/phase59_proposition_addressed_grounding.py",
                "phase62_consensus": "pfe-core/pfe_core/phase62_risk_asymmetric_candidate_consensus.py",
                "phase63_wire": "pfe-core/pfe_core/phase63_field_typed_candidate_wire.py",
                "phase69_core": "pfe-core/pfe_core/phase69_minimal_runtime_ab.py",
                "phase69_prepare": "tools/phase69_prepare.py",
                "phase69_generate": "tools/phase69_generate.py",
                "phase69_prepare_eval": "tools/phase69_prepare_eval.py",
                "phase69_execute_eval": "tools/phase69_execute_eval.py",
                "phase69_finalize": "tools/phase69_finalize_evidence.py",
            }[name]
        ]
    )
    protocol_copy = {key: value for key, value in protocol.items() if key != "protocol_sha256"}
    return {
        "kind": "phase69_generation_freeze_check",
        "passed": (
            source_ok
            and stable_hash(holdout) == freeze.get("holdout_sha256")
            and stable_hash(protocol_copy)
            == protocol.get("protocol_sha256")
            == freeze.get("protocol_sha256")
            and _read_json(EVIDENCE_ROOT / "preparation_decision.json").get("status")
            == "ready_for_real_generation"
        ),
        "source_checks_passed": source_ok,
        "holdout_check_passed": stable_hash(holdout) == freeze.get("holdout_sha256"),
        "protocol_check_passed": stable_hash(protocol_copy)
        == protocol.get("protocol_sha256")
        == freeze.get("protocol_sha256"),
    }


def _run_session(
    *,
    session: Mapping[str, Any],
    variant: str,
    torch: Any,
    tokenizer: Any,
    model: Any,
    device: str,
    generation_protocol: Mapping[str, Any],
) -> dict[str, Any]:
    system = f"{PHASE46_LENGTH_CONTRACT}\n{PHASE46_LATEST_INTENT_CONTRACT}".strip()
    raw_history: list[dict[str, str]] = [{"role": "system", "content": system}]
    persisted_turns: list[dict[str, str]] = []
    generations = []
    privacy_manifests = []
    output_audits = []
    runtime_manifests = []
    user_turns = [
        str(session.get("user_goal") or ""),
        str(session.get("user_correction") or ""),
        f"{session.get('continuation_request') or ''}\n{session.get('acceptance_request') or ''}".strip(),
    ]
    for turn_index, user_text in enumerate(user_turns, start=1):
        raw_history.append({"role": "user", "content": user_text})
        runtime = build_phase69_runtime_messages(raw_history, variant=variant)
        privacy_manifests.append(runtime.privacy.manifest)
        runtime_manifests.append(runtime.manifest)
        persisted_turns.append(dict(runtime.messages[-1]))
        raw_output, info = _generate_raw(
            torch=torch,
            tokenizer=tokenizer,
            model=model,
            device=device,
            messages=runtime.messages,
            protocol=generation_protocol,
        )
        persisted_raw, audit = sanitize_privacy_output(raw_output, runtime.privacy)
        output_audits.append({"turn": turn_index, **audit})
        cleaned, think_leak = _strip_thinking(persisted_raw)
        if not cleaned:
            raise RuntimeError("Phase69 output became empty after privacy handling")
        info.update(
            {
                "turn": turn_index,
                "raw_content": persisted_raw,
                "raw_content_sanitized_before_persistence": True,
                "raw_output_sha256_before_sanitization": audit[
                    "raw_output_sha256_before_sanitization"
                ],
                "output_redaction_count": audit["output_redaction_count"],
                "think_leak_detected": think_leak,
            }
        )
        assistant = {"role": "assistant", "content": cleaned}
        persisted_turns.append(assistant)
        raw_history.append(assistant)
        generations.append(info)
    return {
        "kind": "phase69_real_runtime_ab_transcript",
        "session_id": session.get("session_id"),
        "task_type": session.get("task_type"),
        "category": session.get("category"),
        "variant": variant,
        "model_id": str(MODEL_PATH),
        "device": device,
        "adapter_loaded": False,
        "actual_model_call": True,
        "hardcoded_response": False,
        "status": "completed",
        "turns": persisted_turns,
        "generation": generations,
        "latency_seconds": [row["latency_seconds"] for row in generations],
        "truncated_response": any(row["truncated"] for row in generations),
        "think_leak_detected": any(row["think_leak_detected"] for row in generations),
        "privacy_runtime_enabled": True,
        "latest_intent_runtime_enabled": True,
        "candidate_contract_enabled": variant == "candidate_boundary_contract",
        "only_ab_variable": "candidate_provenance_boundary_contract",
        "generation_protocol_sha256": generation_protocol.get(
            "generation_protocol_sha256"
        ),
        "task_sha256": stable_hash(session),
        "system_contract_sha256": hashlib.sha256(
            runtime.messages[0]["content"].encode("utf-8")
        ).hexdigest(),
        "runtime_manifests": runtime_manifests,
        "privacy_runtime": {
            "input_manifest": _aggregate_privacy_manifests(privacy_manifests),
            "input_manifests": privacy_manifests,
            "output_audits": output_audits,
            "raw_private_values_entered_model": False,
            "raw_private_values_persisted": False,
        },
        "simulated_usage": True,
        "actual_user_feedback": False,
        "not_for_training": True,
        "actual_product_benefit_claim_allowed": False,
        "created_at": _utcnow(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=PHASE69_VARIANTS, required=True)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    holdout = _read_json(EVIDENCE_ROOT / "evidence-holdout/holdout.json")
    protocol = _read_json(EVIDENCE_ROOT / "runtime_ab_protocol.json")
    freeze = _freeze_check(holdout, protocol)
    output_dir = EVIDENCE_ROOT / "evidence-real-generation"
    _write_json(output_dir / f"freeze_check_{args.variant}.json", freeze)
    if freeze["passed"] is not True:
        raise SystemExit(f"Phase69 generation freeze failed: {freeze}")

    output_path = output_dir / f"transcripts_{args.variant}.jsonl"
    metrics_path = output_dir / f"metrics_{args.variant}.json"
    if args.clean:
        output_path.unlink(missing_ok=True)
        metrics_path.unlink(missing_ok=True)
    sessions = [dict(row) for row in holdout.get("sessions") or []]
    existing = [] if args.clean else _read_jsonl(output_path)
    by_id = {
        str(row.get("session_id")): row
        for row in existing
        if row.get("status") == "completed"
    }
    generation_protocol = dict(protocol.get("generation") or {})
    torch, tokenizer, model, device, runtime = _load_runtime(None)
    if runtime.get("adapter_loaded") is not False:
        raise SystemExit("Phase69 runtime A/B must not load an adapter")
    try:
        for index, session in enumerate(sessions, start=1):
            session_id = str(session.get("session_id") or "")
            if session_id in by_id:
                continue
            try:
                transcript = _run_session(
                    session=session,
                    variant=args.variant,
                    torch=torch,
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                    generation_protocol=generation_protocol,
                )
            except Exception as exc:
                transcript = {
                    "kind": "phase69_real_runtime_ab_transcript",
                    "session_id": session_id,
                    "task_type": session.get("task_type"),
                    "category": session.get("category"),
                    "variant": args.variant,
                    "model_id": str(MODEL_PATH),
                    "device": device,
                    "actual_model_call": False,
                    "hardcoded_response": False,
                    "status": "failed",
                    "error": f"{exc.__class__.__name__}: {exc}",
                    "turns": [],
                    "generation": [],
                    "latency_seconds": [],
                    "simulated_usage": True,
                    "actual_user_feedback": False,
                    "not_for_training": True,
                    "created_at": _utcnow(),
                }
            by_id[session_id] = transcript
            rows = [by_id[key] for key in sorted(by_id)]
            _write_jsonl_atomic(output_path, rows)
            if transcript["status"] != "completed" or index % 6 == 0 or index == len(sessions):
                print(
                    f"[{args.variant}] {index}/{len(sessions)} {session_id} "
                    f"status={transcript['status']}",
                    flush=True,
                )
    finally:
        del model
        if device == "mps" and hasattr(torch, "mps"):
            torch.mps.empty_cache()

    rows = [by_id[key] for key in sorted(by_id)]
    completed = [row for row in rows if row.get("status") == "completed"]
    all_latency = [
        float(value)
        for row in completed
        for value in row.get("latency_seconds") or []
    ]
    metrics = {
        "kind": "phase69_real_generation_metrics",
        "variant": args.variant,
        "session_count": len(rows),
        "completed_count": len(completed),
        "failed_count": len(rows) - len(completed),
        "actual_model_session_count": sum(row.get("actual_model_call") is True for row in rows),
        "actual_generation_call_count": sum(len(row.get("generation") or []) for row in completed),
        "truncated_session_count": sum(row.get("truncated_response") is True for row in completed),
        "think_leak_session_count": sum(row.get("think_leak_detected") is True for row in completed),
        "privacy_persistence_failure_count": sum(
            bool(dict(row.get("privacy_runtime") or {}).get("raw_private_values_persisted"))
            for row in completed
        ),
        "mean_generation_latency_seconds": round(sum(all_latency) / len(all_latency), 4)
        if all_latency
        else None,
        "model_id": str(MODEL_PATH),
        "device": device,
        "adapter_loaded": False,
        "simulated_usage": True,
        "actual_user_feedback_count": 0,
        "training_executed": False,
        "created_at": _utcnow(),
    }
    _write_json(metrics_path, metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0 if len(completed) == len(sessions) else 1


if __name__ == "__main__":
    raise SystemExit(main())
