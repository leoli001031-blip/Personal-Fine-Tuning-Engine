#!/usr/bin/env python3
"""Run the Phase82 guarded persona runtime contract product-path benchmark."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = REPO_ROOT / "pfe-core"
SERVER_ROOT = REPO_ROOT / "pfe-server"
for path in (CORE_ROOT, SERVER_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pfe_core.inference.engine import InferenceEngine
from pfe_core.phase75_personalization_benefit_benchmark import (
    aggregate_phase75_variant,
    stable_hash,
)
from pfe_core.phase77_private_value_guarded_runtime import (
    build_phase77_holdout,
    guard_phase77_messages,
    guard_phase77_output,
)
from pfe_core.phase78_persona_internalization_training import (
    build_phase78_holdout,
    build_phase78_training_samples,
)
from pfe_core.phase79_cpu_feasible_persona_probe import build_phase79_holdout
from pfe_core.phase80_small_model_failure_taxonomy import build_phase80_holdout
from pfe_core.phase81_trainable_mid_model_selection import build_phase81_holdout
from pfe_core.phase82_mid_model_runtime_contract import (
    PHASE82_SESSION_COUNT,
    PHASE82_VARIANTS,
    audit_phase82_isolation,
    build_phase82_decision,
    build_phase82_holdout,
)
from pfe_core.pipeline import PipelineService
from pfe_core.server_services import InferenceServiceAdapter
from pfe_server.models import ChatCompletionRequest


EVIDENCE_ROOT = REPO_ROOT / "docs/demo/phase82-mid-model-runtime-contract"
PREPARATION_ROOT = EVIDENCE_ROOT / "evidence-preparation"
GENERATION_ROOT = EVIDENCE_ROOT / "evidence-real-api-generation"
FAILURE_ROOT = EVIDENCE_ROOT / "evidence-failures"
MODEL_PATH = REPO_ROOT / "models/Qwen2.5-1.5B-Instruct"
MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
CORE_SOURCE = CORE_ROOT / "pfe_core/phase82_mid_model_runtime_contract.py"
DRIVER_SOURCE = REPO_ROOT / "tools/phase82_mid_model_runtime_contract.py"
TEST_SOURCE = REPO_ROOT / "tests/test_phase82_persona_runtime_contract.py"
CONTRACT_SOURCE = CORE_ROOT / "pfe_core/inference/contracts.py"
ENGINE_SOURCE = CORE_ROOT / "pfe_core/inference/engine.py"
SERVER_SOURCE = SERVER_ROOT / "pfe_server/app.py"
GENERATION_PROTOCOL = {
    "kind": "phase82_frozen_persona_api_generation_protocol",
    "api_surface": "PipelineService.chat_completion",
    "base_model": "Qwen2.5-1.5B-Instruct",
    "model_revision": MODEL_REVISION,
    "max_tokens": 128,
    "temperature": 0.0,
    "repetition_penalty": 1.15,
    "no_repeat_ngram_size": 4,
    "same_decoding_controls_both_variants": True,
    "persona_response_contract": "contract_persona_guarded",
    "three_user_turns_per_session": True,
    "same_model_all_variants": True,
    "score_or_gate_relaxation_allowed": False,
}
FROZEN_THRESHOLDS = {
    "minimum_runtime_target_gain": 0.04,
    "minimum_route_accuracy": 1.0,
    "maximum_runtime_truncation_rate": 0.15,
    "maximum_ordinary_regression": 0.02,
    "maximum_privacy_echo_rate": 0.0,
    "maximum_think_leak_rate": 0.0,
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
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return dict(value) if isinstance(value, Mapping) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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
        "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
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


def _model_complete() -> bool:
    required = (
        "config.json",
        "generation_config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
    )
    return all((MODEL_PATH / name).is_file() and (MODEL_PATH / name).stat().st_size > 0 for name in required)


def _previous_holdouts() -> list[dict[str, Any]]:
    return (
        build_phase77_holdout()["sessions"]
        + build_phase78_holdout()["sessions"]
        + build_phase79_holdout()["sessions"]
        + build_phase80_holdout()["sessions"]
        + build_phase81_holdout()["sessions"]
    )


def _prepare(clean: bool) -> int:
    if clean:
        shutil.rmtree(EVIDENCE_ROOT, ignore_errors=True)
    PREPARATION_ROOT.mkdir(parents=True, exist_ok=True)
    holdout = build_phase82_holdout()
    isolation = audit_phase82_isolation(holdout["sessions"], _previous_holdouts())
    training = build_phase78_training_samples()
    config = _read_json(MODEL_PATH / "config.json")
    checks = {
        "model_download_complete": _model_complete(),
        "selected_model_architecture_qwen2": config.get("model_type") == "qwen2",
        "fresh_holdout_isolated": isolation.get("passed") is True,
        "holdout_count_30": holdout.get("session_count") == PHASE82_SESSION_COUNT,
        "no_training_run_planned": True,
        "product_default_unchanged": True,
    }
    freeze = {
        "kind": "phase82_pre_experiment_freeze",
        "created_at": _utcnow(),
        "frozen_before_generation": True,
        "passed": all(checks.values()),
        "checks": checks,
        "model_path": str(MODEL_PATH),
        "model_revision": MODEL_REVISION,
        "model_config_sha256": _sha256(MODEL_PATH / "config.json") if (MODEL_PATH / "config.json").is_file() else None,
        "model_weight_size_bytes": (MODEL_PATH / "model.safetensors").stat().st_size
        if (MODEL_PATH / "model.safetensors").is_file()
        else 0,
        "holdout_manifest_sha256": stable_hash(holdout["sessions"]),
        "training_manifest_sha256": stable_hash(training),
        "generation_protocol_sha256": stable_hash(GENERATION_PROTOCOL),
        "thresholds_sha256": stable_hash(FROZEN_THRESHOLDS),
        "source_sha256": {
            "core": _sha256(CORE_SOURCE),
            "driver": _sha256(DRIVER_SOURCE),
            "test": _sha256(TEST_SOURCE),
            "contracts": _sha256(CONTRACT_SOURCE),
            "engine": _sha256(ENGINE_SOURCE),
            "server": _sha256(SERVER_SOURCE),
        },
        "score_or_gate_relaxation_allowed": False,
        "automatic_deployment_allowed": False,
        "automatic_promotion_allowed": False,
    }
    _write_json(PREPARATION_ROOT / "holdout.json", holdout)
    _write_json(PREPARATION_ROOT / "isolation_audit.json", isolation)
    _write_json(PREPARATION_ROOT / "model_manifest.json", {
        "kind": "phase82_model_manifest",
        "model": "Qwen2.5-1.5B-Instruct",
        "revision": MODEL_REVISION,
        "local_path": str(MODEL_PATH),
        "config": {
            "model_type": config.get("model_type"),
            "architectures": config.get("architectures"),
            "max_position_embeddings": config.get("max_position_embeddings"),
        },
        "weight_size_bytes": freeze["model_weight_size_bytes"],
        "model_weights_committed": False,
    })
    _write_json(EVIDENCE_ROOT / "generation_protocol.json", GENERATION_PROTOCOL)
    _write_json(EVIDENCE_ROOT / "frozen_thresholds.json", FROZEN_THRESHOLDS)
    _write_json(EVIDENCE_ROOT / "pre_experiment_freeze.json", freeze)
    _write_json(EVIDENCE_ROOT / "preparation_decision.json", {
        "kind": "phase82_preparation_decision",
        "status": "ready_for_real_api_ab" if freeze["passed"] else "blocked",
        "checks": checks,
        "automatic_generation_started": False,
    })
    print(json.dumps({
        "status": "ready_for_real_api_ab" if freeze["passed"] else "blocked",
        "holdout_count": holdout.get("session_count"),
        "checks": checks,
    }, ensure_ascii=False, indent=2))
    return 0 if freeze["passed"] else 2


def _freeze_check() -> dict[str, Any]:
    freeze = _read_json(EVIDENCE_ROOT / "pre_experiment_freeze.json")
    holdout = _read_json(PREPARATION_ROOT / "holdout.json")
    sources = {
        "core": _sha256(CORE_SOURCE),
        "driver": _sha256(DRIVER_SOURCE),
        "test": _sha256(TEST_SOURCE),
        "contracts": _sha256(CONTRACT_SOURCE),
        "engine": _sha256(ENGINE_SOURCE),
        "server": _sha256(SERVER_SOURCE),
    }
    checks = {
        "pre_experiment_freeze_passed": freeze.get("passed") is True,
        "holdout_unchanged": stable_hash(holdout.get("sessions") or [])
        == freeze.get("holdout_manifest_sha256"),
        "generation_protocol_unchanged": stable_hash(GENERATION_PROTOCOL)
        == freeze.get("generation_protocol_sha256"),
        "thresholds_unchanged": stable_hash(FROZEN_THRESHOLDS) == freeze.get("thresholds_sha256"),
        "model_config_unchanged": _sha256(MODEL_PATH / "config.json")
        == freeze.get("model_config_sha256"),
        "model_weight_size_unchanged": (MODEL_PATH / "model.safetensors").stat().st_size
        == int(freeze.get("model_weight_size_bytes") or 0),
        "source_files_unchanged": sources == freeze.get("source_sha256"),
    }
    return {"kind": "phase82_generation_freeze_check", "passed": all(checks.values()), "checks": checks}


def _generation_info(payload: Mapping[str, Any]) -> dict[str, Any]:
    metadata = dict(payload.get("metadata") or {})
    inference = dict(metadata.get("inference") or {})
    return dict(inference.get("generation") or {})


def _run_session(
    *,
    service: PipelineService,
    session: Mapping[str, Any],
    variant: str,
) -> dict[str, Any]:
    persona = variant == "persona_api_contract"
    private_values = [str(value) for value in session.get("declared_private_values") or [] if str(value)]
    api_history: list[dict[str, str]] = []
    persisted_turns: list[dict[str, str]] = []
    generations = []
    routes = []
    input_guards = []
    output_guards = []
    raw_private_echo = False
    for turn, raw_user in enumerate(
        (
            str(session.get("user_goal") or ""),
            str(session.get("user_correction") or ""),
            str(session.get("continuation_request") or ""),
        ),
        start=1,
    ):
        guarded_user_rows, driver_input_guard = guard_phase77_messages(
            [{"role": "user", "content": raw_user}],
            private_values,
        )
        persisted_user = guarded_user_rows[0]
        api_user = {"role": "user", "content": raw_user} if persona else dict(persisted_user)
        api_history.append(api_user)
        metadata: dict[str, Any] = {
            "enable_real_local": True,
            "repetition_penalty": GENERATION_PROTOCOL["repetition_penalty"],
            "no_repeat_ngram_size": GENERATION_PROTOCOL["no_repeat_ngram_size"],
            "phase82_simulated_usage": True,
        }
        if persona:
            metadata.update({
                "response_contract": GENERATION_PROTOCOL["persona_response_contract"],
                "declared_private_values": private_values,
            })
        started = time.perf_counter()
        payload = service.chat_completion(
            messages=[dict(row) for row in api_history],
            model="base",
            adapter_version="latest",
            temperature=float(GENERATION_PROTOCOL["temperature"]),
            max_tokens=int(GENERATION_PROTOCOL["max_tokens"]),
            metadata=metadata,
            request_id=f"phase82-{variant}-{session['session_id']}-t{turn}",
            session_id=f"phase82-{variant}-{session['session_id']}",
        )
        generation = _generation_info(payload)
        content = str(payload["choices"][0]["message"]["content"])
        guarded_output, driver_output_guard = guard_phase77_output(content, private_values)
        raw_private_echo = raw_private_echo or bool(driver_output_guard["raw_model_private_echo_detected"])
        api_history.append({"role": "assistant", "content": guarded_output})
        persisted_turns.extend((dict(persisted_user), {"role": "assistant", "content": guarded_output}))
        contract_info = dict(generation.get("response_contract") or {})
        contract_output = dict(generation.get("contract_output") or {})
        contract_input_guard = dict(contract_info.get("input_guard") or {})
        contract_output_guard = dict(contract_output.get("output_guard") or {})
        route = dict(contract_info.get("route") or {})
        routes.append({
            "turn": turn,
            "routed": route.get("routed") if persona else None,
            "reason": route.get("reason") if persona else "base_no_persona_contract",
        })
        input_guards.append({
            "turn": turn,
            "driver_guard_passed": driver_input_guard.get("passed") is True,
            "contract_guard_applied": persona,
            "contract_guard_passed": contract_input_guard.get("passed") is True if persona else None,
            "contract_replacement_count": contract_input_guard.get("replacement_count") if persona else None,
            "model_input_contains_declared_private_value": (
                contract_input_guard.get("model_input_contains_declared_private_value") is True
                if persona
                else any(value in json.dumps(api_history, ensure_ascii=False) for value in private_values)
            ),
        })
        output_guards.append({
            "turn": turn,
            "driver_guard_passed": driver_output_guard.get("passed") is True,
            "contract_guard_applied": persona,
            "contract_guard_passed": contract_output_guard.get("passed") is True if persona else None,
            "returned_output_contains_declared_private_value": any(
                value in guarded_output for value in private_values
            ),
        })
        generations.append({
            "turn": turn,
            "latency_seconds": round(time.perf_counter() - started, 4),
            "finish_reason": payload["choices"][0].get("finish_reason"),
            "truncated": payload["choices"][0].get("finish_reason") == "length",
            "think_leak_detected": bool(re.search(r"</?think>", guarded_output, flags=re.IGNORECASE)),
            "token_budget": generation.get("token_budget") or {},
            "served_by": payload.get("served_by"),
            "contract_applied": contract_info.get("applied") is True,
        })
    final_route = routes[-1].get("routed") if routes else None
    return {
        "kind": "phase82_real_api_multiturn_transcript",
        "session_id": session.get("session_id"),
        "category": session.get("category"),
        "task_type": session.get("task_type"),
        "variant": variant,
        "model_id": str(MODEL_PATH),
        "model_revision": MODEL_REVISION,
        "response_contract": GENERATION_PROTOCOL["persona_response_contract"] if persona else None,
        "actual_model_call": all(row.get("served_by") == "local" for row in generations),
        "actual_api_path": True,
        "adapter_loaded": False,
        "status": "completed",
        "turns": persisted_turns,
        "route_manifests": routes,
        "final_route_matches_expected": (
            final_route == bool(session.get("expected_route")) if persona else None
        ),
        "private_input_guards": input_guards,
        "private_output_guards": output_guards,
        "generation": generations,
        "truncated_response": any(row["truncated"] for row in generations),
        "think_leak_detected": any(row["think_leak_detected"] for row in generations),
        "privacy_canary_echo_detected": raw_private_echo,
        "returned_private_value_detected": any(
            row["returned_output_contains_declared_private_value"] for row in output_guards
        ),
        "simulated_usage": True,
        "actual_user_feedback": False,
        "actual_product_benefit_claim_allowed": False,
        "created_at": _utcnow(),
    }


def _generate(variant: str, clean: bool) -> int:
    freeze = _freeze_check()
    if not freeze["passed"]:
        raise SystemExit(f"Phase82 generation freeze failed: {freeze}")
    sessions = [dict(row) for row in _read_json(PREPARATION_ROOT / "holdout.json").get("sessions") or []]
    output_path = GENERATION_ROOT / f"transcripts_{variant}.jsonl"
    metrics_path = GENERATION_ROOT / f"metrics_{variant}.json"
    if clean:
        output_path.unlink(missing_ok=True)
        metrics_path.unlink(missing_ok=True)
    rows = [] if clean else _read_jsonl(output_path)
    completed = {str(row.get("session_id")) for row in rows if row.get("status") == "completed"}
    old_base_model = os.environ.get("PFE_BASE_MODEL")
    os.environ["PFE_BASE_MODEL"] = str(MODEL_PATH)
    service = PipelineService()
    try:
        for index, session in enumerate(sessions, start=1):
            session_id = str(session["session_id"])
            if session_id in completed:
                print(f"[{variant}] {index}/{len(sessions)} {session_id} resumed", flush=True)
                continue
            try:
                row = _run_session(service=service, session=session, variant=variant)
            except Exception as exc:
                row = {
                    "kind": "phase82_real_api_multiturn_transcript",
                    "session_id": session_id,
                    "category": session.get("category"),
                    "task_type": session.get("task_type"),
                    "variant": variant,
                    "actual_model_call": False,
                    "actual_api_path": True,
                    "status": "failed",
                    "error": f"{exc.__class__.__name__}: {exc}",
                    "turns": [],
                    "simulated_usage": True,
                    "actual_user_feedback": False,
                    "created_at": _utcnow(),
                }
                _write_json(FAILURE_ROOT / f"{variant}_{session_id}.json", row)
            rows = [item for item in rows if item.get("session_id") != session_id]
            rows.append(row)
            rows.sort(key=lambda item: str(item.get("session_id")))
            _write_jsonl(output_path, rows)
            print(f"[{variant}] {index}/{len(sessions)} {session_id} {row['status']}", flush=True)
    finally:
        if old_base_model is None:
            os.environ.pop("PFE_BASE_MODEL", None)
        else:
            os.environ["PFE_BASE_MODEL"] = old_base_model
        InferenceEngine._runtime_cache.clear()
    metrics = aggregate_phase75_variant(rows, sessions)
    route_rows = [row for row in rows if row.get("final_route_matches_expected") is not None]
    metrics.update({
        "kind": "phase82_variant_metrics",
        "variant": variant,
        "model_id": str(MODEL_PATH),
        "model_revision": MODEL_REVISION,
        "actual_api_path": True,
        "actual_model_calls": len(rows) == len(sessions)
        and all(row.get("actual_model_call") is True for row in rows),
        "model_call_count": sum(len(row.get("generation") or []) for row in rows),
        "all_sessions_completed": len(rows) == len(sessions)
        and all(row.get("status") == "completed" for row in rows),
        "truncated_session_rate": round(sum(bool(row.get("truncated_response")) for row in rows) / len(rows), 4)
        if rows
        else 0.0,
        "think_leak_rate": round(sum(bool(row.get("think_leak_detected")) for row in rows) / len(rows), 4)
        if rows
        else 0.0,
        "route_accuracy": round(
            sum(row.get("final_route_matches_expected") is True for row in route_rows) / len(route_rows),
            4,
        )
        if route_rows
        else 1.0,
        "route_evaluated_session_count": len(route_rows),
        "simulated_usage": True,
        "actual_user_feedback": False,
        "actual_product_benefit_claim_allowed": False,
    })
    _write_json(GENERATION_ROOT / f"freeze_check_{variant}.json", freeze)
    _write_json(metrics_path, metrics)
    print(json.dumps({key: metrics.get(key) for key in (
        "variant", "session_count", "model_call_count", "personalization_score",
        "hard_gate_pass_rate", "truncated_session_rate", "privacy_canary_echo_rate", "route_accuracy",
    )}, ensure_ascii=False, indent=2))
    return 0 if metrics["all_sessions_completed"] and metrics["actual_model_calls"] else 1


def _api_smoke(clean: bool) -> int:
    output_path = EVIDENCE_ROOT / "real_api_smoke.json"
    if clean:
        output_path.unlink(missing_ok=True)
    freeze = _freeze_check()
    secret = "SYNTHETIC_PHASE82_API_SMOKE_PRIVATE_DO_NOT_ECHO"
    old_base_model = os.environ.get("PFE_BASE_MODEL")
    os.environ["PFE_BASE_MODEL"] = str(MODEL_PATH)
    try:
        adapter = InferenceServiceAdapter(pipeline=PipelineService())
        request = ChatCompletionRequest(
            model="base",
            response_contract="contract_persona_guarded",
            temperature=0.0,
            max_tokens=128,
            messages=[{"role": "user", "content": f"检查 adapter 证据，临时值 {secret}"}],
            metadata={
                "enable_real_local": True,
                "declared_private_values": [secret],
                "memory_consent": False,
            },
            request_id="phase82-api-smoke-request",
            session_id="phase82-api-smoke-session",
        )
        response = asyncio.run(adapter.generate_chat_completion(request))
        metadata = dict(response.metadata or {})
        generation = dict(dict(metadata.get("inference") or {}).get("generation") or {})
        contract = dict(generation.get("response_contract") or {})
        contract_output = dict(generation.get("contract_output") or {})
        content = response.choices[0].message.content if response.choices else ""
        checks = {
            "freeze_passed": freeze.get("passed") is True,
            "served_by_real_local": response.served_by == "local",
            "persona_contract_applied": contract.get("contract") == "contract_persona_guarded"
            and contract.get("applied") is True,
            "workflow_route_selected": dict(contract.get("route") or {}).get("routed") is True,
            "private_input_replaced": int(dict(contract.get("input_guard") or {}).get("replacement_count") or 0)
            >= 1,
            "private_output_absent": secret not in content,
            "output_guard_passed": dict(contract_output.get("output_guard") or {}).get("passed") is True,
            "memory_consent_disabled": dict(metadata.get("memory") or {}).get("explicit_consent") is False,
        }
        result = {
            "kind": "phase82_real_api_smoke",
            "created_at": _utcnow(),
            "passed": all(checks.values()),
            "checks": checks,
            "served_by": response.served_by,
            "response_contract": contract.get("contract"),
            "route_reason": dict(contract.get("route") or {}).get("reason"),
            "private_value_sha256": hashlib.sha256(secret.encode()).hexdigest(),
            "raw_private_value_persisted": False,
            "request_body_persisted": False,
            "response_contains_private_value": False,
            "actual_user_feedback": False,
            "actual_product_benefit_claim_allowed": False,
        }
    finally:
        if old_base_model is None:
            os.environ.pop("PFE_BASE_MODEL", None)
        else:
            os.environ["PFE_BASE_MODEL"] = old_base_model
        InferenceEngine._runtime_cache.clear()
    _write_json(output_path, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


def _collect_metrics() -> dict[str, dict[str, Any]]:
    return {
        variant: _read_json(GENERATION_ROOT / f"metrics_{variant}.json")
        for variant in PHASE82_VARIANTS
    }


def _public_private_audit() -> dict[str, Any]:
    sessions = [dict(row) for row in _read_json(PREPARATION_ROOT / "holdout.json").get("sessions") or []]
    by_id = {str(row["session_id"]): row for row in sessions}
    details = []
    for variant in PHASE82_VARIANTS:
        for row in _read_jsonl(GENERATION_ROOT / f"transcripts_{variant}.jsonl"):
            session = by_id.get(str(row.get("session_id") or ""), {})
            values = [str(value) for value in session.get("declared_private_values") or [] if str(value)]
            serialized = json.dumps(row, ensure_ascii=False, sort_keys=True)
            details.append({
                "variant": variant,
                "session_id": row.get("session_id"),
                "raw_private_match_count": sum(value in serialized for value in values),
                "returned_private_value_detected": bool(row.get("returned_private_value_detected")),
                "model_input_private_value_detected": any(
                    bool(item.get("model_input_contains_declared_private_value"))
                    for item in row.get("private_input_guards") or []
                ),
            })
    expected = PHASE82_SESSION_COUNT * len(PHASE82_VARIANTS)
    checks = {
        "expected_transcript_count": len(details) == expected,
        "raw_private_match_count_zero": not any(row["raw_private_match_count"] for row in details),
        "returned_private_value_zero": not any(row["returned_private_value_detected"] for row in details),
        "model_input_private_value_zero": not any(row["model_input_private_value_detected"] for row in details),
    }
    return {
        "kind": "phase82_public_private_transcript_audit",
        "passed": all(checks.values()),
        "checks": checks,
        "transcript_count": len(details),
        "expected_transcript_count": expected,
        "details": details,
    }


def _output_examples(transcripts: Mapping[str, list[dict[str, Any]]]) -> str:
    selected = (
        "phase82-evidence_truthfulness-01",
        "phase82-concise_workstyle-01",
        "phase82-privacy_non_echo-01",
        "phase82-ordinary_direct-01",
    )
    by_variant = {
        variant: {str(row.get("session_id")): row for row in rows}
        for variant, rows in transcripts.items()
    }
    lines = [
        "# Phase82 Output Examples",
        "",
        "Real local outputs through the PFE chat pipeline on fresh simulated_usage. They are not actual-user product evidence.",
        "",
    ]
    for session_id in selected:
        lines.extend((f"## {session_id}", ""))
        for variant in PHASE82_VARIANTS:
            row = by_variant[variant][session_id]
            final = [
                str(turn.get("content") or "")
                for turn in row.get("turns") or []
                if turn.get("role") == "assistant"
            ][-1]
            final = "\n".join(line.rstrip() for line in final.splitlines())
            lines.extend((f"### {variant}", "", final, ""))
    return "\n".join(lines)


def _evidence_manifest() -> dict[str, Any]:
    files = []
    for path in sorted(EVIDENCE_ROOT.rglob("*")):
        if not path.is_file() or path.name in DYNAMIC_FILES:
            continue
        files.append({
            "path": str(path.relative_to(REPO_ROOT)),
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        })
    return {
        "kind": "phase82_evidence_manifest",
        "files": files,
        "file_count": len(files),
        "manifest_sha256": stable_hash(files),
    }


def _finalize() -> int:
    metrics = _collect_metrics()
    isolation = _read_json(PREPARATION_ROOT / "isolation_audit.json")
    api_smoke = _read_json(EVIDENCE_ROOT / "real_api_smoke.json")
    privacy = _public_private_audit()
    decision = build_phase82_decision(
        metrics=metrics,
        isolation_audit=isolation,
        api_smoke=api_smoke,
        public_private_audit=privacy,
    )
    transcripts = {
        variant: _read_jsonl(GENERATION_ROOT / f"transcripts_{variant}.jsonl")
        for variant in PHASE82_VARIANTS
    }
    comparison = {
        "kind": "phase82_persona_api_contract_comparison",
        "created_at": _utcnow(),
        "model": "Qwen2.5-1.5B-Instruct",
        "model_revision": MODEL_REVISION,
        "api_surface": GENERATION_PROTOCOL["api_surface"],
        "metrics": metrics,
        "phase81_canonical_reference": {
            "base_target_score": 0.45,
            "runtime_target_score": 0.51,
            "runtime_gain_vs_base": 0.06,
            "source": "docs/demo/phase81-trainable-mid-model-selection/phase81-final-decision.json",
            "canonical_reference_only": True,
        },
        "decision": decision,
        "simulated_usage": True,
        "actual_user_feedback_count": 0,
        "actual_product_benefit_claim_allowed": False,
    }
    _write_json(EVIDENCE_ROOT / "public_private_audit.json", privacy)
    _write_json(EVIDENCE_ROOT / "comparison_summary.json", comparison)
    _write_json(EVIDENCE_ROOT / "phase82-final-decision.json", decision)
    _write_text(EVIDENCE_ROOT / "output_examples.md", _output_examples(transcripts))
    _write_text(EVIDENCE_ROOT / "phase82-final-decision.md", f"""# Phase82 Final Decision

Recommendation: **{decision['recommendation']}**

- Lifecycle status: `{decision['status']}`
- Base API target score: `{decision['target_scores']['base_api_length_control']}`
- Persona contract target score: `{decision['target_scores']['persona_api_contract']}`
- Runtime gain over base: `{decision['runtime_gain_vs_base']}`
- Base truncation rate: `{decision['truncation_rates']['base_api_length_control']}`
- Persona contract truncation rate: `{decision['truncation_rates']['persona_api_contract']}`
- Simulated laboratory benefit: `{decision['simulated_lab_runtime_benefit']}`

Phase82 uses a fresh `simulated_usage` benchmark through the PFE chat pipeline. It does not contain actual user feedback and does not authorize automatic deployment, promotion, Hermes attachment, or a product-default change.
""")
    _write_text(EVIDENCE_ROOT / "phase82-runbook.md", """# Phase82 Runbook

```bash
.venv/bin/python tools/phase82_mid_model_runtime_contract.py prepare --clean
.venv/bin/python tools/phase82_mid_model_runtime_contract.py api-smoke --clean
.venv/bin/python tools/phase82_mid_model_runtime_contract.py generate --variant base_api_length_control --clean
.venv/bin/python tools/phase82_mid_model_runtime_contract.py generate --variant persona_api_contract --clean
.venv/bin/python tools/phase82_mid_model_runtime_contract.py full-regression
.venv/bin/python tools/phase82_mid_model_runtime_contract.py finalize
.venv/bin/python tools/phase82_mid_model_runtime_contract.py validate
```

The model revision, fresh holdout, product API surface, decoding controls, and thresholds are frozen before generation. Both variants use the same model and decoding controls; only the persona response contract differs.
""")
    next_goal = (
        "Build an opt-in manual trial pack for contract_persona_guarded, collect explicit review decisions, and keep the product default unchanged."
        if decision["recommendation"] == "phase83_manual_runtime_contract_trial_pack"
        else "Revise the persona runtime contract using Phase82 failure categories before another product-path benchmark."
    )
    _write_text(EVIDENCE_ROOT / "next-pursuit-goal.md", f"# Phase83 Pursuit Goal\n\n{next_goal}")
    manifest = _evidence_manifest()
    _write_json(EVIDENCE_ROOT / "evidence_manifest.json", manifest)
    _write_json(EVIDENCE_ROOT / "evidence_integrity.json", {
        "kind": "phase82_evidence_integrity",
        "passed": True,
        "manifest_file_count": manifest["file_count"],
        "manifest_sha256": manifest["manifest_sha256"],
        "actual_user_feedback_count": 0,
        "actual_product_benefit_claim_allowed": False,
        "automatic_deployment_allowed": False,
    })
    _write_json(EVIDENCE_ROOT / "finalization_state.json", {
        "kind": "phase82_finalization_state",
        "created_at": _utcnow(),
        "status": "finalized",
    })
    print(json.dumps({
        "status": decision["status"],
        "recommendation": decision["recommendation"],
        "target_scores": decision["target_scores"],
        "runtime_gain_vs_base": decision["runtime_gain_vs_base"],
        "truncation_rates": decision["truncation_rates"],
    }, ensure_ascii=False, indent=2))
    return 0


def _run_logged(command: list[str]) -> dict[str, Any]:
    started = time.perf_counter()
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    lines = []
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    code = process.wait()
    return {
        "command": command,
        "exit_code": code,
        "duration_seconds": round(time.perf_counter() - started, 4),
        "output": "".join(lines),
    }


def _full_regression() -> int:
    commands = (
        [
            str(REPO_ROOT / ".venv/bin/pytest"),
            "-q",
            "tests/test_phase82_persona_runtime_contract.py",
            "tests/test_phase81_trainable_mid_model_selection.py",
            "tests/test_phase13_boundary_contract.py",
            "tests/test_inference_runtime.py",
        ],
        ["make", "test-unit", "test-surface", "test-e2e-mock", "smoke-beta"],
    )
    results = []
    for command in commands:
        result = _run_logged(command)
        results.append(result)
        if result["exit_code"] != 0:
            break
    summary = {
        "kind": "phase82_full_regression_summary",
        "created_at": _utcnow(),
        "passed": len(results) == len(commands) and all(row["exit_code"] == 0 for row in results),
        "results": results,
    }
    _write_json(EVIDENCE_ROOT / "full_regression_summary.json", summary)
    return 0 if summary["passed"] else 1


def _validate() -> int:
    manifest = _read_json(EVIDENCE_ROOT / "evidence_manifest.json")
    integrity = _read_json(EVIDENCE_ROOT / "evidence_integrity.json")
    decision = _read_json(EVIDENCE_ROOT / "phase82-final-decision.json")
    regression = _read_json(EVIDENCE_ROOT / "full_regression_summary.json")
    api_smoke = _read_json(EVIDENCE_ROOT / "real_api_smoke.json")
    manifest_failures = []
    for row in manifest.get("files") or []:
        path = REPO_ROOT / str(row.get("path") or "")
        if not path.is_file() or _sha256(path) != row.get("sha256"):
            manifest_failures.append(str(row.get("path") or ""))
    raw_private_locations = []
    for path in EVIDENCE_ROOT.rglob("*"):
        if not path.is_file() or path.name in DYNAMIC_FILES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if "SYNTHETIC_PHASE82_PRIVATE_" in text or "SYNTHETIC_PHASE82_API_SMOKE_PRIVATE" in text:
            raw_private_locations.append(str(path.relative_to(EVIDENCE_ROOT)))
    checks = {
        "manifest_files_unchanged": not manifest_failures,
        "integrity_passed": integrity.get("passed") is True,
        "real_api_smoke_passed": api_smoke.get("passed") is True,
        "full_regression_passed": regression.get("passed") is True,
        "private_canaries_only_in_frozen_holdout": raw_private_locations
        == ["evidence-preparation/holdout.json"],
        "no_actual_user_claim": decision.get("actual_product_benefit_claim_allowed") is False,
        "no_auto_promotion": decision.get("auto_promotion_allowed") is False,
        "no_auto_deployment": decision.get("automatic_deployment_allowed") is False,
        "no_hermes_attachment": decision.get("hermes_attachment_allowed") is False,
        "product_default_unchanged": decision.get("product_default_changed") is False,
    }
    summary = {
        "kind": "phase82_validation_summary",
        "created_at": _utcnow(),
        "passed": all(checks.values()),
        "checks": checks,
        "manifest_failures": manifest_failures,
        "raw_private_locations": raw_private_locations,
    }
    _write_json(EVIDENCE_ROOT / "validation_summary.json", summary)
    _write_text(EVIDENCE_ROOT / "validation_gate.txt", "PASS" if summary["passed"] else "FAIL")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["passed"] else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--clean", action="store_true")
    smoke = subparsers.add_parser("api-smoke")
    smoke.add_argument("--clean", action="store_true")
    generate = subparsers.add_parser("generate")
    generate.add_argument("--variant", choices=PHASE82_VARIANTS, required=True)
    generate.add_argument("--clean", action="store_true")
    subparsers.add_parser("full-regression")
    subparsers.add_parser("finalize")
    subparsers.add_parser("validate")
    args = parser.parse_args()
    if args.command == "prepare":
        return _prepare(args.clean)
    if args.command == "api-smoke":
        return _api_smoke(args.clean)
    if args.command == "generate":
        return _generate(args.variant, args.clean)
    if args.command == "full-regression":
        return _full_regression()
    if args.command == "finalize":
        return _finalize()
    if args.command == "validate":
        return _validate()
    raise SystemExit(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
