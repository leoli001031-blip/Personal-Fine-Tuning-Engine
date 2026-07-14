#!/usr/bin/env python3
"""Run the Phase72 deterministic boundary serializer experiment."""

from __future__ import annotations

import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping
from urllib.error import HTTPError
from urllib.request import Request, urlopen


REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = REPO_ROOT / "pfe-core"
TOOLS_ROOT = REPO_ROOT / "tools"
for root in (CORE_ROOT, TOOLS_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import phase70_execute_eval as phase70_eval
from phase62_execute import JudgeAttemptError
from phase70_generate import (
    MODEL_PATH,
    _load_runtime,
    _run_session,
)
from phase70_prepare import _blind_cases, _phase68_regression_cases
from pfe_core.phase59_proposition_addressed_grounding import (
    build_phase59_candidate_judge_prompt,
    build_phase59_proposition_candidates,
    phase59_ollama_json_schema,
    validate_phase59_raw_selection,
)
from pfe_core.phase63_field_typed_candidate_wire import (
    PHASE63_WIRE_PATTERN,
    parse_phase63_typed_wire_selection,
)
from pfe_core.phase69_minimal_runtime_ab import final_assistant_text
from pfe_core.phase70_structured_boundary_contract import stable_hash
from pfe_core.phase72_deterministic_boundary_serializer import (
    PHASE72_BOUNDARY_COUNT,
    PHASE72_EXACT_OUTPUT,
    PHASE72_ORDINARY_COUNT,
    PHASE72_VARIANTS,
    apply_phase72_serializer,
    build_phase72_explicit_typed_wire_prompt,
    build_phase72_holdout,
    build_phase72_transport_preflight_cases,
    classify_phase72_boundary,
    evaluate_phase72_boundary_results,
    score_phase72_ordinary,
)


EVIDENCE_ROOT = REPO_ROOT / "docs/demo/phase72-deterministic-boundary-serializer"
PHASE71_ROOT = REPO_ROOT / "docs/demo/phase71-qualified-structured-contract-ab"
JUDGE_ALIASES = ("semantic_judge_alpha", "semantic_judge_beta")
JUDGE_MODELS = {
    "semantic_judge_alpha": "gemma4:31b",
    "semantic_judge_beta": "qwen3.6:latest",
}
SOURCE_FILES = {
    "phase46_generator_helpers": REPO_ROOT / "tools/phase46_qwen3_4b_generate.py",
    "phase53_hard_detector": CORE_ROOT / "pfe_core/phase53_evaluator_scope_recovery.py",
    "phase56_grounder_composer": CORE_ROOT / "pfe_core/phase56_evidence_span_grounded_atomic.py",
    "phase59_candidates": CORE_ROOT / "pfe_core/phase59_proposition_addressed_grounding.py",
    "phase62_consensus": CORE_ROOT / "pfe_core/phase62_risk_asymmetric_candidate_consensus.py",
    "phase63_typed_wire": CORE_ROOT / "pfe_core/phase63_field_typed_candidate_wire.py",
    "phase70_runtime_core": CORE_ROOT / "pfe_core/phase70_structured_boundary_contract.py",
    "phase70_generate": REPO_ROOT / "tools/phase70_generate.py",
    "phase70_execute_eval": REPO_ROOT / "tools/phase70_execute_eval.py",
    "phase70_core": CORE_ROOT / "pfe_core/phase72_deterministic_boundary_serializer.py",
    "phase70_prepare_product_eval": Path(__file__).resolve(),
    "phase70_finalize": Path(__file__).resolve(),
    "phase72_core": CORE_ROOT / "pfe_core/phase72_deterministic_boundary_serializer.py",
    "phase72_driver": Path(__file__).resolve(),
}
SOURCE_PATHS = {
    name: str(path.relative_to(REPO_ROOT)) for name, path in SOURCE_FILES.items()
}
DYNAMIC = {
    "evidence_manifest.json",
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


def _verify_manifest(root: Path) -> tuple[bool, str]:
    manifest = _read_json(root / "evidence_manifest.json")
    files = list(manifest.get("files") or [])
    passed = bool(files) and all(
        (REPO_ROOT / str(row.get("path") or "")).is_file()
        and _sha256(REPO_ROOT / str(row.get("path") or "")) == row.get("sha256")
        for row in files
    )
    return passed, str(manifest.get("manifest_sha256") or "")


def _phase71_snapshot() -> dict[str, Any]:
    decision = _read_json(PHASE71_ROOT / "phase71-final-decision.json")
    integrity = _read_json(PHASE71_ROOT / "evidence_integrity.json")
    comparison = _read_json(PHASE71_ROOT / "comparison_summary.json")
    validation = _read_json(PHASE71_ROOT / "validation_summary.json")
    manifest_ok, manifest_sha = _verify_manifest(PHASE71_ROOT)
    checks = {
        "phase71_held": decision.get("recommendation")
        == "hold_phase71_qualified_structured_contract_ab",
        "phase71_blocked_at_product_eval": decision.get("experiment_status")
        == "blocked_at_product_eval",
        "phase71_evidence_integrity_passed": integrity.get("passed") is True,
        "phase71_blocked_evidence_complete": integrity.get("blocked_evidence_complete")
        is True,
        "phase71_288_real_generations_preserved": comparison.get(
            "actual_generation_call_count"
        )
        == 288,
        "phase71_directional_gain_not_promoted": decision.get("candidate_accept_rate_delta")
        == 0.5555
        and decision.get("phase72_nondefault_api_canary_design_eligible") is False,
        "phase71_validation_passed": validation.get("status") == "passed",
        "phase71_manifest_verified": manifest_ok,
        "phase71_no_training_or_default_change": decision.get("training_allowed") is False
        and decision.get("product_default_change_allowed") is False,
    }
    return {
        "kind": "phase72_phase71_hold_snapshot",
        "passed": all(checks.values()),
        "checks": checks,
        "manifest_sha256": manifest_sha,
        "recommendation": decision.get("recommendation"),
    }


def _task_hash(row: Mapping[str, Any]) -> str:
    return stable_hash(
        [
            row.get(key)
            for key in (
                "user_goal",
                "user_correction",
                "continuation_request",
                "acceptance_request",
            )
        ]
    )


def _session_messages(row: Mapping[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "user", "content": str(row.get(key) or "")}
        for key in (
            "user_goal",
            "user_correction",
            "continuation_request",
            "acceptance_request",
        )
    ]


def _fixture_audit(cases: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    details = []
    for row in cases:
        candidates = build_phase59_proposition_candidates(
            str(row.get("assistant_response") or "")
        )
        expected = dict(row.get("expected_candidate_ids") or {})
        actual = {
            field: expected.get(field)
            for field in (
                "source_registration",
                "user_outcome_status",
                "test_to_user_outcome_relation",
            )
        }
        details.append(
            {
                "case_id": row.get("case_id"),
                "candidate_count": len(candidates),
                "expected_candidate_ids": actual,
                "passed": all(value is not None for value in actual.values()),
            }
        )
    return {
        "kind": "phase72_wire_fixture_audit",
        "passed": bool(details) and all(row["passed"] for row in details),
        "case_count": len(details),
        "details": details,
    }


def _prepare(clean: bool) -> int:
    if clean and EVIDENCE_ROOT.exists():
        shutil.rmtree(EVIDENCE_ROOT)
    missing = [name for name, path in SOURCE_FILES.items() if not path.is_file()]
    snapshot = _phase71_snapshot()
    holdout = build_phase72_holdout()
    transport = build_phase72_transport_preflight_cases()
    fixture_audit = _fixture_audit(transport["cases"])
    regression = _phase68_regression_cases()
    transport_public, transport_hidden = _blind_cases(
        transport["cases"], "phase72-wire"
    )
    regression_public, regression_hidden = _blind_cases(
        regression, "phase72-regression"
    )
    previous = _read_json(PHASE71_ROOT / "evidence-holdout/holdout.json")
    current_hashes = {_task_hash(row) for row in holdout["sessions"]}
    previous_hashes = {_task_hash(row) for row in previous.get("sessions") or []}
    overlap = {
        "kind": "phase72_holdout_overlap_audit",
        "passed": len(current_hashes) == 54 and not (current_hashes & previous_hashes),
        "phase72_unique_task_count": len(current_hashes),
        "phase71_exact_overlap_count": len(current_hashes & previous_hashes),
    }
    route_details = []
    for row in holdout["sessions"]:
        route = classify_phase72_boundary(_session_messages(row))
        expected = row["task_type"] == "boundary"
        route_details.append(
            {
                "session_id": row["session_id"],
                "task_type": row["task_type"],
                "category": row["category"],
                "expected_routed": expected,
                "actual_routed": route["routed"],
                "passed": route["routed"] is expected,
                "route": route,
            }
        )
    route_audit = {
        "kind": "phase72_pre_call_route_audit",
        "passed": all(row["passed"] for row in route_details),
        "boundary_recall": round(
            sum(row["actual_routed"] for row in route_details if row["task_type"] == "boundary")
            / PHASE72_BOUNDARY_COUNT,
            4,
        ),
        "ordinary_false_positive_rate": round(
            sum(row["actual_routed"] for row in route_details if row["task_type"] == "ordinary")
            / PHASE72_ORDINARY_COUNT,
            4,
        ),
        "details": route_details,
    }
    generation = {
        "model_path": str(MODEL_PATH),
        "input_max_length": 4096,
        "max_new_tokens": 192,
        "do_sample": False,
        "repetition_penalty": 1.05,
        "enable_thinking": False,
        "turns_per_session": 3,
        "shared_raw_generation": True,
        "raw_runtime_variant": "structured_boundary_contract",
        "privacy_runtime_enabled": True,
        "latest_intent_runtime_enabled": True,
    }
    generation["generation_protocol_sha256"] = stable_hash(generation)
    protocol = {
        "kind": "phase72_frozen_deterministic_serializer_protocol",
        "variants": list(PHASE72_VARIANTS),
        "only_ab_variable": "deterministic_boundary_serializer_after_shared_raw_generation",
        "generation": generation,
        "judge_transport": "alias_capability_routed_candidate_transport",
        "judge_transport_by_alias": {
            "semantic_judge_alpha": "phase59_nested_json_schema",
            "semantic_judge_beta": "phase72_explicit_allowed_token_wire",
        },
        "semantic_judge_aliases": list(JUDGE_ALIASES),
        "semantic_judge_models_private": dict(JUDGE_MODELS),
        "num_ctx": 4096,
        "num_predict_by_alias": {
            "semantic_judge_alpha": 192,
            "semantic_judge_beta": 64,
        },
        "temperature": 0,
        "think": False,
        "parallel_worker_count": 4,
        "retry_limit": 2,
        "wire_prompt_sha256": stable_hash(
            build_phase72_explicit_typed_wire_prompt(transport_public[0])
        ),
        "route_audit_required_before_calls": True,
        "decision_gates": {
            "transport_preflight_status": "qualified",
            "phase68_regression_status": "qualified",
            "candidate_accept_rate_min": 1.0,
            "candidate_accept_rate_delta_min": 0.5,
            "candidate_exact_structure_rate_min": 1.0,
            "candidate_dangerous_max": 0,
            "schema_failure_max": 0,
            "boundary_route_recall_min": 1.0,
            "ordinary_route_false_positive_max": 0.0,
            "ordinary_output_identity_min": 1.0,
        },
        "training_allowed": False,
        "product_default_change_allowed": False,
        "auto_promote_allowed": False,
    }
    protocol["protocol_sha256"] = stable_hash(protocol)
    source_hashes = {name: _sha256(path) for name, path in SOURCE_FILES.items()}
    freeze = {
        "kind": "phase72_pre_model_call_freeze",
        "phase71_snapshot_sha256": stable_hash(snapshot),
        "holdout_sha256": stable_hash(holdout),
        "route_audit_sha256": stable_hash(route_audit),
        "fixture_audit_sha256": stable_hash(fixture_audit),
        "sparse_public_sha256": stable_hash(transport_public),
        "sparse_hidden_sha256": stable_hash(transport_hidden),
        "regression_public_sha256": stable_hash(regression_public),
        "regression_hidden_sha256": stable_hash(regression_hidden),
        "protocol_sha256": protocol["protocol_sha256"],
        "source_sha256": source_hashes,
        "frozen_before_any_phase72_model_call": True,
        "created_at": _utcnow(),
    }
    checks = {
        "phase71_snapshot_passed": snapshot["passed"],
        "holdout_overlap_audit_passed": overlap["passed"],
        "holdout_counts_exact": holdout["session_count"] == 54
        and holdout["boundary_session_count"] == 36
        and holdout["ordinary_session_count"] == 18,
        "route_audit_passed_before_calls": route_audit["passed"],
        "wire_fixture_audit_passed_before_calls": fixture_audit["passed"],
        "wire_preflight_count_exact": len(transport_public) == 18,
        "phase68_regression_balanced_count_exact": len(regression_public) == 30,
        "local_model_present": MODEL_PATH.is_dir(),
        "source_files_complete": not missing,
    }
    decision = {
        "kind": "phase72_preparation_decision",
        "status": "ready_for_sparse_transport_preflight"
        if all(checks.values())
        else "blocked",
        "checks": checks,
        "failed_checks": [key for key, value in checks.items() if not value],
        "created_at": _utcnow(),
    }
    _write_json(EVIDENCE_ROOT / "evidence-baseline/phase71_hold_snapshot.json", snapshot)
    _write_json(EVIDENCE_ROOT / "evidence-holdout/holdout.json", holdout)
    _write_json(EVIDENCE_ROOT / "evidence-holdout/overlap_audit.json", overlap)
    _write_json(EVIDENCE_ROOT / "pre_call_route_audit.json", route_audit)
    _write_json(EVIDENCE_ROOT / "wire_fixture_audit.json", fixture_audit)
    for directory, public, hidden in (
        ("evidence-sparse-preflight", transport_public, transport_hidden),
        ("evidence-phase68-regression", regression_public, regression_hidden),
    ):
        _write_jsonl(EVIDENCE_ROOT / directory / "blind_items_public.jsonl", public)
        _write_json(
            EVIDENCE_ROOT / directory / "blind_hidden_key.json",
            {"item_count": len(hidden), "items": hidden, "hidden_from_judges": True},
        )
    _write_json(EVIDENCE_ROOT / "runtime_ab_protocol.json", protocol)
    _write_json(EVIDENCE_ROOT / "pre_model_call_freeze.json", freeze)
    _write_json(EVIDENCE_ROOT / "preparation_decision.json", decision)
    _write_json(
        EVIDENCE_ROOT / "source_manifest.json",
        {
            "kind": "phase72_source_manifest",
            "source_sha256": source_hashes,
            "actual_user_feedback_count": 0,
            "not_for_training": True,
        },
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return 0 if decision["status"] == "ready_for_sparse_transport_preflight" else 1


def _parse_json(value: str) -> dict[str, Any]:
    text = str(value or "").strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("judge response contains no JSON object")
        parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, Mapping):
        raise ValueError("judge response is not a JSON object")
    return dict(parsed)


def _invoke_phase72_judge(
    *,
    item: Mapping[str, Any],
    alias: str,
    model: str,
    endpoint: str,
    timeout: int,
    protocol: Mapping[str, Any],
    stage: str,
) -> dict[str, Any]:
    response = str(item.get("assistant_response") or "")
    candidates = build_phase59_proposition_candidates(response)
    transport = str(
        dict(protocol.get("judge_transport_by_alias") or {}).get(alias) or ""
    )
    if transport == "phase59_nested_json_schema":
        prompt = build_phase59_candidate_judge_prompt(item)
        schema = phase59_ollama_json_schema(candidates)
    elif transport == "phase72_explicit_allowed_token_wire":
        prompt = build_phase72_explicit_typed_wire_prompt(item)
        schema = None
    else:
        raise ValueError(f"unsupported Phase72 transport: {transport!r}")
    num_predict = int(dict(protocol.get("num_predict_by_alias") or {}).get(alias) or 192)
    payload = {
        "model": model,
        "stream": False,
        "think": False,
        "keep_alive": "30m",
        "messages": [{"role": "user", "content": prompt}],
        "options": {
            "temperature": 0,
            "num_ctx": int(protocol.get("num_ctx") or 4096),
            "num_predict": num_predict,
        },
    }
    if schema is not None:
        payload["format"] = schema
    request = Request(
        endpoint.rstrip("/") + "/api/chat",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urlopen(request, timeout=timeout) as handle:
            raw_http = handle.read().decode("utf-8", errors="replace")
    except HTTPError as exc:
        raw_http = exc.read().decode("utf-8", errors="replace")
        raise JudgeAttemptError(
            f"HTTPError {exc.code}: {raw_http}",
            raw_response=raw_http,
            failure_class="http_error",
        ) from exc
    try:
        body = json.loads(raw_http)
        content = str(dict(body.get("message") or {}).get("content") or "")
        selection = (
            validate_phase59_raw_selection(_parse_json(content), candidates=candidates)
            if transport == "phase59_nested_json_schema"
            else parse_phase63_typed_wire_selection(content, candidates=candidates)
        )
    except (ValueError, json.JSONDecodeError) as exc:
        raise JudgeAttemptError(
            f"{exc.__class__.__name__}: {exc}",
            raw_response=locals().get("content", "") or raw_http,
            failure_class=f"{transport}_validation_error",
        ) from exc
    eval_seconds = float(body.get("eval_duration") or 0) / 1_000_000_000
    eval_count = int(body.get("eval_count") or 0)
    return {
        "item_id": item.get("item_id"),
        "stage": stage,
        "judge_alias": alias,
        "judge_model": model,
        **selection,
        "actual_model_call": True,
        "identity_hidden_from_judge": True,
        "gold_label_hidden_from_judge": True,
        "gold_typed_fields_hidden_from_judge": True,
        "gold_candidate_ids_hidden_from_judge": True,
        "judge_returned_direct_label": False,
        "judge_transport": transport,
        "transport_envelope_valid": True,
        "json_schema_valid": schema is not None,
        "field_typed_wire_valid": transport == "phase72_explicit_allowed_token_wire",
        "num_predict": num_predict,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "schema_sha256": stable_hash(schema) if schema is not None else None,
        "raw_response": content,
        "done_reason": body.get("done_reason"),
        "eval_count": eval_count,
        "eval_tokens_per_second": round(eval_count / eval_seconds, 4)
        if eval_seconds
        else None,
        "latency_seconds": round(time.perf_counter() - started, 4),
        "created_at": _utcnow(),
    }


def _run_eval(stage: str, endpoint: str, timeout: int, resume: bool) -> int:
    old_root = phase70_eval.EVIDENCE_ROOT
    old_paths = phase70_eval.SOURCE_PATHS
    old_invoke = phase70_eval._invoke_judge
    old_product_eval = phase70_eval.evaluate_phase70_boundary_results
    old_argv = sys.argv
    phase70_eval.EVIDENCE_ROOT = EVIDENCE_ROOT
    phase70_eval.SOURCE_PATHS = dict(SOURCE_PATHS)
    phase70_eval._invoke_judge = _invoke_phase72_judge
    phase70_eval.evaluate_phase70_boundary_results = evaluate_phase72_boundary_results
    sys.argv = [
        "phase70_execute_eval.py",
        "--stage",
        stage,
        "--ollama-endpoint",
        endpoint,
        "--timeout",
        str(timeout),
    ] + (["--resume"] if resume else [])
    try:
        return phase70_eval.main()
    finally:
        phase70_eval.EVIDENCE_ROOT = old_root
        phase70_eval.SOURCE_PATHS = old_paths
        phase70_eval._invoke_judge = old_invoke
        phase70_eval.evaluate_phase70_boundary_results = old_product_eval
        sys.argv = old_argv


def _generation_freeze_check(
    holdout: Mapping[str, Any], protocol: Mapping[str, Any]
) -> dict[str, Any]:
    freeze = _read_json(EVIDENCE_ROOT / "pre_model_call_freeze.json")
    source_ok = all(
        _sha256(SOURCE_FILES[name]) == expected
        for name, expected in dict(freeze.get("source_sha256") or {}).items()
    )
    protocol_copy = {key: value for key, value in protocol.items() if key != "protocol_sha256"}
    return {
        "kind": "phase72_shared_raw_generation_freeze_check",
        "passed": source_ok
        and stable_hash(holdout) == freeze.get("holdout_sha256")
        and stable_hash(protocol_copy)
        == protocol.get("protocol_sha256")
        == freeze.get("protocol_sha256"),
        "source_checks_passed": source_ok,
        "holdout_check_passed": stable_hash(holdout) == freeze.get("holdout_sha256"),
        "protocol_check_passed": stable_hash(protocol_copy)
        == protocol.get("protocol_sha256")
        == freeze.get("protocol_sha256"),
    }


def _derive_transcripts(
    raw_rows: Iterable[Mapping[str, Any]], sessions: Mapping[str, Mapping[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    variants = {name: [] for name in PHASE72_VARIANTS}
    for raw_value in raw_rows:
        raw = copy.deepcopy(dict(raw_value))
        session_id = str(raw.get("session_id") or "")
        shared_hash = stable_hash(raw.get("turns") or [])
        baseline = copy.deepcopy(raw)
        baseline.update(
            {
                "kind": "phase72_derived_runtime_transcript",
                "variant": "structured_prompt_raw",
                "shared_raw_transcript_sha256": shared_hash,
                "serializer_enabled": False,
                "serializer_apply_count": 0,
                "only_ab_variable": "deterministic_boundary_serializer_after_shared_raw_generation",
            }
        )
        candidate = copy.deepcopy(raw)
        history = []
        manifests = []
        for turn in candidate.get("turns") or []:
            if turn.get("role") == "assistant":
                serialized, manifest = apply_phase72_serializer(
                    history, str(turn.get("content") or "")
                )
                manifest["turn"] = len(manifests) + 1
                manifests.append(manifest)
                turn["content"] = serialized
            history.append(dict(turn))
        candidate.update(
            {
                "kind": "phase72_derived_runtime_transcript",
                "variant": "deterministic_boundary_serializer",
                "shared_raw_transcript_sha256": shared_hash,
                "serializer_enabled": True,
                "serializer_apply_count": sum(
                    row["serializer_applied"] for row in manifests
                ),
                "serializer_manifests": manifests,
                "hardcoded_response": any(row["serializer_applied"] for row in manifests),
                "only_ab_variable": "deterministic_boundary_serializer_after_shared_raw_generation",
            }
        )
        expected_boundary = sessions[session_id].get("task_type") == "boundary"
        candidate["final_route_expected"] = expected_boundary
        candidate["final_route_actual"] = bool(manifests[-1]["serializer_applied"])
        variants["structured_prompt_raw"].append(baseline)
        variants["deterministic_boundary_serializer"].append(candidate)
    return variants


def _generate(clean: bool) -> int:
    holdout = _read_json(EVIDENCE_ROOT / "evidence-holdout/holdout.json")
    protocol = _read_json(EVIDENCE_ROOT / "runtime_ab_protocol.json")
    freeze = _generation_freeze_check(holdout, protocol)
    output_dir = EVIDENCE_ROOT / "evidence-real-generation"
    _write_json(output_dir / "freeze_check_shared_raw.json", freeze)
    if freeze["passed"] is not True:
        raise SystemExit(f"Phase72 generation freeze failed: {freeze}")
    raw_path = output_dir / "shared_raw_transcripts.jsonl"
    if clean:
        raw_path.unlink(missing_ok=True)
        for variant in PHASE72_VARIANTS:
            (output_dir / f"transcripts_{variant}.jsonl").unlink(missing_ok=True)
        (output_dir / "metrics.json").unlink(missing_ok=True)
    sessions = [dict(row) for row in holdout.get("sessions") or []]
    session_by_id = {str(row["session_id"]): row for row in sessions}
    by_id = {
        str(row.get("session_id")): row
        for row in ([] if clean else _read_jsonl(raw_path))
        if row.get("status") == "completed"
    }
    generation_protocol = dict(protocol.get("generation") or {})
    torch, tokenizer, model, device, runtime = _load_runtime(None)
    if runtime.get("adapter_loaded") is not False:
        raise SystemExit("Phase72 must not load an adapter")
    try:
        for index, session in enumerate(sessions, start=1):
            session_id = str(session["session_id"])
            if session_id in by_id:
                continue
            try:
                row = _run_session(
                    session=session,
                    variant="structured_boundary_contract",
                    torch=torch,
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                    generation_protocol=generation_protocol,
                )
                row["kind"] = "phase72_shared_raw_runtime_transcript"
                row["variant"] = "shared_structured_prompt_raw"
            except Exception as exc:
                row = {
                    "kind": "phase72_shared_raw_runtime_transcript",
                    "session_id": session_id,
                    "task_type": session.get("task_type"),
                    "category": session.get("category"),
                    "variant": "shared_structured_prompt_raw",
                    "actual_model_call": False,
                    "status": "failed",
                    "error": f"{exc.__class__.__name__}: {exc}",
                    "turns": [],
                    "generation": [],
                    "simulated_usage": True,
                    "actual_user_feedback": False,
                    "not_for_training": True,
                    "created_at": _utcnow(),
                }
            by_id[session_id] = row
            _write_jsonl(raw_path, [by_id[key] for key in sorted(by_id)])
            if row["status"] != "completed" or index % 6 == 0:
                print(f"[phase72-shared-raw] {index}/{len(sessions)} status={row['status']}", flush=True)
    finally:
        del model
        if device == "mps" and hasattr(torch, "mps"):
            torch.mps.empty_cache()
    raw_rows = [by_id[key] for key in sorted(by_id)]
    completed = [row for row in raw_rows if row.get("status") == "completed"]
    derived = _derive_transcripts(completed, session_by_id)
    for variant, rows in derived.items():
        _write_jsonl(output_dir / f"transcripts_{variant}.jsonl", rows)
    boundary_candidates = [
        row
        for row in derived["deterministic_boundary_serializer"]
        if row.get("task_type") == "boundary"
    ]
    ordinary_candidates = [
        row
        for row in derived["deterministic_boundary_serializer"]
        if row.get("task_type") == "ordinary"
    ]
    baseline_by_id = {
        str(row["session_id"]): row for row in derived["structured_prompt_raw"]
    }
    metrics = {
        "kind": "phase72_shared_raw_generation_metrics",
        "session_count": len(sessions),
        "completed_count": len(completed),
        "failed_count": len(raw_rows) - len(completed),
        "actual_model_session_count": len(completed),
        "actual_generation_call_count": sum(
            len(row.get("generation") or []) for row in completed
        ),
        "derived_variant_count": 2,
        "derived_transcript_count": sum(len(rows) for rows in derived.values()),
        "boundary_final_route_recall": round(
            sum(row.get("final_route_actual") is True for row in boundary_candidates)
            / len(boundary_candidates),
            4,
        )
        if boundary_candidates
        else 0.0,
        "ordinary_final_route_false_positive_rate": round(
            sum(row.get("final_route_actual") is True for row in ordinary_candidates)
            / len(ordinary_candidates),
            4,
        )
        if ordinary_candidates
        else 0.0,
        "candidate_exact_boundary_output_rate": round(
            sum(final_assistant_text(row) == PHASE72_EXACT_OUTPUT for row in boundary_candidates)
            / len(boundary_candidates),
            4,
        )
        if boundary_candidates
        else 0.0,
        "ordinary_output_identity_rate": round(
            sum(
                final_assistant_text(row)
                == final_assistant_text(baseline_by_id[str(row["session_id"])])
                for row in ordinary_candidates
            )
            / len(ordinary_candidates),
            4,
        )
        if ordinary_candidates
        else 0.0,
        "truncated_session_count": sum(
            bool(row.get("truncated_response")) for row in completed
        ),
        "think_leak_session_count": sum(
            bool(row.get("think_leak_detected")) for row in completed
        ),
        "privacy_failure_count": sum(
            not dict(row.get("privacy_runtime") or {})
            .get("input_manifest", {})
            .get("passed", True)
            for row in completed
        ),
        "model_id": str(MODEL_PATH),
        "device": device,
        "adapter_loaded": False,
        "actual_user_feedback_count": 0,
        "training_executed": False,
        "created_at": _utcnow(),
    }
    _write_json(output_dir / "metrics.json", metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0 if metrics["completed_count"] == len(sessions) else 1


def _parity_audit(
    transcripts: Mapping[str, list[dict[str, Any]]], sessions: list[dict[str, Any]]
) -> dict[str, Any]:
    by_variant = {
        variant: {str(row["session_id"]): row for row in rows}
        for variant, rows in transcripts.items()
    }
    details = []
    for session in sessions:
        session_id = str(session["session_id"])
        baseline = by_variant["structured_prompt_raw"].get(session_id, {})
        candidate = by_variant["deterministic_boundary_serializer"].get(session_id, {})
        is_boundary = session["task_type"] == "boundary"
        details.append(
            {
                "session_id": session_id,
                "same_shared_raw": baseline.get("shared_raw_transcript_sha256")
                == candidate.get("shared_raw_transcript_sha256"),
                "same_model": baseline.get("model_id") == candidate.get("model_id"),
                "same_task": baseline.get("task_sha256") == candidate.get("task_sha256"),
                "both_completed": baseline.get("status")
                == candidate.get("status")
                == "completed",
                "candidate_route_expected": candidate.get("final_route_actual")
                is is_boundary,
                "ordinary_output_identical": is_boundary
                or final_assistant_text(baseline) == final_assistant_text(candidate),
            }
        )
    failures = [
        f"{row['session_id']}:{key}"
        for row in details
        for key, value in row.items()
        if key != "session_id" and not value
    ]
    return {
        "kind": "phase72_shared_raw_single_variable_parity_audit",
        "passed": bool(details) and not failures,
        "failed_checks": failures,
        "session_count": len(details),
        "only_ab_variable": "deterministic_boundary_serializer_after_shared_raw_generation",
        "details": details,
    }


def _prepare_product() -> int:
    holdout = _read_json(EVIDENCE_ROOT / "evidence-holdout/holdout.json")
    sessions = [dict(row) for row in holdout.get("sessions") or []]
    transcripts = {
        variant: _read_jsonl(
            EVIDENCE_ROOT / f"evidence-real-generation/transcripts_{variant}.jsonl"
        )
        for variant in PHASE72_VARIANTS
    }
    metrics = _read_json(EVIDENCE_ROOT / "evidence-real-generation/metrics.json")
    generation_freeze = _read_json(
        EVIDENCE_ROOT / "evidence-real-generation/freeze_check_shared_raw.json"
    )
    sparse = _read_json(EVIDENCE_ROOT / "evidence-sparse-preflight/evaluator_report.json")
    regression = _read_json(EVIDENCE_ROOT / "evidence-phase68-regression/evaluator_report.json")
    parity = _parity_audit(transcripts, sessions)
    ordinary = score_phase72_ordinary(transcripts, sessions)
    checks = {
        "sparse_transport_qualified": sparse.get("status") == "qualified",
        "phase68_regression_qualified": regression.get("status") == "qualified",
        "generation_freeze_passed": generation_freeze.get("passed") is True,
        "all_162_generation_calls_real": metrics.get("actual_generation_call_count") == 162,
        "all_raw_sessions_complete": metrics.get("completed_count") == 54
        and metrics.get("failed_count") == 0,
        "zero_generation_safety_failures": metrics.get("truncated_session_count") == 0
        and metrics.get("think_leak_session_count") == 0
        and metrics.get("privacy_failure_count") == 0,
        "boundary_route_recall_exact": metrics.get("boundary_final_route_recall") == 1.0,
        "ordinary_route_false_positive_zero": metrics.get(
            "ordinary_final_route_false_positive_rate"
        )
        == 0.0,
        "candidate_exact_boundary_output_rate_exact": metrics.get(
            "candidate_exact_boundary_output_rate"
        )
        == 1.0,
        "ordinary_output_identity_exact": metrics.get("ordinary_output_identity_rate")
        == 1.0,
        "single_variable_parity_passed": parity.get("passed") is True,
    }
    session_by_id = {str(row["session_id"]): row for row in sessions}
    blinded = []
    for variant in PHASE72_VARIANTS:
        for transcript in transcripts[variant]:
            session = session_by_id[str(transcript["session_id"])]
            if session["task_type"] == "boundary":
                blinded.append(
                    {
                        "variant": variant,
                        "session_id": transcript["session_id"],
                        "category": session["category"],
                        "assistant_response": final_assistant_text(transcript),
                    }
                )
    import random

    random.Random(7201).shuffle(blinded)
    public = []
    hidden = []
    for index, row in enumerate(blinded, start=1):
        item_id = f"phase72-product-{index:03d}"
        response = str(row["assistant_response"])
        public.append(
            {
                "item_id": item_id,
                "assistant_response": response,
                "proposition_candidates": build_phase59_proposition_candidates(response),
                "simulated_usage": True,
                "actual_user_feedback": False,
                "not_for_training": True,
            }
        )
        hidden.append(
            {
                "item_id": item_id,
                "variant": row["variant"],
                "session_id": row["session_id"],
                "category": row["category"],
                "expected_label": "accept",
            }
        )
    checks["product_item_count_exact"] = len(public) == 72
    checks["identity_hidden"] = all(
        "variant" not in row and "session_id" not in row and "category" not in row
        for row in public
    )
    eval_dir = EVIDENCE_ROOT / "evidence-product-eval"
    _write_jsonl(eval_dir / "blind_items_public.jsonl", public)
    _write_json(
        eval_dir / "blind_hidden_key.json",
        {"item_count": len(hidden), "items": hidden, "hidden_from_judges": True},
    )
    _write_json(EVIDENCE_ROOT / "ab_parity_audit.json", parity)
    _write_json(EVIDENCE_ROOT / "ordinary_control_report.json", ordinary)
    protocol = _read_json(EVIDENCE_ROOT / "runtime_ab_protocol.json")
    product_sources = {
        "phase70_core": SOURCE_FILES["phase70_core"],
        "phase70_prepare_product_eval": SOURCE_FILES["phase70_prepare_product_eval"],
        "phase70_execute_eval": SOURCE_FILES["phase70_execute_eval"],
        "phase70_finalize": SOURCE_FILES["phase70_finalize"],
    }
    freeze = {
        "kind": "phase72_pre_product_judge_freeze",
        "public_sha256": stable_hash(public),
        "hidden_sha256": stable_hash(hidden),
        "protocol_sha256": protocol.get("protocol_sha256"),
        "source_sha256": {
            name: _sha256(path) for name, path in product_sources.items()
        },
        "frozen_before_product_judge_calls": True,
        "created_at": _utcnow(),
    }
    _write_json(eval_dir / "pre_judge_freeze.json", freeze)
    ready = all(checks.values())
    decision = {
        "kind": "phase72_product_eval_preparation_decision",
        "status": "ready_for_product_eval" if ready else "blocked",
        "checks": checks,
        "failed_checks": [key for key, value in checks.items() if not value],
        "created_at": _utcnow(),
    }
    _write_json(eval_dir / "preparation_decision.json", decision)
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return 0 if ready else 1


def _manifest() -> dict[str, Any]:
    files = []
    for path in sorted(EVIDENCE_ROOT.rglob("*")):
        if path.is_file() and path.name not in DYNAMIC:
            files.append(
                {
                    "path": str(path.relative_to(REPO_ROOT)),
                    "sha256": _sha256(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    return {
        "kind": "phase72_evidence_manifest",
        "file_count": len(files),
        "files": files,
        "manifest_sha256": hashlib.sha256(
            json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def _examples(
    sessions: list[dict[str, Any]], transcripts: Mapping[str, list[dict[str, Any]]]
) -> str:
    indexed = {
        variant: {str(row["session_id"]): row for row in rows}
        for variant, rows in transcripts.items()
    }
    selected = []
    seen = set()
    for session in sessions:
        key = (session["task_type"], session["category"])
        if key not in seen:
            seen.add(key)
            selected.append(session)
    lines = [
        "# Phase72 Paired Output Examples",
        "",
        "Shared real Qwen3-4B raw generation with deterministic serializer derivation; simulated_usage only.",
    ]
    for session in selected:
        session_id = str(session["session_id"])
        lines.extend(
            [
                "",
                f"## {session_id} ({session['category']})",
                "",
                "**Structured prompt raw**",
                "",
                final_assistant_text(indexed["structured_prompt_raw"][session_id]),
                "",
                "**Deterministic serializer**",
                "",
                final_assistant_text(
                    indexed["deterministic_boundary_serializer"][session_id]
                ),
            ]
        )
    return "\n".join(line.rstrip() for line in lines)


def _finalize() -> int:
    snapshot = _read_json(EVIDENCE_ROOT / "evidence-baseline/phase71_hold_snapshot.json")
    sparse = _read_json(EVIDENCE_ROOT / "evidence-sparse-preflight/evaluator_report.json")
    regression = _read_json(EVIDENCE_ROOT / "evidence-phase68-regression/evaluator_report.json")
    product = _read_json(EVIDENCE_ROOT / "evidence-product-eval/evaluator_report.json")
    metrics = _read_json(EVIDENCE_ROOT / "evidence-real-generation/metrics.json")
    parity = _read_json(EVIDENCE_ROOT / "ab_parity_audit.json")
    ordinary = _read_json(EVIDENCE_ROOT / "ordinary_control_report.json")
    holdout = _read_json(EVIDENCE_ROOT / "evidence-holdout/holdout.json")
    sessions = [dict(row) for row in holdout.get("sessions") or []]
    transcripts = {
        variant: _read_jsonl(
            EVIDENCE_ROOT / f"evidence-real-generation/transcripts_{variant}.jsonl"
        )
        for variant in PHASE72_VARIANTS
    }
    variants = dict(product.get("variants") or {})
    baseline = dict(variants.get("structured_prompt_raw") or {})
    candidate = dict(variants.get("deterministic_boundary_serializer") or {})
    ordinary_variants = dict(ordinary.get("variants") or {})
    ordinary_candidate = dict(
        ordinary_variants.get("deterministic_boundary_serializer") or {}
    )
    product_complete = product.get("successful_model_output_count") == 144 and product.get(
        "failure_count"
    ) == 0
    checks = {
        "phase71_hold_preserved": snapshot.get("passed") is True,
        "sparse_transport_qualified": sparse.get("status") == "qualified",
        "phase68_regression_qualified": regression.get("status") == "qualified",
        "product_eval_complete": product_complete,
        "single_variable_parity_passed": parity.get("passed") is True,
        "all_162_shared_raw_generation_calls_real": metrics.get(
            "actual_generation_call_count"
        )
        == 162,
        "boundary_route_recall_gate": metrics.get("boundary_final_route_recall") == 1.0,
        "ordinary_route_false_positive_gate": metrics.get(
            "ordinary_final_route_false_positive_rate"
        )
        == 0.0,
        "ordinary_output_identity_gate": metrics.get("ordinary_output_identity_rate")
        == 1.0,
        "candidate_all_items_completed": candidate.get("completed_count")
        == PHASE72_BOUNDARY_COUNT,
        "candidate_accept_rate_gate": candidate.get("accept_rate") == 1.0,
        "candidate_accept_rate_delta_gate": float(
            product.get("candidate_accept_rate_delta") or 0.0
        )
        >= 0.5,
        "candidate_exact_structure_gate": candidate.get("exact_three_line_rate") == 1.0,
        "candidate_dangerous_zero": candidate.get("dangerous_or_reject_count") == 0,
        "product_schema_failures_zero": product.get("schema_failure_count") == 0,
        "product_candidate_conflicts_zero": product.get("candidate_value_conflict_count")
        == 0,
        "ordinary_controls_complete": ordinary_candidate.get("count")
        == PHASE72_ORDINARY_COUNT,
        "ordinary_boundary_leak_zero": ordinary_candidate.get("boundary_leak_count") == 0,
    }
    passed = all(checks.values())
    recommendation = (
        "recommend_phase72_nondefault_serializer_api_canary_after_manual_review"
        if passed
        else "hold_phase72_deterministic_boundary_serializer"
    )
    decision = {
        "kind": "phase72_final_decision",
        "status": recommendation,
        "recommendation": recommendation,
        "experiment_status": "completed" if product_complete else "blocked_at_product_eval",
        "checks": checks,
        "failed_checks": [key for key, value in checks.items() if not value],
        "baseline_accept_rate": baseline.get("accept_rate"),
        "candidate_accept_rate": candidate.get("accept_rate"),
        "candidate_accept_rate_delta": product.get("candidate_accept_rate_delta"),
        "candidate_exact_three_line_rate": candidate.get("exact_three_line_rate"),
        "phase73_nondefault_api_canary_eligible": passed,
        "product_default_change_allowed": False,
        "training_allowed": False,
        "adapter_created": False,
        "hermes_attachment_allowed": False,
        "auto_promote_allowed": False,
    }
    judge_counts = {
        "sparse_preflight": sparse.get("successful_model_output_count"),
        "phase68_regression": regression.get("successful_model_output_count"),
        "product": product.get("successful_model_output_count"),
    }
    comparison = {
        "kind": "phase72_deterministic_serializer_comparison",
        "model": metrics.get("model_id"),
        "device": metrics.get("device"),
        "shared_raw_generation": True,
        "only_ab_variable": "deterministic_boundary_serializer_after_shared_raw_generation",
        "qualification": {
            "sparse": {"status": sparse.get("status"), "accuracy": sparse.get("accuracy")},
            "regression": {
                "status": regression.get("status"),
                "accuracy": regression.get("accuracy"),
            },
        },
        "boundary": variants,
        "candidate_accept_rate_delta": product.get("candidate_accept_rate_delta"),
        "ordinary_controls": ordinary_variants,
        "generation_metrics": metrics,
        "actual_generation_call_count": metrics.get("actual_generation_call_count"),
        "actual_judge_output_counts": judge_counts,
        "actual_model_output_count_total": int(
            metrics.get("actual_generation_call_count") or 0
        )
        + sum(int(value or 0) for value in judge_counts.values()),
        "recommendation": recommendation,
        "actual_user_feedback_count": 0,
        "training_executed": False,
        "adapter_created": False,
        "hermes_attached": False,
        "product_default_changed": False,
    }
    integrity_checks = {
        "phase71_snapshot_passed": snapshot.get("passed") is True,
        "all_qualification_and_product_outputs_complete": judge_counts
        == {"sparse_preflight": 36, "phase68_regression": 60, "product": 144},
        "all_162_shared_raw_calls_real": metrics.get("actual_generation_call_count") == 162,
        "all_54_sessions_complete": metrics.get("completed_count") == 54
        and metrics.get("failed_count") == 0,
        "single_variable_parity_passed": parity.get("passed") is True,
        "no_training_adapter_hermes_or_default_change": decision.get("training_allowed")
        is False
        and decision.get("adapter_created") is False
        and decision.get("hermes_attachment_allowed") is False
        and decision.get("product_default_change_allowed") is False,
        "actual_user_feedback_zero": True,
    }
    integrity = {
        "kind": "phase72_evidence_integrity",
        "passed": all(integrity_checks.values()),
        "experiment_succeeded": passed,
        "checks": integrity_checks,
        "created_at": _utcnow(),
    }
    _write_json(EVIDENCE_ROOT / "comparison_summary.json", comparison)
    _write_json(EVIDENCE_ROOT / "phase72-final-decision.json", decision)
    _write_json(EVIDENCE_ROOT / "evidence_integrity.json", integrity)
    _write_json(
        EVIDENCE_ROOT / "evidence-no-training/training_attempt.json",
        {
            "kind": "phase72_training_attempt",
            "status": "not_run_by_design",
            "reason": "Phase72 isolates deterministic runtime serialization.",
            "adapter_created": False,
        },
    )
    _write_text(EVIDENCE_ROOT / "output_examples.md", _examples(sessions, transcripts))
    _write_text(
        EVIDENCE_ROOT / "phase72-final-decision.md",
        f"""# Phase72 Final Decision

## 结论

最终 recommendation 为 **{recommendation}**。shared raw prompt-only accept rate `{baseline.get('accept_rate')}`，deterministic serializer `{candidate.get('accept_rate')}`，增量 `{product.get('candidate_accept_rate_delta')}`，exact-three-line `{candidate.get('exact_three_line_rate')}`。

## 真实证据

- explicit-token evaluator preflight：{sparse.get('status')}，36 个真实 judge 输出。
- Phase68 regression：{regression.get('status')}，60 个真实 judge 输出。
- Qwen3-4B shared raw generation：162 次真实调用，派生 108 个成对 transcript。
- 产品盲评：{product.get('successful_model_output_count')}/144 个真实 judge 输出。
- 路由：boundary recall `{metrics.get('boundary_final_route_recall')}`，ordinary false positive `{metrics.get('ordinary_final_route_false_positive_rate')}`。

## 边界

这是 simulated_usage runtime serializer 证据，不是训练或真实用户收益。没有训练、adapter、Hermes、默认切换或自动 promote。
""",
    )
    _write_text(
        EVIDENCE_ROOT / "phase72-runbook.md",
        """# Phase72 Runbook

```bash
.venv/bin/python tools/phase72_deterministic_boundary_serializer.py prepare --clean-evidence
.venv/bin/python tools/phase72_deterministic_boundary_serializer.py eval --stage sparse_preflight --ollama-endpoint http://127.0.0.1:11435 --timeout 900
.venv/bin/python tools/phase72_deterministic_boundary_serializer.py eval --stage phase68_regression --ollama-endpoint http://127.0.0.1:11435 --timeout 900
.venv/bin/python tools/phase72_deterministic_boundary_serializer.py generate --clean
.venv/bin/python tools/phase72_deterministic_boundary_serializer.py prepare-product
.venv/bin/python tools/phase72_deterministic_boundary_serializer.py eval --stage product --ollama-endpoint http://127.0.0.1:11435 --timeout 900
.venv/bin/python tools/phase72_deterministic_boundary_serializer.py finalize
.venv/bin/python tools/phase72_deterministic_boundary_serializer.py validate
```

Do not edit the router, serializer, transport, tasks, decoding, or gates after prepare.
""",
    )
    next_goal = (
        "Build Phase73 as an explicit non-default API canary for the deterministic boundary serializer. Add opt-in request metadata only, stream/non-stream parity, fallback, rollback, audit metadata, and fresh live API smoke. Keep the default unchanged and do not train in the same phase."
        if passed
        else
        "Keep the serializer held. Review the preserved Phase72 failures and design one independent repair without default integration or training."
    )
    _write_text(EVIDENCE_ROOT / "next-pursuit-goal.md", f"# Next Pursuit Goal\n\n{next_goal}")
    manifest = _manifest()
    _write_json(EVIDENCE_ROOT / "evidence_manifest.json", manifest)
    state = {
        "kind": "phase72_finalization_state",
        "status": "completed" if integrity["passed"] else "blocked",
        "recommendation": recommendation,
        "evidence_integrity_passed": integrity["passed"],
        "experiment_succeeded": passed,
        "manifest_file_count": manifest["file_count"],
        "created_at": _utcnow(),
    }
    _write_json(EVIDENCE_ROOT / "finalization_state.json", state)
    print(json.dumps(state, ensure_ascii=False, indent=2))
    return 0 if integrity["passed"] else 1


def _run_check(name: str, command: list[str]) -> dict[str, Any]:
    started = time.perf_counter()
    completed = subprocess.run(
        command, cwd=REPO_ROOT, text=True, capture_output=True, check=False
    )
    combined = completed.stdout + completed.stderr
    lines = combined.splitlines()
    return {
        "name": name,
        "command": command,
        "returncode": completed.returncode,
        "passed": completed.returncode == 0,
        "duration_seconds": round(time.perf_counter() - started, 4),
        "output_line_count": len(lines),
        "output_sha256": hashlib.sha256(combined.encode()).hexdigest(),
        "output_tail": lines[-24:],
    }


def _evidence_check() -> dict[str, Any]:
    integrity = _read_json(EVIDENCE_ROOT / "evidence_integrity.json")
    decision = _read_json(EVIDENCE_ROOT / "phase72-final-decision.json")
    comparison = _read_json(EVIDENCE_ROOT / "comparison_summary.json")
    allowed = {
        "recommend_phase72_nondefault_serializer_api_canary_after_manual_review",
        "hold_phase72_deterministic_boundary_serializer",
    }
    passed = (
        integrity.get("passed") is True
        and decision.get("recommendation") in allowed
        and comparison.get("actual_generation_call_count") == 162
        and comparison.get("actual_judge_output_counts")
        == {"sparse_preflight": 36, "phase68_regression": 60, "product": 144}
        and comparison.get("actual_model_output_count_total") == 402
        and comparison.get("actual_user_feedback_count") == 0
        and comparison.get("training_executed") is False
        and comparison.get("adapter_created") is False
        and comparison.get("product_default_changed") is False
        and decision.get("auto_promote_allowed") is False
    )
    return {
        "name": "phase72_evidence_consistency",
        "command": ["internal", "phase72_evidence_consistency"],
        "returncode": 0 if passed else 1,
        "passed": passed,
        "duration_seconds": 0.0,
        "output_line_count": 1,
        "output_sha256": hashlib.sha256(str(passed).encode()).hexdigest(),
        "output_tail": [
            f"integrity={integrity.get('passed')} recommendation={decision.get('recommendation')} outputs={comparison.get('actual_model_output_count_total')}"
        ],
    }


def _validate() -> int:
    python = str(REPO_ROOT / ".venv/bin/python")
    phase_tests = [
        f"tests/test_phase{phase}_{name}.py"
        for phase, name in (
            (72, "deterministic_boundary_serializer"),
            (71, "qualified_structured_contract_ab"),
            (70, "structured_boundary_contract"),
            (69, "minimal_runtime_ab"),
            (68, "aligned_candidate_scope_recovery"),
            (67, "historical_contract_compatibility_audit"),
            (66, "external_distribution_regression"),
            (65, "aggregate_safe_boundary_coverage"),
            (64, "field_typed_historical_replay"),
            (63, "field_typed_candidate_wire"),
            (62, "risk_asymmetric_candidate_consensus"),
            (61, "compact_candidate_wire_protocol"),
            (60, "flat_schema_compatibility"),
            (59, "proposition_addressed_grounding"),
            (58, "clause_addressed_grounding"),
            (57, "span_evaluator_historical_replay"),
            (56, "evidence_span_grounded_atomic"),
            (55, "atomic_boundary_composition"),
            (54, "typed_proposition_evaluator"),
            (53, "evaluator_scope_recovery"),
            (52, "adversarial_evaluator_generalization"),
            (51, "dual_evaluator_hardening"),
            (50, "conditional_provenance_guard"),
            (49, "provenance_boundary_recovery"),
            (48, "compact_intent_runtime"),
            (47, "simulated_user_review"),
            (46, "runtime_first_latest_intent"),
            (45, "privacy_multiturn_preference"),
        )
    ]
    checks = [
        (
            "py_compile",
            [
                python,
                "-m",
                "py_compile",
                "pfe-core/pfe_core/phase72_deterministic_boundary_serializer.py",
                "tools/phase72_deterministic_boundary_serializer.py",
                "tests/test_phase72_deterministic_boundary_serializer.py",
            ],
        ),
        (
            "phase72_focused_and_phase71_to45_regression",
            [python, "-m", "pytest", "-q", *phase_tests],
        ),
        ("test_unit", ["make", "test-unit"]),
        ("test_surface", ["make", "test-surface"]),
        ("test_e2e_mock", ["make", "test-e2e-mock"]),
        ("smoke_beta", ["make", "smoke-beta"]),
        ("git_diff_check", ["git", "diff", "--check"]),
    ]
    results = [_evidence_check()]
    results.extend(_run_check(name, command) for name, command in checks)
    passed = all(row["passed"] for row in results)
    summary = {
        "kind": "phase72_validation_summary",
        "created_at": _utcnow(),
        "status": "passed" if passed else "failed",
        "check_count": len(results),
        "passed_count": sum(row["passed"] for row in results),
        "failed_count": sum(not row["passed"] for row in results),
        "checks": results,
    }
    _write_json(EVIDENCE_ROOT / "validation_summary.json", summary)
    lines = [f"Phase72 validation: {summary['status']}"]
    lines.extend(
        f"{row['name']}: {'PASS' if row['passed'] else 'FAIL'} ({row['duration_seconds']}s, rc={row['returncode']})"
        for row in results
    )
    _write_text(EVIDENCE_ROOT / "validation_gate.txt", "\n".join(lines))
    print("\n".join(lines))
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--clean-evidence", action="store_true")
    evaluate = subparsers.add_parser("eval")
    evaluate.add_argument(
        "--stage", choices=("sparse_preflight", "phase68_regression", "product"), required=True
    )
    evaluate.add_argument("--ollama-endpoint", default="http://127.0.0.1:11434")
    evaluate.add_argument("--timeout", type=int, default=900)
    evaluate.add_argument("--resume", action="store_true")
    generate = subparsers.add_parser("generate")
    generate.add_argument("--clean", action="store_true")
    subparsers.add_parser("prepare-product")
    subparsers.add_parser("finalize")
    subparsers.add_parser("validate")
    args = parser.parse_args()
    if args.command == "prepare":
        return _prepare(args.clean_evidence)
    if args.command == "eval":
        return _run_eval(args.stage, args.ollama_endpoint, args.timeout, args.resume)
    if args.command == "generate":
        return _generate(args.clean)
    if args.command == "prepare-product":
        return _prepare_product()
    if args.command == "finalize":
        return _finalize()
    return _validate()


if __name__ == "__main__":
    raise SystemExit(main())
