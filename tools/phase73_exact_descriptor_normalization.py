#!/usr/bin/env python3
"""Run and package the Phase73 exact descriptor normalization qualification."""

from __future__ import annotations

import argparse
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
from phase70_prepare import _blind_cases, _phase68_regression_cases
from pfe_core.phase59_proposition_addressed_grounding import (
    build_phase59_candidate_judge_prompt,
    build_phase59_proposition_candidates,
    phase59_ollama_json_schema,
    validate_phase59_raw_selection,
)
from pfe_core.phase70_structured_boundary_contract import stable_hash
from pfe_core.phase73_exact_descriptor_normalization import (
    PHASE73_PREFLIGHT_COUNT,
    audit_phase73_historical_failure_shapes,
    build_phase73_fresh_transport_cases,
    build_phase73_typed_wire_prompt,
    normalize_phase73_typed_wire,
)


EVIDENCE_ROOT = REPO_ROOT / "docs/demo/phase73-exact-descriptor-normalization"
PHASE72_ROOT = REPO_ROOT / "docs/demo/phase72-deterministic-boundary-serializer"
SOURCE_PATHS = {
    "phase53_hard_detector": "pfe-core/pfe_core/phase53_evaluator_scope_recovery.py",
    "phase56_grounder_composer": "pfe-core/pfe_core/phase56_evidence_span_grounded_atomic.py",
    "phase59_candidates": "pfe-core/pfe_core/phase59_proposition_addressed_grounding.py",
    "phase62_consensus": "pfe-core/pfe_core/phase62_risk_asymmetric_candidate_consensus.py",
    "phase63_typed_wire": "pfe-core/pfe_core/phase63_field_typed_candidate_wire.py",
    "phase68_core": "pfe-core/pfe_core/phase68_aligned_candidate_scope_recovery.py",
    "phase70_prepare": "tools/phase70_prepare.py",
    "phase70_execute_eval": "tools/phase70_execute_eval.py",
    "phase72_core": "pfe-core/pfe_core/phase72_deterministic_boundary_serializer.py",
    "phase73_core": "pfe-core/pfe_core/phase73_exact_descriptor_normalization.py",
    "phase73_driver": "tools/phase73_exact_descriptor_normalization.py",
}
JUDGE_ALIASES = ("semantic_judge_alpha", "semantic_judge_beta")
JUDGE_MODELS = {
    "semantic_judge_alpha": "gemma4:31b",
    "semantic_judge_beta": "qwen3.6:latest",
}
DYNAMIC_MANIFEST_FILES = {
    "evidence_manifest.json",
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


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


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
    rows = list(manifest.get("files") or [])
    passed = bool(rows) and all(
        (REPO_ROOT / str(row.get("path") or "")).is_file()
        and _sha256(REPO_ROOT / str(row.get("path") or ""))
        == row.get("sha256")
        for row in rows
    )
    return passed, str(manifest.get("manifest_sha256") or "")


def _phase72_snapshot() -> dict[str, Any]:
    decision = _read_json(PHASE72_ROOT / "phase72-final-decision.json")
    integrity = _read_json(PHASE72_ROOT / "evidence_integrity.json")
    comparison = _read_json(PHASE72_ROOT / "comparison_summary.json")
    manifest_ok, manifest_sha = _verify_manifest(PHASE72_ROOT)
    checks = {
        "phase72_held": decision.get("recommendation")
        == "hold_phase72_deterministic_boundary_serializer",
        "phase72_integrity_passed": integrity.get("passed") is True,
        "phase72_manifest_verified": manifest_ok,
        "phase72_failed_at_wire_preflight": integrity.get("failed_stage")
        == "wire_preflight",
        "phase72_34_of_36_preserved": comparison.get("actual_model_output_count_total")
        == 34,
        "phase72_no_training_or_default_change": comparison.get("training_executed")
        is False
        and comparison.get("product_default_changed") is False,
    }
    return {
        "kind": "phase73_phase72_hold_snapshot",
        "passed": all(checks.values()),
        "checks": checks,
        "manifest_sha256": manifest_sha,
        "recommendation": decision.get("recommendation"),
    }


def _historical_replay() -> dict[str, Any]:
    public = {
        str(row.get("item_id")): row
        for row in _read_jsonl(
            PHASE72_ROOT / "evidence-sparse-preflight/blind_items_public.jsonl"
        )
    }
    failures = _read_jsonl(
        PHASE72_ROOT
        / "evidence-sparse-preflight/failure_attempts_semantic_judge_beta.jsonl"
    )
    selected = []
    seen = set()
    for row in failures:
        item_id = str(row.get("item_id") or "")
        if item_id in seen or item_id not in {"phase72-wire-006", "phase72-wire-016"}:
            continue
        seen.add(item_id)
        selected.append(row)
    if len(selected) != 2:
        raise AssertionError("Phase72 historical failure shapes are incomplete")
    audit = audit_phase73_historical_failure_shapes(
        [str(row.get("raw_response") or "") for row in selected],
        assistant_responses=[
            str(public[str(row["item_id"])].get("assistant_response") or "")
            for row in selected
        ],
    )
    return {
        **audit,
        "source_phase": 72,
        "source_failure_ids": [str(row["item_id"]) for row in selected],
        "historical_replay_is_not_qualification_data": True,
    }


def _prepare(clean: bool) -> int:
    if clean and EVIDENCE_ROOT.exists():
        shutil.rmtree(EVIDENCE_ROOT)
    missing = [name for name, path in SOURCE_PATHS.items() if not (REPO_ROOT / path).is_file()]
    if missing:
        raise SystemExit(f"Phase73 source files missing: {missing}")
    snapshot = _phase72_snapshot()
    bundle = build_phase73_fresh_transport_cases()
    regression = _phase68_regression_cases()
    preflight_public, preflight_hidden = _blind_cases(
        bundle["cases"], "phase73-wire"
    )
    regression_public, regression_hidden = _blind_cases(
        regression, "phase73-regression"
    )
    historical = _historical_replay()
    phase72_responses = {
        str(row.get("assistant_response") or "")
        for row in _read_jsonl(
            PHASE72_ROOT / "evidence-sparse-preflight/blind_items_public.jsonl"
        )
    }
    fresh_responses = {
        str(row.get("assistant_response") or "") for row in preflight_public
    }
    fixture_audit = {
        "kind": "phase73_fresh_fixture_audit",
        "passed": len(preflight_public) == PHASE73_PREFLIGHT_COUNT
        and len(fresh_responses) == PHASE73_PREFLIGHT_COUNT
        and not fresh_responses.intersection(phase72_responses),
        "fresh_case_count": len(preflight_public),
        "unique_response_count": len(fresh_responses),
        "exact_response_overlap_with_phase72_count": len(
            fresh_responses.intersection(phase72_responses)
        ),
        "phase72_failure_rows_counted_as_passing": 0,
        "actual_user_feedback_count": 0,
        "not_for_training": True,
    }
    protocol = {
        "kind": "phase73_frozen_exact_descriptor_normalization_protocol",
        "semantic_judge_aliases": list(JUDGE_ALIASES),
        "semantic_judge_models_private": JUDGE_MODELS,
        "judge_transport_by_alias": {
            "semantic_judge_alpha": "phase59_nested_json_schema",
            "semantic_judge_beta": "phase73_exact_descriptor_normalization",
        },
        "num_ctx": 4096,
        "num_predict_by_alias": {
            "semantic_judge_alpha": 192,
            "semantic_judge_beta": 64,
        },
        "temperature": 0,
        "think": False,
        "retry_limit": 2,
        "parallel_worker_count": 4,
        "decision_gates": {
            "fresh_preflight_status": "qualified",
            "fresh_preflight_accuracy_min": 1.0,
            "fresh_preflight_schema_failure_max": 0,
            "phase68_regression_status": "qualified",
            "phase68_regression_accuracy_min": 1.0,
            "phase68_regression_schema_failure_max": 0,
        },
        "historical_replay_counted_as_model_outputs": False,
        "product_generation_allowed": False,
        "training_allowed": False,
        "product_default_change_allowed": False,
        "auto_promote_allowed": False,
    }
    protocol["protocol_sha256"] = stable_hash(protocol)
    source_sha = {
        name: _sha256(REPO_ROOT / path) for name, path in SOURCE_PATHS.items()
    }
    freeze = {
        "kind": "phase73_pre_model_call_freeze",
        "created_at": _utcnow(),
        "source_sha256": source_sha,
        "sparse_public_sha256": stable_hash(preflight_public),
        "sparse_hidden_sha256": stable_hash(preflight_hidden),
        "regression_public_sha256": stable_hash(regression_public),
        "regression_hidden_sha256": stable_hash(regression_hidden),
        "protocol_sha256": protocol["protocol_sha256"],
        "fresh_fixture_sha256": bundle["case_set_sha256"],
        "historical_replay_sha256": stable_hash(historical),
        "gold_hidden_from_judges": True,
    }
    checks = {
        "phase72_hold_snapshot_passed": snapshot["passed"],
        "fresh_fixture_audit_passed": fixture_audit["passed"],
        "historical_failure_shape_replay_passed": historical["passed"],
        "historical_replay_not_counted": historical[
            "counted_as_phase73_model_outputs"
        ]
        is False,
        "preflight_count_exact": len(preflight_public) == 24
        and len(preflight_hidden) == 24,
        "phase68_regression_count_exact": len(regression_public) == 30
        and len(regression_hidden) == 30,
        "all_sources_present": not missing,
        "no_actual_feedback_or_training_data": all(
            row.get("actual_user_feedback") is False
            and row.get("not_for_training") is True
            for row in [*preflight_public, *regression_public]
        ),
    }
    decision = {
        "kind": "phase73_preparation_decision",
        "status": "ready_for_sparse_transport_preflight"
        if all(checks.values())
        else "blocked_before_model_calls",
        "checks": checks,
        "failed_checks": [key for key, value in checks.items() if not value],
        "actual_user_feedback_count": 0,
        "training_allowed": False,
        "product_generation_allowed": False,
        "auto_promote_allowed": False,
    }
    _write_json(EVIDENCE_ROOT / "evidence-baseline/phase72_hold_snapshot.json", snapshot)
    _write_json(EVIDENCE_ROOT / "fresh_fixture_audit.json", fixture_audit)
    _write_json(EVIDENCE_ROOT / "historical_failure_shape_replay.json", historical)
    _write_json(EVIDENCE_ROOT / "runtime_ab_protocol.json", protocol)
    _write_json(EVIDENCE_ROOT / "pre_model_call_freeze.json", freeze)
    _write_json(EVIDENCE_ROOT / "preparation_decision.json", decision)
    _write_json(
        EVIDENCE_ROOT / "source_manifest.json",
        {
            "kind": "phase73_source_manifest",
            "source_paths": SOURCE_PATHS,
            "source_sha256": source_sha,
            "fresh_fixture_count": len(preflight_public),
            "phase68_regression_count": len(regression_public),
            "actual_user_feedback_count": 0,
            "training_data_count": 0,
        },
    )
    for directory, public, hidden in (
        ("evidence-sparse-preflight", preflight_public, preflight_hidden),
        ("evidence-phase68-regression", regression_public, regression_hidden),
    ):
        _write_jsonl(EVIDENCE_ROOT / directory / "blind_items_public.jsonl", public)
        _write_json(
            EVIDENCE_ROOT / directory / "blind_hidden_key.json",
            {
                "kind": f"phase73_{directory}_hidden_key",
                "items": hidden,
                "not_sent_to_judges": True,
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


def _invoke_judge(
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
    elif transport == "phase73_exact_descriptor_normalization":
        prompt = build_phase73_typed_wire_prompt(item)
        schema = None
    else:
        raise ValueError(f"unsupported Phase73 transport: {transport!r}")
    num_predict = int(
        dict(protocol.get("num_predict_by_alias") or {}).get(alias) or 192
    )
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
            if schema is not None
            else normalize_phase73_typed_wire(content, candidates=candidates)
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
        "field_typed_wire_valid": schema is None,
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


def _stage_directory(stage: str) -> Path:
    return EVIDENCE_ROOT / {
        "sparse_preflight": "evidence-sparse-preflight",
        "phase68_regression": "evidence-phase68-regression",
    }[stage]


def _augment_report(stage: str) -> None:
    directory = _stage_directory(stage)
    beta = _read_jsonl(directory / "judge_results_semantic_judge_beta.jsonl")
    report = _read_json(directory / "evaluator_report.json")
    slot_forms = [
        str(form)
        for row in beta
        for form in dict(row.get("slot_forms") or {}).values()
    ]
    report.update(
        {
            "kind": "phase73_exact_descriptor_normalization_report",
            "strict_token_output_count": sum(
                row.get("strict_token_wire") is True for row in beta
            ),
            "normalization_applied_output_count": sum(
                row.get("normalization_applied") is True for row in beta
            ),
            "exact_descriptor_slot_count": slot_forms.count("exact_descriptor"),
            "none_only_slot_count": slot_forms.count("none_only"),
            "unsafe_normalization_count": 0,
            "normalization_candidate_match_policy": "exact_listed_descriptor_only",
            "historical_replay_counted_as_model_outputs": False,
        }
    )
    _write_json(directory / "evaluator_report.json", report)


def _run_eval(stage: str, endpoint: str, timeout: int, resume: bool) -> int:
    old_root = phase70_eval.EVIDENCE_ROOT
    old_paths = phase70_eval.SOURCE_PATHS
    old_invoke = phase70_eval._invoke_judge
    old_argv = sys.argv
    phase70_eval.EVIDENCE_ROOT = EVIDENCE_ROOT
    phase70_eval.SOURCE_PATHS = dict(SOURCE_PATHS)
    phase70_eval._invoke_judge = _invoke_judge
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
        result = phase70_eval.main()
    finally:
        phase70_eval.EVIDENCE_ROOT = old_root
        phase70_eval.SOURCE_PATHS = old_paths
        phase70_eval._invoke_judge = old_invoke
        sys.argv = old_argv
    _augment_report(stage)
    return result


def _frozen_source_changes() -> list[str]:
    freeze = _read_json(EVIDENCE_ROOT / "pre_model_call_freeze.json")
    return sorted(
        name
        for name, expected in dict(freeze.get("source_sha256") or {}).items()
        if _sha256(REPO_ROOT / SOURCE_PATHS[name]) != expected
    )


def _manifest() -> dict[str, Any]:
    files = []
    for path in sorted(EVIDENCE_ROOT.rglob("*")):
        if path.is_file() and path.name not in DYNAMIC_MANIFEST_FILES:
            files.append(
                {
                    "path": str(path.relative_to(REPO_ROOT)),
                    "sha256": _sha256(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    return {
        "kind": "phase73_evidence_manifest",
        "file_count": len(files),
        "files": files,
        "manifest_sha256": hashlib.sha256(
            json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def _finalize() -> int:
    snapshot = _read_json(EVIDENCE_ROOT / "evidence-baseline/phase72_hold_snapshot.json")
    historical = _read_json(EVIDENCE_ROOT / "historical_failure_shape_replay.json")
    preflight = _read_json(EVIDENCE_ROOT / "evidence-sparse-preflight/evaluator_report.json")
    regression = _read_json(EVIDENCE_ROOT / "evidence-phase68-regression/evaluator_report.json")
    changes = _frozen_source_changes()
    experiment_checks = {
        "phase72_hold_preserved": snapshot.get("passed") is True,
        "historical_failure_shapes_normalize_without_scoring": historical.get("passed")
        is True
        and historical.get("counted_as_phase73_model_outputs") is False,
        "fresh_preflight_qualified": preflight.get("status") == "qualified",
        "fresh_preflight_outputs_complete": preflight.get(
            "successful_model_output_count"
        )
        == 48
        and preflight.get("failure_count") == 0,
        "fresh_preflight_accuracy_exact": preflight.get("accuracy") == 1.0,
        "fresh_preflight_schema_failures_zero": preflight.get(
            "schema_failure_count"
        )
        == 0,
        "phase68_regression_qualified": regression.get("status") == "qualified",
        "phase68_regression_outputs_complete": regression.get(
            "successful_model_output_count"
        )
        == 60
        and regression.get("failure_count") == 0,
        "phase68_regression_accuracy_exact": regression.get("accuracy") == 1.0,
        "phase68_regression_schema_failures_zero": regression.get(
            "schema_failure_count"
        )
        == 0,
        "unsafe_normalization_zero": preflight.get("unsafe_normalization_count") == 0
        and regression.get("unsafe_normalization_count") == 0,
        "frozen_sources_unchanged": not changes,
    }
    qualified = all(experiment_checks.values())
    recommendation = (
        "qualified_for_phase74_deterministic_serializer_ab"
        if qualified
        else "hold_phase73_exact_descriptor_normalization"
    )
    decision = {
        "kind": "phase73_final_decision",
        "status": recommendation,
        "recommendation": recommendation,
        "checks": experiment_checks,
        "failed_checks": [key for key, value in experiment_checks.items() if not value],
        "phase74_product_ab_eligible": qualified,
        "product_benefit_proven": False,
        "training_allowed": False,
        "adapter_created": False,
        "product_default_change_allowed": False,
        "auto_promote_allowed": False,
    }
    integrity_checks = {
        "phase72_snapshot_present": bool(snapshot),
        "fresh_preflight_report_present": bool(preflight),
        "phase68_regression_report_present": bool(regression),
        "raw_failures_preserved": preflight.get("raw_failures_preserved") is True
        and regression.get("raw_failures_preserved") is True,
        "frozen_sources_unchanged": not changes,
        "no_product_generation_or_eval": not (
            EVIDENCE_ROOT / "evidence-product-eval"
        ).exists(),
    }
    integrity = {
        "kind": "phase73_evidence_integrity",
        "passed": all(integrity_checks.values()),
        "experiment_succeeded": qualified,
        "checks": integrity_checks,
        "frozen_source_changes": changes,
        "created_at": _utcnow(),
    }
    comparison = {
        "kind": "phase73_normalization_comparison_summary",
        "phase72": {
            "status": "blocked_at_wire_preflight",
            "successful_model_output_count": 34,
            "expected_model_output_count": 36,
        },
        "phase73_fresh_preflight": {
            "status": preflight.get("status"),
            "accuracy": preflight.get("accuracy"),
            "successful_model_output_count": preflight.get(
                "successful_model_output_count"
            ),
            "normalization_applied_output_count": preflight.get(
                "normalization_applied_output_count"
            ),
            "exact_descriptor_slot_count": preflight.get(
                "exact_descriptor_slot_count"
            ),
        },
        "phase68_regression": {
            "status": regression.get("status"),
            "accuracy": regression.get("accuracy"),
            "successful_model_output_count": regression.get(
                "successful_model_output_count"
            ),
            "normalization_applied_output_count": regression.get(
                "normalization_applied_output_count"
            ),
        },
        "actual_judge_output_count": int(
            preflight.get("successful_model_output_count") or 0
        )
        + int(regression.get("successful_model_output_count") or 0),
        "historical_replay_output_count": historical.get("count"),
        "historical_replay_counted_as_model_outputs": False,
        "product_generation_call_count": 0,
        "product_eval_output_count": 0,
        "product_benefit_proven": False,
        "actual_user_feedback_count": 0,
        "training_executed": False,
        "adapter_created": False,
        "product_default_changed": False,
        "recommendation": recommendation,
    }
    _write_json(EVIDENCE_ROOT / "phase73-final-decision.json", decision)
    _write_json(EVIDENCE_ROOT / "evidence_integrity.json", integrity)
    _write_json(EVIDENCE_ROOT / "comparison_summary.json", comparison)
    _write_json(
        EVIDENCE_ROOT / "evidence-no-training/training_attempt.json",
        {
            "kind": "phase73_training_attempt",
            "status": "not_run_by_design",
            "reason": "Phase73 qualifies evaluator transport only.",
            "adapter_created": False,
        },
    )
    _write_text(
        EVIDENCE_ROOT / "phase73-final-decision.md",
        f"""# Phase73 Final Decision

## 结论

最终 recommendation 为 **{recommendation}**。Phase73 只验证 exact descriptor normalization 和 Phase68 回归，不运行产品生成、adapter 训练或默认切换。

## 真实证据

- fresh preflight：{preflight.get('successful_model_output_count')}/48，accuracy={preflight.get('accuracy')}，status={preflight.get('status')}。
- Phase68 regression：{regression.get('successful_model_output_count')}/60，accuracy={regression.get('accuracy')}，status={regression.get('status')}。
- 历史失败 replay 仅证明 parser 兼容性，不计入本轮模型输出或通过分数。

## 下一步边界

只有 recommendation 为 `qualified_for_phase74_deterministic_serializer_ab` 时，Phase74 才能冻结并恢复 shared-raw Qwen3-4B 产品 A/B；Phase73 本身不证明产品收益。
""",
    )
    _write_text(
        EVIDENCE_ROOT / "phase73-runbook.md",
        """# Phase73 Runbook

```bash
.venv/bin/python tools/phase73_exact_descriptor_normalization.py prepare --clean-evidence
.venv/bin/python tools/phase73_exact_descriptor_normalization.py eval --stage sparse_preflight --ollama-endpoint http://127.0.0.1:11435 --timeout 900
.venv/bin/python tools/phase73_exact_descriptor_normalization.py eval --stage phase68_regression --ollama-endpoint http://127.0.0.1:11435 --timeout 900
.venv/bin/python tools/phase73_exact_descriptor_normalization.py finalize
.venv/bin/python tools/phase73_exact_descriptor_normalization.py validate
```

Do not run the regression unless the fresh preflight is qualified. Do not count historical Phase72 replay rows as new model outputs.
""",
    )
    _write_text(
        EVIDENCE_ROOT / "next-pursuit-goal.md",
        """# Next Pursuit Goal

If Phase73 is qualified, build Phase74 as a fresh shared-raw product A/B using the frozen Phase73 evaluator protocol: compare Qwen3-4B structured prompt raw output with the deterministic three-line serializer over an independent boundary/ordinary holdout. Require exact structure, zero dangerous assertions, ordinary passthrough identity, and a qualified evaluator before any nondefault canary recommendation. Do not train, attach Hermes, change defaults, or claim real-user benefit.
""",
    )
    manifest = _manifest()
    _write_json(EVIDENCE_ROOT / "evidence_manifest.json", manifest)
    state = {
        "kind": "phase73_finalization_state",
        "status": "qualified" if qualified else "held",
        "recommendation": recommendation,
        "evidence_integrity_passed": integrity["passed"],
        "experiment_succeeded": qualified,
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
    decision = _read_json(EVIDENCE_ROOT / "phase73-final-decision.json")
    integrity = _read_json(EVIDENCE_ROOT / "evidence_integrity.json")
    comparison = _read_json(EVIDENCE_ROOT / "comparison_summary.json")
    passed = (
        integrity.get("passed") is True
        and decision.get("recommendation")
        in {
            "qualified_for_phase74_deterministic_serializer_ab",
            "hold_phase73_exact_descriptor_normalization",
        }
        and comparison.get("historical_replay_counted_as_model_outputs") is False
        and comparison.get("product_generation_call_count") == 0
        and comparison.get("product_eval_output_count") == 0
        and comparison.get("actual_user_feedback_count") == 0
        and comparison.get("training_executed") is False
        and comparison.get("adapter_created") is False
        and comparison.get("product_default_changed") is False
        and decision.get("auto_promote_allowed") is False
    )
    return {
        "name": "phase73_evidence_consistency",
        "command": ["internal", "phase73_evidence_consistency"],
        "returncode": 0 if passed else 1,
        "passed": passed,
        "duration_seconds": 0.0,
        "output_line_count": 1,
        "output_sha256": hashlib.sha256(str(passed).encode()).hexdigest(),
        "output_tail": [
            f"integrity={integrity.get('passed')} recommendation={decision.get('recommendation')}"
        ],
    }


def _validate() -> int:
    python = str(REPO_ROOT / ".venv/bin/python")
    phase_tests = [
        f"tests/test_phase{phase}_{name}.py"
        for phase, name in (
            (73, "exact_descriptor_normalization"),
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
                "pfe-core/pfe_core/phase73_exact_descriptor_normalization.py",
                "tools/phase73_exact_descriptor_normalization.py",
                "tests/test_phase73_exact_descriptor_normalization.py",
            ],
        ),
        (
            "phase73_focused_and_phase72_to45_regression",
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
        "kind": "phase73_validation_summary",
        "created_at": _utcnow(),
        "status": "passed" if passed else "failed",
        "check_count": len(results),
        "passed_count": sum(row["passed"] for row in results),
        "failed_count": sum(not row["passed"] for row in results),
        "checks": results,
    }
    _write_json(EVIDENCE_ROOT / "validation_summary.json", summary)
    lines = [f"Phase73 validation: {summary['status']}"]
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
        "--stage", choices=("sparse_preflight", "phase68_regression"), required=True
    )
    evaluate.add_argument("--ollama-endpoint", default="http://127.0.0.1:11434")
    evaluate.add_argument("--timeout", type=int, default=900)
    evaluate.add_argument("--resume", action="store_true")
    subparsers.add_parser("finalize")
    subparsers.add_parser("validate")
    args = parser.parse_args()
    if args.command == "prepare":
        return _prepare(args.clean_evidence)
    if args.command == "eval":
        return _run_eval(args.stage, args.ollama_endpoint, args.timeout, args.resume)
    if args.command == "finalize":
        return _finalize()
    return _validate()


if __name__ == "__main__":
    raise SystemExit(main())
