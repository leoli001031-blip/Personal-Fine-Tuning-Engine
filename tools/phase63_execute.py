#!/usr/bin/env python3
"""Run frozen Phase63 field-typed wire calls and Phase62 consensus evaluation."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = REPO_ROOT / "pfe-core"
TOOLS_ROOT = REPO_ROOT / "tools"
for root in (CORE_ROOT, TOOLS_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from pfe_core.phase46_runtime_first_latest_intent import stable_hash
from pfe_core.phase59_proposition_addressed_grounding import build_phase59_proposition_candidates
from pfe_core.phase63_field_typed_candidate_wire import (
    PHASE63_WIRE_PATTERN,
    build_phase63_failure_record,
    build_phase63_typed_wire_prompt,
    evaluate_phase63_candidate_consensus,
    parse_phase63_typed_wire_selection,
)
from phase62_execute import (
    JudgeAttemptError,
    _ollama_tags,
    _read_json,
    _read_jsonl,
    _sha256,
    _utcnow,
    _write_json,
    _write_jsonl,
)


EVIDENCE_ROOT = REPO_ROOT / "docs/demo/phase63-field-typed-candidate-wire"
PHASE53_SOURCE = CORE_ROOT / "pfe_core/phase53_evaluator_scope_recovery.py"
PHASE56_SOURCE = CORE_ROOT / "pfe_core/phase56_evidence_span_grounded_atomic.py"
PHASE58_SOURCE = CORE_ROOT / "pfe_core/phase58_clause_addressed_grounding.py"
PHASE59_SOURCE = CORE_ROOT / "pfe_core/phase59_proposition_addressed_grounding.py"
PHASE60_SOURCE = CORE_ROOT / "pfe_core/phase60_flat_schema_compatibility.py"
PHASE61_SOURCE = CORE_ROOT / "pfe_core/phase61_compact_candidate_wire_protocol.py"
PHASE62_SOURCE = CORE_ROOT / "pfe_core/phase62_risk_asymmetric_candidate_consensus.py"
PHASE63_SOURCE = CORE_ROOT / "pfe_core/phase63_field_typed_candidate_wire.py"
PHASE63_EXECUTOR = TOOLS_ROOT / "phase63_execute.py"


def _invoke_typed_judge(
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
    prompt = build_phase63_typed_wire_prompt(item)
    num_ctx = int(protocol.get("num_ctx") or 4096)
    num_predict = int(protocol.get("num_predict") or 64)
    payload = {
        "model": model,
        "stream": False,
        "think": False,
        "keep_alive": "30m",
        "messages": [{"role": "user", "content": prompt}],
        "options": {"temperature": 0, "num_ctx": num_ctx, "num_predict": num_predict},
    }
    request = Request(
        endpoint.rstrip("/") + "/api/chat",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urlopen(request, timeout=timeout) as response_handle:
            raw_http = response_handle.read().decode("utf-8", errors="replace")
    except HTTPError as exc:
        raw_http = exc.read().decode("utf-8", errors="replace")
        raise JudgeAttemptError(
            f"HTTPError {exc.code}: {raw_http}", raw_response=raw_http, failure_class="http_error"
        ) from exc
    try:
        body = json.loads(raw_http)
    except json.JSONDecodeError as exc:
        raise JudgeAttemptError(
            f"invalid Ollama response JSON: {exc}",
            raw_response=raw_http,
            failure_class="http_response_json_error",
        ) from exc
    content = str(dict(body.get("message") or {}).get("content") or "")
    try:
        selection = parse_phase63_typed_wire_selection(content, candidates=candidates)
    except ValueError as exc:
        raise JudgeAttemptError(
            f"ValueError: {exc}",
            raw_response=content or raw_http,
            failure_class="field_typed_wire_validation_error",
        ) from exc
    match = PHASE63_WIRE_PATTERN.fullmatch(content)
    if match is None:
        raise AssertionError("validated Phase63 wire must match the frozen pattern")
    typed_ids = {
        f"{field}_typed_candidate_id": typed_id
        for field, typed_id in zip(
            ("source_registration", "user_outcome_status", "test_to_user_outcome_relation"),
            match.groups(),
            strict=True,
        )
    }
    eval_seconds = float(body.get("eval_duration") or 0) / 1_000_000_000
    eval_count = int(body.get("eval_count") or 0)
    return {
        "item_id": item.get("item_id"),
        "stage": stage,
        "judge_alias": alias,
        "judge_model": model,
        **selection,
        **typed_ids,
        "actual_model_call": True,
        "identity_hidden_from_judge": True,
        "gold_label_hidden_from_judge": True,
        "gold_typed_fields_hidden_from_judge": True,
        "gold_candidate_ids_hidden_from_judge": True,
        "other_judge_identity_hidden": True,
        "judge_returned_direct_label": False,
        "field_typed_wire_valid": True,
        "think": False,
        "temperature": 0,
        "num_ctx": num_ctx,
        "num_predict": num_predict,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "typed_wire_spec_sha256": stable_hash(dict(protocol.get("typed_wire_spec") or {})),
        "raw_response": content,
        "raw_http_response_sha256": hashlib.sha256(raw_http.encode("utf-8")).hexdigest(),
        "done_reason": body.get("done_reason"),
        "total_duration_seconds": round(float(body.get("total_duration") or 0) / 1_000_000_000, 4),
        "eval_count": eval_count,
        "eval_duration_seconds": round(eval_seconds, 4),
        "eval_tokens_per_second": round(eval_count / eval_seconds, 4) if eval_seconds else None,
        "latency_seconds": round(time.perf_counter() - started, 4),
        "created_at": _utcnow(),
    }


def _stage_paths(stage: str) -> tuple[Path, Path, Path | None]:
    if stage == "preflight":
        directory = EVIDENCE_ROOT / "evidence-typed-wire-preflight"
        return directory, directory / "preflight_items_public.jsonl", None
    directory = EVIDENCE_ROOT / f"evidence-evaluator-{stage}"
    return directory, directory / "blind_items_public.jsonl", directory / "blind_hidden_key.json"


def _freeze_check(stage: str, public: list[dict[str, Any]], hidden: list[dict[str, Any]]) -> dict[str, Any]:
    freeze = _read_json(EVIDENCE_ROOT / "pre_model_call_freeze.json")
    protocol = _read_json(EVIDENCE_ROOT / "evaluator_protocol.json")
    protocol_copy = {key: value for key, value in protocol.items() if key != "protocol_sha256"}
    source_ok = (
        freeze.get("phase53_hard_detector_source_sha256") == _sha256(PHASE53_SOURCE)
        and freeze.get("phase56_composer_source_sha256") == _sha256(PHASE56_SOURCE)
        and freeze.get("phase58_clause_grounder_source_sha256") == _sha256(PHASE58_SOURCE)
        and freeze.get("phase59_candidate_source_sha256") == _sha256(PHASE59_SOURCE)
        and freeze.get("phase60_flat_protocol_source_sha256") == _sha256(PHASE60_SOURCE)
        and freeze.get("phase61_wire_source_sha256") == _sha256(PHASE61_SOURCE)
        and freeze.get("phase62_consensus_source_sha256") == _sha256(PHASE62_SOURCE)
        and freeze.get("phase63_typed_wire_source_sha256") == _sha256(PHASE63_SOURCE)
        and freeze.get("phase63_executor_source_sha256") == _sha256(PHASE63_EXECUTOR)
    )
    protocol_ok = (
        bool(protocol.get("protocol_sha256"))
        and stable_hash(protocol_copy) == protocol.get("protocol_sha256") == freeze.get("protocol_sha256")
    )
    if stage == "preflight":
        data_ok = stable_hash(public) == freeze.get("preflight_public_sha256")
        prerequisite_ok = _read_json(EVIDENCE_ROOT / "preparation_decision.json").get("status") == "ready_for_typed_wire_preflight"
    else:
        data_ok = (
            stable_hash(public) == freeze.get(f"{stage}_public_sha256")
            and stable_hash(hidden) == freeze.get(f"{stage}_hidden_sha256")
        )
        if stage == "calibration":
            prerequisite_ok = _read_json(
                EVIDENCE_ROOT / "evidence-typed-wire-preflight/preflight_report.json"
            ).get("status") == "passed"
        else:
            prerequisite_ok = _read_json(
                EVIDENCE_ROOT / "evidence-evaluator-calibration/candidate_evaluator_report.json"
            ).get("status") == "qualified"
    return {
        "kind": "phase63_typed_wire_freeze_check",
        "stage": stage,
        "passed": source_ok and protocol_ok and data_ok and prerequisite_ok,
        "source_checks_passed": source_ok,
        "protocol_check_passed": protocol_ok,
        "data_check_passed": data_ok,
        "prerequisite_check_passed": prerequisite_ok,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("preflight", "calibration", "holdout"), required=True)
    parser.add_argument("--ollama-endpoint", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    directory, public_path, hidden_path = _stage_paths(args.stage)
    public = _read_jsonl(public_path)
    hidden = list(_read_json(hidden_path).get("items") or []) if hidden_path else []
    freeze_check = _freeze_check(args.stage, public, hidden)
    _write_json(directory / "freeze_check.json", freeze_check)
    if freeze_check["passed"] is not True:
        raise SystemExit(f"Phase63 {args.stage} freeze check failed: {freeze_check}")

    protocol = _read_json(EVIDENCE_ROOT / "evaluator_protocol.json")
    workers = int(protocol.get("parallel_worker_count") or 1)
    if args.workers is not None and args.workers != workers:
        raise SystemExit(f"Phase63 worker count is frozen at {workers}; received --workers={args.workers}")
    aliases = tuple(str(value) for value in protocol.get("semantic_judge_aliases") or [])
    models = {str(key): str(value) for key, value in dict(protocol.get("semantic_judge_models_private") or {}).items()}
    tags = _ollama_tags(args.ollama_endpoint, max(10, args.timeout))
    missing = [models[alias] for alias in aliases if models.get(alias) not in tags["available"]]
    availability = {
        "kind": "phase63_judge_model_availability",
        "stage": args.stage,
        "ollama_endpoint": args.ollama_endpoint,
        "judge_aliases": list(aliases),
        "judge_models": models,
        "ollama_models_available": tags["available"],
        "missing_models": missing,
        "parallel_worker_count": workers,
        "passed": not missing and len(aliases) == 2,
        "queried_at": tags["queried_at"],
    }
    _write_json(directory / "model_availability.json", availability)
    if availability["passed"] is not True:
        raise SystemExit(f"Phase63 judge models unavailable: {missing}")

    all_results: list[dict[str, Any]] = []
    failed_items: list[dict[str, Any]] = []
    all_failure_attempts: list[dict[str, Any]] = []
    retry_limit = int(protocol.get("frozen_retry_limit_per_failed_item") or 2)
    for alias in aliases:
        output_path = directory / f"judge_typed_wire_results_{alias}.jsonl"
        failure_path = directory / f"typed_wire_failure_attempts_{alias}.jsonl"
        if (output_path.exists() or failure_path.exists()) and not args.resume:
            raise SystemExit(f"Phase63 {args.stage} is frozen one-shot evidence; use --resume only after interruption")
        existing_results = _read_jsonl(output_path) if args.resume else []
        failure_attempts = _read_jsonl(failure_path) if args.resume else []
        by_id: dict[str, dict[str, Any]] = {}
        for row in existing_results:
            item = next((entry for entry in public if entry.get("item_id") == row.get("item_id")), {})
            candidates = build_phase59_proposition_candidates(str(item.get("assistant_response") or ""))
            try:
                parse_phase63_typed_wire_selection(str(row.get("raw_response") or ""), candidates=candidates)
            except ValueError:
                continue
            if row.get("actual_model_call") is True:
                by_id[str(row.get("item_id") or "")] = dict(row)
        attempts_by_item: dict[str, int] = {}
        for row in failure_attempts:
            item_id = str(row.get("item_id") or "")
            attempts_by_item[item_id] = max(attempts_by_item.get(item_id, 0), int(row.get("attempt") or 0))
        pending = [
            (index, dict(item))
            for index, item in enumerate(public, start=1)
            if str(item.get("item_id") or "") not in by_id
            and attempts_by_item.get(str(item.get("item_id") or ""), 0) < retry_limit
        ]

        def run_pending(entry: tuple[int, dict[str, Any]]) -> dict[str, Any]:
            index, item = entry
            item_id = str(item.get("item_id") or "")
            new_failures = []
            start_attempt = attempts_by_item.get(item_id, 0) + 1
            for attempt in range(start_attempt, retry_limit + 1):
                try:
                    result = _invoke_typed_judge(
                        item=item,
                        alias=alias,
                        model=models[alias],
                        endpoint=args.ollama_endpoint,
                        timeout=max(30, args.timeout),
                        protocol=protocol,
                        stage=args.stage,
                    )
                    result["parallel_worker_count"] = workers
                    return {"index": index, "item_id": item_id, "result": result, "failures": new_failures}
                except JudgeAttemptError as exc:
                    raw_response = exc.raw_response
                    failure_class = exc.failure_class
                    error = str(exc)
                except (OSError, URLError, ValueError, json.JSONDecodeError) as exc:
                    raw_response = ""
                    failure_class = "transport_or_unexpected_error"
                    error = f"{exc.__class__.__name__}: {exc}"
                record = build_phase63_failure_record(
                    item_id=item_id,
                    judge_alias=alias,
                    attempt=attempt,
                    raw_response=raw_response,
                    error=error,
                )
                record.update(
                    {
                        "stage": args.stage,
                        "judge_model": models[alias],
                        "failure_class": failure_class,
                        "created_at": _utcnow(),
                    }
                )
                new_failures.append(record)
                print(
                    f"[{args.stage}:{alias}] {index}/{len(public)} {item_id} attempt={attempt} failed: {error}",
                    flush=True,
                )
                time.sleep(attempt)
            return {"index": index, "item_id": item_id, "result": None, "failures": new_failures}

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(run_pending, entry) for entry in pending]
            for future in as_completed(futures):
                outcome = future.result()
                item_id = str(outcome["item_id"])
                failure_attempts.extend(outcome["failures"])
                _write_jsonl(failure_path, failure_attempts)
                result = outcome["result"]
                if result is not None:
                    by_id[item_id] = result
                    _write_jsonl(output_path, [by_id[key] for key in sorted(by_id)])
                    print(
                        f"[{args.stage}:{alias}] {outcome['index']}/{len(public)} {item_id} "
                        f"wire={result['raw_response']} latency={result['latency_seconds']}s",
                        flush=True,
                    )
        for item in public:
            item_id = str(item.get("item_id") or "")
            if item_id not in by_id:
                rows = [row for row in failure_attempts if row.get("item_id") == item_id]
                failed_items.append(
                    {
                        "item_id": item_id,
                        "judge_alias": alias,
                        "attempt_count": len(rows),
                        "error": rows[-1].get("error") if rows else "missing_result_without_attempt_record",
                    }
                )
        all_results.extend(by_id[key] for key in sorted(by_id))
        all_failure_attempts.extend(failure_attempts)

    expected = len(public) * len(aliases)
    if args.stage == "preflight":
        report = {
            "kind": "phase63_typed_wire_preflight_report",
            "stage": args.stage,
            "status": "passed" if len(all_results) == expected and not failed_items else "failed",
            "item_count": len(public),
            "judge_alias_count": len(aliases),
            "expected_judge_item_count": expected,
            "successful_model_output_count": len(all_results),
            "failed_judge_item_count": len(failed_items),
            "judge_item_outcome_count": len(all_results) + len(failed_items),
            "raw_failure_attempt_count": len(all_failure_attempts),
            "raw_failures_preserved": all("raw_response" in row for row in all_failure_attempts),
            "failures": failed_items,
            "judge_models": models,
            "ollama_endpoint": args.ollama_endpoint,
            "parallel_worker_count": workers,
            "request_timeout_seconds": max(30, args.timeout),
            "field_typed_wire_only": True,
            "scored_as_calibration": False,
            "fabricated_scores": False,
            "created_at": _utcnow(),
        }
        _write_json(directory / "preflight_report.json", report)
    else:
        report = evaluate_phase63_candidate_consensus(
            split=args.stage,
            public_items=public,
            hidden_key=hidden,
            judge_results=all_results,
            judge_aliases=aliases,
        )
        report.update(
            {
                "kind": "phase63_field_typed_wire_consensus_report",
                "successful_model_output_count": len(all_results),
                "failure_count": len(failed_items),
                "failures": failed_items,
                "judge_item_outcome_count": len(all_results) + len(failed_items),
                "raw_failure_attempt_count": len(all_failure_attempts),
                "raw_failures_preserved": all("raw_response" in row for row in all_failure_attempts),
                "judge_models": models,
                "ollama_endpoint": args.ollama_endpoint,
                "parallel_worker_count": workers,
                "request_timeout_seconds": max(30, args.timeout),
                "field_typed_wire_only": True,
                "fabricated_scores": False,
                "created_at": _utcnow(),
            }
        )
        _write_json(directory / "candidate_evaluator_report.json", report)
    print(
        json.dumps(
            {key: report.get(key) for key in (
                "stage", "split", "status", "item_count", "completed_item_count", "accuracy",
                "typed_exact_match_rate", "candidate_selection_exact_match_rate",
                "raw_judge_typed_exact_match_rate", "safe_abstention_recovery_count",
                "dangerous_any_consensus_count", "candidate_value_conflict_count",
                "successful_model_output_count", "failed_judge_item_count", "failure_count",
                "raw_failure_attempt_count", "schema_failure_count",
            )},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if len(all_results) == expected and not failed_items and report.get("status") in {"passed", "qualified"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
