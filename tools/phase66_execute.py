#!/usr/bin/env python3
"""Run the frozen Phase66 external holdout and historical replay."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import time
from typing import Any
from urllib.error import URLError


REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = REPO_ROOT / "pfe-core"
TOOLS_ROOT = REPO_ROOT / "tools"
for root in (CORE_ROOT, TOOLS_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from pfe_core.phase46_runtime_first_latest_intent import stable_hash
from pfe_core.phase59_proposition_addressed_grounding import (
    build_phase59_proposition_candidates,
)
from pfe_core.phase63_field_typed_candidate_wire import (
    build_phase63_failure_record,
    parse_phase63_typed_wire_selection,
)
from pfe_core.phase66_external_distribution_regression import (
    evaluate_phase66_external_holdout,
    evaluate_phase66_historical_replay,
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
from phase63_execute import _invoke_typed_judge


EVIDENCE_ROOT = REPO_ROOT / "docs/demo/phase66-external-distribution-regression"
PHASE53_SOURCE = CORE_ROOT / "pfe_core/phase53_evaluator_scope_recovery.py"
PHASE56_SOURCE = CORE_ROOT / "pfe_core/phase56_evidence_span_grounded_atomic.py"
PHASE58_SOURCE = CORE_ROOT / "pfe_core/phase58_clause_addressed_grounding.py"
PHASE59_SOURCE = CORE_ROOT / "pfe_core/phase59_proposition_addressed_grounding.py"
PHASE62_SOURCE = CORE_ROOT / "pfe_core/phase62_risk_asymmetric_candidate_consensus.py"
PHASE63_SOURCE = CORE_ROOT / "pfe_core/phase63_field_typed_candidate_wire.py"
PHASE63_EXECUTOR = TOOLS_ROOT / "phase63_execute.py"
PHASE65_SOURCE = CORE_ROOT / "pfe_core/phase65_aggregate_safe_boundary_coverage.py"
PHASE66_SOURCE = CORE_ROOT / "pfe_core/phase66_external_distribution_regression.py"
PHASE66_EXECUTOR = TOOLS_ROOT / "phase66_execute.py"


def _stage_paths(stage: str) -> tuple[Path, Path, Path | None]:
    if stage == "preflight":
        directory = EVIDENCE_ROOT / "evidence-typed-wire-preflight"
        return directory, directory / "preflight_items_public.jsonl", None
    directory = EVIDENCE_ROOT / f"evidence-{stage.replace('_', '-')}"
    return directory, directory / "blind_items_public.jsonl", directory / "blind_hidden_key.json"


def _freeze_check(
    stage: str, public: list[dict[str, Any]], hidden: list[dict[str, Any]]
) -> dict[str, Any]:
    freeze = _read_json(EVIDENCE_ROOT / "pre_model_call_freeze.json")
    protocol = _read_json(EVIDENCE_ROOT / "evaluator_protocol.json")
    protocol_copy = {key: value for key, value in protocol.items() if key != "protocol_sha256"}
    source_ok = (
        freeze.get("phase53_hard_detector_source_sha256") == _sha256(PHASE53_SOURCE)
        and freeze.get("phase56_composer_source_sha256") == _sha256(PHASE56_SOURCE)
        and freeze.get("phase58_clause_grounder_source_sha256") == _sha256(PHASE58_SOURCE)
        and freeze.get("phase59_candidate_source_sha256") == _sha256(PHASE59_SOURCE)
        and freeze.get("phase62_consensus_source_sha256") == _sha256(PHASE62_SOURCE)
        and freeze.get("phase63_typed_wire_source_sha256") == _sha256(PHASE63_SOURCE)
        and freeze.get("phase63_executor_source_sha256") == _sha256(PHASE63_EXECUTOR)
        and freeze.get("phase65_source_sha256") == _sha256(PHASE65_SOURCE)
        and freeze.get("phase66_source_sha256") == _sha256(PHASE66_SOURCE)
        and freeze.get("phase66_executor_source_sha256") == _sha256(PHASE66_EXECUTOR)
    )
    protocol_ok = (
        bool(protocol.get("protocol_sha256"))
        and stable_hash(protocol_copy)
        == protocol.get("protocol_sha256")
        == freeze.get("protocol_sha256")
    )
    if stage == "preflight":
        data_ok = stable_hash(public) == freeze.get("preflight_public_sha256")
        prerequisite_ok = (
            _read_json(EVIDENCE_ROOT / "preparation_decision.json").get("status")
            == "ready_for_typed_wire_preflight"
        )
    else:
        data_ok = (
            stable_hash(public) == freeze.get(f"{stage}_public_sha256")
            and stable_hash(hidden) == freeze.get(f"{stage}_hidden_sha256")
        )
        if stage == "external_holdout":
            prerequisite_ok = (
                _read_json(
                    EVIDENCE_ROOT / "evidence-typed-wire-preflight/preflight_report.json"
                ).get("status")
                == "passed"
            )
        else:
            external = _read_json(
                EVIDENCE_ROOT / "evidence-external-holdout/candidate_evaluator_report.json"
            )
            prerequisite_ok = (
                external.get("status") == "qualified"
                and int(external.get("false_accept_count_on_reject_cases") or 0) == 0
                and int(external.get("schema_failure_count") or 0) == 0
                and int(external.get("candidate_value_conflict_count") or 0) == 0
            )
    return {
        "kind": "phase66_external_distribution_freeze_check",
        "stage": stage,
        "passed": source_ok and protocol_ok and data_ok and prerequisite_ok,
        "source_checks_passed": source_ok,
        "protocol_check_passed": protocol_ok,
        "data_check_passed": data_ok,
        "prerequisite_check_passed": prerequisite_ok,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("preflight", "external_holdout", "historical_replay"),
        required=True,
    )
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
        raise SystemExit(f"Phase66 {args.stage} freeze check failed: {freeze_check}")

    protocol = _read_json(EVIDENCE_ROOT / "evaluator_protocol.json")
    workers = int(protocol.get("parallel_worker_count") or 1)
    if args.workers is not None and args.workers != workers:
        raise SystemExit(
            f"Phase66 worker count is frozen at {workers}; received --workers={args.workers}"
        )
    aliases = tuple(str(value) for value in protocol.get("semantic_judge_aliases") or [])
    models = {
        str(key): str(value)
        for key, value in dict(protocol.get("semantic_judge_models_private") or {}).items()
    }
    tags = _ollama_tags(args.ollama_endpoint, max(10, args.timeout))
    missing = [models[alias] for alias in aliases if models.get(alias) not in tags["available"]]
    availability = {
        "kind": "phase66_judge_model_availability",
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
        raise SystemExit(f"Phase66 judge models unavailable: {missing}")

    all_results: list[dict[str, Any]] = []
    failed_items: list[dict[str, Any]] = []
    all_failure_attempts: list[dict[str, Any]] = []
    retry_limit = int(protocol.get("frozen_retry_limit_per_failed_item") or 2)
    for alias in aliases:
        output_path = directory / f"judge_typed_wire_results_{alias}.jsonl"
        failure_path = directory / f"typed_wire_failure_attempts_{alias}.jsonl"
        if (output_path.exists() or failure_path.exists()) and not args.resume:
            raise SystemExit(
                f"Phase66 {args.stage} is frozen one-shot evidence; use --resume only after interruption"
            )
        existing_results = _read_jsonl(output_path) if args.resume else []
        failure_attempts = _read_jsonl(failure_path) if args.resume else []
        by_id: dict[str, dict[str, Any]] = {}
        public_by_id = {str(row.get("item_id") or ""): row for row in public}
        for row in existing_results:
            item = public_by_id.get(str(row.get("item_id") or ""), {})
            candidates = build_phase59_proposition_candidates(
                str(item.get("assistant_response") or "")
            )
            try:
                parse_phase63_typed_wire_selection(
                    str(row.get("raw_response") or ""), candidates=candidates
                )
            except ValueError:
                continue
            if row.get("actual_model_call") is True:
                by_id[str(row.get("item_id") or "")] = dict(row)
        attempts_by_item: dict[str, int] = {}
        for row in failure_attempts:
            item_id = str(row.get("item_id") or "")
            attempts_by_item[item_id] = max(
                attempts_by_item.get(item_id, 0), int(row.get("attempt") or 0)
            )
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
                    return {
                        "index": index,
                        "item_id": item_id,
                        "result": result,
                        "failures": new_failures,
                    }
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
                    f"[{args.stage}:{alias}] {index}/{len(public)} {item_id} "
                    f"attempt={attempt} failed: {error}",
                    flush=True,
                )
                time.sleep(attempt)
            return {
                "index": index,
                "item_id": item_id,
                "result": None,
                "failures": new_failures,
            }

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
                        f"[{args.stage}:{alias}] {outcome['index']}/{len(public)} "
                        f"{item_id} wire={result['raw_response']} "
                        f"latency={result['latency_seconds']}s",
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
                        "error": rows[-1].get("error")
                        if rows
                        else "missing_result_without_attempt_record",
                    }
                )
        all_results.extend(by_id[key] for key in sorted(by_id))
        all_failure_attempts.extend(failure_attempts)

    expected = len(public) * len(aliases)
    if args.stage == "preflight":
        report = {
            "kind": "phase66_typed_wire_preflight_report",
            "stage": args.stage,
            "status": "passed"
            if len(all_results) == expected and not failed_items
            else "failed",
            "item_count": len(public),
            "expected_judge_item_count": expected,
            "successful_model_output_count": len(all_results),
            "failed_judge_item_count": len(failed_items),
            "judge_item_outcome_count": len(all_results) + len(failed_items),
            "raw_failure_attempt_count": len(all_failure_attempts),
            "raw_failures_preserved": all(
                "raw_response" in row for row in all_failure_attempts
            ),
            "failures": failed_items,
            "scored_as_holdout": False,
        }
        output_name = "preflight_report.json"
    elif args.stage == "external_holdout":
        report = evaluate_phase66_external_holdout(
            public_items=public,
            hidden_key=hidden,
            judge_results=all_results,
            judge_aliases=aliases,
        )
        output_name = "candidate_evaluator_report.json"
    else:
        report = evaluate_phase66_historical_replay(
            public_items=public,
            hidden_key=hidden,
            judge_results=all_results,
            judge_aliases=aliases,
        )
        output_name = "historical_replay_report.json"
    report.update(
        {
            "successful_model_output_count": len(all_results),
            "failure_count": len(failed_items),
            "failures": failed_items,
            "judge_item_outcome_count": len(all_results) + len(failed_items),
            "raw_failure_attempt_count": len(all_failure_attempts),
            "raw_failures_preserved": all(
                "raw_response" in row for row in all_failure_attempts
            ),
            "judge_models": models,
            "ollama_endpoint": args.ollama_endpoint,
            "parallel_worker_count": workers,
            "request_timeout_seconds": max(30, args.timeout),
            "scope_aware_candidate_rule": True,
            "field_typed_wire_only": True,
            "fabricated_scores": False,
            "created_at": _utcnow(),
        }
    )
    _write_json(directory / output_name, report)
    print(
        json.dumps(
            {
                key: report.get(key)
                for key in (
                    "stage",
                    "split",
                    "status",
                    "item_count",
                    "completed_item_count",
                    "accuracy",
                    "accuracy_delta_from_phase64",
                    "per_phase",
                    "typed_exact_match_rate",
                    "candidate_selection_exact_match_rate",
                    "raw_judge_typed_exact_match_rate",
                    "safe_abstention_recovery_count",
                    "dangerous_any_consensus_count",
                    "candidate_value_conflict_count",
                    "successful_model_output_count",
                    "failure_count",
                    "raw_failure_attempt_count",
                    "schema_failure_count",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    complete = len(all_results) == expected and not failed_items
    return 0 if complete and report.get("status") in {"passed", "qualified"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
