#!/usr/bin/env python3
"""Execute frozen Phase68 dual-judge stages."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping
from urllib.error import URLError


REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = REPO_ROOT / "pfe-core"
TOOLS_ROOT = REPO_ROOT / "tools"
for root in (CORE_ROOT, TOOLS_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from pfe_core.phase46_runtime_first_latest_intent import stable_hash
from pfe_core.phase59_proposition_addressed_grounding import build_phase59_proposition_candidates
from pfe_core.phase63_field_typed_candidate_wire import (
    build_phase63_failure_record,
    parse_phase63_typed_wire_selection,
)
from pfe_core.phase68_aligned_candidate_scope_recovery import (
    evaluate_phase68_candidate_consensus,
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


EVIDENCE_ROOT = REPO_ROOT / "docs/demo/phase68-aligned-candidate-scope-recovery"
SOURCE_FILES = {
    "phase53_hard_detector": CORE_ROOT / "pfe_core/phase53_evaluator_scope_recovery.py",
    "phase56_grounder_composer": CORE_ROOT / "pfe_core/phase56_evidence_span_grounded_atomic.py",
    "phase58_clause_grounder": CORE_ROOT / "pfe_core/phase58_clause_addressed_grounding.py",
    "phase59_candidates": CORE_ROOT / "pfe_core/phase59_proposition_addressed_grounding.py",
    "phase62_consensus": CORE_ROOT / "pfe_core/phase62_risk_asymmetric_candidate_consensus.py",
    "phase63_wire": CORE_ROOT / "pfe_core/phase63_field_typed_candidate_wire.py",
    "phase68_core": CORE_ROOT / "pfe_core/phase68_aligned_candidate_scope_recovery.py",
    "phase68_executor": Path(__file__).resolve(),
}
STAGES = ("preflight", "calibration", "holdout", "phase55_regression")


def _stage_paths(stage: str) -> tuple[Path, Path, Path | None]:
    if stage == "preflight":
        directory = EVIDENCE_ROOT / "evidence-typed-wire-preflight"
        return directory, directory / "preflight_items_public.jsonl", None
    directory = EVIDENCE_ROOT / (
        "evidence-aligned-phase55-regression"
        if stage == "phase55_regression"
        else f"evidence-evaluator-{stage}"
    )
    return directory, directory / "blind_items_public.jsonl", directory / "blind_hidden_key.json"


def _prerequisite_passed(stage: str) -> bool:
    if stage == "preflight":
        return _read_json(EVIDENCE_ROOT / "preparation_decision.json").get("status") == (
            "ready_for_typed_wire_preflight"
        )
    if stage == "calibration":
        return _read_json(
            EVIDENCE_ROOT / "evidence-typed-wire-preflight/preflight_report.json"
        ).get("status") == "passed"
    if stage == "holdout":
        return _read_json(
            EVIDENCE_ROOT / "evidence-evaluator-calibration/candidate_evaluator_report.json"
        ).get("status") == "qualified"
    return _read_json(
        EVIDENCE_ROOT / "evidence-evaluator-holdout/candidate_evaluator_report.json"
    ).get("status") == "qualified"


def _freeze_check(
    stage: str, public: list[dict[str, Any]], hidden: list[dict[str, Any]]
) -> dict[str, Any]:
    freeze = _read_json(EVIDENCE_ROOT / "pre_model_call_freeze.json")
    protocol = _read_json(EVIDENCE_ROOT / "evaluator_protocol.json")
    protocol_copy = {key: value for key, value in protocol.items() if key != "protocol_sha256"}
    source_hashes = dict(freeze.get("source_sha256") or {})
    source_ok = all(source_hashes.get(name) == _sha256(path) for name, path in SOURCE_FILES.items())
    protocol_ok = (
        stable_hash(protocol_copy)
        == protocol.get("protocol_sha256")
        == freeze.get("protocol_sha256")
    )
    if stage == "preflight":
        data_ok = stable_hash(public) == freeze.get("preflight_public_sha256")
    else:
        data_ok = (
            stable_hash(public) == freeze.get(f"{stage}_public_sha256")
            and stable_hash(hidden) == freeze.get(f"{stage}_hidden_sha256")
        )
    return {
        "kind": "phase68_freeze_check",
        "stage": stage,
        "passed": source_ok and protocol_ok and data_ok and _prerequisite_passed(stage),
        "source_checks_passed": source_ok,
        "protocol_check_passed": protocol_ok,
        "data_check_passed": data_ok,
        "prerequisite_check_passed": _prerequisite_passed(stage),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=STAGES, required=True)
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
        raise SystemExit(f"Phase68 {args.stage} freeze check failed: {freeze_check}")

    protocol = _read_json(EVIDENCE_ROOT / "evaluator_protocol.json")
    workers = int(protocol.get("parallel_worker_count") or 1)
    if args.workers is not None and args.workers != workers:
        raise SystemExit(f"Phase68 worker count is frozen at {workers}")
    aliases = tuple(str(value) for value in protocol.get("semantic_judge_aliases") or [])
    models = {
        str(key): str(value)
        for key, value in dict(protocol.get("semantic_judge_models_private") or {}).items()
    }
    tags = _ollama_tags(args.ollama_endpoint, max(10, args.timeout))
    missing = [models[alias] for alias in aliases if models.get(alias) not in tags["available"]]
    availability = {
        "kind": "phase68_judge_model_availability",
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
        raise SystemExit(f"Phase68 judge models unavailable: {missing}")

    all_results: list[dict[str, Any]] = []
    failed_items: list[dict[str, Any]] = []
    all_failure_attempts: list[dict[str, Any]] = []
    retry_limit = int(protocol.get("frozen_retry_limit_per_failed_item") or 2)
    for alias in aliases:
        output_path = directory / f"judge_typed_wire_results_{alias}.jsonl"
        failure_path = directory / f"typed_wire_failure_attempts_{alias}.jsonl"
        if (output_path.exists() or failure_path.exists()) and not args.resume:
            raise SystemExit(
                f"Phase68 {args.stage} is one-shot evidence; use --resume only after interruption"
            )
        existing = _read_jsonl(output_path) if args.resume else []
        failures = _read_jsonl(failure_path) if args.resume else []
        by_id: dict[str, dict[str, Any]] = {}
        for row in existing:
            item = next(
                (entry for entry in public if entry.get("item_id") == row.get("item_id")), {}
            )
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
        for row in failures:
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
            for attempt in range(attempts_by_item.get(item_id, 0) + 1, retry_limit + 1):
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
                    f"[{args.stage}:{alias}] {index}/{len(public)} {item_id} "
                    f"attempt={attempt} failed: {error}",
                    flush=True,
                )
                time.sleep(attempt)
            return {"index": index, "item_id": item_id, "result": None, "failures": new_failures}

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(run_pending, entry) for entry in pending]
            for future in as_completed(futures):
                outcome = future.result()
                item_id = str(outcome["item_id"])
                failures.extend(outcome["failures"])
                _write_jsonl(failure_path, failures)
                if outcome["result"] is not None:
                    by_id[item_id] = outcome["result"]
                    _write_jsonl(output_path, [by_id[key] for key in sorted(by_id)])
                    print(
                        f"[{args.stage}:{alias}] {outcome['index']}/{len(public)} "
                        f"{item_id} wire={outcome['result']['raw_response']}",
                        flush=True,
                    )
        for item in public:
            item_id = str(item.get("item_id") or "")
            if item_id not in by_id:
                rows = [row for row in failures if row.get("item_id") == item_id]
                failed_items.append(
                    {
                        "item_id": item_id,
                        "judge_alias": alias,
                        "attempt_count": len(rows),
                        "error": rows[-1].get("error") if rows else "missing_result",
                    }
                )
        all_results.extend(by_id[key] for key in sorted(by_id))
        all_failure_attempts.extend(failures)

    expected = len(public) * len(aliases)
    if args.stage == "preflight":
        report = {
            "kind": "phase68_typed_wire_preflight_report",
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
            "fabricated_scores": False,
            "created_at": _utcnow(),
        }
        _write_json(directory / "preflight_report.json", report)
    else:
        categories = (
            protocol.get("aligned_phase55_categories")
            if args.stage == "phase55_regression"
            else protocol.get("fresh_categories")
        )
        report = evaluate_phase68_candidate_consensus(
            split="calibration" if args.stage == "calibration" else "holdout",
            public_items=public,
            hidden_key=hidden,
            judge_results=all_results,
            judge_aliases=aliases,
            categories=tuple(str(value) for value in categories or []),
        )
        report.update(
            {
                "stage": args.stage,
                "successful_model_output_count": len(all_results),
                "failure_count": len(failed_items),
                "failures": failed_items,
                "judge_item_outcome_count": len(all_results) + len(failed_items),
                "raw_failure_attempt_count": len(all_failure_attempts),
                "raw_failures_preserved": all(
                    "raw_response" in row for row in all_failure_attempts
                ),
                "aligned_label_contract_only": args.stage == "phase55_regression",
                "typed_metrics_diagnostic_only": args.stage == "phase55_regression",
                "fabricated_scores": False,
                "created_at": _utcnow(),
            }
        )
        filename = (
            "aligned_regression_report.json"
            if args.stage == "phase55_regression"
            else "candidate_evaluator_report.json"
        )
        _write_json(directory / filename, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not failed_items else 1


if __name__ == "__main__":
    raise SystemExit(main())
