#!/usr/bin/env python3
"""Run the frozen qualified dual-judge evaluation for Phase69 outputs."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
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

from pfe_core.phase59_proposition_addressed_grounding import build_phase59_proposition_candidates
from pfe_core.phase63_field_typed_candidate_wire import (
    build_phase63_failure_record,
    parse_phase63_typed_wire_selection,
)
from pfe_core.phase69_minimal_runtime_ab import evaluate_phase69_boundary_results, stable_hash
from phase62_execute import JudgeAttemptError, _ollama_tags, _utcnow
from phase63_execute import _invoke_typed_judge


EVIDENCE_ROOT = REPO_ROOT / "docs/demo/phase69-minimal-runtime-ab"
EVAL_DIR = EVIDENCE_ROOT / "evidence-qualified-evaluator"


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


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _freeze_check(public: list[dict[str, Any]], hidden: list[dict[str, Any]]) -> dict[str, Any]:
    freeze = _read_json(EVAL_DIR / "pre_judge_freeze.json")
    protocol = _read_json(EVIDENCE_ROOT / "runtime_ab_protocol.json")
    source_paths = {
        "phase69_core": REPO_ROOT / "pfe-core/pfe_core/phase69_minimal_runtime_ab.py",
        "phase69_prepare_eval": REPO_ROOT / "tools/phase69_prepare_eval.py",
        "phase69_execute_eval": Path(__file__).resolve(),
        "phase69_finalize": REPO_ROOT / "tools/phase69_finalize_evidence.py",
    }
    source_ok = all(
        freeze.get("source_sha256", {}).get(name) == _sha256(path)
        for name, path in source_paths.items()
    )
    return {
        "kind": "phase69_qualified_evaluator_freeze_check",
        "passed": (
            source_ok
            and stable_hash(public) == freeze.get("public_items_sha256")
            and stable_hash(hidden) == freeze.get("hidden_key_sha256")
            and protocol.get("protocol_sha256") == freeze.get("protocol_sha256")
            and _read_json(EVAL_DIR / "eval_preparation_decision.json").get("status")
            == "ready_for_qualified_evaluator"
        ),
        "source_checks_passed": source_ok,
        "public_items_check_passed": stable_hash(public) == freeze.get("public_items_sha256"),
        "hidden_key_check_passed": stable_hash(hidden) == freeze.get("hidden_key_sha256"),
        "protocol_check_passed": protocol.get("protocol_sha256") == freeze.get("protocol_sha256"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ollama-endpoint", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    public = _read_jsonl(EVAL_DIR / "blind_items_public.jsonl")
    hidden = list(_read_json(EVAL_DIR / "blind_hidden_key.json").get("items") or [])
    freeze = _freeze_check(public, hidden)
    _write_json(EVAL_DIR / "freeze_check.json", freeze)
    if freeze["passed"] is not True:
        raise SystemExit(f"Phase69 judge freeze failed: {freeze}")

    protocol = _read_json(EVIDENCE_ROOT / "runtime_ab_protocol.json")
    workers = int(protocol.get("parallel_worker_count") or 1)
    if args.workers is not None and args.workers != workers:
        raise SystemExit(f"Phase69 worker count is frozen at {workers}")
    aliases = tuple(str(value) for value in protocol.get("semantic_judge_aliases") or [])
    models = {
        str(key): str(value)
        for key, value in dict(protocol.get("semantic_judge_models_private") or {}).items()
    }
    tags = _ollama_tags(args.ollama_endpoint, max(10, args.timeout))
    missing = [models[alias] for alias in aliases if models.get(alias) not in tags["available"]]
    availability = {
        "kind": "phase69_judge_model_availability",
        "ollama_endpoint": args.ollama_endpoint,
        "judge_aliases": list(aliases),
        "judge_models": models,
        "ollama_models_available": tags["available"],
        "missing_models": missing,
        "parallel_worker_count": workers,
        "passed": not missing and len(aliases) == 2,
        "queried_at": tags["queried_at"],
    }
    _write_json(EVAL_DIR / "model_availability.json", availability)
    if availability["passed"] is not True:
        raise SystemExit(f"Phase69 judge models unavailable: {missing}")

    all_results = []
    all_failures = []
    exhausted = []
    retry_limit = int(protocol.get("frozen_retry_limit_per_failed_item") or 2)
    for alias in aliases:
        result_path = EVAL_DIR / f"judge_typed_wire_results_{alias}.jsonl"
        failure_path = EVAL_DIR / f"typed_wire_failure_attempts_{alias}.jsonl"
        if (result_path.exists() or failure_path.exists()) and not args.resume:
            raise SystemExit("Phase69 evaluation is one-shot evidence; use --resume after interruption")
        existing = _read_jsonl(result_path) if args.resume else []
        failures = _read_jsonl(failure_path) if args.resume else []
        by_id: dict[str, dict[str, Any]] = {}
        for row in existing:
            item = next((entry for entry in public if entry.get("item_id") == row.get("item_id")), {})
            candidates = build_phase59_proposition_candidates(str(item.get("assistant_response") or ""))
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

        def run_item(entry: tuple[int, dict[str, Any]]) -> dict[str, Any]:
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
                        stage="phase69_runtime_ab",
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
                        "stage": "phase69_runtime_ab",
                        "judge_model": models[alias],
                        "failure_class": failure_class,
                        "created_at": _utcnow(),
                    }
                )
                new_failures.append(record)
                time.sleep(attempt)
            return {"index": index, "item_id": item_id, "result": None, "failures": new_failures}

        completed_since_start = 0
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(run_item, entry) for entry in pending]
            for future in as_completed(futures):
                outcome = future.result()
                failures.extend(outcome["failures"])
                _write_jsonl(failure_path, failures)
                if outcome["result"] is not None:
                    by_id[str(outcome["item_id"])] = outcome["result"]
                    _write_jsonl(result_path, [by_id[key] for key in sorted(by_id)])
                    completed_since_start += 1
                    if completed_since_start % 12 == 0 or completed_since_start == len(pending):
                        print(
                            f"[phase69:{alias}] completed {len(by_id)}/{len(public)}",
                            flush=True,
                        )
        for item in public:
            item_id = str(item.get("item_id") or "")
            if item_id not in by_id:
                rows = [row for row in failures if row.get("item_id") == item_id]
                exhausted.append(
                    {
                        "item_id": item_id,
                        "judge_alias": alias,
                        "attempt_count": len(rows),
                        "error": rows[-1].get("error") if rows else "missing_result",
                    }
                )
        all_results.extend(by_id[key] for key in sorted(by_id))
        all_failures.extend(failures)

    report = evaluate_phase69_boundary_results(
        public_items=public,
        hidden_key=hidden,
        judge_results=all_results,
        judge_aliases=aliases,
    )
    report.update(
        {
            "status": "completed" if not exhausted and len(all_results) == len(public) * len(aliases) else "incomplete",
            "successful_model_output_count": len(all_results),
            "expected_model_output_count": len(public) * len(aliases),
            "failure_count": len(exhausted),
            "failures": exhausted,
            "raw_failure_attempt_count": len(all_failures),
            "raw_failures_preserved": all("raw_response" in row for row in all_failures),
            "fabricated_scores": False,
            "created_at": _utcnow(),
        }
    )
    _write_json(EVAL_DIR / "boundary_evaluator_report.json", report)
    summary = {
        "status": report["status"],
        "actual_judge_outputs": len(all_results),
        "schema_failures": report["schema_failure_count"],
        "candidate_conflicts": report["candidate_value_conflict_count"],
        "baseline_accept_rate": report["variants"]["baseline_runtime"]["accept_rate"],
        "candidate_accept_rate": report["variants"]["candidate_boundary_contract"]["accept_rate"],
        "delta": report["candidate_accept_rate_delta"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
