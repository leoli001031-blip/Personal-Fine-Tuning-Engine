#!/usr/bin/env python3
"""Run frozen Phase59 proposition-candidate selection with two local judges."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Iterable, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = REPO_ROOT / "pfe-core"
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

from pfe_core.phase46_runtime_first_latest_intent import stable_hash
from pfe_core.phase59_proposition_addressed_grounding import (
    build_phase59_candidate_judge_prompt,
    build_phase59_proposition_candidates,
    evaluate_phase59_candidate_evaluator,
    phase59_ollama_json_schema,
    validate_phase59_raw_selection,
)


EVIDENCE_ROOT = REPO_ROOT / "docs/demo/phase59-proposition-addressed-grounding"
PHASE53_SOURCE = CORE_ROOT / "pfe_core/phase53_evaluator_scope_recovery.py"
PHASE56_SOURCE = CORE_ROOT / "pfe_core/phase56_evidence_span_grounded_atomic.py"
PHASE58_SOURCE = CORE_ROOT / "pfe_core/phase58_clause_addressed_grounding.py"
PHASE59_SOURCE = CORE_ROOT / "pfe_core/phase59_proposition_addressed_grounding.py"


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


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_json_object(value: Any) -> dict[str, Any]:
    text = str(value or "").strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return {}
        parsed = json.loads(text[start:end + 1])
    return dict(parsed) if isinstance(parsed, Mapping) else {}


def _ollama_tags(endpoint: str, timeout: int) -> dict[str, Any]:
    request = Request(endpoint.rstrip("/") + "/api/tags", method="GET")
    with urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    return {
        "available": sorted(str(row.get("name") or "") for row in body.get("models") or []),
        "queried_at": _utcnow(),
    }


def _judge_one(
    *, item: Mapping[str, Any], alias: str, model: str, endpoint: str, timeout: int, protocol: Mapping[str, Any]
) -> dict[str, Any]:
    response = str(item.get("assistant_response") or "")
    candidates = build_phase59_proposition_candidates(response)
    prompt = build_phase59_candidate_judge_prompt(item)
    schema = phase59_ollama_json_schema(candidates)
    num_ctx = int(protocol.get("num_ctx") or 4096)
    num_predict = int(protocol.get("num_predict") or 192)
    payload = {
        "model": model,
        "stream": False,
        "think": False,
        "format": schema,
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
    with urlopen(request, timeout=timeout) as response_handle:
        body = json.loads(response_handle.read().decode("utf-8"))
    content = str(dict(body.get("message") or {}).get("content") or "")
    parsed = _parse_json_object(content)
    selection = validate_phase59_raw_selection(parsed, candidates=candidates)
    eval_seconds = float(body.get("eval_duration") or 0) / 1_000_000_000
    eval_count = int(body.get("eval_count") or 0)
    return {
        "item_id": item.get("item_id"),
        "judge_alias": alias,
        "judge_model": model,
        **selection,
        "actual_model_call": True,
        "identity_hidden_from_judge": True,
        "gold_label_hidden_from_judge": True,
        "gold_typed_fields_hidden_from_judge": True,
        "gold_candidate_ids_hidden_from_judge": True,
        "other_judge_identity_hidden": True,
        "judge_returned_direct_label": False,
        "think": False,
        "temperature": 0,
        "num_ctx": num_ctx,
        "num_predict": num_predict,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "schema_sha256": stable_hash(schema),
        "raw_response": content,
        "done_reason": body.get("done_reason"),
        "total_duration_seconds": round(float(body.get("total_duration") or 0) / 1_000_000_000, 4),
        "eval_count": eval_count,
        "eval_duration_seconds": round(eval_seconds, 4),
        "eval_tokens_per_second": round(eval_count / eval_seconds, 4) if eval_seconds else None,
        "latency_seconds": round(time.perf_counter() - started, 4),
        "created_at": _utcnow(),
    }


def _fixture_paths(split: str) -> tuple[Path, Path, Path]:
    directory = EVIDENCE_ROOT / f"evidence-evaluator-{split}"
    return directory, directory / "blind_items_public.jsonl", directory / "blind_hidden_key.json"


def _freeze_check(split: str, public: list[dict[str, Any]], hidden: list[dict[str, Any]]) -> dict[str, Any]:
    freeze = _read_json(EVIDENCE_ROOT / "pre_model_call_freeze.json")
    protocol = _read_json(EVIDENCE_ROOT / "evaluator_protocol.json")
    protocol_copy = {key: value for key, value in protocol.items() if key != "protocol_sha256"}
    source_ok = (
        freeze.get("phase53_hard_detector_source_sha256") == _sha256(PHASE53_SOURCE)
        and freeze.get("phase56_composer_source_sha256") == _sha256(PHASE56_SOURCE)
        and freeze.get("phase58_clause_grounder_source_sha256") == _sha256(PHASE58_SOURCE)
        and freeze.get("phase59_evaluator_source_sha256") == _sha256(PHASE59_SOURCE)
    )
    protocol_ok = (
        bool(protocol.get("protocol_sha256"))
        and stable_hash(protocol_copy) == protocol.get("protocol_sha256") == freeze.get("protocol_sha256")
    )
    data_ok = (
        stable_hash(public) == freeze.get(f"{split}_public_sha256")
        and stable_hash(hidden) == freeze.get(f"{split}_hidden_sha256")
    )
    prerequisite_ok = True
    if split == "holdout":
        prerequisite_ok = _read_json(
            EVIDENCE_ROOT / "evidence-evaluator-calibration/candidate_evaluator_report.json"
        ).get("status") == "qualified"
    return {
        "kind": "phase59_candidate_evaluator_freeze_check",
        "split": split,
        "passed": source_ok and protocol_ok and data_ok and prerequisite_ok,
        "source_checks_passed": source_ok,
        "protocol_check_passed": protocol_ok,
        "data_check_passed": data_ok,
        "prerequisite_check_passed": prerequisite_ok,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("calibration", "holdout"), required=True)
    parser.add_argument("--ollama-endpoint", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    directory, public_path, hidden_path = _fixture_paths(args.split)
    public = _read_jsonl(public_path)
    hidden = list(_read_json(hidden_path).get("items") or [])
    freeze_check = _freeze_check(args.split, public, hidden)
    _write_json(directory / "freeze_check.json", freeze_check)
    if freeze_check["passed"] is not True:
        raise SystemExit(f"Phase59 {args.split} freeze check failed: {freeze_check}")

    protocol = _read_json(EVIDENCE_ROOT / "evaluator_protocol.json")
    workers = int(protocol.get("parallel_worker_count") or 1)
    if args.workers is not None and args.workers != workers:
        raise SystemExit(f"Phase59 worker count is frozen at {workers}; received --workers={args.workers}")
    aliases = tuple(str(value) for value in protocol.get("semantic_judge_aliases") or [])
    models = {str(key): str(value) for key, value in dict(protocol.get("semantic_judge_models_private") or {}).items()}
    tags = _ollama_tags(args.ollama_endpoint, max(10, args.timeout))
    missing = [models[alias] for alias in aliases if models.get(alias) not in tags["available"]]
    availability = {
        "kind": "phase59_judge_model_availability",
        "split": args.split,
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
        raise SystemExit(f"Phase59 judge models unavailable: {missing}")

    all_results = []
    failures = []
    for alias in aliases:
        output_path = directory / f"judge_candidate_results_{alias}.jsonl"
        if output_path.exists() and not args.resume:
            raise SystemExit(f"Phase59 {args.split} is frozen one-shot evidence; use --resume only after interruption")
        existing = _read_jsonl(output_path) if args.resume else []
        by_id = {}
        for row in existing:
            item = next((entry for entry in public if entry.get("item_id") == row.get("item_id")), {})
            candidates = build_phase59_proposition_candidates(str(item.get("assistant_response") or ""))
            try:
                validate_phase59_raw_selection(row, candidates=candidates)
            except ValueError:
                continue
            if row.get("actual_model_call") is True:
                by_id[str(row.get("item_id") or "")] = dict(row)
        pending = [
            (index, dict(item))
            for index, item in enumerate(public, start=1)
            if str(item.get("item_id") or "") not in by_id
        ]

        def run_pending(entry: tuple[int, dict[str, Any]]) -> dict[str, Any]:
            index, item = entry
            item_id = str(item.get("item_id") or "")
            error = None
            for attempt in range(1, 3):
                try:
                    result = _judge_one(
                        item=item, alias=alias, model=models[alias], endpoint=args.ollama_endpoint,
                        timeout=max(30, args.timeout), protocol=protocol,
                    )
                    result["parallel_worker_count"] = workers
                    return {"index": index, "item_id": item_id, "result": result, "error": None}
                except (OSError, HTTPError, URLError, ValueError, json.JSONDecodeError) as exc:
                    error = f"{exc.__class__.__name__}: {exc}"
                    print(
                        f"[{args.split}:{alias}] {index}/{len(public)} {item_id} attempt={attempt} failed: {error}",
                        flush=True,
                    )
                    time.sleep(attempt)
            return {"index": index, "item_id": item_id, "result": None, "error": error}

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(run_pending, entry) for entry in pending]
            for future in as_completed(futures):
                outcome = future.result()
                item_id = str(outcome["item_id"])
                result = outcome["result"]
                if result is None:
                    failures.append({"item_id": item_id, "judge_alias": alias, "error": outcome["error"]})
                    continue
                by_id[item_id] = result
                _write_jsonl(output_path, [by_id[key] for key in sorted(by_id)])
                print(
                    f"[{args.split}:{alias}] {outcome['index']}/{len(public)} {item_id} "
                    f"source={result['source_registration_candidate_id']} "
                    f"outcome={result['user_outcome_status_candidate_id']} "
                    f"relation={result['test_to_user_outcome_relation_candidate_id']} "
                    f"latency={result['latency_seconds']}s",
                    flush=True,
                )
        all_results.extend(by_id[key] for key in sorted(by_id))

    report = evaluate_phase59_candidate_evaluator(
        split=args.split,
        public_items=public,
        hidden_key=hidden,
        judge_results=all_results,
        judge_aliases=aliases,
    )
    report.update(
        {
            "failure_count": len(failures),
            "failures": failures,
            "judge_models": models,
            "ollama_endpoint": args.ollama_endpoint,
            "parallel_worker_count": workers,
            "request_timeout_seconds": max(30, args.timeout),
            "fabricated_scores": False,
            "created_at": _utcnow(),
        }
    )
    _write_json(directory / "candidate_evaluator_report.json", report)
    print(
        json.dumps(
            {key: report.get(key) for key in (
                "split", "status", "item_count", "completed_item_count", "accuracy",
                "typed_exact_match_rate", "candidate_selection_exact_match_rate", "per_field",
                "per_category", "invalid_dangerous_atom_count", "false_accept_count_on_reject_cases",
                "schema_failure_count", "failure_count",
            )},
            ensure_ascii=False,
            indent=2,
        )
    )
    expected = len(public) * len(aliases)
    return 0 if len(all_results) == expected and not failures and report.get("status") == "qualified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
