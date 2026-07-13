#!/usr/bin/env python3
"""Run the frozen Phase54 Gemma/Qwen typed proposition protocol."""

from __future__ import annotations

import argparse
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
from pfe_core.phase54_typed_proposition_evaluator import (
    build_phase54_typed_judge_prompt,
    evaluate_phase54_typed_evaluator,
    validate_phase54_typed_extraction,
)


EVIDENCE_ROOT = REPO_ROOT / "docs/demo/phase54-typed-proposition-evaluator"
PHASE53_SOURCE = CORE_ROOT / "pfe_core/phase53_evaluator_scope_recovery.py"
PHASE54_SOURCE = CORE_ROOT / "pfe_core/phase54_typed_proposition_evaluator.py"


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
    names = sorted(str(row.get("name") or "") for row in body.get("models") or [])
    return {"available": names, "queried_at": _utcnow()}


def _judge_one(
    *, item: Mapping[str, Any], alias: str, model: str, endpoint: str, timeout: int, protocol: Mapping[str, Any]
) -> dict[str, Any]:
    prompt = build_phase54_typed_judge_prompt(item)
    num_predict = int(protocol.get("num_predict") or 256)
    payload = {
        "model": model,
        "stream": False,
        "think": False,
        "format": dict(protocol.get("ollama_json_schema") or {}),
        "keep_alive": "30m",
        "messages": [{"role": "user", "content": prompt}],
        "options": {"temperature": 0, "num_predict": num_predict},
    }
    request = Request(
        endpoint.rstrip("/") + "/api/chat",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    content = str(dict(body.get("message") or {}).get("content") or "")
    parsed = _parse_json_object(content)
    typed = validate_phase54_typed_extraction(parsed)
    return {
        "item_id": item.get("item_id"),
        "judge_alias": alias,
        "judge_model": model,
        **typed,
        "actual_model_call": True,
        "identity_hidden_from_judge": True,
        "gold_label_hidden_from_judge": True,
        "gold_typed_fields_hidden_from_judge": True,
        "other_judge_identity_hidden": True,
        "judge_returned_direct_label": False,
        "think": False,
        "temperature": 0,
        "num_predict": num_predict,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "raw_response": content,
        "done_reason": body.get("done_reason"),
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
        freeze.get("phase53_evaluator_source_sha256") == _sha256(PHASE53_SOURCE)
        and freeze.get("phase54_evaluator_source_sha256") == _sha256(PHASE54_SOURCE)
    )
    protocol_ok = (
        bool(protocol.get("protocol_sha256"))
        and stable_hash(protocol_copy) == protocol.get("protocol_sha256") == freeze.get("protocol_sha256")
    )
    if split == "calibration":
        data_ok = (
            stable_hash(public) == freeze.get("calibration_public_sha256")
            and stable_hash(hidden) == freeze.get("calibration_hidden_sha256")
        )
        prerequisite_ok = True
    else:
        data_ok = (
            stable_hash(public) == freeze.get("holdout_public_sha256")
            and stable_hash(hidden) == freeze.get("holdout_hidden_sha256")
        )
        prerequisite_ok = _read_json(
            EVIDENCE_ROOT / "evidence-evaluator-calibration/typed_evaluator_report.json"
        ).get("status") == "qualified"
    return {
        "kind": "phase54_typed_evaluator_freeze_check",
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
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    directory, public_path, hidden_path = _fixture_paths(args.split)
    public = _read_jsonl(public_path)
    hidden = list(_read_json(hidden_path).get("items") or [])
    freeze_check = _freeze_check(args.split, public, hidden)
    _write_json(directory / "freeze_check.json", freeze_check)
    if freeze_check["passed"] is not True:
        raise SystemExit(f"Phase54 {args.split} freeze check failed: {freeze_check}")

    protocol = _read_json(EVIDENCE_ROOT / "evaluator_protocol.json")
    aliases = tuple(str(value) for value in protocol.get("semantic_judge_aliases") or [])
    models = {
        str(key): str(value)
        for key, value in dict(protocol.get("semantic_judge_models_private") or {}).items()
    }
    tags = _ollama_tags(args.ollama_endpoint, max(10, args.timeout))
    missing = [models[alias] for alias in aliases if models.get(alias) not in tags["available"]]
    availability = {
        "kind": "phase54_judge_model_availability",
        "split": args.split,
        "judge_aliases": list(aliases),
        "judge_models": models,
        "ollama_models_available": tags["available"],
        "missing_models": missing,
        "passed": not missing and len(aliases) == 2,
        "queried_at": tags["queried_at"],
    }
    _write_json(directory / "model_availability.json", availability)
    if availability["passed"] is not True:
        raise SystemExit(f"Phase54 judge models unavailable: {missing}")

    all_results = []
    failures = []
    for alias in aliases:
        output_path = directory / f"judge_typed_results_{alias}.jsonl"
        existing = _read_jsonl(output_path) if args.resume else []
        by_id = {}
        for row in existing:
            try:
                validate_phase54_typed_extraction(row)
            except ValueError:
                continue
            if row.get("actual_model_call") is True:
                by_id[str(row.get("item_id") or "")] = dict(row)
        for index, item in enumerate(public, start=1):
            item_id = str(item.get("item_id") or "")
            if item_id in by_id:
                print(f"[{args.split}:{alias}] {index}/{len(public)} {item_id} resumed", flush=True)
                continue
            error = None
            for attempt in range(1, 3):
                try:
                    result = _judge_one(
                        item=item,
                        alias=alias,
                        model=models[alias],
                        endpoint=args.ollama_endpoint,
                        timeout=max(30, args.timeout),
                        protocol=protocol,
                    )
                    by_id[item_id] = result
                    _write_jsonl(output_path, [by_id[key] for key in sorted(by_id)])
                    print(
                        f"[{args.split}:{alias}] {index}/{len(public)} {item_id} "
                        f"source={result['source_eligibility']} benefit={result['current_benefit_assertion']}",
                        flush=True,
                    )
                    error = None
                    break
                except (OSError, HTTPError, URLError, ValueError, json.JSONDecodeError) as exc:
                    error = f"{exc.__class__.__name__}: {exc}"
                    print(
                        f"[{args.split}:{alias}] {index}/{len(public)} {item_id} attempt={attempt} failed: {error}",
                        flush=True,
                    )
                    time.sleep(attempt)
            if error is not None:
                failures.append({"item_id": item_id, "judge_alias": alias, "error": error})
        all_results.extend(by_id[key] for key in sorted(by_id))

    report = evaluate_phase54_typed_evaluator(
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
            "fabricated_scores": False,
            "created_at": _utcnow(),
        }
    )
    _write_json(directory / "typed_evaluator_report.json", report)
    print(
        json.dumps(
            {
                key: report.get(key)
                for key in (
                    "split",
                    "status",
                    "item_count",
                    "completed_item_count",
                    "accuracy",
                    "typed_exact_match_rate",
                    "per_field",
                    "false_accept_count_on_reject_cases",
                    "hard_reject_vs_two_safe_accept_conflict_count",
                    "judge_direct_label_count",
                    "per_category",
                    "failure_count",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    expected = len(public) * len(aliases)
    return 0 if len(all_results) == expected and not failures and report.get("status") == "qualified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
