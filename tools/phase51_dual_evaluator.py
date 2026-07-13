#!/usr/bin/env python3
"""Run Phase51 identity-hidden Gemma/Qwen semantic evaluation."""

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
from pfe_core.phase51_dual_evaluator_hardening import (
    build_phase51_runtime_blind_items,
    build_phase51_semantic_judge_prompt,
    evaluate_phase51_dual_evaluator,
    summarize_phase51_runtime_judgments,
)


EVIDENCE_ROOT = REPO_ROOT / "docs" / "demo" / "phase51-dual-evaluator-hardening"
PHASE50_SOURCE = CORE_ROOT / "pfe_core" / "phase50_conditional_provenance_guard.py"
PHASE51_SOURCE = CORE_ROOT / "pfe_core" / "phase51_dual_evaluator_hardening.py"
RUNTIME_DIR = EVIDENCE_ROOT / "evidence-real-runtime"
VARIANTS = ("base_compact_v1", "base_global_v2", "base_conditional_guard")


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
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
        try:
            parsed = json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return {}
    return dict(parsed) if isinstance(parsed, Mapping) else {}


def _ollama_tags(endpoint: str, timeout: int) -> dict[str, Any]:
    request = Request(endpoint.rstrip("/") + "/api/tags", method="GET")
    with urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    names = sorted(str(row.get("name") or "") for row in body.get("models") or [])
    return {"available": names, "queried_at": _utcnow()}


def _judge_one(
    *, item: Mapping[str, Any], alias: str, model: str, endpoint: str, timeout: int
) -> dict[str, Any]:
    prompt = build_phase51_semantic_judge_prompt(item)
    payload = {
        "model": model,
        "stream": False,
        "think": False,
        "format": "json",
        "keep_alive": "30m",
        "messages": [{"role": "user", "content": prompt}],
        "options": {"temperature": 0, "num_predict": 192},
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
    label = str(parsed.get("label") or "").strip().lower()
    if label not in {"accept", "edit", "reject"}:
        raise ValueError(f"invalid semantic judge label: {label or '<empty>'}")
    try:
        confidence = max(0.0, min(1.0, float(parsed.get("confidence") or 0.0)))
    except (TypeError, ValueError):
        confidence = 0.0
    return {
        "item_id": item.get("item_id"),
        "judge_alias": alias,
        "judge_model": model,
        "label": label,
        "confidence": confidence,
        "reason": str(parsed.get("reason") or "").strip(),
        "actual_model_call": True,
        "identity_hidden_from_judge": True,
        "gold_label_hidden_from_judge": True,
        "other_judge_identity_hidden": True,
        "think": False,
        "temperature": 0,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "raw_response": content,
        "latency_seconds": round(time.perf_counter() - started, 4),
        "created_at": _utcnow(),
    }


def _fixture_paths(split: str) -> tuple[Path, Path, Path]:
    if split == "calibration":
        directory = EVIDENCE_ROOT / "evidence-evaluator-calibration"
    elif split == "holdout":
        directory = EVIDENCE_ROOT / "evidence-evaluator-holdout"
    else:
        directory = EVIDENCE_ROOT / "evidence-runtime-dual-eval"
    return directory, directory / "blind_items_public.jsonl", directory / "blind_hidden_key.json"


def _prepare_runtime_blind(public_path: Path, hidden_path: Path) -> None:
    sessions = _read_json(EVIDENCE_ROOT / "evidence-runtime-holdout" / "holdout.json").get("sessions") or []
    transcripts = {
        variant: _read_jsonl(RUNTIME_DIR / f"transcripts_{variant}.jsonl")
        for variant in VARIANTS
    }
    if any(
        len(rows) != 48
        or any(row.get("status") != "completed" or row.get("actual_model_call") is not True for row in rows)
        for rows in transcripts.values()
    ):
        raise SystemExit("Phase51 runtime dual eval requires 48 completed actual transcripts per arm")
    blind = build_phase51_runtime_blind_items(transcripts, sessions, seed=51)
    _write_jsonl(public_path, blind["public_items"])
    _write_json(hidden_path, {"items": blind["hidden_key"]})
    _write_json(
        public_path.parent / "runtime_blind_freeze.json",
        {
            "kind": "phase51_runtime_blind_freeze",
            "public_items_sha256": stable_hash(blind["public_items"]),
            "hidden_key_sha256": stable_hash(blind["hidden_key"]),
            "item_count": len(blind["public_items"]),
            "expected_item_count": 72,
            "identity_hidden_from_judges": True,
            "frozen_before_runtime_judge_calls": True,
            "created_at": _utcnow(),
        },
    )


def _freeze_check(split: str, public: list[dict[str, Any]], hidden: list[dict[str, Any]]) -> dict[str, Any]:
    freeze = _read_json(EVIDENCE_ROOT / "pre_model_call_freeze.json")
    protocol = _read_json(EVIDENCE_ROOT / "evaluator_runtime_protocol.json")
    source_ok = (
        freeze.get("phase50_runtime_source_sha256") == _sha256(PHASE50_SOURCE)
        and freeze.get("phase51_evaluator_source_sha256") == _sha256(PHASE51_SOURCE)
    )
    protocol_hash = str(protocol.get("protocol_sha256") or "")
    protocol_copy = {key: value for key, value in protocol.items() if key != "protocol_sha256"}
    protocol_ok = bool(protocol_hash) and stable_hash(protocol_copy) == protocol_hash == freeze.get("protocol_sha256")
    if split == "calibration":
        data_ok = (
            stable_hash(public) == freeze.get("calibration_public_sha256")
            and stable_hash(hidden) == freeze.get("calibration_hidden_sha256")
        )
        prerequisite_ok = True
    elif split == "holdout":
        data_ok = (
            stable_hash(public) == freeze.get("holdout_public_sha256")
            and stable_hash(hidden) == freeze.get("holdout_hidden_sha256")
        )
        calibration_report = _read_json(
            EVIDENCE_ROOT / "evidence-evaluator-calibration" / "dual_evaluator_report.json"
        )
        prerequisite_ok = calibration_report.get("status") == "qualified"
    else:
        runtime_freeze = _read_json(public_path := EVIDENCE_ROOT / "evidence-runtime-dual-eval" / "runtime_blind_freeze.json")
        data_ok = (
            public_path.exists()
            and stable_hash(public) == runtime_freeze.get("public_items_sha256")
            and stable_hash(hidden) == runtime_freeze.get("hidden_key_sha256")
            and int(runtime_freeze.get("item_count") or 0) == 72
        )
        calibration_report = _read_json(
            EVIDENCE_ROOT / "evidence-evaluator-calibration" / "dual_evaluator_report.json"
        )
        holdout_report = _read_json(
            EVIDENCE_ROOT / "evidence-evaluator-holdout" / "dual_evaluator_report.json"
        )
        prerequisite_ok = (
            calibration_report.get("status") == "qualified"
            and holdout_report.get("status") == "qualified"
        )
    return {
        "kind": "phase51_dual_evaluator_freeze_check",
        "split": split,
        "passed": source_ok and protocol_ok and data_ok and prerequisite_ok,
        "source_checks_passed": source_ok,
        "protocol_check_passed": protocol_ok,
        "data_check_passed": data_ok,
        "prerequisite_check_passed": prerequisite_ok,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("calibration", "holdout", "runtime"), required=True)
    parser.add_argument("--ollama-endpoint", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    directory, public_path, hidden_path = _fixture_paths(args.split)
    directory.mkdir(parents=True, exist_ok=True)
    if args.split == "runtime" and (not args.resume or not public_path.exists()):
        _prepare_runtime_blind(public_path, hidden_path)
    public = _read_jsonl(public_path)
    hidden = list(_read_json(hidden_path).get("items") or [])
    freeze_check = _freeze_check(args.split, public, hidden)
    _write_json(directory / "freeze_check.json", freeze_check)
    if freeze_check["passed"] is not True:
        raise SystemExit(f"Phase51 {args.split} freeze check failed: {freeze_check}")

    protocol = _read_json(EVIDENCE_ROOT / "evaluator_runtime_protocol.json")
    evaluator = dict(protocol.get("evaluator") or {})
    aliases = tuple(str(value) for value in evaluator.get("semantic_judge_aliases") or [])
    models = {str(key): str(value) for key, value in dict(evaluator.get("semantic_judge_models_private") or {}).items()}
    tags = _ollama_tags(args.ollama_endpoint, max(10, args.timeout))
    missing = [models[alias] for alias in aliases if models.get(alias) not in tags["available"]]
    availability = {
        "kind": "phase51_judge_model_availability",
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
        raise SystemExit(f"Phase51 judge models unavailable: {missing}")

    selected = public[:max(0, args.limit)] if args.limit is not None else public
    all_results = []
    failures = []
    for alias in aliases:
        output_path = directory / f"judge_results_{alias}.jsonl"
        existing = _read_jsonl(output_path) if args.resume else []
        by_id = {
            str(row.get("item_id") or ""): dict(row)
            for row in existing
            if row.get("label") in {"accept", "edit", "reject"}
            and row.get("actual_model_call") is True
        }
        for index, item in enumerate(selected, start=1):
            item_id = str(item.get("item_id") or "")
            if item_id in by_id:
                print(f"[{args.split}:{alias}] {index}/{len(selected)} {item_id} resumed", flush=True)
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
                    )
                    by_id[item_id] = result
                    _write_jsonl(output_path, [by_id[key] for key in sorted(by_id)])
                    print(
                        f"[{args.split}:{alias}] {index}/{len(selected)} {item_id} {result['label']}",
                        flush=True,
                    )
                    error = None
                    break
                except (OSError, HTTPError, URLError, ValueError, json.JSONDecodeError) as exc:
                    error = f"{exc.__class__.__name__}: {exc}"
                    print(
                        f"[{args.split}:{alias}] {index}/{len(selected)} {item_id} attempt={attempt} failed: {error}",
                        flush=True,
                    )
                    time.sleep(attempt)
            if error is not None:
                failures.append({"item_id": item_id, "judge_alias": alias, "error": error})
        all_results.extend(by_id[key] for key in sorted(by_id))

    if args.split in {"calibration", "holdout"}:
        report = evaluate_phase51_dual_evaluator(
            split=args.split,
            public_items=public,
            hidden_key=hidden,
            judge_results=all_results,
            judge_aliases=aliases,
        )
    else:
        report = summarize_phase51_runtime_judgments(
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
    _write_json(directory / "dual_evaluator_report.json", report)
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
                    "false_accept_count_on_reject_cases",
                    "judge_agreement_rate",
                    "by_variant",
                    "failure_count",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    expected = len(public) * len(aliases)
    complete = len(all_results) == expected and not failures
    qualified = report.get("status") == ("completed" if args.split == "runtime" else "qualified")
    return 0 if complete and qualified else 1


if __name__ == "__main__":
    raise SystemExit(main())
