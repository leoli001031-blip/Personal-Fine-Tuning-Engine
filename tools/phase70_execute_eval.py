#!/usr/bin/env python3
"""Execute Phase70 evaluator qualification and product stages."""

from __future__ import annotations

import argparse
from collections import Counter
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

from pfe_core.phase59_proposition_addressed_grounding import (
    build_phase59_candidate_judge_prompt,
    build_phase59_proposition_candidates,
    phase59_ollama_json_schema,
    validate_phase59_raw_selection,
)
from pfe_core.phase63_field_typed_candidate_wire import (
    build_phase63_typed_wire_prompt,
    parse_phase63_typed_wire_selection,
)
from pfe_core.phase62_risk_asymmetric_candidate_consensus import evaluate_phase62_candidate_consensus
from pfe_core.phase68_aligned_candidate_scope_recovery import PHASE68_CATEGORIES
from pfe_core.phase70_structured_boundary_contract import (
    evaluate_phase70_boundary_results,
    stable_hash,
)
from phase62_execute import JudgeAttemptError, _ollama_tags, _utcnow


EVIDENCE_ROOT = REPO_ROOT / "docs/demo/phase70-structured-boundary-contract"
STAGES = ("sparse_preflight", "phase68_regression", "product")
SOURCE_PATHS = {
    "phase46_generator_helpers": "tools/phase46_qwen3_4b_generate.py",
    "phase53_hard_detector": "pfe-core/pfe_core/phase53_evaluator_scope_recovery.py",
    "phase56_grounder_composer": "pfe-core/pfe_core/phase56_evidence_span_grounded_atomic.py",
    "phase59_candidates": "pfe-core/pfe_core/phase59_proposition_addressed_grounding.py",
    "phase63_typed_wire": "pfe-core/pfe_core/phase63_field_typed_candidate_wire.py",
    "phase62_consensus": "pfe-core/pfe_core/phase62_risk_asymmetric_candidate_consensus.py",
    "phase70_core": "pfe-core/pfe_core/phase70_structured_boundary_contract.py",
    "phase70_prepare": "tools/phase70_prepare.py",
    "phase70_generate": "tools/phase70_generate.py",
    "phase70_prepare_product_eval": "tools/phase70_prepare_product_eval.py",
    "phase70_execute_eval": "tools/phase70_execute_eval.py",
    "phase70_finalize": "tools/phase70_finalize_evidence.py",
}


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
    transports = dict(protocol.get("judge_transport_by_alias") or {})
    transport = str(transports.get(alias) or "")
    if transport == "phase59_nested_json_schema":
        prompt = build_phase59_candidate_judge_prompt(item)
        schema = phase59_ollama_json_schema(candidates)
    elif transport == "phase63_typed_wire":
        prompt = build_phase63_typed_wire_prompt(item)
        schema = None
    else:
        raise ValueError(f"unsupported Phase70 judge transport for {alias}: {transport!r}")
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
            f"HTTPError {exc.code}: {raw_http}", raw_response=raw_http, failure_class="http_error"
        ) from exc
    try:
        body = json.loads(raw_http)
        content = str(dict(body.get("message") or {}).get("content") or "")
        if transport == "phase59_nested_json_schema":
            selection = validate_phase59_raw_selection(_parse_json(content), candidates=candidates)
        else:
            selection = parse_phase63_typed_wire_selection(content, candidates=candidates)
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
        "field_typed_wire_valid": transport == "phase63_typed_wire",
        "num_predict": num_predict,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "schema_sha256": stable_hash(schema) if schema is not None else None,
        "typed_wire_spec_sha256": (
            stable_hash(dict(protocol.get("typed_wire_spec") or {}))
            if transport == "phase63_typed_wire"
            else None
        ),
        "raw_response": content,
        "done_reason": body.get("done_reason"),
        "eval_count": eval_count,
        "eval_tokens_per_second": round(eval_count / eval_seconds, 4) if eval_seconds else None,
        "latency_seconds": round(time.perf_counter() - started, 4),
        "created_at": _utcnow(),
    }


def _paths(stage: str) -> tuple[Path, Path, Path]:
    directory = EVIDENCE_ROOT / {
        "sparse_preflight": "evidence-sparse-preflight",
        "phase68_regression": "evidence-phase68-regression",
        "product": "evidence-product-eval",
    }[stage]
    return directory, directory / "blind_items_public.jsonl", directory / "blind_hidden_key.json"


def _freeze_check(stage: str, public: list[dict[str, Any]], hidden: list[dict[str, Any]]) -> dict[str, Any]:
    protocol = _read_json(EVIDENCE_ROOT / "runtime_ab_protocol.json")
    if stage == "product":
        freeze = _read_json(EVIDENCE_ROOT / "evidence-product-eval/pre_judge_freeze.json")
        expected_sources = dict(freeze.get("source_sha256") or {})
        product_paths = {
            "phase70_core": SOURCE_PATHS["phase70_core"],
            "phase70_prepare_product_eval": SOURCE_PATHS["phase70_prepare_product_eval"],
            "phase70_execute_eval": SOURCE_PATHS["phase70_execute_eval"],
            "phase70_finalize": SOURCE_PATHS["phase70_finalize"],
        }
        source_ok = all(_sha256(REPO_ROOT / product_paths[name]) == value for name, value in expected_sources.items())
        data_ok = stable_hash(public) == freeze.get("public_sha256") and stable_hash(hidden) == freeze.get("hidden_sha256")
        prerequisite = _read_json(EVIDENCE_ROOT / "evidence-product-eval/preparation_decision.json").get("status") == "ready_for_product_eval"
    else:
        freeze = _read_json(EVIDENCE_ROOT / "pre_model_call_freeze.json")
        source_ok = all(_sha256(REPO_ROOT / SOURCE_PATHS[name]) == value for name, value in dict(freeze.get("source_sha256") or {}).items())
        prefix = "sparse" if stage == "sparse_preflight" else "regression"
        data_ok = stable_hash(public) == freeze.get(f"{prefix}_public_sha256") and stable_hash(hidden) == freeze.get(f"{prefix}_hidden_sha256")
        prerequisite = (
            _read_json(EVIDENCE_ROOT / "preparation_decision.json").get("status") == "ready_for_sparse_transport_preflight"
            if stage == "sparse_preflight"
            else _read_json(EVIDENCE_ROOT / "evidence-sparse-preflight/evaluator_report.json").get("status") == "qualified"
        )
    return {
        "kind": "phase70_eval_freeze_check",
        "stage": stage,
        "passed": source_ok and data_ok and prerequisite and protocol.get("protocol_sha256") == freeze.get("protocol_sha256"),
        "source_checks_passed": source_ok,
        "data_check_passed": data_ok,
        "prerequisite_passed": prerequisite,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--ollama-endpoint", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    directory, public_path, hidden_path = _paths(args.stage)
    public = _read_jsonl(public_path)
    hidden = list(_read_json(hidden_path).get("items") or [])
    freeze = _freeze_check(args.stage, public, hidden)
    _write_json(directory / "freeze_check.json", freeze)
    if freeze["passed"] is not True:
        raise SystemExit(f"Phase70 {args.stage} freeze failed: {freeze}")
    protocol = _read_json(EVIDENCE_ROOT / "runtime_ab_protocol.json")
    aliases = tuple(str(value) for value in protocol.get("semantic_judge_aliases") or [])
    models = {str(key): str(value) for key, value in dict(protocol.get("semantic_judge_models_private") or {}).items()}
    workers = int(protocol.get("parallel_worker_count") or 1)
    tags = _ollama_tags(args.ollama_endpoint, max(10, args.timeout))
    missing = [models[alias] for alias in aliases if models.get(alias) not in tags["available"]]
    availability = {
        "kind": "phase70_model_availability",
        "stage": args.stage,
        "judge_models": models,
        "available": tags["available"],
        "missing": missing,
        "passed": not missing and len(aliases) == 2,
    }
    _write_json(directory / "model_availability.json", availability)
    if availability["passed"] is not True:
        raise SystemExit(f"Phase70 judge models unavailable: {missing}")
    all_results = []
    all_failure_attempts = []
    exhausted = []
    retry_limit = int(protocol.get("retry_limit") or 2)
    for alias in aliases:
        result_path = directory / f"judge_results_{alias}.jsonl"
        failure_path = directory / f"failure_attempts_{alias}.jsonl"
        if (result_path.exists() or failure_path.exists()) and not args.resume:
            raise SystemExit(f"Phase70 {args.stage} is one-shot evidence; use --resume after interruption")
        existing = _read_jsonl(result_path) if args.resume else []
        failures = _read_jsonl(failure_path) if args.resume else []
        by_id = {str(row.get("item_id")): row for row in existing if row.get("actual_model_call") is True}
        attempts = Counter()
        for row in failures:
            attempts[str(row.get("item_id"))] = max(attempts[str(row.get("item_id"))], int(row.get("attempt") or 0))
        pending = [(index, row) for index, row in enumerate(public, start=1) if str(row.get("item_id")) not in by_id and attempts[str(row.get("item_id"))] < retry_limit]

        def run(entry: tuple[int, dict[str, Any]]) -> dict[str, Any]:
            index, item = entry
            item_id = str(item.get("item_id"))
            local_failures = []
            for attempt in range(attempts[item_id] + 1, retry_limit + 1):
                try:
                    return {
                        "index": index,
                        "item_id": item_id,
                        "result": _invoke_judge(
                            item=item,
                            alias=alias,
                            model=models[alias],
                            endpoint=args.ollama_endpoint,
                            timeout=max(30, args.timeout),
                            protocol=protocol,
                            stage=args.stage,
                        ),
                        "failures": local_failures,
                    }
                except JudgeAttemptError as exc:
                    local_failures.append(
                        {
                            "item_id": item_id,
                            "judge_alias": alias,
                            "attempt": attempt,
                            "failure_class": exc.failure_class,
                            "error": str(exc),
                            "raw_response": exc.raw_response,
                            "created_at": _utcnow(),
                        }
                    )
                    time.sleep(attempt)
                except (OSError, URLError) as exc:
                    local_failures.append(
                        {
                            "item_id": item_id,
                            "judge_alias": alias,
                            "attempt": attempt,
                            "failure_class": "transport_error",
                            "error": f"{exc.__class__.__name__}: {exc}",
                            "raw_response": "",
                            "created_at": _utcnow(),
                        }
                    )
            return {"index": index, "item_id": item_id, "result": None, "failures": local_failures}

        done = 0
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for future in as_completed([executor.submit(run, entry) for entry in pending]):
                outcome = future.result()
                failures.extend(outcome["failures"])
                _write_jsonl(failure_path, failures)
                if outcome["result"] is not None:
                    by_id[str(outcome["item_id"])] = outcome["result"]
                    _write_jsonl(result_path, [by_id[key] for key in sorted(by_id)])
                    done += 1
                    if done % 12 == 0 or done == len(pending):
                        print(f"[{args.stage}:{alias}] {len(by_id)}/{len(public)}", flush=True)
        for item in public:
            item_id = str(item.get("item_id"))
            if item_id not in by_id:
                exhausted.append({"item_id": item_id, "judge_alias": alias})
        all_results.extend(by_id[key] for key in sorted(by_id))
        all_failure_attempts.extend(failures)
    if args.stage == "product":
        report = evaluate_phase70_boundary_results(
            public_items=public,
            hidden_key=hidden,
            judge_results=all_results,
            judge_aliases=aliases,
        )
    else:
        categories = tuple(sorted({str(row.get("category") or "") for row in hidden})) if args.stage == "sparse_preflight" else PHASE68_CATEGORIES
        report = evaluate_phase62_candidate_consensus(
            split="calibration" if args.stage == "sparse_preflight" else "holdout",
            public_items=public,
            hidden_key=hidden,
            judge_results=all_results,
            judge_aliases=aliases,
            categories=categories,
        )
    report.update(
        {
            "stage": args.stage,
            "successful_model_output_count": len(all_results),
            "expected_model_output_count": len(public) * len(aliases),
            "failure_count": len(exhausted),
            "failures": exhausted,
            "raw_failure_attempt_count": len(all_failure_attempts),
            "raw_failures_preserved": all("raw_response" in row for row in all_failure_attempts),
            "transport": "alias_capability_routed_candidate_transport",
            "judge_transport_by_alias": dict(protocol.get("judge_transport_by_alias") or {}),
            "fabricated_scores": False,
            "created_at": _utcnow(),
        }
    )
    _write_json(directory / "evaluator_report.json", report)
    summary = {
        "stage": args.stage,
        "status": report.get("status"),
        "outputs": len(all_results),
        "failures": len(exhausted),
        "accuracy": report.get("accuracy"),
        "baseline_accept": dict(report.get("variants") or {}).get("natural_boundary_contract", {}).get("accept_rate"),
        "candidate_accept": dict(report.get("variants") or {}).get("structured_boundary_contract", {}).get("accept_rate"),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not exhausted else 1


if __name__ == "__main__":
    raise SystemExit(main())
