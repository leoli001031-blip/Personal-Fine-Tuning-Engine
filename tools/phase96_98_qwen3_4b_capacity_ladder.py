#!/usr/bin/env python3
"""Run the local Qwen3-4B capacity, SFT, and DPO ladder for Phase96-98."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = REPO_ROOT / "pfe-core"
TOOLS_ROOT = REPO_ROOT / "tools"
for path in (CORE_ROOT, TOOLS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pfe_core.phase75_personalization_benefit_benchmark import stable_hash
from pfe_core.phase91_controlled_dpo_preference import score_phase91_output
from pfe_core.phase93_95_dpo_product_proof import aggregate_phase94_scores, has_repeated_output
from pfe_core.phase96_98_qwen3_4b_capacity_ladder import (
    audit_phase96_capacity_holdout,
    build_phase96_capacity_decision,
    build_phase96_capacity_holdout,
)
from phase87_89_failure_driven_adapter_loop import _release_runtime, _run_eval_session


EVIDENCE_ROOT = REPO_ROOT / "docs/demo/phase96-98-qwen3-4b-capacity-ladder"
PHASE96_ROOT = EVIDENCE_ROOT / "phase96-capacity-diagnostic"
PHASE96_PREPARATION_ROOT = PHASE96_ROOT / "evidence-preparation"
PHASE96_EVAL_ROOT = PHASE96_ROOT / "evidence-eval"
PRIVATE_REVIEW_ROOT = Path("/private/tmp/pfe-phase96-98-simulated-review")
QWEN25_15B_PATH = REPO_ROOT / "models/Qwen2.5-1.5B-Instruct"
QWEN3_4B_PATH = REPO_ROOT / "models/Qwen3-4B"
MODEL_VARIANTS = {
    "qwen25_1_5b": QWEN25_15B_PATH,
    "qwen3_4b": QWEN3_4B_PATH,
}
CAPACITY_THRESHOLDS = {
    "core_regression_allowed": False,
    "strict_core_improvement_required": True,
    "ordinary_control_regression_allowed": False,
    "unsupported_regression_allowed": False,
    "repetition_regression_allowed": False,
    "think_leak_maximum": 0.0,
    "privacy_echo_maximum": 0.0,
    "automatic_promotion_allowed": False,
}


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _write_private_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_clean(path: Path, parent: Path) -> None:
    resolved = path.resolve()
    if resolved.parent != parent.resolve():
        raise RuntimeError(f"refusing to clean outside {parent}: {path}")
    if resolved.exists():
        shutil.rmtree(resolved)


def _source_hashes() -> dict[str, str]:
    paths = {
        "core": CORE_ROOT / "pfe_core/phase96_98_qwen3_4b_capacity_ladder.py",
        "driver": REPO_ROOT / "tools/phase96_98_qwen3_4b_capacity_ladder.py",
        "core_test": REPO_ROOT / "tests/test_phase96_98_qwen3_4b_capacity_ladder.py",
        "driver_test": REPO_ROOT / "tests/test_phase96_driver_safety.py",
    }
    return {name: _sha256(path) for name, path in paths.items()}


def _model_integrity(model_path: Path) -> dict[str, Any]:
    index_path = model_path / "model.safetensors.index.json"
    if index_path.is_file():
        index = _read_json(index_path)
        shards = sorted(set(dict(index.get("weight_map") or {}).values()))
    else:
        shards = ["model.safetensors"]
    return {
        "model_path": str(model_path),
        "shards": shards,
        "shard_count": len(shards),
        "all_shards_present": all((model_path / shard).is_file() for shard in shards),
        "total_weight_bytes": sum((model_path / shard).stat().st_size for shard in shards if (model_path / shard).is_file()),
        "config_present": (model_path / "config.json").is_file(),
        "tokenizer_present": (model_path / "tokenizer.json").is_file(),
    }


def _previous_holdouts() -> list[dict[str, Any]]:
    paths = (
        REPO_ROOT / "docs/demo/phase43-qwen3-4b-personal-preference-benefit-proof/evidence-holdout/holdout.json",
        REPO_ROOT / "docs/demo/phase87-89-failure-driven-adapter-loop/evidence-preparation/holdout.json",
        REPO_ROOT / "docs/demo/phase90-native-format-curriculum-repair/evidence-preparation/holdout.json",
        REPO_ROOT / "docs/demo/phase91-controlled-dpo-preference-diagnostic/evidence-preparation/holdout.json",
    )
    payloads = [_read_json(path) for path in paths]
    phase93 = _read_json(
        REPO_ROOT / "docs/demo/phase92-95-autonomous-dpo-stability-product-proof/phase93-stable-dpo-training/evidence-preparation/fresh_holdouts.json"
    )
    payloads.append({"sessions": list(phase93.get("sanity_sessions") or []) + list(phase93.get("product_sessions") or [])})
    return payloads


def _prepare(clean: bool) -> int:
    if clean and PHASE96_ROOT.exists():
        _safe_clean(PHASE96_ROOT, EVIDENCE_ROOT)
    PHASE96_PREPARATION_ROOT.mkdir(parents=True, exist_ok=True)
    holdout = build_phase96_capacity_holdout()
    isolation = audit_phase96_capacity_holdout(holdout, _previous_holdouts())
    models = {name: _model_integrity(path) for name, path in MODEL_VARIANTS.items()}
    checks = {
        "holdout_isolation_passed": isolation.get("passed") is True,
        "both_models_complete": all(
            model["all_shards_present"] and model["config_present"] and model["tokenizer_present"]
            for model in models.values()
        ),
        "capacity_session_count_8": holdout.get("session_count") == 8,
        "planned_model_calls_48": int(holdout.get("session_count") or 0) * 3 * 2 == 48,
        "all_simulated_not_actual": all(
            row.get("simulated_usage") is True and row.get("actual_user_feedback") is False
            for row in holdout.get("sessions") or []
        ),
    }
    freeze = {
        "kind": "phase96_capacity_pre_generation_freeze",
        "created_at": _utcnow(),
        "passed": all(checks.values()),
        "checks": checks,
        "models": models,
        "holdout_manifest_sha256": holdout["manifest_sha256"],
        "holdout_isolation_sha256": stable_hash(isolation),
        "thresholds": CAPACITY_THRESHOLDS,
        "thresholds_sha256": stable_hash(CAPACITY_THRESHOLDS),
        "source_sha256": _source_hashes(),
        "model_call_budget": {"planned": 48, "maximum": 48},
        "simulated_usage": True,
        "actual_user_feedback_count": 0,
        "automatic_promotion_allowed": False,
    }
    _write_json(PHASE96_PREPARATION_ROOT / "capacity_holdout.json", holdout)
    _write_json(PHASE96_PREPARATION_ROOT / "holdout_isolation_audit.json", isolation)
    _write_json(PHASE96_PREPARATION_ROOT / "model_integrity.json", {"models": models})
    _write_json(PHASE96_ROOT / "pre_generation_freeze.json", freeze)
    print(json.dumps({"status": "ready" if freeze["passed"] else "blocked", "checks": checks}, ensure_ascii=False, indent=2))
    return 0 if freeze["passed"] else 2


def _load_base_runtime(model_path: Path) -> tuple[Any, Any, Any, str]:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device == "mps" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), local_files_only=True, low_cpu_mem_usage=True, dtype=dtype
    )
    model.to(device)
    model.eval()
    return torch, tokenizer, model, device


def _generation_freeze_check(variant: str) -> dict[str, Any]:
    freeze = _read_json(PHASE96_ROOT / "pre_generation_freeze.json")
    holdout = _read_json(PHASE96_PREPARATION_ROOT / "capacity_holdout.json")
    existing_calls = sum(
        int(_read_json(path).get("model_call_count") or 0)
        for path in PHASE96_EVAL_ROOT.glob("metrics_*.json")
    ) if PHASE96_EVAL_ROOT.exists() else 0
    checks = {
        "pre_generation_freeze_passed": freeze.get("passed") is True,
        "source_files_unchanged": _source_hashes() == freeze.get("source_sha256"),
        "holdout_unchanged": stable_hash(holdout.get("sessions") or []) == freeze.get("holdout_manifest_sha256"),
        "thresholds_unchanged": stable_hash(CAPACITY_THRESHOLDS) == freeze.get("thresholds_sha256"),
        "variant_allowed": variant in MODEL_VARIANTS,
        "model_calls_within_48": existing_calls + 24 <= 48,
    }
    return {"kind": "phase96_generation_freeze_check", "variant": variant, "passed": all(checks.values()), "checks": checks}


def _generate(variant: str, clean: bool) -> int:
    if variant not in MODEL_VARIANTS:
        raise SystemExit("unsupported Phase96 model variant")
    metrics_path = PHASE96_EVAL_ROOT / f"metrics_{variant}.json"
    structural_path = PHASE96_EVAL_ROOT / f"structural_sessions_{variant}.jsonl"
    cache_path = PRIVATE_REVIEW_ROOT / f"capacity_{variant}.jsonl"
    if metrics_path.exists():
        raise SystemExit(f"refusing to repeat completed model calls: {metrics_path}")
    if clean:
        structural_path.unlink(missing_ok=True)
        cache_path.unlink(missing_ok=True)
    freeze = _generation_freeze_check(variant)
    _write_json(PHASE96_EVAL_ROOT / f"freeze_check_{variant}.json", freeze)
    if not freeze["passed"]:
        return 2
    sessions = [dict(row) for row in _read_json(PHASE96_PREPARATION_ROOT / "capacity_holdout.json").get("sessions") or []]
    rows: list[dict[str, Any]] = []
    private_rows: list[dict[str, Any]] = []
    torch = tokenizer = model = device = None
    try:
        torch, tokenizer, model, device = _load_base_runtime(MODEL_VARIANTS[variant])
        for index, session in enumerate(sessions, start=1):
            try:
                structural, private = _run_eval_session(
                    session=session,
                    torch=torch,
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                    adapter_loaded=False,
                )
                raw_score = score_phase91_output(private["raw_output"], session)
                post_score = score_phase91_output(private["post_output"], session)
                latency = sum(float(turn.get("latency_seconds") or 0.0) for turn in structural.get("turns") or [])
                raw_score.update({"repeated_output": has_repeated_output(private["raw_output"]), "latency_seconds": round(latency, 4)})
                post_score.update({"repeated_output": has_repeated_output(private["post_output"]), "latency_seconds": round(latency, 4)})
                structural.update({
                    "kind": "phase96_capacity_structural_session",
                    "variant": variant,
                    "model_path": str(MODEL_VARIANTS[variant]),
                    "raw_score": raw_score,
                    "post_score": post_score,
                })
            except Exception as exc:
                structural = {
                    "kind": "phase96_capacity_structural_session",
                    "session_id": session.get("session_id"),
                    "category": session.get("category"),
                    "variant": variant,
                    "status": "failed",
                    "actual_model_call": False,
                    "error_type": exc.__class__.__name__,
                    "raw_model_output_persisted": False,
                    "simulated_usage": True,
                    "actual_user_feedback": False,
                }
                private = {"session_id": session.get("session_id"), "category": session.get("category"), "error_type": exc.__class__.__name__}
            rows.append(structural)
            private_rows.append(private)
            _write_jsonl(structural_path, rows)
            _write_private_jsonl(cache_path, private_rows)
            print(f"[capacity:{variant}] {index}/{len(sessions)} {session.get('session_id')} {structural['status']}", flush=True)
    finally:
        if torch is not None and model is not None and device is not None:
            _release_runtime(torch, model, device)
    completed = [row for row in rows if row.get("status") == "completed"]
    raw = aggregate_phase94_scores({"category": row.get("category"), **dict(row.get("raw_score") or {})} for row in completed)
    post = aggregate_phase94_scores({"category": row.get("category"), **dict(row.get("post_score") or {})} for row in completed)
    fallback_count = sum(row.get("final_fallback_used") is True for row in completed)
    post["fallback_rate"] = round(fallback_count / len(completed), 4) if completed else 0.0
    metrics = {
        "kind": "phase96_capacity_variant_metrics",
        "variant": variant,
        "model_path": str(MODEL_VARIANTS[variant]),
        "session_count": len(completed),
        "model_call_count": sum(int(row.get("turn_count") or 0) for row in completed),
        "all_sessions_completed": len(completed) == len(sessions),
        "raw": raw,
        "post_contract": post,
        "raw_output_cache": str(cache_path),
        "raw_output_cache_outside_repo": True,
        "simulated_usage": True,
        "actual_user_feedback_count": 0,
        "product_gate_qualified": False,
    }
    _write_json(metrics_path, metrics)
    print(json.dumps({"variant": variant, "raw": raw, "model_calls": metrics["model_call_count"]}, ensure_ascii=False, indent=2))
    return 0 if metrics["all_sessions_completed"] else 1


def _decide() -> int:
    smaller_payload = _read_json(PHASE96_EVAL_ROOT / "metrics_qwen25_1_5b.json")
    larger_payload = _read_json(PHASE96_EVAL_ROOT / "metrics_qwen3_4b.json")
    smaller = dict(smaller_payload.get("raw") or {})
    larger = dict(larger_payload.get("raw") or {})
    decision = build_phase96_capacity_decision(smaller, larger)
    decision["metrics"] = {"qwen25_1_5b": smaller, "qwen3_4b": larger}
    decision["model_call_count"] = int(smaller_payload.get("model_call_count") or 0) + int(larger_payload.get("model_call_count") or 0)
    decision["model_call_budget_maximum"] = 48
    _write_json(PHASE96_ROOT / "capacity_decision.json", decision)
    lines = [
        "# Phase96 Capacity Decision",
        "",
        f"- Status: `{decision['status']}`",
        f"- Passed: {str(decision['passed']).lower()}",
        f"- Model calls: {decision['model_call_count']}/48",
        "- Product gate qualified: false",
        "- Evidence: simulated usage only",
        "",
        "This gate only decides whether Qwen3-4B is a justified training target. It does not prove adapter benefit.",
    ]
    (PHASE96_ROOT / "phase96-decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return 0 if decision["passed"] else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("phase96-prepare")
    prepare.add_argument("--clean", action="store_true")
    generate = sub.add_parser("phase96-generate")
    generate.add_argument("--variant", choices=tuple(MODEL_VARIANTS), required=True)
    generate.add_argument("--clean", action="store_true")
    sub.add_parser("phase96-decide")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "phase96-prepare":
        return _prepare(args.clean)
    if args.command == "phase96-generate":
        return _generate(args.variant, args.clean)
    if args.command == "phase96-decide":
        return _decide()
    raise SystemExit("unsupported command")


if __name__ == "__main__":
    raise SystemExit(main())
