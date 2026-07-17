#!/usr/bin/env python3
"""Run Phase103 paired simulated-user acceptance for base and archived DPO."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = REPO_ROOT / "pfe-core"
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

from pfe_core.phase75_personalization_benefit_benchmark import stable_hash
from pfe_core.phase103_simulated_user_acceptance import (
    aggregate_phase103_scores,
    audit_phase103_sessions,
    build_phase103_decision,
    build_phase103_sessions,
    compare_phase103_variants,
    score_phase103_session,
)
from phase101_failure_targeted_sft import _load_runtime, _run_session, _write_private_jsonl


EVIDENCE_ROOT = REPO_ROOT / "docs/demo/phase100-104-autonomous-qwen3-training-benefit-loop"
PHASE100_ROOT = EVIDENCE_ROOT / "phase100-generation-boundary"
PHASE101_ROOT = EVIDENCE_ROOT / "phase101-failure-targeted-sft"
PHASE102_ROOT = EVIDENCE_ROOT / "phase102-failure-targeted-dpo"
PHASE_ROOT = EVIDENCE_ROOT / "phase103-simulated-user-acceptance"
PREPARATION_ROOT = PHASE_ROOT / "evidence-preparation"
EVAL_ROOT = PHASE_ROOT / "evidence-eval"
PRIVATE_ROOT = Path("/private/tmp/pfe-phase103-simulated-review")


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_clean(path: Path, parent: Path) -> None:
    resolved = path.resolve()
    if resolved.parent != parent.resolve():
        raise RuntimeError(f"refusing to clean outside {parent}: {path}")
    if resolved.exists():
        shutil.rmtree(resolved)


def _source_hashes() -> dict[str, str]:
    paths = {
        "core": CORE_ROOT / "pfe_core/phase103_simulated_user_acceptance.py",
        "driver": REPO_ROOT / "tools/phase103_simulated_user_acceptance.py",
        "core_test": REPO_ROOT / "tests/test_phase103_simulated_user_acceptance.py",
        "driver_test": REPO_ROOT / "tests/test_phase103_driver_safety.py",
        "phase101_driver": REPO_ROOT / "tools/phase101_failure_targeted_sft.py",
    }
    return {name: _sha256(path) for name, path in paths.items()}


def _dpo_adapter_dir() -> Path:
    attempt = _read_json(PHASE102_ROOT / "evidence-training/30step/training_attempt.json")
    validation = dict(attempt.get("adapter_validation") or {})
    path = Path(str(validation.get("artifact_dir") or ""))
    if attempt.get("status") != "completed" or validation.get("valid") is not True or not path.is_dir():
        raise SystemExit("Phase103 requires the valid archived Phase102 30-step DPO adapter")
    return path


def _previous_holdouts() -> list[dict[str, Any]]:
    paths = (
        PHASE100_ROOT / "evidence-preparation/diagnostic_holdout.json",
        PHASE100_ROOT / "evidence-preparation/final_holdout.json",
        PHASE101_ROOT / "evidence-preparation/holdout.json",
    )
    return [_read_json(path) for path in paths if path.is_file()]


def _training_rows() -> list[dict[str, Any]]:
    return [
        *_read_jsonl(PHASE101_ROOT / "evidence-preparation/selected_sft_samples.jsonl"),
        *_read_jsonl(PHASE102_ROOT / "evidence-preparation/selected_dpo_pairs.jsonl"),
    ]


def _prepare(clean: bool) -> int:
    if clean and PHASE_ROOT.exists():
        _safe_clean(PHASE_ROOT, EVIDENCE_ROOT)
    PREPARATION_ROOT.mkdir(parents=True, exist_ok=True)
    payload = build_phase103_sessions()
    audit = audit_phase103_sessions(payload, _training_rows(), _previous_holdouts())
    phase102 = _read_json(PHASE102_ROOT / "phase102-decision.json")
    adapter = _dpo_adapter_dir()
    checks = {
        "phase102_remains_archive": str(phase102.get("status") or "").startswith("archive_"),
        "phase102_product_gate_false": phase102.get("product_gate_qualified") is False,
        "session_audit_passed": audit.get("passed") is True,
        "session_count_20": payload.get("session_count") == 20,
        "total_calls_120": payload.get("total_model_call_budget") == 120,
        "dpo_adapter_available_as_archived_candidate": adapter.is_dir(),
    }
    freeze = {
        "kind": "phase103_pre_acceptance_freeze",
        "created_at": _utcnow(),
        "passed": all(checks.values()),
        "checks": checks,
        "session_manifest_sha256": payload["manifest_sha256"],
        "source_sha256": _source_hashes(),
        "variants": ["base", "dpo"],
        "model_calls_per_variant": 60,
        "phase103_model_call_budget": 120,
        "cumulative_model_call_budget": 240,
        "long_run_total_call_budget": 270,
        "dpo_adapter_path": str(adapter),
        "dpo_adapter_sha256": _sha256(adapter / "adapter_model.safetensors"),
        "simulated_usage": True,
        "actual_user_feedback_count": 0,
        "product_gate_qualified": False,
        "automatic_promotion_allowed": False,
    }
    _write_json(PREPARATION_ROOT / "sessions.json", payload)
    _write_json(PREPARATION_ROOT / "session_isolation_audit.json", audit)
    _write_json(PHASE_ROOT / "pre_acceptance_freeze.json", freeze)
    print(json.dumps({"status": "ready" if freeze["passed"] else "blocked", "checks": checks}, ensure_ascii=False, indent=2))
    return 0 if freeze["passed"] else 2


def _eval_freeze_check(variant: str, adapter_path: Path | None) -> dict[str, Any]:
    freeze = _read_json(PHASE_ROOT / "pre_acceptance_freeze.json")
    sessions = _read_json(PREPARATION_ROOT / "sessions.json")
    checks = {
        "pre_acceptance_freeze_passed": freeze.get("passed") is True,
        "source_files_unchanged": _source_hashes() == freeze.get("source_sha256"),
        "session_manifest_unchanged": stable_hash(sessions.get("sessions") or []) == freeze.get("session_manifest_sha256"),
        "variant_frozen": variant in {"base", "dpo"},
        "adapter_available_or_base": adapter_path is None or adapter_path.is_dir(),
        "adapter_hash_unchanged": variant == "base" or _sha256(adapter_path / "adapter_model.safetensors") == freeze.get("dpo_adapter_sha256"),
        "no_completed_eval_exists": not (EVAL_ROOT / variant / "metrics.json").exists(),
    }
    return {"kind": "phase103_eval_freeze_check", "variant": variant, "passed": all(checks.values()), "checks": checks}


def _evaluate(variant: str, clean: bool) -> int:
    if variant not in {"base", "dpo"}:
        raise SystemExit("Phase103 variant must be base or dpo")
    adapter_path = None if variant == "base" else _dpo_adapter_dir()
    output_root = EVAL_ROOT / variant
    if clean and output_root.exists():
        _safe_clean(output_root, EVAL_ROOT)
    cache_path = PRIVATE_ROOT / f"{variant}.jsonl"
    if clean:
        cache_path.unlink(missing_ok=True)
    freeze = _eval_freeze_check(variant, adapter_path)
    _write_json(output_root / "freeze_check.json", freeze)
    if not freeze["passed"]:
        return 2
    sessions = [dict(row) for row in _read_json(PREPARATION_ROOT / "sessions.json").get("sessions") or []]
    rows = []
    scores = []
    private_rows = []
    torch = tokenizer = model = device = None
    try:
        torch, tokenizer, model, device = _load_runtime(adapter_path)
        for index, session in enumerate(sessions, start=1):
            structural, private = _run_session(session=session, torch=torch, tokenizer=tokenizer, model=model, device=device)
            structural["kind"] = "phase103_structural_session"
            outputs = [str(row.get("raw_output") or "") for row in private.get("turns") or []]
            user_score = score_phase103_session(session=session, outputs=outputs, structural_turns=structural.get("turns") or [])
            structural["simulated_user_score"] = user_score
            rows.append(structural)
            scores.append(user_score)
            private_rows.append({
                "session_id": session.get("session_id"),
                "category": session.get("category"),
                "user_goal": session.get("user_goal"),
                "user_correction": session.get("user_correction"),
                "continuation_request": session.get("continuation_request"),
                "turns": private.get("turns") or [],
                "final_acceptance": user_score.get("accepted"),
            })
            _write_jsonl(output_root / "structural_sessions.jsonl", rows)
            _write_jsonl(output_root / "simulated_user_scores.jsonl", scores)
            _write_private_jsonl(cache_path, private_rows)
            print(f"[phase103:{variant}] {index}/{len(sessions)} {session.get('session_id')} accepted={user_score['accepted']}", flush=True)
    finally:
        if torch is not None and model is not None and device is not None:
            del model
            if device == "mps":
                torch.mps.empty_cache()
    metrics = aggregate_phase103_scores(scores)
    payload = {
        "kind": "phase103_variant_metrics",
        "variant": variant,
        "model_call_count": sum(int(row.get("turn_count") or 0) for row in rows),
        "metrics": metrics,
        "adapter_loaded": adapter_path is not None,
        "adapter_is_archived_candidate": variant == "dpo",
        "guided_generation_used": False,
        "private_cache": str(cache_path),
        "private_cache_outside_repo": True,
        "simulated_usage": True,
        "actual_user_feedback_count": 0,
    }
    _write_json(output_root / "metrics.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _decide() -> int:
    base_payload = _read_json(EVAL_ROOT / "base/metrics.json")
    adapter_payload = _read_json(EVAL_ROOT / "dpo/metrics.json")
    base_scores = _read_jsonl(EVAL_ROOT / "base/simulated_user_scores.jsonl")
    adapter_scores = _read_jsonl(EVAL_ROOT / "dpo/simulated_user_scores.jsonl")
    base = dict(base_payload.get("metrics") or {})
    adapter = dict(adapter_payload.get("metrics") or {})
    paired = compare_phase103_variants(base_scores, adapter_scores)
    decision = build_phase103_decision(base_metrics=base, adapter_metrics=adapter, paired=paired)
    decision.update({
        "base_metrics": base,
        "archived_dpo_metrics": adapter,
        "paired_comparison": paired,
        "phase103_model_call_count": 120,
        "cumulative_model_call_count": 240,
        "long_run_total_call_budget": 270,
    })
    _write_json(PHASE_ROOT / "phase103-decision.json", decision)
    lines = [
        "# Phase103 Decision",
        "",
        f"- Status: `{decision['status']}`",
        f"- Passed: {str(decision['passed']).lower()}",
        f"- Recommendation: `{decision['recommendation']}`",
        "- Sessions: 20 paired, three turns each",
        "- Evidence: simulated_usage only",
        "- Product gate qualified: false",
        "- Automatic promotion allowed: false",
    ]
    (PHASE_ROOT / "phase103-decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return 0 if decision["passed"] else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--clean", action="store_true")
    evaluate = sub.add_parser("eval")
    evaluate.add_argument("--variant", required=True)
    evaluate.add_argument("--clean", action="store_true")
    sub.add_parser("decide")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "prepare":
        return _prepare(args.clean)
    if args.command == "eval":
        return _evaluate(args.variant, args.clean)
    if args.command == "decide":
        return _decide()
    raise SystemExit("unsupported command")


if __name__ == "__main__":
    raise SystemExit(main())
