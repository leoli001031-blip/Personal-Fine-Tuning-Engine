#!/usr/bin/env python3
"""Run a real Phase44 Qwen3-4B completion-only LoRA probe."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import shutil
import sys
import time
import traceback
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = REPO_ROOT / "pfe-core"
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

from pfe_core.adapter_store.quality import validate_adapter_artifact
from pfe_core.phase44_preference_curriculum import build_phase44_sft_job_spec
from pfe_core.trainer.executors import _encode_sft_examples, _run_real_local_peft_training


EVIDENCE_ROOT = REPO_ROOT / "docs" / "demo" / "phase44-preference-curriculum-privacy-safe-retraining"
TRAINING_EVIDENCE = EVIDENCE_ROOT / "evidence-training-sft"
CANDIDATE_PATH = EVIDENCE_ROOT / "evidence-curriculum" / "selected_preference_pairs.jsonl"
SCORER_FREEZE_PATH = EVIDENCE_ROOT / "evidence-scorer-calibration" / "scorer_freeze.json"
SCORER_SOURCE = CORE_ROOT / "pfe_core" / "phase44_preference_curriculum.py"
MODEL_PATH = REPO_ROOT / "models" / "Qwen3-4B"


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return dict(value) if isinstance(value, Mapping) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _completion_boundary_report(job_spec: Mapping[str, Any]) -> dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), local_files_only=True)
    maximum = int(dict(dict(job_spec.get("recipe") or {}).get("training") or {}).get("max_length") or 384)
    examples = list(job_spec.get("training_examples") or [])
    rows = _encode_sft_examples(
        tokenizer=tokenizer,
        training_examples=examples,
        max_length=maximum,
        vocab_size=int(getattr(tokenizer, "vocab_size", 0) or 151936),
    )
    details = []
    for source, encoded in zip(examples, rows):
        labels = list(encoded.get("labels") or [])
        details.append({
            "sample_id": source.get("sample_id"),
            "taxonomy_dimension": source.get("taxonomy_dimension"),
            "completion_label_token_count": sum(int(value) != -100 for value in labels),
            "masked_or_padding_token_count": sum(int(value) == -100 for value in labels),
        })
    minimum = min((row["completion_label_token_count"] for row in details), default=0)
    return {
        "kind": "phase44_sft_completion_boundary_report",
        "passed": len(rows) == len(examples) and bool(details) and minimum >= 8,
        "source_sample_count": len(examples),
        "encoded_sample_count": len(rows),
        "max_length": maximum,
        "minimum_completion_label_token_count": minimum,
        "prompt_tokens_use_loss": False,
        "completion_tokens_use_loss": True,
        "details": details,
    }


def _scorer_freeze_check() -> dict[str, Any]:
    freeze = _read_json(SCORER_FREEZE_PATH)
    current = _sha256(SCORER_SOURCE)
    expected = str(freeze.get("source_sha256") or "")
    return {
        "kind": "phase44_scorer_freeze_check",
        "passed": bool(expected) and expected == current and freeze.get("calibration_status") == "passed",
        "expected_sha256": expected,
        "current_sha256": current,
        "calibration_status": freeze.get("calibration_status"),
        "checked_at": _utcnow(),
    }


def _coverage_report(real: Mapping[str, Any], *, source_count: int, requested_steps: int) -> dict[str, Any]:
    samples = dict(real.get("sample_exposure_counts") or {})
    categories = dict(real.get("category_exposure_counts") or {})
    sample_values = [int(value) for value in samples.values()]
    category_values = [int(value) for value in categories.values()]
    full = len(samples) == source_count and all(value >= 1 for value in sample_values)
    return {
        "kind": "phase44_actual_exposure_report",
        "requested_steps": requested_steps,
        "sampling_strategy": real.get("sampling_strategy"),
        "source_sample_count": source_count,
        "actual_step_count": real.get("steps"),
        "sample_exposure_counts": dict(sorted(samples.items())),
        "category_exposure_counts": dict(sorted(categories.items())),
        "unique_samples_exposed": len(samples),
        "unique_categories_exposed": len(categories),
        "minimum_sample_exposure": min(sample_values) if sample_values else 0,
        "maximum_sample_exposure": max(sample_values) if sample_values else 0,
        "category_exposure_spread": max(category_values) - min(category_values) if category_values else None,
        "full_coverage": full,
        "eligible_as_final_candidate": requested_steps >= source_count and full and len(categories) == 10,
    }


def _update_index(attempt: Mapping[str, Any]) -> None:
    path = TRAINING_EVIDENCE / "probe_index.json"
    current = _read_json(path) if path.exists() else {"kind": "phase44_sft_probe_index", "probes": {}}
    probes = dict(current.get("probes") or {})
    probes[str(attempt.get("requested_steps"))] = dict(attempt)
    current["probes"] = probes
    current["updated_at"] = _utcnow()
    _write_json(path, current)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    steps = max(1, int(args.steps))
    probe_dir = TRAINING_EVIDENCE / f"probe-{steps}step"
    output_dir = REPO_ROOT / "trainer_job_outputs" / f"phase44-qwen3-4b-sft-{steps}step"
    if args.clean and probe_dir.exists():
        sampler_plan = _read_json(probe_dir / "sampler_plan.json") if (probe_dir / "sampler_plan.json").exists() else {}
        manifest = _read_json(probe_dir / "training_manifest.json") if (probe_dir / "training_manifest.json").exists() else {}
        shutil.rmtree(probe_dir)
        probe_dir.mkdir(parents=True, exist_ok=True)
        if sampler_plan:
            _write_json(probe_dir / "sampler_plan.json", sampler_plan)
        if manifest:
            _write_json(probe_dir / "training_manifest.json", manifest)
    if args.clean and output_dir.exists():
        shutil.rmtree(output_dir)
    probe_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = _read_jsonl(CANDIDATE_PATH)
    job_spec = build_phase44_sft_job_spec(
        pairs=candidates,
        base_model=str(MODEL_PATH),
        output_dir=str(output_dir),
        max_steps=steps,
    )
    boundary = _completion_boundary_report(job_spec)
    freeze_check = _scorer_freeze_check()
    _write_json(probe_dir / "training_manifest.json", job_spec)
    _write_json(probe_dir / "completion_boundary_report.json", boundary)
    _write_json(probe_dir / "scorer_freeze_check.json", freeze_check)
    if boundary.get("passed") is not True or freeze_check.get("passed") is not True or len(candidates) < 80:
        attempt = {
            "kind": "phase44_qwen3_4b_sft_training_attempt",
            "status": "blocked",
            "requested_steps": steps,
            "reason": "training_preflight_failed",
            "completion_boundary_report": boundary,
            "scorer_freeze_check": freeze_check,
            "approved_candidate_count": len(candidates),
            "actual_product_benefit_claim_allowed": False,
            "auto_promotion_allowed": False,
        }
        _write_json(probe_dir / "training_attempt.json", attempt)
        _update_index(attempt)
        return 2

    started = time.perf_counter()
    started_at = _utcnow()
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    try:
        result = _run_real_local_peft_training(job_spec)
        real = dict(result.get("real_execution") or {})
        artifact_dir = Path(str(real.get("artifact_dir") or ""))
        adapter_path = artifact_dir / "adapter_model.safetensors"
        validation = validate_adapter_artifact(
            artifact_dir, {"artifact_name": adapter_path.name, "artifact_format": "peft_lora"},
        )
        validation.update({
            "sha256": _sha256(adapter_path) if adapter_path.exists() else None,
            "artifact_dir": str(artifact_dir), "adapter_path": str(adapter_path),
            "parameters_updated": real.get("parameters_updated"), "steps": real.get("steps"),
        })
        coverage = _coverage_report(real, source_count=len(candidates), requested_steps=steps)
        completed = (
            result.get("status") == "completed" and real.get("success") is True
            and real.get("parameters_updated") is True and int(real.get("steps") or 0) >= steps
            and validation.get("valid") is True
        )
        if steps >= len(candidates):
            completed = completed and coverage["eligible_as_final_candidate"]
        attempt = {
            "kind": "phase44_qwen3_4b_sft_training_attempt",
            "status": "completed" if completed else "failed",
            "real_training": completed,
            "candidate_eligible": completed and coverage["eligible_as_final_candidate"],
            "model": str(MODEL_PATH), "requested_steps": steps,
            "started_at": started_at, "finished_at": _utcnow(),
            "duration_seconds": round(time.perf_counter() - started, 4),
            "max_rss_before_bytes": rss_before,
            "max_rss_after_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "execution": real, "adapter_validation": validation, "exposure": coverage,
            "actual_user_feedback": False, "simulated_usage": True,
            "actual_product_benefit_claim_allowed": False, "auto_promotion_allowed": False,
        }
        _write_json(probe_dir / "training_attempt.json", attempt)
        _write_json(probe_dir / "adapter_validation.json", validation)
        _write_json(probe_dir / "actual_exposure_report.json", coverage)
        _write_json(probe_dir / "train_log.json", {"loss_history": real.get("loss_history") or []})
        _write_json(probe_dir / "loss_history.json", real.get("loss_history") or [])
        _write_json(probe_dir / "parameter_fingerprint_before_after.json", {
            "before": real.get("parameter_fingerprint_before"), "after": real.get("parameter_fingerprint_after"),
            "parameters_updated": real.get("parameters_updated"),
        })
    except Exception as exc:
        attempt = {
            "kind": "phase44_qwen3_4b_sft_training_attempt", "status": "failed", "real_training": False,
            "candidate_eligible": False, "model": str(MODEL_PATH), "requested_steps": steps,
            "started_at": started_at, "finished_at": _utcnow(),
            "duration_seconds": round(time.perf_counter() - started, 4),
            "max_rss_before_bytes": rss_before,
            "max_rss_after_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "error": f"{exc.__class__.__name__}: {exc}", "traceback": traceback.format_exc(),
            "actual_product_benefit_claim_allowed": False, "auto_promotion_allowed": False,
        }
        _write_json(probe_dir / "training_attempt.json", attempt)
        _write_json(probe_dir / "train_log.json", attempt)

    _update_index(attempt)
    _write_json(TRAINING_EVIDENCE / "latest_training_attempt.json", attempt)
    print(json.dumps({key: attempt.get(key) for key in ("status", "requested_steps", "candidate_eligible", "duration_seconds", "error")}, ensure_ascii=False, indent=2))
    return 0 if attempt.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
