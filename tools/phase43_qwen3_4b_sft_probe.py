#!/usr/bin/env python3
"""Run an evidenced real Qwen3-4B completion-only LoRA probe for Phase43."""

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
from pfe_core.phase43_personal_preference_benefit import build_phase43_sft_job_spec
from pfe_core.trainer.executors import _encode_sft_examples, _run_real_local_peft_training


EVIDENCE_ROOT = REPO_ROOT / "docs" / "demo" / "phase43-qwen3-4b-personal-preference-benefit-proof"
TRAINING_EVIDENCE = EVIDENCE_ROOT / "evidence-training-sft"
CANDIDATE_PATH = EVIDENCE_ROOT / "evidence-candidate-review" / "selected_preference_pairs.jsonl"
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
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _completion_boundary_report(job_spec: Mapping[str, Any]) -> dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), local_files_only=True)
    max_length = int(dict(dict(job_spec.get("recipe") or {}).get("training") or {}).get("max_length") or 384)
    rows = _encode_sft_examples(
        tokenizer=tokenizer,
        training_examples=list(job_spec.get("training_examples") or []),
        max_length=max_length,
        vocab_size=int(getattr(tokenizer, "vocab_size", 0) or 151936),
    )
    details = []
    for source, encoded in zip(job_spec.get("training_examples") or [], rows):
        label_count = sum(int(value) != -100 for value in encoded.get("labels") or [])
        prompt_mask_count = sum(int(value) == -100 for value in encoded.get("labels") or [])
        details.append(
            {
                "sample_id": source.get("sample_id"),
                "completion_label_token_count": label_count,
                "masked_or_padding_token_count": prompt_mask_count,
            }
        )
    minimum = min((row["completion_label_token_count"] for row in details), default=0)
    return {
        "kind": "phase43_sft_completion_boundary_report",
        "passed": bool(details) and minimum >= 8,
        "sample_count": len(details),
        "max_length": max_length,
        "minimum_completion_label_token_count": minimum,
        "prompt_tokens_use_loss": False,
        "completion_tokens_use_loss": True,
        "details": details,
    }


def _update_probe_index(probe: Mapping[str, Any]) -> None:
    path = TRAINING_EVIDENCE / "probe_index.json"
    current = _read_json(path) if path.exists() else {"kind": "phase43_sft_probe_index", "probes": {}}
    probes = dict(current.get("probes") or {})
    probes[str(probe.get("requested_steps"))] = dict(probe)
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
    output_dir = REPO_ROOT / "trainer_job_outputs" / f"phase43-qwen3-4b-sft-{steps}step"
    if args.clean and probe_dir.exists():
        shutil.rmtree(probe_dir)
    if args.clean and output_dir.exists():
        shutil.rmtree(output_dir)
    probe_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not CANDIDATE_PATH.exists():
        raise SystemExit("Phase43 candidates are missing; run phase43_qwen3_4b_prepare.py first")
    candidates = _read_jsonl(CANDIDATE_PATH)
    job_spec = build_phase43_sft_job_spec(
        pairs=candidates,
        base_model=str(MODEL_PATH),
        output_dir=str(output_dir),
        max_steps=steps,
    )
    boundary = _completion_boundary_report(job_spec)
    _write_json(probe_dir / "training_manifest.json", job_spec)
    _write_json(probe_dir / "completion_boundary_report.json", boundary)
    if boundary.get("passed") is not True:
        attempt = {
            "kind": "phase43_qwen3_4b_sft_training_attempt",
            "status": "blocked",
            "requested_steps": steps,
            "reason": "completion_boundary_preflight_failed",
            "completion_boundary_report": boundary,
            "actual_product_benefit_claim_allowed": False,
            "auto_promotion_allowed": False,
        }
        _write_json(probe_dir / "training_attempt.json", attempt)
        _update_probe_index(attempt)
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
            artifact_dir,
            {"artifact_name": adapter_path.name, "artifact_format": "peft_lora"},
        )
        validation.update(
            {
                "sha256": _sha256(adapter_path) if adapter_path.exists() else None,
                "artifact_dir": str(artifact_dir),
                "adapter_path": str(adapter_path),
                "parameters_updated": real.get("parameters_updated"),
                "steps": real.get("steps"),
            }
        )
        completed = (
            result.get("status") == "completed"
            and real.get("success") is True
            and real.get("parameters_updated") is True
            and int(real.get("steps") or 0) >= steps
            and validation.get("valid") is True
        )
        attempt = {
            "kind": "phase43_qwen3_4b_sft_training_attempt",
            "status": "completed" if completed else "failed",
            "real_training": completed,
            "model": str(MODEL_PATH),
            "requested_steps": steps,
            "started_at": started_at,
            "finished_at": _utcnow(),
            "duration_seconds": round(time.perf_counter() - started, 4),
            "max_rss_before_bytes": rss_before,
            "max_rss_after_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "execution": real,
            "adapter_validation": validation,
            "actual_user_feedback": False,
            "simulated_lab_benefit_not_yet_evaluated": True,
            "actual_product_benefit_claim_allowed": False,
            "auto_promotion_allowed": False,
        }
        _write_json(probe_dir / "training_attempt.json", attempt)
        _write_json(probe_dir / "adapter_validation.json", validation)
        _write_json(probe_dir / "train_log.json", {"loss_history": real.get("loss_history") or []})
        _write_json(probe_dir / "loss_history.json", real.get("loss_history") or [])
        _write_json(
            probe_dir / "parameter_fingerprint_before_after.json",
            {
                "before": real.get("parameter_fingerprint_before"),
                "after": real.get("parameter_fingerprint_after"),
                "parameters_updated": real.get("parameters_updated"),
            },
        )
    except Exception as exc:
        attempt = {
            "kind": "phase43_qwen3_4b_sft_training_attempt",
            "status": "failed",
            "real_training": False,
            "model": str(MODEL_PATH),
            "requested_steps": steps,
            "started_at": started_at,
            "finished_at": _utcnow(),
            "duration_seconds": round(time.perf_counter() - started, 4),
            "max_rss_before_bytes": rss_before,
            "max_rss_after_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "error": f"{exc.__class__.__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "actual_product_benefit_claim_allowed": False,
            "auto_promotion_allowed": False,
        }
        _write_json(probe_dir / "training_attempt.json", attempt)
        _write_json(probe_dir / "train_log.json", attempt)

    _update_probe_index(attempt)
    _write_json(TRAINING_EVIDENCE / "latest_training_attempt.json", attempt)
    print(json.dumps({key: attempt.get(key) for key in ("status", "requested_steps", "duration_seconds", "error")}, indent=2))
    return 0 if attempt.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
