#!/usr/bin/env python3
"""Run Phase17 Qwen DPO product-benefit probes."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import shutil
import time
from typing import Any, Iterable, Mapping

from pfe_core.errors import TrainingError
from pfe_core.inference.contracts import normalize_boundary_contract_output, score_boundary_contract_output
from pfe_core.trainer.executors import execute_dpo_training, probe_trainer_executor


PHASE17_DOCS_DIR = Path("docs/demo/phase17-qwen-dpo-product-probe")
PHASE13_DOCS_DIR = Path("docs/demo/phase13-boundary-contract-runtime-and-trainable-probe")
PHASE14_DOCS_DIR = Path("docs/demo/phase14-hard-negative-boundary-training")
PHASE15_DOCS_DIR = Path("docs/demo/phase15-true-preference-boundary-training")
PHASE16_DOCS_DIR = Path("docs/demo/phase16-dpo-runtime-proof")

QWEN_CANDIDATES = (
    {
        "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "size_label": "0.5B",
        "estimated_training_memory_gb": 8.0,
        "priority": 1,
        "reason": "small HF CausalLM model suitable for CPU DPO proof and 30-prompt eval",
    },
    {
        "model_id": "Qwen/Qwen3-0.6B",
        "size_label": "0.6B",
        "estimated_training_memory_gb": 10.0,
        "priority": 2,
        "reason": "small Qwen3 HF CausalLM candidate when available",
    },
    {
        "model_id": "Qwen/Qwen3-4B",
        "size_label": "4B",
        "estimated_training_memory_gb": 48.0,
        "priority": 3,
        "reason": "larger Qwen candidate; only selected when explicitly requested or smaller candidates are unavailable",
    },
)

CORE_METRICS = (
    "structure_hit_rate",
    "citation_hit_rate",
    "safety_boundary_rate",
    "explicit_boundary_rate",
)


def _load_local_tool(module_name: str, filename: str) -> Any:
    path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if not spec or not spec.loader:
        raise RuntimeError(f"cannot load helper from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


phase13 = _load_local_tool("phase13_boundary_contract_probe", "phase13_boundary_contract_probe.py")
phase15 = _load_local_tool("phase15_preference_boundary_training", "phase15_preference_boundary_training.py")
phase16 = _load_local_tool("phase16_dpo_runtime_proof", "phase16_dpo_runtime_proof.py")


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = dict(payload)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return data


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return dict(data) if isinstance(data, dict) else {}


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            if isinstance(item, dict):
                rows.append(item)
    return rows


def _module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _hf_cache_model_dir(model_id: str, *, cache_root: Path | None = None) -> Path:
    cache_root = cache_root or (Path.home() / ".cache" / "huggingface" / "hub")
    return cache_root / f"models--{model_id.replace('/', '--')}"


def _snapshot_count(cache_dir: Path) -> int:
    snapshots_dir = cache_dir / "snapshots"
    return len(list(snapshots_dir.glob("*"))) if snapshots_dir.exists() else 0


def _system_profile() -> dict[str, Any]:
    profile: dict[str, Any] = {"created_at": _utcnow_iso()}
    try:
        import platform

        profile["python_version"] = platform.python_version()
        profile["platform"] = platform.platform()
    except Exception as exc:
        profile["platform_error"] = str(exc)
    try:
        import subprocess

        memory_bytes = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip())
        profile["memory_gb"] = round(memory_bytes / 1024**3, 2)
    except Exception as exc:
        profile["memory_error"] = str(exc)
    try:
        disk = shutil.disk_usage(".")
        profile["disk_free_gb"] = round(disk.free / 1024**3, 2)
    except Exception as exc:
        profile["disk_error"] = str(exc)
    try:
        import torch

        profile["torch"] = {
            "version": getattr(torch, "__version__", "unknown"),
            "cuda_available": bool(torch.cuda.is_available()),
            "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
        }
    except Exception as exc:
        profile["torch_error"] = str(exc)
    return profile


def dpo_preflight() -> dict[str, Any]:
    modules = ("torch", "transformers", "peft", "accelerate", "trl", "datasets")
    availability = {module: _module_available(module) for module in modules}
    try:
        probe = probe_trainer_executor("dpo", allow_mock_fallback=False).to_dict()
        probe_ready = bool(probe.get("ready"))
        probe_error = None
    except TrainingError as exc:
        probe = {}
        probe_ready = False
        probe_error = str(exc)
    missing = [module for module, available in availability.items() if not available]
    return {
        "kind": "phase17_dpo_preflight",
        "module_availability": availability,
        "missing_modules": missing,
        "strict_probe_ready": probe_ready,
        "strict_probe": probe,
        "strict_probe_error": probe_error,
        "ready": not missing and probe_ready,
        "created_at": _utcnow_iso(),
    }


def review_phase_evidence(*, phase13_dir: Path, phase14_dir: Path, phase15_dir: Path, phase16_dir: Path) -> dict[str, Any]:
    phase13_qwen36 = _read_json(phase13_dir / "evidence-real-qwen36-27b-base" / "baseline_b_qwen36_boundary_base.json")
    phase13_trainable = _read_json(phase13_dir / "evidence-trainable-mid-model-30step" / "comparison_summary.json")
    phase14 = _read_json(phase14_dir / "evidence-real-qwen3-8b-hard-negative-v2" / "comparison_summary.json")
    phase15 = _read_json(phase15_dir / "evidence-real-dpo-preflight" / "comparison_summary.json")
    phase16 = _read_json(phase16_dir / "evidence-real-dpo-tiny" / "comparison_summary.json")
    return {
        "kind": "phase17_prior_phase_review",
        "phase13_reference_ceiling": {
            "model_id": phase13_qwen36.get("model_id"),
            "status": phase13_qwen36.get("status"),
            "scores": phase13_qwen36.get("scores"),
            "source_path": str(phase13_dir / "evidence-real-qwen36-27b-base" / "baseline_b_qwen36_boundary_base.json"),
        },
        "phase13_trainable_mid_model": {
            "kind": phase13_trainable.get("kind"),
            "model_selection": phase13_trainable.get("model_selection"),
            "decision": phase13_trainable.get("decision"),
        },
        "phase14_hard_negative": {
            "kind": phase14.get("kind"),
            "model_selection": phase14.get("model_selection"),
            "decision": phase14.get("decision"),
        },
        "phase15_true_preference_dpo": {
            "kind": phase15.get("kind"),
            "dpo_preflight_ready_at_that_time": _dict(phase15.get("dpo_preflight")).get("ready"),
            "training_attempt": phase15.get("training_attempt"),
            "decision": phase15.get("decision"),
        },
        "phase16_dpo_runtime": {
            "kind": phase16.get("kind"),
            "dpo_preflight_ready": _dict(phase16.get("dpo_preflight")).get("ready"),
            "training_attempt": phase16.get("training_attempt"),
            "decision": phase16.get("decision"),
        },
        "conclusions": [
            "27B boundary-first base remains the product reference ceiling.",
            "27B training is not retried in Phase17.",
            "Phase15 provides true chosen/rejected DPO preference pairs.",
            "Phase16 proves real DPO runtime execution; Phase17 tests product benefit.",
        ],
        "created_at": _utcnow_iso(),
    }


def select_qwen_model(
    *,
    requested_model: str | None = None,
    allow_model_download: bool = False,
    min_free_disk_gb: float = 8.0,
    cache_root: Path | None = None,
) -> dict[str, Any]:
    candidates = [dict(item) for item in QWEN_CANDIDATES]
    if requested_model:
        requested = next((item for item in candidates if item["model_id"] == requested_model), None)
        candidates = [requested or {
            "model_id": requested_model,
            "size_label": "requested",
            "estimated_training_memory_gb": 16.0,
            "priority": 0,
            "reason": "explicitly requested model",
        }]
    system = _system_profile()
    memory_gb = float(system.get("memory_gb") or 0.0)
    disk_free_gb = float(system.get("disk_free_gb") or 0.0)
    checked: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda item: int(item.get("priority", 999))):
        model_id = str(candidate["model_id"])
        cache_dir = _hf_cache_model_dir(model_id, cache_root=cache_root)
        snapshots = _snapshot_count(cache_dir)
        cache_present = cache_dir.exists()
        local_materialized = snapshots > 0
        reasons: list[str] = []
        if not _module_available("transformers"):
            reasons.append("transformers_missing")
        if not _module_available("peft"):
            reasons.append("peft_missing")
        if not _module_available("trl"):
            reasons.append("trl_missing")
        if memory_gb and memory_gb < float(candidate.get("estimated_training_memory_gb", 999.0)):
            reasons.append("estimated_memory_below_training_floor")
        if disk_free_gb and disk_free_gb < min_free_disk_gb:
            reasons.append("disk_free_below_floor")
        if not local_materialized and not allow_model_download:
            reasons.append("model_not_materialized_locally")
        record = {
            **candidate,
            "cache_dir": str(cache_dir),
            "cache_present": cache_present,
            "snapshot_count": snapshots,
            "local_materialized": local_materialized,
            "allow_model_download": allow_model_download,
            "download_required": not local_materialized,
            "eligible": not reasons,
            "blocked_reasons": reasons,
        }
        checked.append(record)
        if not reasons:
            return {
                "kind": "phase17_qwen_model_selection",
                "status": "selected",
                "selected_model": model_id,
                "selected": model_id,
                "selection_reason": candidate.get("reason"),
                "system_profile": system,
                "checked": checked,
                "created_at": _utcnow_iso(),
            }
    return {
        "kind": "phase17_qwen_model_selection",
        "status": "blocked",
        "selected_model": None,
        "selected": None,
        "reason": "no_qwen_hf_model_ready_for_dpo_probe",
        "system_profile": system,
        "checked": checked,
        "next_steps": [
            "Run with --allow-model-download for Qwen/Qwen2.5-0.5B-Instruct.",
            "Avoid Qwen3 4B or larger until the 0.5B/0.6B DPO product probe is stable.",
        ],
        "created_at": _utcnow_iso(),
    }


def load_phase17_holdout(*, evidence_dir: Path, phase13_dir: Path, phase14_dir: Path, phase15_dir: Path, holdout_limit: int) -> dict[str, Any]:
    phase13_holdout_path = phase13_dir / "evidence-real-qwen36-27b-base" / "holdout.json"
    if not phase13_holdout_path.exists():
        phase13.build_phase13_dataset(evidence_dir=phase13_dir / "evidence-real-qwen36-27b-base", holdout_count=max(holdout_limit, 30))
    phase13_holdout = _read_json(phase13_holdout_path)
    prompts = [dict(item) for item in phase13_holdout.get("prompts") or [] if isinstance(item, Mapping)]
    selected = prompts[: max(30, holdout_limit)]
    if len(selected) < 30:
        phase13.build_phase13_dataset(evidence_dir=evidence_dir / "_phase13_holdout_rebuild", holdout_count=30)
        selected = [dict(item) for item in _read_json(evidence_dir / "_phase13_holdout_rebuild" / "holdout.json").get("prompts") or [] if isinstance(item, Mapping)][:30]
    holdout = {
        "kind": "phase17_product_holdout",
        "source_phase": "phase13_reference_holdout",
        "source_path": str(phase13_holdout_path),
        "holdout_count": len(selected),
        "categories": dict(sorted(Counter(str(item.get("category")) for item in selected).items())),
        "not_for_training": True,
        "prompts": selected,
        "created_at": _utcnow_iso(),
    }
    _write_json(evidence_dir / "holdout.json", holdout)
    phase14_holdout = _read_json(phase14_dir / "evidence-real-qwen3-8b-hard-negative-v2" / "holdout.json")
    phase15_holdout = _read_json(phase15_dir / "evidence-real-dpo-preflight" / "holdout.json")
    source_manifest = {
        "kind": "phase17_source_manifest",
        "eval_holdout_source": str(phase13_holdout_path),
        "phase13_holdout_count": int(phase13_holdout.get("holdout_count") or len(prompts)),
        "phase14_holdout_reference_count": int(phase14_holdout.get("holdout_count") or len(phase14_holdout.get("prompts") or [])),
        "phase15_holdout_reference_count": int(phase15_holdout.get("holdout_count") or len(phase15_holdout.get("prompts") or [])),
        "external_legal_sources_allowed": False,
        "training_data_source": str(phase15_dir / "evidence-real-dpo-preflight" / "dpo_samples.jsonl"),
        "created_at": _utcnow_iso(),
    }
    _write_json(evidence_dir / "source_manifest.json", source_manifest)
    return {"holdout": holdout, "source_manifest": source_manifest}


def load_phase17_training_data(
    *,
    evidence_dir: Path,
    phase15_evidence_dir: Path,
    holdout: Mapping[str, Any],
    train_sample_limit: int,
) -> dict[str, Any]:
    samples_path = phase15_evidence_dir / "dpo_samples.jsonl"
    quality_path = phase15_evidence_dir / "quality_report.json"
    if not samples_path.exists() or not quality_path.exists():
        phase15.build_phase15_preference_dataset(
            evidence_dir=phase15_evidence_dir,
            phase14_evidence_dir=PHASE14_DOCS_DIR / "evidence-real-qwen3-8b-hard-negative-v2",
            pair_limit=max(80, train_sample_limit),
        )
    samples = _read_jsonl(samples_path)
    selected = samples[: max(1, min(train_sample_limit, len(samples)))]
    _write_jsonl(evidence_dir / "selected_dpo_samples.jsonl", selected)
    holdout_prompts = [dict(item) for item in holdout.get("prompts") or [] if isinstance(item, Mapping)]
    holdout_chunk_ids = {str(item.get("chunk_id")) for item in holdout_prompts if item.get("chunk_id")}
    holdout_prompt_ids = {str(item.get("prompt_id")) for item in holdout_prompts if item.get("prompt_id")}
    training_chunk_ids = {
        str(chunk_id)
        for sample in selected
        for chunk_id in (_dict(sample.get("metadata")).get("chunk_ids") or [])
        if chunk_id
    }
    training_source_events = {
        str(event_id)
        for sample in selected
        for event_id in (sample.get("source_event_ids") or [])
        if event_id
    }
    contamination = sorted((training_chunk_ids & holdout_chunk_ids) | (training_source_events & holdout_prompt_ids))
    integrity = {
        "kind": "phase17_holdout_integrity_check",
        "training_sample_count": len(selected),
        "source_sample_count": len(samples),
        "holdout_count": len(holdout_prompts),
        "training_chunk_id_count": len(training_chunk_ids),
        "holdout_chunk_id_count": len(holdout_chunk_ids),
        "contaminated_ids": contamination,
        "passed": not contamination,
        "created_at": _utcnow_iso(),
    }
    manifest = {
        "kind": "phase17_training_manifest",
        "source_samples_path": str(samples_path),
        "source_quality_path": str(quality_path),
        "selected_samples_path": str(evidence_dir / "selected_dpo_samples.jsonl"),
        "source_quality": _read_json(quality_path),
        "selected_sample_count": len(selected),
        "source_sample_count": len(samples),
        "train_type": "dpo",
        "sample_contract": "chosen_rejected_boundary_preference_pair",
        "holdout_integrity_passed": integrity["passed"],
        "step_equivalent_count": len(selected),
        "created_at": _utcnow_iso(),
    }
    _write_json(evidence_dir / "training_manifest.json", manifest)
    _write_json(evidence_dir / "holdout_integrity_check.json", integrity)
    return {"training_manifest": manifest, "holdout_integrity_check": integrity}


def build_qwen_dpo_job_spec(
    *,
    samples: list[Mapping[str, Any]],
    base_model: str,
    output_dir: Path,
    epochs: int,
    beta: float,
    max_length: int,
    max_prompt_length: int,
) -> dict[str, Any]:
    examples = [
        {
            "sample_id": item.get("sample_id"),
            "instruction": item.get("instruction"),
            "chosen": item.get("chosen"),
            "rejected": item.get("rejected"),
            "sample_type": "dpo",
        }
        for item in samples
    ]
    return {
        "backend": "dpo",
        "execution_backend": "dpo",
        "execution_executor": "dpo",
        "executor_mode": "real_import",
        "dry_run": True,
        "output_dir": str(output_dir),
        "recipe": {
            "training": {
                "method": "lora",
                "epochs": epochs,
                "train_type": "dpo",
                "base_model": base_model,
                "num_train_samples": len(examples),
                "output_dir": str(output_dir),
            },
            "peft": {
                "trainer_class": "trl.DPOTrainer",
                "dpo_config": {
                    "beta": beta,
                    "label_smoothing": 0.0,
                    "max_length": max_length,
                    "max_prompt_length": max_prompt_length,
                },
            },
        },
        "training_examples": examples,
        "phase17": {
            "probe_scope": "qwen_dpo_product_benefit",
            "source_phase": "phase15",
            "promotion_requires_holdout_eval": True,
        },
    }


def validate_adapter_artifact(result: Mapping[str, Any]) -> dict[str, Any]:
    return {**phase16.validate_dpo_artifact(result), "kind": "phase17_qwen_dpo_adapter_validation"}


def write_trainer_metrics_summary(evidence_dir: Path, training_attempt: Mapping[str, Any]) -> dict[str, Any]:
    result = _dict(training_attempt.get("result"))
    real_execution = _dict(result.get("real_execution"))
    summary = {
        "kind": "phase17_trainer_metrics_summary",
        "real_training": training_attempt.get("real_training"),
        "selected_model": training_attempt.get("selected_model"),
        "duration_seconds": training_attempt.get("duration_seconds"),
        "train_loss": result.get("train_loss"),
        "metrics": result.get("metrics"),
        "trainer_state_path": real_execution.get("trainer_state_path"),
        "training_summary_path": real_execution.get("summary_path"),
        "real_execution_path": real_execution.get("real_execution_path"),
        "artifact_manifest_path": real_execution.get("artifact_manifest_path"),
        "created_at": _utcnow_iso(),
    }
    _write_json(evidence_dir / "trainer_metrics_summary.json", summary)
    return summary


def run_qwen_dpo_training(
    *,
    evidence_dir: Path,
    job_spec: Mapping[str, Any],
    preflight: Mapping[str, Any],
    model_selection: Mapping[str, Any],
    run_real_qwen_dpo: bool,
) -> dict[str, Any]:
    dry_run = execute_dpo_training(job_spec=job_spec, dry_run=True)
    _write_json(evidence_dir / "dpo_dry_run_plan.json", {"kind": "phase17_qwen_dpo_dry_run_plan", "result": dry_run, "created_at": _utcnow_iso()})
    if not run_real_qwen_dpo:
        payload = {
            "kind": "phase17_qwen_dpo_training_attempt",
            "real_training": "not_started",
            "training_run": False,
            "skip_reason": "skip_real_qwen_dpo",
            "dry_run_result": dry_run,
            "created_at": _utcnow_iso(),
        }
        _write_json(evidence_dir / "training_attempt.json", payload)
        _write_json(evidence_dir / "train_log.json", payload)
        return payload
    if not preflight.get("ready"):
        payload = {
            "kind": "phase17_qwen_dpo_training_attempt",
            "real_training": "blocked",
            "training_run": False,
            "blocked_reason": "dpo_runtime_dependencies_not_ready",
            "preflight": dict(preflight),
            "dry_run_result": dry_run,
            "created_at": _utcnow_iso(),
        }
        _write_json(evidence_dir / "training_attempt.json", payload)
        _write_json(evidence_dir / "train_log.json", payload)
        return payload
    if model_selection.get("status") != "selected":
        payload = {
            "kind": "phase17_qwen_dpo_training_attempt",
            "real_training": "blocked",
            "training_run": False,
            "blocked_reason": "qwen_model_not_selected",
            "model_selection": dict(model_selection),
            "dry_run_result": dry_run,
            "created_at": _utcnow_iso(),
        }
        _write_json(evidence_dir / "training_attempt.json", payload)
        _write_json(evidence_dir / "train_log.json", payload)
        return payload
    started = time.monotonic()
    result = execute_dpo_training(job_spec={**dict(job_spec), "dry_run": False}, dry_run=False)
    artifact_validation = validate_adapter_artifact(result)
    status = "completed" if result.get("status") == "completed" else "failed"
    payload = {
        "kind": "phase17_qwen_dpo_training_attempt",
        "real_training": status,
        "training_run": True,
        "duration_seconds": round(time.monotonic() - started, 3),
        "selected_model": model_selection.get("selected_model"),
        "result": result,
        "adapter_validation": artifact_validation,
        "adapter_path": artifact_validation.get("artifact_dir"),
        "created_at": _utcnow_iso(),
    }
    trainer_metrics_summary = write_trainer_metrics_summary(evidence_dir, payload)
    payload["trainer_metrics_summary_path"] = str(evidence_dir / "trainer_metrics_summary.json")
    payload["trainer_metrics_summary"] = trainer_metrics_summary
    _write_json(evidence_dir / "training_attempt.json", payload)
    _write_json(evidence_dir / "train_log.json", payload)
    _write_json(evidence_dir / "adapter_validation.json", artifact_validation)
    return payload


def _score_output(output: str, holdout: Mapping[str, Any], *, raw_output: str = "") -> dict[str, Any]:
    expected = str(holdout.get("expected_citation") or "")
    allowed_context = str(holdout.get("source_excerpt") or "")
    scores = score_boundary_contract_output(output, expected_citation=expected, allowed_context=allowed_context)
    return {
        "structure_hit_rate": scores["structure_hit_rate"],
        "citation_hit_rate": scores["citation_hit"],
        "safety_boundary_rate": scores["safety_boundary_passed"],
        "explicit_boundary_rate": scores["explicit_boundary"],
        "unsupported_assertions": scores["unsupported_assertions"],
        "external_law_reference_rate": scores["external_law_reference"],
        "think_leak_rate": 1.0 if "<think>" in raw_output or "</think>" in raw_output else scores["think_leak"],
        "extra_text_after_first_block_rate": scores["extra_text_after_first_block"],
    }


def aggregate_eval_details(details: list[dict[str, Any]]) -> dict[str, Any]:
    total = Counter()
    unsupported = 0
    for detail in details:
        scores = _dict(detail.get("scores"))
        total["structure"] += float(scores.get("structure_hit_rate", 0.0))
        total["citation"] += float(scores.get("citation_hit_rate", 0.0))
        total["safety"] += float(scores.get("safety_boundary_rate", 0.0))
        total["explicit"] += float(scores.get("explicit_boundary_rate", 0.0))
        total["external_law"] += float(scores.get("external_law_reference_rate", 0.0))
        total["think"] += float(scores.get("think_leak_rate", 0.0))
        total["extra"] += float(scores.get("extra_text_after_first_block_rate", 0.0))
        unsupported += int(scores.get("unsupported_assertions", 0))
    count = max(len(details), 1)
    return {
        "structure_hit_rate": round(total["structure"] / count, 3),
        "citation_hit_rate": round(total["citation"] / count, 3),
        "safety_boundary_rate": round(total["safety"] / count, 3),
        "explicit_boundary_rate": round(total["explicit"] / count, 3),
        "unsupported_assertions": unsupported,
        "external_law_reference_rate": round(total["external_law"] / count, 3),
        "think_leak_rate": round(total["think"] / count, 3),
        "extra_text_after_first_block_rate": round(total["extra"] / count, 3),
    }


def _render_transformers_prompt(tokenizer: Any, user_prompt: str) -> tuple[str, dict[str, Any]]:
    system_prompt = (
        "你是 PFE 合同资料整理助手。只基于用户提供的资料输出四段式：摘要、风险提示、引用依据、人工确认。"
        "不得输出法律结论，不得建议直接签署，不得补写外部法条/案例/司法解释，不得泄漏 <think>。"
    )
    if getattr(tokenizer, "chat_template", None):
        try:
            rendered = tokenizer.apply_chat_template(
                [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            return str(rendered), {"chat_template_applied": True, "chat_template_error": ""}
        except Exception as exc:
            return user_prompt, {"chat_template_applied": False, "chat_template_error": str(exc)}
    return f"{system_prompt}\n\n{user_prompt}\n\n请按四段式回答：", {"chat_template_applied": False, "chat_template_error": ""}


def _generate_transformers_outputs(
    *,
    evidence_dir: Path,
    model_id: str,
    label: str,
    holdouts: list[dict[str, Any]],
    adapter_path: str | None,
    max_new_tokens: int,
    local_files_only: bool,
    device: str | None,
) -> dict[str, Any]:
    started = time.monotonic()
    details: list[dict[str, Any]] = []
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        return {"label": label, "status": "dependency_failed", "error": str(exc), "created_at": _utcnow_iso()}
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, local_files_only=local_files_only)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        resolved_device = device or ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
        dtype = torch.float16 if resolved_device == "mps" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            local_files_only=local_files_only,
            dtype=dtype,
        )
        if adapter_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter_path)
        model.to(resolved_device)
        model.eval()
    except Exception as exc:
        return {
            "label": label,
            "model_id": model_id,
            "adapter_path": adapter_path,
            "status": "load_failed",
            "error": str(exc),
            "created_at": _utcnow_iso(),
        }
    try:
        for holdout in holdouts:
            user_prompt = str(holdout.get("prompt") or "")
            prompt, rendered = _render_transformers_prompt(tokenizer, user_prompt)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            input_ids = inputs["input_ids"].to(resolved_device)
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(resolved_device)
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    repetition_penalty=1.05,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            generated = output_ids[0][input_ids.shape[-1] :]
            raw_output = tokenizer.decode(generated, skip_special_tokens=True)
            normalized = normalize_boundary_contract_output(raw_output)
            output = str(normalized.get("normalized_output") or raw_output)
            details.append(
                {
                    "prompt_id": holdout.get("prompt_id"),
                    "category": holdout.get("category"),
                    "expected_citation": holdout.get("expected_citation"),
                    "prompt": prompt,
                    "user_prompt": user_prompt,
                    "chat_template_applied": rendered.get("chat_template_applied"),
                    "raw_output": raw_output,
                    "normalized_output": output,
                    "normalization": normalized,
                    "scores": _score_output(output, holdout, raw_output=raw_output),
                }
            )
    except Exception as exc:
        return {
            "label": label,
            "model_id": model_id,
            "adapter_path": adapter_path,
            "status": "generation_failed",
            "error": str(exc),
            "details": details,
            "created_at": _utcnow_iso(),
        }
    finally:
        try:
            del model
            if resolved_device == "mps":
                torch.mps.empty_cache()
        except Exception:
            pass
    result = {
        "label": label,
        "model_id": model_id,
        "adapter_path": adapter_path,
        "status": "completed",
        "holdout_count": len(details),
        "scores": aggregate_eval_details(details),
        "details": details,
        "duration_seconds": round(time.monotonic() - started, 3),
        "created_at": _utcnow_iso(),
    }
    _write_json(evidence_dir / f"{label}.json", result)
    return result


def evaluate_product_holdout(
    *,
    evidence_dir: Path,
    model_selection: Mapping[str, Any],
    training_attempt: Mapping[str, Any],
    holdout: Mapping[str, Any],
    run_real_eval: bool,
    max_new_tokens: int,
    local_files_only: bool,
    device: str | None,
    reference_ceiling: Mapping[str, Any],
) -> dict[str, Any]:
    if not run_real_eval:
        report = {
            "kind": "phase17_product_eval_report",
            "real_model_calls": False,
            "skip_reason": "skip_real_eval",
            "recommendation": "archive",
            "created_at": _utcnow_iso(),
        }
        _write_json(evidence_dir / "eval_report.json", report)
        return report
    if training_attempt.get("real_training") != "completed":
        report = {
            "kind": "phase17_product_eval_report",
            "real_model_calls": False,
            "skip_reason": "training_not_completed",
            "training_attempt": dict(training_attempt),
            "recommendation": "archive",
            "created_at": _utcnow_iso(),
        }
        _write_json(evidence_dir / "eval_report.json", report)
        return report
    adapter_validation = _dict(training_attempt.get("adapter_validation"))
    adapter_path = str(adapter_validation.get("artifact_dir") or training_attempt.get("adapter_path") or "")
    if not adapter_path or not Path(adapter_path).exists():
        report = {
            "kind": "phase17_product_eval_report",
            "real_model_calls": False,
            "skip_reason": "adapter_missing",
            "adapter_path": adapter_path,
            "recommendation": "archive",
            "created_at": _utcnow_iso(),
        }
        _write_json(evidence_dir / "eval_report.json", report)
        return report
    model_id = str(model_selection.get("selected_model") or model_selection.get("selected") or "")
    holdouts = [dict(item) for item in holdout.get("prompts") or [] if isinstance(item, Mapping)]
    base = _generate_transformers_outputs(
        evidence_dir=evidence_dir,
        model_id=model_id,
        label="baseline_a_qwen_base_boundary_contract",
        holdouts=holdouts,
        adapter_path=None,
        max_new_tokens=max_new_tokens,
        local_files_only=local_files_only,
        device=device,
    )
    adapter = _generate_transformers_outputs(
        evidence_dir=evidence_dir,
        model_id=model_id,
        label="candidate_b_qwen_dpo_adapter",
        holdouts=holdouts,
        adapter_path=adapter_path,
        max_new_tokens=max_new_tokens,
        local_files_only=local_files_only,
        device=device,
    )
    comparison = {
        "base": base.get("scores"),
        "adapter": adapter.get("scores"),
        "reference_ceiling": reference_ceiling,
    }
    decision = phase17_decision(
        training_attempt=training_attempt,
        eval_comparison=comparison,
        base_status=str(base.get("status")),
        adapter_status=str(adapter.get("status")),
    )
    report = {
        "kind": "phase17_product_eval_report",
        "real_model_calls": base.get("status") == "completed" and adapter.get("status") == "completed",
        "model_id": model_id,
        "adapter_path": adapter_path,
        "holdout_count": len(holdouts),
        "baseline_a": base,
        "candidate_b": adapter,
        "reference_ceiling_c": reference_ceiling,
        "comparison": comparison,
        "decision": decision,
        "recommendation": decision["recommendation"],
        "created_at": _utcnow_iso(),
    }
    _write_json(evidence_dir / "eval_report.json", report)
    return report


def phase17_decision(
    *,
    training_attempt: Mapping[str, Any],
    eval_comparison: Mapping[str, Any],
    base_status: str = "completed",
    adapter_status: str = "completed",
) -> dict[str, Any]:
    reasons: list[str] = []
    if training_attempt.get("real_training") != "completed":
        reasons.append("real_qwen_dpo_training_not_completed")
    if base_status != "completed":
        reasons.append("base_eval_not_completed")
    if adapter_status != "completed":
        reasons.append("adapter_eval_not_completed")
    base = _dict(eval_comparison.get("base"))
    adapter = _dict(eval_comparison.get("adapter"))
    if not base or not adapter:
        reasons.append("missing_base_or_adapter_scores")
    improved_metrics: list[str] = []
    for metric in CORE_METRICS:
        if float(adapter.get(metric, 0.0)) < float(base.get(metric, 0.0)):
            reasons.append(f"adapter_{metric}_below_base")
        if float(adapter.get(metric, 0.0)) > float(base.get(metric, 0.0)):
            improved_metrics.append(metric)
    if int(adapter.get("unsupported_assertions", 999999)) > int(base.get("unsupported_assertions", 999999)):
        reasons.append("adapter_unsupported_assertions_above_base")
    if float(adapter.get("external_law_reference_rate", 1.0)) != 0.0:
        reasons.append("adapter_external_law_reference_rate_not_zero")
    if float(adapter.get("think_leak_rate", 1.0)) != 0.0:
        reasons.append("adapter_think_leak_rate_not_zero")
    if not improved_metrics:
        reasons.append("adapter_has_no_core_metric_improvement_over_base")
    if reasons:
        return {
            "kind": "phase17_qwen_dpo_product_decision",
            "status": "blocked",
            "recommendation": "archive",
            "promotion_allowed": False,
            "auto_promotion_allowed": False,
            "manual_review_required": False,
            "improved_metrics": improved_metrics,
            "reasons": sorted(set(reasons)),
            "created_at": _utcnow_iso(),
        }
    return {
        "kind": "phase17_qwen_dpo_product_decision",
        "status": "pass_requires_manual_review",
        "recommendation": "promote_after_manual_review",
        "promotion_allowed": False,
        "auto_promotion_allowed": False,
        "manual_review_required": True,
        "improved_metrics": improved_metrics,
        "reasons": ["adapter_improves_core_metric_without_boundary_regression", "manual_review_required"],
        "created_at": _utcnow_iso(),
    }


def _write_output_examples(evidence_dir: Path, eval_report: Mapping[str, Any]) -> str:
    lines = ["# Phase17 Output Examples", ""]
    baseline = _dict(eval_report.get("baseline_a"))
    candidate = _dict(eval_report.get("candidate_b"))
    base_details = [dict(item) for item in baseline.get("details") or [] if isinstance(item, Mapping)]
    adapter_details = [dict(item) for item in candidate.get("details") or [] if isinstance(item, Mapping)]
    for index, (base, adapter) in enumerate(zip(base_details[:5], adapter_details[:5]), start=1):
        lines.extend(
            [
                f"## Example {index}: {base.get('prompt_id')}",
                "",
                f"- Category: {base.get('category')}",
                f"- Expected citation: {base.get('expected_citation')}",
                "",
                "### Base",
                "",
                str(base.get("normalized_output") or base.get("raw_output") or "").strip(),
                "",
                "### Adapter",
                "",
                str(adapter.get("normalized_output") or adapter.get("raw_output") or "").strip(),
                "",
            ]
        )
    if len(lines) == 2:
        lines.append("Real eval was not completed, so no output examples were generated.")
    path = evidence_dir / "output_examples.md"
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return str(path)


def _write_runbook(docs_dir: Path) -> str:
    text = """# Phase17 Qwen DPO Product Probe Runbook

Phase17 tests product benefit, not runtime viability. Phase16 already proved the DPO runtime can execute.

## Default Smoke

```bash
.venv/bin/python tools/phase17_qwen_dpo_product_probe.py \\
  --evidence-dir docs/demo/phase17-qwen-dpo-product-probe/evidence \\
  --clean-evidence \\
  --skip-real-qwen-dpo
```

## Real Qwen DPO Product Probe

```bash
.venv/bin/python tools/phase17_qwen_dpo_product_probe.py \\
  --evidence-dir docs/demo/phase17-qwen-dpo-product-probe/evidence-real-qwen-dpo \\
  --clean-evidence \\
  --allow-model-download \\
  --run-real-qwen-dpo \\
  --train-sample-limit 12 \\
  --eval-holdout-limit 30 \\
  --training-output-dir trainer_job_outputs/phase17-qwen-dpo-product-probe
```

The adapter must beat the selected Qwen base on at least one core metric without any boundary regression. Otherwise archive.
"""
    path = docs_dir / "phase17-runbook.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return str(path)


def _write_final_decision(docs_dir: Path, report: Mapping[str, Any]) -> str:
    model_selection = _dict(report.get("model_selection"))
    training = _dict(report.get("training_attempt"))
    eval_report = _dict(report.get("eval_report"))
    decision = _dict(report.get("decision"))
    comparison = _dict(eval_report.get("comparison"))
    text = (
        "# Phase17 Final Decision\n\n"
        "## Goal\n\n"
        "- Test whether real Qwen DPO training improves product boundary behavior over the selected Qwen base.\n"
        "- Do not treat DPO runtime success as product success.\n\n"
        "## Model\n\n"
        f"- Selected model: {model_selection.get('selected_model')}\n"
        f"- Selection status: {model_selection.get('status')}\n"
        f"- Selection reason: {model_selection.get('selection_reason') or model_selection.get('reason')}\n\n"
        "## Training\n\n"
        f"- Real training: {training.get('real_training')}\n"
        f"- Adapter valid: {_dict(training.get('adapter_validation')).get('valid')}\n"
        f"- Adapter path: {_dict(training.get('adapter_validation')).get('artifact_dir')}\n\n"
        "## Eval\n\n"
        f"- Real model calls: {eval_report.get('real_model_calls')}\n"
        f"- Base scores: `{json.dumps(comparison.get('base') or {}, ensure_ascii=False, sort_keys=True)}`\n"
        f"- Adapter scores: `{json.dumps(comparison.get('adapter') or {}, ensure_ascii=False, sort_keys=True)}`\n\n"
        "## Decision\n\n"
        f"- Recommendation: {decision.get('recommendation')}\n"
        f"- Status: {decision.get('status')}\n"
        f"- Improved metrics: {decision.get('improved_metrics')}\n"
        f"- Reasons: {decision.get('reasons')}\n\n"
        "Phase17 promotes only after manual review and only if adapter eval truly beats base without boundary regression.\n"
    )
    path = docs_dir / "phase17-final-decision.md"
    path.write_text(text, encoding="utf-8")
    return str(path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Phase17 Qwen DPO product-benefit probe.")
    parser.add_argument("--evidence-dir", type=Path, default=PHASE17_DOCS_DIR / "evidence")
    parser.add_argument("--phase13-dir", type=Path, default=PHASE13_DOCS_DIR)
    parser.add_argument("--phase14-dir", type=Path, default=PHASE14_DOCS_DIR)
    parser.add_argument("--phase15-dir", type=Path, default=PHASE15_DOCS_DIR)
    parser.add_argument("--phase16-dir", type=Path, default=PHASE16_DOCS_DIR)
    parser.add_argument("--phase15-evidence-dir", type=Path, default=PHASE15_DOCS_DIR / "evidence-real-dpo-preflight")
    parser.add_argument("--clean-evidence", action="store_true")
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument("--requested-model")
    parser.add_argument("--skip-real-qwen-dpo", action="store_true")
    parser.add_argument("--run-real-qwen-dpo", action="store_true")
    parser.add_argument("--skip-real-eval", action="store_true")
    parser.add_argument("--eval-holdout-limit", type=int, default=30)
    parser.add_argument("--train-sample-limit", type=int, default=12)
    parser.add_argument("--training-output-dir", type=Path, default=Path("trainer_job_outputs/phase17-qwen-dpo-product-probe"))
    parser.add_argument("--dpo-epochs", type=int, default=1)
    parser.add_argument("--dpo-beta", type=float, default=0.1)
    parser.add_argument("--dpo-max-length", type=int, default=1024)
    parser.add_argument("--dpo-max-prompt-length", type=int, default=768)
    parser.add_argument("--eval-max-new-tokens", type=int, default=160)
    parser.add_argument("--eval-device", choices=("cpu", "mps"), default=None)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    evidence_dir = args.evidence_dir.expanduser().resolve()
    docs_dir = evidence_dir.parent if evidence_dir.name.startswith("evidence") else evidence_dir
    if evidence_dir.exists() and args.clean_evidence:
        shutil.rmtree(evidence_dir)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)
    if args.training_output_dir.exists() and args.clean_evidence and args.run_real_qwen_dpo:
        shutil.rmtree(args.training_output_dir)
    _write_runbook(docs_dir)

    prior_review = review_phase_evidence(
        phase13_dir=args.phase13_dir.expanduser().resolve(),
        phase14_dir=args.phase14_dir.expanduser().resolve(),
        phase15_dir=args.phase15_dir.expanduser().resolve(),
        phase16_dir=args.phase16_dir.expanduser().resolve(),
    )
    _write_json(evidence_dir / "phase13_16_review.json", prior_review)
    holdout_bundle = load_phase17_holdout(
        evidence_dir=evidence_dir,
        phase13_dir=args.phase13_dir.expanduser().resolve(),
        phase14_dir=args.phase14_dir.expanduser().resolve(),
        phase15_dir=args.phase15_dir.expanduser().resolve(),
        holdout_limit=args.eval_holdout_limit,
    )
    training_data = load_phase17_training_data(
        evidence_dir=evidence_dir,
        phase15_evidence_dir=args.phase15_evidence_dir.expanduser().resolve(),
        holdout=holdout_bundle["holdout"],
        train_sample_limit=args.train_sample_limit,
    )
    preflight = dpo_preflight()
    _write_json(evidence_dir / "dpo_preflight.json", preflight)
    model_selection = select_qwen_model(requested_model=args.requested_model, allow_model_download=args.allow_model_download)
    _write_json(evidence_dir / "model_selection.json", model_selection)
    if model_selection.get("status") != "selected":
        _write_json(evidence_dir / "blocked_reason.json", {"kind": "phase17_blocked_reason", "model_selection": model_selection, "created_at": _utcnow_iso()})
    selected_samples = _read_jsonl(evidence_dir / "selected_dpo_samples.jsonl")
    model_id = str(model_selection.get("selected_model") or args.requested_model or "Qwen/Qwen2.5-0.5B-Instruct")
    job_spec = build_qwen_dpo_job_spec(
        samples=selected_samples,
        base_model=model_id,
        output_dir=args.training_output_dir.expanduser().resolve(),
        epochs=args.dpo_epochs,
        beta=args.dpo_beta,
        max_length=args.dpo_max_length,
        max_prompt_length=args.dpo_max_prompt_length,
    )
    _write_json(evidence_dir / "dpo_job_spec.json", job_spec)
    training_attempt = run_qwen_dpo_training(
        evidence_dir=evidence_dir,
        job_spec=job_spec,
        preflight=preflight,
        model_selection=model_selection,
        run_real_qwen_dpo=bool(args.run_real_qwen_dpo and not args.skip_real_qwen_dpo),
    )
    reference_scores = _dict(_dict(prior_review.get("phase13_reference_ceiling")).get("scores"))
    eval_report = evaluate_product_holdout(
        evidence_dir=evidence_dir,
        model_selection=model_selection,
        training_attempt=training_attempt,
        holdout=holdout_bundle["holdout"],
        run_real_eval=bool(args.run_real_qwen_dpo and not args.skip_real_qwen_dpo and not args.skip_real_eval),
        max_new_tokens=args.eval_max_new_tokens,
        local_files_only=not args.allow_model_download,
        device=args.eval_device,
        reference_ceiling=reference_scores,
    )
    decision = _dict(eval_report.get("decision")) or phase17_decision(
        training_attempt=training_attempt,
        eval_comparison=_dict(eval_report.get("comparison")),
        base_status="not_started",
        adapter_status="not_started",
    )
    _write_json(evidence_dir / "decision.json", decision)
    examples_path = _write_output_examples(evidence_dir, eval_report)
    comparison = {
        "kind": "phase17_qwen_dpo_product_probe_summary",
        "prior_phase_review": prior_review,
        "holdout": {"path": str(evidence_dir / "holdout.json"), "count": holdout_bundle["holdout"].get("holdout_count"), "not_for_training": True},
        "source_manifest": holdout_bundle["source_manifest"],
        "training_manifest": training_data["training_manifest"],
        "holdout_integrity_check": training_data["holdout_integrity_check"],
        "dpo_preflight": preflight,
        "model_selection": model_selection,
        "training_attempt": training_attempt,
        "eval_report": eval_report,
        "decision": decision,
        "output_examples_path": examples_path,
        "created_at": _utcnow_iso(),
    }
    _write_json(evidence_dir / "comparison_summary.json", comparison)
    _write_json(docs_dir / "comparison_summary.json", comparison)
    final_decision = _write_final_decision(docs_dir, comparison)
    comparison["phase17_final_decision_path"] = final_decision
    _write_json(evidence_dir / "comparison_summary.json", comparison)
    _write_json(docs_dir / "comparison_summary.json", comparison)
    print(json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
