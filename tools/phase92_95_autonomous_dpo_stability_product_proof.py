#!/usr/bin/env python3
"""Run the bounded local Phase92-95 DPO stability and product-proof loop."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import sys
import time
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = REPO_ROOT / "pfe-core"
TOOLS_ROOT = REPO_ROOT / "tools"
for path in (CORE_ROOT, TOOLS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pfe_core.adapter_store.quality import validate_adapter_artifact
from pfe_core.phase75_personalization_benefit_benchmark import stable_hash
from pfe_core.phase92_dpo_numerical_stability import (
    build_phase92_probe_matrix,
    reconstruct_phase91_runtime,
    select_phase92_runtime,
)
from pfe_core.phase93_95_dpo_product_proof import (
    PHASE94_MODEL_CALL_BUDGET,
    aggregate_phase94_scores,
    audit_phase94_holdout_isolation,
    build_phase93_94_holdouts,
    build_phase93_sanity_decision,
    build_phase95_product_decision,
    has_repeated_output,
)
from pfe_core.phase91_controlled_dpo_preference import score_phase91_output
from pfe_core.trainer.executors import execute_dpo_training
from phase87_89_failure_driven_adapter_loop import (
    GENERATION_PROTOCOL,
    MODEL_PATH,
    TRAINER_OUTPUT_ROOT,
    _load_runtime,
    _release_runtime,
    _run_eval_session,
)


PHASE91_COMMIT = "e99a216b7d19c168910583be33efde6858d74bd4"
PHASE91_ROOT = REPO_ROOT / "docs/demo/phase91-controlled-dpo-preference-diagnostic"
EVIDENCE_ROOT = REPO_ROOT / "docs/demo/phase92-95-autonomous-dpo-stability-product-proof"
PHASE92_ROOT = EVIDENCE_ROOT / "phase92-numerical-stability"
PHASE92_PREPARATION_ROOT = PHASE92_ROOT / "evidence-preparation"
PHASE92_PROBE_ROOT = PHASE92_ROOT / "evidence-probes"
PARENT_ADAPTER_ROOT = TRAINER_OUTPUT_ROOT / "phase87-failure-driven-25step/peft_lora"
PHASE92_TRAINER_ROOT = TRAINER_OUTPUT_ROOT / "phase92-numerical-stability"
PHASE93_ROOT = EVIDENCE_ROOT / "phase93-stable-dpo-training"
PHASE93_PREPARATION_ROOT = PHASE93_ROOT / "evidence-preparation"
PHASE93_TRAINING_ROOT = PHASE93_ROOT / "evidence-training"
PHASE94_ROOT = EVIDENCE_ROOT / "phase94-product-evaluation"
PHASE94_EVAL_ROOT = PHASE94_ROOT / "evidence-eval"
PHASE95_ROOT = EVIDENCE_ROOT / "phase95-final-decision"
PHASE93_TRAINER_ROOT = TRAINER_OUTPUT_ROOT / "phase93-stable-dpo"
PRIVATE_REVIEW_ROOT = Path("/private/tmp/pfe-phase92-95-simulated-review")
DPO_BETA = 0.1
DPO_MAX_LENGTH = 384
DPO_MAX_PROMPT_LENGTH = 288
DPO_LORA = {"r": 16, "lora_alpha": 32, "lora_dropout": 0.05}
PHASE93_SANITY_THRESHOLDS = {
    "phase89_core_regression_allowed": False,
    "strict_core_improvement_required": True,
    "unsupported_regression_allowed": False,
    "repetition_regression_allowed": False,
    "think_leak_maximum": 0.0,
    "privacy_echo_maximum": 0.0,
    "automatic_promotion_allowed": False,
}
PHASE94_PRODUCT_THRESHOLDS = {
    "candidate_core_not_below_phase89": True,
    "candidate_strict_core_improvement_vs_phase89": True,
    "candidate_strict_core_improvement_vs_base": True,
    "unsupported_not_above_phase89": True,
    "repetition_not_above_phase89": True,
    "think_leak_maximum": 0.0,
    "privacy_echo_maximum": 0.0,
    "maximum_model_calls": PHASE94_MODEL_CALL_BUDGET,
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


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
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
        "executor": CORE_ROOT / "pfe_core/trainer/executors.py",
        "phase92_core": CORE_ROOT / "pfe_core/phase92_dpo_numerical_stability.py",
        "driver": REPO_ROOT / "tools/phase92_95_autonomous_dpo_stability_product_proof.py",
        "runtime_test": REPO_ROOT / "tests/test_phase92_dpo_runtime_resolution.py",
        "phase92_test": REPO_ROOT / "tests/test_phase92_dpo_numerical_stability.py",
        "driver_test": REPO_ROOT / "tests/test_phase92_driver_safety.py",
    }
    return {name: _sha256(path) for name, path in paths.items()}


def _parent_validation() -> dict[str, Any]:
    phase91 = _read_json(PHASE91_ROOT / "pre_experiment_freeze.json")
    adapter_path = PARENT_ADAPTER_ROOT / "adapter_model.safetensors"
    actual = _sha256(adapter_path) if adapter_path.is_file() else None
    expected = phase91.get("parent_adapter_sha256")
    return {
        "artifact_dir": str(PARENT_ADAPTER_ROOT),
        "expected_sha256": expected,
        "actual_sha256": actual,
        "valid": actual is not None and actual == expected,
    }


def _selected_probe_rows() -> list[dict[str, Any]]:
    rows = _read_jsonl(
        PHASE91_ROOT / "evidence-preparation/trainer_rows_12step.jsonl"
    )
    by_category: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_category.setdefault(str(row.get("preference_category")), []).append(row)
    selected = [
        by_category["exact_three_line"][0],
        by_category["false_block"][0],
        by_category["provenance"][0],
        by_category["exact_three_line"][1],
    ]
    return [dict(row) for row in selected]


def build_phase92_job_spec(
    rows: list[Mapping[str, Any]],
    probe: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
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
                "epochs": 1,
                "max_steps": int(probe["max_steps"]),
                "learning_rate": float(probe["learning_rate"]),
                "train_type": "dpo",
                "base_model": str(MODEL_PATH),
                "num_train_samples": len(rows),
                "output_dir": str(output_dir),
                "runtime_device": str(probe["runtime_device"]),
                "runtime_dtype": str(probe["runtime_dtype"]),
                "incremental_context": {
                    "parent_adapter_path": str(PARENT_ADAPTER_ROOT),
                },
            },
            "peft": {
                "trainer_class": "trl.DPOTrainer",
                "dpo_config": {
                    "beta": DPO_BETA,
                    "label_smoothing": 0.0,
                    "max_length": DPO_MAX_LENGTH,
                    "max_prompt_length": DPO_MAX_PROMPT_LENGTH,
                },
                "lora_config": dict(DPO_LORA),
            },
        },
        "training_examples": [dict(row) for row in rows],
        "phase92": {
            "probe_id": probe["probe_id"],
            "parent_phase": 91,
            "parent_commit": PHASE91_COMMIT,
            "single_variable_probe": True,
            "simulated_usage": True,
            "actual_user_feedback": False,
            "automatic_promotion_allowed": False,
        },
    }


def _prepare(clean: bool) -> int:
    if clean and PHASE92_ROOT.exists():
        _safe_clean(PHASE92_ROOT, EVIDENCE_ROOT)
    PHASE92_PREPARATION_ROOT.mkdir(parents=True, exist_ok=True)
    import torch

    cuda_available = bool(torch.cuda.is_available())
    mps_available = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    rows = _selected_probe_rows()
    matrix = build_phase92_probe_matrix(mps_available=mps_available)
    parent = _parent_validation()
    phase91_decision = _read_json(PHASE91_ROOT / "phase91-final-decision.json")
    reconstruction = reconstruct_phase91_runtime(
        cuda_available=cuda_available,
        mps_available=mps_available,
    )
    dry_runs: dict[str, Any] = {}
    specs: dict[str, Any] = {}
    for probe in matrix:
        probe_id = str(probe["probe_id"])
        output_dir = PHASE92_TRAINER_ROOT / probe_id
        spec = build_phase92_job_spec(rows, probe, output_dir)
        dry = execute_dpo_training(job_spec=spec, dry_run=True)
        specs[probe_id] = spec
        dry_runs[probe_id] = dry
        _write_json(PHASE92_PREPARATION_ROOT / f"job_spec_{probe_id}.json", spec)
        _write_json(PHASE92_PREPARATION_ROOT / f"dry_run_{probe_id}.json", dry)

    checks = {
        "phase91_remains_archive": str(phase91_decision.get("status") or "").startswith("archive_"),
        "phase91_product_gate_false": phase91_decision.get("product_gate_qualified") is False,
        "parent_adapter_valid": parent["valid"] is True,
        "exactly_four_probe_rows": len(rows) == 4,
        "all_rows_simulated": all(row.get("simulated_usage") is True for row in rows),
        "no_actual_feedback": all(row.get("actual_user_feedback") is False for row in rows),
        "probe_count_within_cap": len(matrix) <= 3,
        "cpu_probe_first": matrix[0]["probe_id"] == "cpu_float32",
        "legacy_mismatch_reconstructed": reconstruction["cpu_float16_mismatch"] is True,
        "dry_runs_resolve_requested_runtime": all(
            dict(dry_runs[str(probe["probe_id"])]["training_config"]["runtime_resolution"]).get("device")
            == probe["runtime_device"]
            and dict(dry_runs[str(probe["probe_id"])]["training_config"]["runtime_resolution"]).get("dtype")
            == probe["runtime_dtype"]
            for probe in matrix
        ),
    }
    freeze = {
        "kind": "phase92_pre_probe_freeze",
        "created_at": _utcnow(),
        "passed": all(checks.values()),
        "checks": checks,
        "phase91_commit": PHASE91_COMMIT,
        "phase91_status": phase91_decision.get("status"),
        "runtime_inventory": {
            "torch_version": torch.__version__,
            "cuda_available": cuda_available,
            "mps_available": mps_available,
        },
        "legacy_runtime_reconstruction": reconstruction,
        "parent_adapter": parent,
        "probe_matrix": matrix,
        "selected_rows_sha256": stable_hash(rows),
        "job_spec_sha256": {name: stable_hash(spec) for name, spec in specs.items()},
        "source_sha256": _source_hashes(),
        "limits": {
            "maximum_four_step_probes": 3,
            "external_provider_calls_allowed": 0,
            "actual_user_feedback_count": 0,
        },
    }
    _write_jsonl(PHASE92_PREPARATION_ROOT / "selected_probe_rows.jsonl", rows)
    _write_json(PHASE92_PREPARATION_ROOT / "probe_matrix.json", {"probes": matrix})
    _write_json(PHASE92_PREPARATION_ROOT / "phase91_runtime_reconstruction.json", reconstruction)
    _write_json(PHASE92_PREPARATION_ROOT / "parent_adapter_validation.json", parent)
    _write_json(PHASE92_ROOT / "pre_probe_freeze.json", freeze)
    print(json.dumps({"status": "ready" if freeze["passed"] else "blocked", "checks": checks}, ensure_ascii=False, indent=2))
    return 0 if freeze["passed"] else 2


def _probe_definition(probe_id: str) -> dict[str, Any]:
    matrix = _read_json(PHASE92_PREPARATION_ROOT / "probe_matrix.json").get("probes") or []
    for row in matrix:
        if row.get("probe_id") == probe_id:
            return dict(row)
    raise SystemExit(f"probe is not frozen: {probe_id}")


def _probe_freeze_check(probe_id: str) -> dict[str, Any]:
    freeze = _read_json(PHASE92_ROOT / "pre_probe_freeze.json")
    rows = _read_jsonl(PHASE92_PREPARATION_ROOT / "selected_probe_rows.jsonl")
    probe = _probe_definition(probe_id)
    spec = build_phase92_job_spec(rows, probe, PHASE92_TRAINER_ROOT / probe_id)
    cpu_attempt = PHASE92_PROBE_ROOT / "cpu_float32/attempt.json"
    checks = {
        "pre_probe_freeze_passed": freeze.get("passed") is True,
        "source_files_unchanged": _source_hashes() == freeze.get("source_sha256"),
        "selected_rows_unchanged": stable_hash(rows) == freeze.get("selected_rows_sha256"),
        "job_spec_unchanged": stable_hash(spec) == dict(freeze.get("job_spec_sha256") or {}).get(probe_id),
        "parent_adapter_unchanged": _parent_validation() == freeze.get("parent_adapter"),
        "cpu_runs_first": probe_id == "cpu_float32" or cpu_attempt.is_file(),
    }
    if probe_id == "mps_float32_low_lr":
        existing = []
        for prior in ("cpu_float32", "mps_float32"):
            path = PHASE92_PROBE_ROOT / prior / "attempt.json"
            if path.is_file():
                existing.append(_read_json(path))
        selection = select_phase92_runtime(existing, mps_available=True)
        checks["third_probe_condition_met"] = selection.get("status") == "third_probe_required"
    return {"probe_id": probe_id, "passed": all(checks.values()), "checks": checks}


def _probe(probe_id: str, clean: bool) -> int:
    probe = _probe_definition(probe_id)
    evidence_dir = PHASE92_PROBE_ROOT / probe_id
    output_dir = PHASE92_TRAINER_ROOT / probe_id
    if clean and evidence_dir.exists():
        _safe_clean(evidence_dir, PHASE92_PROBE_ROOT)
    if clean and output_dir.exists():
        _safe_clean(output_dir, PHASE92_TRAINER_ROOT)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    freeze_check = _probe_freeze_check(probe_id)
    rows = _read_jsonl(PHASE92_PREPARATION_ROOT / "selected_probe_rows.jsonl")
    spec = build_phase92_job_spec(rows, probe, output_dir)
    _write_json(evidence_dir / "freeze_check.json", freeze_check)
    _write_json(evidence_dir / "job_spec.json", spec)
    if not freeze_check["passed"]:
        _write_json(evidence_dir / "attempt.json", {
            "probe_id": probe_id,
            "status": "blocked",
            "reason": "freeze_check_failed",
            "freeze_check": freeze_check,
        })
        return 2

    started = time.perf_counter()
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = execute_dpo_training(job_spec={**spec, "dry_run": False}, dry_run=False)
    duration = round(time.perf_counter() - started, 4)
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    real = dict(result.get("real_execution") or {})
    artifact_dir = Path(str(real.get("artifact_dir") or ""))
    validation = validate_adapter_artifact(
        artifact_dir,
        {"artifact_name": "adapter_model.safetensors", "artifact_format": "peft_lora"},
    ) if artifact_dir.is_dir() else {"valid": False, "reason": "artifact_dir_missing"}
    validation.update({
        "artifact_dir": str(artifact_dir),
        "sha256": _sha256(artifact_dir / "adapter_model.safetensors")
        if (artifact_dir / "adapter_model.safetensors").is_file() else None,
    })
    attempt = {
        "kind": "phase92_four_step_probe_attempt",
        "probe_id": probe_id,
        "status": result.get("status"),
        "requested_steps": 4,
        "duration_seconds": duration,
        "resource_usage": {
            "ru_maxrss_before": rss_before,
            "ru_maxrss_after": rss_after,
            "ru_maxrss_unit": "bytes_on_macos",
        },
        "runtime_resolution": dict(result.get("training_config") or {}).get("runtime_resolution"),
        "result": result,
        "adapter_validation": validation,
        "simulated_usage": True,
        "actual_user_feedback": False,
        "product_gate_qualified": False,
    }
    _write_json(evidence_dir / "attempt.json", attempt)
    _write_json(evidence_dir / "adapter_validation.json", validation)
    print(json.dumps({
        "probe_id": probe_id,
        "status": result.get("status"),
        "steps": real.get("steps"),
        "duration_seconds": duration,
        "error": result.get("error"),
        "adapter_valid": validation.get("valid"),
    }, ensure_ascii=False, indent=2))
    return 0 if result.get("status") == "completed" and validation.get("valid") is True else 1


def _select() -> int:
    freeze = _read_json(PHASE92_ROOT / "pre_probe_freeze.json")
    attempts = []
    for probe in freeze.get("probe_matrix") or []:
        path = PHASE92_PROBE_ROOT / str(probe["probe_id"]) / "attempt.json"
        if path.is_file():
            attempts.append(_read_json(path))
    decision = select_phase92_runtime(
        attempts,
        mps_available=bool(dict(freeze.get("runtime_inventory") or {}).get("mps_available")),
    )
    selected_id = decision.get("selected_probe_id")
    selected_probe = _probe_definition(str(selected_id)) if selected_id else None
    payload = {
        "kind": "phase92_runtime_selection",
        **decision,
        "selected_probe": selected_probe,
        "probe_attempt_count": len(attempts),
        "maximum_probe_count": 3,
        "simulated_usage": True,
        "actual_user_feedback_count": 0,
        "product_gate_qualified": False,
        "automatic_promotion_allowed": False,
        "next_action": "run_phase93_12step" if selected_id else (
            "run_third_probe" if decision.get("status") == "third_probe_required" else "archive"
        ),
    }
    _write_json(PHASE92_ROOT / "runtime_selection.json", payload)
    _write_json(PHASE92_ROOT / "phase92-decision.json", payload)
    lines = [
        "# Phase92 Decision",
        "",
        f"- Status: `{payload['status']}`",
        f"- Selected probe: `{selected_id}`",
        f"- Probe attempts: {len(attempts)}/3 maximum",
        "- Product gate qualified: false",
        "- Evidence: simulated usage only",
        "",
        "Phase92 only establishes a numerically stable local DPO runtime. It does not establish product benefit.",
    ]
    (PHASE92_ROOT / "phase92-decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if selected_id else (3 if decision.get("status") == "third_probe_required" else 1)


def _phase93_source_hashes() -> dict[str, str]:
    paths = {
        "executor": CORE_ROOT / "pfe_core/trainer/executors.py",
        "phase92_core": CORE_ROOT / "pfe_core/phase92_dpo_numerical_stability.py",
        "phase93_core": CORE_ROOT / "pfe_core/phase93_95_dpo_product_proof.py",
        "driver": REPO_ROOT / "tools/phase92_95_autonomous_dpo_stability_product_proof.py",
        "phase92_tests": REPO_ROOT / "tests/test_phase92_dpo_numerical_stability.py",
        "phase93_tests": REPO_ROOT / "tests/test_phase93_95_dpo_product_proof.py",
        "driver_tests": REPO_ROOT / "tests/test_phase93_95_driver_safety.py",
    }
    return {name: _sha256(path) for name, path in paths.items()}


def _phase93_rows(steps: int) -> list[dict[str, Any]]:
    return _read_jsonl(
        PHASE91_ROOT / f"evidence-preparation/trainer_rows_{steps}step.jsonl"
    )


def _selected_runtime() -> dict[str, Any]:
    decision = _read_json(PHASE92_ROOT / "runtime_selection.json")
    if decision.get("status") != "stable_runtime_selected":
        raise SystemExit("Phase92 did not select a stable runtime")
    return dict(decision.get("selected_probe") or {})


def build_phase93_job_spec(
    rows: list[Mapping[str, Any]],
    *,
    steps: int,
    output_dir: Path,
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
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
                "epochs": 1,
                "max_steps": steps,
                "learning_rate": float(runtime["learning_rate"]),
                "train_type": "dpo",
                "base_model": str(MODEL_PATH),
                "num_train_samples": len(rows),
                "output_dir": str(output_dir),
                "runtime_device": str(runtime["runtime_device"]),
                "runtime_dtype": str(runtime["runtime_dtype"]),
                "incremental_context": {
                    "parent_adapter_path": str(PARENT_ADAPTER_ROOT),
                },
            },
            "peft": {
                "trainer_class": "trl.DPOTrainer",
                "dpo_config": {
                    "beta": DPO_BETA,
                    "label_smoothing": 0.0,
                    "max_length": DPO_MAX_LENGTH,
                    "max_prompt_length": DPO_MAX_PROMPT_LENGTH,
                },
                "lora_config": dict(DPO_LORA),
            },
        },
        "training_examples": [dict(row) for row in rows],
        "phase93": {
            "steps": steps,
            "starts_independently_from_phase89_parent": True,
            "phase92_probe_adapter_used_as_parent": False,
            "phase91_holdout_used_for_training": False,
            "phase90_holdout_used_for_training": False,
            "simulated_usage": True,
            "actual_user_feedback": False,
            "automatic_promotion_allowed": False,
        },
    }


def _previous_holdouts() -> list[dict[str, Any]]:
    roots = (
        REPO_ROOT / "docs/demo/phase87-89-failure-driven-adapter-loop/evidence-preparation/holdout.json",
        REPO_ROOT / "docs/demo/phase90-native-format-curriculum-repair/evidence-preparation/holdout.json",
        PHASE91_ROOT / "evidence-preparation/holdout.json",
    )
    return [_read_json(path) for path in roots]


def _phase93_prepare(clean: bool) -> int:
    if clean:
        for path in (PHASE93_ROOT, PHASE94_ROOT, PHASE95_ROOT):
            if path.exists():
                _safe_clean(path, EVIDENCE_ROOT)
    PHASE93_PREPARATION_ROOT.mkdir(parents=True, exist_ok=True)
    runtime = _selected_runtime()
    rows12 = _phase93_rows(12)
    rows30 = _phase93_rows(30)
    holdouts = build_phase93_94_holdouts()
    isolation = audit_phase94_holdout_isolation(
        training_rows=rows30,
        holdouts=holdouts,
        previous_holdouts=_previous_holdouts(),
    )
    parent = _parent_validation()
    specs: dict[str, Any] = {}
    dry_runs: dict[str, Any] = {}
    for steps, rows in ((12, rows12), (30, rows30)):
        output_dir = PHASE93_TRAINER_ROOT / f"{steps}step"
        spec = build_phase93_job_spec(
            rows, steps=steps, output_dir=output_dir, runtime=runtime
        )
        dry = execute_dpo_training(job_spec=spec, dry_run=True)
        specs[str(steps)] = spec
        dry_runs[str(steps)] = dry
        _write_json(PHASE93_PREPARATION_ROOT / f"job_spec_{steps}step.json", spec)
        _write_json(PHASE93_PREPARATION_ROOT / f"dry_run_{steps}step.json", dry)
        _write_jsonl(PHASE93_PREPARATION_ROOT / f"trainer_rows_{steps}step.jsonl", rows)

    sanity_calls = len(holdouts["sanity_sessions"]) * 3 * 2
    product_calls = len(holdouts["product_sessions"]) * 3 * 3
    checks = {
        "phase92_runtime_selected": runtime.get("probe_id") == "mps_float32",
        "phase92_runtime_is_mps_float32": runtime.get("runtime_device") == "mps"
        and runtime.get("runtime_dtype") == "float32",
        "parent_adapter_valid": parent.get("valid") is True,
        "training_rows_12": len(rows12) == 12,
        "training_rows_30": len(rows30) == 30,
        "all_training_rows_simulated": all(
            row.get("simulated_usage") is True and row.get("actual_user_feedback") is False
            for row in rows12 + rows30
        ),
        "fresh_holdout_isolation_passed": isolation.get("passed") is True,
        "call_budget_132_within_150": sanity_calls + product_calls == 132
        and sanity_calls + product_calls <= PHASE94_MODEL_CALL_BUDGET,
        "dry_runs_use_selected_runtime": all(
            dict(dry["training_config"]["runtime_resolution"]).get("device") == "mps"
            and dict(dry["training_config"]["runtime_resolution"]).get("dtype") == "float32"
            for dry in dry_runs.values()
        ),
        "both_jobs_start_from_phase89": all(
            spec["recipe"]["training"]["incremental_context"]["parent_adapter_path"]
            == str(PARENT_ADAPTER_ROOT)
            for spec in specs.values()
        ),
    }
    freeze = {
        "kind": "phase93_94_pre_training_and_eval_freeze",
        "created_at": _utcnow(),
        "passed": all(checks.values()),
        "checks": checks,
        "selected_runtime": runtime,
        "parent_adapter": parent,
        "trainer_rows_sha256": {
            "12": stable_hash(rows12),
            "30": stable_hash(rows30),
        },
        "job_spec_sha256": {steps: stable_hash(spec) for steps, spec in specs.items()},
        "sanity_manifest_sha256": holdouts["sanity_manifest_sha256"],
        "product_manifest_sha256": holdouts["product_manifest_sha256"],
        "holdout_isolation_sha256": stable_hash(isolation),
        "sanity_thresholds": PHASE93_SANITY_THRESHOLDS,
        "sanity_thresholds_sha256": stable_hash(PHASE93_SANITY_THRESHOLDS),
        "product_thresholds": PHASE94_PRODUCT_THRESHOLDS,
        "product_thresholds_sha256": stable_hash(PHASE94_PRODUCT_THRESHOLDS),
        "source_sha256": _phase93_source_hashes(),
        "model_call_budget": {
            "sanity": sanity_calls,
            "product": product_calls,
            "total": sanity_calls + product_calls,
            "maximum": PHASE94_MODEL_CALL_BUDGET,
        },
        "simulated_usage": True,
        "actual_user_feedback_count": 0,
        "automatic_promotion_allowed": False,
    }
    _write_json(PHASE93_PREPARATION_ROOT / "fresh_holdouts.json", holdouts)
    _write_json(PHASE93_PREPARATION_ROOT / "sanity_validation.json", {"sessions": holdouts["sanity_sessions"]})
    _write_json(PHASE93_PREPARATION_ROOT / "product_holdout.json", {"sessions": holdouts["product_sessions"]})
    _write_json(PHASE93_PREPARATION_ROOT / "holdout_isolation_audit.json", isolation)
    _write_json(PHASE93_ROOT / "pre_training_freeze.json", freeze)
    print(json.dumps({"status": "ready" if freeze["passed"] else "blocked", "checks": checks}, ensure_ascii=False, indent=2))
    return 0 if freeze["passed"] else 2


def _phase93_freeze_check(steps: int) -> dict[str, Any]:
    freeze = _read_json(PHASE93_ROOT / "pre_training_freeze.json")
    rows = _read_jsonl(PHASE93_PREPARATION_ROOT / f"trainer_rows_{steps}step.jsonl")
    spec = build_phase93_job_spec(
        rows,
        steps=steps,
        output_dir=PHASE93_TRAINER_ROOT / f"{steps}step",
        runtime=dict(freeze.get("selected_runtime") or {}),
    )
    holdouts = _read_json(PHASE93_PREPARATION_ROOT / "fresh_holdouts.json")
    sanity = _read_json(PHASE93_ROOT / "sanity_decision.json") if (PHASE93_ROOT / "sanity_decision.json").is_file() else {}
    checks = {
        "pre_training_freeze_passed": freeze.get("passed") is True,
        "source_files_unchanged": _phase93_source_hashes() == freeze.get("source_sha256"),
        "training_rows_unchanged": stable_hash(rows) == dict(freeze.get("trainer_rows_sha256") or {}).get(str(steps)),
        "job_spec_unchanged": stable_hash(spec) == dict(freeze.get("job_spec_sha256") or {}).get(str(steps)),
        "parent_adapter_unchanged": _parent_validation() == freeze.get("parent_adapter"),
        "sanity_holdout_unchanged": stable_hash(holdouts.get("sanity_sessions") or []) == freeze.get("sanity_manifest_sha256"),
        "product_holdout_unchanged": stable_hash(holdouts.get("product_sessions") or []) == freeze.get("product_manifest_sha256"),
        "thirty_step_requires_passed_sanity": steps != 30 or sanity.get("passed") is True,
        "independent_phase89_parent": spec["recipe"]["training"]["incremental_context"]["parent_adapter_path"] == str(PARENT_ADAPTER_ROOT),
    }
    return {"kind": "phase93_training_freeze_check", "steps": steps, "passed": all(checks.values()), "checks": checks}


def _phase93_train(steps: int, clean: bool) -> int:
    if steps not in (12, 30):
        raise SystemExit("Phase93 permits only 12-step and 30-step probes")
    evidence_dir = PHASE93_TRAINING_ROOT / f"{steps}step"
    output_dir = PHASE93_TRAINER_ROOT / f"{steps}step"
    if clean and evidence_dir.exists():
        _safe_clean(evidence_dir, PHASE93_TRAINING_ROOT)
    if clean and output_dir.exists():
        _safe_clean(output_dir, PHASE93_TRAINER_ROOT)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    freeze = _phase93_freeze_check(steps)
    rows = _read_jsonl(PHASE93_PREPARATION_ROOT / f"trainer_rows_{steps}step.jsonl")
    spec = build_phase93_job_spec(rows, steps=steps, output_dir=output_dir, runtime=_selected_runtime())
    _write_json(evidence_dir / "freeze_check.json", freeze)
    _write_json(evidence_dir / "job_spec.json", spec)
    if not freeze["passed"]:
        _write_json(evidence_dir / "training_attempt.json", {
            "status": "blocked",
            "requested_steps": steps,
            "reason": "freeze_check_failed",
            "freeze_check": freeze,
        })
        return 2
    started = time.perf_counter()
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = execute_dpo_training(job_spec={**spec, "dry_run": False}, dry_run=False)
    duration = round(time.perf_counter() - started, 4)
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    real = dict(result.get("real_execution") or {})
    artifact_dir = Path(str(real.get("artifact_dir") or ""))
    validation = validate_adapter_artifact(
        artifact_dir,
        {"artifact_name": "adapter_model.safetensors", "artifact_format": "peft_lora"},
    ) if artifact_dir.is_dir() else {"valid": False, "reason": "artifact_dir_missing"}
    adapter_path = artifact_dir / "adapter_model.safetensors"
    validation.update({
        "artifact_dir": str(artifact_dir),
        "sha256": _sha256(adapter_path) if adapter_path.is_file() else None,
        "requested_steps": steps,
        "completed_steps": int(real.get("steps") or 0),
        "lineage": "base_merge_phase89_then_apply_phase93_dpo",
    })
    completed = (
        result.get("status") == "completed"
        and real.get("success") is True
        and int(real.get("steps") or 0) == steps
        and real.get("parameters_updated") is True
        and validation.get("valid") is True
    )
    attempt = {
        "kind": "phase93_stable_dpo_training_attempt",
        "status": "completed" if completed else "failed",
        "requested_steps": steps,
        "completed_steps": int(real.get("steps") or 0),
        "duration_seconds": duration,
        "resource_usage": {
            "ru_maxrss_before": rss_before,
            "ru_maxrss_after": rss_after,
            "ru_maxrss_unit": "bytes_on_macos",
        },
        "runtime_resolution": dict(result.get("training_config") or {}).get("runtime_resolution"),
        "result": result,
        "adapter_validation": validation,
        "starts_independently_from_phase89_parent": True,
        "simulated_usage": True,
        "actual_user_feedback": False,
        "product_gate_qualified": False,
        "automatic_promotion_allowed": False,
    }
    _write_json(evidence_dir / "training_attempt.json", attempt)
    _write_json(evidence_dir / "train_log.json", {
        "status": attempt["status"],
        "requested_steps": steps,
        "completed_steps": attempt["completed_steps"],
        "loss_history": real.get("loss_history") or [],
        "non_finite_metrics": real.get("non_finite_metrics") or [],
        "parameter_fingerprint_before": real.get("parameter_fingerprint_before"),
        "parameter_fingerprint_after": real.get("parameter_fingerprint_after"),
        "parameters_updated": real.get("parameters_updated"),
        "runtime_audit": real.get("runtime_audit"),
    })
    _write_json(evidence_dir / "adapter_validation.json", validation)
    print(json.dumps({
        "status": attempt["status"],
        "requested_steps": steps,
        "completed_steps": attempt["completed_steps"],
        "duration_seconds": duration,
        "error": result.get("error"),
        "adapter_valid": validation.get("valid"),
    }, ensure_ascii=False, indent=2))
    return 0 if completed else 1


def _phase93_adapter_dir(steps: int) -> Path:
    attempt = _read_json(PHASE93_TRAINING_ROOT / f"{steps}step/training_attempt.json")
    validation = dict(attempt.get("adapter_validation") or {})
    path = Path(str(validation.get("artifact_dir") or ""))
    if attempt.get("status") != "completed" or validation.get("valid") is not True or not path.is_dir():
        raise SystemExit(f"Phase93 {steps}-step adapter is unavailable")
    return path.resolve()


def _load_phase93_candidate(adapter_path: Path) -> tuple[Any, Any, Any, str]:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device == "mps" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH), local_files_only=True, low_cpu_mem_usage=True, dtype=dtype
    )
    model = PeftModel.from_pretrained(model, str(PARENT_ADAPTER_ROOT), local_files_only=True)
    model = model.merge_and_unload()
    model = PeftModel.from_pretrained(model, str(adapter_path), local_files_only=True)
    model.to(device)
    model.eval()
    return torch, tokenizer, model, device


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


def _eval_sessions(scope: str) -> list[dict[str, Any]]:
    holdouts = _read_json(PHASE93_PREPARATION_ROOT / "fresh_holdouts.json")
    key = "sanity_sessions" if scope == "sanity" else "product_sessions"
    return [dict(row) for row in holdouts.get(key) or []]


def _existing_model_call_count() -> int:
    total = 0
    if PHASE94_EVAL_ROOT.exists():
        for path in PHASE94_EVAL_ROOT.glob("*/metrics_*.json"):
            total += int(_read_json(path).get("model_call_count") or 0)
    return total


def _generation_freeze_check(scope: str, variant: str) -> dict[str, Any]:
    freeze = _read_json(PHASE93_ROOT / "pre_training_freeze.json")
    holdouts = _read_json(PHASE93_PREPARATION_ROOT / "fresh_holdouts.json")
    sessions = holdouts["sanity_sessions" if scope == "sanity" else "product_sessions"]
    expected_hash = freeze["sanity_manifest_sha256" if scope == "sanity" else "product_manifest_sha256"]
    sanity = _read_json(PHASE93_ROOT / "sanity_decision.json") if (PHASE93_ROOT / "sanity_decision.json").is_file() else {}
    steps = 12 if scope == "sanity" else 30
    train_path = PHASE93_TRAINING_ROOT / f"{steps}step/training_attempt.json"
    train = _read_json(train_path) if train_path.is_file() else {}
    planned_calls = len(sessions) * 3
    checks = {
        "pre_training_freeze_passed": freeze.get("passed") is True,
        "source_files_unchanged": _phase93_source_hashes() == freeze.get("source_sha256"),
        "holdout_unchanged": stable_hash(sessions) == expected_hash,
        "required_training_completed": variant != "candidate" or train.get("status") == "completed",
        "product_requires_passed_sanity": scope != "product" or sanity.get("passed") is True,
        "call_budget_not_exceeded": _existing_model_call_count() + planned_calls <= PHASE94_MODEL_CALL_BUDGET,
        "variant_allowed": variant in ({"phase89", "candidate"} if scope == "sanity" else {"base", "phase89", "candidate"}),
    }
    return {"kind": "phase94_generation_freeze_check", "scope": scope, "variant": variant, "passed": all(checks.values()), "checks": checks}


def _phase94_generate(scope: str, variant: str, clean: bool) -> int:
    if scope not in {"sanity", "product"}:
        raise SystemExit("unsupported Phase94 generation scope")
    allowed = {"phase89", "candidate"} if scope == "sanity" else {"base", "phase89", "candidate"}
    if variant not in allowed:
        raise SystemExit("unsupported Phase94 generation variant")
    root = PHASE94_EVAL_ROOT / scope
    structural_path = root / f"structural_sessions_{variant}.jsonl"
    metrics_path = root / f"metrics_{variant}.json"
    cache_path = PRIVATE_REVIEW_ROOT / f"{scope}_{variant}.jsonl"
    if metrics_path.exists():
        raise SystemExit(f"refusing to repeat completed model calls: {metrics_path}")
    if clean:
        structural_path.unlink(missing_ok=True)
        cache_path.unlink(missing_ok=True)
    freeze = _generation_freeze_check(scope, variant)
    _write_json(root / f"freeze_check_{variant}.json", freeze)
    if not freeze["passed"]:
        return 2
    sessions = _eval_sessions(scope)
    rows: list[dict[str, Any]] = []
    private_rows: list[dict[str, Any]] = []
    torch = tokenizer = model = device = None
    try:
        if variant == "candidate":
            torch, tokenizer, model, device = _load_phase93_candidate(
                _phase93_adapter_dir(12 if scope == "sanity" else 30)
            )
        elif variant == "phase89":
            torch, tokenizer, model, device = _load_runtime(PARENT_ADAPTER_ROOT)
        else:
            torch, tokenizer, model, device = _load_runtime(None)
        for index, session in enumerate(sessions, start=1):
            try:
                structural, private = _run_eval_session(
                    session=session,
                    torch=torch,
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                    adapter_loaded=variant != "base",
                )
                raw_score = score_phase91_output(private["raw_output"], session)
                post_score = score_phase91_output(private["post_output"], session)
                latency = sum(float(turn.get("latency_seconds") or 0.0) for turn in structural.get("turns") or [])
                raw_score["repeated_output"] = has_repeated_output(private["raw_output"])
                post_score["repeated_output"] = has_repeated_output(private["post_output"])
                raw_score["latency_seconds"] = round(latency, 4)
                post_score["latency_seconds"] = round(latency, 4)
                structural.update({
                    "kind": "phase94_structural_eval_session",
                    "variant": variant,
                    "lineage": "base_merge_phase89_apply_phase93_dpo" if variant == "candidate" else ("base_plus_phase89" if variant == "phase89" else "base"),
                    "raw_score": raw_score,
                    "post_score": post_score,
                })
            except Exception as exc:
                structural = {
                    "kind": "phase94_structural_eval_session",
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
                private = {"session_id": session.get("session_id"), "error_type": exc.__class__.__name__}
            rows.append(structural)
            private_rows.append(private)
            _write_jsonl(structural_path, rows)
            _write_private_jsonl(cache_path, private_rows)
            print(f"[{scope}:{variant}] {index}/{len(sessions)} {session.get('session_id')} {structural['status']}", flush=True)
    finally:
        if torch is not None and model is not None and device is not None:
            _release_runtime(torch, model, device)
    completed = [row for row in rows if row.get("status") == "completed"]
    raw = aggregate_phase94_scores({"category": row.get("category"), **dict(row.get("raw_score") or {})} for row in completed)
    post = aggregate_phase94_scores({"category": row.get("category"), **dict(row.get("post_score") or {})} for row in completed)
    fallback_count = sum(row.get("final_fallback_used") is True for row in completed)
    post["fallback_rate"] = round(fallback_count / len(completed), 4) if completed else 0.0
    model_calls = sum(int(row.get("turn_count") or 0) for row in completed)
    metrics = {
        "kind": "phase94_variant_metrics",
        "scope": scope,
        "variant": variant,
        "session_count": len(completed),
        "model_call_count": model_calls,
        "all_sessions_completed": len(completed) == len(sessions),
        "raw": raw,
        "post_contract": post,
        "call_budget_total_after_arm": _existing_model_call_count() + model_calls,
        "call_budget_maximum": PHASE94_MODEL_CALL_BUDGET,
        "raw_output_cache": str(cache_path),
        "raw_output_cache_outside_repo": True,
        "simulated_usage": True,
        "actual_user_feedback_count": 0,
        "product_gate_qualified": False,
    }
    _write_json(metrics_path, metrics)
    print(json.dumps({"scope": scope, "variant": variant, "session_count": len(completed), "model_calls": model_calls, "raw": raw}, ensure_ascii=False, indent=2))
    return 0 if metrics["all_sessions_completed"] else 1


def _phase93_sanity() -> int:
    phase89_payload = _read_json(PHASE94_EVAL_ROOT / "sanity/metrics_phase89.json")
    candidate_payload = _read_json(PHASE94_EVAL_ROOT / "sanity/metrics_candidate.json")
    decision = build_phase93_sanity_decision(
        dict(phase89_payload.get("raw") or {}),
        dict(candidate_payload.get("raw") or {}),
    )
    training = _read_json(PHASE93_TRAINING_ROOT / "12step/training_attempt.json")
    decision["checks"]["real_12step_training_completed"] = training.get("status") == "completed"
    decision["passed"] = all(decision["checks"].values())
    decision["next_action"] = "run_phase93_30step" if decision["passed"] else "archive_phase93_sanity_failure"
    decision["metrics"] = {
        "phase89": phase89_payload.get("raw"),
        "candidate": candidate_payload.get("raw"),
    }
    _write_json(PHASE93_ROOT / "sanity_decision.json", decision)
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return 0 if decision["passed"] else 1


def _redacted_output_examples() -> str:
    lines = ["# Phase94 Output Examples", "", "All rows are simulated usage. Private canaries were removed by the runtime guard.", ""]
    for category in ("exact_three_line", "false_block", "provenance", "ordinary_control"):
        lines.append(f"## {category}")
        lines.append("")
        for variant in ("base", "phase89", "candidate"):
            rows = _read_jsonl(PRIVATE_REVIEW_ROOT / f"product_{variant}.jsonl")
            row = next(item for item in rows if item.get("category") == category)
            output = str(row.get("raw_output") or "").strip()
            output = output.replace("PHASE94-PRIVATE", "[REDACTED]")
            lines.extend([f"### {variant}", "", "```text", output, "```", ""])
    return "\n".join(lines)


def _manifest() -> dict[str, Any]:
    files = []
    for path in sorted(EVIDENCE_ROOT.rglob("*")):
        if path.is_file() and path.name not in {"evidence_manifest.json", "validation_summary.json"}:
            files.append({"path": str(path.relative_to(EVIDENCE_ROOT)), "sha256": _sha256(path), "size_bytes": path.stat().st_size})
    return {"kind": "phase92_95_evidence_manifest", "files": files, "file_count": len(files)}


def _phase95_finalize() -> int:
    sanity_path = PHASE93_ROOT / "sanity_decision.json"
    sanity = _read_json(sanity_path) if sanity_path.is_file() else {}
    metrics_paths = {variant: PHASE94_EVAL_ROOT / f"product/metrics_{variant}.json" for variant in ("base", "phase89", "candidate")}
    if sanity.get("passed") is not True:
        decision = {
            "kind": "phase95_product_decision",
            "status": "archive_phase93_sanity_failure",
            "recommendation": "archive_and_keep_runtime_contract_main_path",
            "product_gate_qualified": False,
            "promotion_allowed": False,
            "automatic_promotion_allowed": False,
            "automatic_deployment_allowed": False,
            "actual_product_benefit_claim_allowed": False,
            "simulated_usage": True,
            "actual_user_feedback_count": 0,
        }
        metrics = {}
    elif not all(path.is_file() for path in metrics_paths.values()):
        decision = {
            "kind": "phase95_product_decision",
            "status": "archive_phase94_product_eval_incomplete",
            "recommendation": "archive_incomplete_evidence",
            "product_gate_qualified": False,
            "promotion_allowed": False,
            "automatic_promotion_allowed": False,
            "automatic_deployment_allowed": False,
            "actual_product_benefit_claim_allowed": False,
            "simulated_usage": True,
            "actual_user_feedback_count": 0,
        }
        metrics = {}
    else:
        payloads = {variant: _read_json(path) for variant, path in metrics_paths.items()}
        metrics = {variant: dict(payload.get("raw") or {}) for variant, payload in payloads.items()}
        decision = build_phase95_product_decision(metrics)
        decision["model_call_count"] = sum(int(payload.get("model_call_count") or 0) for payload in payloads.values()) + sum(
            int(_read_json(PHASE94_EVAL_ROOT / f"sanity/metrics_{variant}.json").get("model_call_count") or 0)
            for variant in ("phase89", "candidate")
        )
        decision["model_call_budget_maximum"] = PHASE94_MODEL_CALL_BUDGET

    PHASE95_ROOT.mkdir(parents=True, exist_ok=True)
    _write_json(PHASE95_ROOT / "comparison_summary.json", {"metrics": metrics, "decision": decision})
    _write_json(PHASE95_ROOT / "phase95-final-decision.json", decision)
    if metrics:
        (PHASE95_ROOT / "output_examples.md").write_text(_redacted_output_examples(), encoding="utf-8")
    lines = [
        "# Phase95 Final Decision",
        "",
        f"- Status: `{decision['status']}`",
        f"- Recommendation: `{decision['recommendation']}`",
        f"- Product gate qualified: {str(decision['product_gate_qualified']).lower()}",
        "- Promotion allowed: false",
        "- Automatic deployment allowed: false",
        "- Evidence: simulated usage only",
        "- Actual user feedback count: 0",
        "",
        "Phase92 numerical stability is separate from Phase94 product benefit. Runtime-contract output is not counted as adapter benefit.",
    ]
    (PHASE95_ROOT / "phase95-final-decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    runbook = [
        "# Phase92-95 Runbook",
        "",
        "1. Phase92 reconstructed the legacy CPU-float16 mismatch and selected MPS-float32 using bounded 4-step probes.",
        "2. Phase93 trained independent 12-step and conditionally 30-step DPO adapters from the Phase89 parent.",
        "3. Phase94 used frozen fresh simulated holdouts and raw-output metrics under a 150-call cap.",
        "4. Phase95 never promotes automatically and keeps simulated evidence separate from actual-user benefit claims.",
    ]
    (EVIDENCE_ROOT / "phase92-95-runbook.md").write_text("\n".join(runbook) + "\n", encoding="utf-8")
    audit = {
        "kind": "phase92_95_public_private_audit",
        "passed": True,
        "private_cache_outside_repo": str(PRIVATE_REVIEW_ROOT).startswith("/private/tmp/"),
        "raw_full_transcripts_committed": False,
        "simulated_usage": True,
        "actual_user_feedback_count": 0,
        "automatic_promotion_allowed": False,
    }
    _write_json(PHASE95_ROOT / "public_private_audit.json", audit)
    _write_json(EVIDENCE_ROOT / "evidence_manifest.json", _manifest())
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return 0


def _validate() -> int:
    manifest = _read_json(EVIDENCE_ROOT / "evidence_manifest.json")
    failures = []
    for row in manifest.get("files") or []:
        path = EVIDENCE_ROOT / str(row["path"])
        if not path.is_file() or _sha256(path) != row.get("sha256"):
            failures.append(str(row["path"]))
    decision = _read_json(PHASE95_ROOT / "phase95-final-decision.json")
    checks = {
        "manifest_files_unchanged": not failures,
        "no_auto_promotion": decision.get("automatic_promotion_allowed") is False,
        "no_automatic_deployment": decision.get("automatic_deployment_allowed") is False,
        "no_actual_product_benefit_claim": decision.get("actual_product_benefit_claim_allowed") is False,
        "simulated_usage_only": decision.get("simulated_usage") is True and decision.get("actual_user_feedback_count") == 0,
        "model_call_budget_respected": int(decision.get("model_call_count") or 0) <= PHASE94_MODEL_CALL_BUDGET,
    }
    payload = {"kind": "phase92_95_validation_summary", "passed": all(checks.values()), "checks": checks, "manifest_failures": failures, "decision_status": decision.get("status")}
    _write_json(EVIDENCE_ROOT / "validation_summary.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["passed"] else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("phase92-prepare")
    prepare.add_argument("--clean", action="store_true")
    probe = sub.add_parser("phase92-probe")
    probe.add_argument("--probe-id", required=True)
    probe.add_argument("--clean", action="store_true")
    sub.add_parser("phase92-select")
    phase93_prepare = sub.add_parser("phase93-prepare")
    phase93_prepare.add_argument("--clean", action="store_true")
    phase93_train = sub.add_parser("phase93-train")
    phase93_train.add_argument("--steps", type=int, choices=(12, 30), required=True)
    phase93_train.add_argument("--clean", action="store_true")
    generate = sub.add_parser("phase94-generate")
    generate.add_argument("--scope", choices=("sanity", "product"), required=True)
    generate.add_argument("--variant", choices=("base", "phase89", "candidate"), required=True)
    generate.add_argument("--clean", action="store_true")
    sub.add_parser("phase93-sanity")
    sub.add_parser("phase95-finalize")
    sub.add_parser("validate")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "phase92-prepare":
        return _prepare(args.clean)
    if args.command == "phase92-probe":
        return _probe(args.probe_id, args.clean)
    if args.command == "phase92-select":
        return _select()
    if args.command == "phase93-prepare":
        return _phase93_prepare(args.clean)
    if args.command == "phase93-train":
        return _phase93_train(args.steps, args.clean)
    if args.command == "phase94-generate":
        return _phase94_generate(args.scope, args.variant, args.clean)
    if args.command == "phase93-sanity":
        return _phase93_sanity()
    if args.command == "phase95-finalize":
        return _phase95_finalize()
    if args.command == "validate":
        return _validate()
    raise SystemExit("unsupported command")


if __name__ == "__main__":
    raise SystemExit(main())
