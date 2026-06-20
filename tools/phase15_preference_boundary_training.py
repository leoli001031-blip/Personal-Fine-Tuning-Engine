#!/usr/bin/env python3
"""Run Phase15 true-preference boundary training probes."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import shutil
from typing import Any, Iterable, Mapping

from pfe_core.errors import TrainingError
from pfe_core.inference.contracts import score_boundary_contract_output
from pfe_core.trainer.executors import execute_dpo_training, probe_trainer_executor


PHASE15_DOCS_DIR = Path("docs/demo/phase15-true-preference-boundary-training")
PHASE14_DOCS_DIR = Path("docs/demo/phase14-hard-negative-boundary-training")
PHASE13_DOCS_DIR = Path("docs/demo/phase13-boundary-contract-runtime-and-trainable-probe")


def _load_local_tool(module_name: str, filename: str) -> Any:
    path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if not spec or not spec.loader:
        raise RuntimeError(f"cannot load helper from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


phase14 = _load_local_tool("phase14_hard_negative_boundary_training", "phase14_hard_negative_boundary_training.py")


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


def _score_text(text: str, *, expected_citation: str, source_excerpt: str) -> dict[str, Any]:
    return score_boundary_contract_output(text, expected_citation=expected_citation, allowed_context=source_excerpt)


def _load_or_build_phase14_source(phase14_evidence_dir: Path, *, candidate_count: int, holdout_count: int) -> dict[str, Any]:
    candidate_path = phase14_evidence_dir / "candidate_samples.jsonl"
    pair_path = phase14_evidence_dir / "preference_pairs.jsonl"
    holdout_path = phase14_evidence_dir / "holdout.json"
    if not candidate_path.exists() or not pair_path.exists() or not holdout_path.exists():
        phase14.build_phase14_dataset(evidence_dir=phase14_evidence_dir, candidate_count=candidate_count, holdout_count=holdout_count)
    candidates = _read_jsonl(candidate_path)
    pairs = _read_jsonl(pair_path)
    holdout = _read_json(holdout_path)
    return {
        "candidate_samples": candidates,
        "preference_pairs": pairs,
        "holdout": holdout,
        "source_paths": {
            "candidate_samples": str(candidate_path),
            "preference_pairs": str(pair_path),
            "holdout": str(holdout_path),
        },
    }


def preference_pair_quality_check(sample: Mapping[str, Any], *, holdout_chunk_ids: set[str]) -> dict[str, Any]:
    base_check = phase14.sample_quality_check(sample, holdout_chunk_ids=holdout_chunk_ids)
    metadata = _dict(sample.get("metadata"))
    expected_citation = str(metadata.get("expected_citation") or "")
    source_excerpt = str(metadata.get("source_excerpt") or "")
    rejected = str(sample.get("rejected") or "")
    rejected_scores = _score_text(rejected, expected_citation=expected_citation, source_excerpt=source_excerpt)
    reasons = list(base_check.get("reasons") or [])
    if sample.get("sample_type") != "dpo":
        reasons.append("sample_type_not_dpo")
    if not rejected.strip():
        reasons.append("missing_rejected_answer")
    if not (
        rejected_scores.get("external_law_reference")
        or rejected_scores.get("legal_conclusion")
        or int(rejected_scores.get("unsupported_assertions", 0)) > 0
        or float(rejected_scores.get("safety_boundary_passed", 1.0)) == 0.0
    ):
        reasons.append("rejected_answer_not_preference_negative")
    return {
        "sample_id": sample.get("sample_id"),
        "passed": not reasons,
        "reasons": sorted(set(str(reason) for reason in reasons)),
        "chosen_scores": base_check.get("chosen_scores"),
        "rejected_scores": rejected_scores,
    }


def build_phase15_preference_dataset(
    *,
    evidence_dir: Path,
    phase14_evidence_dir: Path,
    candidate_count: int = 120,
    holdout_count: int = 80,
    pair_limit: int | None = None,
) -> dict[str, Any]:
    evidence_dir.mkdir(parents=True, exist_ok=True)
    source = _load_or_build_phase14_source(phase14_evidence_dir, candidate_count=candidate_count, holdout_count=holdout_count)
    candidates = [dict(item) for item in source["candidate_samples"]]
    pairs = [dict(item) for item in source["preference_pairs"]]
    holdout = _dict(source["holdout"])
    holdouts = [dict(item) for item in holdout.get("prompts") or [] if isinstance(item, Mapping)]
    holdout_chunk_ids = {str(item.get("chunk_id")) for item in holdouts if item.get("chunk_id")}
    candidates_by_id = {str(item.get("sample_id")): item for item in candidates if item.get("sample_id")}

    dpo_samples: list[dict[str, Any]] = []
    quality_checks: list[dict[str, Any]] = []
    selected_pairs = pairs[:pair_limit] if pair_limit is not None else pairs
    for index, pair in enumerate(selected_pairs, start=1):
        source_sample = candidates_by_id.get(str(pair.get("sample_id"))) or {}
        source_metadata = _dict(source_sample.get("metadata"))
        dataset_split = str(source_metadata.get("dataset_split") or ("train" if index / max(len(selected_pairs), 1) <= 0.85 else "val"))
        metadata = {
            **source_metadata,
            "phase": "phase15",
            "dataset_split": dataset_split,
            "eligible_for_training": True,
            "response_contract": source_metadata.get("response_contract"),
            "training_strategy": "true_preference_dpo_boundary_pair",
            "source_phase": "phase14",
            "source_pair_id": pair.get("pair_id"),
            "rejected_is_training_negative": True,
            "signal_quality": {"confidence": 0.99},
        }
        dpo_sample = {
            "sample_id": f"phase15-dpo-boundary-{index:03d}",
            "sample_type": "dpo",
            "instruction": str(pair.get("prompt") or source_sample.get("instruction") or ""),
            "chosen": str(pair.get("chosen") or source_sample.get("chosen") or ""),
            "rejected": str(pair.get("rejected") or source_sample.get("rejected") or ""),
            "score": 0.99,
            "source": "phase15_true_preference_boundary_pair",
            "source_event_ids": list(source_sample.get("source_event_ids") or [pair.get("pair_id"), pair.get("sample_id")]),
            "metadata": metadata,
        }
        check = preference_pair_quality_check(dpo_sample, holdout_chunk_ids=holdout_chunk_ids)
        quality_checks.append(check)
        if check["passed"]:
            dpo_samples.append(dpo_sample)

    split_counts = Counter(str(_dict(item.get("metadata")).get("dataset_split")) for item in dpo_samples)
    category_counts = Counter(str(_dict(item.get("metadata")).get("hard_negative_category")) for item in dpo_samples)
    rejected_failure_counts = Counter()
    for check in quality_checks:
        rejected_scores = _dict(check.get("rejected_scores"))
        if rejected_scores.get("external_law_reference"):
            rejected_failure_counts["external_law_reference"] += 1
        if rejected_scores.get("legal_conclusion"):
            rejected_failure_counts["legal_conclusion"] += 1
        if int(rejected_scores.get("unsupported_assertions", 0)) > 0:
            rejected_failure_counts["unsupported_assertions"] += 1
        if float(rejected_scores.get("safety_boundary_passed", 1.0)) == 0.0:
            rejected_failure_counts["safety_boundary_failed"] += 1

    _write_jsonl(evidence_dir / "dpo_samples.jsonl", dpo_samples)
    _write_jsonl(evidence_dir / "preference_pairs.jsonl", selected_pairs)
    _write_json(evidence_dir / "holdout.json", holdout)
    source_manifest = {
        "kind": "phase15_source_manifest",
        "source_phase": "phase14",
        "source_paths": dict(source["source_paths"]),
        "candidate_source_count": len(candidates),
        "source_preference_pair_count": len(pairs),
        "holdout_count": len(holdouts),
        "external_legal_sources_allowed": False,
        "created_at": _utcnow_iso(),
    }
    _write_json(evidence_dir / "source_manifest.json", source_manifest)
    quality_report = {
        "kind": "phase15_dpo_preference_quality_report",
        "source_preference_pair_count": len(pairs),
        "selected_preference_pair_count": len(selected_pairs),
        "dpo_sample_count": len(dpo_samples),
        "quality_check_count": len(quality_checks),
        "failed_quality_count": len([item for item in quality_checks if not item.get("passed")]),
        "split_counts": dict(sorted(split_counts.items())),
        "hard_negative_categories": dict(sorted(category_counts.items())),
        "rejected_failure_counts": dict(sorted(rejected_failure_counts.items())),
        "holdout_chunk_ids": sorted(holdout_chunk_ids),
        "training_strategy": "true_preference_dpo_boundary_pairs",
        "meets_quality_goal": len(dpo_samples) >= 80 and not [item for item in quality_checks if not item.get("passed")],
        "created_at": _utcnow_iso(),
    }
    _write_json(evidence_dir / "quality_report.json", quality_report)
    _write_json(evidence_dir / "quality_checks.json", {"checks": quality_checks, "created_at": _utcnow_iso()})
    return {
        "source_manifest": source_manifest,
        "quality_report": quality_report,
        "dpo_samples": {"path": str(evidence_dir / "dpo_samples.jsonl"), "count": len(dpo_samples)},
        "holdout": {"path": str(evidence_dir / "holdout.json"), "count": len(holdouts), "not_for_training": True},
    }


def dpo_dependency_preflight() -> dict[str, Any]:
    modules = ("torch", "transformers", "peft", "accelerate", "trl", "datasets")
    availability = {module: _module_available(module) for module in modules}
    try:
        strict_probe = probe_trainer_executor("dpo", allow_mock_fallback=False).to_dict()
        probe_ready = bool(strict_probe.get("ready"))
        probe_error = None
    except TrainingError as exc:
        strict_probe = {}
        probe_ready = False
        probe_error = str(exc)
    missing_modules = [module for module, available in availability.items() if not available]
    return {
        "kind": "phase15_dpo_dependency_preflight",
        "module_availability": availability,
        "missing_modules": missing_modules,
        "strict_probe_ready": probe_ready,
        "strict_probe": strict_probe,
        "strict_probe_error": probe_error,
        "ready": not missing_modules and probe_ready,
        "created_at": _utcnow_iso(),
    }


def build_dpo_job_spec(
    *,
    samples: list[Mapping[str, Any]],
    base_model: str,
    output_dir: Path,
    epochs: int,
    beta: float,
    max_length: int,
    max_prompt_length: int,
    sample_limit: int,
) -> dict[str, Any]:
    train_examples = [
        {
            "sample_id": item.get("sample_id"),
            "instruction": item.get("instruction"),
            "chosen": item.get("chosen"),
            "rejected": item.get("rejected"),
            "sample_type": "dpo",
        }
        for item in samples[:sample_limit]
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
                "method": "qlora",
                "epochs": epochs,
                "train_type": "dpo",
                "base_model": base_model,
                "num_train_samples": len(train_examples),
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
        "training_examples": train_examples,
        "phase15": {
            "training_strategy": "true_preference_dpo_boundary_pairs",
            "source_phase": "phase14",
        },
    }


def run_dpo_dry_run(*, evidence_dir: Path, job_spec: Mapping[str, Any]) -> dict[str, Any]:
    dry_run = execute_dpo_training(job_spec=job_spec, dry_run=True)
    payload = {
        "kind": "phase15_dpo_dry_run_plan",
        "dry_run_result": dry_run,
        "job_spec": dict(job_spec),
        "created_at": _utcnow_iso(),
    }
    _write_json(evidence_dir / "dpo_dry_run_plan.json", payload)
    return payload


def run_real_dpo_attempt(*, evidence_dir: Path, job_spec: Mapping[str, Any], preflight: Mapping[str, Any], run_real_dpo: bool) -> dict[str, Any]:
    if not run_real_dpo:
        payload = {
            "kind": "phase15_real_dpo_training_attempt",
            "real_training": "not_started",
            "training_run": False,
            "skip_reason": "skip_real_dpo",
            "created_at": _utcnow_iso(),
        }
        _write_json(evidence_dir / "training_attempt.json", payload)
        _write_json(evidence_dir / "train_log.json", payload)
        return payload
    if not preflight.get("ready"):
        payload = {
            "kind": "phase15_real_dpo_training_attempt",
            "real_training": "blocked",
            "training_run": False,
            "blocked_reason": "dpo_runtime_dependencies_not_ready",
            "missing_modules": list(preflight.get("missing_modules") or []),
            "preflight": dict(preflight),
            "created_at": _utcnow_iso(),
        }
        _write_json(evidence_dir / "training_attempt.json", payload)
        _write_json(evidence_dir / "train_log.json", payload)
        return payload
    result = execute_dpo_training(job_spec={**dict(job_spec), "dry_run": False}, dry_run=False)
    payload = {
        "kind": "phase15_real_dpo_training_attempt",
        "real_training": "completed" if result.get("status") == "completed" else "failed",
        "training_run": True,
        "result": result,
        "adapter_path": result.get("artifact_dir"),
        "created_at": _utcnow_iso(),
    }
    _write_json(evidence_dir / "training_attempt.json", payload)
    _write_json(evidence_dir / "train_log.json", payload)
    return payload


def phase15_decision(*, quality_report: Mapping[str, Any], preflight: Mapping[str, Any], training_attempt: Mapping[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    if not quality_report.get("meets_quality_goal"):
        reasons.append("dpo_preference_dataset_quality_goal_not_met")
    if not preflight.get("ready"):
        reasons.append("dpo_runtime_dependencies_not_ready")
    if training_attempt.get("real_training") != "completed":
        reasons.append("real_dpo_training_not_completed")
    if reasons:
        return {
            "kind": "phase15_adapter_decision",
            "status": "blocked",
            "recommendation": "archive",
            "promotion_allowed": False,
            "auto_promotion_allowed": False,
            "manual_review_required": False,
            "reasons": sorted(set(reasons)),
            "created_at": _utcnow_iso(),
        }
    return {
        "kind": "phase15_adapter_decision",
        "status": "pass_requires_eval",
        "recommendation": "promote_after_manual_review",
        "promotion_allowed": False,
        "auto_promotion_allowed": False,
        "manual_review_required": True,
        "reasons": ["real_dpo_training_completed", "adapter_eval_required_before_any_product_use"],
        "created_at": _utcnow_iso(),
    }


def _load_phase_references(phase13_dir: Path, phase14_dir: Path) -> dict[str, Any]:
    return {
        "phase13_summary": _read_json(phase13_dir / "comparison_summary.json"),
        "phase14_summary": _read_json(phase14_dir / "comparison_summary.json"),
        "phase14_final_decision": (phase14_dir / "phase14-final-decision.md").read_text(encoding="utf-8")[:1600]
        if (phase14_dir / "phase14-final-decision.md").exists()
        else "",
        "created_at": _utcnow_iso(),
    }


def _write_runbook(docs_dir: Path) -> str:
    text = """# Phase15 True-Preference Boundary Training Runbook

Phase15 turns Phase14 rejected hard negatives into real DPO-shaped preference pairs. It does not treat rejected answers as side evidence anymore: every training row has `sample_type=dpo`, `chosen`, and `rejected`.

## Default Smoke

```bash
.venv/bin/python tools/phase15_preference_boundary_training.py \\
  --evidence-dir docs/demo/phase15-true-preference-boundary-training/evidence \\
  --clean-evidence \\
  --skip-real-dpo
```

## Strict DPO Preflight

```bash
.venv/bin/python tools/phase15_preference_boundary_training.py \\
  --evidence-dir docs/demo/phase15-true-preference-boundary-training/evidence-real-dpo-preflight \\
  --clean-evidence \\
  --run-real-dpo
```

If `trl` or `datasets` are missing, archive with blocked evidence instead of falling back to SFT or mock training.
"""
    path = docs_dir / "phase15-runbook.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return str(path)


def _write_final_decision(docs_dir: Path, report: Mapping[str, Any]) -> str:
    quality = _dict(report.get("dataset")).get("quality_report") or {}
    preflight = _dict(report.get("dpo_preflight"))
    training = _dict(report.get("training_attempt"))
    decision = _dict(report.get("decision"))
    text = (
        "# Phase15 Final Decision\n\n"
        "## Goal\n\n"
        "- Upgrade Phase14 hard-negative pairs from SFT side evidence into true DPO-shaped preference data.\n"
        "- Do not claim adapter improvement unless real DPO training and eval complete.\n\n"
        "## Dataset\n\n"
        f"- DPO sample count: {quality.get('dpo_sample_count')}\n"
        f"- Source preference pair count: {quality.get('source_preference_pair_count')}\n"
        f"- Meets quality goal: {quality.get('meets_quality_goal')}\n"
        f"- Rejected failure counts: `{json.dumps(quality.get('rejected_failure_counts') or {}, ensure_ascii=False, sort_keys=True)}`\n\n"
        "## DPO Runtime Preflight\n\n"
        f"- Ready: {preflight.get('ready')}\n"
        f"- Missing modules: {preflight.get('missing_modules')}\n"
        f"- Strict probe error: {preflight.get('strict_probe_error')}\n\n"
        "## Training\n\n"
        f"- Real training: {training.get('real_training')}\n"
        f"- Training run: {training.get('training_run')}\n"
        f"- Blocked reason: {training.get('blocked_reason')}\n\n"
        "## Decision\n\n"
        f"- Recommendation: {decision.get('recommendation')}\n"
        f"- Status: {decision.get('status')}\n"
        f"- Reasons: {decision.get('reasons')}\n\n"
        "Phase15 archives unless real DPO dependencies, real DPO training, and adapter eval all pass. Runtime boundary contract remains the product path until then.\n"
    )
    path = docs_dir / "phase15-final-decision.md"
    path.write_text(text, encoding="utf-8")
    return str(path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Phase15 true-preference boundary training probes.")
    parser.add_argument("--evidence-dir", type=Path, default=PHASE15_DOCS_DIR / "evidence")
    parser.add_argument("--phase13-dir", type=Path, default=PHASE13_DOCS_DIR)
    parser.add_argument("--phase14-dir", type=Path, default=PHASE14_DOCS_DIR)
    parser.add_argument("--phase14-evidence-dir", type=Path, default=PHASE14_DOCS_DIR / "evidence-real-qwen3-8b-hard-negative-v2")
    parser.add_argument("--clean-evidence", action="store_true")
    parser.add_argument("--skip-real-dpo", action="store_true")
    parser.add_argument("--run-real-dpo", action="store_true")
    parser.add_argument("--candidate-count", type=int, default=120)
    parser.add_argument("--holdout-count", type=int, default=80)
    parser.add_argument("--pair-limit", type=int, default=120)
    parser.add_argument("--train-sample-limit", type=int, default=80)
    parser.add_argument("--base-model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--training-output-dir", type=Path, default=Path("trainer_job_outputs/phase15-dpo-boundary"))
    parser.add_argument("--dpo-epochs", type=int, default=1)
    parser.add_argument("--dpo-beta", type=float, default=0.1)
    parser.add_argument("--dpo-max-length", type=int, default=1024)
    parser.add_argument("--dpo-max-prompt-length", type=int, default=768)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    evidence_dir = args.evidence_dir.expanduser().resolve()
    docs_dir = evidence_dir.parent if evidence_dir.name.startswith("evidence") else evidence_dir
    if evidence_dir.exists() and args.clean_evidence:
        shutil.rmtree(evidence_dir)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)
    _write_runbook(docs_dir)

    references = _load_phase_references(args.phase13_dir.expanduser().resolve(), args.phase14_dir.expanduser().resolve())
    dataset = build_phase15_preference_dataset(
        evidence_dir=evidence_dir,
        phase14_evidence_dir=args.phase14_evidence_dir.expanduser().resolve(),
        candidate_count=args.candidate_count,
        holdout_count=args.holdout_count,
        pair_limit=args.pair_limit,
    )
    dpo_samples = _read_jsonl(evidence_dir / "dpo_samples.jsonl")
    preflight = dpo_dependency_preflight()
    _write_json(evidence_dir / "dpo_preflight.json", preflight)
    job_spec = build_dpo_job_spec(
        samples=dpo_samples,
        base_model=args.base_model,
        output_dir=args.training_output_dir.expanduser().resolve(),
        epochs=args.dpo_epochs,
        beta=args.dpo_beta,
        max_length=args.dpo_max_length,
        max_prompt_length=args.dpo_max_prompt_length,
        sample_limit=args.train_sample_limit,
    )
    _write_json(evidence_dir / "dpo_job_spec.json", job_spec)
    dry_run = run_dpo_dry_run(evidence_dir=evidence_dir, job_spec=job_spec)
    training_attempt = run_real_dpo_attempt(
        evidence_dir=evidence_dir,
        job_spec=job_spec,
        preflight=preflight,
        run_real_dpo=bool(args.run_real_dpo and not args.skip_real_dpo),
    )
    decision = phase15_decision(
        quality_report=_dict(dataset.get("quality_report")),
        preflight=preflight,
        training_attempt=training_attempt,
    )
    _write_json(evidence_dir / "decision.json", decision)
    comparison = {
        "kind": "phase15_true_preference_boundary_training_summary",
        "phase_references": references,
        "dataset": dataset,
        "dpo_preflight": preflight,
        "dpo_dry_run": dry_run,
        "training_attempt": training_attempt,
        "decision": decision,
        "created_at": _utcnow_iso(),
    }
    _write_json(evidence_dir / "comparison_summary.json", comparison)
    _write_json(docs_dir / "comparison_summary.json", comparison)
    final_decision = _write_final_decision(docs_dir, comparison)
    comparison["phase15_final_decision_path"] = final_decision
    _write_json(evidence_dir / "comparison_summary.json", comparison)
    _write_json(docs_dir / "comparison_summary.json", comparison)
    print(json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
