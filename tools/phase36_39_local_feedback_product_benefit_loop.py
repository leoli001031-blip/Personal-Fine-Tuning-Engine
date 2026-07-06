#!/usr/bin/env python3
"""Generate Phase36-39 local feedback product-benefit loop evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import re
import shutil
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
PFE_CORE = ROOT / "pfe-core"
for path in (PFE_CORE,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pfe_core.phase35_local_interaction_capture import load_phase35_state, phase35_store_path
from pfe_core.phase36_39_local_feedback_product_benefit import (
    PHASE39_MODEL_VARIANTS,
    build_phase36_39_comparison_summary,
    build_phase36_39_simulated_lab_records,
    build_phase36_review_decision,
    build_phase36_review_queue,
    build_phase36_review_summary,
    build_phase37_candidate_artifacts,
    build_phase37_holdout,
    build_phase38_model_selection,
    build_phase38_training_attempt,
    build_phase38_training_manifest,
    build_phase39_blind_eval_pairs,
    build_phase39_eval_report,
    build_phase39_simulated_sessions,
    build_phase39_transcripts,
    phase39_final_decision,
    validate_phase39_boundaries,
    write_jsonl,
)
from pfe_core.trainer.executors import execute_peft_training


PHASE35_DIR = Path("docs/demo/phase35-local-interaction-capture")
PHASE36_39_DIR = Path("docs/demo/phase36-39-local-feedback-product-benefit-loop")
_LOCAL_ABS_PATH_RE = re.compile(r"/Users/lichenhao/[^\s\"'，。；;、)）\]]+")


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
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if isinstance(item, Mapping):
            rows.append(dict(item))
    return rows


def _clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _redact_evidence_tree(path: Path) -> None:
    for item in path.rglob("*"):
        if not item.is_file() or item.suffix not in {".json", ".jsonl", ".md", ".txt"}:
            continue
        text = item.read_text(encoding="utf-8")
        redacted = _LOCAL_ABS_PATH_RE.sub("[LOCAL_PATH]", text)
        if redacted != text:
            item.write_text(redacted, encoding="utf-8")


def _discover_local_models() -> list[dict[str, Any]]:
    models_dir = ROOT / "models"
    discovered: list[dict[str, Any]] = []
    if models_dir.exists():
        for config in sorted(models_dir.glob("*/config.json")):
            model_dir = config.parent
            lower = model_dir.name.lower()
            discovered.append(
                {
                    "name": model_dir.name,
                    "path": str(model_dir),
                    "trainable": "qwen" in lower,
                    "quantization": "4bit" if "4bit" in lower else "none",
                    "has_config": True,
                    "safetensor_count": len(list(model_dir.glob("*.safetensors"))),
                }
            )
    return discovered


def _dependency_summary() -> dict[str, Any]:
    modules = ["torch", "transformers", "peft", "accelerate"]
    return {
        "kind": "phase38_dependency_summary",
        "modules": {module: importlib.util.find_spec(module) is not None for module in modules},
        "created_at": _utcnow_iso(),
    }


def _run_phase38_training_probe(
    *,
    evidence_training_dir: Path,
    simulated_candidates: Mapping[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    local_models = _discover_local_models()
    dependency_summary = _dependency_summary()
    model_selection = build_phase38_model_selection(
        local_models=local_models,
        dependency_summary=dependency_summary,
    )
    _write_json(evidence_training_dir / "model_selection.json", model_selection)
    training_manifest = build_phase38_training_manifest(
        candidate_artifacts=simulated_candidates,
        model_selection=model_selection,
        step_count=args.training_steps,
    )
    _write_json(evidence_training_dir / "training_manifest.json", training_manifest)
    selected_pairs = list(simulated_candidates.get("dpo_pairs") or [])[: max(1, args.training_steps)]
    write_jsonl(evidence_training_dir / "selected_training_pairs.jsonl", selected_pairs)

    blocked_reason = None
    execution_result: dict[str, Any] | None = None
    if model_selection.get("status") != "selected":
        blocked_reason = "no_trainable_small_qwen_model_selected"
    elif not selected_pairs:
        blocked_reason = "no_simulated_lab_training_pairs"
    elif args.skip_real_training:
        blocked_reason = "real_training_skipped_by_flag"
    else:
        output_dir = args.training_output_dir.expanduser()
        if args.clean_training_output and output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        training_examples = [
            {
                "sample_id": pair.get("sample_id"),
                "instruction": pair.get("instruction") or pair.get("prompt"),
                "chosen": pair.get("chosen"),
                "metadata": pair.get("metadata"),
            }
            for pair in selected_pairs
        ]
        job_spec = {
            "backend": "peft",
            "base_model": model_selection.get("selected_model"),
            "base_model_path": model_selection.get("selected_model"),
            "local_model_path": model_selection.get("selected_model"),
            "real_local": True,
            "local_only": True,
            "output_dir": str(output_dir),
            "training_examples": training_examples,
            "recipe": {
                "training": {
                    "base_model": model_selection.get("selected_model"),
                    "base_model_path": model_selection.get("selected_model"),
                    "local_model_path": model_selection.get("selected_model"),
                    "local_only": True,
                    "output_dir": str(output_dir),
                    "epochs": 1,
                    "max_steps": args.training_steps,
                    "num_train_samples": len(training_examples),
                }
            },
        }
        _write_json(evidence_training_dir / "phase38_peft_job_spec.json", job_spec)
        execution_result = execute_peft_training(job_spec=job_spec, dry_run=False)
        _write_json(evidence_training_dir / "phase38_peft_execution_result.json", execution_result)
        real_execution = _dict(execution_result.get("real_execution"))
        for key, target_name in (
            ("trainer_state_path", "trainer_state.json"),
            ("summary_path", "training_summary.json"),
            ("real_execution_path", "trainer_real_execution.json"),
            ("artifact_manifest_path", "adapter_artifact_manifest.json"),
        ):
            raw_path = real_execution.get(key)
            if raw_path and Path(str(raw_path)).exists():
                shutil.copyfile(str(raw_path), evidence_training_dir / target_name)

    training_attempt = build_phase38_training_attempt(
        training_manifest=training_manifest,
        execution_result=execution_result,
        blocked_reason=blocked_reason,
    )
    _write_json(evidence_training_dir / "training_attempt.json", training_attempt)
    _write_json(evidence_training_dir / "train_log.json", training_attempt)
    _write_json(evidence_training_dir / "adapter_validation.json", _dict(training_attempt.get("adapter_validation")))
    return model_selection, training_manifest, training_attempt


def _write_output_examples(path: Path, transcripts_by_variant: Mapping[str, list[Mapping[str, Any]]]) -> None:
    lines = ["# Phase39 Output Examples", ""]
    for variant in PHASE39_MODEL_VARIANTS:
        transcript = (transcripts_by_variant.get(variant) or [])[0]
        lines.append(f"## {variant}")
        for turn in transcript.get("turns") or []:
            if _dict(turn).get("role") == "assistant":
                lines.append(str(_dict(turn).get("content") or ""))
                break
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_runbook(path: Path) -> None:
    path.write_text(
        """# Phase36-39 Runbook

Generate the full local feedback product-benefit loop evidence:

```bash
.venv/bin/python tools/phase36_39_local_feedback_product_benefit_loop.py --clean-evidence
```

This run does not integrate Hermes and does not train 27B. It separates the actual feedback lane from the simulated lab lane. Without at least 12 approved actual local interactions, product-benefit claims remain lab-only.
""",
        encoding="utf-8",
    )


def _write_final_decision(path: Path, decision: Mapping[str, Any]) -> None:
    path.write_text(
        f"""# Phase39 Final Decision

## Decision

- Recommendation: {decision.get("recommendation")}
- Evidence type: {decision.get("evidence_type")}
- Actual product benefit claim allowed: {decision.get("actual_product_benefit_claim_allowed")}
- Auto promotion allowed: {decision.get("auto_promotion_allowed")}
- Actual approved count: {decision.get("actual_approved_count")}

## Product Signal

- Adapter over base: {decision.get("adapter_over_base")}
- Adapter over runtime contract: {decision.get("adapter_over_runtime_contract")}
- Adapter + runtime contract over runtime contract: {decision.get("adapter_runtime_contract_over_runtime_contract")}
- Runtime contract primary path: {decision.get("runtime_contract_primary_path")}

## Interpretation

Phase36-39 proves the straight-line loop shape, but current committed evidence is simulated lab evidence unless real approved local feedback reaches the threshold. Do not claim actual user product benefit from this evidence alone.
""",
        encoding="utf-8",
    )


def _write_next_goal(path: Path) -> None:
    path.write_text(
        """目标：开发并验证 PFE Phase40：真实本地反馈采集与人工审核达标。

请继续保持 Hermes 解耦，收集至少 12 条 operator-attested actual local feedback，完成 Phase36 review，重新生成 actual_candidate_lane，并只在真实 approved feedback 达标后再声明产品收益候选。
""",
        encoding="utf-8",
    )


def generate_phase36_39_evidence(args: argparse.Namespace) -> dict[str, Any]:
    if args.clean_evidence:
        _clean_dir(PHASE36_39_DIR)
    review_dir = PHASE36_39_DIR / "evidence-review"
    candidates_dir = PHASE36_39_DIR / "evidence-candidates"
    training_dir = PHASE36_39_DIR / "evidence-training"
    eval_dir = PHASE36_39_DIR / "evidence-eval"
    for path in (review_dir, candidates_dir, training_dir, eval_dir):
        path.mkdir(parents=True, exist_ok=True)

    phase35_state_path = phase35_store_path(PHASE35_DIR / "evidence", "phase35-demo")
    phase35_state = load_phase35_state(phase35_state_path)
    phase35_capture_rows = _read_jsonl(PHASE35_DIR / "evidence-capture" / "simulated_local_interactions.jsonl")
    review_queue = build_phase36_review_queue(phase35_state)
    review_decisions = [
        build_phase36_review_decision(
            row,
            state="exclude",
            reviewer_id="phase36-deterministic-reviewer",
            reason="Phase35 simulated/local lab row is not actual_user_feedback.",
        )
        for row in phase35_capture_rows
    ]
    review_summary = build_phase36_review_summary(
        state=phase35_state,
        review_decisions=review_decisions,
    )
    _write_json(review_dir / "phase35_store_source.json", {"kind": "phase36_phase35_store_source", "path": str(phase35_state_path), "state": phase35_state})
    _write_json(review_dir / "review_queue.json", review_queue)
    _write_json(review_dir / "review_decisions.json", {"kind": "phase36_review_decisions", "items": review_decisions})
    _write_json(review_dir / "phase36_review_summary.json", review_summary)

    holdout = build_phase37_holdout(count=args.holdout_count)
    simulated_lab_records = build_phase36_39_simulated_lab_records(count=args.simulated_lab_count)
    actual_candidates = build_phase37_candidate_artifacts(
        records=list(phase35_state.get("interactions") or []),
        review_decisions=review_decisions,
        holdout=holdout,
        lane="actual_candidate_lane",
    )
    simulated_candidates = build_phase37_candidate_artifacts(
        records=simulated_lab_records,
        review_decisions=[],
        holdout=holdout,
        lane="simulated_lab_candidate_lane",
    )
    _write_json(candidates_dir / "holdout.json", holdout)
    _write_json(candidates_dir / "candidate_manifest.json", simulated_candidates["candidate_manifest"])
    _write_json(candidates_dir / "actual_candidate_manifest.json", actual_candidates["candidate_manifest"])
    _write_json(candidates_dir / "simulated_lab_candidate_manifest.json", simulated_candidates["candidate_manifest"])
    write_jsonl(candidates_dir / "sft_samples.jsonl", simulated_candidates["sft_samples"])
    write_jsonl(candidates_dir / "dpo_pairs.jsonl", simulated_candidates["dpo_pairs"])
    _write_json(candidates_dir / "candidate_quality_report.json", simulated_candidates["candidate_quality_report"])
    _write_json(candidates_dir / "holdout_integrity_check.json", simulated_candidates["holdout_integrity_check"])
    _write_json(candidates_dir / "actual_candidate_artifacts.json", actual_candidates)
    _write_json(candidates_dir / "simulated_lab_candidate_artifacts.json", simulated_candidates)

    _model_selection, _training_manifest, training_attempt = _run_phase38_training_probe(
        evidence_training_dir=training_dir,
        simulated_candidates=simulated_candidates,
        args=args,
    )

    sessions = build_phase39_simulated_sessions(count=args.session_count)
    transcripts_by_variant = {
        variant: build_phase39_transcripts(sessions=sessions, model_variant=variant)
        for variant in PHASE39_MODEL_VARIANTS
    }
    blind_pairs = build_phase39_blind_eval_pairs(sessions=sessions, transcripts_by_variant=transcripts_by_variant)
    eval_report = build_phase39_eval_report(transcripts_by_variant=transcripts_by_variant)
    boundary_check = validate_phase39_boundaries(
        sessions=sessions,
        transcripts=[item for rows in transcripts_by_variant.values() for item in rows],
        blind_pairs=blind_pairs,
    )
    final_decision = phase39_final_decision(
        review_summary=review_summary,
        actual_candidates=actual_candidates,
        simulated_candidates=simulated_candidates,
        training_attempt=training_attempt,
        eval_report=eval_report,
        boundary_check=boundary_check,
    )
    summary = build_phase36_39_comparison_summary(
        review_summary=review_summary,
        actual_candidates=actual_candidates,
        simulated_candidates=simulated_candidates,
        training_attempt=training_attempt,
        eval_report=eval_report,
        final_decision=final_decision,
    )
    write_jsonl(eval_dir / "simulated_sessions.jsonl", sessions)
    for variant, rows in transcripts_by_variant.items():
        write_jsonl(eval_dir / f"{variant}_transcripts.jsonl", rows)
    write_jsonl(eval_dir / "blind_eval_pairs.jsonl", blind_pairs)
    _write_json(eval_dir / "user_acceptance_scores.json", eval_report)
    _write_json(eval_dir / "boundary_check.json", boundary_check)
    _write_output_examples(eval_dir / "output_examples.md", transcripts_by_variant)
    _write_json(PHASE36_39_DIR / "comparison_summary.json", summary)
    _write_json(eval_dir / "comparison_summary.json", summary)
    _write_json(PHASE36_39_DIR / "phase39-final-decision.json", final_decision)
    _write_runbook(PHASE36_39_DIR / "phase36-39-runbook.md")
    _write_final_decision(PHASE36_39_DIR / "phase39-final-decision.md", final_decision)
    _write_next_goal(PHASE36_39_DIR / "next-pursuit-goal.md")
    _redact_evidence_tree(PHASE36_39_DIR)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-evidence", action="store_true")
    parser.add_argument("--simulated-lab-count", type=int, default=24)
    parser.add_argument("--holdout-count", type=int, default=40)
    parser.add_argument("--session-count", type=int, default=64)
    parser.add_argument("--training-steps", type=int, default=12)
    parser.add_argument("--skip-real-training", action="store_true")
    parser.add_argument("--training-output-dir", type=Path, default=Path("trainer_job_outputs/phase36-39-local-feedback-product-benefit-loop"))
    parser.add_argument("--clean-training-output", action="store_true")
    args = parser.parse_args()
    summary = generate_phase36_39_evidence(args)
    print(
        json.dumps(
            {
                "kind": summary["kind"],
                "status": summary["status"],
                "phase38_training_probe_status": summary["phase38_training_probe_status"],
                "phase39_product_eval_status": summary["phase39_product_eval_status"],
                "evidence_type": summary["evidence_type"],
                "final_recommendation": summary["final_recommendation"],
                "actual_product_benefit_claim_allowed": summary["actual_product_benefit_claim_allowed"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
