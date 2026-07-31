#!/usr/bin/env python3
"""Generate Phase30 simulated-human feedback quality-loop evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
PFE_CORE = ROOT / "pfe-core"
if str(PFE_CORE) not in sys.path:
    sys.path.insert(0, str(PFE_CORE))

from pfe_core.phase30_simulated_feedback_quality import (
    aggregate_phase30_quality,
    build_phase30_candidate_artifacts,
    build_phase30_feedback_batch,
    build_phase30_feedback_routing_report,
    build_phase30_personas,
    build_phase30_tasks,
    phase30_final_decision,
    score_phase30_output,
    write_jsonl,
)


PHASE30_DIR = Path("docs/demo/phase30-simulated-human-feedback-quality-loop")
PHASE29_DIR = Path("docs/demo/phase29-feedback-driven-tuning-benefit-proof")


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


def _read_text(path: Path, *, max_chars: int = 2000) -> str:
    try:
        return path.read_text(encoding="utf-8")[:max_chars]
    except Exception:
        return ""


def _clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _load_local_phase17_tool() -> Any:
    path = Path(__file__).resolve().parent / "phase17_qwen_dpo_product_probe.py"
    spec = importlib.util.spec_from_file_location("phase17_qwen_dpo_product_probe", path)
    if not spec or not spec.loader:
        raise RuntimeError(f"cannot load Phase17 helper from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_phase30_phase29_review() -> dict[str, Any]:
    final_decision = _read_text(PHASE29_DIR / "phase29-final-decision.md")
    audit = _read_text(PHASE29_DIR / "phase29-completion-audit.md")
    eval_report = _read_json(PHASE29_DIR / "evidence-training" / "dpo-fallback-qwen25-0_5b" / "eval_report_rescored.json")
    scores = _dict(eval_report.get("scores"))
    decision = _dict(eval_report.get("decision"))
    return {
        "kind": "phase30_phase29_review",
        "reviewed_paths": [
            str(PHASE29_DIR / "phase29-final-decision.md"),
            str(PHASE29_DIR / "phase29-completion-audit.md"),
            str(PHASE29_DIR / "evidence-eval" / "output_examples.md"),
            str(PHASE29_DIR / "evidence-training" / "dpo-fallback-qwen25-0_5b" / "eval_report_rescored.json"),
        ],
        "phase29_final_decision_excerpt": final_decision,
        "phase29_completion_audit_excerpt": audit,
        "phase29_scores": scores,
        "phase29_decision": decision,
        "failure_causes": [
            "four_section_structure_not_stable",
            "user_preference_adherence_not_learned",
            "external_law_reference_still_present",
            "manual_confirmation_unstable",
            "missing_material_boundary_not_strict_enough",
        ],
        "phase30_response": [
            "simulate stricter user/operator feedback personas",
            "make chosen outputs exact four-section targets",
            "make rejected outputs explicit hard negatives",
            "keep simulated feedback separate from actual_user_feedback",
        ],
        "created_at": _utcnow_iso(),
    }


def _phase29_review_markdown(review: Mapping[str, Any]) -> str:
    lines = ["# Phase30 Review Of Phase29", ""]
    lines.append("Phase29 proved the loop can train, but the adapter did not pass product gates.")
    lines.append("")
    lines.append("## Failure Causes")
    for item in review.get("failure_causes") or []:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Phase30 Response")
    for item in review.get("phase30_response") or []:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("No Phase30 simulated feedback is counted as actual user feedback.")
    return "\n".join(lines) + "\n"


def phase17_compatible_dpo_samples(dpo_pairs: list[Mapping[str, Any]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for index, item in enumerate(dpo_pairs):
        rows.append(
            {
                "sample_id": str(item.get("sample_id") or item.get("pair_id") or f"phase30_dpo_{index:03d}"),
                "instruction": str(item.get("instruction") or item.get("prompt") or ""),
                "chosen": str(item.get("chosen") or ""),
                "rejected": str(item.get("rejected") or ""),
            }
        )
    return rows


def _rescore_phase30_probe(result: Mapping[str, Any]) -> dict[str, Any]:
    details = []
    for item in result.get("details") or []:
        if not isinstance(item, Mapping):
            continue
        output = str(item.get("raw_output") or item.get("normalized_output") or item.get("output") or "")
        scores = score_phase30_output(
            output,
            expected_citation=str(item.get("expected_citation") or ""),
            category=str(item.get("category") or ""),
        )
        details.append({**dict(item), "phase30_scores": scores})
    return {
        "status": result.get("status"),
        "model_id": result.get("model_id"),
        "adapter_path": result.get("adapter_path"),
        "holdout_count": len(details),
        "scores": aggregate_phase30_quality([_dict(item.get("phase30_scores")) for item in details]),
        "details": details,
        "created_at": _utcnow_iso(),
    }


def run_phase30_training_probe(
    *,
    evidence_training_dir: Path,
    evidence_eval_dir: Path,
    dpo_pairs: list[Mapping[str, Any]],
    holdout: Mapping[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    if not args.run_real_training:
        payload = {
            "kind": "phase30_training_probe",
            "status": "not_started",
            "real_training": "not_started",
            "training_run": False,
            "skip_reason": "run with --run-real-training",
            "created_at": _utcnow_iso(),
        }
        _write_json(evidence_training_dir / "training_attempt.json", payload)
        _write_json(evidence_training_dir / "train_log.json", payload)
        return payload

    phase17 = _load_local_phase17_tool()
    selected_pairs = dpo_pairs[: max(1, min(args.dpo_sample_limit, len(dpo_pairs)))]
    phase17_samples = phase17_compatible_dpo_samples(selected_pairs)
    write_jsonl(evidence_training_dir / "selected_dpo_pairs.jsonl", selected_pairs)
    write_jsonl(evidence_training_dir / "selected_phase17_compatible_dpo_pairs.jsonl", phase17_samples)
    if args.clean_training_output and args.training_output_dir.exists():
        shutil.rmtree(args.training_output_dir)
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    job_spec = phase17.build_qwen_dpo_job_spec(
        samples=phase17_samples,
        base_model=model_id,
        output_dir=args.training_output_dir.expanduser().resolve(),
        epochs=1,
        beta=0.1,
        max_length=1024,
        max_prompt_length=768,
    )
    job_spec = dict(job_spec)
    recipe = dict(job_spec.get("recipe") or {})
    training_recipe = dict(recipe.get("training") or {})
    training_recipe["use_cpu"] = True
    recipe["training"] = training_recipe
    job_spec["recipe"] = recipe
    job_spec["use_cpu"] = True
    preflight = phase17.dpo_preflight()
    model_selection = {
        "kind": "phase30_training_probe_model_selection",
        "status": "selected",
        "selected_model": model_id,
        "selected": model_id,
        "training_role": "simulation_quality_training_probe_not_product_benefit",
        "created_at": _utcnow_iso(),
    }
    _write_json(evidence_training_dir / "dpo_job_spec.json", job_spec)
    _write_json(evidence_training_dir / "dpo_preflight.json", preflight)
    started = time.monotonic()
    training = phase17.run_qwen_dpo_training(
        evidence_dir=evidence_training_dir,
        job_spec=job_spec,
        preflight=preflight,
        model_selection=model_selection,
        run_real_qwen_dpo=True,
    )
    if training.get("real_training") != "completed":
        payload = {
            "kind": "phase30_training_probe",
            "status": "failed",
            "real_training": training.get("real_training"),
            "training_attempt": training,
            "duration_seconds": round(time.monotonic() - started, 3),
            "created_at": _utcnow_iso(),
        }
        _write_json(evidence_training_dir / "phase30_training_probe.json", payload)
        return payload

    holdouts = [
        {**dict(item), "prompt": str(item.get("original_prompt") or item.get("task") or "")}
        for item in list(holdout.get("prompts") or [])[: args.eval_holdout_limit]
        if isinstance(item, Mapping)
    ]
    adapter_path = str(_dict(training.get("adapter_validation")).get("artifact_dir") or training.get("adapter_path") or "")
    base_raw = phase17._generate_transformers_outputs(  # noqa: SLF001
        evidence_dir=evidence_eval_dir,
        model_id=model_id,
        label="phase30_base_eval",
        holdouts=holdouts,
        adapter_path=None,
        max_new_tokens=args.eval_max_tokens,
        local_files_only=True,
        device="cpu",
    )
    adapter_raw = phase17._generate_transformers_outputs(  # noqa: SLF001
        evidence_dir=evidence_eval_dir,
        model_id=model_id,
        label="phase30_adapter_eval",
        holdouts=holdouts,
        adapter_path=adapter_path,
        max_new_tokens=args.eval_max_tokens,
        local_files_only=True,
        device="cpu",
    )
    base = _rescore_phase30_probe(base_raw)
    adapter = _rescore_phase30_probe(adapter_raw)
    report = {
        "kind": "phase30_training_probe",
        "status": "completed",
        "real_training": "completed",
        "training_role": "simulation_quality_training_probe_not_product_benefit",
        "model_id": model_id,
        "adapter_path": adapter_path,
        "training_attempt": training,
        "base_result": base,
        "adapter_result": adapter,
        "scores": {"base": base["scores"], "adapter": adapter["scores"]},
        "recommendation": "simulation_quality_ready_for_real_feedback",
        "duration_seconds": round(time.monotonic() - started, 3),
        "created_at": _utcnow_iso(),
    }
    _write_json(evidence_eval_dir / "eval_report.json", report)
    _write_json(evidence_training_dir / "phase30_training_probe.json", report)
    return report


def _write_output_examples(path: Path, training_probe: Mapping[str, Any]) -> None:
    lines = ["# Phase30 Output Examples", ""]
    for label, result in (("Base", _dict(training_probe.get("base_result"))), ("Adapter", _dict(training_probe.get("adapter_result")))):
        if not result:
            continue
        lines.extend([f"## {label}", "", f"- Scores: `{json.dumps(result.get('scores') or {}, ensure_ascii=False, sort_keys=True)}`", ""])
        for detail in list(result.get("details") or [])[:3]:
            lines.extend(["```text", str(detail.get("raw_output") or detail.get("output") or "")[:1200], "```", ""])
    if len(lines) == 2:
        lines.extend(["No real training probe output was generated in this run.", ""])
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def _write_runbook(path: Path) -> None:
    path.write_text(
        """# Phase30 Runbook

Phase30 builds simulated-human feedback quality evidence. It does not claim actual user feedback or production product lift.

## Default Smoke

```bash
.venv/bin/python tools/phase30_simulated_human_feedback_quality_loop.py --clean-evidence
```

## Optional 12-Step Training Probe

```bash
.venv/bin/python tools/phase30_simulated_human_feedback_quality_loop.py \\
  --clean-evidence \\
  --run-real-training \\
  --clean-training-output
```
""",
        encoding="utf-8",
    )


def _write_final_decision(path: Path, summary: Mapping[str, Any]) -> None:
    decision = _dict(summary.get("decision"))
    quality = _dict(summary.get("candidate_quality_report"))
    manifest = _dict(summary.get("candidate_manifest"))
    training = _dict(summary.get("training_probe"))
    aggregate = _dict(quality.get("aggregate"))
    path.write_text(
        f"""# Phase30 Final Decision

## Decision

- Recommendation: {decision.get("recommendation")}
- Status: {decision.get("status")}
- Promotion allowed: false
- Product benefit claim allowed: false
- Actual user feedback collected: false
- Simulated human perspective only: true

## Evidence

- Personas: {len(summary.get("personas") or [])}
- Training tasks: {_dict(summary.get("task_set")).get("training_task_count")}
- Preference tasks: {_dict(summary.get("task_set")).get("preference_task_count")}
- Holdout tasks: {_dict(_dict(summary.get("task_set")).get("holdout")).get("holdout_count")}
- Simulated feedback: {manifest.get("simulated_human_feedback_count")}
- SFT samples: {manifest.get("sft_sample_count")}
- DPO pairs: {manifest.get("dpo_pair_count")}
- Hard negatives: {manifest.get("hard_negative_pair_count")}
- Correction samples: {manifest.get("correction_sample_count")}
- Training probe: {training.get("status")}

## Quality Scores

| Metric | Score |
| --- | ---: |
| four_section_exact_rate | {aggregate.get("four_section_exact_rate")} |
| citation_exact_match_rate | {aggregate.get("citation_exact_match_rate")} |
| no_external_law_rate | {aggregate.get("no_external_law_rate")} |
| no_legal_conclusion_rate | {aggregate.get("no_legal_conclusion_rate")} |
| manual_confirmation_rate | {aggregate.get("manual_confirmation_rate")} |
| missing_info_first_rate | {aggregate.get("missing_info_first_rate")} |
| preference_adherence_rate | {aggregate.get("preference_adherence_rate")} |
| concise_output_rate | {aggregate.get("concise_output_rate")} |
| hard_negative_contrast_score | {aggregate.get("hard_negative_contrast_score")} |

## Boundary

Phase30 simulated feedback can validate sample format and preference-data quality, but it cannot prove production product benefit. The next step is collecting actual user feedback.
""",
        encoding="utf-8",
    )


def generate_phase30_evidence(args: argparse.Namespace) -> dict[str, Any]:
    if args.clean_evidence:
        _clean_dir(PHASE30_DIR)
    for subdir in (
        "evidence",
        "evidence-personas",
        "evidence-feedback",
        "evidence-candidates",
        "evidence-training",
        "evidence-eval",
    ):
        (PHASE30_DIR / subdir).mkdir(parents=True, exist_ok=True)

    evidence_dir = PHASE30_DIR / "evidence"
    persona_dir = PHASE30_DIR / "evidence-personas"
    feedback_dir = PHASE30_DIR / "evidence-feedback"
    candidate_dir = PHASE30_DIR / "evidence-candidates"
    training_dir = PHASE30_DIR / "evidence-training"
    eval_dir = PHASE30_DIR / "evidence-eval"

    phase29_review = build_phase30_phase29_review()
    personas = build_phase30_personas()
    task_set = build_phase30_tasks(
        training_count=args.training_task_count,
        preference_count=args.preference_task_count,
        holdout_count=args.holdout_count,
    )
    feedback = build_phase30_feedback_batch(tasks=task_set["training_tasks"] + task_set["preference_tasks"], personas=personas)
    routing = build_phase30_feedback_routing_report(feedback)
    candidates = build_phase30_candidate_artifacts(feedback=feedback, routing_report=routing, holdout=task_set["holdout"])
    training_probe = run_phase30_training_probe(
        evidence_training_dir=training_dir,
        evidence_eval_dir=eval_dir,
        dpo_pairs=candidates["dpo_pairs"],
        holdout=task_set["holdout"],
        args=args,
    )
    decision = phase30_final_decision(quality_report=candidates["candidate_quality_report"], training_report=training_probe)

    _write_json(evidence_dir / "phase30_phase29_review.json", phase29_review)
    (evidence_dir / "phase30_phase29_review.md").write_text(_phase29_review_markdown(phase29_review), encoding="utf-8")
    _write_json(evidence_dir / "source_manifest.json", task_set["source_manifest"])
    _write_json(evidence_dir / "task_set.json", task_set)
    _write_json(evidence_dir / "holdout.json", task_set["holdout"])
    _write_json(persona_dir / "personas.json", {"kind": "phase30_personas", "items": personas})
    write_jsonl(persona_dir / "personas.jsonl", personas)
    _write_json(feedback_dir / "simulated_feedback_batch.json", {"kind": "phase30_simulated_feedback_batch", "items": feedback})
    write_jsonl(feedback_dir / "simulated_feedback_batch.jsonl", feedback)
    _write_json(feedback_dir / "feedback_routing_report.json", routing)
    _write_json(
        feedback_dir / "review_decisions.json",
        {
            "kind": "phase30_review_decisions",
            "items": [
                {
                    "feedback_id": item["feedback_id"],
                    "state": item["review_state"],
                    "reason": item["human_feedback_text"],
                    "reviewer_id": item["reviewer_id"],
                }
                for item in feedback
            ],
        },
    )
    write_jsonl(candidate_dir / "selected_sft_samples.jsonl", candidates["sft_samples"])
    write_jsonl(candidate_dir / "selected_dpo_pairs.jsonl", candidates["dpo_pairs"])
    write_jsonl(candidate_dir / "selected_hard_negative_pairs.jsonl", candidates["hard_negative_pairs"])
    write_jsonl(candidate_dir / "selected_correction_samples.jsonl", candidates["correction_samples"])
    _write_json(candidate_dir / "candidate_manifest.json", candidates["candidate_manifest"])
    _write_json(candidate_dir / "candidate_quality_report.json", candidates["candidate_quality_report"])
    _write_json(candidate_dir / "holdout_integrity_check.json", candidates["holdout_integrity_check"])
    _write_json(eval_dir / "decision.json", decision)
    if not (eval_dir / "eval_report.json").exists():
        _write_json(eval_dir / "eval_report.json", {"kind": "phase30_eval_report", "status": "not_started", "skip_reason": "training_probe_not_run", "decision": decision, "created_at": _utcnow_iso()})
    _write_output_examples(eval_dir / "output_examples.md", training_probe)

    summary = {
        "kind": "phase30_simulated_human_feedback_quality_summary",
        "status": "completed",
        "phase29_review": phase29_review,
        "personas": personas,
        "task_set": {
            "training_task_count": task_set["training_task_count"],
            "preference_task_count": task_set["preference_task_count"],
            "holdout": {"holdout_count": task_set["holdout"]["holdout_count"]},
            "total_task_count": task_set["total_task_count"],
        },
        "feedback_routing_report": routing,
        "candidate_manifest": candidates["candidate_manifest"],
        "candidate_quality_report": candidates["candidate_quality_report"],
        "holdout_integrity_check": candidates["holdout_integrity_check"],
        "training_probe": training_probe,
        "decision": decision,
        "final_recommendation": decision["recommendation"],
        "created_at": _utcnow_iso(),
    }
    _write_json(PHASE30_DIR / "comparison_summary.json", summary)
    _write_json(evidence_dir / "comparison_summary.json", summary)
    _write_runbook(PHASE30_DIR / "phase30-runbook.md")
    _write_final_decision(PHASE30_DIR / "phase30-final-decision.md", summary)
    (PHASE30_DIR / "next-pursuit-goal.md").write_text(
        "目标：基于 Phase30 的高质量 simulated feedback 样本，进入 actual_user_feedback 采集并复跑真实训练收益门。\n",
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate Phase30 simulated-human feedback quality evidence.")
    parser.add_argument("--clean-evidence", action="store_true")
    parser.add_argument("--training-task-count", type=int, default=40)
    parser.add_argument("--preference-task-count", type=int, default=20)
    parser.add_argument("--holdout-count", type=int, default=20)
    parser.add_argument("--run-real-training", action="store_true")
    parser.add_argument("--clean-training-output", action="store_true")
    parser.add_argument("--dpo-sample-limit", type=int, default=12)
    parser.add_argument("--eval-holdout-limit", type=int, default=20)
    parser.add_argument("--eval-max-tokens", type=int, default=160)
    parser.add_argument("--training-output-dir", type=Path, default=Path("trainer_job_outputs/phase30-dpo-simulated-feedback-qwen25-0_5b"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = generate_phase30_evidence(args)
    compact = {
        "kind": summary.get("kind"),
        "status": summary.get("status"),
        "candidate_manifest": summary.get("candidate_manifest"),
        "quality": _dict(summary.get("candidate_quality_report")).get("aggregate"),
        "training_probe": {key: _dict(summary.get("training_probe")).get(key) for key in ("status", "real_training", "model_id", "adapter_path")},
        "decision": summary.get("decision"),
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
