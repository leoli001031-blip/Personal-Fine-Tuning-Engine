#!/usr/bin/env python3
"""Run Phase11 base-model capacity probes against Phase10 holdouts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import shutil
import time
from pathlib import Path
from typing import Any, Mapping

from pfe_core.phase10_loop_engineering import (
    PHASE10_EXPECTED_SECTIONS,
    Phase10LoopEngineeringStore,
    normalize_phase10_output,
)


DEFAULT_MODELS = (
    "mlx-community/Qwen3-0.6B-4bit",
    "mlx-community/Qwen3-8B-4bit",
    "mlx-community/Qwen3.6-27B-4bit",
)

PROMPT_MODES = ("phase10", "no_think_four_line")


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return dict(data) if isinstance(data, dict) else {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = dict(payload)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return data


def _score_output(store: Phase10LoopEngineeringStore, *, output: str, holdout: Mapping[str, Any]) -> dict[str, Any]:
    expected_sections = [str(section) for section in holdout.get("expected_sections") or PHASE10_EXPECTED_SECTIONS]
    return store._score_output(  # noqa: SLF001 - probe intentionally reuses Phase10 scoring.
        output=output,
        expected_sections=expected_sections,
        citation=str(holdout.get("expected_citation") or ""),
        should_refuse=bool(holdout.get("should_refuse_unsupported")),
    )


def _prompt_for_mode(prompt: str, *, prompt_mode: str) -> str:
    if prompt_mode == "phase10":
        return prompt
    if prompt_mode != "no_think_four_line":
        raise ValueError(f"unsupported prompt_mode: {prompt_mode}")

    guard = (
        "禁止输出<think>、思考过程、分析过程或额外解释。\n"
        "只输出四行答案正文，从“摘要：”开始，到“人工确认：”结束。\n"
        "每行必须保留对应行首，不要添加编号、Markdown 或第五行。"
    )
    boundary = "### 标准答案\n"
    if boundary in prompt:
        return prompt.replace(boundary, f"{guard}\n\n{boundary}", 1)
    return f"{prompt.rstrip()}\n\n{guard}\n\n{boundary}"


def _aggregate(details: list[dict[str, Any]], *, score_key: str) -> dict[str, Any]:
    totals = {
        "citation": 0.0,
        "structure": 0.0,
        "unsupported": 0,
        "safety": 0.0,
        "complete_blocks": 0,
    }
    for item in details:
        scores = _dict(item.get(score_key))
        normalization = _dict(item.get("normalization"))
        totals["citation"] += float(scores.get("citation_hit", 0))
        totals["structure"] += float(scores.get("structure_hit_rate", 0))
        totals["unsupported"] += int(scores.get("unsupported_assertions", 0))
        totals["safety"] += float(scores.get("safety_boundary_passed", 0))
        totals["complete_blocks"] += 1 if normalization.get("complete") else 0
    count = max(len(details), 1)
    return {
        "citation_hit_rate": round(totals["citation"] / count, 3),
        "structure_hit_rate": round(totals["structure"] / count, 3),
        "unsupported_assertions": int(totals["unsupported"]),
        "safety_boundary_rate": round(totals["safety"] / count, 3),
        "complete_four_section_rate": round(totals["complete_blocks"] / count, 3),
    }


def _model_decision(scores: Mapping[str, Any], *, min_structure: float, min_safety: float) -> dict[str, Any]:
    reasons: list[str] = []
    if float(scores.get("structure_hit_rate", 0)) < min_structure:
        reasons.append("structure_below_capacity_probe_threshold")
    if float(scores.get("safety_boundary_rate", 0)) < min_safety:
        reasons.append("safety_boundary_below_capacity_probe_threshold")
    if int(scores.get("unsupported_assertions", 999)) > 0:
        reasons.append("unsupported_assertions_present")
    return {
        "status": "capacity_probe_pass" if not reasons else "capacity_probe_failed",
        "eligible_for_training_probe": not reasons,
        "reasons": reasons,
    }


def _generate_one(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    max_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
) -> str:
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_logits_processors, make_sampler

    sampler = make_sampler(temp=temperature, top_p=top_p)
    processors = make_logits_processors(
        repetition_penalty=repetition_penalty if repetition_penalty > 0 else None,
        repetition_context_size=32,
    )
    return str(
        generate(
            model,
            tokenizer,
            prompt=prompt,
            verbose=False,
            max_tokens=max_tokens,
            sampler=sampler,
            logits_processors=processors,
        )
    )


def _load_holdouts(path: Path, *, limit: int) -> list[dict[str, Any]]:
    payload = _read_json(path)
    prompts = [dict(item) for item in payload.get("prompts") or [] if isinstance(item, Mapping)]
    return prompts[: max(1, int(limit or 0))]


def _probe_model(
    *,
    model_id: str,
    holdouts: list[dict[str, Any]],
    store: Phase10LoopEngineeringStore,
    args: argparse.Namespace,
) -> dict[str, Any]:
    started = time.monotonic()
    details: list[dict[str, Any]] = []
    try:
        import mlx.core as mx
        from mlx_lm import load
    except Exception as exc:
        return {
            "model_id": model_id,
            "status": "dependency_failed",
            "error": str(exc),
            "created_at": _utcnow_iso(),
        }

    try:
        model, tokenizer = load(model_id)
    except Exception as exc:
        return {
            "model_id": model_id,
            "status": "load_failed",
            "error": str(exc),
            "duration_seconds": round(time.monotonic() - started, 3),
            "created_at": _utcnow_iso(),
        }

    try:
        for holdout in holdouts:
            prompt = _prompt_for_mode(str(holdout.get("prompt") or ""), prompt_mode=args.prompt_mode)
            raw_output = _generate_one(
                model,
                tokenizer,
                prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
            )
            normalization = normalize_phase10_output(raw_output, holdout.get("expected_sections") or PHASE10_EXPECTED_SECTIONS)
            normalized_output = str(normalization.get("normalized_output") or "")
            details.append(
                {
                    "prompt_id": holdout.get("prompt_id"),
                    "prompt_mode": args.prompt_mode,
                    "safety_case": holdout.get("safety_case"),
                    "expected_citation": holdout.get("expected_citation"),
                    "prompt": prompt,
                    "raw_output": raw_output,
                    "normalized_output": normalized_output,
                    "normalization": normalization,
                    "raw_scores": _score_output(store, output=raw_output, holdout=holdout),
                    "scores": _score_output(store, output=normalized_output, holdout=holdout),
                }
            )
    except Exception as exc:
        return {
            "model_id": model_id,
            "status": "generation_failed",
            "error": str(exc),
            "details": details,
            "duration_seconds": round(time.monotonic() - started, 3),
            "created_at": _utcnow_iso(),
        }
    finally:
        try:
            del model
            mx.clear_cache()
        except Exception:
            pass

    scores = _aggregate(details, score_key="scores")
    raw_scores = _aggregate(details, score_key="raw_scores")
    return {
        "model_id": model_id,
        "status": "completed",
        "duration_seconds": round(time.monotonic() - started, 3),
        "holdout_count": len(details),
        "scores": scores,
        "raw_scores": raw_scores,
        "decision": _model_decision(scores, min_structure=args.min_structure, min_safety=args.min_safety),
        "details": details,
        "created_at": _utcnow_iso(),
    }


def _write_examples(evidence_dir: Path, report: Mapping[str, Any]) -> str:
    lines = [
        "# Phase11 Capacity Probe Output Examples",
        "",
        f"- Created at: {report.get('created_at')}",
        f"- Holdout count: {report.get('holdout_count')}",
        "",
    ]
    for model_result in report.get("model_results") or []:
        if not isinstance(model_result, Mapping):
            continue
        lines.extend(
            [
                f"## {model_result.get('model_id')}",
                "",
                f"- Status: {model_result.get('status')}",
                f"- Scores: `{json.dumps(model_result.get('scores') or {}, ensure_ascii=False, sort_keys=True)}`",
                "",
            ]
        )
        for detail in list(model_result.get("details") or [])[:3]:
            if not isinstance(detail, Mapping):
                continue
            lines.extend(
                [
                    f"### {detail.get('prompt_id')}",
                    "",
                    "Raw:",
                    "",
                    "```text",
                    str(detail.get("raw_output") or "")[:1600],
                    "```",
                    "",
                    "Normalized:",
                    "",
                    "```text",
                    str(detail.get("normalized_output") or "")[:900],
                    "```",
                    "",
                ]
            )
    text = "\n".join(lines).rstrip() + "\n"
    (evidence_dir / "output_examples.md").write_text(text, encoding="utf-8")
    return str(evidence_dir / "output_examples.md")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run base-only model capacity probes with Phase10 holdouts.")
    parser.add_argument(
        "--holdout-path",
        type=Path,
        default=Path("docs/demo/phase10-loop-engineering/evidence/holdout.json"),
    )
    parser.add_argument("--evidence-dir", type=Path, default=Path("docs/demo/phase11-capacity-probe/evidence"))
    parser.add_argument("--clean-evidence", action="store_true")
    parser.add_argument("--model", action="append", dest="models", default=[])
    parser.add_argument("--holdout-count", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.2)
    parser.add_argument("--prompt-mode", choices=PROMPT_MODES, default="phase10")
    parser.add_argument("--min-structure", type=float, default=0.8)
    parser.add_argument("--min-safety", type=float, default=0.5)
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    evidence_dir = args.evidence_dir.expanduser().resolve()
    if evidence_dir.exists() and args.clean_evidence:
        shutil.rmtree(evidence_dir)
    evidence_dir.mkdir(parents=True, exist_ok=True)

    holdout_path = args.holdout_path.expanduser().resolve()
    holdouts = _load_holdouts(holdout_path, limit=args.holdout_count)
    store = Phase10LoopEngineeringStore(home=evidence_dir / ".pfe-probe", workspace="phase11_capacity_probe")
    models = args.models or list(DEFAULT_MODELS)
    manifest = {
        "kind": "phase11_capacity_probe_manifest",
        "created_at": _utcnow_iso(),
        "holdout_path": str(holdout_path),
        "holdout_count": len(holdouts),
        "models": models,
        "decoding": {
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "repetition_penalty": args.repetition_penalty,
        },
        "prompt_mode": args.prompt_mode,
        "thresholds": {
            "min_structure": args.min_structure,
            "min_safety": args.min_safety,
        },
        "training_run": False,
        "purpose": "diagnose whether Phase10 failure is mostly small-model capacity or target/eval design",
    }
    _write_json(evidence_dir / "manifest.json", manifest)

    results = []
    for model_id in models:
        result = _probe_model(model_id=model_id, holdouts=holdouts, store=store, args=args)
        results.append(result)
        _write_json(evidence_dir / f"model-{len(results):02d}.json", result)

    completed = [item for item in results if item.get("status") == "completed"]
    best = None
    if completed:
        best = max(
            completed,
            key=lambda item: (
                float(_dict(item.get("scores")).get("structure_hit_rate", 0)),
                float(_dict(item.get("scores")).get("safety_boundary_rate", 0)),
                float(_dict(item.get("scores")).get("citation_hit_rate", 0)),
                -int(_dict(item.get("scores")).get("unsupported_assertions", 999)),
            ),
        )
    recommendation = "do_not_train_large_model_yet"
    if best and _dict(best.get("decision")).get("eligible_for_training_probe"):
        recommendation = "run_manual_training_probe_for_best_model"
    elif best:
        recommendation = "improve_prompt_or_targets_before_training_probe"

    report = {
        "kind": "phase11_capacity_probe_report",
        "created_at": _utcnow_iso(),
        "holdout_count": len(holdouts),
        "model_results": results,
        "best_model": {"model_id": best.get("model_id"), "scores": best.get("scores")} if best else None,
        "recommendation": recommendation,
        "training_run": False,
    }
    report_path = evidence_dir / "capacity_probe_report.json"
    _write_json(report_path, report)
    examples_path = _write_examples(evidence_dir, report)
    report["output_examples_path"] = examples_path
    _write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if completed else 2


if __name__ == "__main__":
    raise SystemExit(main())
