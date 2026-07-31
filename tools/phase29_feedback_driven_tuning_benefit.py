#!/usr/bin/env python3
"""Generate Phase29 feedback-driven tuning benefit evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import shutil
import time
from typing import Any, Iterable, Mapping
from urllib import error, request

from pfe_core.phase29_feedback_tuning_benefit import (
    aggregate_phase29_eval,
    build_phase29_benefit_contract,
    build_phase29_candidate_artifacts,
    build_phase29_feedback_batch,
    build_phase29_model_selection,
    build_phase29_signal_routing_report,
    build_phase29_tasks,
    phase29_adapter_decision,
    score_phase29_output,
    write_jsonl,
)


PHASE29_DIR = Path("docs/demo/phase29-feedback-driven-tuning-benefit-proof")
PHASE13_DIR = Path("docs/demo/phase13-boundary-contract-runtime-and-trainable-probe")
PHASE14_DIR = Path("docs/demo/phase14-hard-negative-boundary-training")
PHASE17_DIR = Path("docs/demo/phase17-qwen-dpo-product-probe")
PHASE23_DIR = Path("docs/demo/phase23-runtime-contract-product-loop")
PHASE24_DIR = Path("docs/demo/phase24-real-signal-review-candidate-value-probe")
PHASE25_DIR = Path("docs/demo/phase25-actual-user-feedback-readiness-loop")
PHASE26_DIR = Path("docs/demo/phase26-actual-feedback-collection-training-probe")
PHASE27_DIR = Path("docs/demo/phase27-actual-feedback-review-training-loop")
PHASE28_DIR = Path("docs/demo/phase28-real-feedback-loop-engineering")


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


def _read_text(path: Path, *, max_chars: int = 1600) -> str:
    try:
        return path.read_text(encoding="utf-8")[:max_chars]
    except Exception:
        return ""


def _clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _load_phase13_tool() -> Any:
    path = Path(__file__).resolve().parent / "phase13_boundary_contract_probe.py"
    spec = importlib.util.spec_from_file_location("phase13_boundary_contract_probe", path)
    if not spec or not spec.loader:
        raise RuntimeError(f"cannot load Phase13 helper from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _phase_summary(name: str, path: Path) -> dict[str, Any]:
    summary = _read_json(path / "comparison_summary.json")
    candidate = _dict(summary.get("candidate_c_mid_model") or summary.get("candidate") or {})
    candidate_eval = _dict(candidate.get("eval"))
    return {
        "phase": name,
        "exists": path.exists(),
        "comparison_kind": summary.get("kind"),
        "decision": summary.get("decision") or candidate_eval.get("eval_gate"),
        "scores": summary.get("scores") or candidate_eval.get("scores"),
        "model_selection": summary.get("model_selection"),
        "final_decision_excerpt": _read_text(path / f"{name.lower()}-final-decision.md"),
    }


def build_phase29_prereq_review() -> dict[str, Any]:
    phase13 = _phase_summary("phase13", PHASE13_DIR)
    phase14 = _phase_summary("phase14", PHASE14_DIR)
    phase17 = _phase_summary("phase17", PHASE17_DIR)
    phase28_decision = _read_text(PHASE28_DIR / "phase28-final-decision.md")
    return {
        "kind": "phase29_prereq_review",
        "reviewed_paths": [
            str(PHASE13_DIR),
            str(PHASE14_DIR),
            str(PHASE17_DIR),
            str(PHASE23_DIR),
            str(PHASE24_DIR),
            str(PHASE25_DIR),
            str(PHASE26_DIR),
            str(PHASE27_DIR),
            str(PHASE28_DIR),
        ],
        "phase13": phase13,
        "phase14": phase14,
        "phase17": phase17,
        "phase28_final_decision_excerpt": phase28_decision,
        "conclusions": [
            "PFE already has training, eval, gate, and archive mechanics.",
            "Runtime boundary contract is a strong product path but not a tuning-benefit proof.",
            "Prior SFT/DPO adapters showed partial movement but did not stably beat base/runtime contract.",
            "Phase28's current blocker is insufficient attested actual feedback.",
            "Phase29 uses operator-reviewed feedback only as a technical proof unless actual feedback is present.",
        ],
        "created_at": _utcnow_iso(),
    }


def _prereq_review_markdown(review: Mapping[str, Any]) -> str:
    lines = ["# Phase29 Prerequisite Review", ""]
    for conclusion in review.get("conclusions") or []:
        lines.append(f"- {conclusion}")
    lines.extend(
        [
            "",
            "## Evidence Boundary",
            "",
            "Phase29 is not a Hermes integration task. Ollama qwen3.6 is a strong runtime reference, not the training target.",
            "Operator-reviewed feedback can prove the training loop technically, but production benefit still requires actual user feedback.",
            "",
        ]
    )
    return "\n".join(lines)


def _ollama_json(path: str, payload: Mapping[str, Any] | None = None, *, timeout: int = 45) -> dict[str, Any]:
    url = f"http://127.0.0.1:11434{path}"
    data = None if payload is None else json.dumps(dict(payload), ensure_ascii=False).encode("utf-8")
    req = request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with request.urlopen(req, timeout=timeout) as response:  # noqa: S310 - local-only URL
            raw = response.read().decode("utf-8")
        obj = json.loads(raw)
        return dict(obj) if isinstance(obj, Mapping) else {"raw": obj}
    except (OSError, error.URLError, json.JSONDecodeError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def run_ollama_qwen36_reference(*, holdout: Mapping[str, Any], limit: int, run_outputs: bool) -> dict[str, Any]:
    tags = _ollama_json("/api/tags", timeout=5)
    models = [dict(item) for item in tags.get("models") or [] if isinstance(item, Mapping)]
    qwen = next((item for item in models if str(item.get("name") or item.get("model") or "").startswith("qwen3.6")), None)
    result: dict[str, Any] = {
        "kind": "phase29_ollama_qwen36_runtime_reference",
        "status": "available" if qwen else "unavailable",
        "model": qwen,
        "role": "strong_runtime_reference_not_training_target",
        "real_model_calls": False,
        "created_at": _utcnow_iso(),
    }
    if not qwen:
        result["tags_response"] = tags
        return result
    if not run_outputs:
        result["skip_reason"] = "run with --run-runtime-reference to capture qwen3.6 outputs"
        return result
    details = []
    started = time.monotonic()
    for prompt in list(holdout.get("prompts") or [])[:limit]:
        response = _ollama_json(
            "/api/chat",
            {
                "model": "qwen3.6",
                "stream": False,
                "think": False,
                "options": {"temperature": 0, "num_predict": 192},
                "messages": [{"role": "user", "content": str(prompt.get("user_prompt") or prompt.get("task") or "")}],
            },
            timeout=90,
        )
        content = str(_dict(response.get("message")).get("content") or response.get("response") or "")
        details.append(
            {
                "prompt_id": prompt.get("prompt_id"),
                "category": prompt.get("category"),
                "raw_response": response,
                "output": content,
                "scores": score_phase29_output(
                    content,
                    expected_citation=str(prompt.get("expected_citation") or ""),
                    category=str(prompt.get("category") or ""),
                ),
            }
        )
    result.update(
        {
            "status": "completed",
            "real_model_calls": True,
            "holdout_count": len(details),
            "duration_seconds": round(time.monotonic() - started, 3),
            "scores": aggregate_phase29_eval(details),
            "details": details,
        }
    )
    return result


def _rescore_probe_result(result: Mapping[str, Any]) -> dict[str, Any]:
    details = []
    for item in result.get("details") or []:
        if not isinstance(item, Mapping):
            continue
        output = str(item.get("normalized_output") or item.get("raw_output") or "")
        scores = score_phase29_output(
            output,
            expected_citation=str(item.get("expected_citation") or ""),
            category=str(item.get("category") or ""),
        )
        details.append({**dict(item), "phase29_scores": scores})
    return {
        "status": result.get("status"),
        "model_id": result.get("model_id"),
        "adapter_path": result.get("adapter_path"),
        "holdout_count": len(details),
        "scores": aggregate_phase29_eval(details, score_key="phase29_scores"),
        "details": details,
        "created_at": _utcnow_iso(),
    }


def run_phase29_training(
    *,
    evidence_training_dir: Path,
    model_selection: Mapping[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    if not args.run_real_training:
        payload = {
            "kind": "phase29_training_attempt",
            "real_training": "not_started",
            "training_run": False,
            "skip_reason": "run with --run-real-training",
            "created_at": _utcnow_iso(),
        }
        _write_json(evidence_training_dir / "training_attempt.json", payload)
        _write_json(evidence_training_dir / "train_log.json", payload)
        return payload
    if model_selection.get("status") != "selected":
        payload = {
            "kind": "phase29_training_attempt",
            "real_training": "blocked",
            "training_run": False,
            "skip_reason": "no_trainable_model_selected",
            "model_selection": dict(model_selection),
            "created_at": _utcnow_iso(),
        }
        _write_json(evidence_training_dir / "training_attempt.json", payload)
        _write_json(evidence_training_dir / "train_log.json", payload)
        return payload
    phase13 = _load_phase13_tool()
    namespace = argparse.Namespace(
        run_mid_training=True,
        mid_model_id=str(model_selection.get("selected_model") or model_selection.get("selected") or ""),
        training_steps=args.training_steps,
        train_sample_limit=args.train_sample_limit,
        train_max_seq_length=args.train_max_seq_length,
        training_output_dir=args.training_output_dir.expanduser().resolve(),
        clean_training_output=args.clean_training_output,
        training_timeout_seconds=args.training_timeout_seconds,
    )
    attempt = phase13.run_mid_training_probe(evidence_dir=evidence_training_dir, args=namespace, model_selection=model_selection)
    payload = {"kind": "phase29_training_attempt", **dict(attempt)}
    _write_json(evidence_training_dir / "training_attempt.json", payload)
    _write_json(evidence_training_dir / "train_log.json", payload)
    adapter_path = str(payload.get("adapter_path") or "")
    validation = {
        "kind": "phase29_adapter_validation",
        "adapter_path": adapter_path,
        "exists": bool(adapter_path and Path(adapter_path).exists()),
        "files": sorted([item.name for item in Path(adapter_path).glob("*")]) if adapter_path and Path(adapter_path).exists() else [],
        "created_at": _utcnow_iso(),
    }
    _write_json(evidence_training_dir / "adapter_validation.json", validation)
    return {**payload, "adapter_validation": validation}


def run_phase29_eval(
    *,
    evidence_eval_dir: Path,
    holdout: Mapping[str, Any],
    model_selection: Mapping[str, Any],
    training_attempt: Mapping[str, Any],
    args: argparse.Namespace,
    data_source_summary: Mapping[str, Any],
) -> dict[str, Any]:
    if training_attempt.get("real_training") != "completed":
        report = {
            "kind": "phase29_eval_report",
            "real_model_calls": False,
            "skip_reason": "training_not_completed",
            "training_attempt": dict(training_attempt),
            "decision": {
                "kind": "phase29_adapter_decision",
                "status": "blocked",
                "recommendation": "archive",
                "reasons": ["training_not_completed"],
                "promotion_allowed": False,
                "auto_promotion_allowed": False,
            },
            "created_at": _utcnow_iso(),
        }
        _write_json(evidence_eval_dir / "eval_report.json", report)
        _write_json(evidence_eval_dir / "decision.json", report["decision"])
        return report
    phase13 = _load_phase13_tool()
    model_id = str(model_selection.get("selected_model") or model_selection.get("selected") or "")
    adapter_path = str(training_attempt.get("adapter_path") or _dict(training_attempt.get("adapter_validation")).get("adapter_path") or "")
    holdouts = [dict(item) for item in holdout.get("prompts") or []][: args.eval_holdout_limit]
    base_raw = phase13._probe_model(  # noqa: SLF001
        evidence_dir=evidence_eval_dir,
        model_id=model_id,
        label="phase29_selected_base_eval",
        prompt_mode="boundary_first_chat_no_think",
        holdouts=holdouts,
        max_tokens=args.eval_max_tokens,
        temperature=0.0,
        top_p=0.0,
        repetition_penalty=args.repetition_penalty,
    )
    adapter_raw = phase13._probe_model(  # noqa: SLF001
        evidence_dir=evidence_eval_dir,
        model_id=model_id,
        label="phase29_adapter_eval",
        prompt_mode="boundary_first_chat_no_think",
        holdouts=holdouts,
        max_tokens=args.eval_max_tokens,
        temperature=0.0,
        top_p=0.0,
        repetition_penalty=args.repetition_penalty,
        adapter_path=adapter_path,
    )
    base = _rescore_probe_result(base_raw)
    adapter = _rescore_probe_result(adapter_raw)
    _write_json(evidence_eval_dir / "phase29_selected_base_eval_rescored.json", base)
    _write_json(evidence_eval_dir / "phase29_adapter_eval_rescored.json", adapter)
    decision = phase29_adapter_decision(
        base_scores=_dict(base.get("scores")),
        adapter_scores=_dict(adapter.get("scores")),
        data_source_summary=data_source_summary,
    )
    report = {
        "kind": "phase29_eval_report",
        "real_model_calls": base_raw.get("status") == "completed" and adapter_raw.get("status") == "completed",
        "model_id": model_id,
        "adapter_path": adapter_path,
        "scores": {"base": base.get("scores"), "adapter": adapter.get("scores")},
        "base_result": base,
        "adapter_result": adapter,
        "decision": decision,
        "recommendation": decision["recommendation"],
        "training_attempt": dict(training_attempt),
        "created_at": _utcnow_iso(),
    }
    _write_json(evidence_eval_dir / "eval_report.json", report)
    _write_json(evidence_eval_dir / "decision.json", decision)
    return report


def phase17_compatible_dpo_samples(dpo_pairs: Iterable[Mapping[str, Any]]) -> list[dict[str, str]]:
    samples: list[dict[str, str]] = []
    for index, item in enumerate(dpo_pairs):
        if not isinstance(item, Mapping):
            continue
        samples.append(
            {
                "sample_id": str(item.get("sample_id") or item.get("pair_id") or f"phase29_dpo_{index:03d}"),
                "instruction": str(item.get("instruction") or item.get("prompt") or ""),
                "chosen": str(item.get("chosen") or ""),
                "rejected": str(item.get("rejected") or ""),
            }
        )
    return samples


def run_phase29_dpo_fallback(
    *,
    evidence_dir: Path,
    dpo_pairs: list[Mapping[str, Any]],
    holdout: Mapping[str, Any],
    args: argparse.Namespace,
    data_source_summary: Mapping[str, Any],
) -> dict[str, Any]:
    if not args.run_dpo_fallback:
        payload = {
            "kind": "phase29_dpo_fallback",
            "status": "not_started",
            "skip_reason": "run with --run-dpo-fallback",
            "created_at": _utcnow_iso(),
        }
        _write_json(evidence_dir / "dpo_fallback.json", payload)
        return payload
    phase17 = _load_local_phase17_tool()
    fallback_dir = evidence_dir / "dpo-fallback-qwen25-0_5b"
    fallback_dir.mkdir(parents=True, exist_ok=True)
    selected = dpo_pairs[: max(1, min(args.dpo_fallback_sample_limit, len(dpo_pairs)))]
    write_jsonl(fallback_dir / "selected_dpo_pairs.jsonl", selected)
    phase17_samples = phase17_compatible_dpo_samples(selected)
    write_jsonl(fallback_dir / "selected_phase17_compatible_dpo_pairs.jsonl", phase17_samples)
    if args.clean_training_output and args.dpo_fallback_output_dir.exists():
        shutil.rmtree(args.dpo_fallback_output_dir)
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    job_spec = phase17.build_qwen_dpo_job_spec(
        samples=phase17_samples,
        base_model=model_id,
        output_dir=args.dpo_fallback_output_dir.expanduser().resolve(),
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
        "kind": "phase29_dpo_fallback_model_selection",
        "status": "selected",
        "selected_model": model_id,
        "selected": model_id,
        "training_role": "fallback_training_format_proof_not_primary_product_model",
        "created_at": _utcnow_iso(),
    }
    _write_json(fallback_dir / "dpo_job_spec.json", job_spec)
    _write_json(fallback_dir / "dpo_preflight.json", preflight)
    training = phase17.run_qwen_dpo_training(
        evidence_dir=fallback_dir,
        job_spec=job_spec,
        preflight=preflight,
        model_selection=model_selection,
        run_real_qwen_dpo=True,
    )
    if training.get("real_training") != "completed":
        payload = {
            "kind": "phase29_dpo_fallback",
            "status": "failed",
            "training_attempt": training,
            "created_at": _utcnow_iso(),
        }
        _write_json(evidence_dir / "dpo_fallback.json", payload)
        return payload
    holdouts = [
        {**dict(item), "prompt": str(item.get("user_prompt") or item.get("task") or "")}
        for item in list(holdout.get("prompts") or [])[: args.eval_holdout_limit]
        if isinstance(item, Mapping)
    ]
    adapter_path = str(_dict(training.get("adapter_validation")).get("artifact_dir") or training.get("adapter_path") or "")
    base_raw = phase17._generate_transformers_outputs(  # noqa: SLF001
        evidence_dir=fallback_dir,
        model_id=model_id,
        label="phase29_dpo_fallback_base_eval",
        holdouts=holdouts,
        adapter_path=None,
        max_new_tokens=args.eval_max_tokens,
        local_files_only=True,
        device="cpu",
    )
    adapter_raw = phase17._generate_transformers_outputs(  # noqa: SLF001
        evidence_dir=fallback_dir,
        model_id=model_id,
        label="phase29_dpo_fallback_adapter_eval",
        holdouts=holdouts,
        adapter_path=adapter_path,
        max_new_tokens=args.eval_max_tokens,
        local_files_only=True,
        device="cpu",
    )
    base = _rescore_probe_result(base_raw)
    adapter = _rescore_probe_result(adapter_raw)
    decision = phase29_adapter_decision(
        base_scores=_dict(base.get("scores")),
        adapter_scores=_dict(adapter.get("scores")),
        data_source_summary=data_source_summary,
    )
    report = {
        "kind": "phase29_dpo_fallback",
        "status": "completed",
        "training_role": "fallback_training_format_proof_not_primary_product_model",
        "model_id": model_id,
        "training_attempt": training,
        "base_result": base,
        "adapter_result": adapter,
        "scores": {"base": base.get("scores"), "adapter": adapter.get("scores")},
        "decision": decision,
        "recommendation": decision["recommendation"],
        "created_at": _utcnow_iso(),
    }
    _write_json(fallback_dir / "eval_report_rescored.json", report)
    _write_json(evidence_dir / "dpo_fallback.json", report)
    return report


def _load_local_phase17_tool() -> Any:
    path = Path(__file__).resolve().parent / "phase17_qwen_dpo_product_probe.py"
    spec = importlib.util.spec_from_file_location("phase17_qwen_dpo_product_probe", path)
    if not spec or not spec.loader:
        raise RuntimeError(f"cannot load Phase17 helper from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_output_examples(path: Path, eval_report: Mapping[str, Any], runtime_reference: Mapping[str, Any]) -> None:
    lines = ["# Phase29 Output Examples", ""]
    for label, result in (("Selected Base", _dict(eval_report.get("base_result"))), ("Adapter", _dict(eval_report.get("adapter_result")))):
        lines.extend([f"## {label}", "", f"- Scores: `{json.dumps(result.get('scores') or {}, ensure_ascii=False, sort_keys=True)}`", ""])
        for detail in list(result.get("details") or [])[:3]:
            lines.extend(["```text", str(detail.get("raw_output") or detail.get("output") or "")[:1200], "```", ""])
    if runtime_reference.get("details"):
        lines.extend(["## Ollama qwen3.6 Runtime Reference", ""])
        for detail in list(runtime_reference.get("details") or [])[:2]:
            lines.extend(["```text", str(detail.get("output") or "")[:1200], "```", ""])
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def _write_runbook(path: Path) -> None:
    text = """# Phase29 Runbook

Phase29 proves whether PFE can convert reviewed feedback into training candidates and then into measurable adapter benefit.

## Default Smoke

```bash
.venv/bin/python tools/phase29_feedback_driven_tuning_benefit.py --clean-evidence
```

## Real 12-Step Probe

```bash
.venv/bin/python tools/phase29_feedback_driven_tuning_benefit.py \\
  --clean-evidence \\
  --run-real-training \\
  --training-steps 12 \\
  --train-sample-limit 40 \\
  --eval-holdout-limit 30 \\
  --run-runtime-reference \\
  --run-dpo-fallback
```

Ollama qwen3.6 is a strong runtime reference only. Phase29 does not train Ollama GGUF and does not default to the 52G Qwen3.6-27B safetensors.
"""
    path.write_text(text, encoding="utf-8")


def _write_final_decision(path: Path, summary: Mapping[str, Any]) -> None:
    eval_report = _dict(summary.get("effective_eval_report") or summary.get("eval_report"))
    decision = _dict(eval_report.get("decision"))
    manifest = _dict(summary.get("candidate_manifest"))
    training_attempt = _dict(summary.get("training_attempt"))
    dpo_fallback = _dict(summary.get("dpo_fallback"))
    dpo_training = _dict(dpo_fallback.get("training_attempt"))
    dpo_validation = _dict(dpo_training.get("adapter_validation"))
    scores = _dict(eval_report.get("scores"))
    base_scores = _dict(scores.get("base"))
    adapter_scores = _dict(scores.get("adapter"))
    runtime_scores = _dict(_dict(summary.get("runtime_reference")).get("scores"))
    reasons = decision.get("reasons") or []
    improved_metrics = decision.get("improved_metrics") or []
    text = f"""# Phase29 Final Decision

## Decision

- Recommendation: {decision.get("recommendation", "archive")}
- Status: {decision.get("status", "blocked")}
- Auto promotion allowed: false
- Improved metrics: {", ".join(str(item) for item in improved_metrics) or "none"}
- Gate reasons: {", ".join(str(item) for item in reasons) or "none"}

## Evidence

- Data source: operator_reviewed_feedback={manifest.get("operator_reviewed_feedback_count", 0)}, actual_user_feedback={manifest.get("actual_user_feedback_count", 0)}
- SFT samples: {manifest.get("sft_sample_count", 0)}
- DPO pairs: {manifest.get("dpo_pair_count", 0)}
- Primary 8B MLX training: {training_attempt.get("real_training")} ({training_attempt.get("error_type") or "no_error"})
- Primary adapter path: {training_attempt.get("adapter_path")}
- DPO fallback: {dpo_fallback.get("status")} on {dpo_fallback.get("model_id")}
- DPO adapter valid: {dpo_validation.get("valid")} at {dpo_validation.get("artifact_dir")}
- Effective eval: {eval_report.get("kind")}

## Holdout Scores

| Metric | Base | Adapter |
| --- | ---: | ---: |
| structure_hit_rate | {base_scores.get("structure_hit_rate")} | {adapter_scores.get("structure_hit_rate")} |
| citation_hit_rate | {base_scores.get("citation_hit_rate")} | {adapter_scores.get("citation_hit_rate")} |
| safety_boundary_rate | {base_scores.get("safety_boundary_rate")} | {adapter_scores.get("safety_boundary_rate")} |
| explicit_boundary_rate | {base_scores.get("explicit_boundary_rate")} | {adapter_scores.get("explicit_boundary_rate")} |
| missing_info_ack_rate | {base_scores.get("missing_info_ack_rate")} | {adapter_scores.get("missing_info_ack_rate")} |
| user_preference_adherence_rate | {base_scores.get("user_preference_adherence_rate")} | {adapter_scores.get("user_preference_adherence_rate")} |
| external_law_reference_rate | {base_scores.get("external_law_reference_rate")} | {adapter_scores.get("external_law_reference_rate")} |
| unsupported_assertions | {base_scores.get("unsupported_assertions")} | {adapter_scores.get("unsupported_assertions")} |
| think_leak_rate | {base_scores.get("think_leak_rate")} | {adapter_scores.get("think_leak_rate")} |

## Ollama qwen3.6 Reference

Ollama qwen3.6 is a strong runtime reference, not a Phase29 training target.

| Metric | qwen3.6 reference |
| --- | ---: |
| structure_hit_rate | {runtime_scores.get("structure_hit_rate")} |
| citation_hit_rate | {runtime_scores.get("citation_hit_rate")} |
| safety_boundary_rate | {runtime_scores.get("safety_boundary_rate")} |
| explicit_boundary_rate | {runtime_scores.get("explicit_boundary_rate")} |
| external_law_reference_rate | {runtime_scores.get("external_law_reference_rate")} |
| unsupported_assertions | {runtime_scores.get("unsupported_assertions")} |

## Boundary

This is a PFE tuning-benefit proof, not Hermes integration. If the data source is mainly operator-reviewed feedback, a pass is technical success only and requires actual feedback collection next.
"""
    path.write_text(text, encoding="utf-8")


def generate_phase29_evidence(args: argparse.Namespace) -> dict[str, Any]:
    if args.clean_evidence:
        _clean_dir(PHASE29_DIR)
    for subdir in (
        "evidence",
        "evidence-feedback",
        "evidence-candidates",
        "evidence-training",
        "evidence-eval",
        "evidence-runtime-reference",
    ):
        (PHASE29_DIR / subdir).mkdir(parents=True, exist_ok=True)

    evidence_dir = PHASE29_DIR / "evidence"
    feedback_dir = PHASE29_DIR / "evidence-feedback"
    candidate_dir = PHASE29_DIR / "evidence-candidates"
    training_dir = PHASE29_DIR / "evidence-training"
    eval_dir = PHASE29_DIR / "evidence-eval"
    runtime_dir = PHASE29_DIR / "evidence-runtime-reference"

    prereq = build_phase29_prereq_review()
    benefit = build_phase29_benefit_contract()
    task_set = build_phase29_tasks(train_count=args.training_task_count, holdout_count=args.holdout_count)
    signals = build_phase29_feedback_batch(tasks=task_set["training_tasks"], operator_count=args.operator_feedback_count)
    routing = build_phase29_signal_routing_report(signals)
    candidates = build_phase29_candidate_artifacts(signals=signals, routing_report=routing, holdout=task_set["holdout"])
    model_selection = build_phase29_model_selection(requested_model=args.training_model_id or None)
    runtime_reference = run_ollama_qwen36_reference(
        holdout=task_set["holdout"],
        limit=args.runtime_reference_limit,
        run_outputs=args.run_runtime_reference,
    )

    _write_json(evidence_dir / "phase29_prereq_review.json", prereq)
    (evidence_dir / "phase29_prereq_review.md").write_text(_prereq_review_markdown(prereq), encoding="utf-8")
    _write_json(evidence_dir / "phase29_benefit_contract.json", benefit)
    _write_json(evidence_dir / "source_manifest.json", task_set["source_manifest"])
    _write_json(evidence_dir / "holdout.json", task_set["holdout"])
    _write_json(evidence_dir / "model_selection.json", model_selection)
    _write_json(feedback_dir / "phase29_feedback_batch.json", {"kind": "phase29_feedback_batch", "items": signals})
    write_jsonl(feedback_dir / "phase29_feedback_batch.jsonl", signals)
    _write_json(
        feedback_dir / "phase29_review_decisions.json",
        {
            "kind": "phase29_review_decisions",
            "items": [
                {
                    "signal_id": signal["signal_id"],
                    "state": signal["review_state"],
                    "reason": signal["eligibility_reason"],
                    "reviewer_id": signal["reviewer_id"],
                }
                for signal in signals
            ],
        },
    )
    _write_json(feedback_dir / "phase29_signal_routing_report.json", routing)
    write_jsonl(candidate_dir / "selected_sft_samples.jsonl", candidates["sft_samples"])
    write_jsonl(candidate_dir / "selected_dpo_pairs.jsonl", candidates["dpo_pairs"])
    _write_json(candidate_dir / "candidate_manifest.json", candidates["candidate_manifest"])
    _write_json(candidate_dir / "candidate_quality_report.json", candidates["quality_report"])
    _write_json(candidate_dir / "holdout_integrity_check.json", candidates["holdout_integrity_check"])
    _write_json(runtime_dir / "ollama_qwen36_reference.json", runtime_reference)

    write_jsonl(training_dir / "candidate_samples.jsonl", candidates["sft_samples"])
    training_attempt = run_phase29_training(
        evidence_training_dir=training_dir,
        model_selection=model_selection,
        args=args,
    )
    eval_report = run_phase29_eval(
        evidence_eval_dir=eval_dir,
        holdout=task_set["holdout"],
        model_selection=model_selection,
        training_attempt=training_attempt,
        args=args,
        data_source_summary=candidates["candidate_manifest"],
    )
    dpo_fallback = run_phase29_dpo_fallback(
        evidence_dir=training_dir,
        dpo_pairs=candidates["dpo_pairs"],
        holdout=task_set["holdout"],
        args=args,
        data_source_summary=candidates["candidate_manifest"],
    )
    effective_eval = eval_report
    if _dict(eval_report.get("decision")).get("recommendation") == "archive" and dpo_fallback.get("status") == "completed":
        effective_eval = dpo_fallback
    preference_report = {
        "kind": "phase29_preference_adherence_report",
        "base": _dict(_dict(effective_eval.get("scores")).get("base")),
        "adapter": _dict(_dict(effective_eval.get("scores")).get("adapter")),
        "preference": benefit["persona_scenario_preference"],
        "created_at": _utcnow_iso(),
    }
    _write_json(eval_dir / "preference_adherence_report.json", preference_report)
    _write_output_examples(eval_dir / "output_examples.md", effective_eval, runtime_reference)

    summary = {
        "kind": "phase29_feedback_driven_tuning_benefit_summary",
        "status": "completed",
        "prereq_review": prereq,
        "benefit_contract": benefit,
        "candidate_manifest": candidates["candidate_manifest"],
        "model_selection": model_selection,
        "runtime_reference": runtime_reference,
        "training_attempt": training_attempt,
        "eval_report": eval_report,
        "dpo_fallback": dpo_fallback,
        "effective_eval_report": effective_eval,
        "final_recommendation": _dict(effective_eval.get("decision")).get("recommendation"),
        "created_at": _utcnow_iso(),
    }
    _write_json(PHASE29_DIR / "comparison_summary.json", summary)
    _write_json(evidence_dir / "comparison_summary.json", summary)
    _write_runbook(PHASE29_DIR / "phase29-runbook.md")
    _write_final_decision(PHASE29_DIR / "phase29-final-decision.md", summary)
    (PHASE29_DIR / "next-pursuit-goal.md").write_text(
        "目标：基于 Phase29 结果继续收集 actual_user_feedback，或优化 preference/DPO training 后再跑真实 adapter gate。\n",
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate Phase29 feedback-driven tuning benefit proof evidence.")
    parser.add_argument("--clean-evidence", action="store_true")
    parser.add_argument("--training-task-count", type=int, default=40)
    parser.add_argument("--holdout-count", type=int, default=30)
    parser.add_argument("--operator-feedback-count", type=int, default=40)
    parser.add_argument("--training-model-id", default="")
    parser.add_argument("--run-real-training", action="store_true")
    parser.add_argument("--training-steps", type=int, default=12)
    parser.add_argument("--train-sample-limit", type=int, default=40)
    parser.add_argument("--train-max-seq-length", type=int, default=768)
    parser.add_argument("--training-output-dir", type=Path, default=Path("trainer_job_outputs/phase29-feedback-driven-sft-12step"))
    parser.add_argument("--clean-training-output", action="store_true")
    parser.add_argument("--training-timeout-seconds", type=int, default=2400)
    parser.add_argument("--eval-holdout-limit", type=int, default=30)
    parser.add_argument("--eval-max-tokens", type=int, default=192)
    parser.add_argument("--repetition-penalty", type=float, default=1.2)
    parser.add_argument("--run-runtime-reference", action="store_true")
    parser.add_argument("--runtime-reference-limit", type=int, default=5)
    parser.add_argument("--run-dpo-fallback", action="store_true")
    parser.add_argument("--dpo-fallback-sample-limit", type=int, default=12)
    parser.add_argument("--dpo-fallback-output-dir", type=Path, default=Path("trainer_job_outputs/phase29-dpo-fallback-qwen25-0_5b"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = generate_phase29_evidence(args)
    compact = {
        "kind": summary.get("kind"),
        "status": summary.get("status"),
        "candidate_manifest": summary.get("candidate_manifest"),
        "model_selection": summary.get("model_selection"),
        "training_attempt": {
            key: _dict(summary.get("training_attempt")).get(key)
            for key in ("real_training", "training_run", "model_id", "adapter_path", "error_type", "skip_reason")
        },
        "eval_decision": _dict(_dict(summary.get("effective_eval_report")).get("decision")),
        "dpo_fallback": {
            key: _dict(summary.get("dpo_fallback")).get(key)
            for key in ("status", "training_role", "model_id", "recommendation")
        },
        "runtime_reference": {
            key: _dict(summary.get("runtime_reference")).get(key)
            for key in ("status", "real_model_calls", "role", "skip_reason")
        },
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
