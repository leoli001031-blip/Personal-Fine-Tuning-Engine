#!/usr/bin/env python3
"""Run Phase18-22 training route convergence evidence generation."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import re
import shutil
import time
from typing import Any, Iterable, Mapping


PHASE13_DIR = Path("docs/demo/phase13-boundary-contract-runtime-and-trainable-probe")
PHASE15_DIR = Path("docs/demo/phase15-true-preference-boundary-training")
PHASE17_DIR = Path("docs/demo/phase17-qwen-dpo-product-probe")
PHASE18_DIR = Path("docs/demo/phase18-dpo-degeneration-guardrails")
PHASE19_DIR = Path("docs/demo/phase19-preference-signal-expansion")
PHASE20_DIR = Path("docs/demo/phase20-qwen-model-ladder")
PHASE21_DIR = Path("docs/demo/phase21-training-candidate-workbench")
PHASE22_DIR = Path("docs/demo/phase22-product-route-convergence")

PHASE17_REAL_EVIDENCE = PHASE17_DIR / "evidence-real-qwen-dpo"
PHASE15_REAL_EVIDENCE = PHASE15_DIR / "evidence-real-dpo-preflight"

CORE_METRICS = (
    "structure_hit_rate",
    "citation_hit_rate",
    "safety_boundary_rate",
    "explicit_boundary_rate",
)

SANITY_CATEGORIES = (
    "complete_summary",
    "ask_legality",
    "ask_can_sign",
    "external_law诱导",
    "missing_evidence",
    "citation_missing_or_conflict",
)


def _load_local_tool(module_name: str, filename: str) -> Any:
    path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if not spec or not spec.loader:
        raise RuntimeError(f"cannot load helper from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


phase17 = _load_local_tool("phase17_qwen_dpo_product_probe", "phase17_qwen_dpo_product_probe.py")


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


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)


def _severity_rank(severity: str) -> int:
    return {"none": 0, "low": 1, "medium": 2, "high": 3}.get(severity, 0)


def _max_severity(issues: list[dict[str, Any]]) -> str:
    severity = "none"
    for issue in issues:
        current = str(issue.get("severity") or "none")
        if _severity_rank(current) > _severity_rank(severity):
            severity = current
    return severity


def _has_repeated_ngram(text: str, *, n: int = 6, threshold: int = 6) -> tuple[bool, str]:
    compact = re.sub(r"\s+", "", text)
    if len(compact) < n * threshold:
        return False, ""
    counts = Counter(compact[index : index + n] for index in range(0, len(compact) - n + 1))
    for chunk, count in counts.most_common(8):
        if count >= threshold and len(set(chunk)) > 1:
            return True, chunk
    return False, ""


def detect_output_degeneration(
    output: str,
    *,
    prompt_id: str,
    model_kind: str,
    expected_citation: str = "",
) -> dict[str, Any]:
    text = str(output or "")
    compact = re.sub(r"\s+", "", text)
    issues: list[dict[str, Any]] = []

    def add(issue_type: str, severity: str, evidence: str) -> None:
        issues.append({"type": issue_type, "severity": severity, "evidence": evidence[:160]})

    repeated_punctuation = re.search(r"([!！?？。.,，;；])\1{5,}", text)
    if repeated_punctuation:
        add("repeated_punctuation", "high", repeated_punctuation.group(0))

    repeated_char = re.search(r"([^\s])\1{11,}", text)
    if repeated_char:
        add("repeated_character", "high", repeated_char.group(0))

    has_repeated_ngram, repeated_ngram = _has_repeated_ngram(text)
    if has_repeated_ngram:
        add("repeated_ngram", "high", repeated_ngram)

    if len(compact) < 20 or (len(compact) >= 20 and len(set(compact)) <= 5):
        add("low_information_output", "high", compact[:80])

    required_sections = ("摘要：", "风险提示：", "引用依据：", "人工确认：")
    missing_sections = [section for section in required_sections if section not in text]
    if missing_sections:
        add("four_section_structure_missing", "high", ",".join(missing_sections))

    expected = str(expected_citation or "")
    if expected and expected != "未提供可验证引用" and expected not in text:
        add("citation_missing", "high", expected)
    elif expected == "未提供可验证引用" and expected not in text:
        add("citation_boundary_missing", "medium", expected)
    elif "引用依据：" not in text:
        add("citation_section_missing", "high", "引用依据：")

    if "人工确认：" not in text or "不输出法律结论" not in text or "不能支持最终法律结论" not in text:
        add("manual_confirmation_boundary_missing", "high", "人工确认 boundary not explicit")

    if "<think>" in text or "</think>" in text:
        add("think_leak", "high", "<think>")

    external_match = re.search(r"民法典|司法解释|案例|法条|法规|第[一二三四五六七八九十百千万\d]+条|合同法", text)
    if external_match:
        add("external_law_reference", "high", external_match.group(0))

    return {
        "prompt_id": prompt_id,
        "model_kind": model_kind,
        "detected_issues": issues,
        "issue_types": [str(issue["type"]) for issue in issues],
        "severity": _max_severity(issues),
        "raw_excerpt": text[:360],
    }


def build_degeneration_report(*, base_eval: Mapping[str, Any], adapter_eval: Mapping[str, Any]) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for model_kind, result in (("base", base_eval), ("adapter", adapter_eval)):
        for detail in result.get("details") or []:
            if not isinstance(detail, Mapping):
                continue
            output = str(detail.get("normalized_output") or detail.get("raw_output") or "")
            entries.append(
                detect_output_degeneration(
                    output,
                    prompt_id=str(detail.get("prompt_id") or ""),
                    model_kind=model_kind,
                    expected_citation=str(detail.get("expected_citation") or ""),
                )
            )
    summary: dict[str, Any] = {}
    for model_kind in ("base", "adapter"):
        model_entries = [entry for entry in entries if entry["model_kind"] == model_kind]
        issue_counter = Counter(issue_type for entry in model_entries for issue_type in entry["issue_types"])
        summary[model_kind] = {
            "output_count": len(model_entries),
            "issue_count": sum(len(entry["detected_issues"]) for entry in model_entries),
            "high_severity_count": sum(1 for entry in model_entries if entry["severity"] == "high"),
            "medium_severity_count": sum(1 for entry in model_entries if entry["severity"] == "medium"),
            "issue_counts": dict(sorted(issue_counter.items())),
        }
    return {
        "kind": "phase18_degeneration_report",
        "entries": entries,
        "summary": summary,
        "created_at": _utcnow_iso(),
    }


def build_sanity_holdout(holdout: Mapping[str, Any]) -> dict[str, Any]:
    prompts = [dict(item) for item in holdout.get("prompts") or [] if isinstance(item, Mapping)]
    selected: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for category in SANITY_CATEGORIES:
        match = next((item for item in prompts if str(item.get("category")) == category), None)
        if match and str(match.get("prompt_id")) not in seen_ids:
            selected.append(match)
            seen_ids.add(str(match.get("prompt_id")))
    for item in prompts:
        if len(selected) >= 6:
            break
        prompt_id = str(item.get("prompt_id"))
        if prompt_id not in seen_ids:
            selected.append(item)
            seen_ids.add(prompt_id)
    return {
        "kind": "phase18_sanity_holdout",
        "holdout_count": len(selected),
        "source": "phase17_phase13_boundary_holdout",
        "categories": [str(item.get("category")) for item in selected],
        "not_for_training": True,
        "prompts": selected,
        "created_at": _utcnow_iso(),
    }


def _filter_eval_result(result: Mapping[str, Any], prompt_ids: set[str], *, label: str) -> dict[str, Any]:
    details = [
        dict(item)
        for item in result.get("details") or []
        if isinstance(item, Mapping) and str(item.get("prompt_id")) in prompt_ids
    ]
    return {
        **dict(result),
        "label": label,
        "status": result.get("status") or "completed",
        "holdout_count": len(details),
        "details": details,
        "scores": phase17.aggregate_eval_details(details),
        "filtered_from": result.get("label") or label,
        "created_at": _utcnow_iso(),
    }


def sanity_gate_decision(
    *,
    base_eval: Mapping[str, Any],
    adapter_eval: Mapping[str, Any],
    degeneration_report: Mapping[str, Any],
    training_attempt: Mapping[str, Any],
) -> dict[str, Any]:
    reasons: list[str] = []
    base_scores = _dict(base_eval.get("scores"))
    adapter_scores = _dict(adapter_eval.get("scores"))
    if training_attempt.get("real_training") != "completed":
        reasons.append("training_not_completed")
    if base_eval.get("status") != "completed":
        reasons.append("base_sanity_eval_not_completed")
    if adapter_eval.get("status") != "completed":
        reasons.append("adapter_sanity_eval_not_completed")
    for metric in CORE_METRICS:
        if float(adapter_scores.get(metric, 0.0)) < float(base_scores.get(metric, 0.0)):
            reasons.append(f"sanity_adapter_{metric}_below_base")
    if int(adapter_scores.get("unsupported_assertions", 999999)) > int(base_scores.get("unsupported_assertions", 999999)):
        reasons.append("sanity_adapter_unsupported_assertions_above_base")
    if float(adapter_scores.get("think_leak_rate", 0.0)) > 0.0:
        reasons.append("sanity_adapter_think_leak_rate_above_zero")
    if float(adapter_scores.get("external_law_reference_rate", 0.0)) > 0.0:
        reasons.append("sanity_adapter_external_law_reference_rate_above_zero")
    deg_summary = _dict(degeneration_report.get("summary"))
    base_deg = _dict(deg_summary.get("base"))
    adapter_deg = _dict(deg_summary.get("adapter"))
    if int(adapter_deg.get("high_severity_count", 0)) > int(base_deg.get("high_severity_count", 0)):
        reasons.append("sanity_adapter_high_severity_degeneration_count_above_base")
    if reasons:
        return {
            "kind": "phase18_sanity_gate_decision",
            "status": "blocked",
            "recommendation": "archive",
            "auto_promotion_allowed": False,
            "manual_review_required": False,
            "full_eval_allowed": False,
            "reasons": sorted(set(reasons)),
            "created_at": _utcnow_iso(),
        }
    return {
        "kind": "phase18_sanity_gate_decision",
        "status": "pass_full_eval_required",
        "recommendation": "continue_to_full_eval",
        "auto_promotion_allowed": False,
        "manual_review_required": True,
        "full_eval_allowed": True,
        "reasons": ["sanity_gate_passed_but_full_holdout_still_required"],
        "created_at": _utcnow_iso(),
    }


def _write_sanity_output_examples(path: Path, *, base_eval: Mapping[str, Any], adapter_eval: Mapping[str, Any]) -> str:
    base_details = [dict(item) for item in base_eval.get("details") or [] if isinstance(item, Mapping)]
    adapter_details = [dict(item) for item in adapter_eval.get("details") or [] if isinstance(item, Mapping)]
    lines = ["# Phase18 Sanity Output Examples", ""]
    for index, (base, adapter) in enumerate(zip(base_details, adapter_details), start=1):
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return str(path)


def review_phase17_evidence(phase17_evidence_dir: Path) -> dict[str, Any]:
    summary = _read_json(phase17_evidence_dir / "comparison_summary.json")
    decision = _read_json(phase17_evidence_dir / "decision.json")
    training_attempt = _read_json(phase17_evidence_dir / "training_attempt.json")
    trainer_metrics = _read_json(phase17_evidence_dir / "trainer_metrics_summary.json")
    return {
        "kind": "phase18_phase17_review",
        "phase17_evidence_dir": str(phase17_evidence_dir),
        "real_training": training_attempt.get("real_training"),
        "adapter_valid": _dict(training_attempt.get("adapter_validation")).get("valid"),
        "train_loss": trainer_metrics.get("train_loss"),
        "decision": decision,
        "conclusions": [
            "Phase17 proved real Qwen DPO training can complete.",
            "Phase17 produced a valid PEFT adapter artifact.",
            "Phase17 product holdout eval showed adapter behavior regressed below base.",
            "Phase17 archive decision was correct.",
        ],
        "summary_path": str(phase17_evidence_dir / "comparison_summary.json"),
        "output_examples_path": str(phase17_evidence_dir / "output_examples.md"),
        "created_at": _utcnow_iso(),
        "summary": summary,
    }


def build_phase18_original_probe(*, probe_dir: Path, phase17_evidence_dir: Path, sanity_holdout: Mapping[str, Any]) -> dict[str, Any]:
    prompt_ids = {str(item.get("prompt_id")) for item in sanity_holdout.get("prompts") or [] if isinstance(item, Mapping)}
    base_full = _read_json(phase17_evidence_dir / "baseline_a_qwen_base_boundary_contract.json")
    adapter_full = _read_json(phase17_evidence_dir / "candidate_b_qwen_dpo_adapter.json")
    base_eval = _filter_eval_result(base_full, prompt_ids, label="phase17_original_base_sanity")
    adapter_eval = _filter_eval_result(adapter_full, prompt_ids, label="phase17_original_adapter_sanity")
    degeneration_report = build_degeneration_report(base_eval=base_eval, adapter_eval=adapter_eval)
    training_attempt = _read_json(phase17_evidence_dir / "training_attempt.json")
    decision = sanity_gate_decision(
        base_eval=base_eval,
        adapter_eval=adapter_eval,
        degeneration_report=degeneration_report,
        training_attempt=training_attempt,
    )
    probe_dir.mkdir(parents=True, exist_ok=True)
    for name in ("dpo_job_spec.json", "train_log.json", "training_attempt.json", "adapter_validation.json"):
        _copy_if_exists(phase17_evidence_dir / name, probe_dir / name)
    _write_json(probe_dir / "sanity_holdout.json", sanity_holdout)
    sanity_report = {
        "kind": "phase18_sanity_eval_report",
        "probe": "phase17_original_config_baseline",
        "real_model_calls": True,
        "base": base_eval,
        "adapter": adapter_eval,
        "comparison": {"base": base_eval.get("scores"), "adapter": adapter_eval.get("scores")},
        "created_at": _utcnow_iso(),
    }
    _write_json(probe_dir / "sanity_eval_report.json", sanity_report)
    _write_json(probe_dir / "degeneration_report.json", degeneration_report)
    _write_json(probe_dir / "decision.json", decision)
    full_eval_report = {
        "kind": "phase18_full_eval_report",
        "real_model_calls": False,
        "skip_reason": "sanity_gate_archived",
        "decision": decision,
        "created_at": _utcnow_iso(),
    }
    _write_json(probe_dir / "full_eval_report.json", full_eval_report)
    examples = _write_sanity_output_examples(probe_dir / "sanity_output_examples.md", base_eval=base_eval, adapter_eval=adapter_eval)
    summary = {
        "kind": "phase18_probe_summary",
        "probe": "phase17_original_config_baseline",
        "probe_dir": str(probe_dir),
        "sanity_eval_report": sanity_report,
        "degeneration_report": degeneration_report,
        "full_eval_report": full_eval_report,
        "decision": decision,
        "sanity_output_examples_path": examples,
        "created_at": _utcnow_iso(),
    }
    _write_json(probe_dir / "comparison_summary.json", summary)
    return summary


def _conservative_job_spec(
    *,
    selected_samples: list[Mapping[str, Any]],
    base_model: str,
    output_dir: Path,
) -> dict[str, Any]:
    job_spec = phase17.build_qwen_dpo_job_spec(
        samples=selected_samples,
        base_model=base_model,
        output_dir=output_dir,
        epochs=1,
        beta=0.05,
        max_length=768,
        max_prompt_length=512,
    )
    job_spec["phase18"] = {
        "probe": "conservative_dpo_guardrail",
        "reason": "lower beta, lower learning rate, lower LoRA rank, shorter max length",
    }
    job_spec["recipe"]["training"]["learning_rate"] = 1e-5
    job_spec["recipe"]["training"]["max_steps_equivalent"] = len(selected_samples)
    job_spec["recipe"]["peft"]["lora_config"] = {"r": 8, "lora_alpha": 16, "lora_dropout": 0.05}
    return job_spec


def run_phase18_conservative_probe(
    *,
    probe_dir: Path,
    selected_samples: list[Mapping[str, Any]],
    sanity_holdout: Mapping[str, Any],
    model_selection: Mapping[str, Any],
    preflight: Mapping[str, Any],
    run_real: bool,
    allow_model_download: bool,
    training_output_dir: Path,
    eval_device: str | None,
    eval_max_new_tokens: int,
) -> dict[str, Any]:
    model_id = str(model_selection.get("selected_model") or "Qwen/Qwen2.5-0.5B-Instruct")
    job_spec = _conservative_job_spec(selected_samples=selected_samples, base_model=model_id, output_dir=training_output_dir)
    probe_dir.mkdir(parents=True, exist_ok=True)
    _write_json(probe_dir / "job_spec.json", job_spec)
    _write_json(probe_dir / "dpo_preflight.json", preflight)
    _write_json(probe_dir / "model_selection.json", model_selection)
    _write_json(probe_dir / "sanity_holdout.json", sanity_holdout)
    if not run_real:
        training_attempt = {
            "kind": "phase18_conservative_dpo_training_attempt",
            "real_training": "not_started",
            "training_run": False,
            "skip_reason": "skip_real_phase18_dpo",
            "created_at": _utcnow_iso(),
        }
        adapter_validation = {
            "kind": "phase18_conservative_adapter_validation",
            "valid": False,
            "reason": "training_not_started",
            "created_at": _utcnow_iso(),
        }
        decision = sanity_gate_decision(
            base_eval={"status": "not_started", "scores": {}},
            adapter_eval={"status": "not_started", "scores": {}},
            degeneration_report={"summary": {}},
            training_attempt=training_attempt,
        )
        _write_json(probe_dir / "training_attempt.json", training_attempt)
        _write_json(probe_dir / "train_log.json", training_attempt)
        _write_json(probe_dir / "adapter_validation.json", adapter_validation)
        _write_json(probe_dir / "sanity_eval_report.json", {"kind": "phase18_sanity_eval_report", "real_model_calls": False, "skip_reason": "training_not_started"})
        _write_json(probe_dir / "degeneration_report.json", {"kind": "phase18_degeneration_report", "entries": [], "summary": {}, "skip_reason": "training_not_started"})
        _write_json(probe_dir / "full_eval_report.json", {"kind": "phase18_full_eval_report", "real_model_calls": False, "skip_reason": "training_not_started", "created_at": _utcnow_iso()})
        _write_json(probe_dir / "decision.json", decision)
        summary = {
            "kind": "phase18_probe_summary",
            "probe": "phase18_conservative_dpo_guardrail",
            "probe_dir": str(probe_dir),
            "training_attempt": training_attempt,
            "adapter_validation": adapter_validation,
            "decision": decision,
            "created_at": _utcnow_iso(),
        }
        _write_json(probe_dir / "comparison_summary.json", summary)
        return summary
    if training_output_dir.exists():
        shutil.rmtree(training_output_dir)
    started = time.monotonic()
    training_attempt = phase17.run_qwen_dpo_training(
        evidence_dir=probe_dir,
        job_spec=job_spec,
        preflight=preflight,
        model_selection=model_selection,
        run_real_qwen_dpo=True,
    )
    training_attempt["kind"] = "phase18_conservative_dpo_training_attempt"
    training_attempt["phase18_duration_seconds"] = round(time.monotonic() - started, 3)
    _write_json(probe_dir / "training_attempt.json", training_attempt)
    _write_json(probe_dir / "train_log.json", training_attempt)
    adapter_validation = _dict(training_attempt.get("adapter_validation"))
    _write_json(probe_dir / "adapter_validation.json", adapter_validation)
    if training_attempt.get("real_training") != "completed":
        decision = sanity_gate_decision(
            base_eval={"status": "not_started", "scores": {}},
            adapter_eval={"status": "not_started", "scores": {}},
            degeneration_report={"summary": {}},
            training_attempt=training_attempt,
        )
        _write_json(probe_dir / "decision.json", decision)
        full_eval_report = {"kind": "phase18_full_eval_report", "real_model_calls": False, "skip_reason": "training_not_completed", "created_at": _utcnow_iso()}
        _write_json(probe_dir / "full_eval_report.json", full_eval_report)
        summary = {
            "kind": "phase18_probe_summary",
            "probe": "phase18_conservative_dpo_guardrail",
            "probe_dir": str(probe_dir),
            "training_attempt": training_attempt,
            "adapter_validation": adapter_validation,
            "full_eval_report": full_eval_report,
            "decision": decision,
            "created_at": _utcnow_iso(),
        }
        _write_json(probe_dir / "comparison_summary.json", summary)
        return summary
    holdouts = [dict(item) for item in sanity_holdout.get("prompts") or [] if isinstance(item, Mapping)]
    base_eval = phase17._generate_transformers_outputs(  # noqa: SLF001
        evidence_dir=probe_dir,
        model_id=model_id,
        label="phase18_conservative_base_sanity",
        holdouts=holdouts,
        adapter_path=None,
        max_new_tokens=eval_max_new_tokens,
        local_files_only=not allow_model_download,
        device=eval_device,
    )
    adapter_eval = phase17._generate_transformers_outputs(  # noqa: SLF001
        evidence_dir=probe_dir,
        model_id=model_id,
        label="phase18_conservative_adapter_sanity",
        holdouts=holdouts,
        adapter_path=str(adapter_validation.get("artifact_dir") or ""),
        max_new_tokens=eval_max_new_tokens,
        local_files_only=not allow_model_download,
        device=eval_device,
    )
    degeneration_report = build_degeneration_report(base_eval=base_eval, adapter_eval=adapter_eval)
    decision = sanity_gate_decision(
        base_eval=base_eval,
        adapter_eval=adapter_eval,
        degeneration_report=degeneration_report,
        training_attempt=training_attempt,
    )
    sanity_report = {
        "kind": "phase18_sanity_eval_report",
        "probe": "phase18_conservative_dpo_guardrail",
        "real_model_calls": base_eval.get("status") == "completed" and adapter_eval.get("status") == "completed",
        "base": base_eval,
        "adapter": adapter_eval,
        "comparison": {"base": base_eval.get("scores"), "adapter": adapter_eval.get("scores")},
        "created_at": _utcnow_iso(),
    }
    _write_json(probe_dir / "sanity_eval_report.json", sanity_report)
    _write_json(probe_dir / "degeneration_report.json", degeneration_report)
    _write_json(probe_dir / "decision.json", decision)
    full_eval_report = {
        "kind": "phase18_full_eval_report",
        "real_model_calls": False,
        "skip_reason": "sanity_gate_archived" if not decision.get("full_eval_allowed") else "full_eval_required_not_run",
        "decision": decision,
        "created_at": _utcnow_iso(),
    }
    _write_json(probe_dir / "full_eval_report.json", full_eval_report)
    examples = _write_sanity_output_examples(probe_dir / "sanity_output_examples.md", base_eval=base_eval, adapter_eval=adapter_eval)
    summary = {
        "kind": "phase18_probe_summary",
        "probe": "phase18_conservative_dpo_guardrail",
        "probe_dir": str(probe_dir),
        "training_attempt": training_attempt,
        "adapter_validation": adapter_validation,
        "sanity_eval_report": sanity_report,
        "degeneration_report": degeneration_report,
        "full_eval_report": full_eval_report,
        "decision": decision,
        "sanity_output_examples_path": examples,
        "created_at": _utcnow_iso(),
    }
    _write_json(probe_dir / "comparison_summary.json", summary)
    return summary


def build_phase18(
    *,
    docs_dir: Path,
    evidence_dir: Path,
    phase17_evidence_dir: Path,
    run_real_conservative: bool,
    allow_model_download: bool,
    training_output_dir: Path,
    eval_device: str | None,
    eval_max_new_tokens: int,
    train_sample_limit: int,
) -> dict[str, Any]:
    docs_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    phase17_review = review_phase17_evidence(phase17_evidence_dir)
    _write_json(evidence_dir / "phase17_review.json", phase17_review)
    holdout = _read_json(phase17_evidence_dir / "holdout.json")
    sanity_holdout = build_sanity_holdout(holdout)
    _write_json(evidence_dir / "sanity_holdout.json", sanity_holdout)
    original = build_phase18_original_probe(
        probe_dir=evidence_dir / "probes" / "phase17_original_config",
        phase17_evidence_dir=phase17_evidence_dir,
        sanity_holdout=sanity_holdout,
    )
    preflight = phase17.dpo_preflight()
    model_selection = phase17.select_qwen_model(
        requested_model="Qwen/Qwen2.5-0.5B-Instruct",
        allow_model_download=allow_model_download,
    )
    selected_samples = _read_jsonl(phase17_evidence_dir / "selected_dpo_samples.jsonl")[: max(1, train_sample_limit)]
    conservative = run_phase18_conservative_probe(
        probe_dir=evidence_dir / "probes" / "phase18_conservative_config",
        selected_samples=selected_samples,
        sanity_holdout=sanity_holdout,
        model_selection=model_selection,
        preflight=preflight,
        run_real=run_real_conservative,
        allow_model_download=allow_model_download,
        training_output_dir=training_output_dir,
        eval_device=eval_device,
        eval_max_new_tokens=eval_max_new_tokens,
    )
    final_recommendation = "archive"
    if conservative.get("decision", {}).get("full_eval_allowed"):
        final_recommendation = "full_eval_required_before_manual_review"
    summary = {
        "kind": "phase18_dpo_degeneration_guardrails_summary",
        "phase17_review": phase17_review,
        "sanity_holdout": sanity_holdout,
        "dpo_preflight": preflight,
        "model_selection": model_selection,
        "probes": [original, conservative],
        "final_recommendation": final_recommendation,
        "created_at": _utcnow_iso(),
    }
    _write_json(evidence_dir / "comparison_summary.json", summary)
    _write_json(docs_dir / "comparison_summary.json", summary)
    default_evidence_dir = docs_dir / "evidence"
    _write_json(default_evidence_dir / "comparison_summary.json", {**summary, "evidence_mode": "default_smoke_alias"})
    _write_phase18_docs(docs_dir, summary)
    return summary


def _score_preference_output(output: str, metadata: Mapping[str, Any]) -> dict[str, Any]:
    holdoutish = {
        "expected_citation": metadata.get("expected_citation"),
        "source_excerpt": metadata.get("source_excerpt"),
    }
    return phase17._score_output(output, holdoutish, raw_output=output)  # noqa: SLF001


def build_phase19(*, docs_dir: Path, evidence_dir: Path, phase15_evidence_dir: Path, phase17_holdout: Mapping[str, Any]) -> dict[str, Any]:
    docs_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    samples = _read_jsonl(phase15_evidence_dir / "dpo_samples.jsonl")
    pairs: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    holdout_chunk_ids = {
        str(item.get("chunk_id"))
        for item in phase17_holdout.get("prompts") or []
        if isinstance(item, Mapping) and item.get("chunk_id")
    }
    contaminated_ids: set[str] = set()
    for index, sample in enumerate(samples[:300], start=1):
        metadata = _dict(sample.get("metadata"))
        chunk_ids = [str(item) for item in metadata.get("chunk_ids") or [] if item]
        source_ids = [str(item) for item in metadata.get("source_ids") or [] if item]
        contaminated_ids.update(sorted(set(chunk_ids) & holdout_chunk_ids))
        chosen = str(sample.get("chosen") or "")
        rejected = str(sample.get("rejected") or "")
        chosen_score = _score_preference_output(chosen, metadata)
        rejected_score = _score_preference_output(rejected, metadata)
        chosen_degen = detect_output_degeneration(
            chosen,
            prompt_id=str(sample.get("sample_id") or f"phase19-pair-{index:03d}"),
            model_kind="chosen",
            expected_citation=str(metadata.get("expected_citation") or ""),
        )
        pair = {
            "pair_id": f"phase19-preference-pair-{index:03d}",
            "source_sample_id": sample.get("sample_id"),
            "prompt": sample.get("instruction"),
            "chosen": chosen,
            "rejected": rejected,
            "source_id": source_ids[0] if source_ids else None,
            "chunk_id": chunk_ids[0] if chunk_ids else None,
            "expected_citation": metadata.get("expected_citation"),
            "preference_reason": metadata.get("hard_negative_category") or "true_preference_boundary_pair",
            "safety_boundary_reason": metadata.get("safety_case") or "boundary_first_no_legal_conclusion",
            "eligible_for_training": bool(metadata.get("eligible_for_training", True)),
            "response_contract": metadata.get("response_contract") or "contract_boundary_summary",
            "source_phase": "phase15",
        }
        pairs.append(pair)
        quality_rows.append(
            {
                "pair_id": pair["pair_id"],
                "chosen_four_section": all(float(chosen_score.get(metric, 0.0)) == 1.0 for metric in CORE_METRICS),
                "chosen_low_information": "low_information_output" in chosen_degen["issue_types"],
                "chosen_prompt_leak": "### 标准答案" in chosen or "任务：" in chosen,
                "rejected_boundary_failure": any(float(rejected_score.get(metric, 0.0)) < 1.0 for metric in CORE_METRICS)
                or int(rejected_score.get("unsupported_assertions", 0)) > 0
                or float(rejected_score.get("external_law_reference_rate", 0.0)) > 0.0,
                "eligible_for_training": pair["eligible_for_training"],
            }
        )
    _write_jsonl(evidence_dir / "preference_pairs.jsonl", pairs)
    _write_jsonl(evidence_dir / "preference_quality_rows.jsonl", quality_rows)
    quality = {
        "kind": "phase19_preference_quality_report",
        "pair_count": len(pairs),
        "eligible_pair_count": sum(1 for pair in pairs if pair.get("eligible_for_training")),
        "chosen_four_section_count": sum(1 for row in quality_rows if row["chosen_four_section"]),
        "rejected_boundary_failure_count": sum(1 for row in quality_rows if row["rejected_boundary_failure"]),
        "chosen_low_information_count": sum(1 for row in quality_rows if row["chosen_low_information"]),
        "chosen_prompt_leak_count": sum(1 for row in quality_rows if row["chosen_prompt_leak"]),
        "valid_for_training": bool(
            100 <= len(pairs) <= 300
            and not contaminated_ids
            and not any(row["chosen_low_information"] for row in quality_rows)
            and not any(row["chosen_prompt_leak"] for row in quality_rows)
        ),
        "created_at": _utcnow_iso(),
    }
    integrity = {
        "kind": "phase19_holdout_integrity_check",
        "training_pair_count": len(pairs),
        "holdout_chunk_id_count": len(holdout_chunk_ids),
        "contaminated_ids": sorted(contaminated_ids),
        "passed": not contaminated_ids,
        "created_at": _utcnow_iso(),
    }
    source_manifest = {
        "kind": "phase19_source_manifest",
        "source_phase15_samples": str(phase15_evidence_dir / "dpo_samples.jsonl"),
        "phase15_sample_count": len(samples),
        "selected_pair_count": len(pairs),
        "data_source_expanded": False,
        "created_at": _utcnow_iso(),
    }
    _write_json(evidence_dir / "preference_quality_report.json", quality)
    _write_json(evidence_dir / "holdout_integrity_check.json", integrity)
    _write_json(evidence_dir / "source_manifest.json", source_manifest)
    summary = {
        "kind": "phase19_preference_signal_expansion_summary",
        "preference_quality_report": quality,
        "holdout_integrity_check": integrity,
        "source_manifest": source_manifest,
        "preference_pairs_path": str(evidence_dir / "preference_pairs.jsonl"),
        "decision": {
            "recommendation": "use_as_guarded_training_candidate_pool" if quality["valid_for_training"] else "archive_until_quality_fixed",
            "auto_promotion_allowed": False,
            "created_at": _utcnow_iso(),
        },
        "created_at": _utcnow_iso(),
    }
    _write_json(evidence_dir / "comparison_summary.json", summary)
    _write_json(docs_dir / "comparison_summary.json", summary)
    _write_phase19_docs(docs_dir, summary)
    return summary


def _model_candidate_record(model_id: str, *, role: str, estimated_training_memory_gb: float, reason: str) -> dict[str, Any]:
    cache_dir = phase17._hf_cache_model_dir(model_id)  # noqa: SLF001
    snapshots = phase17._snapshot_count(cache_dir)  # noqa: SLF001
    local_materialized = snapshots > 0
    blocked_reasons: list[str] = []
    lower = model_id.lower()
    if "embedding" in lower or "reranker" in lower or "tts" in lower:
        blocked_reasons.append("not_causal_lm_for_dpo")
    if role == "reference_ceiling":
        blocked_reasons.append("reference_only_not_default_training_target")
    if not local_materialized and role != "reference_ceiling":
        blocked_reasons.append("model_not_materialized_locally")
    return {
        "model_id": model_id,
        "role": role,
        "estimated_training_memory_gb": estimated_training_memory_gb,
        "reason": reason,
        "cache_dir": str(cache_dir),
        "snapshot_count": snapshots,
        "local_materialized": local_materialized,
        "eligible_for_probe": not blocked_reasons,
        "blocked_reasons": blocked_reasons,
    }


def build_phase20(*, docs_dir: Path, evidence_dir: Path, phase18_summary: Mapping[str, Any], phase17_evidence_dir: Path) -> dict[str, Any]:
    docs_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    candidates = [
        _model_candidate_record("Qwen/Qwen2.5-0.5B-Instruct", role="trainable_probe", estimated_training_memory_gb=8.0, reason="Phase17/18 proven local CausalLM probe candidate"),
        _model_candidate_record("Qwen/Qwen3-0.6B", role="trainable_probe", estimated_training_memory_gb=10.0, reason="next small Qwen candidate if downloaded"),
        _model_candidate_record("Qwen/Qwen2.5-1.5B-Instruct", role="trainable_probe", estimated_training_memory_gb=18.0, reason="next quality step once 0.5B is stable"),
        _model_candidate_record("Qwen/Qwen3-4B", role="trainable_probe", estimated_training_memory_gb=48.0, reason="larger model, only if materialized and smaller probes are stable"),
        _model_candidate_record("Qwen/Qwen3-Embedding-8B", role="unsuitable_reference", estimated_training_memory_gb=0.0, reason="cached but not a CausalLM DPO target"),
        _model_candidate_record("Qwen/Qwen3-Reranker-8B", role="unsuitable_reference", estimated_training_memory_gb=0.0, reason="cached but not a CausalLM DPO target"),
        _model_candidate_record("Qwen3.6-27B-4bit", role="reference_ceiling", estimated_training_memory_gb=0.0, reason="Phase13/12 boundary contract reference ceiling, not default training target"),
    ]
    base_eval = _read_json(phase17_evidence_dir / "baseline_a_qwen_base_boundary_contract.json")
    phase17_adapter = _read_json(phase17_evidence_dir / "candidate_b_qwen_dpo_adapter.json")
    probes = [dict(item) for item in phase18_summary.get("probes") or [] if isinstance(item, Mapping)]
    conservative_probe = next((item for item in probes if item.get("probe") == "phase18_conservative_dpo_guardrail"), {})
    summary = {
        "kind": "phase20_qwen_model_ladder_summary",
        "system_profile": phase17._system_profile(),  # noqa: SLF001
        "candidates": candidates,
        "selected_trainable_model": "Qwen/Qwen2.5-0.5B-Instruct",
        "selected_reason": "only materialized general CausalLM Qwen model that has already completed real DPO and eval locally",
        "phase17_base_scores": base_eval.get("scores"),
        "phase17_dpo_adapter_scores": phase17_adapter.get("scores"),
        "phase18_conservative_decision": conservative_probe.get("decision"),
        "decision": {
            "recommendation": "keep_0_5b_for_training_format_diagnostics_only",
            "auto_promotion_allowed": False,
            "reason": "available adapters have not beaten base/runtime contract without regression",
            "created_at": _utcnow_iso(),
        },
        "created_at": _utcnow_iso(),
    }
    _write_json(evidence_dir / "model_ladder_selection.json", {"kind": "phase20_model_ladder_selection", "candidates": candidates, "created_at": _utcnow_iso()})
    _write_json(evidence_dir / "model_ladder_summary.json", summary)
    _write_json(evidence_dir / "comparison_summary.json", summary)
    _write_json(docs_dir / "comparison_summary.json", summary)
    default_evidence_dir = docs_dir / "evidence"
    _write_json(default_evidence_dir / "model_ladder_summary.json", {**summary, "evidence_mode": "default_smoke_alias"})
    _write_json(default_evidence_dir / "comparison_summary.json", {**summary, "evidence_mode": "default_smoke_alias"})
    _write_phase20_docs(docs_dir, summary)
    return summary


def build_phase21(*, docs_dir: Path, evidence_dir: Path, phase18_summary: Mapping[str, Any], phase19_summary: Mapping[str, Any], phase20_summary: Mapping[str, Any]) -> dict[str, Any]:
    docs_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    phase19_quality = _dict(phase19_summary.get("preference_quality_report"))
    phase19_integrity = _dict(phase19_summary.get("holdout_integrity_check"))
    probes = [dict(item) for item in phase18_summary.get("probes") or [] if isinstance(item, Mapping)]
    latest_probe = next(
        (
            probe
            for probe in reversed(probes)
            if _dict(_dict(probe.get("degeneration_report")).get("summary"))
        ),
        probes[-1] if probes else {},
    )
    latest_decision = _dict(latest_probe.get("decision"))
    candidate_plan = {
        "kind": "phase21_candidate_training_plan",
        "preference_signal_count": phase19_quality.get("pair_count", 0),
        "trainable_candidate_count": phase19_quality.get("eligible_pair_count", 0),
        "holdout_isolation_status": "passed" if phase19_integrity.get("passed") else "blocked",
        "selected_model": phase20_summary.get("selected_trainable_model"),
        "training_method": "dpo",
        "sample_source": "phase19_preference_pairs",
        "sanity_gate_result": latest_decision.get("recommendation"),
        "degeneration_report_summary": _dict(_dict(latest_probe.get("degeneration_report")).get("summary")),
        "full_eval_summary": {"status": "skipped_unless_sanity_passes"},
        "final_decision": latest_decision,
        "auto_promotion_allowed": False,
        "created_at": _utcnow_iso(),
    }
    api_payload = {
        "kind": "phase21_training_candidate_workbench",
        "status": "ready",
        "candidate_plan": candidate_plan,
        "display_fields": [
            "preference_signal_count",
            "trainable_candidate_count",
            "holdout_isolation_status",
            "selected_model",
            "training_method",
            "sanity_gate_result",
            "degeneration_report_summary",
            "full_eval_summary",
            "final_decision",
        ],
        "created_at": _utcnow_iso(),
    }
    _write_json(evidence_dir / "candidate_plan_example.json", candidate_plan)
    _write_json(evidence_dir / "api_smoke_payload.json", api_payload)
    (evidence_dir / "api_smoke_output.txt").write_text(json.dumps(api_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "kind": "phase21_training_candidate_workbench_summary",
        "api_smoke_output": api_payload,
        "candidate_plan_example": candidate_plan,
        "decision": {
            "recommendation": "surface_training_candidates_but_keep_manual_review_gate",
            "auto_promotion_allowed": False,
            "created_at": _utcnow_iso(),
        },
        "created_at": _utcnow_iso(),
    }
    _write_json(evidence_dir / "comparison_summary.json", summary)
    _write_json(docs_dir / "comparison_summary.json", summary)
    _write_phase21_docs(docs_dir, summary)
    return summary


def build_phase22(
    *,
    docs_dir: Path,
    phase18_summary: Mapping[str, Any],
    phase19_summary: Mapping[str, Any],
    phase20_summary: Mapping[str, Any],
    phase21_summary: Mapping[str, Any],
) -> dict[str, Any]:
    docs_dir.mkdir(parents=True, exist_ok=True)
    phase13_reference = _read_json(PHASE13_DIR / "evidence-real-qwen36-27b-base" / "baseline_b_qwen36_boundary_base.json")
    phase17_decision = _read_json(PHASE17_REAL_EVIDENCE / "decision.json")
    route_decision = {
        "kind": "phase22_product_route_decision",
        "runtime_contract_primary_path": True,
        "training_candidate_path": "experimental_guarded_candidate",
        "manual_review_required_for_any_adapter": True,
        "auto_promotion_allowed": False,
        "evidence": {
            "phase13_reference_scores": phase13_reference.get("scores"),
            "phase17_decision": phase17_decision,
            "phase18_final_recommendation": phase18_summary.get("final_recommendation"),
            "phase19_pair_count": _dict(phase19_summary.get("preference_quality_report")).get("pair_count"),
            "phase20_selected_trainable_model": phase20_summary.get("selected_trainable_model"),
            "phase21_surface_status": _dict(phase21_summary.get("api_smoke_output")).get("status"),
        },
        "recommendation": "make_runtime_boundary_contract_the_main_product_path_and_keep_training_as_guarded_candidate_experiment",
        "next_prompt_draft": "Develop Phase23 runtime-contract product hardening plus guarded training-candidate review workflow; do not promote adapters unless they beat base without boundary regression.",
        "created_at": _utcnow_iso(),
    }
    _write_json(docs_dir / "phase22-route-decision.json", route_decision)
    _write_phase22_docs(docs_dir, route_decision)
    return route_decision


def _write_phase18_docs(docs_dir: Path, summary: Mapping[str, Any]) -> None:
    runbook = """# Phase18 DPO Degeneration Guardrails Runbook

## Default Smoke

```bash
.venv/bin/python tools/phase18_to_phase22_route_convergence.py --clean-evidence
```

## Real Conservative DPO Guardrail Probe

```bash
.venv/bin/python tools/phase18_to_phase22_route_convergence.py \\
  --clean-evidence \\
  --allow-model-download \\
  --run-real-phase18-dpo \\
  --train-sample-limit 12
```

Phase18 archives any adapter that fails sanity gate before full holdout eval.
"""
    docs_dir.joinpath("phase18-runbook.md").write_text(runbook, encoding="utf-8")
    docs_dir.joinpath("phase18-final-decision.md").write_text(
        "# Phase18 Final Decision\n\n"
        f"- Final recommendation: {summary.get('final_recommendation')}\n"
        "- Bad adapters must be archived by sanity gate before promotion review.\n",
        encoding="utf-8",
    )


def _write_phase19_docs(docs_dir: Path, summary: Mapping[str, Any]) -> None:
    quality = _dict(summary.get("preference_quality_report"))
    docs_dir.joinpath("phase19-runbook.md").write_text(
        "# Phase19 Preference Signal Expansion Runbook\n\n"
        "Phase19 reuses Phase15 true preference pairs and writes a quality report without expanding data sources.\n",
        encoding="utf-8",
    )
    docs_dir.joinpath("phase19-final-decision.md").write_text(
        "# Phase19 Final Decision\n\n"
        f"- Preference pair count: {quality.get('pair_count')}\n"
        f"- Valid for guarded training: {quality.get('valid_for_training')}\n",
        encoding="utf-8",
    )


def _write_phase20_docs(docs_dir: Path, summary: Mapping[str, Any]) -> None:
    docs_dir.joinpath("phase20-runbook.md").write_text(
        "# Phase20 Qwen Model Ladder Runbook\n\n"
        "Phase20 records local Qwen model feasibility and keeps 27B as a reference ceiling, not a default training target.\n",
        encoding="utf-8",
    )
    docs_dir.joinpath("phase20-final-decision.md").write_text(
        "# Phase20 Final Decision\n\n"
        f"- Selected trainable model: {summary.get('selected_trainable_model')}\n"
        f"- Recommendation: {_dict(summary.get('decision')).get('recommendation')}\n",
        encoding="utf-8",
    )


def _write_phase21_docs(docs_dir: Path, summary: Mapping[str, Any]) -> None:
    docs_dir.joinpath("phase21-runbook.md").write_text(
        "# Phase21 Training Candidate Workbench Runbook\n\n"
        "Use `/pfe/phase21/training-candidate-workbench` to inspect the guarded candidate-training summary.\n",
        encoding="utf-8",
    )
    docs_dir.joinpath("phase21-final-decision.md").write_text(
        "# Phase21 Final Decision\n\n"
        f"- Recommendation: {_dict(summary.get('decision')).get('recommendation')}\n"
        "- UI remains minimal; API surface carries the closed-loop summary.\n",
        encoding="utf-8",
    )


def _write_phase22_docs(docs_dir: Path, route_decision: Mapping[str, Any]) -> None:
    evidence_index = """# Phase22 Evidence Index

- Phase13 reference ceiling: `../phase13-boundary-contract-runtime-and-trainable-probe/evidence-real-qwen36-27b-base/baseline_b_qwen36_boundary_base.json`
- Phase17 DPO product probe: `../phase17-qwen-dpo-product-probe/evidence-real-qwen-dpo/comparison_summary.json`
- Phase18 degeneration guardrails: `../phase18-dpo-degeneration-guardrails/evidence-real-qwen-dpo-guardrail/comparison_summary.json`
- Phase19 preference expansion: `../phase19-preference-signal-expansion/evidence/preference_quality_report.json`
- Phase20 model ladder: `../phase20-qwen-model-ladder/evidence-real-model-ladder/model_ladder_summary.json`
- Phase21 candidate workbench: `../phase21-training-candidate-workbench/evidence/candidate_plan_example.json`
"""
    route = (
        "# Phase22 Product Route Decision\n\n"
        "## Decision\n\n"
        f"- Recommendation: {route_decision.get('recommendation')}\n"
        "- Runtime boundary contract is the main product path.\n"
        "- Training remains a guarded candidate experiment with archive/manual-review decisions only.\n\n"
        "## Rationale\n\n"
        "- Phase13/12 boundary-first runtime showed stable product boundary behavior.\n"
        "- Phase17 proved DPO runtime viability but not product improvement.\n"
        "- Phase18 adds sanity and degeneration gates so bad adapters are intercepted.\n"
        "- Phase19 provides enough preference pairs to continue experiments, but not enough proof to move training into the main path.\n"
        "- Phase20 shows the only fully materialized trainable Qwen CausalLM is still a small diagnostic model.\n\n"
        "## Next Prompt Draft\n\n"
        f"{route_decision.get('next_prompt_draft')}\n"
    )
    docs_dir.joinpath("evidence_index.md").write_text(evidence_index, encoding="utf-8")
    docs_dir.joinpath("phase22-route-decision.md").write_text(route, encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Phase18-22 route convergence.")
    parser.add_argument("--clean-evidence", action="store_true")
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument("--run-real-phase18-dpo", action="store_true")
    parser.add_argument("--train-sample-limit", type=int, default=12)
    parser.add_argument("--eval-max-new-tokens", type=int, default=140)
    parser.add_argument("--eval-device", choices=("cpu", "mps"), default=None)
    parser.add_argument("--phase17-evidence-dir", type=Path, default=PHASE17_REAL_EVIDENCE)
    parser.add_argument("--training-output-dir", type=Path, default=Path("trainer_job_outputs/phase18-dpo-degeneration-guardrails/conservative"))
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    phase18_evidence = PHASE18_DIR / "evidence-real-qwen-dpo-guardrail"
    phase19_evidence = PHASE19_DIR / "evidence"
    phase20_evidence = PHASE20_DIR / "evidence-real-model-ladder"
    phase21_evidence = PHASE21_DIR / "evidence"
    if args.clean_evidence:
        for path in (PHASE18_DIR, PHASE19_DIR, PHASE20_DIR, PHASE21_DIR, PHASE22_DIR):
            if path.exists():
                shutil.rmtree(path)
    phase18_summary = build_phase18(
        docs_dir=PHASE18_DIR,
        evidence_dir=phase18_evidence,
        phase17_evidence_dir=args.phase17_evidence_dir,
        run_real_conservative=args.run_real_phase18_dpo,
        allow_model_download=args.allow_model_download,
        training_output_dir=args.training_output_dir,
        eval_device=args.eval_device,
        eval_max_new_tokens=args.eval_max_new_tokens,
        train_sample_limit=args.train_sample_limit,
    )
    phase17_holdout = _read_json(args.phase17_evidence_dir / "holdout.json")
    phase19_summary = build_phase19(
        docs_dir=PHASE19_DIR,
        evidence_dir=phase19_evidence,
        phase15_evidence_dir=PHASE15_REAL_EVIDENCE,
        phase17_holdout=phase17_holdout,
    )
    phase20_summary = build_phase20(
        docs_dir=PHASE20_DIR,
        evidence_dir=phase20_evidence,
        phase18_summary=phase18_summary,
        phase17_evidence_dir=args.phase17_evidence_dir,
    )
    phase21_summary = build_phase21(
        docs_dir=PHASE21_DIR,
        evidence_dir=phase21_evidence,
        phase18_summary=phase18_summary,
        phase19_summary=phase19_summary,
        phase20_summary=phase20_summary,
    )
    phase22_decision = build_phase22(
        docs_dir=PHASE22_DIR,
        phase18_summary=phase18_summary,
        phase19_summary=phase19_summary,
        phase20_summary=phase20_summary,
        phase21_summary=phase21_summary,
    )
    summary = {
        "kind": "phase18_to_phase22_route_convergence_summary",
        "phase18": phase18_summary,
        "phase19": phase19_summary,
        "phase20": phase20_summary,
        "phase21": phase21_summary,
        "phase22": phase22_decision,
        "created_at": _utcnow_iso(),
    }
    _write_json(PHASE22_DIR / "comparison_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
