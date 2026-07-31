#!/usr/bin/env python3
"""Generate Phase32 personal Agent preference training-loop evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping
import re

ROOT = Path(__file__).resolve().parents[1]
PFE_CORE = ROOT / "pfe-core"
if str(PFE_CORE) not in sys.path:
    sys.path.insert(0, str(PFE_CORE))

from pfe_core.phase32_personal_agent_preference import (
    PHASE32_MIN_HOLDOUT_PROMPTS,
    aggregate_phase32_eval_details,
    build_phase32_candidate_artifacts,
    build_phase32_holdout,
    build_phase32_phase31_review,
    build_phase32_review_decisions,
    build_phase32_taxonomy,
    phase32_final_decision,
    score_phase32_output,
    write_jsonl,
)


PHASE31_DIR = Path("docs/demo/phase31-obsidian-agent-conversation-signal-mining")
PHASE32_DIR = Path("docs/demo/phase32-personal-agent-preference-training-loop")
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


def _load_phase17_tool() -> Any:
    path = Path(__file__).resolve().parent / "phase17_qwen_dpo_product_probe.py"
    spec = importlib.util.spec_from_file_location("phase17_qwen_dpo_product_probe", path)
    if not spec or not spec.loader:
        raise RuntimeError(f"cannot load Phase17 helper from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _phase17_compatible_dpo_pairs(dpo_pairs: Iterable[Mapping[str, Any]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for index, pair in enumerate(dpo_pairs, start=1):
        rows.append(
            {
                "sample_id": str(pair.get("sample_id") or pair.get("pair_id") or f"phase32-dpo-{index:03d}"),
                "instruction": str(pair.get("instruction") or pair.get("prompt") or ""),
                "chosen": str(pair.get("chosen") or ""),
                "rejected": str(pair.get("rejected") or ""),
            }
        )
    return rows


def build_phase32_training_manifest(
    *,
    selected_dpo_pairs: list[Mapping[str, Any]],
    candidate_manifest: Mapping[str, Any],
    holdout_integrity: Mapping[str, Any],
    model_selection: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "kind": "phase32_training_manifest",
        "train_type": "dpo",
        "training_strategy": "personal_agent_preference_dpo_probe",
        "selected_sample_count": len(selected_dpo_pairs),
        "step_equivalent_count": len(selected_dpo_pairs),
        "source_candidate_manifest": dict(candidate_manifest),
        "holdout_integrity_passed": bool(holdout_integrity.get("passed")),
        "model_selection_status": model_selection.get("status"),
        "selected_model": model_selection.get("selected_model") or model_selection.get("selected"),
        "not_27b_training": True,
        "raw_private_text_committed": False,
        "created_at": _utcnow_iso(),
    }


def run_phase32_training_probe(
    *,
    evidence_training_dir: Path,
    dpo_pairs: list[Mapping[str, Any]],
    candidate_manifest: Mapping[str, Any],
    holdout_integrity: Mapping[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    phase17 = _load_phase17_tool()
    model_selection = phase17.select_qwen_model(
        requested_model=args.training_model_id,
        allow_model_download=args.allow_model_download,
    )
    _write_json(evidence_training_dir / "model_selection.json", model_selection)
    selected = dpo_pairs[: max(1, min(args.train_sample_limit, len(dpo_pairs)))]
    write_jsonl(evidence_training_dir / "selected_dpo_pairs.jsonl", selected)
    compatible = _phase17_compatible_dpo_pairs(selected)
    write_jsonl(evidence_training_dir / "selected_phase17_compatible_dpo_pairs.jsonl", compatible)
    training_manifest = build_phase32_training_manifest(
        selected_dpo_pairs=selected,
        candidate_manifest=candidate_manifest,
        holdout_integrity=holdout_integrity,
        model_selection=model_selection,
    )
    _write_json(evidence_training_dir / "training_manifest.json", training_manifest)
    if model_selection.get("status") != "selected":
        payload = {
            "kind": "phase32_training_attempt",
            "real_training": "blocked",
            "training_run": False,
            "blocked_reason": "qwen_model_not_selected",
            "model_selection": model_selection,
            "training_manifest": training_manifest,
            "created_at": _utcnow_iso(),
        }
        _write_json(evidence_training_dir / "training_attempt.json", payload)
        _write_json(evidence_training_dir / "train_log.json", payload)
        _write_json(
            evidence_training_dir / "adapter_validation.json",
            {"kind": "phase32_adapter_validation", "valid": False, "reason": "training_not_completed"},
        )
        return payload
    output_dir = args.training_output_dir.expanduser().resolve()
    if args.clean_training_output and output_dir.exists():
        shutil.rmtree(output_dir)
    job_spec = phase17.build_qwen_dpo_job_spec(
        samples=compatible,
        base_model=str(model_selection.get("selected_model") or model_selection.get("selected")),
        output_dir=output_dir,
        epochs=args.dpo_epochs,
        beta=args.dpo_beta,
        max_length=args.dpo_max_length,
        max_prompt_length=args.dpo_max_prompt_length,
    )
    recipe = dict(job_spec.get("recipe") or {})
    training_recipe = dict(recipe.get("training") or {})
    training_recipe["use_cpu"] = True
    recipe["training"] = training_recipe
    job_spec = {**job_spec, "recipe": recipe, "use_cpu": True}
    preflight = phase17.dpo_preflight()
    _write_json(evidence_training_dir / "dpo_preflight.json", preflight)
    _write_json(evidence_training_dir / "dpo_job_spec.json", job_spec)
    if not args.run_real_training:
        payload = {
            "kind": "phase32_training_attempt",
            "real_training": "not_started",
            "training_run": False,
            "skip_reason": "run with --run-real-training",
            "preflight": preflight,
            "model_selection": model_selection,
            "selected_model": model_selection.get("selected_model") or model_selection.get("selected"),
            "training_manifest": training_manifest,
            "created_at": _utcnow_iso(),
        }
        _write_json(evidence_training_dir / "training_attempt.json", payload)
        _write_json(evidence_training_dir / "train_log.json", payload)
        _write_json(
            evidence_training_dir / "adapter_validation.json",
            {"kind": "phase32_adapter_validation", "valid": False, "reason": "training_not_started"},
        )
        return payload
    training = phase17.run_qwen_dpo_training(
        evidence_dir=evidence_training_dir,
        job_spec=job_spec,
        preflight=preflight,
        model_selection=model_selection,
        run_real_qwen_dpo=True,
    )
    payload = {
        "kind": "phase32_training_attempt",
        **dict(training),
        "training_manifest": training_manifest,
        "phase32_training_role": "personal_agent_preference_probe",
        "not_27b_training": True,
    }
    _write_json(evidence_training_dir / "training_attempt.json", payload)
    _write_json(evidence_training_dir / "train_log.json", payload)
    validation = {
        "kind": "phase32_adapter_validation",
        **_dict(training.get("adapter_validation")),
        "selected_model": model_selection.get("selected_model") or model_selection.get("selected"),
    }
    _write_json(evidence_training_dir / "adapter_validation.json", validation)
    return payload


def _render_phase32_prompt(tokenizer: Any, user_prompt: str) -> tuple[str, dict[str, Any]]:
    system_prompt = (
        "你是 PFE 个人 Agent 协作助手。回答必须体现用户长期偏好：先执行、证据优先、状态简洁、"
        "被纠正后快速转向、理解本机路径/分支/进程上下文、保护隐私边界。不要宣称未实际完成的提交、PR 或关停。"
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
            return f"{system_prompt}\n\n用户：{user_prompt}\n助手：", {"chat_template_applied": False, "chat_template_error": str(exc)}
    return f"{system_prompt}\n\n用户：{user_prompt}\n助手：", {"chat_template_applied": False, "chat_template_error": ""}


def _generate_phase32_transformers_outputs(
    *,
    evidence_eval_dir: Path,
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
        return {"kind": "phase32_eval_report", "label": label, "status": "dependency_failed", "error": str(exc), "created_at": _utcnow_iso()}
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
            "kind": "phase32_eval_report",
            "label": label,
            "model_id": model_id,
            "adapter_path": adapter_path,
            "status": "load_failed",
            "error": str(exc),
            "created_at": _utcnow_iso(),
        }
    try:
        for item in holdouts:
            prompt, rendered = _render_phase32_prompt(tokenizer, str(item.get("prompt") or ""))
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1536)
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
            details.append(
                {
                    "prompt_id": item.get("prompt_id"),
                    "category": item.get("category"),
                    "expected_taxonomy": item.get("expected_taxonomy"),
                    "user_prompt": item.get("prompt"),
                    "chat_template_applied": rendered.get("chat_template_applied"),
                    "raw_output": raw_output,
                    "output": raw_output,
                    "scores": score_phase32_output(raw_output, item),
                }
            )
    except Exception as exc:
        return {
            "kind": "phase32_eval_report",
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
    report = {
        "kind": "phase32_eval_report",
        "label": label,
        "status": "completed",
        "model_id": model_id,
        "adapter_path": adapter_path,
        "holdout_count": len(details),
        "scores": aggregate_phase32_eval_details(details),
        "details": details,
        "duration_seconds": round(time.monotonic() - started, 3),
        "created_at": _utcnow_iso(),
    }
    _write_json(evidence_eval_dir / f"{label}.json", report)
    return report


def run_phase32_eval(
    *,
    evidence_eval_dir: Path,
    holdout: Mapping[str, Any],
    training_attempt: Mapping[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    model_id = str(training_attempt.get("selected_model") or _dict(training_attempt.get("result")).get("base_model") or "")
    adapter_path = str(_dict(training_attempt.get("adapter_validation")).get("artifact_dir") or training_attempt.get("adapter_path") or "")
    if not args.run_real_eval:
        base = {"kind": "phase32_eval_report", "label": "eval_report_base", "status": "not_started", "skip_reason": "run with --run-real-eval"}
        adapter = {"kind": "phase32_eval_report", "label": "eval_report_adapter", "status": "not_started", "skip_reason": "run with --run-real-eval"}
    elif training_attempt.get("real_training") != "completed" or not model_id or not adapter_path:
        base = {"kind": "phase32_eval_report", "label": "eval_report_base", "status": "blocked", "blocked_reason": "training_not_completed"}
        adapter = {"kind": "phase32_eval_report", "label": "eval_report_adapter", "status": "blocked", "blocked_reason": "training_not_completed"}
    else:
        prompts = [dict(item) for item in holdout.get("prompts") or []][: args.eval_holdout_limit]
        base = _generate_phase32_transformers_outputs(
            evidence_eval_dir=evidence_eval_dir,
            model_id=model_id,
            label="eval_report_base",
            holdouts=prompts,
            adapter_path=None,
            max_new_tokens=args.eval_max_new_tokens,
            local_files_only=not args.allow_model_download,
            device=args.eval_device,
        )
        adapter = _generate_phase32_transformers_outputs(
            evidence_eval_dir=evidence_eval_dir,
            model_id=model_id,
            label="eval_report_adapter",
            holdouts=prompts,
            adapter_path=adapter_path,
            max_new_tokens=args.eval_max_new_tokens,
            local_files_only=not args.allow_model_download,
            device=args.eval_device,
        )
    _write_json(evidence_eval_dir / "eval_report_base.json", base)
    _write_json(evidence_eval_dir / "eval_report_adapter.json", adapter)
    return {"base": base, "adapter": adapter}


def capture_ollama_reference(*, evidence_eval_dir: Path, run_outputs: bool) -> dict[str, Any]:
    report: dict[str, Any] = {
        "kind": "phase32_ollama_runtime_reference",
        "run_outputs": run_outputs,
        "models": [],
        "created_at": _utcnow_iso(),
    }
    try:
        proc = subprocess.run(["ollama", "list"], text=True, capture_output=True, timeout=15, check=False)
        report["ollama_list_returncode"] = proc.returncode
        report["ollama_list_stdout"] = proc.stdout
        report["qwen36_available"] = "qwen3.6" in proc.stdout
        report["gemma4_31b_available"] = "gemma4:31b" in proc.stdout
    except Exception as exc:
        report["status"] = "unavailable"
        report["error"] = str(exc)
    _write_json(evidence_eval_dir / "ollama_reference.json", report)
    return report


def _write_output_examples(path: Path, base_eval: Mapping[str, Any], adapter_eval: Mapping[str, Any]) -> None:
    lines = ["# Phase32 Output Examples", ""]
    for label, report in (("Base", base_eval), ("Adapter", adapter_eval)):
        lines.extend([f"## {label}", "", f"- Status: {report.get('status')}", f"- Scores: `{json.dumps(report.get('scores') or {}, ensure_ascii=False, sort_keys=True)}`", ""])
        for detail in list(report.get("details") or [])[:5]:
            lines.extend(
                [
                    f"### {detail.get('prompt_id')} / {detail.get('category')}",
                    "",
                    "```text",
                    str(detail.get("raw_output") or detail.get("output") or "")[:1200],
                    "```",
                    "",
                ]
            )
    while lines and lines[-1] == "":
        lines.pop()
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_runbook(path: Path) -> None:
    path.write_text(
        """# Phase32 Runbook

Phase32 reviews Phase31 historical Agent collaboration signals, builds abstract personal preference candidates, runs a small Qwen DPO probe when available, and evaluates base vs adapter on personal Agent holdout prompts.

Historical conversations are not realtime actual feedback. Do not commit raw Obsidian/AgentMemory text.

## Default Evidence Smoke

```bash
.venv/bin/python tools/phase32_personal_agent_preference_training_loop.py --clean-evidence
```

## Real Training And Eval Probe

```bash
.venv/bin/python tools/phase32_personal_agent_preference_training_loop.py \\
  --clean-evidence \\
  --run-real-training \\
  --run-real-eval \\
  --eval-device cpu
```
""",
        encoding="utf-8",
    )


def _write_final_decision(path: Path, summary: Mapping[str, Any]) -> None:
    decision = _dict(summary.get("decision"))
    review = _dict(summary.get("review_summary"))
    manifest = _dict(summary.get("candidate_manifest"))
    training = _dict(summary.get("training_attempt"))
    path.write_text(
        f"""# Phase32 Final Decision

## Decision

- Recommendation: {decision.get("recommendation")}
- Status: {decision.get("status")}
- Promotion allowed: {decision.get("promotion_allowed")}
- Auto promotion allowed: false
- Product benefit claim allowed: {decision.get("product_benefit_claim_allowed")}
- Actual user feedback collected: false
- Historical Agent conversations used: true

## Review

- Decisions: {review.get("decision_count")}
- Approved for training: {review.get("approved_for_training_count")}
- Excluded: {review.get("excluded_count")}
- Quarantined: {review.get("quarantined_count")}
- Taxonomy counts: `{json.dumps(review.get("taxonomy_counts") or {}, ensure_ascii=False, sort_keys=True)}`

## Candidates

- SFT samples: {manifest.get("sft_sample_count")}
- DPO pairs: {manifest.get("dpo_pair_count")}
- Hard negatives: {manifest.get("hard_negative_pair_count")}
- Profile candidates: {manifest.get("profile_candidate_count")}
- Memory candidates: {manifest.get("memory_candidate_count")}
- Raw private text committed: {manifest.get("raw_private_text_committed")}

## Training

- Real training: {training.get("real_training")}
- Selected model: {training.get("selected_model")}
- Adapter path: {training.get("adapter_path")}

## Scores

- Base: `{json.dumps(decision.get("base_scores") or {}, ensure_ascii=False, sort_keys=True)}`
- Adapter: `{json.dumps(decision.get("adapter_scores") or {}, ensure_ascii=False, sort_keys=True)}`

## Reasons

{chr(10).join(f"- {reason}" for reason in decision.get("reasons") or ["no blocking reasons"])}
""",
        encoding="utf-8",
    )


def generate_phase32_evidence(args: argparse.Namespace) -> dict[str, Any]:
    if args.clean_evidence:
        _clean_dir(PHASE32_DIR)
    for subdir in ("evidence", "evidence-review", "evidence-candidates", "evidence-training", "evidence-eval"):
        (PHASE32_DIR / subdir).mkdir(parents=True, exist_ok=True)
    evidence_dir = PHASE32_DIR / "evidence"
    review_dir = PHASE32_DIR / "evidence-review"
    candidate_dir = PHASE32_DIR / "evidence-candidates"
    training_dir = PHASE32_DIR / "evidence-training"
    eval_dir = PHASE32_DIR / "evidence-eval"

    phase31_summary = _read_json(PHASE31_DIR / "comparison_summary.json")
    phase31_decision_text = (PHASE31_DIR / "phase31-final-decision.md").read_text(encoding="utf-8") if (PHASE31_DIR / "phase31-final-decision.md").exists() else ""
    phase31_review = build_phase32_phase31_review(phase31_summary=phase31_summary, phase31_decision=phase31_decision_text)
    signals = _read_jsonl(PHASE31_DIR / "evidence-signals" / "historical_signal_batch.jsonl")
    review_batch = build_phase32_review_decisions(signals)
    holdout = build_phase32_holdout(count=max(PHASE32_MIN_HOLDOUT_PROMPTS, args.holdout_count))
    candidates = build_phase32_candidate_artifacts(
        signals=signals,
        review_decisions=list(review_batch["review_decisions"]),
        holdout=holdout,
    )
    training_attempt = run_phase32_training_probe(
        evidence_training_dir=training_dir,
        dpo_pairs=list(candidates["dpo_pairs"]),
        candidate_manifest=candidates["candidate_manifest"],
        holdout_integrity=candidates["holdout_integrity_check"],
        args=args,
    )
    eval_reports = run_phase32_eval(
        evidence_eval_dir=eval_dir,
        holdout=holdout,
        training_attempt=training_attempt,
        args=args,
    )
    ollama_reference = capture_ollama_reference(evidence_eval_dir=eval_dir, run_outputs=args.run_ollama_reference)
    decision = phase32_final_decision(
        candidate_quality_report=candidates["candidate_quality_report"],
        training_attempt=training_attempt,
        base_eval=eval_reports["base"],
        adapter_eval=eval_reports["adapter"],
    )

    _write_json(evidence_dir / "phase31_review.json", phase31_review)
    _write_json(review_dir / "taxonomy.json", build_phase32_taxonomy())
    _write_json(review_dir / "review_decisions.json", {"kind": "phase32_review_decisions", "items": review_batch["review_decisions"]})
    _write_json(review_dir / "review_summary.json", review_batch["review_summary"])
    _write_json(evidence_dir / "holdout.json", holdout)
    _write_json(candidate_dir / "candidate_manifest.json", candidates["candidate_manifest"])
    _write_json(candidate_dir / "candidate_quality_report.json", candidates["candidate_quality_report"])
    _write_json(candidate_dir / "holdout_integrity_check.json", candidates["holdout_integrity_check"])
    write_jsonl(candidate_dir / "selected_sft_samples.jsonl", candidates["sft_samples"])
    write_jsonl(candidate_dir / "selected_dpo_pairs.jsonl", candidates["dpo_pairs"])
    write_jsonl(candidate_dir / "hard_negative_pairs.jsonl", candidates["hard_negative_pairs"])
    write_jsonl(candidate_dir / "profile_candidates.jsonl", candidates["profile_candidates"])
    write_jsonl(candidate_dir / "memory_candidates.jsonl", candidates["memory_candidates"])
    _write_json(eval_dir / "decision.json", decision)
    _write_output_examples(eval_dir / "output_examples.md", eval_reports["base"], eval_reports["adapter"])

    summary = {
        "kind": "phase32_personal_agent_preference_training_loop_summary",
        "status": "completed",
        "phase31_review": phase31_review,
        "review_summary": review_batch["review_summary"],
        "candidate_manifest": candidates["candidate_manifest"],
        "candidate_quality_report": candidates["candidate_quality_report"],
        "training_attempt": training_attempt,
        "base_eval": eval_reports["base"],
        "adapter_eval": eval_reports["adapter"],
        "ollama_reference": ollama_reference,
        "decision": decision,
        "final_recommendation": decision["recommendation"],
        "created_at": _utcnow_iso(),
    }
    _write_json(PHASE32_DIR / "comparison_summary.json", summary)
    _write_json(evidence_dir / "comparison_summary.json", summary)
    _write_runbook(PHASE32_DIR / "phase32-runbook.md")
    _write_final_decision(PHASE32_DIR / "phase32-final-decision.md", summary)
    (PHASE32_DIR / "next-pursuit-goal.md").write_text(
        "目标：将 Phase32 通过人工复核的个人协作偏好 adapter 接入 Hermes/PFE runtime，采集真实在线反馈并形成持续闭环。\n",
        encoding="utf-8",
    )
    _redact_evidence_tree(PHASE32_DIR)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-evidence", action="store_true")
    parser.add_argument("--holdout-count", type=int, default=40)
    parser.add_argument("--train-sample-limit", type=int, default=12)
    parser.add_argument("--training-model-id", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument("--run-real-training", action="store_true")
    parser.add_argument("--clean-training-output", action="store_true")
    parser.add_argument("--training-output-dir", type=Path, default=Path("trainer_job_outputs/phase32-personal-agent-preference-qwen25-0_5b"))
    parser.add_argument("--dpo-epochs", type=int, default=1)
    parser.add_argument("--dpo-beta", type=float, default=0.1)
    parser.add_argument("--dpo-max-length", type=int, default=1024)
    parser.add_argument("--dpo-max-prompt-length", type=int, default=768)
    parser.add_argument("--run-real-eval", action="store_true")
    parser.add_argument("--eval-holdout-limit", type=int, default=40)
    parser.add_argument("--eval-max-new-tokens", type=int, default=128)
    parser.add_argument("--eval-device", default="cpu")
    parser.add_argument("--run-ollama-reference", action="store_true")
    args = parser.parse_args()

    summary = generate_phase32_evidence(args)
    print(
        json.dumps(
            {
                "kind": summary["kind"],
                "status": summary["status"],
                "review": summary["review_summary"],
                "candidate_manifest": summary["candidate_manifest"],
                "training": {
                    "real_training": _dict(summary.get("training_attempt")).get("real_training"),
                    "selected_model": _dict(summary.get("training_attempt")).get("selected_model"),
                },
                "decision": summary["decision"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
