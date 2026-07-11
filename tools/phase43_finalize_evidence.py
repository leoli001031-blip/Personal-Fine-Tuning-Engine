#!/usr/bin/env python3
"""Finalize Phase43 comparison, decision, examples, and documentation."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = REPO_ROOT / "pfe-core"
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

from pfe_core.phase43_personal_preference_benefit import build_phase43_decision


ROOT = REPO_ROOT / "docs" / "demo" / "phase43-qwen3-4b-personal-preference-benefit-proof"


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _copy_selected_sft_evidence() -> dict[str, Any]:
    source = ROOT / "evidence-training-sft" / "probe-12step"
    target = ROOT / "evidence-training-sft"
    mapping = {
        "training_manifest.json": "training_manifest_selected.json",
        "training_attempt.json": "training_attempt.json",
        "train_log.json": "train_log.json",
        "loss_history.json": "loss_history.json",
        "parameter_fingerprint_before_after.json": "parameter_fingerprint_before_after.json",
        "adapter_validation.json": "adapter_validation.json",
        "completion_boundary_report.json": "completion_boundary_report.json",
    }
    for source_name, target_name in mapping.items():
        shutil.copy2(source / source_name, target / target_name)
    sanity12 = _read_json(ROOT / "evidence-training-sft" / "probe-12step" / "sanity_report_sft.json")
    sanity30 = _read_json(ROOT / "evidence-training-sft" / "probe-30step" / "sanity_report_sft.json")
    return {
        "kind": "phase43_sft_probe_selection",
        "selected_probe": "12step",
        "selected_adapter_path": _read_json(source / "adapter_validation.json").get("artifact_dir"),
        "reason": (
            "12-step had a slightly higher sanity preference score and lower repetition than 30-step; "
            "lower 30-step loss was not treated as product improvement"
        ),
        "probe_12step": {
            "preference_score": sanity12.get("user_preference_score"),
            "response_diversity": sanity12.get("response_diversity"),
            "repetition_rate": sanity12.get("repetition_rate"),
        },
        "probe_30step": {
            "preference_score": sanity30.get("user_preference_score"),
            "response_diversity": sanity30.get("response_diversity"),
            "repetition_rate": sanity30.get("repetition_rate"),
        },
        "actual_product_benefit_claim_allowed": False,
    }


def _delta(candidate: Mapping[str, Any], base: Mapping[str, Any], key: str) -> float:
    return round(float(candidate.get(key) or 0.0) - float(base.get(key) or 0.0), 4)


def _final_assistant(transcript: Mapping[str, Any]) -> str:
    answers = [
        str(row.get("content") or "")
        for row in transcript.get("turns") or []
        if isinstance(row, Mapping) and row.get("role") == "assistant"
    ]
    return answers[-1] if answers else ""


def _output_examples(sessions: list[Mapping[str, Any]], transcripts: Mapping[str, list[Mapping[str, Any]]]) -> str:
    by_variant = {
        variant: {str(row.get("session_id")): row for row in rows}
        for variant, rows in transcripts.items()
    }
    wanted_categories = ("drift_correction", "failure_handling", "privacy", "git_pr")
    selected = []
    for category in wanted_categories:
        item = next((row for row in sessions if row.get("category") == category), None)
        if item:
            selected.append(item)
    lines = [
        "# Phase43 real output examples",
        "",
        "All answers below are direct final-turn outputs from real local Qwen3-4B calls. They are not rewritten.",
        "",
    ]
    for session in selected:
        session_id = str(session.get("session_id"))
        lines.extend(
            [
                f"## {session_id} - {session.get('category')}",
                "",
                f"**Goal:** {session.get('user_goal')}",
                "",
                f"**Correction:** {session.get('user_correction')}",
                "",
            ]
        )
        for variant in ("base", "runtime", "sft"):
            text = _final_assistant(by_variant[variant][session_id])
            lines.extend([f"### {variant}", "", text, ""])
    rendered = "\n".join(lines)
    return "\n".join(line.rstrip() for line in rendered.splitlines()).rstrip() + "\n"


def main() -> int:
    selection = _copy_selected_sft_evidence()
    _write_json(ROOT / "evidence-training-sft" / "probe_selection.json", selection)

    metrics = {
        variant: _read_json(ROOT / "evidence-holdout" / f"metrics_{variant}.json")
        for variant in ("base", "runtime", "sft")
    }
    deterministic = _read_json(ROOT / "evidence-blind-eval" / "deterministic_summary.json")
    independent = _read_json(ROOT / "evidence-blind-eval" / "independent_judge_summary.json")
    sft_attempt = _read_json(ROOT / "evidence-training-sft" / "training_attempt.json")
    dpo_attempt = _read_json(ROOT / "evidence-training-dpo" / "training_attempt.json")
    decision = build_phase43_decision(
        base_metrics=metrics["base"],
        candidate_metrics={"sft": metrics["sft"]},
        deterministic_blind=deterministic,
        independent_blind=independent,
        training_status={"sft": str(sft_attempt.get("status") or "failed")},
    )
    decision.update(
        {
            "selected_sft_probe": "12step",
            "dpo_status": dpo_attempt.get("status"),
            "dpo_decision": "archive",
            "dpo_reason": dpo_attempt.get("error") or "non_finite_training_metrics",
            "runtime_contract_status": "effective_but_not_trainable_candidate",
            "runtime_contract_tradeoff": "higher blind wins and preference score, but lower diversity, higher repetition, and privacy canary leakage",
            "final_recommendation": "archive",
            "hermes_real_acceptance_recommendation": (
                "do_not_attach_the_archived_adapter; harden privacy and candidate data first. "
                "A guarded runtime-contract-only Hermes trial can be considered after that fix."
            ),
            "created_at": _utcnow(),
        }
    )

    comparison = {
        "kind": "phase43_qwen3_4b_personal_preference_comparison",
        "status": "completed",
        "model": "Qwen3-4B",
        "real_model_calls": True,
        "holdout_session_count": 40,
        "model_call_count": 360,
        "variants": metrics,
        "deltas_vs_base": {
            variant: {
                key: _delta(metrics[variant], metrics["base"], key)
                for key in (
                    "user_preference_score",
                    "follows_latest_user_intent_rate",
                    "evidence_before_claim_rate",
                    "correction_responsiveness_rate",
                    "useful_next_action_rate",
                    "generic_answer_rate",
                    "false_completion_rate",
                    "privacy_violation_rate",
                    "training_leakage_rate",
                    "response_diversity",
                    "repetition_rate",
                    "latency_seconds",
                )
            }
            for variant in ("runtime", "sft")
        },
        "user_preference_win_rate": {
            variant: dict(dict(independent.get("variants") or {}).get(variant) or {}).get("candidate_win_rate")
            for variant in ("runtime", "sft")
        },
        "deterministic_blind": deterministic,
        "independent_blind": independent,
        "sft_probe_selection": selection,
        "sft_training": {
            "status": sft_attempt.get("status"),
            "requested_steps": sft_attempt.get("requested_steps"),
            "duration_seconds": sft_attempt.get("duration_seconds"),
            "initial_loss": dict(sft_attempt.get("execution") or {}).get("initial_loss"),
            "final_loss": dict(sft_attempt.get("execution") or {}).get("final_loss"),
            "parameters_updated": dict(sft_attempt.get("execution") or {}).get("parameters_updated"),
            "adapter_sha256": dict(sft_attempt.get("adapter_validation") or {}).get("sha256"),
        },
        "dpo_training": {
            "status": dpo_attempt.get("status"),
            "requested_steps": dpo_attempt.get("requested_steps"),
            "error": dpo_attempt.get("error"),
        },
        "decision": decision,
        "simulated_usage": True,
        "actual_user_feedback": False,
        "actual_user_benefit_claim_allowed": False,
        "actual_product_benefit_claim_allowed": False,
        "auto_promotion_allowed": False,
        "created_at": _utcnow(),
    }
    _write_json(ROOT / "comparison_summary.json", comparison)
    _write_json(ROOT / "phase43-final-decision.json", decision)
    _write_json(ROOT / "evidence-blind-eval" / "blind_eval_report.json", {
        "kind": "phase43_dual_blind_eval_report",
        "deterministic": deterministic,
        "independent": independent,
        "decision": decision,
    })

    holdout = _read_json(ROOT / "evidence-holdout" / "holdout.json")
    transcript_rows = {
        variant: _read_jsonl(ROOT / "evidence-holdout" / f"transcripts_{variant}.jsonl")
        for variant in ("base", "runtime", "sft")
    }
    (ROOT / "evidence-holdout" / "output_examples.md").write_text(
        _output_examples(list(holdout.get("sessions") or []), transcript_rows),
        encoding="utf-8",
    )

    runbook = """# PFE Phase43 runbook

## Scope

Phase43 tests simulated laboratory preference benefit on local unquantized Qwen3-4B. It does not claim actual user benefit and never auto-promotes an adapter.

## Reproduce

```bash
cd /Users/lichenhao/Desktop/PFE
.venv/bin/python tools/phase43_qwen3_4b_prepare.py --clean-evidence
.venv/bin/python tools/phase43_qwen3_4b_sft_probe.py --steps 1 --clean
.venv/bin/python tools/phase43_qwen3_4b_sft_probe.py --steps 12 --clean
.venv/bin/python tools/phase43_qwen3_4b_generate.py --variant base --mode sanity --max-new-tokens 80
.venv/bin/python tools/phase43_qwen3_4b_generate.py --variant runtime --mode sanity --max-new-tokens 80
.venv/bin/python tools/phase43_qwen3_4b_generate.py --variant sft --mode sanity --steps 12 --max-new-tokens 80
.venv/bin/python tools/phase43_qwen3_4b_sft_probe.py --steps 30 --clean
.venv/bin/python tools/phase43_qwen3_4b_generate.py --variant sft --mode sanity --steps 30 --max-new-tokens 80
.venv/bin/python tools/phase43_qwen3_4b_dpo_probe.py --steps 12 --clean
.venv/bin/python tools/phase43_qwen3_4b_generate.py --variant base --mode holdout --max-new-tokens 96
.venv/bin/python tools/phase43_qwen3_4b_generate.py --variant runtime --mode holdout --max-new-tokens 96
.venv/bin/python tools/phase43_qwen3_4b_generate.py --variant sft --mode holdout --steps 12 --max-new-tokens 96
.venv/bin/python tools/phase43_blind_eval.py
.venv/bin/python tools/phase43_finalize_evidence.py
```

## Validation

```bash
.venv/bin/python -m py_compile pfe-core/pfe_core/phase43_personal_preference_benefit.py tools/phase43_*.py
.venv/bin/pytest -q tests/test_phase43_personal_preference_benefit.py
.venv/bin/pytest -q tests/test_phase42_reliability_hardening.py
make test-unit test-surface test-e2e-mock smoke-beta
git diff --check
```

## Decision boundary

Only a candidate passing both deterministic and independent blind gates may become `ready_for_manual_acceptance_trial`. Actual benefit still requires later real Hermes feedback and human review.
"""
    (ROOT / "phase43-runbook.md").write_text(runbook, encoding="utf-8")

    sft_checks = dict(dict(decision.get("candidate_decisions") or {}).get("sft") or {})
    final_md = f"""# PFE Phase43 final decision

## Decision

**archive**. Qwen3-4B SFT trained successfully, but the adapter did not beat base under the frozen dual-blind gate. No adapter is promoted.

## Real training

- SFT 1/12/30-step probes all performed real MPS optimizer/backward updates and produced valid PEFT safetensors.
- 12-step was selected over 30-step because its sanity score was slightly higher and repetition was lower, despite the 30-step loss being lower.
- DPO executed 12 real steps but produced non-finite metrics. The new gate correctly marked it failed and excluded it from eval.

## Product holdout

- 40 independent multi-turn sessions per arm, 360 real Qwen3-4B generation calls total.
- Base preference score: {metrics['base'].get('user_preference_score')}.
- Runtime contract preference score: {metrics['runtime'].get('user_preference_score')}.
- SFT preference score: {metrics['sft'].get('user_preference_score')}.
- SFT correction responsiveness: {metrics['sft'].get('correction_responsiveness_rate')} vs base {metrics['base'].get('correction_responsiveness_rate')}.
- SFT training leakage: {metrics['sft'].get('training_leakage_rate')}; diversity: {metrics['sft'].get('response_diversity')}.
- SFT privacy violation rate: {metrics['sft'].get('privacy_violation_rate')}.

## Blind evaluation

- Deterministic SFT win rate: {dict(deterministic.get('variants') or {}).get('sft', {}).get('candidate_win_rate')}.
- Independent Gemma4 SFT win rate: {dict(independent.get('variants') or {}).get('sft', {}).get('candidate_win_rate')}.
- Deterministic runtime win rate: {dict(deterministic.get('variants') or {}).get('runtime', {}).get('candidate_win_rate')}.
- Independent Gemma4 runtime win rate: {dict(independent.get('variants') or {}).get('runtime', {}).get('candidate_win_rate')}.

## Gate failures

`{json.dumps(sft_checks.get('failed_checks') or [], ensure_ascii=False)}`

## Interpretation

The training runtime is trustworthy and Qwen3-4B is locally trainable, but this 24-pair simulated dataset does not yet produce a better personal assistant. The runtime contract is much more effective than SFT on evidence and correction behavior, but it over-applies the contract, reduces diversity, increases repetition, and still repeats the privacy canary.

Phase43 proves simulated laboratory results only. It does not prove actual user benefit. Do not attach the archived adapter to the fourth Hermes Agent. First improve scenario-specific preference data and privacy behavior; only then run a guarded Hermes manual acceptance trial.
"""
    (ROOT / "phase43-final-decision.md").write_text(final_md, encoding="utf-8")

    integrity = {
        "kind": "phase43_evidence_integrity",
        "passed": (
            all(len(rows) == 40 for rows in transcript_rows.values())
            and all(all(row.get("actual_model_call") is True for row in rows) for rows in transcript_rows.values())
            and int(independent.get("completed_pair_count") or 0) == 80
            and independent.get("failure_count") == 0
            and sft_attempt.get("status") == "completed"
            and dpo_attempt.get("status") == "failed"
            and decision.get("status") == "archive"
        ),
        "transcript_counts": {key: len(value) for key, value in transcript_rows.items()},
        "all_transcripts_real_model_calls": {
            key: all(row.get("actual_model_call") is True for row in value)
            for key, value in transcript_rows.items()
        },
        "independent_judge_count": independent.get("completed_pair_count"),
        "independent_judge_failure_count": independent.get("failure_count"),
        "sft_status": sft_attempt.get("status"),
        "dpo_status": dpo_attempt.get("status"),
        "final_status": decision.get("status"),
        "actual_user_benefit_claim_allowed": False,
        "auto_promotion_allowed": False,
    }
    _write_json(ROOT / "evidence_integrity.json", integrity)
    print(json.dumps({"integrity_passed": integrity["passed"], "decision": decision["status"], "failed_checks": sft_checks.get("failed_checks")}, ensure_ascii=False, indent=2))
    return 0 if integrity["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
