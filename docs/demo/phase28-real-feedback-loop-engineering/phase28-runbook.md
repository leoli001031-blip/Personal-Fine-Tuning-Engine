# Phase28 Runbook

## Goal

Run the real-feedback loop-engineering path: collect attested feedback, import,
validate, review, build candidates, and open training readiness only when the
approved actual-feedback threshold is met.

## Current Evidence

- Task count: 36
- Actual feedback count: 0
- Approved actual candidates: 0
- Loop state: observe
- Readiness: collect_actual_feedback
- Blockers: insufficient_approved_actual_user_feedback, insufficient_sft_candidate_samples, insufficient_dpo_candidate_pairs

## Default Command

```bash
.venv/bin/python tools/phase28_real_feedback_loop_engineering.py --clean-evidence
```

## Optional Real Feedback Import

```bash
.venv/bin/python tools/phase28_real_feedback_loop_engineering.py --feedback-jsonl path/to/actual_feedback.jsonl --review-decisions-json path/to/review_decisions.json
```

Templates and Phase27 simulation rows are not valid actual feedback.
