# Phase27 Runbook

## Goal

Run the actual-feedback review workflow after Phase26: collect real feedback, import it, review it, then open Qwen3-4B training only when 12 approved actual candidates exist.

## Current State

- Collection tasks prepared: 12
- Actual feedback count: 0
- Accepted pending review: 0
- Approved actual candidates: 0
- Readiness: collect_actual_feedback
- Blockers: insufficient_approved_actual_user_feedback, insufficient_sft_candidate_samples, insufficient_dpo_candidate_pairs

## Commands

```bash
.venv/bin/python tools/phase27_actual_feedback_review_training_loop.py --clean-evidence
```

## API

- `GET /pfe/phase27/collection-pack`
- `POST /pfe/phase27/actual-feedback-batch`
- `GET /pfe/phase27/review-queue`
- `POST /pfe/phase27/review-decisions`
- `GET /pfe/phase27/training-readiness`
