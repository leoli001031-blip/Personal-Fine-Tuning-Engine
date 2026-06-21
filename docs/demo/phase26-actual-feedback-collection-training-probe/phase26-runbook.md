# Phase26 Runbook

## Goal

Execute the next product step after Phase25: prepare a real feedback collection pack, accept attested actual-feedback batches, manually approve candidates, then run a Qwen3-4B SFT/DPO probe only when the gate is ready.

## Current State

- Collection tasks prepared: 12
- Actual feedback count: 0
- Approved actual candidates: 0
- Readiness: collect_actual_feedback
- Blockers: insufficient_approved_actual_user_feedback, insufficient_sft_candidate_samples, insufficient_dpo_candidate_pairs

## Commands

```bash
.venv/bin/python tools/phase26_actual_feedback_collection_training_probe.py --clean-evidence
```

## API

- `GET /pfe/phase26/feedback-collection-pack`
- `POST /pfe/phase26/actual-feedback-batch`
- `GET /pfe/phase26/training-probe-readiness`
