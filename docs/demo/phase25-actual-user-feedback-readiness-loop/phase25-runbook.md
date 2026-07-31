# Phase25 Runbook

## Goal

Collect attested actual user feedback before any product-value SFT/DPO training probe.

## Current Evidence

- Actual feedback count: 0
- Approved actual candidates: 0
- Readiness status: collect_actual_feedback
- Blockers: insufficient_approved_actual_user_feedback, insufficient_sft_candidate_samples, insufficient_dpo_candidate_pairs
- Runtime holdout scores: {"citation_hit_rate": 1.0, "explicit_boundary_rate": 1.0, "external_law_reference_rate": 0.0, "extra_text_after_first_block_rate": 0.0, "safety_boundary_rate": 1.0, "structure_hit_rate": 1.0, "think_leak_rate": 0.0, "unsupported_assertions": 0}

## Generate Evidence

```bash
.venv/bin/python tools/phase25_actual_user_feedback_readiness_loop.py --clean-evidence
```

## Collect One Actual Feedback Signal

Use `/pfe/phase25/actual-feedback` with `feedback_source=actual_user_feedback` and an attestation that confirms the feedback is not scripted or curated. The signal remains pending review until explicitly approved.
