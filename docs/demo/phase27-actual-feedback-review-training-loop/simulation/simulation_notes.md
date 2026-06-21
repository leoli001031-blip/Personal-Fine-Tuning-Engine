# Phase27 Workflow Simulation

This package is a dry run. It is not actual user-feedback evidence and must not
be used for product-value training claims.

## What Was Simulated

1. Exported the Phase27 collection pack.
2. Created 12 simulation-only feedback payloads.
3. Imported them through the Phase27 intake validator.
4. Persisted review decisions into a sandbox store.
5. Approved the 12 simulation signals for candidate generation.
6. Generated SFT/DPO candidate artifacts.
7. Opened the readiness gate with a simulated Qwen3-4B inventory.
8. Stopped at `ready_to_launch`; no real training or adapter eval was run.

## Guardrail Replay

The simulation also checks three negative branches:

- template feedback -> non_training
- missing consent -> blocked
- PII plus missing citation -> quarantined

## Summary

- accepted_pending_review_count: 12
- approved_for_candidate_count: 12
- sft_sample_count: 12
- dpo_pair_count: 12
- readiness_status: ready_for_real_training_probe
- training_attempt_status: ready_to_launch

## Important Boundary

This proves the workflow shape, not product value. The real next step is still
collecting attested human feedback and rerunning the same path without
simulation markers.
