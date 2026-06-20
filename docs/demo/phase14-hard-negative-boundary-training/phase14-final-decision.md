# Phase14 Final Decision

## Goal

- Test whether hard-negative boundary training can reduce external-law leakage and unsupported assertions.
- Keep the Phase13 runtime boundary contract as the product path unless the trainable adapter matches the 27B boundary reference and improves over its own 8B base.

## Phase13 Reference

- Qwen3.6-27B boundary base scores: `{"citation_hit_rate": 1.0, "explicit_boundary_rate": 1.0, "external_law_reference_rate": 0.0, "extra_text_after_first_block_rate": 0.0, "safety_boundary_rate": 1.0, "structure_hit_rate": 1.0, "think_leak_rate": 0.0, "unsupported_assertions": 0}`
- Phase13 conclusion remains valid: the runtime boundary contract is stable, while trainable adapters must be manually reviewed and must not auto-promote.

## Dataset And Training Strategy

- Holdout prompts: 80 hard-negative prompts, not used for training.
- Candidate samples: 120 chosen completions plus 120 rejected contrast answers.
- MLX backend mode: SFT only.
- Training strategy: train only safe chosen completions; save rejected answers as contrast evidence, not as true DPO/preference training.
- Output-only loss masking: enabled by the MLX training worker.

## Probe Iterations

### Attempt 1

- Evidence: `docs/demo/phase14-hard-negative-boundary-training/evidence-real-qwen3-8b-hard-negative/`
- Model: `mlx-community/Qwen3-8B-4bit`
- Steps: 12
- Result: training completed and adapter evaluated.
- Observation: adapter removed external-law leakage from `0.05` to `0.0`, but unsupported assertions stayed at `4` and safety boundary dropped to `0.95`.
- Diagnosis: one safe target used the phrase `不能建议直接签署`, which still contains signing-advice wording and can contaminate the safety scorer/model target.
- Decision: archive.

### Attempt 2

- Evidence: `docs/demo/phase14-hard-negative-boundary-training/evidence-real-qwen3-8b-hard-negative-v2/`
- Model: `mlx-community/Qwen3-8B-4bit`
- Steps: 12
- Training completed: true.
- Adapter path: `trainer_job_outputs/phase14-hard-negative-qwen3-8b-v2/adapters`
- Change: removed the problematic signing-advice wording from safe targets and increased missing-citation hard negatives.

## V2 Results

- 8B base scores: `{"citation_hit_rate": 0.9, "explicit_boundary_rate": 1.0, "external_law_reference_rate": 0.05, "extra_text_after_first_block_rate": 0.0, "safety_boundary_rate": 1.0, "structure_hit_rate": 1.0, "think_leak_rate": 0.0, "unsupported_assertions": 4}`
- 8B adapter scores: `{"citation_hit_rate": 0.95, "explicit_boundary_rate": 1.0, "external_law_reference_rate": 0.0, "extra_text_after_first_block_rate": 0.0, "safety_boundary_rate": 0.975, "structure_hit_rate": 1.0, "think_leak_rate": 0.0, "unsupported_assertions": 2}`
- Deltas vs 8B base: `{"citation_delta_vs_mid_base": 0.05, "external_law_delta_vs_mid_base": -0.05, "safety_delta_vs_mid_base": -0.025, "unsupported_delta_vs_mid_base": -2}`

## Adapter Gate

- Recommendation: archive.
- Status: blocked.
- Improved vs 8B base: false.
- Reasons:
  - `adapter_citation_hit_rate_below_qwen36_boundary_base`
  - `adapter_safety_boundary_rate_below_qwen36_boundary_base`
  - `adapter_unsupported_assertions_above_qwen36_boundary_base`
  - `hard_negative_training_not_improved_vs_mid_base`

## Why No 30-Step Probe

The 12-step V2 adapter produced useful movement on external-law leakage and unsupported assertions, but it still regressed safety boundary from `1.0` to `0.975`. Phase13 already showed that longer SFT can amplify boundary regressions, so more blind SFT steps are not justified here.

## Final Decision

Phase14 archives the trained adapter. The hard-negative SFT curriculum is directionally useful, but it does not beat the Phase13 runtime boundary contract and does not satisfy the adapter gate.

Next work should keep the runtime boundary contract as the main product path while exploring true preference/DPO-capable training or more precise boundary-error sampling.
