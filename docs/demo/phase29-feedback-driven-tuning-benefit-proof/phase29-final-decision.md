# Phase29 Final Decision

## Decision

- Recommendation: archive
- Status: blocked
- Auto promotion allowed: false
- Improved metrics: citation_hit_rate, unsupported_assertions
- Gate reasons: adapter_external_law_reference_rate_present, adapter_missing_info_ack_rate_not_above_base, adapter_structure_hit_rate_below_base, adapter_user_preference_adherence_rate_not_above_base

## Evidence

- Data source: operator_reviewed_feedback=40, actual_user_feedback=0
- SFT samples: 40
- DPO pairs: 40
- Primary 8B MLX training: failed (training_worker_failed)
- Primary adapter path: None
- DPO fallback: completed on Qwen/Qwen2.5-0.5B-Instruct
- DPO adapter valid: true at /Users/lichenhao/Desktop/PFE/trainer_job_outputs/phase29-dpo-fallback-qwen25-0_5b/dpo_adapter
- Effective eval: phase29_dpo_fallback

## Holdout Scores

| Metric | Base | Adapter |
| --- | ---: | ---: |
| structure_hit_rate | 0.033 | 0.0 |
| citation_hit_rate | 0.733 | 0.767 |
| safety_boundary_rate | 0.0 | 0.067 |
| explicit_boundary_rate | 0.0 | 0.067 |
| missing_info_ack_rate | 0.9 | 0.9 |
| user_preference_adherence_rate | 0.0 | 0.0 |
| external_law_reference_rate | 0.233 | 0.133 |
| unsupported_assertions | 36 | 33 |
| think_leak_rate | 0.0 | 0.0 |

## Ollama qwen3.6 Reference

Ollama qwen3.6 is a strong runtime reference, not a Phase29 training target.

| Metric | qwen3.6 reference |
| --- | ---: |
| structure_hit_rate | 0.0 |
| citation_hit_rate | 0.0 |
| safety_boundary_rate | 0.0 |
| explicit_boundary_rate | 0.0 |
| external_law_reference_rate | 0.4 |
| unsupported_assertions | 7 |

## Boundary

This is a PFE tuning-benefit proof, not Hermes integration. If the data source is mainly operator-reviewed feedback, a pass is technical success only and requires actual feedback collection next.
