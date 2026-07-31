# Phase13 Final Decision

## Runtime Contract

- Contract id: contract_boundary_summary
- Runtime/API field: response_contract
- Output: 摘要 / 风险提示 / 引用依据 / 人工确认

## Qwen3.6 Boundary Base

- Status: completed
- Scores: `{"citation_hit_rate": 1.0, "explicit_boundary_rate": 1.0, "external_law_reference_rate": 0.0, "extra_text_after_first_block_rate": 0.0, "safety_boundary_rate": 1.0, "structure_hit_rate": 1.0, "think_leak_rate": 0.0, "unsupported_assertions": 0}`

## Mid Model Training

- Model: mlx-community/Qwen3-8B-4bit
- Real training: completed
- Adapter path: /Users/lichenhao/Desktop/PFE/trainer_job_outputs/phase13-mid-model-30step/adapters
- Error type: None

## Mid Model Probe Results

- 12-step adapter: structure 1.0, citation 0.933, safety 0.9, external law reference 0.033, unsupported assertions 4, recommendation archive.
- 30-step adapter: structure 1.0, citation 0.967, safety 0.967, external law reference 0.5, unsupported assertions 16, recommendation archive.
- 8B base reference: structure 1.0, citation 0.933, safety 1.0, external law reference 0.033, unsupported assertions 1.
- 30-step improves citation slightly over the 8B base, but it increases external-law leakage and unsupported assertions; it does not beat the 27B boundary contract.

## Adapter Gate

- Recommendation: archive
- Status: blocked
- Reasons: ['adapter_citation_hit_rate_below_qwen36_boundary_base', 'adapter_safety_boundary_rate_below_qwen36_boundary_base', 'adapter_unsupported_assertions_above_qwen36_boundary_base', 'adapter_external_law_reference_present']

Phase13 never auto-promotes. Passing adapters are limited to `promote_after_manual_review`.
Final recommendation: archive the Phase13 adapters and keep the runtime boundary contract as the primary product path while improving training data and safety negatives.
