# Phase17 Final Decision

## Goal

- Test whether real Qwen DPO training improves product boundary behavior over the selected Qwen base.
- Do not treat DPO runtime success as product success.

## Model

- Selected model: Qwen/Qwen2.5-0.5B-Instruct
- Selection status: selected
- Selection reason: small HF CausalLM model suitable for CPU DPO proof and 30-prompt eval

## Training

- Real training: completed
- Adapter valid: True
- Adapter path: /Users/lichenhao/Desktop/PFE/trainer_job_outputs/phase17-qwen-dpo-product-probe/dpo_adapter

## Eval

- Real model calls: True
- Base scores: `{"citation_hit_rate": 0.567, "explicit_boundary_rate": 0.6, "external_law_reference_rate": 0.067, "extra_text_after_first_block_rate": 0.0, "safety_boundary_rate": 0.6, "structure_hit_rate": 0.633, "think_leak_rate": 0.0, "unsupported_assertions": 14}`
- Adapter scores: `{"citation_hit_rate": 0.0, "explicit_boundary_rate": 0.0, "external_law_reference_rate": 0.0, "extra_text_after_first_block_rate": 0.0, "safety_boundary_rate": 0.0, "structure_hit_rate": 0.0, "think_leak_rate": 0.0, "unsupported_assertions": 30}`

## Decision

- Recommendation: archive
- Status: blocked
- Improved metrics: []
- Reasons: ['adapter_citation_hit_rate_below_base', 'adapter_explicit_boundary_rate_below_base', 'adapter_has_no_core_metric_improvement_over_base', 'adapter_safety_boundary_rate_below_base', 'adapter_structure_hit_rate_below_base', 'adapter_unsupported_assertions_above_base']

Phase17 promotes only after manual review and only if adapter eval truly beats base without boundary regression.
