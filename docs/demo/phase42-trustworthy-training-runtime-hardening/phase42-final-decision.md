# PFE Phase42 Final Decision

## Decision

`reliability_gate_passed=true`

Recommendation: return to personal-preference training only after manual review
of the diverse simulated candidate set. This is a runtime/training-integrity
decision, not a product-benefit claim.

## Verified Results

### Adapter 20260617-006

- Root adapter is a valid 34 MB PEFT safetensors file with 336 LoRA tensors and
  8,798,208 parameters.
- Base passed the 20-prompt generic holdout: relevance `0.90`, unique ratio
  `1.00`, duplicate rate `0.00`.
- Adapter failed: relevance `0.00`, unique ratio `0.25`, duplicate rate `0.75`.
- All 20 responses leaked `未量化零点五B` or `深蓝闭环` training strings.
- The adapter was archived, not deleted. `latest` was removed and PFE now reports
  `base_ready`.

### Real Local Training

- Qwen2.5-0.5B completed 12 real optimizer/backward steps.
- LoRA trainable parameters changed and the complete loss history is persisted.
- The resulting adapter has 336 LoRA tensors and passes safetensors parsing.
- The artifact is an integrity probe only and was not evaluated for product gain,
  installed, promoted, or set as latest.

### Hermes Runtime

- Live HTTP SSE passed.
- The standard OpenAI SDK consumed five chunks, non-empty content, and terminal
  `finish_reason=stop` followed by `[DONE]`.
- Client cancellation propagates to the upstream task.
- Streaming is honestly reported as `buffered_backend`; token-level generation
  remains future work.

### Context and Security

- Real tokenizer smoke: original `4427`, retained prompt `4088`, effective
  context `4096`, completion budget `8`, `input_truncated=true`, and
  `finish_reason=length`.
- Filesystem-backed user IDs are safely mapped and cannot traverse directories.
- Long-term memory is read/written only with `metadata.memory_consent=true`.
- Dashboard metrics, training, signals, adapters, and health all use management
  access control; remote and wrong-key requests are rejected.

### Candidate Quality

- Original Phase41 candidate is blocked by the new diversity gate.
- Phase41-v2 contains 24 scenario-specific pairs with chosen, rejected, and
  prompt unique ratios of `1.00`; exact duplicate rate is `0.00`.
- Phase41-v2 remains simulated evidence and is not approved for training.

## Boundaries

- `actual_product_benefit_claim_allowed=false`
- `auto_training_allowed=false`
- `auto_promotion_allowed=false`
- No 27B training or download was performed.
- `videos/` and existing Hermes profiles were not modified.

## Verification Gate

- Phase42 focused/regression suite: 95 passed.
- Unit suite: 1204 passed, 30 deselected.
- Surface suite: 162 passed.
- E2E mock suite: 13 passed, 22 deselected.
- `smoke-beta`, live HTTP SSE, OpenAI SDK streaming, and Dashboard smoke passed.

Authoritative machine decision: `phase42-final-decision.json`.
