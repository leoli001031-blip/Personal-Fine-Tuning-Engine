# Phase104 Final Decision

- Status: `phase104_runtime_contract_primary_adapters_archived`
- Recommendation: `runtime_contract_remains_primary`
- Product gate qualified: false
- Automatic promotion allowed: false
- Deployment allowed: false
- Evidence type: simulated_usage only
- Local generation calls: 240/270

## What was proved

- Phase100 closed the native generation boundary with no post-hoc truncation.
- Qwen3-4B SFT completed at 1, 12, and 30 steps with valid LoRA artifacts.
- Qwen3-4B DPO completed at 12 and 30 steps with finite MPS float32 metrics.

## What was not proved

- SFT did not improve the fresh product holdout and regressed format stability.
- DPO matched base on the fresh holdout but did not exceed it.
- In 20 paired three-turn simulated sessions, DPO had 0 wins, 19 ties, and 1 loss.
- No adapter product benefit was established; both adapters remain archived.

## Product path

Keep the Phase100 runtime contract as the primary path. The next investment should be a more diverse provenance and correction-following curriculum plus loss-target diagnostics, not additional steps on the same 32 examples.
