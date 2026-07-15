# Phase79 Final Decision

Recommendation: **phase80_small_model_failure_taxonomy**

- Lifecycle status: `archive_12_step_sanity_failed`
- Real Qwen2.5-0.5B-Instruct 12-step training: `completed`
- Adapter artifact: `valid`
- 12-step sanity: `failed`
- Adapter sanity truncation rate: `0.1429`
- Base sanity target score: `0.53`
- Adapter sanity target score: `0.49`
- Runtime-contract sanity target score: `0.59`
- 120-step training: `not started by frozen sanity gate`
- Full 48-session eval: `not reached`
- Independent judge calls: `0`

Phase79 proves that real completion-only LoRA training can complete on CPU for the local 0.5B Qwen model. It does not prove adapter benefit. The first independent sanity exposed truncated repetition and weak persona answers, so the frozen gate correctly stopped the expensive 120-step run. All sessions are `simulated_usage`; there is no `actual_user_feedback`, product-benefit claim, promotion, Hermes attachment, or default change.
