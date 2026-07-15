# Phase80 Final Decision

Recommendation: **phase81_trainable_mid_model_selection**

- Lifecycle status: `diagnosis_completed`
- Failure classification: `small_model_capacity_dominant_with_length_cost`
- Real low-LR 12-step training: `completed`
- 0.5B base target score: `0.4967`
- Phase79 high-LR adapter target score: `0.5633`
- Phase80 low-LR adapter target score: `0.4689`
- 0.5B runtime-contract score: `0.5333`
- 4B base score: `0.6222`
- 4B runtime-contract score: `0.7122`
- Low-LR adapter gain over 0.5B base: `-0.0278`
- Runtime gain over 0.5B base: `0.0366`
- 4B runtime gap over 0.5B runtime: `0.1789`
- Real generation calls: `441`

Phase80 is a diagnostic experiment, not an adapter promotion test. All sessions are fresh `simulated_usage`; there is no `actual_user_feedback`. Training completion, decoding stability, runtime-contract benefit, and model-capacity gap are reported separately. No adapter is promoted, attached to Hermes, or made a product default.
