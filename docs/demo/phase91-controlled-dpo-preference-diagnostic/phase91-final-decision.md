# Phase91 Final Decision

- Status: `archive_phase91_12step_dpo_non_finite`
- Recommendation: `archive_and_move_to_larger_model_or_stable_dpo_backend`
- Model: local Qwen2.5-1.5B-Instruct
- Parent: Phase89 25-step SFT adapter
- Evidence: simulated usage only
- Actual user feedback: 0
- Product gate qualified: false

## What completed

- 72 controlled DPO pairs passed quality checks, with 24 pairs for each of exact three-line format, false block, and provenance.
- The 40-session holdout had zero workflow, exact-text, and near-duplicate overlap with training, Phase89 holdout, and Phase90 holdout.
- DPO dependencies, dry-run plans, parent adapter SHA, and token boundaries passed.
- Real local progressive DPO started with the required `base + merged Phase89 parent` lineage.

## Why it stopped

The 12-step trainer produced non-finite metrics from log index 3. The executor reported `grad_norm=nan` and rejected the run before a valid adapter artifact existed. The attempt therefore records zero accepted optimizer steps and no candidate adapter.

Under the frozen gate, no candidate means:

- no sanity generation;
- no 30-step probe;
- no base / Phase89 / DPO full comparison;
- no promotion, deployment, or Hermes attachment.

This is a runtime/numerical-stability failure, not evidence that DPO improved or harmed product behavior.

## Next judgment

Do not retry the same 1.5B float16 MPS DPO recipe by silently changing thresholds. The next controlled choice is either a numerically stable DPO backend/dtype experiment with a fresh freeze, or moving the same preference contract to a larger model/runtime already proven stable for local training.
