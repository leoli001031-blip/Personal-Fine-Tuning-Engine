# Phase22 Product Route Decision

## Decision

- Recommendation: make_runtime_boundary_contract_the_main_product_path_and_keep_training_as_guarded_candidate_experiment
- Runtime boundary contract is the main product path.
- Training remains a guarded candidate experiment with archive/manual-review decisions only.

## Rationale

- Phase13/12 boundary-first runtime showed stable product boundary behavior.
- Phase17 proved DPO runtime viability but not product improvement.
- Phase18 adds sanity and degeneration gates so bad adapters are intercepted.
- Phase19 provides enough preference pairs to continue experiments, but not enough proof to move training into the main path.
- Phase20 shows the only fully materialized trainable Qwen CausalLM is still a small diagnostic model.

## Next Prompt Draft

Develop Phase23 runtime-contract product hardening plus guarded training-candidate review workflow; do not promote adapters unless they beat base without boundary regression.
