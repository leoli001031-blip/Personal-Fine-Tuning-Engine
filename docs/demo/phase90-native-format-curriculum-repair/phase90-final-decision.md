# Phase90 Final Decision

- Status: `archive_phase90_native_format_not_qualified`
- Recommendation: `archive_and_reassess_model_capacity_or_objective`
- Selected curriculum: `balanced`
- Evidence type: simulated benchmark with real local Qwen2.5-1.5B training and generation
- Actual user feedback: 0
- Product gate qualified: false
- Automatic promotion, deployment, and Hermes attachment: forbidden

## Three-arm result

| Variant | Raw score | Native format | False block | Truncation | Runtime fallback |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base | 0.685 | 0.200 | 0.050 | 0.150 | 0.825 |
| Phase89 archived adapter | 0.740 | 0.200 | 0.050 | 0.075 | 0.825 |
| Phase90 balanced candidate | 0.715 | 0.200 | 0.100 | 0.100 | 0.825 |

Phase90 improved raw score over base by 0.030 but underperformed the archived Phase89 adapter by 0.025. Prompt-contract alignment did not improve native format or runtime fallback and increased false blocks versus both comparators.

## Simulated blind user review

- Candidate wins: 5
- Base wins: 3
- Ties: 32
- Candidate finding count: 44
- Review status: failed

This review is explicitly simulated and is not actual user feedback. Raw outputs were kept only in a mode-0600 temporary cache and deleted during finalization.

## Next engineering judgment

Do not add more steps to this 1.5B curriculum. The failure pattern now points to either model-capacity limits or an objective that needs stronger preference supervision. Preserve the Phase89 adapter as the better archived diagnostic and test one controlled next move: a small DPO preference probe on exact native-format and false-block contrasts, or the same frozen SFT objective on a larger locally trainable model. Neither path may reuse the Phase90 holdout for training.
