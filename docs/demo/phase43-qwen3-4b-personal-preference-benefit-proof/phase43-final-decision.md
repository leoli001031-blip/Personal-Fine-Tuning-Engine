# PFE Phase43 final decision

## Decision

**archive**. Qwen3-4B SFT trained successfully, but the adapter did not beat base under the frozen dual-blind gate. No adapter is promoted.

## Real training

- SFT 1/12/30-step probes all performed real MPS optimizer/backward updates and produced valid PEFT safetensors.
- 12-step was selected over 30-step because its sanity score was slightly higher and repetition was lower, despite the 30-step loss being lower.
- DPO executed 12 real steps but produced non-finite metrics. The new gate correctly marked it failed and excluded it from eval.

## Product holdout

- 40 independent multi-turn sessions per arm, 360 real Qwen3-4B generation calls total.
- Base preference score: 0.6895.
- Runtime contract preference score: 0.9265.
- SFT preference score: 0.77.
- SFT correction responsiveness: 0.7 vs base 0.625.
- SFT training leakage: 0.0; diversity: 1.0.
- SFT privacy violation rate: 0.1.

## Blind evaluation

- Deterministic SFT win rate: 0.475.
- Independent Gemma4 SFT win rate: 0.175.
- Deterministic runtime win rate: 0.775.
- Independent Gemma4 runtime win rate: 0.8.

## Gate failures

`["deterministic_blind_win_rate_at_least_0_60", "independent_blind_win_rate_at_least_0_60", "preference_score_gain_at_least_0_10", "correction_gain_at_least_0_10", "generic_answer_rate_drop_at_least_0_15", "privacy_violation_zero"]`

## Interpretation

The training runtime is trustworthy and Qwen3-4B is locally trainable, but this 24-pair simulated dataset does not yet produce a better personal assistant. The runtime contract is much more effective than SFT on evidence and correction behavior, but it over-applies the contract, reduces diversity, increases repetition, and still repeats the privacy canary.

Phase43 proves simulated laboratory results only. It does not prove actual user benefit. Do not attach the archived adapter to the fourth Hermes Agent. First improve scenario-specific preference data and privacy behavior; only then run a guarded Hermes manual acceptance trial.
