# Phase96-98 Runbook

1. Validate the local Qwen2.5-1.5B and Qwen3-4B model artifacts.
2. Freeze eight fresh simulated capacity sessions with zero overlap against prior holdouts.
3. Run 24 local calls per model with identical prompts, order, decoding, and raw-output scoring.
4. Continue to Qwen3-4B SFT only if the larger base has a strict core gain without ordinary-task, repetition, think-leak, safety, or privacy regression.
5. Archive when the capacity gate fails. Do not train, promote, deploy, attach Hermes, or claim actual-user benefit.

Phase96 stopped at step 5. The larger model has meaningful capacity gains, but its current no-think and generation-stop behavior is not stable enough to justify adapter training.
