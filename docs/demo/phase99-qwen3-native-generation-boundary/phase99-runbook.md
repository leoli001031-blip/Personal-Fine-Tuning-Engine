# Phase99 Runbook

1. Freeze eight fresh simulated sessions and verify isolation from prior holdouts.
2. Render Qwen3 prompts with `enable_thinking=false` and verify the empty thinking block suffix.
3. Block think tokens, fake assistant/user roles, tool-response continuation, and Qwen chat restart tokens inside `model.generate`.
4. Stop generation inside `model.generate` when the first expected answer block is complete; preserve model EOS as a separate native termination path.
5. Persist the full raw output without post-hoc truncation and score format, provenance, repetition, think leak, extra text, and termination evidence.
6. Unlock 4B SFT only when every frozen gate passes. Otherwise archive without training.

Phase99 stopped at step 6. The runaway problem is fixed, but provenance explicitness and complete-boundary-before-EOS remain unresolved.
