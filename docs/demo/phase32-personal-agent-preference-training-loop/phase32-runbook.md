# Phase32 Runbook

Phase32 reviews Phase31 historical Agent collaboration signals, builds abstract personal preference candidates, runs a small Qwen DPO probe when available, and evaluates base vs adapter on personal Agent holdout prompts.

Historical conversations are not realtime actual feedback. Do not commit raw Obsidian/AgentMemory text.

## Default Evidence Smoke

```bash
.venv/bin/python tools/phase32_personal_agent_preference_training_loop.py --clean-evidence
```

## Real Training And Eval Probe

```bash
.venv/bin/python tools/phase32_personal_agent_preference_training_loop.py \
  --clean-evidence \
  --run-real-training \
  --run-real-eval \
  --eval-device cpu
```
