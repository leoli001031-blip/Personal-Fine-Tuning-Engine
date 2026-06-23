# Phase29 Runbook

Phase29 proves whether PFE can convert reviewed feedback into training candidates and then into measurable adapter benefit.

## Default Smoke

```bash
.venv/bin/python tools/phase29_feedback_driven_tuning_benefit.py --clean-evidence
```

## Real 12-Step Probe

```bash
.venv/bin/python tools/phase29_feedback_driven_tuning_benefit.py \
  --clean-evidence \
  --run-real-training \
  --training-steps 12 \
  --train-sample-limit 40 \
  --eval-holdout-limit 30 \
  --run-runtime-reference \
  --run-dpo-fallback
```

Ollama qwen3.6 is a strong runtime reference only. Phase29 does not train Ollama GGUF and does not default to the 52G Qwen3.6-27B safetensors.
