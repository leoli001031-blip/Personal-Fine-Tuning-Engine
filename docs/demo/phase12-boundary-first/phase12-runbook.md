# Phase12 Boundary-First Runbook

Phase12 tests whether Qwen3.6-27B can obey PFE's product boundary before adapter training.

## Smoke

```bash
.venv/bin/python tools/phase12_boundary_first.py \
  --evidence-dir docs/demo/phase12-boundary-first/evidence \
  --clean-evidence \
  --skip-model-probe
```

## Qwen3.6 Base Probe

```bash
.venv/bin/python tools/phase12_boundary_first.py \
  --evidence-dir docs/demo/phase12-boundary-first/evidence-real-qwen36-27b \
  --clean-evidence \
  --model mlx-community/Qwen3.6-27B-4bit \
  --prompt-mode phase10 \
  --prompt-mode no_think_four_line \
  --prompt-mode boundary_first_four_line \
  --prompt-mode boundary_first_chat_no_think \
  --holdout-count 10 \
  --candidate-count 40 \
  --max-tokens 192 \
  --repetition-penalty 1.2
```

## 12-Step Training Probe

Only run this after the base probe has selected `boundary_first_chat_no_think`.
On the 128GB local Mac, the first real 27B attempt reached MLX/Metal training and terminated with
`kIOGPUCommandBufferCallbackErrorOutOfMemory` before producing an adapter artifact. Treat that as an
archive/blocking result unless a later runner proves otherwise.

```bash
.venv/bin/python tools/phase12_boundary_first.py \
  --evidence-dir docs/demo/phase12-boundary-first/evidence-real-qwen36-27b \
  --clean-evidence \
  --model mlx-community/Qwen3.6-27B-4bit \
  --prompt-mode phase10 \
  --prompt-mode no_think_four_line \
  --prompt-mode boundary_first_four_line \
  --prompt-mode boundary_first_chat_no_think \
  --holdout-count 10 \
  --candidate-count 40 \
  --max-tokens 192 \
  --repetition-penalty 1.2 \
  --run-training-probe \
  --training-steps 12 \
  --train-sample-limit 40 \
  --train-max-seq-length 1024 \
  --clean-training-output
```

Passing the probe requires strong structure, citation, explicit safety boundary, fewer unsupported assertions than Phase11, no external law references, and no raw `<think>` leak.
