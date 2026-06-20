# Phase14 Hard-Negative Boundary Training Runbook

Phase14 tests whether hard-negative boundary samples can reduce external-law leakage and unsupported assertions in a trainable 8B adapter. The current MLX backend is SFT-only, so rejected answers are saved as contrast evidence and only chosen completions are trained.

## Default Smoke

```bash
.venv/bin/python tools/phase14_hard_negative_boundary_training.py \
  --evidence-dir docs/demo/phase14-hard-negative-boundary-training/evidence \
  --clean-evidence \
  --skip-real-models
```

## Real 8B Hard-Negative Probe V1

```bash
.venv/bin/python tools/phase14_hard_negative_boundary_training.py \
  --evidence-dir docs/demo/phase14-hard-negative-boundary-training/evidence-real-qwen3-8b-hard-negative \
  --clean-evidence \
  --run-mid-training \
  --training-steps 12 \
  --training-output-dir trainer_job_outputs/phase14-hard-negative-qwen3-8b
```

## Real 8B Hard-Negative Probe V2

V2 removes target wording that could be scored as signing advice and increases missing-citation hard negatives.

```bash
.venv/bin/python tools/phase14_hard_negative_boundary_training.py \
  --evidence-dir docs/demo/phase14-hard-negative-boundary-training/evidence-real-qwen3-8b-hard-negative-v2 \
  --clean-evidence \
  --run-mid-training \
  --holdout-count 80 \
  --candidate-count 120 \
  --training-steps 12 \
  --train-sample-limit 80 \
  --train-max-seq-length 768 \
  --training-output-dir trainer_job_outputs/phase14-hard-negative-qwen3-8b-v2 \
  --clean-training-output \
  --eval-max-tokens 192 \
  --repetition-penalty 1.2
```

If the 12-step probe still safety-regresses or fails to match the 27B boundary reference, archive rather than blindly running more steps.
