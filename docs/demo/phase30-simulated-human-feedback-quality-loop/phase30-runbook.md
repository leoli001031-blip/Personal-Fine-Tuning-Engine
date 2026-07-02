# Phase30 Runbook

Phase30 builds simulated-human feedback quality evidence. It does not claim actual user feedback or production product lift.

## Default Smoke

```bash
.venv/bin/python tools/phase30_simulated_human_feedback_quality_loop.py --clean-evidence
```

## Optional 12-Step Training Probe

```bash
.venv/bin/python tools/phase30_simulated_human_feedback_quality_loop.py \
  --clean-evidence \
  --run-real-training \
  --clean-training-output
```
