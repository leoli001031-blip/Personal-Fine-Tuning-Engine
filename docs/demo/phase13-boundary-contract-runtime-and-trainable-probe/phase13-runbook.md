# Phase13 Boundary Contract Runtime And Trainable Probe Runbook

Phase13 productizes the Phase12 boundary-first contract and tests whether a trainable mid-size model can beat the prompt contract.

## Default Smoke

```bash
.venv/bin/python tools/phase13_boundary_contract_probe.py \
  --evidence-dir docs/demo/phase13-boundary-contract-runtime-and-trainable-probe/evidence \
  --clean-evidence \
  --skip-real-models
```

## Qwen3.6-27B Boundary Base Probe

```bash
.venv/bin/python tools/phase13_boundary_contract_probe.py \
  --evidence-dir docs/demo/phase13-boundary-contract-runtime-and-trainable-probe/evidence-real-qwen36-27b-base \
  --clean-evidence \
  --run-qwen36-base
```

## Trainable Mid-Model Probe

```bash
.venv/bin/python tools/phase13_boundary_contract_probe.py \
  --evidence-dir docs/demo/phase13-boundary-contract-runtime-and-trainable-probe/evidence-trainable-mid-model \
  --clean-evidence \
  --run-mid-training \
  --training-steps 12
```

## Trainable Mid-Model 30-Step Probe

```bash
.venv/bin/python tools/phase13_boundary_contract_probe.py \
  --evidence-dir docs/demo/phase13-boundary-contract-runtime-and-trainable-probe/evidence-trainable-mid-model-30step \
  --clean-evidence \
  --run-mid-training \
  --training-steps 30 \
  --training-output-dir trainer_job_outputs/phase13-mid-model-30step
```
