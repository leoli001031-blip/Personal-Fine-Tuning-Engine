# Phase18 DPO Degeneration Guardrails Runbook

## Default Smoke

```bash
.venv/bin/python tools/phase18_to_phase22_route_convergence.py --clean-evidence
```

## Real Conservative DPO Guardrail Probe

```bash
.venv/bin/python tools/phase18_to_phase22_route_convergence.py \
  --clean-evidence \
  --allow-model-download \
  --run-real-phase18-dpo \
  --train-sample-limit 12
```

Phase18 archives any adapter that fails sanity gate before full holdout eval.
