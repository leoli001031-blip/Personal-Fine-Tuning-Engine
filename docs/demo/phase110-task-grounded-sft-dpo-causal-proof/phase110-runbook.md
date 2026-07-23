# Phase110 Runbook

```bash
.venv/bin/python tools/phase110_task_grounded_sft_dpo_causal_proof.py prepare --clean
.venv/bin/python tools/phase110_task_grounded_sft_dpo_causal_proof.py diagnose-adapter
.venv/bin/python tools/phase110_task_grounded_sft_dpo_causal_proof.py train-sft --steps 1 --clean
.venv/bin/python tools/phase110_task_grounded_sft_dpo_causal_proof.py train-sft --steps 12 --clean
.venv/bin/python tools/phase110_task_grounded_sft_dpo_causal_proof.py train-sft --steps 30 --clean
.venv/bin/python tools/phase110_task_grounded_sft_dpo_causal_proof.py eval --variant base
.venv/bin/python tools/phase110_task_grounded_sft_dpo_causal_proof.py eval --variant phase109_dpo
.venv/bin/python tools/phase110_task_grounded_sft_dpo_causal_proof.py eval --variant phase110_sft
.venv/bin/python tools/phase110_task_grounded_sft_dpo_causal_proof.py analyze-sft
# Run DPO 1/12/30 and its eval only when analyze-sft exits 0.
.venv/bin/python tools/phase110_task_grounded_sft_dpo_causal_proof.py decide
.venv/bin/python tools/phase110_task_grounded_sft_dpo_causal_proof.py validate
```

All data is `simulated_usage` derived only from Phase31/32 aggregate signals. The 42-session holdout is fresh and excluded from training. Raw generations stay under `/private/tmp`. No external provider, paid API, push, deployment, automatic retraining, or automatic promotion is permitted.
