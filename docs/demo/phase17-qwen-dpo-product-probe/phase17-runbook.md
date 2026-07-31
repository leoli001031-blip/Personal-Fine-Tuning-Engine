# Phase17 Qwen DPO Product Probe Runbook

Phase17 tests product benefit, not runtime viability. Phase16 already proved the DPO runtime can execute.

## Default Smoke

```bash
.venv/bin/python tools/phase17_qwen_dpo_product_probe.py \
  --evidence-dir docs/demo/phase17-qwen-dpo-product-probe/evidence \
  --clean-evidence \
  --skip-real-qwen-dpo
```

## Real Qwen DPO Product Probe

```bash
.venv/bin/python tools/phase17_qwen_dpo_product_probe.py \
  --evidence-dir docs/demo/phase17-qwen-dpo-product-probe/evidence-real-qwen-dpo \
  --clean-evidence \
  --allow-model-download \
  --run-real-qwen-dpo \
  --train-sample-limit 12 \
  --eval-holdout-limit 30 \
  --training-output-dir trainer_job_outputs/phase17-qwen-dpo-product-probe
```

The adapter must beat the selected Qwen base on at least one core metric without any boundary regression. Otherwise archive.
