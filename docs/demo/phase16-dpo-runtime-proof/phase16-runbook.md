# Phase16 DPO Runtime Proof Runbook

Phase16 proves that the DPO runtime can execute a real `trl.DPOTrainer` job and materialize an adapter artifact. This is a runtime proof only, not a product adapter quality claim.

## Default Smoke

```bash
.venv/bin/python tools/phase16_dpo_runtime_proof.py \
  --evidence-dir docs/demo/phase16-dpo-runtime-proof/evidence \
  --clean-evidence \
  --skip-real-dpo-proof
```

## Real Tiny DPO Runtime Proof

```bash
.venv/bin/python tools/phase16_dpo_runtime_proof.py \
  --evidence-dir docs/demo/phase16-dpo-runtime-proof/evidence-real-dpo-tiny \
  --clean-evidence \
  --run-real-dpo-proof \
  --train-sample-limit 2 \
  --training-output-dir trainer_job_outputs/phase16-dpo-runtime-proof-tiny
```

Passing this proof only permits a later Qwen DPO probe after manual review. It never promotes a product adapter by itself.
