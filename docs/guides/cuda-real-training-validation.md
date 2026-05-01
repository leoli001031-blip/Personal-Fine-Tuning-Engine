# CUDA Real Training Validation

This guide is for validating PEFT and Unsloth real-training paths on a Linux/CUDA host. The command uses PFE's normal real-training gate, parent preflight, materialized subprocess isolation, and diagnostics files.

## Dry Run

From the repository root:

```bash
.venv/bin/python tools/verify_cuda_real_training.py --backend all
```

Dry-run mode does not launch real training. It writes `cuda_real_training_summary.json` under a temporary `/tmp/pfe-cuda-verify-*` directory.

## Real PEFT Smoke

```bash
.venv/bin/python tools/verify_cuda_real_training.py \
  --backend peft \
  --run \
  --base-model sshleifer/tiny-gpt2 \
  --timeout-seconds 300
```

Expected outcome:

- `status=completed`
- `runner_status=completed`
- `returncode=0`
- `diagnostics_json`, `stdout_log`, and `stderr_log` paths are present
- adapter artifacts exist under the reported `adapter_path`

## Real Unsloth Smoke

```bash
.venv/bin/python tools/verify_cuda_real_training.py \
  --backend unsloth \
  --run \
  --unsloth-base-model unsloth/tinyllama-bnb-4bit \
  --timeout-seconds 600
```

If Unsloth or CUDA is unavailable, the result should be `blocked` or `failed` with diagnostics instead of crashing the parent process.

## Combined CUDA Pass

```bash
.venv/bin/python tools/verify_cuda_real_training.py \
  --backend all \
  --run \
  --peft-base-model sshleifer/tiny-gpt2 \
  --unsloth-base-model unsloth/tinyllama-bnb-4bit \
  --timeout-seconds 600
```

The script exits non-zero if any requested real backend does not complete. The most useful file to collect is:

```text
/tmp/pfe-cuda-verify-*/cuda_real_training_summary.json
```

Do not commit generated `trainer_job.py`, `trainer_job.json`, `training_job_result.json`, logs, diagnostics, or adapter output directories.
