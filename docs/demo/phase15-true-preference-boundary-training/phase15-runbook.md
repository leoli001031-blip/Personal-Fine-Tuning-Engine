# Phase15 True-Preference Boundary Training Runbook

Phase15 turns Phase14 rejected hard negatives into real DPO-shaped preference pairs. It does not treat rejected answers as side evidence anymore: every training row has `sample_type=dpo`, `chosen`, and `rejected`.

## Default Smoke

```bash
.venv/bin/python tools/phase15_preference_boundary_training.py \
  --evidence-dir docs/demo/phase15-true-preference-boundary-training/evidence \
  --clean-evidence \
  --skip-real-dpo
```

## Strict DPO Preflight

```bash
.venv/bin/python tools/phase15_preference_boundary_training.py \
  --evidence-dir docs/demo/phase15-true-preference-boundary-training/evidence-real-dpo-preflight \
  --clean-evidence \
  --run-real-dpo
```

If `trl` or `datasets` are missing, archive with blocked evidence instead of falling back to SFT or mock training.
