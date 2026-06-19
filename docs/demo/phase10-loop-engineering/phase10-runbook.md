# Phase10 Runbook

Phase10 uses loop engineering: isolate one behavior, run it, measure it, and
only widen the loop when the gate passes.

## Default Smoke

```bash
.venv/bin/python tools/phase10_loop_engineering_smoke.py \
  --workdir /tmp/pfe-phase10-default-smoke \
  --evidence-dir docs/demo/phase10-loop-engineering/evidence \
  --clean-evidence
```

Expected result:

- creates 60 format-curriculum candidate samples;
- does not run real training;
- archives the candidate because real training/eval evidence is missing;
- skips Qwen3.6 because the small-model gate has not passed.

## Stage A Real Qwen3-0.6B

```bash
.venv/bin/python tools/phase10_loop_engineering_smoke.py \
  --workdir /tmp/pfe-phase10-stage-a-real \
  --evidence-dir docs/demo/phase10-loop-engineering/evidence-qwen3-0.6b-stage-a \
  --clean-evidence \
  --allow-remote-download \
  --run-real-training \
  --run-real-eval \
  --epochs 12 \
  --timeout 3600 \
  --eval-samples 10 \
  --eval-max-tokens 96 \
  --eval-temperature 0.0 \
  --eval-top-p 0.0 \
  --eval-repetition-penalty 1.1
```

Expected artifacts:

- `train_log.json`
- `training_job_result.json`
- `training_attempt.json`
- `eval_report.json`
- `decision.json`
- `output_examples.md`
- `comparison_summary.json`

## Decode Probe

The first real Stage A run failed the gate. A decode probe was run without
changing the scorer:

```bash
.venv/bin/python tools/phase10_loop_engineering_smoke.py \
  --workdir /tmp/pfe-phase10-stage-a-decode-probe \
  --evidence-dir docs/demo/phase10-loop-engineering/evidence-qwen3-0.6b-stage-a-decode-probe \
  --clean-evidence \
  --allow-remote-download \
  --run-real-training \
  --run-real-eval \
  --epochs 12 \
  --timeout 3600 \
  --eval-samples 10 \
  --eval-max-tokens 192 \
  --eval-temperature 0.0 \
  --eval-top-p 0.0 \
  --eval-repetition-penalty 1.2
```

This probe checks whether the failure is mainly eval truncation. It still
archives because structure remains below base and safety boundary has no
measured lift.

## Gate

The adapter may only become `promote_after_manual_review` when all of these
are true on holdout prompts:

- normalized structure hit rate is higher than base;
- citation hit rate is not lower than base;
- unsupported assertions are not higher than base;
- safety boundary rate has a real lift.

Even then, Phase10 does not auto-promote. Qwen3.6 4-bit is only eligible for
preflight/load smoke after the small-model gate passes.
