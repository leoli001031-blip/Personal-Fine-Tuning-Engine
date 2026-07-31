# Phase11 Capacity Probe Runbook

This runbook reproduces the Phase11 base-model capacity probe. It does not train adapters.

## Preconditions

- Worktree: `/Users/lichenhao/Desktop/PFE`
- Branch: `codex/phase11-capacity-probe-qwen-models`
- Python: `.venv/bin/python`
- Required package path: local editable `pfe-core`
- Required runtime: `mlx_lm`

## Sanity Checks

```bash
.venv/bin/python -m py_compile tools/phase11_capacity_probe.py tests/test_phase11_capacity_probe.py
.venv/bin/python -m pytest tests/test_phase11_capacity_probe.py -q
```

Expected result:

```text
3 passed
```

## 0.6B Smoke

```bash
.venv/bin/python tools/phase11_capacity_probe.py \
  --evidence-dir docs/demo/phase11-capacity-probe/evidence-smoke-0.6b \
  --clean-evidence \
  --model mlx-community/Qwen3-0.6B-4bit \
  --holdout-count 2 \
  --max-tokens 96 \
  --repetition-penalty 1.2
```

Saved evidence:

- `docs/demo/phase11-capacity-probe/evidence-smoke-0.6b/capacity_probe_report.json`
- `docs/demo/phase11-capacity-probe/evidence-smoke-0.6b/output_examples.md`

## 0.6B And 8B Baseline Prompt

```bash
.venv/bin/python tools/phase11_capacity_probe.py \
  --evidence-dir docs/demo/phase11-capacity-probe/evidence-qwen3-0.6b-8b-base \
  --clean-evidence \
  --model mlx-community/Qwen3-0.6B-4bit \
  --model mlx-community/Qwen3-8B-4bit \
  --holdout-count 10 \
  --max-tokens 192 \
  --repetition-penalty 1.2
```

Saved evidence:

- `docs/demo/phase11-capacity-probe/evidence-qwen3-0.6b-8b-base/capacity_probe_report.json`
- `docs/demo/phase11-capacity-probe/evidence-qwen3-0.6b-8b-base/output_examples.md`

## 27B Baseline Prompt

```bash
.venv/bin/python tools/phase11_capacity_probe.py \
  --evidence-dir docs/demo/phase11-capacity-probe/evidence-qwen36-27b-base-full \
  --clean-evidence \
  --model mlx-community/Qwen3.6-27B-4bit \
  --holdout-count 10 \
  --max-tokens 192 \
  --repetition-penalty 1.2
```

Saved evidence:

- `docs/demo/phase11-capacity-probe/evidence-qwen36-27b-base-full/capacity_probe_report.json`
- `docs/demo/phase11-capacity-probe/evidence-qwen36-27b-base-full/output_examples.md`

## 8B No-Think Prompt

```bash
.venv/bin/python tools/phase11_capacity_probe.py \
  --evidence-dir docs/demo/phase11-capacity-probe/evidence-qwen3-8b-no-think-base \
  --clean-evidence \
  --model mlx-community/Qwen3-8B-4bit \
  --holdout-count 10 \
  --max-tokens 192 \
  --repetition-penalty 1.2 \
  --prompt-mode no_think_four_line
```

Saved evidence:

- `docs/demo/phase11-capacity-probe/evidence-qwen3-8b-no-think-base/capacity_probe_report.json`
- `docs/demo/phase11-capacity-probe/evidence-qwen3-8b-no-think-base/output_examples.md`

## 27B No-Think Prompt

```bash
.venv/bin/python tools/phase11_capacity_probe.py \
  --evidence-dir docs/demo/phase11-capacity-probe/evidence-qwen36-27b-no-think-base \
  --clean-evidence \
  --model mlx-community/Qwen3.6-27B-4bit \
  --holdout-count 10 \
  --max-tokens 192 \
  --repetition-penalty 1.2 \
  --prompt-mode no_think_four_line
```

Saved evidence:

- `docs/demo/phase11-capacity-probe/evidence-qwen36-27b-no-think-base/capacity_probe_report.json`
- `docs/demo/phase11-capacity-probe/evidence-qwen36-27b-no-think-base/output_examples.md`

## Read The Summary

```bash
jq '.runs[] | {label, model_id, prompt_mode, scores, decision}' \
  docs/demo/phase11-capacity-probe/comparison_summary.json
```

The current recommendation is to avoid immediate large-model training. Qwen3.6-27B-4bit is the best capacity candidate, but the next development loop should first fix the explicit safety-boundary target and raw generation end boundary.
