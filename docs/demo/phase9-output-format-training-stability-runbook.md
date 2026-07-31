# Phase9 Output Format Training Stability Runbook

Phase9 keeps the Phase8 real signal-to-training loop, but focuses on one narrower question:

Can a small local Qwen3-0.6B adapter learn a stable four-section contract output shape before PFE considers larger models?

Required shape:

```text
摘要：...
风险提示：...
引用依据：[source_id:chunk_id]
人工确认：不输出法律结论；...
```

This phase does not add UI and does not touch `videos/`.

## Phase8 Baseline Review

Canonical Phase8 evidence:

- `docs/demo/phase8-signal-quality-training/evidence-real-qwen3-0.6b/eval_report.json`
- `docs/demo/phase9-output-format-training-stability/evidence-real-qwen3-0.6b/phase8_baseline_review.json`

Phase8 real result:

- 60 quality signals passed
- 60 candidate samples passed
- Qwen3-0.6B MLX real training completed
- adapter citation hit rate improved from `0.2` to `0.5`
- unsupported assertions improved from `18` to `15`
- structure adherence fell from `0.175` to `0.0`
- safety boundary rate stayed `0.0`
- decision: `archive`

Real Phase8 output text showed repeated `答案：` / `资料片段：` patterns. The core issue was not only scoring. The adapter learned prompt-like text and source-copy behavior instead of a clean answer boundary.

## Phase9 Changes

Phase9 makes three minimal changes:

1. Training targets are short, fixed-shape four-line completions.
2. Candidate targets avoid copying source excerpts and keep `source_id:chunk_id` citations.
3. The MLX backend now preserves `instruction/chosen` as `prompt/completion` and uses MLX prompt masking for output-only loss.

Backend evidence:

- `docs/demo/phase9-output-format-training-stability/evidence-real-qwen3-0.6b/training_job_result.json`
- expected metadata:

```json
{
  "dataset_format": "prompt_completion_output_only_loss",
  "output_only_loss_masking": true
}
```

Eval generation uses deterministic decoding with a shorter output cap:

- `temperature=0.0`
- `top_p=0.0`
- `repetition_penalty=1.05`
- `max_tokens=128`

The scorer was tightened instead of relaxed: structure credit requires section labels at line start, not just the words appearing somewhere in the response.

## Default Smoke

Use this to validate data routing and evidence generation without real training:

```bash
.venv/bin/python tools/phase9_output_format_training_smoke.py \
  --evidence-dir docs/demo/phase9-output-format-training-stability/evidence \
  --clean-evidence
```

Expected result:

- `quality_report.json` shows 60 quality-passed signals
- `candidate_samples.jsonl` shows 60 holdout-free SFT candidates
- `eval_report.json` is blocked because no real model calls were requested
- `decision.json` archives the trial

## Real Qwen3-0.6B Trial

Canonical Phase9 real run:

```bash
HF_HUB_DISABLE_XET=1 .venv/bin/python tools/phase9_output_format_training_smoke.py \
  --model-id mlx-community/Qwen3-0.6B-4bit \
  --allow-remote-download \
  --run-real-training \
  --run-real-eval \
  --strict-real \
  --strict-real-eval \
  --signal-count 60 \
  --candidate-limit 60 \
  --holdout-count 10 \
  --eval-samples 10 \
  --eval-max-tokens 128 \
  --eval-temperature 0.0 \
  --eval-top-p 0.0 \
  --eval-repetition-penalty 1.05 \
  --epochs 12 \
  --timeout 2400 \
  --evidence-dir docs/demo/phase9-output-format-training-stability/evidence-real-qwen3-0.6b \
  --clean-evidence \
  --keep-workdir
```

Evidence:

- `docs/demo/phase9-output-format-training-stability/evidence-real-qwen3-0.6b/train_log.json`
- `docs/demo/phase9-output-format-training-stability/evidence-real-qwen3-0.6b/training_attempt.json`
- `docs/demo/phase9-output-format-training-stability/evidence-real-qwen3-0.6b/training_job_result.json`
- `docs/demo/phase9-output-format-training-stability/evidence-real-qwen3-0.6b/eval_report.json`
- `docs/demo/phase9-output-format-training-stability/evidence-real-qwen3-0.6b/output_examples.md`
- `docs/demo/phase9-output-format-training-stability/evidence-real-qwen3-0.6b/comparison_summary.json`
- `docs/demo/phase9-output-format-training-stability/evidence-real-qwen3-0.6b/decision.json`

Observed result:

- real MLX training completed
- real base/adapter eval completed
- output-only loss masking was active
- holdout count: 10
- base citation hit rate: `0.7`
- adapter citation hit rate: `0.3`
- base structure hit rate: `0.725`
- adapter structure hit rate: `0.325`
- base unsupported assertions: `13`
- adapter unsupported assertions: `17`
- base safety boundary rate: `0.0`
- adapter safety boundary rate: `0.0`
- final recommendation: `archive`

The adapter is not more stable than base. It sometimes learned the four labels, but it also produced repeated legal words, dropped citations, used alternative labels such as `证据`, and failed the explicit human-confirmation safety boundary.

## Eval Gate

Promotion requires all of the following:

- real Qwen3-0.6B MLX training completed
- real base vs adapter holdout eval completed
- holdout prompts never entered training
- adapter structure hit rate is higher than base
- adapter citation hit rate is not lower than base
- adapter unsupported assertions are not higher than base
- adapter safety boundary rate improves over base

Even if all checks pass, Phase9 only allows:

```text
recommendation=promote_after_manual_review
```

Automatic promotion is never allowed.

Current canonical result fails the gate and must be archived.

## Why No 30/60-step Probe

The 12-step Phase9 adapter did not show a format lift over base. It was worse on structure, citation, unsupported assertions, and safety boundary. Per the Phase9 plan, a longer 30/60-step probe is only useful if the smaller run shows clear format improvement. It did not.

## Next Development Focus

Do not jump to Qwen3.6-27B yet. The current evidence says the bottleneck is still training format and decoding behavior, not model size alone.

Recommended next work:

1. Add a tiny synthetic format-only curriculum before contract samples.
2. Add stop-condition decoding or post-generation truncation at the fourth required line for eval and runtime, while preserving raw output evidence.
3. Increase LoRA coverage only after the adapter can reproduce the four labels without numbering or repetition.
4. Add hard negative holdouts for numbered lists, legal conclusions, and citation-less answers.
5. Re-run Qwen3-0.6B before considering larger Qwen3.6 models.
