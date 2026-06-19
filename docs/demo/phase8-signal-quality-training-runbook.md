# Phase8 Signal Quality Training Runbook

Phase8 validates the product idea at the signal-quality layer:

1. collect public, citable, low-PII contract material
2. synthesize high-quality edit / correction / preference signals
3. quality-gate candidate samples before training
4. run a real Qwen3-0.6B MLX LoRA/SFT trial
5. compare base vs adapter on holdout prompts that never enter training
6. archive by default unless the real adapter passes the eval gate

This phase does not add UI and does not touch `videos/`.

## Scenario

- Scenario: contract summary / risk flagging / citation grounding / human confirmation
- Allowed behavior: material organization, risk hints, source citation, uncertainty and human-review boundaries
- Disallowed behavior: legal conclusion, professional advice, unsupported inference, high-risk deterministic claims
- Required output sections: `摘要 / 风险提示 / 引用依据 / 人工确认`

## Sources

The source pack uses Common Paper public contract templates under the Common Paper standard-agreement attribution note.

Evidence:

- `docs/demo/phase8-signal-quality-training/evidence/source_manifest.json`
- `docs/demo/phase8-signal-quality-training/evidence-real-qwen3-0.6b/source_manifest.json`

Current canonical real run:

- 11 public sources collected
- 10 sources allowed for training
- 1 source routed to review-only by PII audit
- source goal passed

## Default Smoke

Use the default smoke when validating the data and quality gate without running MLX training:

```bash
.venv/bin/python tools/phase8_signal_quality_training_smoke.py \
  --evidence-dir docs/demo/phase8-signal-quality-training/evidence \
  --clean-evidence
```

Expected result:

- `quality_report.json` shows at least 30 quality-passed signals
- `candidate_samples.jsonl` contains holdout-free training candidates
- `eval_report.json` is blocked because no real training/eval was requested
- `decision.json` archives the trial

## Real Qwen3-0.6B Trial

Canonical real run:

```bash
HF_HUB_DISABLE_XET=1 .venv/bin/python tools/phase8_signal_quality_training_smoke.py \
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
  --eval-max-tokens 160 \
  --epochs 12 \
  --timeout 2400 \
  --evidence-dir docs/demo/phase8-signal-quality-training/evidence-real-qwen3-0.6b \
  --clean-evidence \
  --keep-workdir
```

Evidence:

- `docs/demo/phase8-signal-quality-training/evidence-real-qwen3-0.6b/quality_report.json`
- `docs/demo/phase8-signal-quality-training/evidence-real-qwen3-0.6b/train_log.json`
- `docs/demo/phase8-signal-quality-training/evidence-real-qwen3-0.6b/training_attempt.json`
- `docs/demo/phase8-signal-quality-training/evidence-real-qwen3-0.6b/eval_report.json`
- `docs/demo/phase8-signal-quality-training/evidence-real-qwen3-0.6b/decision.json`

Observed result:

- real MLX training completed
- real base/adapter eval completed
- 60 quality signals passed
- 60 candidate samples passed
- holdout count: 10
- adapter citation hit rate improved from `0.2` to `0.5`
- unsupported assertions improved from `18` to `15`
- structure adherence fell from `0.175` to `0.0`
- safety boundary rate remained `0.0`
- final recommendation: `archive`

This is a partial lift, not a promotable adapter. It proves the loop can collect quality data and run a real training/eval cycle, but the adapter did not learn the required output structure strongly enough.

## 60-step Probe

An exploratory 60-step run is saved separately:

- `docs/demo/phase8-signal-quality-training/evidence-real-qwen3-0.6b-epochs60/eval_report.json`

Observed result:

- real training completed
- real eval completed
- adapter underperformed base on citation, structure, and unsupported assertions
- final recommendation: `archive`

This suggests training strength alone is not the next fix. The next iteration should improve completion formatting, output-only loss masking, decoder settings, or LoRA coverage before trying larger models.

## Qwen3.6-27B Boundary

Do not rerun Qwen3.6-27B by default in Phase8.

Phase7 established the boundary evidence: `mlx-community/Qwen3.6-27B-4bit` hit Metal memory pressure on this Mac during real training. Keep it as a target-model boundary only:

- preflight is allowed
- dry-run recipe documentation is allowed
- long real training retries are not part of this phase

## Eval Gate

Promotion requires all of the following:

- real MLX training completed
- real base vs adapter holdout eval completed
- adapter citation hit rate >= `0.85`
- adapter structure hit rate >= `0.85`
- adapter safety boundary rate >= `0.85`
- adapter does not increase unsupported assertions
- adapter beats base on structure adherence

Even if the gate passes, Phase8 only allows:

```text
recommendation=promote_after_manual_review
```

Automatic promotion is never allowed in Phase8.

## Next Development Focus

The next phase should not jump straight to Qwen3.6-27B. The current evidence points to these higher-leverage fixes first:

1. Make training examples use a cleaner completion boundary and avoid copying `资料片段` patterns into the target.
2. Add output-only loss masking or chat-template formatting if the MLX path supports it.
3. Increase LoRA coverage beyond the current minimal adapter layer only after the formatting issue is fixed.
4. Add decoding controls for eval to reduce repetition.
5. Keep collecting stronger correction/edit signals, but require each sample to demonstrate the exact four-section output form.
