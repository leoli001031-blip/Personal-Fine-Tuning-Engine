# Phase10 Loop Summary

## What Changed

- Added `pfe_core.phase10_loop_engineering`.
- Added a Stage A format-only curriculum for the four-line output:
  `摘要 / 风险提示 / 引用依据 / 人工确认`.
- Added raw and normalized eval evidence:
  - `raw_output` is preserved;
  - `normalized_output` only truncates the first complete four-section block;
  - missing sections are never synthesized.
- Added Qwen3.6 4-bit preflight decision logic that stays blocked until the
  Qwen3-0.6B gate passes.
- Added Phase10 smoke and unit tests.

## Real Results

### Default Smoke

- Evidence: `docs/demo/phase10-loop-engineering/evidence/`
- Candidate samples: 60
- Real training: not started
- Decision: archive
- Qwen3.6: skipped

### Stage A Qwen3-0.6B 12-step

- Evidence: `docs/demo/phase10-loop-engineering/evidence-qwen3-0.6b-stage-a/`
- Real training: completed
- Candidate samples: 60
- Holdout prompts: 10
- Decision: archive

| metric | base | adapter | delta |
|---|---:|---:|---:|
| citation hit rate | 0.3 | 0.3 | 0.0 |
| structure hit rate | 0.6 | 0.15 | -0.45 |
| safety boundary rate | 0.0 | 0.0 | 0.0 |
| unsupported assertions | 17 | 17 | 0 |

### Decode Probe

- Evidence: `docs/demo/phase10-loop-engineering/evidence-qwen3-0.6b-stage-a-decode-probe/`
- Real training: completed
- Eval max tokens: 192
- Repetition penalty: 1.2
- Decision: archive

| metric | base | adapter | delta |
|---|---:|---:|---:|
| citation hit rate | 0.3 | 0.4 | 0.1 |
| structure hit rate | 0.6 | 0.25 | -0.35 |
| safety boundary rate | 0.0 | 0.0 | 0.0 |
| unsupported assertions | 17 | 16 | 1 |

## Decision

Phase10 did not solve the four-section stability problem yet. The adapter
showed modest citation and unsupported-assertion improvement in the decode
probe, but it still failed the primary structure gate and did not improve the
safety boundary rate.

The correct product decision is `archive`.

Stage B and Qwen3.6 4-bit training were not run because Stage A did not pass.

## Next Loop

The next loop should change the training target, not the model size:

- make target completions even shorter;
- add exact label-order contrastive examples;
- add negative examples where partial labels, repeated labels, and legal
  conclusions are corrected into the four-line format;
- consider a two-pass eval prompt that asks for the four labels without source
  prose copied into the answer;
- rerun Qwen3-0.6B before any Qwen3.6 work.
