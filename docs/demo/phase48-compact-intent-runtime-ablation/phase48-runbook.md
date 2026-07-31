# Phase48 Runbook

## Prepare and freeze

```bash
.venv/bin/python tools/phase48_prepare.py --clean-evidence
.venv/bin/pytest -q tests/test_phase48_compact_intent_runtime.py tests/test_phase47_simulated_user_review.py tests/test_phase46_runtime_first_latest_intent.py tests/test_phase45_privacy_multiturn_preference.py
```

## Real three-arm runtime ablation

```bash
.venv/bin/python tools/phase48_qwen3_4b_generate.py --variant base_privacy --clean
.venv/bin/python tools/phase48_qwen3_4b_generate.py --variant base_compact_intent --clean
.venv/bin/python tools/phase48_qwen3_4b_generate.py --variant base_full_intent --clean
```

All arms use the same Qwen3-4B base, privacy transform, length contract, 64-session fresh holdout, deterministic decoding, and no adapter. Only the latest-intent runtime expression varies.

## Blind evaluation and finalization

```bash
.venv/bin/python tools/phase48_blind_eval.py --resume
.venv/bin/python tools/phase48_finalize_evidence.py
.venv/bin/python tools/phase48_validate.py
```

The independent judge is local Ollama `gemma4:31b` with `think=false`. Phase48 permits neither training nor automatic promotion.
