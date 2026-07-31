# Phase46 Runbook

## Prepare and freeze

```bash
.venv/bin/python tools/phase46_prepare.py --clean-evidence
.venv/bin/pytest -q tests/test_phase46_runtime_first_latest_intent.py tests/test_phase45_privacy_multiturn_preference.py
```

## Real three-arm runtime ablation

```bash
.venv/bin/python tools/phase46_qwen3_4b_generate.py --variant base_privacy --clean
.venv/bin/python tools/phase46_qwen3_4b_generate.py --variant base_privacy_intent --clean
.venv/bin/python tools/phase46_qwen3_4b_generate.py --variant adapter_privacy_intent --clean
```

All arms use the same privacy transform, output-length contract, frozen 72-session holdout, deterministic decoding, and Qwen3-4B base. Only the latest-intent envelope and eval-only archived adapter vary.

## Blind evaluation and finalization

```bash
.venv/bin/python tools/phase46_blind_eval.py --resume
.venv/bin/python tools/phase46_finalize_evidence.py
.venv/bin/python tools/phase46_validate.py
```

The independent judge is local Ollama `gemma4:31b` with `think=false`. No Phase46 training or automatic promotion is allowed.
