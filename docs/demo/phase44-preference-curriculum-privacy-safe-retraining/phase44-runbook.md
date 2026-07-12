# Phase44 Runbook

## Prepare and freeze

```bash
.venv/bin/python tools/phase44_prepare.py --clean-evidence
.venv/bin/pytest -q tests/test_phase44_preference_curriculum.py tests/test_phase43_personal_preference_benefit.py
```

## Real Qwen3-4B SFT

```bash
.venv/bin/python tools/phase44_qwen3_4b_sft_probe.py --steps 1 --clean
.venv/bin/python tools/phase44_qwen3_4b_sft_probe.py --steps 12 --clean
.venv/bin/python tools/phase44_qwen3_4b_sft_probe.py --steps 120 --clean
```

## Diagnostic and frozen holdout

Run `tools/phase44_qwen3_4b_generate.py` once per `base`, `runtime_v1`, `soft_runtime`, and `sft`, first with `--mode diagnostic --clean`, then with `--mode holdout --clean`. Use `--steps 120` for `sft`.

## Blind evaluation and finalization

```bash
.venv/bin/python tools/phase44_blind_eval.py
.venv/bin/python tools/phase44_finalize_evidence.py
```

The independent judge must be local Ollama `gemma4:31b` with `think=false`. Never promote automatically; a passing outcome can only recommend `ready_for_hermes_shadow_trial`.
