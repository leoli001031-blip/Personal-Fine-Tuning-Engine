# PFE Phase43 runbook

## Scope

Phase43 tests simulated laboratory preference benefit on local unquantized Qwen3-4B. It does not claim actual user benefit and never auto-promotes an adapter.

## Reproduce

```bash
cd /Users/lichenhao/Desktop/PFE
.venv/bin/python tools/phase43_qwen3_4b_prepare.py --clean-evidence
.venv/bin/python tools/phase43_qwen3_4b_sft_probe.py --steps 1 --clean
.venv/bin/python tools/phase43_qwen3_4b_sft_probe.py --steps 12 --clean
.venv/bin/python tools/phase43_qwen3_4b_generate.py --variant base --mode sanity --max-new-tokens 80
.venv/bin/python tools/phase43_qwen3_4b_generate.py --variant runtime --mode sanity --max-new-tokens 80
.venv/bin/python tools/phase43_qwen3_4b_generate.py --variant sft --mode sanity --steps 12 --max-new-tokens 80
.venv/bin/python tools/phase43_qwen3_4b_sft_probe.py --steps 30 --clean
.venv/bin/python tools/phase43_qwen3_4b_generate.py --variant sft --mode sanity --steps 30 --max-new-tokens 80
.venv/bin/python tools/phase43_qwen3_4b_dpo_probe.py --steps 12 --clean
.venv/bin/python tools/phase43_qwen3_4b_generate.py --variant base --mode holdout --max-new-tokens 96
.venv/bin/python tools/phase43_qwen3_4b_generate.py --variant runtime --mode holdout --max-new-tokens 96
.venv/bin/python tools/phase43_qwen3_4b_generate.py --variant sft --mode holdout --steps 12 --max-new-tokens 96
.venv/bin/python tools/phase43_blind_eval.py
.venv/bin/python tools/phase43_finalize_evidence.py
```

## Validation

```bash
.venv/bin/python -m py_compile pfe-core/pfe_core/phase43_personal_preference_benefit.py tools/phase43_*.py
.venv/bin/pytest -q tests/test_phase43_personal_preference_benefit.py
.venv/bin/pytest -q tests/test_phase42_reliability_hardening.py
make test-unit test-surface test-e2e-mock smoke-beta
git diff --check
```

## Decision boundary

Only a candidate passing both deterministic and independent blind gates may become `ready_for_manual_acceptance_trial`. Actual benefit still requires later real Hermes feedback and human review.
