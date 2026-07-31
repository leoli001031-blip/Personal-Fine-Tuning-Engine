# Phase50 Runbook

```bash
.venv/bin/python tools/phase50_prepare.py --clean-evidence
.venv/bin/pytest -q tests/test_phase50_conditional_provenance_guard.py tests/test_phase49_provenance_boundary_recovery.py tests/test_phase48_compact_intent_runtime.py tests/test_phase47_simulated_user_review.py tests/test_phase46_runtime_first_latest_intent.py tests/test_phase45_privacy_multiturn_preference.py
.venv/bin/python tools/phase50_qwen3_4b_generate.py --variant base_compact_v1 --clean
.venv/bin/python tools/phase50_qwen3_4b_generate.py --variant base_global_v2 --clean
.venv/bin/python tools/phase50_qwen3_4b_generate.py --variant base_conditional_guard --clean
.venv/bin/python tools/phase50_blind_eval.py --resume
.venv/bin/python tools/phase50_posthoc_evaluator_audit.py
.venv/bin/python tools/phase50_finalize_evidence.py
.venv/bin/python tools/phase50_validate.py
```

The router, scorer, holdout, protocol, and gates must be frozen before model calls. No training, adapter, Hermes attachment, automatic promotion, or default-path change is allowed.
