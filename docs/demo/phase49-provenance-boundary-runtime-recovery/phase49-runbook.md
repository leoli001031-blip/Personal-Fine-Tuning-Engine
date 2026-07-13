# Phase49 Runbook

## Prepare and scorer-debug attempt

```bash
.venv/bin/python tools/phase49_prepare.py --clean-evidence
.venv/bin/python tools/phase49_qwen3_4b_generate.py --variant base_privacy --clean
```

Attempt-01 exposed a boundary-paraphrase scoring gap after 192 real calls. Its holdout, scorer freeze, transcript, metrics, and invalidation decision are preserved under `evidence-scorer-debug/attempt-01-boundary-paraphrase-gap/` and are not eligible for formal conclusions.

## Fresh formal attempt

After extending semantic calibration, regenerate preparation evidence without deleting the debug directory:

```bash
.venv/bin/python tools/phase49_prepare.py
.venv/bin/pytest -q tests/test_phase49_provenance_boundary_recovery.py tests/test_phase48_compact_intent_runtime.py tests/test_phase47_simulated_user_review.py tests/test_phase46_runtime_first_latest_intent.py tests/test_phase45_privacy_multiturn_preference.py
.venv/bin/python tools/phase49_qwen3_4b_generate.py --variant base_privacy --clean
.venv/bin/python tools/phase49_qwen3_4b_generate.py --variant base_compact_v1 --clean
.venv/bin/python tools/phase49_qwen3_4b_generate.py --variant base_compact_v2 --clean
```

## Blind evaluation and finalization

```bash
.venv/bin/python tools/phase49_blind_eval.py --resume
.venv/bin/python tools/phase49_finalize_evidence.py
.venv/bin/python tools/phase49_validate.py
```

No training, adapter, Hermes attachment, automatic promotion, or product-default change is allowed.
