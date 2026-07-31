# Phase45 Runbook

## Prepare and freeze

```bash
.venv/bin/python tools/phase45_prepare.py --clean-evidence
.venv/bin/pytest -q tests/test_phase45_privacy_multiturn_preference.py tests/test_phase44_preference_curriculum.py tests/test_trainer_real_peft_job.py
```

## Real Qwen3-4B SFT

```bash
.venv/bin/python tools/phase45_qwen3_4b_sft_probe.py --candidate candidate_a --steps 1 --clean
.venv/bin/python tools/phase45_qwen3_4b_sft_probe.py --candidate candidate_a --steps 12 --clean
.venv/bin/python tools/phase45_qwen3_4b_sft_probe.py --candidate candidate_a --steps 160 --clean
.venv/bin/python tools/phase45_qwen3_4b_sft_probe.py --candidate candidate_b --steps 160 --clean
```

## Diagnostic selection and fair-generation preflight

Run `base_privacy`, `candidate_a_privacy`, and `candidate_b_privacy` in diagnostic mode, then run `tools/phase45_select_candidate.py`. Run `base_raw` and the selected candidate raw arm, then rerun selection to freeze the four-arm preflight. Protocol v1 failed on truncation and is retained under `evidence-diagnostic/protocol-v1-failed/`; `tools/phase45_revise_generation_protocol.py` records the v2 revision.

## Frozen 80-session holdout

```bash
.venv/bin/python tools/phase45_qwen3_4b_generate.py --mode holdout --variant base_raw --clean
.venv/bin/python tools/phase45_qwen3_4b_generate.py --mode holdout --variant base_privacy --clean
.venv/bin/python tools/phase45_qwen3_4b_generate.py --mode holdout --variant adapter_raw --clean
.venv/bin/python tools/phase45_qwen3_4b_generate.py --mode holdout --variant adapter_privacy --clean
```

## Blind eval and finalization

```bash
.venv/bin/python tools/phase45_blind_eval.py --resume
.venv/bin/python tools/phase45_finalize_evidence.py
```

The independent judge is local Ollama `gemma4:31b` with `think=false`. Phase45 never auto-promotes; passing would only recommend a manual Hermes shadow trial.
